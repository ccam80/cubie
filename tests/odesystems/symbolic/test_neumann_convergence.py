"""Tests for the Neumann preconditioner convergence diagnostic.

Covers the numeric check and its solver-owned wiring.
"""

from pathlib import Path
import warnings

import numpy as np
import pytest

from cubie import SymbolicODE
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.cubie_cache import CUBIECache
from cubie.odesystems.symbolic.codegen.neumann_convergence import (
    NeumannRHSEvaluator,
    check_neumann_convergence,
    neumann_spectral_radius,
)


@pytest.fixture(scope="session")
def diagonally_dominant_system():
    """Decoupled, strongly diagonal system (Neumann converges)."""
    return SymbolicODE.create(
        dxdt=["dx = -10.0 * x", "dy = -10.0 * y"],
        states={"x": 1.0, "y": 1.0},
        precision=np.float64,
    )


@pytest.fixture(scope="session")
def off_diagonal_heavy_system():
    """Strong cross-coupling breaks diagonal dominance (diverges)."""
    return SymbolicODE.create(
        dxdt=["dx = -x + 100.0 * y", "dy = 100.0 * x - y"],
        states={"x": 1.0, "y": 1.0},
        precision=np.float64,
    )


@pytest.fixture(scope="session")
def gating_singularity_system():
    """Diagonally dominant system with a guarded ``min`` term.

    The off-diagonal coupling uses ``min`` so that the analytic
    derivative is a ``Piecewise``. Finite-differencing the guarded
    right-hand side evaluates it cleanly at the initial state.
    """
    return SymbolicODE.create(
        dxdt=["dx = -10.0 * x + min(y, 1.0)", "dy = -10.0 * y"],
        states={"x": 0.5, "y": 0.5},
        precision=np.float64,
    )


@pytest.fixture(scope="session")
def singular_initial_state_system():
    """System whose Jacobian is non-finite at the initial state.

    ``log(x)`` is undefined for the backward finite-difference step at
    ``x == 0``, so the Jacobian cannot be evaluated there.
    """
    return SymbolicODE.create(
        dxdt=["dx = log(x)", "dy = -10.0 * y"],
        states={"x": 0.0, "y": 1.0},
        precision=np.float64,
    )


# Keys every check_neumann_convergence return must expose, regardless of
# which path produced it.
_RESULT_KEYS = {
    "rho_N",
    "max_ratio",
    "converges",
    "n_states",
    "n_stages",
    "worst_rows",
    "J_numeric",
}


def _evaluator(system):
    return NeumannRHSEvaluator(
        lambda: system.evaluate_f,
        lambda: system.precision,
    )


def test_converges_for_diagonally_dominant_system(
    diagonally_dominant_system,
):
    """A diagonally dominant Jacobian reports convergence."""
    result = check_neumann_convergence(
        diagonally_dominant_system.indices,
        _evaluator(diagonally_dominant_system),
    )
    assert result["converges"] is True
    assert result["rho_N"] < 1.0
    assert result["worst_rows"] == []


def test_diverges_and_warns_for_off_diagonal_heavy_system(
    off_diagonal_heavy_system,
):
    """A non-dominant Jacobian reports divergence and warns."""
    with pytest.warns(UserWarning):
        result = check_neumann_convergence(
            off_diagonal_heavy_system.indices,
            _evaluator(off_diagonal_heavy_system),
        )
    assert result["converges"] is False
    assert result["rho_N"] >= 1.0
    assert result["worst_rows"]


def test_gating_singularity_converges_without_false_divergence(
    gating_singularity_system,
):
    """A guarded ``min`` term evaluates instead of forcing divergence."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = check_neumann_convergence(
            gating_singularity_system.indices,
            _evaluator(gating_singularity_system),
        )
    assert result["converges"] is True
    assert result["rho_N"] < 1.0
    assert not [w for w in record if "DIVERGE" in str(w.message)]


def test_non_finite_jacobian_reports_not_verified(
    singular_initial_state_system,
):
    """A non-finite Jacobian returns the full signature with no verdict."""
    result = check_neumann_convergence(
        singular_initial_state_system.indices,
        _evaluator(singular_initial_state_system),
    )
    assert result["converges"] is None
    # The early-return path must expose the same keys as the normal path.
    assert set(result) == _RESULT_KEYS


def test_finite_and_non_finite_paths_share_return_signature(
    diagonally_dominant_system,
    singular_initial_state_system,
):
    """Both convergence paths return an identical set of keys."""
    finite = check_neumann_convergence(
        diagonally_dominant_system.indices,
        _evaluator(diagonally_dominant_system),
    )
    non_finite = check_neumann_convergence(
        singular_initial_state_system.indices,
        _evaluator(singular_initial_state_system),
    )
    assert set(finite) == set(non_finite) == _RESULT_KEYS


def test_solver_runs_diagnostic_for_neumann_type(
    off_diagonal_heavy_system,
):
    """Requesting a Neumann helper triggers the convergence warning."""
    kernel = BatchSolverKernel(
        off_diagonal_heavy_system,
        algorithm_settings={"algorithm": "euler"},
        cache=False,
    )
    try:
        with pytest.warns(UserWarning):
            kernel._get_solver_helper(
                "neumann_preconditioner",
                beta=1.0,
                gamma=1.0,
                preconditioner_order=0,
            )
    finally:
        kernel.close()


def test_each_solver_owns_its_neumann_cache_policy(
    diagonally_dominant_system,
    tmp_path,
):
    """Each solver applies its own cache policy to the diagnostic."""
    disabled = BatchSolverKernel(
        diagonally_dominant_system,
        algorithm_settings={"algorithm": "euler"},
        cache=False,
    )
    enabled = BatchSolverKernel(
        diagonally_dominant_system,
        algorithm_settings={"algorithm": "euler"},
        cache=tmp_path,
    )
    try:
        disabled_helper = disabled._get_solver_helper(
            "neumann_preconditioner", preconditioner_order=0
        )
        enabled_helper = enabled._get_solver_helper(
            "neumann_preconditioner", preconditioner_order=0
        )
        disabled_again = disabled._get_solver_helper(
            "neumann_preconditioner", preconditioner_order=0
        )
        disabled_cache = disabled._neumann_cache()
        enabled_cache = enabled._neumann_cache()

        assert disabled_helper is enabled_helper is disabled_again
        assert disabled_cache is None
        assert isinstance(enabled_cache, CUBIECache)
        assert Path(enabled_cache._cache_path).parent == tmp_path
        assert (
            disabled._neumann_rhs_evaluator._kernel
            is not enabled._neumann_rhs_evaluator._kernel
        )
    finally:
        disabled.close()
        enabled.close()


def test_cache_policy_update_rebuilds_neumann_kernel(
    diagonally_dominant_system,
    tmp_path,
):
    """A cache-policy update rebuilds the diagnostic dispatcher."""
    kernel = BatchSolverKernel(
        diagonally_dominant_system,
        algorithm_settings={"algorithm": "euler"},
        cache=False,
    )
    try:
        kernel._get_solver_helper(
            "neumann_preconditioner", preconditioner_order=0
        )
        uncached = kernel._neumann_rhs_evaluator._kernel

        initial_path = tmp_path / "initial"
        kernel.update(cache_enabled=True, cache_dir=initial_path)
        kernel._get_solver_helper(
            "neumann_preconditioner", preconditioner_order=0
        )
        cached = kernel._neumann_rhs_evaluator._kernel

        relocated_path = tmp_path / "relocated"
        kernel.set_cache_dir(relocated_path)
        kernel._get_solver_helper(
            "neumann_preconditioner", preconditioner_order=0
        )
        relocated = kernel._neumann_rhs_evaluator._kernel
        configured_cache = kernel._neumann_cache()

        assert cached is not uncached
        assert relocated is not cached
        assert isinstance(configured_cache, CUBIECache)
        assert Path(configured_cache._cache_path).parent == relocated_path
    finally:
        kernel.close()


def test_spectral_radius_tracks_beta():
    """beta shifts the operator diagonal, controlling convergence."""
    jacobian = np.array([[-1.0, 100.0], [100.0, -1.0]])
    diverging = neumann_spectral_radius(jacobian, beta=1.0, gamma=1.0)
    converging = neumann_spectral_radius(jacobian, beta=1000.0, gamma=1.0)
    assert diverging["converges"] is False
    assert diverging["rho_N"] >= 1.0
    assert converging["converges"] is True
    assert converging["rho_N"] < 1.0


def test_spectral_radius_single_stage_matches_unit_tableau():
    """A one-entry unit tableau reproduces the single-stage result."""
    jacobian = np.array([[-2.0, 0.5], [0.5, -2.0]])
    single = neumann_spectral_radius(jacobian, beta=1.0, gamma=1.0)
    staged = neumann_spectral_radius(
        jacobian, beta=1.0, gamma=1.0, stage_coefficients=[[1.0]]
    )
    assert staged["n_stages"] == 1
    assert np.isclose(single["rho_N"], staged["rho_N"])
