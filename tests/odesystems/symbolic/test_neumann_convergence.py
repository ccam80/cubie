"""Tests for the Neumann preconditioner convergence diagnostic.

Covers the pure-numeric :func:`neumann_spectral_radius`,
:func:`check_neumann_convergence`, the wiring into
:meth:`SymbolicODE.get_solver_helper`, and the cache configuration
flowing from a solver into the system-owned evaluator.
"""

from pathlib import Path
import warnings

import numpy as np
import pytest

from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.cubie_cache import CUBIECache
from cubie.odesystems.symbolic.codegen.neumann_convergence import (
    check_neumann_convergence,
    neumann_spectral_radius,
)

_DIAGONALLY_DOMINANT = {
    "system_type": "diagonally_dominant",
    "precision": np.float64,
}
_OFF_DIAGONAL_HEAVY = {
    "system_type": "off_diagonal_heavy",
    "precision": np.float64,
}
_GATING_SINGULARITY = {
    "system_type": "gating_singularity",
    "precision": np.float64,
}
_SINGULAR_INITIAL_STATE = {
    "system_type": "singular_initial_state",
    "precision": np.float64,
}

# Keys every check_neumann_convergence return must expose, regardless of
# which path produced it.
_RESULT_KEYS = {
    "rho_per_unit_step_factor",
    "critical_step_factor",
    "step_factor",
    "rho_series",
    "series_converges",
    "n_states",
    "n_stages",
    "J_numeric",
}


@pytest.mark.parametrize(
    "solver_settings_override", [_DIAGONALLY_DOMINANT], indirect=True
)
def test_small_step_converges_for_diagonally_dominant_system(system):
    """A sufficiently small supplied step reports convergence."""
    result = check_neumann_convergence(
        system.indices,
        system._get_neumann_evaluator(),
        step_size=1e-4,
        stage_coefficient=1.0,
    )
    assert result["series_converges"] is True
    assert result["rho_series"] < 1.0
    assert set(result) == _RESULT_KEYS


@pytest.mark.parametrize(
    "solver_settings_override", [_OFF_DIAGONAL_HEAVY], indirect=True
)
def test_large_step_diverges_for_off_diagonal_heavy_system(system):
    """A supplied step beyond the critical magnitude warns."""
    with pytest.warns(UserWarning):
        result = check_neumann_convergence(
            system.indices,
            system._get_neumann_evaluator(),
            step_size=1.0,
            stage_coefficient=1.0,
        )
    assert result["series_converges"] is False
    assert result["rho_series"] >= 1.0
    assert result["critical_step_factor"] <= 1.0


@pytest.mark.parametrize(
    "solver_settings_override", [_GATING_SINGULARITY], indirect=True
)
def test_gating_singularity_converges_without_false_divergence(system):
    """Guarded gating terms do not trigger a false divergence report."""
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        result = check_neumann_convergence(
            system.indices,
            system._get_neumann_evaluator(),
            step_size=1e-4,
            stage_coefficient=1.0,
        )
    assert result["series_converges"] is True
    assert result["rho_series"] < 1.0


@pytest.mark.parametrize(
    "solver_settings_override", [_SINGULAR_INITIAL_STATE], indirect=True
)
def test_non_finite_jacobian_reports_not_verified(system):
    """A non-finite Jacobian returns the full signature with no verdict."""
    result = check_neumann_convergence(
        system.indices,
        system._get_neumann_evaluator(),
    )
    assert result["series_converges"] is None
    # The early-return path must expose the same keys as the normal path.
    assert set(result) == _RESULT_KEYS


@pytest.mark.parametrize(
    "solver_settings_override", [_OFF_DIAGONAL_HEAVY], indirect=True
)
def test_get_solver_helper_runs_diagnostic_for_neumann_type(system):
    """Static helper check reports a step limit, not divergence."""
    with pytest.warns(UserWarning, match="not a divergence verdict"):
        system.get_solver_helper(
            "neumann_preconditioner", solver_beta=1.0, solver_gamma=1.0
        )


@pytest.mark.parametrize(
    "solver_settings_override", [_DIAGONALLY_DOMINANT], indirect=True
)
def test_solver_cache_policy_reaches_evaluator(system, tmp_path):
    """A solver's cache settings flow down to the system's evaluator."""
    evaluator = system._diagnostic_factories["neumann_rhs"]
    try:
        disabled = BatchSolverKernel(
            system,
            algorithm_settings={"algorithm": "euler"},
            cache=False,
        )
        try:
            cache_config = evaluator.compile_settings.cache_config
            assert cache_config.cache_enabled is False
        finally:
            disabled.close()

        enabled = BatchSolverKernel(
            system,
            algorithm_settings={"algorithm": "euler"},
            cache=tmp_path,
        )
        try:
            cache_config = evaluator.compile_settings.cache_config
            assert cache_config.cache_enabled is True
            assert cache_config.cache_dir == tmp_path
        finally:
            enabled.close()
    finally:
        system.update(
            {"cache_enabled": False, "cache_dir": None}, silent=True
        )


@pytest.mark.parametrize(
    "solver_settings_override", [_DIAGONALLY_DOMINANT], indirect=True
)
def test_cache_change_rebuilds_evaluator_kernel(system, tmp_path):
    """A cache-setting update rebuilds the diagnostic kernel."""
    try:
        evaluator = system._get_neumann_evaluator()
        first = evaluator.get_cached_output("evaluation_kernel")
        again = system._get_neumann_evaluator().get_cached_output(
            "evaluation_kernel"
        )
        assert again is first

        system.update(
            {"cache_enabled": True, "cache_dir": tmp_path}, silent=True
        )
        rebuilt = system._get_neumann_evaluator().get_cached_output(
            "evaluation_kernel"
        )
        assert rebuilt is not first
    finally:
        system.update(
            {"cache_enabled": False, "cache_dir": None}, silent=True
        )


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override", [_DIAGONALLY_DOMINANT], indirect=True
)
def test_evaluator_attaches_configured_disk_cache(system, tmp_path):
    """With caching enabled the built kernel carries a CUBIECache."""
    try:
        system.update(
            {"cache_enabled": True, "cache_dir": tmp_path}, silent=True
        )
        evaluator = system._get_neumann_evaluator()
        kernel = evaluator.get_cached_output("evaluation_kernel")
        assert isinstance(kernel._cache, CUBIECache)
        assert Path(kernel._cache.cache_path).parent == tmp_path
    finally:
        system.update(
            {"cache_enabled": False, "cache_dir": None}, silent=True
        )


def test_spectral_radius_tracks_beta():
    """beta scales the implemented Neumann series matrix."""
    jacobian = np.array([[-1.0, 100.0], [100.0, -1.0]])
    diverging = neumann_spectral_radius(
        jacobian,
        beta=1.0,
        gamma=1.0,
        stage_coefficient=1.0,
        step_factor_value=1.0,
    )
    converging = neumann_spectral_radius(
        jacobian,
        beta=1000.0,
        gamma=1.0,
        stage_coefficient=1.0,
        step_factor_value=1.0,
    )
    assert diverging["series_converges"] is False
    assert diverging["rho_series"] >= 1.0
    assert converging["series_converges"] is True
    assert converging["rho_series"] < 1.0


def test_spectral_radius_reports_exact_critical_step():
    """The static limit is the reciprocal unit-step radius."""
    jacobian = np.array([[0.0, 2.0], [-3.0, 0.0]])
    static = neumann_spectral_radius(
        jacobian, beta=2.0, gamma=3.0, stage_coefficient=0.25
    )
    critical = static["critical_step_factor"]

    below = neumann_spectral_radius(
        jacobian,
        beta=2.0,
        gamma=3.0,
        stage_coefficient=0.25,
        step_factor_value=0.5 * critical,
    )
    above = neumann_spectral_radius(
        jacobian,
        beta=2.0,
        gamma=3.0,
        stage_coefficient=0.25,
        step_factor_value=2.0 * critical,
    )

    assert static["step_factor"] == "h"
    assert np.isclose(
        critical,
        1.0 / static["rho_per_unit_step_factor"],
    )
    assert static["rho_series"] is None
    assert static["series_converges"] is None
    assert np.isclose(below["rho_series"], 0.5)
    assert below["series_converges"] is True
    assert np.isclose(above["rho_series"], 2.0)
    assert above["series_converges"] is False


def test_single_stage_without_coefficient_reports_effective_step():
    """Unknown runtime a_ij yields a bound on abs(a_ij*h)."""
    result = neumann_spectral_radius(np.array([[4.0]]))
    assert result["step_factor"] == "a_ij * h"
    assert np.isclose(result["rho_per_unit_step_factor"], 4.0)
    assert np.isclose(result["critical_step_factor"], 0.25)


def test_nontrivial_firk_tableau_uses_kronecker_coupling():
    """FIRK radius matches the full A-tensor-J spectrum."""
    jacobian = np.array([[2.0, 1.0], [-1.0, 3.0]])
    coefficients = np.array([[0.2, -0.1], [0.4, 0.3]])
    expected = max(abs(np.linalg.eigvals(np.kron(coefficients, jacobian))))
    result = neumann_spectral_radius(
        jacobian,
        stage_coefficients=coefficients,
    )
    assert result["step_factor"] == "h"
    assert np.isclose(result["rho_per_unit_step_factor"], expected)


def test_spectral_radius_single_stage_matches_unit_tableau():
    """A one-entry unit tableau reproduces the single-stage result."""
    jacobian = np.array([[-2.0, 0.5], [0.5, -2.0]])
    single = neumann_spectral_radius(jacobian, beta=1.0, gamma=1.0)
    staged = neumann_spectral_radius(
        jacobian, beta=1.0, gamma=1.0, stage_coefficients=[[1.0]]
    )
    assert staged["n_stages"] == 1
    assert np.isclose(
        single["rho_per_unit_step_factor"],
        staged["rho_per_unit_step_factor"],
    )
