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
    "rho_N",
    "max_ratio",
    "converges",
    "n_states",
    "n_stages",
    "worst_rows",
    "J_numeric",
}


@pytest.mark.parametrize(
    "solver_settings_override", [_DIAGONALLY_DOMINANT], indirect=True
)
def test_converges_for_diagonally_dominant_system(system):
    """A diagonally dominant Jacobian reports convergence."""
    result = check_neumann_convergence(
        system.indices,
        system._get_neumann_evaluator(),
    )
    assert result["converges"] is True
    assert result["rho_N"] < 1.0
    assert set(result) == _RESULT_KEYS


@pytest.mark.parametrize(
    "solver_settings_override", [_OFF_DIAGONAL_HEAVY], indirect=True
)
def test_diverges_and_warns_for_off_diagonal_heavy_system(system):
    """A strongly cross-coupled Jacobian warns of divergence."""
    with pytest.warns(UserWarning):
        result = check_neumann_convergence(
            system.indices,
            system._get_neumann_evaluator(),
        )
    assert result["converges"] is False
    assert result["rho_N"] >= 1.0
    assert result["worst_rows"]


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
        )
    assert result["converges"] is True
    assert result["rho_N"] < 1.0


@pytest.mark.parametrize(
    "solver_settings_override", [_SINGULAR_INITIAL_STATE], indirect=True
)
def test_non_finite_jacobian_reports_not_verified(system):
    """A non-finite Jacobian returns the full signature with no verdict."""
    result = check_neumann_convergence(
        system.indices,
        system._get_neumann_evaluator(),
    )
    assert result["converges"] is None
    # The early-return path must expose the same keys as the normal path.
    assert set(result) == _RESULT_KEYS


@pytest.mark.parametrize(
    "solver_settings_override", [_OFF_DIAGONAL_HEAVY], indirect=True
)
def test_get_solver_helper_runs_diagnostic_for_neumann_type(system):
    """Requesting a Neumann helper triggers the convergence warning."""
    with pytest.warns(UserWarning):
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
