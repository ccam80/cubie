"""Tests for the Neumann preconditioner convergence diagnostic.

Covers the pure-numeric :func:`neumann_spectral_radius`,
:func:`check_neumann_convergence`, and the wiring into
:meth:`SymbolicODE.get_solver_helper`.
"""

import warnings

import numpy as np
import pytest

from cubie import SymbolicODE
from cubie.odesystems.symbolic.codegen.neumann_convergence import (
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


def test_converges_for_diagonally_dominant_system(
    diagonally_dominant_system,
):
    """A diagonally dominant Jacobian reports convergence."""
    result = check_neumann_convergence(
        diagonally_dominant_system.equations,
        diagonally_dominant_system.indices,
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
            off_diagonal_heavy_system.equations,
            off_diagonal_heavy_system.indices,
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
            gating_singularity_system.equations,
            gating_singularity_system.indices,
        )
    assert result["converges"] is True
    assert result["rho_N"] < 1.0
    assert not [w for w in record if "DIVERGE" in str(w.message)]


def test_get_solver_helper_runs_diagnostic_for_neumann_type(
    off_diagonal_heavy_system,
):
    """Requesting a Neumann helper triggers the convergence warning."""
    with pytest.warns(UserWarning):
        off_diagonal_heavy_system.get_solver_helper(
            "neumann_preconditioner", beta=1.0, gamma=1.0
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
