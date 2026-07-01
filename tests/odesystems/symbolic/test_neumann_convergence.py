"""Tests for the Neumann preconditioner convergence diagnostic.

Covers :func:`check_neumann_convergence` directly and its wiring into
:meth:`SymbolicODE.get_solver_helper`.
"""

import numpy as np
import pytest

from cubie import SymbolicODE
from cubie.odesystems.symbolic.codegen.neumann_convergence import (
    check_neumann_convergence,
)


def _diagonally_dominant_system():
    """Decoupled, strongly diagonal system (Neumann converges)."""
    return SymbolicODE.create(
        dxdt=["dx = -10.0 * x", "dy = -10.0 * y"],
        states={"x": 1.0, "y": 1.0},
        precision=np.float64,
    )


def _off_diagonal_heavy_system():
    """Strong cross-coupling breaks diagonal dominance (diverges)."""
    return SymbolicODE.create(
        dxdt=["dx = -x + 100.0 * y", "dy = 100.0 * x - y"],
        states={"x": 1.0, "y": 1.0},
        precision=np.float64,
    )


def test_converges_for_diagonally_dominant_system():
    """A diagonally dominant Jacobian reports convergence."""
    ode = _diagonally_dominant_system()
    result = check_neumann_convergence(ode.equations, ode.indices)
    assert result["converges"] is True
    assert result["rho_N"] < 1.0
    assert result["worst_rows"] == []


def test_diverges_and_warns_for_off_diagonal_heavy_system():
    """A non-dominant Jacobian reports divergence and warns."""
    ode = _off_diagonal_heavy_system()
    with pytest.warns(UserWarning):
        result = check_neumann_convergence(ode.equations, ode.indices)
    assert result["converges"] is False
    assert result["rho_N"] >= 1.0
    assert result["worst_rows"]


def test_get_solver_helper_runs_diagnostic_for_neumann_type():
    """Requesting a Neumann helper triggers the convergence warning."""
    ode = _off_diagonal_heavy_system()
    with pytest.warns(UserWarning):
        ode.get_solver_helper(
            "neumann_preconditioner", beta=1.0, gamma=1.0
        )
