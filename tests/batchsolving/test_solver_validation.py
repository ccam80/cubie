"""Integration tests for solver-level validation.

This test module validates that compatibility checking propagates correctly
through the high-level solver API, including :class:`~cubie.Solver` and
:func:`~cubie.solve_ivp`.

The tests ensure that:

1. :class:`Solver` raises ValueError for incompatible configurations
   (adaptive controller + errorless algorithm).
2. :func:`solve_ivp` raises ValueError for incompatible configurations.
3. Compatible configurations succeed without errors.
4. Error messages propagate from the underlying
   :class:`~cubie.integrators.SingleIntegratorRunCore.SingleIntegratorRunCore`
   validation.

These integration tests validate the full API surface to ensure users
receive clear error messages regardless of which entry point they use.

Notes
-----
These tests use real system fixtures and actual solver components rather
than mocks, ensuring that validation behaves correctly in real-world usage
scenarios.
"""

import pytest
import numpy as np

from cubie import Solver, solve_ivp
from cubie.integrators.algorithms.generic_erk_tableaus import (
    CLASSICAL_RK4_TABLEAU,
)


def test_solver_with_incompatible_config_raises(system):
    """Solver raises ValueError for incompatible configuration."""

    with pytest.raises(ValueError) as exc_info:
        Solver(
            system=system,
            algorithm="explicit_euler",
            step_controller="pi",
            dt_min=1e-6,
            dt_max=1e-1,
        )

    error_msg = str(exc_info.value).lower()
    assert "explicit_euler" in error_msg
    assert "pi" in error_msg


def test_solve_ivp_with_incompatible_config_raises(
    system
):
    """solve_ivp raises ValueError for incompatible configuration."""

    initial_values = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    parameters = np.array([1.0], dtype=np.float32)

    with pytest.raises(ValueError) as exc_info:
        solve_ivp(
            system=system,
            initial_values=initial_values,
            parameters=parameters,
            t_span=(0.0, 10.0),
            algorithm="erk",
            tableau=CLASSICAL_RK4_TABLEAU,
            step_controller="pi",
            dt_min=1e-6,
            dt_max=1e-1,
        )

    error_msg = str(exc_info.value).lower()
    assert "erk" in error_msg
    assert "pi" in error_msg


def test_solver_with_compatible_config_succeeds(
    system
):
    """Solver succeeds with compatible configuration."""

    solver = Solver(
        system=system,
        algorithm="explicit_euler",
        step_controller="fixed",
        dt=1e-3,
    )

    assert solver is not None
