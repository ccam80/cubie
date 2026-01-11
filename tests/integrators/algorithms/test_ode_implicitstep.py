"""Tests for ODEImplicitStep tolerance parameter routing."""

import numpy as np

from cubie.integrators.algorithms.backwards_euler import BackwardsEulerStep


def test_implicit_step_accepts_tolerance_arrays(precision):
    """Verify implicit step forwards tolerance arrays to nested solvers."""
    n = 3
    krylov_atol = np.array([1e-6, 1e-7, 1e-8], dtype=precision)
    krylov_rtol = np.array([1e-4, 1e-5, 1e-6], dtype=precision)
    newton_atol = np.array([1e-3, 1e-4, 1e-5], dtype=precision)
    newton_rtol = np.array([1e-2, 1e-3, 1e-4], dtype=precision)

    step = BackwardsEulerStep(
        precision=precision,
        n=n,
        krylov_atol=krylov_atol,
        krylov_rtol=krylov_rtol,
        newton_atol=newton_atol,
        newton_rtol=newton_rtol,
    )

    assert np.allclose(step.krylov_atol, krylov_atol)
    assert np.allclose(step.krylov_rtol, krylov_rtol)
    assert np.allclose(step.newton_atol, newton_atol)
    assert np.allclose(step.newton_rtol, newton_rtol)


def test_implicit_step_exposes_tolerance_properties(precision):
    """Verify tolerance array properties return correct values."""
    n = 5
    krylov_atol_scalar = 1e-6
    krylov_rtol_scalar = 1e-4
    newton_atol_scalar = 1e-3
    newton_rtol_scalar = 1e-2

    step = BackwardsEulerStep(
        precision=precision,
        n=n,
        krylov_atol=krylov_atol_scalar,
        krylov_rtol=krylov_rtol_scalar,
        newton_atol=newton_atol_scalar,
        newton_rtol=newton_rtol_scalar,
    )

    # Verify arrays have correct shape
    assert step.krylov_atol.shape == (n,)
    assert step.krylov_rtol.shape == (n,)
    assert step.newton_atol.shape == (n,)
    assert step.newton_rtol.shape == (n,)

    # Verify arrays have correct values (scalar broadcast to array)
    assert np.all(step.krylov_atol == precision(krylov_atol_scalar))
    assert np.all(step.krylov_rtol == precision(krylov_rtol_scalar))
    assert np.all(step.newton_atol == precision(newton_atol_scalar))
    assert np.all(step.newton_rtol == precision(newton_rtol_scalar))


def test_implicit_step_linear_solver_newton_atol_returns_none(precision):
    """Verify newton_atol/rtol return None when solver is LinearSolver."""
    n = 3

    step = BackwardsEulerStep(
        precision=precision,
        n=n,
        solver_type='linear',
    )

    # LinearSolver doesn't have newton_atol/rtol, so properties return None
    assert step.newton_atol is None
    assert step.newton_rtol is None

    # But krylov_atol/rtol are still available
    assert step.krylov_atol is not None
    assert step.krylov_rtol is not None
