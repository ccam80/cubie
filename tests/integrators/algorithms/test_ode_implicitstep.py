"""Tests for ODEImplicitStep tolerance parameter routing."""

import numpy as np
import pytest

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
    """Verify newton_atol/rtol return None when solver is MRLinearSolver."""
    n = 3

    step = BackwardsEulerStep(
        precision=precision,
        n=n,
        solver_type='linear',
    )

    # MRLinearSolver doesn't have newton_atol/rtol, so properties return None
    assert step.newton_atol is None
    assert step.newton_rtol is None

    # But krylov_atol/rtol are still available
    assert step.krylov_atol is not None
    assert step.krylov_rtol is not None


def test_implicit_step_rejects_invalid_solver_type(precision):
    """Constructing with an unknown solver_type raises ValueError."""
    with pytest.raises(ValueError, match="solver_type must be"):
        BackwardsEulerStep(
            precision=precision,
            n=3,
            solver_type='not_a_real_solver',
        )


def test_implicit_config_settings_dict_includes_implicit_fields(precision):
    """ImplicitStepConfig.settings_dict merges base and implicit fields."""
    step = BackwardsEulerStep(precision=precision, n=3)
    settings = step.compile_settings.settings_dict
    assert settings['beta'] == step.compile_settings.beta
    assert settings['gamma'] == step.compile_settings.gamma
    assert 'M' not in settings
    assert (
        settings['preconditioner_order']
        == step.compile_settings.preconditioner_order
    )
    assert (
        settings['preconditioner_type']
        == step.compile_settings.preconditioner_type
    )
    assert (
        settings['get_solver_helper_fn']
        == step.compile_settings.get_solver_helper_fn
    )


def test_implicit_step_beta_gamma_properties(precision):
    """beta and gamma forward to compile_settings."""
    step = BackwardsEulerStep(precision=precision, n=3)
    assert step.beta == step.compile_settings.beta
    assert step.gamma == step.compile_settings.gamma


def test_implicit_step_preconditioner_type_property(precision):
    """preconditioner_type forwards to compile_settings."""
    step = BackwardsEulerStep(
        precision=precision, n=3, preconditioner_type='jacobi',
    )
    assert step.preconditioner_type == 'jacobi'


def test_implicit_step_update_invokes_register_buffers_override(precision):
    """update() dispatches to ODEImplicitStep's no-op register_buffers."""
    step = BackwardsEulerStep(precision=precision, n=3)
    recognised = step.update(newton_atol=1e-5)
    assert 'newton_atol' in recognised


def test_implicit_step_settings_dict_merges_solver_settings(precision):
    """ODEImplicitStep.settings_dict merges algorithm and solver keys."""
    step = BackwardsEulerStep(precision=precision, n=3)
    settings = step.settings_dict
    solver_settings = step.solver.settings_dict
    for key, value in solver_settings.items():
        assert key in settings


_RESIDUAL_SETTINGS = {
    "krylov_residual_reduction": 0.2,
    "krylov_residual_floor": 0.03,
}

_RESIDUAL_ARRANGEMENTS = [
    {**_RESIDUAL_SETTINGS, "algorithm": "backwards_euler"},
    {
        **_RESIDUAL_SETTINGS,
        "algorithm": "backwards_euler",
        "linear_correction_type": "bicgstab",
    },
    {**_RESIDUAL_SETTINGS, "algorithm": "ros3p"},
]
_RESIDUAL_IDS = ["newton-mr", "newton-bicgstab", "direct-linear"]


@pytest.mark.parametrize(
    "solver_settings_override",
    _RESIDUAL_ARRANGEMENTS,
    ids=_RESIDUAL_IDS,
    indirect=True,
)
def test_implicit_step_routes_residual_settings(step_object, precision):
    """Every solver arrangement routes the linear stopping settings."""
    step = step_object
    assert step.krylov_residual_reduction == precision(0.2)
    assert step.krylov_residual_floor == precision(0.03)
    assert step.settings_dict["krylov_residual_reduction"] == precision(0.2)
    assert step.settings_dict["krylov_residual_floor"] == precision(0.03)


@pytest.mark.parametrize(
    "solver_settings_override",
    _RESIDUAL_ARRANGEMENTS,
    ids=_RESIDUAL_IDS,
    indirect=True,
)
def test_implicit_step_updates_residual_settings(
    step_object_mutable, precision
):
    """update() reroutes the linear stopping settings to the solver."""
    step = step_object_mutable
    recognized = step.update(
        krylov_residual_reduction=0.25,
        krylov_residual_floor=0.04,
    )
    assert {
        "krylov_residual_reduction",
        "krylov_residual_floor",
    } <= recognized
    assert step.krylov_residual_reduction == precision(0.25)
    assert step.krylov_residual_floor == precision(0.04)
