"""End-to-end coverage for inner-solver tolerance defaults.

An implicit adaptive method solves each stage with an inner Krylov/Newton
solver.  If the inner-solver tolerance is not tighter than the step
controller's tolerance, the stage-solve residual noise floors the embedded
error estimate, so a controller tolerance at or below the inner-solver
tolerance can reject every step down to ``dt_min``.

Leaving ``krylov_atol``/``krylov_rtol``/``newton_atol``/``newton_rtol`` unset
now defaults them to the controller atol/rtol divided by ten.  These tests
confirm the derived default is wired through the whole public ``Solver`` stack
and that a real GPU solve with the new defaults still succeeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from cubie import create_ODE_system, solve_ivp
from cubie.batchsolving.solver import Solver


def _decay(t, y, p):
    """Stiff linear decay ``u' = -k u``."""
    return [-p["k"] * y[0]]


@pytest.fixture
def decay_system():
    """Build the scalar stiff-decay ODE system."""
    return create_ODE_system(
        dxdt=_decay,
        states={"u": 1.0},
        parameters={"k": 50.0},
        name="stiff_decay_tolerance_test",
    )


def test_default_inner_tolerances_wired_through_solver(decay_system):
    """The controller-scaled default reaches the real inner solver.

    Exercises the full public construction path (``Solver`` kwarg routing ->
    ``BatchSolverKernel`` -> ``SingleIntegratorRun``) to confirm the derived
    default lands on the solver the device step actually reads.
    """
    solver = Solver(
        decay_system,
        algorithm="ros3p",
        step_controller="pid",
        atol=1e-8,
        rtol=1e-8,
        dt_min=1e-10,
        dt_max=0.1,
    )
    integrator = solver.kernel.single_integrator
    controller = integrator._step_controller
    algo = integrator._algo_step

    assert controller.is_adaptive
    expected = np.asarray(controller.atol) / 10.0
    assert np.allclose(expected, 1e-9)
    assert np.allclose(algo.krylov_atol, expected)
    assert np.allclose(algo.krylov_rtol, np.asarray(controller.rtol) / 10.0)


def test_explicit_inner_tolerance_wins_through_solver(decay_system):
    """An explicit krylov tolerance passed to ``Solver`` is not overwritten."""
    solver = Solver(
        decay_system,
        algorithm="ros3p",
        step_controller="pid",
        atol=1e-8,
        rtol=1e-8,
        dt_min=1e-10,
        dt_max=0.1,
        krylov_atol=4e-5,
    )
    algo = solver.kernel.single_integrator._algo_step
    assert np.allclose(algo.krylov_atol, 4e-5)


def test_ros3p_default_tolerances_solve_succeeds(decay_system):
    """ROS3P at atol=rtol=1e-8 solves without explicit krylov tolerances.

    The scalar solve returns status 0 and a decaying solution; the run is a
    regression guard that the derived defaults do not break the device path.
    """
    duration = 0.2
    result = solve_ivp(
        system=decay_system,
        y0={"u": np.ones(1, dtype=np.float64)},
        parameters={"k": np.full(1, 50.0, dtype=np.float64)},
        method="ros3p",
        duration=duration,
        dt_min=1e-10,
        dt_max=0.1,
        atol=1e-8,
        rtol=1e-8,
        save_every=duration,
        output_types=["state", "time"],
        grid_type="verbatim",
    )

    status = np.asarray(result.status_codes).ravel()
    assert np.all(status == 0), result.status_messages

    final = np.asarray(result.time_domain_array).ravel()[-1]
    assert np.isfinite(final)
    assert 0.0 < final < 1.0
