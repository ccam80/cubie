"""Tests for the run status word's transient-failure semantics.

The integration loop accumulates step/controller status bits into an
iteration-scoped temporary that is cleared whenever a step is
accepted, and commits the accumulated bits into the persistent status
word only when the run ends irrecoverably.  A step that fails
transiently (for example an inner linear solve exhausting its
iteration budget) and is recovered at a smaller ``dt`` leaves a clean
status, so a completed run's trajectory survives the default
``nan_error_trajectories=True`` masking, while the flags of a fatal
iteration are preserved for diagnosis.
"""

from math import cos  # noqa: F401 — used inside the ODE function

import numpy as np

import pytest

from cubie import CUBIE_RESULT_CODES, create_ODE_system, solve_ivp


STEP_TOO_SMALL = int(CUBIE_RESULT_CODES.STEP_TOO_SMALL)
MAX_LINEAR = int(CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED)


def _stiff_nonlinear(t, y, p):
    """A stiff two-state system requiring an implicit inner solve."""
    x0, x1 = y[0], y[1]
    k = p["k"]
    dx0 = -k * (x0 - cos(x1))
    dx1 = -x1 + x0
    return [dx0, dx1]


def _build_system():
    """Build the moderately stiff two-state system for the recovery test.

    The recovery scenario needs an oversized first step to exhaust a
    two-iteration Krylov budget while reduced steps converge.  The
    shared fixture systems sit outside that window: the nonlinear and
    three-chamber systems converge even at the initial step (no
    transient failure to observe) and the very stiff system never
    converges under the budget (no recovery).
    """
    return create_ODE_system(
        dxdt=_stiff_nonlinear,
        states={"x0": 1.0, "x1": 0.0},
        parameters={"k": 500.0},
        name="status_staining_stiff",
    )


def test_recovered_transient_failure_reports_success():
    """A run that fails transiently then recovers ends with status 0.

    ``rodas3p`` with a deliberately large initial ``dt`` and a tight
    ``krylov_max_iters`` budget forces the first Rosenbrock stage's inner
    linear solve to exhaust its iterations (``MAX_LINEAR_ITERATIONS_EXCEEDED``)
    before the adaptive controller reduces ``dt`` and the integration
    proceeds to completion with a finite trajectory.  The delivered status
    must be ``0`` and the trajectory must be returned unmasked under the
    default ``nan_error_trajectories=True``.
    """
    system = _build_system()
    duration = 1.0
    result = solve_ivp(
        system=system,
        y0={"x0": np.array([1.0], dtype=np.float64),
            "x1": np.array([0.0], dtype=np.float64)},
        parameters={"k": np.array([500.0], dtype=np.float64)},
        method="rodas3p",
        duration=duration,
        dt=duration,
        dt_min=1e-9,
        dt_max=duration,
        atol=1e-6,
        rtol=1e-3,
        save_every=duration / 10.0,
        krylov_max_iters=2,
        output_types=["state", "time"],
        grid_type="verbatim",
    )

    status_codes = result.status_codes
    assert status_codes is not None
    assert int(status_codes[0]) == 0, (
        f"recovered run reported {result.status_messages}"
    )

    # The full trajectory must survive the default NaN-masking.
    tda = result.time_domain_array
    assert tda.size > 0
    assert np.isfinite(tda).all(), "recovered trajectory was NaN-masked"


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "stiff"}],
    indirect=True,
)
def test_irrecoverable_failure_preserves_fatal_flags(system):
    """A run driven to ``dt_min`` reports the fatal iteration's flags.

    With ``dt_min`` pinned just below ``dt_max`` and tolerances too tight to
    satisfy, the adaptive controller cannot shrink the step far enough and
    signals ``STEP_TOO_SMALL``, ending the run irrecoverably.  The persistent
    status word must carry ``STEP_TOO_SMALL`` together with the step-status
    bit of the fatal iteration (``MAX_LINEAR_ITERATIONS_EXCEEDED``),
    demonstrating that the accumulated bits are committed on an
    irrecoverable end rather than discarded.
    """
    duration = 1.0
    initial_values = {
        name: np.array([value], dtype=np.float64)
        for name, value in zip(
            system.initial_values.names,
            system.initial_values.values_array,
        )
    }
    result = solve_ivp(
        system=system,
        y0=initial_values,
        parameters={},
        method="rodas3p",
        duration=duration,
        dt=0.5,
        dt_min=0.4,
        dt_max=0.5,
        atol=1e-13,
        rtol=1e-13,
        save_every=duration / 10.0,
        output_types=["state", "time"],
        grid_type="verbatim",
        nan_error_trajectories=False,
    )

    status_codes = result.status_codes
    assert status_codes is not None
    fatal = int(status_codes[0])
    assert fatal != 0, "irrecoverable run reported success"
    assert fatal & STEP_TOO_SMALL == STEP_TOO_SMALL, (
        f"STEP_TOO_SMALL missing from {result.status_messages}"
    )
    assert fatal & MAX_LINEAR == MAX_LINEAR, (
        "fatal iteration's step-status bit was not preserved: "
        f"{result.status_messages}"
    )
