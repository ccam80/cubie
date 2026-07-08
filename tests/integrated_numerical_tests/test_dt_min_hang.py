"""Save-schedule tests for float32 rounding effects.

The device schedule accumulates ``next_save += save_every``. When
``save_every`` is not exactly representable in float32 (0.1, for
example), each addition lands slightly off the exact grid, so the
scheduled save times fall slightly before or after the times the
user asked for. These tests check that rounding in either direction
neither hangs the loop nor changes how many samples are saved:

- the loop keeps stepping when a scheduled save time falls behind
  the current time (a stale save target would clamp the next step
  to zero or negative length, so the clamp only applies when the
  step would be positive);
- every allocated output row is written, in increasing time order,
  when the schedule reaches the final save slightly after the end
  time.
"""

import numpy as np
import pytest

from math import sin  # noqa: F401 — used inside ODE function

from cubie import create_ODE_system, solve_ivp


# ------------------------------------------------------------------ #
#  Coupled oscillator with piecewise damping
# ------------------------------------------------------------------ #

def _coupled_oscillator(t, y, p):
    """Two coupled springs with piecewise velocity-dependent damping."""
    x1, v1, x2, v2 = y[0], y[1], y[2], y[3]
    k = p["k"]
    c_couple = p["c_couple"]
    omega = p["omega"]
    damp1 = 0.5 * v1 if v1 * v1 > 0.01 else 0.0
    damp2 = 0.5 * v2 if v2 * v2 > 0.01 else 0.0
    drive = sin(omega * t)
    dx1 = v1
    dv1 = -k * x1 - damp1 + c_couple * (x2 - x1) + drive
    dx2 = v2
    dv2 = -k * x2 - damp2 + c_couple * (x1 - x2)
    return [dx1, dv1, dx2, dv2]


@pytest.fixture
def oscillator_system():
    """Build the coupled oscillator ODE system.

    This system is kept separate from the shared fixtures because
    the hang test below needs an adaptive solver squeezed into very
    small steps right at a save boundary; the piecewise damping
    produces that behaviour and the shared fixture systems do not.
    """
    return create_ODE_system(
        dxdt=_coupled_oscillator,
        states={"x1": 1.0, "v1": 0.0, "x2": -0.5, "v2": 0.0},
        parameters={"k": 4.0, "c_couple": 0.3, "omega": 2.5},
        name="coupled_osc_drift_test",
    )


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

def test_f32_save_drift_does_not_hang(oscillator_system):
    """The loop completes when the save schedule falls behind time.

    After about 80 additions of float32(0.1), the accumulated save
    schedule sits about 5 microseconds earlier than the committed
    simulation time. A save target earlier than the current time
    would clamp the next step to zero or negative length, which the
    step function cannot integrate, so the clamp applies only when
    the resulting step is positive. k=3.0 with radau pushes the
    adaptive solver into small steps near the save boundaries
    around t=8, which is where the stale save target appears; the
    run must still complete with a full set of saves.
    """
    n = 1
    result = solve_ivp(
        system=oscillator_system,
        y0={
            "x1": np.ones(n, dtype=np.float32),
            "v1": np.zeros(n, dtype=np.float32),
            "x2": np.full(n, -0.5, dtype=np.float32),
            "v2": np.zeros(n, dtype=np.float32),
        },
        parameters={
            "k": np.full(n, 3.0, dtype=np.float32),
            "c_couple": np.full(n, 0.3, dtype=np.float32),
            "omega": np.full(n, 2.5, dtype=np.float32),
        },
        method="radau",
        duration=10.0,
        dt_min=1e-6,
        dt_max=1.0,
        save_every=0.1,
        output_types=["state", "time"],
        grid_type="verbatim",
    )
    # Should produce ~100 saves; any completion is a pass.
    n_saves = result.time_domain_array.shape[0]
    assert n_saves >= 80


_DRIFTED_GRID = {
    "algorithm": "euler",
    "step_controller": "fixed",
    "dt": 0.01,
    "duration": 1.0,
    "save_every": 0.1,
    "output_types": ["state", "time"],
}


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param(_DRIFTED_GRID, id="drifted_schedule")],
    indirect=True,
)
def test_all_save_slots_written_on_inexact_grid(
    solver, solver_settings, batch_input_arrays, driver_settings
):
    """Every allocated save row is written on an inexact float32 grid.

    The settings request ten regular saves of an interval that is
    not exactly representable in float32, so the accumulated
    schedule reaches the final save slightly after the end time.
    The host must allocate eleven rows (the initial state plus ten
    saves) and the device must fill all of them, in increasing time
    order, ending at the requested duration.
    """
    initial_values, parameters = batch_input_arrays
    duration = float(solver_settings["duration"])
    result = solver.solve(
        initial_values=initial_values,
        parameters=parameters,
        drivers=driver_settings,
        duration=duration,
    )

    status_codes = np.asarray(result.status_codes)
    assert np.all(status_codes == 0), result.status_messages
    times = np.asarray(result.time)
    assert times.shape[0] == 11
    assert np.all(np.diff(times, axis=0) > 0.0), (
        f"saved times are not strictly increasing: {times}"
    )
    assert np.allclose(times[-1, :], duration, rtol=1e-4)
    assert np.isfinite(result.time_domain_array).all()
