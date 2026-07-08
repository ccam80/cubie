"""Save-schedule integrity tests under f32 accumulation drift.

The loop accumulates ``next_save += save_every``, so a ``save_every``
that is inexact in float32 (e.g. 0.1) moves the schedule by ulps
per event in either direction. The schedule-completion test carries
a half-interval margin past ``t_end``, and the boundary clamp
applies only to positive gaps, so drift in either direction can
neither strand a host-allocated slot nor trap the loop. These tests
assert both halves of that invariant:

- the loop terminates even when ``next_save`` falls behind the f64
  time accumulator (with save_every=0.1, ~5.2 µs after ~80 saves),
  because a non-positive gap falls through to the raw step;
- every host-allocated save slot is written, including the final
  one, when drift pushes the last scheduled save past ``t_end``.
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
    """Build the coupled oscillator ODE system."""
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
    """The loop completes when f32 save_every accumulation drifts.

    k=3.0 with radau and duration=10.0 exercises the drifted-
    schedule path: the piecewise damping forces the adaptive
    solver into small steps near save boundaries around t≈8,
    where 80 additions of float32(0.1) leave ``next_save`` 5.2 µs
    below the f64 time accumulator. The positive-gap-only clamp
    keeps ``dt_eff`` positive there, so the run must finish with
    a full set of saves.
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


def test_final_save_slot_written_on_inexact_grid(oscillator_system):
    """Every host-allocated save slot is written, including the last.

    With duration=1.0 and save_every=0.1 in float32 the host
    allocates eleven rows (``floor(duration / save_every) + 1``)
    while the accumulated schedule reaches 1.0000001 for the final
    save. The half-interval stop margin keeps that save inside the
    schedule, and the saved time column exposes any slot left
    unwritten regardless of what the memory happens to contain.
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
        method="dormand-prince-54",
        duration=1.0,
        dt_min=1e-6,
        dt_max=1.0,
        save_every=0.1,
        output_types=["state", "time"],
        grid_type="verbatim",
    )

    assert int(result.status_codes[0]) == 0, result.status_messages
    times = result.time[:, 0]
    assert times.shape[0] == 11
    assert np.all(np.diff(times) > 0.0), (
        f"saved times are not strictly increasing: {times}"
    )
    assert times[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(result.time_domain_array).all()
