"""Regression tests for f32 save-event drift hang.

When ``save_every`` is not exactly representable in float32
(e.g. 0.1), repeated f32 addition of ``next_save += save_every``
accumulates rounding error.  After enough saves, ``next_save``
drifts below its true value while the f64 time accumulator
overshoots it.  This produces a negative ``dt_eff`` at the
event-boundary adjustment, trapping the loop.

The bug triggers when the solver is crawling at small dt near a
save boundary whose f32 ``next_save`` has drifted behind the f64
time.  With save_every=0.1, after ~80 saves the drift is ~5.2 µs.
"""

from __future__ import annotations

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
    """Loop must not hang when f32 save_every accumulation drifts.

    Regression test for negative ``dt_eff`` caused by ``next_save``
    falling behind ``t_prec`` due to f32 rounding of non-binary
    ``save_every``.

    k=3.0 with radau and duration=10.0 reliably triggers the hang
    because the piecewise damping forces the adaptive solver into
    small steps near save boundaries around t≈8, where 80 additions
    of float32(0.1) have drifted ``next_save`` 5.2 µs below its
    true value.
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
