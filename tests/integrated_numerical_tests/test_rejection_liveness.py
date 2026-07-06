"""Regression tests for adaptive-rejection liveness (issue #529).

On the HIRES benchmark several adaptive implicit solvers entered a
perpetual-rejection limit cycle: a failed linear solve injected a
huge error, the controller's history term then grew dt back into the
failing regime on the next rejected step, and neither the stagnation
guard (blind to rejected steps) nor STEP_TOO_SMALL (dt pinned far
above dt_min) ever fired.  The kernel spun forever at 100% GPU.

The fixes under test: adaptive controllers never grow dt on a
rejected step, and the loop declares a run irrecoverable
(MAX_LOOP_ITERS_EXCEEDED) after too many consecutive rejections.
"""

from __future__ import annotations

import numpy as np
import pytest

from cubie import create_ODE_system, Solver

HIRES_DURATION = 321.8122


@pytest.fixture(scope="module")
def hires_system():
    """Stiff HIRES benchmark (Hairer & Wanner)."""
    equations = [
        "du1 = -1.71*u1 + 0.43*u2 + 8.32*u3 + 0.0007",
        "du2 = 1.71*u1 - 8.75*u2",
        "du3 = -10.03*u3 + 0.43*u4 + 0.035*u5",
        "du4 = 8.32*u2 + 1.71*u3 - 1.12*u4",
        "du5 = -1.745*u5 + 0.43*u6 + 0.43*u7",
        "du6 = -280.0*u6*u8 + 0.69*u4 + 1.71*u5 - 0.43*u6 + 0.69*u7",
        "du7 = 280.0*u6*u8 - 1.81*u7",
        "du8 = -280.0*u6*u8 + 1.81*u7",
    ]
    return create_ODE_system(
        equations,
        states=[f"u{i}" for i in range(1, 9)],
        precision=np.float64,
        name="hires_liveness",
    )


def solve_hires(system, algorithm, atol, rtol):
    solver = Solver(
        system,
        algorithm=algorithm,
        step_controller="pi",
        atol=atol,
        rtol=rtol,
        dt_min=1e-10 * HIRES_DURATION,
        dt_max=HIRES_DURATION,
        save_every=HIRES_DURATION / 128,
        output_types=["state", "time"],
    )
    initial = {
        f"u{i}": np.array([value]) for i, value in
        zip(range(1, 9), [1.0, 0, 0, 0, 0, 0, 0, 0.0057])
    }
    return solver.solve(
        initial, {}, duration=HIRES_DURATION, t0=0.0,
        grid_type="verbatim", nan_error_trajectories=False,
    )


@pytest.mark.parametrize(
    "algorithm, atol, rtol",
    [
        ("rodas3p", 1e-6, 1e-3),
        ("rodas3p", 1e-7, 1e-4),
        ("rodas3p", 1e-8, 1e-5),
        ("rodas3p", 1e-9, 1e-6),
        ("rodas3p", 1e-10, 1e-7),
        ("ros3p", 1e-8, 1e-5),
        pytest.param(
            "radau_iia_5", 1e-6, 1e-3, marks=pytest.mark.slow
        ),
        pytest.param(
            "radau_iia_5", 1e-7, 1e-4, marks=pytest.mark.slow
        ),
    ],
)
def test_hires_returns_from_former_hang_configs(
    hires_system, algorithm, atol, rtol
):
    """Formerly-hanging configs either finish or fail decodably.

    A run may carry transient solver flags from rejected attempts
    (status staining); the liveness contract is that it either covers
    the full time span with finite output or terminates with a
    liveness flag — never spins forever.
    """
    result = solve_hires(hires_system, algorithm, atol, rtol)
    flags = result.status_messages.get(0, [])
    died = (
        "STEP_TOO_SMALL" in flags
        or "MAX_LOOP_ITERS_EXCEEDED" in flags
    )
    if not died:
        trajectory = np.asarray(result.time_domain_array)[..., 0]
        # Output arrays are preallocated; unwritten rows stay zero,
        # so the last written save is the maximum time.
        last_saved = float(np.max(np.asarray(result.time)))
        assert np.all(np.isfinite(trajectory))
        assert last_saved >= HIRES_DURATION - 2 * (
            HIRES_DURATION / 128
        )


def test_hires_negative_control_completes(hires_system):
    """l_stable_sdirk_4 never hung and must still complete cleanly."""
    result = solve_hires(hires_system, "l_stable_sdirk_4", 1e-10, 1e-7)
    status = int(np.asarray(result.status_codes)[0])
    trajectory = np.asarray(result.time_domain_array)[..., 0]
    assert status == 0
    assert np.all(np.isfinite(trajectory))
