"""Numerical tests for the rosenbrock23 tableau family.

The rosenbrock23 tableau (aliased as ode23s and rosenbrock23_sciml)
is stored in the transformed increment convention of Hairer & Wanner
(Solving ODEs II, IV.7.17).  These tests pin the order-2 solution and
order-3 embedded estimate on stiff linear decay: consistent tableau
coefficients are required for the adaptive controller to accept steps
at all.
"""

import numpy as np
import pytest

from cubie import create_ODE_system, Solver


@pytest.fixture(scope="module")
def stiff_decay_system():
    """Linear decay u' = -50 u, autonomous and stiff at unit scale."""
    return create_ODE_system(
        ["du1 = -50.0 * u1"],
        states=["u1"],
        precision=np.float64,
        name="stiff_decay_regression",
    )


@pytest.fixture(scope="module")
def forced_decay_system():
    """Forced decay u' = -50 (u - cos t); exercises gamma_stages."""
    return create_ODE_system(
        ["du1 = -50.0 * (u1 - cos(t))"],
        states=["u1"],
        precision=np.float64,
        name="forced_decay_regression",
    )


@pytest.mark.parametrize(
    "algorithm", ["rosenbrock23", "ode23s", "rosenbrock23_sciml"]
)
def test_rosenbrock23_solves_stiff_decay(stiff_decay_system, algorithm):
    """Adaptive solve succeeds with accurate output at tol 1e-6.

    Krylov tolerances sit well below the controller tolerance so the
    test isolates the tableau from the stage-solve residual floor on
    the embedded error estimate.
    """
    solver = Solver(
        stiff_decay_system,
        algorithm=algorithm,
        step_controller="pi",
        atol=1e-6,
        rtol=1e-6,
        krylov_atol=1e-12,
        krylov_rtol=1e-10,
        save_every=0.1,
        output_types=["state", "time"],
    )
    result = solver.solve(
        {"u1": np.array([1.0])}, {}, duration=1.0, t0=0.0,
        grid_type="verbatim",
    )
    status = int(np.asarray(result.status_codes)[0])
    final = float(np.asarray(result.time_domain_array)[-1, 0, 0])
    assert status == 0
    assert abs(final - np.exp(-50.0)) < 1e-4


def test_rosenbrock23_solves_forced_decay(forced_decay_system):
    """Nonautonomous stiff solve stays accurate (time-derivative path)."""
    solver = Solver(
        forced_decay_system,
        algorithm="rosenbrock23",
        step_controller="pi",
        atol=1e-8,
        rtol=1e-8,
        krylov_atol=1e-12,
        krylov_rtol=1e-10,
        save_every=0.1,
        output_types=["state", "time"],
    )
    result = solver.solve(
        {"u1": np.array([0.0])}, {}, duration=1.0, t0=0.0,
        grid_type="verbatim",
    )
    status = int(np.asarray(result.status_codes)[0])
    final = float(np.asarray(result.time_domain_array)[-1, 0, 0])
    reference = (
        (2500.0 * np.cos(1.0) + 50.0 * np.sin(1.0)) / 2501.0
        - 2500.0 / 2501.0 * np.exp(-50.0)
    )
    assert status == 0
    assert abs(final - reference) < 1e-5
