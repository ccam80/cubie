"""Numerical solve of a structurally simplified torn DAE.

Integrates ``dx/dt = -z`` with the implicit constraint
``z**5 + z = x`` (not explicitly solvable, so ``z`` is a torn
algebraic state under a singular mass matrix) and compares against a
high-accuracy reference computed on the reduced ODE with a per-step
Newton solve for ``z``.
"""

import numpy as np
import pytest

from cubie import Solver, solve_ivp
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system


@pytest.fixture(scope="module")
def torn_ode():
    return create_ODE_system(
        dxdt="""
        dx = -z
        0 = z**5 + z - x
        """,
        states={"x": 2.0, "z": 1.0},
        precision=np.float64,
        simplify=True,
        name="dae_guard_torn",
    )


def test_torn_system_rejects_explicit_algorithm(torn_ode):
    # The singular mass matrix cannot be consumed by an explicit
    # step; silently ignoring it would integrate the constraint
    # residuals as derivatives.
    with pytest.raises(ValueError, match="implicit algorithm"):
        Solver(torn_ode, algorithm="euler")


def test_torn_system_rejects_user_mass_matrix(torn_ode):
    # The structural mass matrix is paired to the simplifier's
    # state ordering; a user override is rejected.
    with pytest.raises(ValueError, match="cannot override"):
        Solver(
            torn_ode,
            algorithm="backwards_euler",
            algorithm_settings={"M": np.eye(2)},
        )


def z_of_x(x):
    """Solve z**5 + z = x by Newton iteration."""

    z = x / 2.0
    for _ in range(60):
        f = z**5 + z - x
        z = z - f / (5.0 * z**4 + 1.0)
    return z


def reference_solution(x0, t_end, n_steps):
    """RK4 on the reduced ODE dx/dt = -z(x)."""

    dt = t_end / n_steps
    x = x0
    for _ in range(n_steps):
        k1 = -z_of_x(x)
        k2 = -z_of_x(x + 0.5 * dt * k1)
        k3 = -z_of_x(x + 0.5 * dt * k2)
        k4 = -z_of_x(x + dt * k3)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return x


@pytest.mark.slow
def test_torn_dae_solution_matches_reference():
    ode = create_ODE_system(
        dxdt="""
        dx = -z
        0 = z**5 + z - x
        """,
        states={"x": 2.0, "z": 1.0},
        precision=np.float64,
        simplify=True,
        name="dae_solve_torn",
    )
    t_end = 0.2
    result = solve_ivp(
        ode,
        y0={"x": np.array([2.0]), "z": np.array([1.0])},
        method="backwards_euler",
        duration=t_end,
        dt=1e-3,
        save_every=0.1,
        newton_atol=1e-10,
        newton_rtol=1e-10,
        krylov_atol=1e-12,
        krylov_rtol=1e-12,
    )
    legend = {
        label: idx
        for idx, label in result.time_domain_legend.items()
    }
    trajectory = result.time_domain_array
    x_final = float(trajectory[-1, legend["x"], 0])
    z_final = float(trajectory[-1, legend["z"], 0])

    x_ref = reference_solution(2.0, t_end, 4000)
    z_ref = z_of_x(x_ref)

    # Backward Euler global error is O(dt); dt=1e-3 over t=0.2 with
    # |dz/dt| < 1 keeps it well inside 2e-3.
    assert x_final == pytest.approx(x_ref, abs=2e-3)
    # The algebraic constraint must hold at the solution.
    assert z_final**5 + z_final - x_final == pytest.approx(
        0.0, abs=1e-5
    )
    assert z_final == pytest.approx(z_ref, abs=2e-3)
