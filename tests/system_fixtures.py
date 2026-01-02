"""Symbolic system factories shared across the test-suite.

This module centralises the symbolic ODE problems used by the test fixtures so
that integrator tests, CPU reference implementations, and batch solver tests
operate on the same definitions.  Each system exposes lightweight NumPy helper
functions for evaluating derivatives and Jacobians, plus a
``correct_answer_python`` implementation bound to the generated symbolic
system.  These helpers provide fast reference evaluations that mirror the
behaviour of the compiled device functions.
"""

from typing import Sequence, Union

from numpy import asarray, dtype, floating
from numpy.typing import NDArray

from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system

Array = NDArray[floating]


def _as_array(vector: Union[Sequence[float], Array], dt: dtype) -> Array:
    """Return ``vector`` as a one-dimensional array of ``dt``."""

    arr = asarray(vector, dtype=dt)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional array of samples.")
    return arr

THREE_STATE_LINEAR_EQUATIONS = [
    "dx0 = -x0",
    "dx1 = -x1/2",
    "dx2 = -x2/3",
    "o0 = dx0 * p0 + c0 + d0",
    "o1 = dx1 * p1 + c1 + d0",
    "o2 = dx2 * p2 + c2 + d0",
]

THREE_STATE_LINEAR_STATES = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
THREE_STATE_LINEAR_PARAMETERS = {"p0": 1.0, "p1": 2.0, "p2": 3.0}
THREE_STATE_LINEAR_CONSTANTS = {"c0": 0.5, "c1": 1.0, "c2": 2.0}
THREE_STATE_LINEAR_DRIVERS = ["d0"]
THREE_STATE_LINEAR_OBSERVABLES = ["o0", "o1", "o2"]


def build_three_state_linear_system(precision: np.dtype) -> BaseODE:
    """Return the symbolic three-state linear system."""

    system = create_ODE_system(
        dxdt=THREE_STATE_LINEAR_EQUATIONS,
        states=THREE_STATE_LINEAR_STATES,
        parameters=THREE_STATE_LINEAR_PARAMETERS,
        constants=THREE_STATE_LINEAR_CONSTANTS,
        drivers=THREE_STATE_LINEAR_DRIVERS,
        observables=THREE_STATE_LINEAR_OBSERVABLES,
        precision=precision,
        name="three_state_linear",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Three-state nonlinear system
# ---------------------------------------------------------------------------

THREE_STATE_NONLINEAR_EQUATIONS = [
    "dx0 = p0 * (x1 - x0**3) + d0",
    "dx1 = p1 * x0 * x2 - x1 + c1",
    "dx2 = -p2 * x2 + c2 * tanh(x0)",
    "o0 = x0 + c0",
    "o1 = x1**2 + p1",
    "o2 = x2 + d0",
]

THREE_STATE_NONLINEAR_STATES = {"x0": 0.5, "x1": -0.25, "x2": 1.2}
THREE_STATE_NONLINEAR_PARAMETERS = {"p0": 0.7, "p1": 0.9, "p2": 1.1}
THREE_STATE_NONLINEAR_CONSTANTS = {"c0": 0.5, "c1": -0.3, "c2": 0.25}
THREE_STATE_NONLINEAR_DRIVERS = ["d0"]
THREE_STATE_NONLINEAR_OBSERVABLES = ["o0", "o1", "o2"]

def build_three_state_nonlinear_system(precision: np.dtype) -> BaseODE:
    """Return the symbolic three-state nonlinear system."""

    system = create_ODE_system(
        dxdt=THREE_STATE_NONLINEAR_EQUATIONS,
        states=THREE_STATE_NONLINEAR_STATES,
        parameters=THREE_STATE_NONLINEAR_PARAMETERS,
        constants=THREE_STATE_NONLINEAR_CONSTANTS,
        drivers=THREE_STATE_NONLINEAR_DRIVERS,
        observables=THREE_STATE_NONLINEAR_OBSERVABLES,
        precision=precision,
        name="three_state_nonlinear",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Three chamber cardiovascular system (ThreeCM replacement)
# ---------------------------------------------------------------------------

THREE_CHAMBER_EQUATIONS = [
    "P_a = E_a * V_a",
    "P_v = E_v * V_v",
    "P_h = E_h * V_h * d1",
    "Q_i = (P_v - P_h) / R_i if P_v > P_h else 0",
    "Q_o = (P_h - P_a) / R_o if P_h > P_a else 0",
    "Q_c = (P_a - P_v) / R_c",
    "dV_h = Q_i - Q_o",
    "dV_a = Q_o - Q_c",
    "dV_v = Q_c - Q_i",
]

THREE_CHAMBER_STATES = {"V_h": 1.0, "V_a": 1.0, "V_v": 1.0}
THREE_CHAMBER_PARAMETERS = {
    "E_h": 0.52,
    "E_a": 0.0133,
    "E_v": 0.0624,
    "R_i": 0.012,
    "R_o": 1.0,
    "R_c": 1.0 / 114.0,
    "V_s3": 2.0,
}
THREE_CHAMBER_CONSTANTS: dict[str, float] = {}
THREE_CHAMBER_DRIVERS = ["d1"]
THREE_CHAMBER_OBSERVABLES = ["P_a", "P_v", "P_h", "Q_i", "Q_o", "Q_c"]


def build_three_chamber_system(precision: np.dtype) -> BaseODE:
    """Return the symbolic three chamber cardiovascular system."""

    system = create_ODE_system(
        dxdt=THREE_CHAMBER_EQUATIONS,
        states=THREE_CHAMBER_STATES,
        parameters=THREE_CHAMBER_PARAMETERS,
        constants=THREE_CHAMBER_CONSTANTS,
        drivers=THREE_CHAMBER_DRIVERS,
        observables=THREE_CHAMBER_OBSERVABLES,
        precision=precision,
        name="three_chamber_system",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Three-state very stiff nonlinear system
# ---------------------------------------------------------------------------

THREE_STATE_VERY_STIFF_EQUATIONS = [
    "dx0 = -k1 * (x0 - x1) - n0 * x0**3 + d0",
    "dx1 = k1 * (x0 - x1) - k2 * (x1 - x2) - n1 * x1**3",
    "dx2 = k2 * (x1 - x2) - k3 * (x2 - c0) - n2 * x2**3",
    "r0 = x0 - x1",
    "r1 = x1 - x2",
    "r2 = x0 + x1 + x2",
]

THREE_STATE_VERY_STIFF_STATES = {"x0": 0.5, "x1": 0.25, "x2": 0.1}
THREE_STATE_VERY_STIFF_PARAMETERS = {
    "k1": 150.0,
    "k2": 900.0,
    "k3": 1200.0,
    "n0": 40.0,
    "n1": 30.0,
    "n2": 20.0,
}
THREE_STATE_VERY_STIFF_CONSTANTS = {"c0": 0.5}
THREE_STATE_VERY_STIFF_DRIVERS = ["d0"]
THREE_STATE_VERY_STIFF_OBSERVABLES = ["r0", "r1", "r2"]



def build_three_state_very_stiff_system(precision: np.dtype) -> BaseODE:
    """Return the symbolic very stiff nonlinear system."""

    system = create_ODE_system(
        dxdt=THREE_STATE_VERY_STIFF_EQUATIONS,
        states=THREE_STATE_VERY_STIFF_STATES,
        parameters=THREE_STATE_VERY_STIFF_PARAMETERS,
        constants=THREE_STATE_VERY_STIFF_CONSTANTS,
        drivers=THREE_STATE_VERY_STIFF_DRIVERS,
        observables=THREE_STATE_VERY_STIFF_OBSERVABLES,
        precision=precision,
        name="three_state_very_stiff",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Large nonlinear system (100 states)
# ---------------------------------------------------------------------------

_LARGE_SYSTEM_STATE_VALUES = [0.1 + 0.01 * i for i in range(100)]
_LARGE_SYSTEM_PARAMETER_VALUES = [0.5 + 0.005 * i for i in range(100)]
_LARGE_SYSTEM_CONSTANT_VALUES = [
    ((-1) ** i) * (0.01 + 0.002 * i) for i in range(100)
]


def _large_system_equations() -> list[str]:
    """Generate symbolic equations for the large nonlinear system."""

    equations = []
    for idx in range(100):
        nxt = (idx + 1) % 100
        equations.append(
            "dx{idx} = -p{idx}*x{idx} + c{nxt}*sin(x{nxt}) + "
            "0.01*x{idx}*x{nxt} + d0/{denom}".format(
                idx=idx, nxt=nxt, denom=idx + 1
            )
        )
    return equations


LARGE_SYSTEM_EQUATIONS = _large_system_equations()
LARGE_SYSTEM_STATES = {
    f"x{i}": value for i, value in enumerate(_LARGE_SYSTEM_STATE_VALUES)
}
LARGE_SYSTEM_PARAMETERS = {
    f"p{i}": value for i, value in enumerate(_LARGE_SYSTEM_PARAMETER_VALUES)
}
LARGE_SYSTEM_CONSTANTS = {
    f"c{i}": value for i, value in enumerate(_LARGE_SYSTEM_CONSTANT_VALUES)
}
LARGE_SYSTEM_DRIVERS = ["d0"]



def build_large_nonlinear_system(precision: np.dtype) -> BaseODE:
    """Return the symbolic 100-state nonlinear system."""

    system = create_ODE_system(
        dxdt=LARGE_SYSTEM_EQUATIONS,
        states=LARGE_SYSTEM_STATES,
        parameters=LARGE_SYSTEM_PARAMETERS,
        constants=LARGE_SYSTEM_CONSTANTS,
        drivers=LARGE_SYSTEM_DRIVERS,
        precision=precision,
        name="large_nonlinear_system",
        strict=True,
    )

    return system


# ---------------------------------------------------------------------------
# Three-state constant derivative system (all algorithms reduce to Euler)
# ---------------------------------------------------------------------------

THREE_STATE_CONSTANT_DERIV_EQUATIONS = [
    "dx0 = c0",
    "dx1 = c1",
    "dx2 = c2",
    "o0 = x0 + p0",
    "o1 = x1 + p1",
    "o2 = x2 + p2",
]

THREE_STATE_CONSTANT_DERIV_STATES = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
THREE_STATE_CONSTANT_DERIV_PARAMETERS = {"p0": 1.0, "p1": 2.0, "p2": 3.0}
THREE_STATE_CONSTANT_DERIV_CONSTANTS = {"c0": 1.0, "c1": 2.0, "c2": 3.0}
THREE_STATE_CONSTANT_DERIV_DRIVERS = []
THREE_STATE_CONSTANT_DERIV_OBSERVABLES = ["o0", "o1", "o2"]


def build_three_state_constant_deriv_system(precision: np.dtype) -> BaseODE:
    """Return a system with constant derivatives.

    For this system, dx/dt = constant (independent of state), which means
    all higher-order Taylor terms vanish. Therefore, all numerical
    integration algorithms (Euler, RK4, etc.) produce identical results,
    making it ideal for testing algorithm parity.
    """

    system = create_ODE_system(
        dxdt=THREE_STATE_CONSTANT_DERIV_EQUATIONS,
        states=THREE_STATE_CONSTANT_DERIV_STATES,
        parameters=THREE_STATE_CONSTANT_DERIV_PARAMETERS,
        constants=THREE_STATE_CONSTANT_DERIV_CONSTANTS,
        drivers=THREE_STATE_CONSTANT_DERIV_DRIVERS,
        observables=THREE_STATE_CONSTANT_DERIV_OBSERVABLES,
        precision=precision,
        name="three_state_constant_deriv",
        strict=True,
    )

    return system


__all__ = [
    "build_three_state_linear_system",
    "build_three_state_nonlinear_system",
    "build_three_chamber_system",
    "build_three_state_very_stiff_system",
    "build_large_nonlinear_system",
    "build_three_state_constant_deriv_system",
]
