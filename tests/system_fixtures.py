"""Symbolic system factories shared across the test-suite.

This module centralises the symbolic ODE problems used by the test fixtures so
that integrator tests, CPU reference implementations, and batch solver tests
operate on the same definitions.  Each system exposes lightweight NumPy helper
functions for evaluating derivatives and Jacobians, plus a
``correct_answer_python`` implementation bound to the generated symbolic
system.  These helpers provide fast reference evaluations that mirror the
behaviour of the compiled device functions.
"""

from __future__ import annotations

from types import MethodType
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system

Array = NDArray[np.floating]


def _as_array(vector: Sequence[float] | Array, dtype: np.dtype) -> Array:
    """Return ``vector`` as a one-dimensional array of ``dtype``."""

    array = np.asarray(vector, dtype=dtype)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array of samples.")
    return array


# def _bind_python_helpers(
#     system: BaseODE,
#     dxdt_fn: Callable[[Array, Array, Array, Array], Array],
#     jac_fn: Callable[[Array, Array, Array, Array], Array],
#     observables_fn: Callable[
#         [Array, Array, Array, Array, Array], Array
#     ],
# ) -> None:
#     """Attach Python evaluation helpers to ``system``."""
#     pass
    # def correct_answer_python(
    #     self: BaseODE,
    #     state: Sequence[float] | Array,
    #     parameters: Sequence[float] | Array,
    #     drivers: Sequence[float] | Array,
    # ) -> tuple[Array, Array]:
    #     dtype = self.precision
    #     state_arr = _as_array(state, dtype)
    #     param_arr = _as_array(parameters, dtype)
    #     driver_arr = _as_array(drivers, dtype)
    #     const_arr = self.constants.values_array.astype(dtype)
    #     dxdt = dxdt_fn(state_arr, param_arr, driver_arr, const_arr)
    #     observables = observables_fn(
    #         state_arr, param_arr, driver_arr, const_arr, dxdt
    #     )
    #     return dxdt.astype(dtype, copy=False), observables.astype(
    #         dtype, copy=False
    #     )
    #
    # system.correct_answer_python = MethodType(correct_answer_python, system)
    # system.python_dxdt = dxdt_fn
    # system.python_jacobian = jac_fn
    # system.python_observables = observables_fn


# ---------------------------------------------------------------------------
# Three-state linear system (Decays replacement)
# ---------------------------------------------------------------------------

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

#
# def pydxdt_three_state_linear(
#     state: Array, parameters: Array, drivers: Array, constants: Array
# ) -> Array:
#     """Derivative helper for the three-state linear system."""
#
#     dtype = state.dtype
#     dxdt = np.empty(3, dtype=dtype)
#     dxdt[0] = -state[0]
#     dxdt[1] = -0.5 * state[1]
#     dxdt[2] = -(1.0 / 3.0) * state[2]
#     return dxdt
#
#
# def pyjac_three_state_linear(
#     state: Array, parameters: Array, drivers: Array, constants: Array
# ) -> Array:
#     """Jacobian helper for the three-state linear system."""
#
#     dtype = state.dtype
#     jacobian = np.zeros((3, 3), dtype=dtype)
#     jacobian[0, 0] = -1.0
#     jacobian[1, 1] = -0.5
#     jacobian[2, 2] = -(1.0 / 3.0)
#     return jacobian
#
#
# def pyobservables_three_state_linear(
#     state: Array,
#     parameters: Array,
#     drivers: Array,
#     constants: Array,
#     dxdt: Array,
# ) -> Array:
#     """Observable helper for the three-state linear system."""
#
#     dtype = state.dtype
#     driver = drivers[0] if drivers.size else dtype.type(0)
#     observables = np.empty(3, dtype=dtype)
#     observables = dxdt * parameters
#     observables += constants[:3]
#     observables += driver
#     return observables.astype(dtype)


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
    # system.build()
    # _bind_python_helpers(
    #     system,
    #     pydxdt_three_state_linear,
    #     pyjac_three_state_linear,
    #     pyobservables_three_state_linear,
    # )
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

#
# def pydxdt_three_state_nonlinear(
#     state: Array, parameters: Array, drivers: Array, constants: Array
# ) -> Array:
#     """Derivative helper for the three-state nonlinear system."""
#
#     dtype = state.dtype
#     driver = drivers[0] if drivers.size else dtype.type(0)
#     dxdt = np.empty(3, dtype=dtype)
#     dxdt[0] = parameters[0] * (state[1] - state[0] ** 3) + driver
#     dxdt[1] = parameters[1] * state[0] * state[2] - state[1] + constants[1]
#     dxdt[2] = -parameters[2] * state[2]
#     dxdt[2] += constants[2] * np.tanh(state[0])
#     return dxdt
#
#
# def pyjac_three_state_nonlinear(
#     state: Array, parameters: Array, drivers: Array, constants: Array
# ) -> Array:
#     """Jacobian helper for the three-state nonlinear system."""
#
#     dtype = state.dtype
#     jacobian = np.zeros((3, 3), dtype=dtype)
#     jacobian[0, 0] = -3.0 * parameters[0] * (state[0] ** 2)
#     jacobian[0, 1] = parameters[0]
#     jacobian[1, 0] = parameters[1] * state[2]
#     jacobian[1, 1] = -1.0
#     jacobian[1, 2] = parameters[1] * state[0]
#     sech2 = 1.0 - np.tanh(state[0]) ** 2
#     jacobian[2, 0] = constants[2] * sech2
#     jacobian[2, 2] = -parameters[2]
#     return jacobian
#
#
# def pyobservables_three_state_nonlinear(
#     state: Array,
#     parameters: Array,
#     drivers: Array,
#     constants: Array,
#     dxdt: Array,
# ) -> Array:
#     """Observable helper for the three-state nonlinear system."""
#
#     dtype = state.dtype
#     driver = drivers[0] if drivers.size else dtype.type(0)
#     observables = np.empty(3, dtype=dtype)
#     observables[0] = state[0] + constants[0]
#     observables[1] = state[1] ** 2 + parameters[1]
#     observables[2] = state[2] + driver
#     return observables


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
    # system.build()
    # _bind_python_helpers(
    #     system,
    #     pydxdt_three_state_nonlinear,
    #     pyjac_three_state_nonlinear,
    #     pyobservables_three_state_nonlinear,
    # )
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

#
# def _three_chamber_pressures(
#     state: Array, parameters: Array, drivers: Array
# ) -> tuple[np.floating, np.floating, np.floating]:
#     dtype = state.dtype
#     driver = drivers[0] if drivers.size else dtype.type(0)
#     p_a = parameters[1] * state[1]
#     p_v = parameters[2] * state[2]
#     p_h = parameters[0] * state[0] * driver
#     return p_a, p_v, p_h


# def pydxdt_three_chamber(
#     state: Array, parameters: Array, drivers: Array, constants: Array
# ) -> Array:
#     """Derivative helper for the three chamber cardiovascular model."""
#
#     dtype = state.dtype
#     p_a, p_v, p_h = _three_chamber_pressures(state, parameters, drivers)
#     r_i, r_o, r_c = parameters[3], parameters[4], parameters[5]
#
#     if p_v > p_h:
#         q_i = (p_v - p_h) / r_i
#     else:
#         q_i = dtype.type(0)
#
#     if p_h > p_a:
#         q_o = (p_h - p_a) / r_o
#     else:
#         q_o = dtype.type(0)
#
#     q_c = (p_a - p_v) / r_c
#     dxdt = np.empty(3, dtype=dtype)
#     dxdt[0] = q_i - q_o
#     dxdt[1] = q_o - q_c
#     dxdt[2] = q_c - q_i
#     return dxdt
#
#
# def pyjac_three_chamber(
#     state: Array, parameters: Array, drivers: Array, constants: Array
# ) -> Array:
#     """Jacobian helper mirroring the ThreeCM device implementation."""
#
#     dtype = state.dtype
#     p_a, p_v, p_h = _three_chamber_pressures(state, parameters, drivers)
#     r_i, r_o, r_c = parameters[3], parameters[4], parameters[5]
#     e_h, e_a, e_v = parameters[0], parameters[1], parameters[2]
#     driver = drivers[0] if drivers.size else dtype.type(0)
#
#     qi_active = p_v > p_h
#     qo_active = p_h > p_a
#
#     jacobian = np.zeros((3, 3), dtype=dtype)
#
#     dqi_dvh = -(e_h * driver) / r_i if qi_active else dtype.type(0)
#     dqi_dvv = e_v / r_i if qi_active else dtype.type(0)
#     dqo_dvh = (e_h * driver) / r_o if qo_active else dtype.type(0)
#     dqo_dva = -(e_a) / r_o if qo_active else dtype.type(0)
#     dqc_dva = e_a / r_c
#     dqc_dvv = -e_v / r_c
#
#     jacobian[0, 0] = dqi_dvh - dqo_dvh
#     jacobian[0, 1] = -dqo_dva
#     jacobian[0, 2] = dqi_dvv
#
#     jacobian[1, 0] = dqo_dvh
#     jacobian[1, 1] = dqo_dva - dqc_dva
#     jacobian[1, 2] = -dqc_dvv
#
#     jacobian[2, 0] = -dqi_dvh
#     jacobian[2, 1] = dqc_dva
#     jacobian[2, 2] = dqc_dvv - dqi_dvv
#     return jacobian
#
#
# def pyobservables_three_chamber(
#     state: Array,
#     parameters: Array,
#     drivers: Array,
#     constants: Array,
#     dxdt: Array,
# ) -> Array:
#     """Observable helper for the three chamber cardiovascular model."""
#
#     dtype = state.dtype
#     p_a, p_v, p_h = _three_chamber_pressures(state, parameters, drivers)
#     r_i, r_o, r_c = parameters[3], parameters[4], parameters[5]
#
#     if p_v > p_h:
#         q_i = (p_v - p_h) / r_i
#     else:
#         q_i = dtype.type(0)
#
#     if p_h > p_a:
#         q_o = (p_h - p_a) / r_o
#     else:
#         q_o = dtype.type(0)
#
#     q_c = (p_a - p_v) / r_c
#     return np.array([p_a, p_v, p_h, q_i, q_o, q_c], dtype=dtype)
#

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
    # system.build()
    # _bind_python_helpers(
    #     system,
    #     pydxdt_three_chamber,
    #     pyjac_three_chamber,
    #     pyobservables_three_chamber,
    # )
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

#
# def pydxdt_three_state_very_stiff(
#     state: Array, parameters: Array, drivers: Array, constants: Array
# ) -> Array:
#     """Derivative helper for the very stiff nonlinear system."""
#
#     dtype = state.dtype
#     driver = drivers[0] if drivers.size else dtype.type(0)
#     k1, k2, k3, n0, n1, n2 = parameters[:6]
#     c0 = constants[0]
#
#     dxdt = np.empty(3, dtype=dtype)
#     dxdt[0] = -k1 * (state[0] - state[1])
#     dxdt[0] -= n0 * state[0] ** 3
#     dxdt[0] += driver
#
#     dxdt[1] = k1 * (state[0] - state[1])
#     dxdt[1] -= k2 * (state[1] - state[2])
#     dxdt[1] -= n1 * state[1] ** 3
#
#     dxdt[2] = k2 * (state[1] - state[2])
#     dxdt[2] -= k3 * (state[2] - c0)
#     dxdt[2] -= n2 * state[2] ** 3
#
#     return dxdt
#
#
# def pyjac_three_state_very_stiff(
#     state: Array, parameters: Array, drivers: Array, constants: Array
# ) -> Array:
#     """Jacobian helper for the very stiff nonlinear system."""
#
#     dtype = state.dtype
#     k1, k2, k3, n0, n1, n2 = parameters[:6]
#
#     jacobian = np.zeros((3, 3), dtype=dtype)
#
#     jacobian[0, 0] = -k1 - 3.0 * n0 * (state[0] ** 2)
#     jacobian[0, 1] = k1
#
#     jacobian[1, 0] = k1
#     jacobian[1, 1] = -k1 - k2 - 3.0 * n1 * (state[1] ** 2)
#     jacobian[1, 2] = k2
#
#     jacobian[2, 1] = k2
#     jacobian[2, 2] = -k2 - k3 - 3.0 * n2 * (state[2] ** 2)
#
#     return jacobian
#
#
# def pyobservables_three_state_very_stiff(
#     state: Array,
#     parameters: Array,
#     drivers: Array,
#     constants: Array,
#     dxdt: Array,
# ) -> Array:
#     """Observable helper for the very stiff nonlinear system."""
#
#     dtype = state.dtype
#     observables = np.empty(3, dtype=dtype)
#     observables[0] = state[0] - state[1]
#     observables[1] = state[1] - state[2]
#     observables[2] = state[0] + state[1] + state[2]
#     return observables


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
    # system.build()
    # _bind_python_helpers(
    #     system,
    #     pydxdt_three_state_very_stiff,
    #     pyjac_three_state_very_stiff,
    #     pyobservables_three_state_very_stiff,
    # )
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

#
# def pydxdt_large_nonlinear(
#     state: Array, parameters: Array, drivers: Array, constants: Array
# ) -> Array:
#     """Derivative helper for the 100-state nonlinear system."""
#
#     dtype = state.dtype
#     driver = drivers[0] if drivers.size else dtype.type(0)
#     n = state.size
#     dxdt = np.empty(n, dtype=dtype)
#     for idx in range(n):
#         nxt = (idx + 1) % n
#         dxdt[idx] = -parameters[idx] * state[idx]
#         dxdt[idx] += constants[nxt] * np.sin(state[nxt])
#         dxdt[idx] += 0.01 * state[idx] * state[nxt]
#         dxdt[idx] += driver / (idx + 1)
#     return dxdt
#
#
# def pyjac_large_nonlinear(
#     state: Array, parameters: Array, drivers: Array, constants: Array
# ) -> Array:
#     """Jacobian helper for the 100-state nonlinear system."""
#
#     dtype = state.dtype
#     n = state.size
#     jacobian = np.zeros((n, n), dtype=dtype)
#     for idx in range(n):
#         nxt = (idx + 1) % n
#         jacobian[idx, idx] = -parameters[idx] + 0.01 * state[nxt]
#         jacobian[idx, nxt] = constants[nxt] * np.cos(state[nxt])
#         jacobian[idx, nxt] += 0.01 * state[idx]
#     return jacobian
#
#
# def pyobservables_large_nonlinear(
#     state: Array,
#     parameters: Array,
#     drivers: Array,
#     constants: Array,
#     dxdt: Array,
# ) -> Array:
#     """Observable helper for the 100-state nonlinear system."""
#
#     return np.empty(0, dtype=state.dtype)


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
    # system.build()
    # _bind_python_helpers(
    #     system,
    #     pydxdt_large_nonlinear,
    #     pyjac_large_nonlinear,
    #     pyobservables_large_nonlinear,
    # )
    return system


__all__ = [
    "build_three_state_linear_system",
    "build_three_state_nonlinear_system",
    "build_three_chamber_system",
    "build_three_state_very_stiff_system",
    "build_large_nonlinear_system",
    # "pydxdt_three_state_linear",
    # "pydxdt_three_state_nonlinear",
    # "pydxdt_three_chamber",
    # "pydxdt_three_state_very_stiff",
    # "pydxdt_large_nonlinear",
    # "pyjac_three_state_linear",
    # "pyjac_three_state_nonlinear",
    # "pyjac_three_chamber",
    # "pyjac_three_state_very_stiff",
    # "pyjac_large_nonlinear",
]
