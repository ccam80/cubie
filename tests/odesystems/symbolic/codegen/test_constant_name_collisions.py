"""End-to-end guards against user names aliasing generated bindings.

Every user constant is emitted into generated factories through a
single prefixed local (``CONSTANT_ALIAS_PREFIX``), so a model constant
named after any factory-scope binding (solver scalings, loop bounds,
tableau metadata, generated body locals, even ``precision`` itself)
can neither replace that binding nor be replaced by it. These tests
build a system whose constants are named after each class of internal
binding and compare explicit and implicit solves against an identical
system with unremarkable names.
"""

import numpy as np
import pytest

from cubie import create_ODE_system, solve_ivp
from cubie.odesystems.symbolic.codegen.dxdt import (
    generate_dxdt_fac_code,
)
from cubie.odesystems.symbolic.sym_utils import CONSTANT_ALIAS_PREFIX


# One constant per class of generated binding: factory arguments
# (beta, gamma, order), factory locals (n, total_n, stage_width,
# beta_inv, h_eff_factor, precision), FIRK tableau metadata (c_0,
# a_0_0), and generated body locals (dx_0).
_HOSTILE_CONSTANTS = {
    "beta": 8.0 / 3.0,
    "gamma": 0.5,
    "order": 2.0,
    "n": 1.5,
    "total_n": 0.25,
    "stage_width": 0.125,
    "beta_inv": 0.75,
    "h_eff_factor": 0.375,
    "precision": 1.25,
    "c_0": 0.6,
    "a_0_0": 0.3,
    "dx_0": 0.2,
}

_HOSTILE_EQUATION = (
    "dx = -(beta + gamma + n + dx_0)*x - 0.1*x**order"
    " + c_0 + a_0_0 + beta_inv + h_eff_factor"
    " + total_n + stage_width + precision"
)

_SAFE_CONSTANTS = {
    f"k_{index}": value
    for index, value in enumerate(_HOSTILE_CONSTANTS.values())
}

_SAFE_EQUATION = (
    "dx = -(k_0 + k_1 + k_3 + k_11)*x - 0.1*x**k_2"
    " + k_9 + k_10 + k_6 + k_7 + k_4 + k_5 + k_8"
)


def _solve(system, method):
    result = solve_ivp(
        system,
        y0={"x": 2.0},
        method=method,
        duration=0.2,
        dt=0.01,
        save_every=0.05,
    )
    assert not np.any(result.status_codes)
    return result.time_domain_array


def _hostile_system(precision, name):
    return create_ODE_system(
        _HOSTILE_EQUATION,
        states={"x": 2.0},
        constants=dict(_HOSTILE_CONSTANTS),
        precision=precision,
        name=name,
    )


def _safe_system(precision, name):
    return create_ODE_system(
        _SAFE_EQUATION,
        states={"x": 2.0},
        constants=dict(_SAFE_CONSTANTS),
        precision=precision,
        name=name,
    )


def test_hostile_names_solve_explicit(precision):
    """Explicit solves are unaffected by hostile constant names."""
    hostile = _hostile_system(precision, "hostile_names_explicit")
    safe = _safe_system(precision, "safe_names_explicit")
    np.testing.assert_allclose(
        _solve(hostile, "euler"),
        _solve(safe, "euler"),
        rtol=1e-6,
    )


def test_hostile_names_solve_single_stage_implicit(precision):
    """Single-stage residual/operator/Neumann helpers stay isolated.

    ``backwards_euler`` compiles the single-stage templates whose
    factory scope binds ``beta``, ``gamma``, ``order``, ``n``,
    ``beta_inv``, and ``h_eff_factor`` — every one shadowed here by a
    same-named model constant.
    """
    hostile = _hostile_system(precision, "hostile_names_be")
    safe = _safe_system(precision, "safe_names_be")
    np.testing.assert_allclose(
        _solve(hostile, "backwards_euler"),
        _solve(safe, "backwards_euler"),
        rtol=1e-6,
    )


def test_hostile_names_solve_firk(precision):
    """Flattened FIRK helpers keep tableau metadata isolated.

    The ``n_stage_*`` templates bind ``total_n``, ``stage_width``,
    and the tableau metadata symbols ``c_0``/``a_0_0`` at factory
    scope — every one shadowed here by a same-named model constant.
    """
    hostile = _hostile_system(precision, "hostile_names_firk")
    safe = _safe_system(precision, "safe_names_firk")
    np.testing.assert_allclose(
        _solve(hostile, "firk"),
        _solve(safe, "firk"),
        rtol=1e-6,
    )


def test_hostile_constants_emit_only_prefixed_loads(precision):
    """Generated source loads every hostile constant prefixed."""
    hostile = _hostile_system(precision, "hostile_names_source")
    code = generate_dxdt_fac_code(
        hostile.equations, hostile.indices
    )
    for name in _HOSTILE_CONSTANTS:
        load = (
            f"{CONSTANT_ALIAS_PREFIX}{name} = "
            f"precision(constants['{name}'])"
        )
        assert load in code
        # No unprefixed binding of the user name anywhere in the
        # factory body.
        assert f"\n    {name} = " not in code


def test_reserved_prefix_names_are_rejected(precision):
    """User symbols may not enter the generated-code namespace."""
    with pytest.raises(ValueError, match="reserved"):
        create_ODE_system(
            "dx = -_cubie_codegen_k*x",
            states={"x": 2.0},
            constants={"_cubie_codegen_k": 1.0},
            precision=precision,
            name="reserved_prefix_rejected",
        )
