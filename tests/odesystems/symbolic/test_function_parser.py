"""Tests for function_parser module and end-to-end callable integration."""

import math

import pytest
import sympy as sp

from cubie.odesystems.symbolic.parsing.parser import (
    ParsedEquations,
    parse_input,
)
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system


class TestParseInput:
    """End-to-end tests through parse_input with callable dxdt."""

    def test_simple_exponential_decay(self):
        """Single-state exponential decay via function."""
        def f(t, y):
            return [-0.1 * y[0]]

        index_map, syms, fns, eqs, h = parse_input(
            dxdt=f,
            states={"x": 1.0},
        )
        assert len(eqs.state_derivatives) == 1
        assert len(index_map.state_names) == 1

    def test_two_state_with_params(self):
        """Damped oscillator with parameter k."""
        def f(t, y, p):
            return [-p[0] * y[1], y[0]]

        index_map, syms, fns, eqs, h = parse_input(
            dxdt=f,
            states={"velocity": 1.0, "position": 0.0},
            parameters={"k": 0.5},
        )
        assert len(eqs.state_derivatives) == 2

    def test_named_states_positional(self):
        """Named states accessed by positional index."""
        def f(t, y):
            return [-y[0], y[0] - y[1]]

        index_map, syms, fns, eqs, h = parse_input(
            dxdt=f,
            states={"a": 0.0, "b": 1.0},
        )
        assert len(eqs.state_derivatives) == 2

    def test_string_indexed_states(self):
        """String subscript state access."""
        def f(t, y):
            return [-y["velocity"]]

        index_map, syms, fns, eqs, h = parse_input(
            dxdt=f,
            states={"velocity": 1.0},
        )
        assert len(eqs.state_derivatives) == 1

    def test_constants_attribute_access(self):
        """Constants accessed via attribute pattern."""
        def f(t, y, c):
            return [-c.damping * y[0]]

        index_map, syms, fns, eqs, h = parse_input(
            dxdt=f,
            states={"x": 1.0},
            constants={"damping": 0.1},
        )
        assert len(eqs.state_derivatives) == 1

    def test_constants_string_subscript(self):
        """Constants accessed via string subscript."""
        def f(t, y, c):
            return [-c["damping"] * y[0]]

        index_map, syms, fns, eqs, h = parse_input(
            dxdt=f,
            states={"x": 1.0},
            constants={"damping": 0.1},
        )
        assert len(eqs.state_derivatives) == 1

    def test_equivalence_with_string_input(self):
        """Function input produces equivalent equations to string input."""
        def f(t, y):
            return [-0.1 * y[0]]

        _, _, _, eqs_func, _ = parse_input(
            dxdt=f,
            states={"x": 1.0},
        )

        _, _, _, eqs_str, _ = parse_input(
            dxdt="dx = -0.1 * x",
            states={"x": 1.0},
        )

        assert len(eqs_func.state_derivatives) == len(
            eqs_str.state_derivatives
        )
        # Both should have one derivative equation for dx
        func_rhs = eqs_func.state_derivatives[0][1]
        str_rhs = eqs_str.state_derivatives[0][1]
        assert sp.simplify(func_rhs - str_rhs) == 0

    def test_math_functions(self):
        """Math functions like sin are converted."""
        def f(t, y):
            from math import sin
            return [sin(y[0])]

        index_map, syms, fns, eqs, h = parse_input(
            dxdt=f,
            states={"x": 1.0},
        )
        assert len(eqs.state_derivatives) == 1

    def test_local_alias(self):
        """Local variable aliasing a state does not create auxiliary."""
        def f(t, y):
            v = y[0]
            return [-0.1 * v]

        _, _, _, eqs, _ = parse_input(
            dxdt=f,
            states={"x": 1.0},
        )
        # Alias should not appear as auxiliary
        assert len(eqs.auxiliaries) == 0
        assert len(eqs.state_derivatives) == 1


class TestCreateODESystem:
    """Test create_ODE_system with callable dxdt."""

    def test_simple(self):
        """Basic callable creates SymbolicODE."""
        def f(t, y):
            return [-y[0]]

        ode = create_ODE_system(
            dxdt=f,
            states={"x": 1.0},
        )
        assert ode.num_states == 1

    def test_with_parameters(self):
        """Callable with parameter argument."""
        def f(t, y, p):
            return [-p[0] * y[0]]

        ode = create_ODE_system(
            dxdt=f,
            states={"x": 1.0},
            parameters={"k": 0.5},
        )
        assert ode.num_states == 1


class TestDetectInputType:
    """Test that callable detection works in _detect_input_type."""

    def test_callable_detected(self):
        """Callable dxdt returns 'function' type."""
        from cubie.odesystems.symbolic.parsing.parser import (
            _detect_input_type,
        )

        def f(t, y):
            return [-y[0]]

        assert _detect_input_type(f) == "function"

    def test_string_still_works(self):
        """String dxdt still returns 'string' type."""
        from cubie.odesystems.symbolic.parsing.parser import (
            _detect_input_type,
        )
        assert _detect_input_type("dx = -x") == "string"
