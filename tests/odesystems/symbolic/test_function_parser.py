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


    def test_augmented_assignment_correctness(self):
        """Augmented assignment produces correct summed expression."""
        def f(t, y):
            total = y[0]
            total += y[1]
            total += y[2]
            return [-total, total * 0.5, -total * 0.25]

        _, _, _, eqs, _ = parse_input(
            dxdt=f,
            states={"a": 1.0, "b": 0.5, "c": 0.0},
        )
        assert len(eqs.state_derivatives) == 3
        # ``total`` becomes an auxiliary: total = a + b + c
        # Derivatives reference the auxiliary symbol.
        assert len(eqs.auxiliaries) == 1
        aux_lhs, aux_rhs = eqs.auxiliaries[0]
        assert str(aux_lhs) == "total"
        a = sp.Symbol("a", real=True)
        b = sp.Symbol("b", real=True)
        c = sp.Symbol("c", real=True)
        assert sp.simplify(aux_rhs - (a + b + c)) == 0, (
            f"Expected a + b + c, got {aux_rhs}"
        )


    def test_for_loop_unrolling(self):
        """For-loop over range() unrolled to correct equations."""
        def f(t, y, p):
            total = 0.0
            for i in range(3):
                total += y[i] * p[i]
            return [-total, total * 0.5, -total * 0.25]

        _, _, _, eqs, _ = parse_input(
            dxdt=f,
            states={"a": 1.0, "b": 0.5, "c": 0.0},
            parameters={"p0": 0.1, "p1": 0.2, "p2": 0.3},
        )
        assert len(eqs.state_derivatives) == 3
        # ``total`` auxiliary should contain all three terms
        assert len(eqs.auxiliaries) == 1
        aux_rhs = eqs.auxiliaries[0][1]
        a = sp.Symbol("a", real=True)
        b = sp.Symbol("b", real=True)
        c = sp.Symbol("c", real=True)
        p0 = sp.Symbol("p0", real=True)
        p1 = sp.Symbol("p1", real=True)
        p2 = sp.Symbol("p2", real=True)
        expected = a * p0 + b * p1 + c * p2
        assert sp.simplify(aux_rhs - expected) == 0, (
            f"Expected {expected}, got {aux_rhs}"
        )

    def test_tuple_unpacking(self):
        """Tuple unpacking in assignment works."""
        def f(t, y):
            a, b = y[0], y[1]
            return [-a, b]

        _, _, _, eqs, _ = parse_input(
            dxdt=f,
            states={"x": 1.0, "v": 0.5},
        )
        assert len(eqs.state_derivatives) == 2

    def test_if_else_piecewise(self):
        """If/else produces Piecewise in auxiliary equations."""
        def f(t, y):
            if y[0] > 0:
                result = y[0]
            else:
                result = -y[0]
            return [-result]

        _, _, _, eqs, _ = parse_input(
            dxdt=f,
            states={"x": 1.0},
        )
        # The Piecewise appears in the auxiliary for 'result'
        assert len(eqs.auxiliaries) == 1
        _, aux_rhs = eqs.auxiliaries[0]
        assert aux_rhs.has(sp.Piecewise)

    def test_if_elif_else_piecewise(self):
        """If/elif/else chain produces multi-branch Piecewise."""
        def f(t, y):
            if y[0] > 1.0:
                rate = 2.0 * y[0]
            elif y[0] > 0.0:
                rate = y[0]
            else:
                rate = 0.0
            return [-rate]

        _, _, _, eqs, _ = parse_input(
            dxdt=f,
            states={"x": 1.0},
        )
        assert len(eqs.auxiliaries) == 1
        _, aux_rhs = eqs.auxiliaries[0]
        assert aux_rhs.has(sp.Piecewise)


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
