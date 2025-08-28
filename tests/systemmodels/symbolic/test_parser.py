import warnings

import pytest
import sympy as sp

from cubie.systemmodels.symbolic.indexedbasemaps import (
    IndexedBases,
)
from cubie.systemmodels.symbolic.parser import (
    EquationWarning,
    _lhs_pass,
    _process_calls,
    _process_parameters,
    _replace_if,
    _rhs_pass,
    _sanitise_input_math,
    parse_input,
)
from cubie.systemmodels.symbolic.sym_utils import hash_system_definition


class TestInputCleaning:
    """Test input sanitization functions."""

    def test_sanitise_input_math_simple_if(self):
        """Test sanitizing simple if-else expressions."""
        expr = "a if x > 0 else b"
        result = _sanitise_input_math(expr)
        assert "Piecewise" in result
        assert "a" in result and "b" in result

    def test_sanitise_input_math_nested_if(self):
        """Test sanitizing nested if-else expressions."""
        expr = "a if x > 0 else (b if y > 0 else c)"
        result = _sanitise_input_math(expr)
        assert result.count("Piecewise") == 2

    def test_replace_if_simple(self):
        """Test _replace_if with simple expression."""
        expr = "a if x > 0 else b"
        result = _replace_if(expr)
        expected = "Piecewise((a, x > 0), (b, True))"
        assert result == expected

    def test_replace_if_complex_condition(self):
        """Test _replace_if with complex condition."""
        expr = "x + 1 if y < z and w > 0 else x - 1"
        result = _replace_if(expr)
        assert "Piecewise((x + 1, y < z and w > 0), (x - 1, True))" == result

    def test_replace_if_no_match(self):
        """Test _replace_if when no if-else pattern is found."""
        expr = "x + y"
        result = _replace_if(expr)
        assert result == expr


class TestProcessCalls:
    """Test function call processing."""

    def test_process_calls_sympy_functions(self):
        """Test processing known SymPy functions."""
        equations = ["dx = sin(x) + cos(y)", "dy = exp(z)"]
        funcs = _process_calls(equations)

        assert "sin" in funcs
        assert "cos" in funcs
        assert "exp" in funcs
        assert callable(funcs["sin"])

    def test_process_calls_user_functions(self):
        """Test processing user-defined functions."""
        equations = ["dx = custom_func(x)"]
        user_funcs = {"custom_func": lambda x: x**2}

        funcs = _process_calls(equations, user_funcs)
        assert "custom_func" in funcs
        assert funcs["custom_func"] == user_funcs["custom_func"]

    def test_process_calls_unknown_function(self):
        """Test error when unknown function is called."""
        equations = ["dx = unknown_func(x)"]

        with pytest.raises(ValueError, match="function unknown_func"):
            _process_calls(equations)

    def test_process_calls_no_functions(self):
        """Test when no function calls are present."""
        equations = ["dx = x + y"]
        funcs = _process_calls(equations)
        assert funcs == {}


class TestProcessParameters:
    """Test parameter processing."""

    def test_process_parameters_basic(self):
        """Test basic parameter processing."""
        states = ["x", "y"]
        parameters = ["a", "b"]
        constants = ["c"]
        observables = ["o1"]
        drivers = ["d1"]

        ib = _process_parameters(
            states, parameters, constants, observables, drivers
        )

        assert isinstance(ib, IndexedBases)
        assert list(ib.state_names) == states
        assert list(ib.parameter_names) == parameters

    def test_process_parameters_with_dicts(self):
        """Test parameter processing with dictionaries."""
        states = {"x": 1.0, "y": 2.0}
        parameters = {"a": 0.1, "b": 0.2}
        constants = {"c": 3.14}
        observables = ["o1"]
        drivers = ["d1"]

        ib = _process_parameters(
            states, parameters, constants, observables, drivers
        )

        assert "x" in ib.state_names
        assert "y" in ib.state_names


class TestLhsPass:
    """Test left-hand side processing."""

    def test_lhs_pass_basic(self, simple_system_defaults):
        """Test basic LHS processing."""
        (
            states,
            parameters,
            constants,
            drivers,
            observables,
            dxdt_str,
            dxdt_list,
        ) = simple_system_defaults

        ib = _process_parameters(
            states, parameters, constants, observables, drivers
        )
        lines = dxdt_list

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            anon_aux = _lhs_pass(lines, ib)

            # Should create anonymous auxiliary for 'uninited' and 'done'
            assert "uninited" in anon_aux
            assert "done" in ib.dxdt_names
            assert isinstance(anon_aux["uninited"], sp.Symbol)

    def test_lhs_pass_state_derivative(self):
        """Test processing state derivatives."""
        ib = IndexedBases.from_user_inputs(["x"], ["a"], ["c"], [], ["drv"])
        lines = ["dx = x + a"]

        anon_aux = _lhs_pass(lines, ib)
        assert len(anon_aux) == 0  # No anonymous auxiliaries

    def test_lhs_pass_observable_to_state_conversion(self):
        """Test conversion of observable to state when derivative is defined."""
        ib = IndexedBases.from_user_inputs(["x"], ["a"], ["c"], ["y"], ["drv"])
        lines = ["dy = x + a", "dx = y"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = _lhs_pass(lines, ib)

            # Should warn about observable becoming state
            assert len(w) >= 1
            assert issubclass(w[0].category, EquationWarning)

    def test_lhs_pass_invalid_state_assignment(self):
        """Test error when assigning directly to state."""
        ib = IndexedBases.from_user_inputs(
            ["x"], ["a"], ["c"], ["obs"], ["drv"]
        )
        lines = ["x = a + 1"]  # Direct assignment to state

        with pytest.raises(
            ValueError, match="State x cannot be assigned directly"
        ):
            _lhs_pass(lines, ib)

    def test_lhs_pass_immutable_assignment(self):
        """Test error when assigning to immutable inputs."""
        ib = IndexedBases.from_user_inputs(
            ["x"], ["a"], ["c"], ["obs"], ["drv"]
        )
        lines = ["a = x + 1"]  # Assignment to parameter

        with pytest.raises(
            ValueError, match="was entered as an immutable input"
        ):
            _lhs_pass(lines, ib)

    def test_lhs_pass_missing_observables(self):
        """Test error when observables are never assigned."""
        ib = IndexedBases.from_user_inputs(
            ["x"], ["a"], ["c"], ["obs"], ["drv"]
        )
        lines = ["dx = x + a"]  # 'obs' is never assigned

        with pytest.raises(
            ValueError, match="Observables .* are never assigned"
        ):
            _lhs_pass(lines, ib)


class TestRhsPass:
    """Test right-hand side processing."""

    def test_rhs_pass_basic(self):
        """Test basic RHS processing."""
        lines = ["dx = x + a", "obs = sin(x)"]
        symbols = {
            "dx": sp.Symbol("dx", real=True),
            "x": sp.Symbol("x", real=True),
            "a": sp.Symbol("a", real=True),
            "obs": sp.Symbol("obs", real=True),
            "sin": sp.sin,
        }
        dx, x, a, obs, sin = symbols.values()
        expressions, funcs, new_symbols = _rhs_pass(lines, symbols)

        assert expressions[0] == [dx, x + a]
        assert expressions[1] == [obs, sin(x)]
        assert isinstance(expressions[0][1], sp.Expr)

    def test_rhs_pass_undefined_symbol(self):
        """Test error when undefined symbol is used."""
        lines = ["dx = x + undefined_var"]
        symbols = {
            "dx": sp.Symbol("dx", real=True),
            "x": sp.Symbol("x", real=True),
        }

        with pytest.raises(ValueError, match="Undefined symbols"):
            _rhs_pass(lines, symbols)

    def test_rhs_pass_with_if_else(self):
        """Test RHS processing with if-else expressions."""
        lines = ["dx = a if x > 0 else b"]
        symbols = {
            "dx": sp.Symbol("dx", real=True),
            "x": sp.Symbol("x", real=True),
            "a": sp.Symbol("a", real=True),
            "b": sp.Symbol("b", real=True),
        }
        dx, x, a, b = symbols.values()
        from sympy import Piecewise

        expressions, funcs, new_symbols = _rhs_pass(lines, symbols)
        assert expressions[0] == [dx, Piecewise((a, x > 0), (b, True))]


class TestHashSystemDefinition:
    """Test system definition hashing function."""

    def test_hash_system_definition_string(self):
        """Test hashing string input."""
        dxdt = "dx = x + y\ndy = x - y"
        result = hash_system_definition(dxdt, {})
        assert isinstance(result, str)

    def test_hash_system_definition_list(self):
        """Test hashing list input."""
        dxdt = ["dx = x + y", "dy = x - y"]
        result = hash_system_definition(dxdt, {})
        assert isinstance(result, str)

    def test_hash_system_definition_tuple(self):
        """Test hashing tuple input."""
        dxdt = ("dx = x + y", "dy = x - y")
        result = hash_system_definition(dxdt, {})
        assert isinstance(result, str)

    def test_hash_system_definition_consistency(self):
        """Test that equivalent inputs produce same hash."""
        dxdt1 = "dx = x + y\ndy = x - y"
        dxdt2 = ["dx = x + y", "dy = x - y"]
        dxdt3 = "dx=x+y\ndy=x-y"  # Different whitespace

        hash1 = hash_system_definition(dxdt1, {})
        hash2 = hash_system_definition(dxdt2, {})
        hash3 = hash_system_definition(dxdt3, {})

        assert hash1 == hash2 == hash3

    def test_hash_system_definition_different_content(self):
        """Test that different content produces different hashes."""
        dxdt1 = "dx = x + y"
        dxdt2 = "dx = x - y"

        hash1 = hash_system_definition(dxdt1, {})
        hash2 = hash_system_definition(dxdt2, {})

        assert hash1 != hash2

    def test_hash_system_definition_with_constants(self):
        """Test that different constants produce different hashes."""
        dxdt = "dx = c * x + y"
        constants1 = {"c": 1.0}
        constants2 = {"c": 2.0}

        hash1 = hash_system_definition(dxdt, constants1)
        hash2 = hash_system_definition(dxdt, constants2)

        assert hash1 != hash2


class TestParseInput:
    """Test the main parse_input function."""

    def test_parse_input_basic(self, simple_system_defaults):
        """Test basic parsing functionality."""
        (
            states,
            parameters,
            constants,
            drivers,
            observables,
            dxdt_str,
            dxdt_list,
        ) = simple_system_defaults

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            index_map, all_symbols, _, equation_map, fn_hash = parse_input(
                states=states,
                parameters=parameters,
                constants=constants,
                observables=observables,
                drivers=drivers,
                dxdt=dxdt_str,
            )

        assert isinstance(index_map, IndexedBases)
        assert isinstance(all_symbols, dict)
        assert isinstance(equation_map, list)
        assert isinstance(fn_hash, str)  # Changed from int to str

        # Check that we got the expected symbols
        assert "one" in all_symbols  # state
        assert "zebra" in all_symbols  # parameter
        assert "apple" in all_symbols  # constant

    def test_parse_input_with_list(self, simple_system_defaults):
        """Test parsing with list input."""
        (
            states,
            parameters,
            constants,
            drivers,
            observables,
            dxdt_str,
            dxdt_list,
        ) = simple_system_defaults

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            index_map, all_symbols, _, equation_map, fn_hash = parse_input(
                states=states,
                parameters=parameters,
                constants=constants,
                observables=observables,
                drivers=drivers,
                dxdt=dxdt_list,
            )

        assert isinstance(equation_map, list)
        assert len(equation_map) > 0

    def test_parse_input_with_user_functions(self):
        """Test parsing with user-defined functions."""
        states = ["x"]
        parameters = ["a"]
        constants = {}
        observables = ["y"]
        drivers = []
        dxdt = ["dx = custom_func(x)", "y = x"]
        user_functions = {"custom_func": lambda x: x**2}

        index_map, all_symbols, funcs, equation_map, fn_hash = parse_input(
            states=states,
            parameters=parameters,
            constants=constants,
            observables=observables,
            drivers=drivers,
            user_functions=user_functions,
            dxdt=dxdt,
        )

        assert "custom_func" in all_symbols
        assert callable(all_symbols["custom_func"])
        assert funcs["custom_func"] == user_functions["custom_func"]

    def test_parse_input_invalid_dxdt_type(self):
        """Test error with invalid dxdt type."""
        states = ["x"]
        parameters = ["a"]
        constants = []
        observables = []
        drivers = []
        dxdt = 123  # Invalid type

        with pytest.raises(
            ValueError, match="dxdt must be a string or a list/tuple"
        ):
            parse_input(
                states=states,
                parameters=parameters,
                constants=constants,
                observables=observables,
                drivers=drivers,
                dxdt=dxdt,
            )

    def test_parse_input_empty_lines_filtered(self):
        """Test that empty lines are filtered out."""
        states = ["x"]
        parameters = ["a"]
        constants = []
        observables = ["y"]
        drivers = []
        dxdt = ["dx = x + a", "", "  ", "y = x"]  # Contains empty lines
        dx, x, a, y = sp.symbols("dx x a y", real=True)
        index_map, all_symbols, _, equation_map, fn_hash = parse_input(
            states=states,
            parameters=parameters,
            constants=constants,
            observables=observables,
            drivers=drivers,
            dxdt=dxdt,
        )

        assert equation_map[0][0] == dx
        assert equation_map[0][1] == x + a
        assert equation_map[1][0] == y
        assert equation_map[1][1] == x


class TestIntegrationWithFixtures:
    """Test integration with conftest fixtures."""

    def test_with_simple_system_defaults(self, simple_system_defaults):
        """Test parsing the fixture system."""
        (
            states,
            parameters,
            constants,
            drivers,
            observables,
            dxdt_str,
            dxdt_list,
        ) = simple_system_defaults

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # Test both string and list versions
            result1 = parse_input(
                states=states,
                parameters=parameters,
                constants=constants,
                observables=observables,
                drivers=drivers,
                dxdt=dxdt_str,
            )

            result2 = parse_input(
                states=states,
                parameters=parameters,
                constants=constants,
                observables=observables,
                drivers=drivers,
                dxdt=dxdt_list,
            )

        # Results should be equivalent
        index_map1, all_symbols1, _, equation_map1, fn_hash1 = result1
        index_map2, all_symbols2, _, equation_map2, fn_hash2 = result2

        assert fn_hash1 == fn_hash2  # Same hash for equivalent content
        assert equation_map1 == equation_map2

    def test_equation_parsing_with_fixture(self, simple_system_defaults):
        """Test specific equation parsing behaviors with fixture."""
        (
            states,
            parameters,
            constants,
            drivers,
            observables,
            dxdt_str,
            dxdt_list,
        ) = simple_system_defaults

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            index_map, all_symbols, _, equation_map, fn_hash = parse_input(
                states=states,
                parameters=parameters,
                constants=constants,
                observables=observables,
                drivers=drivers,
                dxdt=dxdt_str,
                strict=True,
            )

        # Should have warnings about anonymous auxiliaries
        warning_messages = [str(warning.message) for warning in w]
        assert any("uninited" in msg for msg in warning_messages)
        assigned_to = [expr[0] for expr in equation_map]
        expr = [expr[1] for expr in equation_map]
        # Check that equations were parsed correctly
        assert sp.Symbol("done", real=True) in assigned_to
        assert (
            sp.Symbol("safari", real=True) + sp.Symbol("zoo", real=True)
            == expr[-1]
        )


class TestNonStrictInput:
    """Test non-strict input parsing."""

    def test_simple(self, simple_system_defaults):
        (
            states,
            parameters,
            constants,
            drivers,
            observables,
            dxdt_str,
            dxdt_list,
        ) = simple_system_defaults

        with pytest.raises(ValueError, match="strict"):
            index_map, all_symbols, _, equation_map, fn_hash = parse_input(
                dxdt=dxdt_str, strict=True
            )
        index_map, all_symbols, _, equation_map, fn_hash = parse_input(
                dxdt=dxdt_str,
                strict=False
        )
        assert "apple" in index_map.parameter_names
        assert "zebra" in index_map.parameter_names
        assert "driver1" in index_map.parameter_names
        assert "one" in index_map.state_names
        assert "safari" in index_map.observable_names
        assert "uninited" in index_map.observable_names
        assert "dfoo" in index_map.dxdt_names