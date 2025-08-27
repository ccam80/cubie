import warnings

import pytest
import sympy as sp

from cubie.systemmodels.symbolic.parser import (
    EquationWarning,
    IndexedBaseMap,
    IndexedBases,
    _lhs_pass,
    _process_calls,
    _process_parameters,
    _replace_if,
    _rhs_pass,
    _sanitise_input_math,
    hash_dxdt,
    parse_input,
)


class TestIndexedBaseMap:
    """Test cases for IndexedBaseMap class."""

    def test_init_basic(self):
        """Test basic initialization of IndexedBaseMap."""
        symbol_labels = ["x", "y", "z"]
        base_map = IndexedBaseMap("test", symbol_labels)

        # Create the expected symbols
        x, y, z = (
            sp.Symbol("x", real=True),
            sp.Symbol("y", real=True),
            sp.Symbol("z", real=True),
        )

        assert base_map.base_name == "test"
        assert base_map.length == 3
        assert base_map.real is True
        assert isinstance(base_map.base, sp.IndexedBase)
        assert len(base_map.index_map) == 3
        assert len(base_map.ref_map) == 3
        assert len(base_map.symbol_map) == 3

        # Check index mappings (keyed by symbol)
        assert base_map.index_map[x] == 0
        assert base_map.index_map[y] == 1
        assert base_map.index_map[z] == 2

        # Check references (keyed by symbol)
        assert str(base_map.ref_map[x]) == "test[0]"
        assert str(base_map.ref_map[y]) == "test[1]"
        assert str(base_map.ref_map[z]) == "test[2]"

        # Check symbol mappings (keyed by string)
        assert base_map.symbol_map["x"] == x
        assert base_map.symbol_map["y"] == y
        assert base_map.symbol_map["z"] == z

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        symbol_labels = ["a", "b"]
        defaults = [1.0, 2.0]
        base_map = IndexedBaseMap(
            "test", symbol_labels, input_defaults=defaults
        )

        # Create the expected symbols
        a, b = sp.Symbol("a", real=True), sp.Symbol("b", real=True)

        assert base_map.default_values == {a: 1.0, b: 2.0}

    def test_init_with_custom_length(self):
        """Test initialization with custom length."""
        symbol_labels = ["x", "y"]
        base_map = IndexedBaseMap("test", symbol_labels, length=5)

        assert base_map.length == 5
        assert base_map.base.shape == (5,)

    def test_init_not_real(self):
        """Test initialization with complex numbers."""
        symbol_labels = ["x"]
        base_map = IndexedBaseMap("test", symbol_labels, real=False)

        assert not base_map.real
        assert not base_map.base.assumptions0.get("real")

    def test_init_defaults_length_mismatch(self):
        """Test error when defaults length doesn't match symbols."""
        symbol_labels = ["x", "y"]
        defaults = [1.0]  # Wrong length

        with pytest.raises(
            ValueError, match="Input defaults must be the same length"
        ):
            IndexedBaseMap("test", symbol_labels, input_defaults=defaults)

    def test_pop_symbol(self):
        """Test removing a symbol from the map."""
        symbol_labels = ["x", "y", "z"]
        base_map = IndexedBaseMap("test", symbol_labels)

        # Create the expected symbols
        x, y, z = (
            sp.Symbol("x", real=True),
            sp.Symbol("y", real=True),
            sp.Symbol("z", real=True),
        )

        initial_length = base_map.length
        base_map.pop(y)

        assert y not in base_map.ref_map
        assert y not in base_map.index_map
        assert "y" not in base_map.symbol_map
        assert base_map.length == initial_length - 1
        assert base_map.base.shape == (2,)

    def test_push_symbol(self):
        """Test adding a symbol to the map."""
        symbol_labels = ["x", "y"]
        base_map = IndexedBaseMap("test", symbol_labels)

        initial_length = base_map.length
        z = sp.Symbol("z", real=True)

        base_map.push(z)
        assert z in base_map.ref_map
        assert z in base_map.index_map
        assert base_map.symbol_map["z"] == z
        assert base_map.index_map[z] == initial_length
        assert base_map.length == initial_length + 1
        assert str(base_map.ref_map[z]) == f"test[{initial_length}]"


class TestIndexedBases:
    """Test cases for IndexedBases class."""

    @pytest.fixture
    def sample_indexed_bases(self):
        """Create a sample IndexedBases instance for testing."""
        states = IndexedBaseMap("state", ["x", "y"])
        parameters = IndexedBaseMap("param", ["a", "b"])
        constants = IndexedBaseMap("const", ["c"])
        observables = IndexedBaseMap("obs", ["o1"])
        drivers = IndexedBaseMap("drv", ["d1"])
        dxdt = IndexedBaseMap("dxdt", ["dx", "dy"])

        return IndexedBases(
            states, parameters, constants, observables, drivers, dxdt
        )

    def test_init(self, sample_indexed_bases):
        """Test IndexedBases initialization."""
        ib = sample_indexed_bases

        assert isinstance(ib.states, IndexedBaseMap)
        assert isinstance(ib.parameters, IndexedBaseMap)
        assert isinstance(ib.constants, IndexedBaseMap)
        assert isinstance(ib.observables, IndexedBaseMap)
        assert isinstance(ib.drivers, IndexedBaseMap)
        assert isinstance(ib.dxdt, IndexedBaseMap)

        # Check all_indices combines all mappings
        assert len(ib.all_indices) == 9  # 2+2+1+1+1+2 symbols

    def test_from_user_inputs(self):
        """Test creating IndexedBases from user inputs."""
        states = ["x", "y"]
        parameters = ["a", "b"]
        constants = ["c"]
        observables = ["o1"]
        drivers = ["d1"]

        ib = IndexedBases.from_user_inputs(
            states, parameters, constants, observables, drivers
        )

        assert list(ib.state_names) == ["x", "y"]
        assert list(ib.parameter_names) == ["a", "b"]
        assert list(ib.constant_names) == ["c"]
        assert list(ib.observable_names) == ["o1"]
        assert list(ib.driver_names) == ["d1"]
        assert list(ib.dxdt_names) == ["dx", "dy"]

    def test_from_user_inputs_with_defaults(self):
        """Test creating IndexedBases from user inputs with default values."""
        states = {"x": 1.5, "y": 2.3}
        parameters = {"a": 0.1, "b": 0.2}
        constants = {"c": 3.14}
        observables = ["o1"]
        drivers = ["d1"]

        ib = IndexedBases.from_user_inputs(
            states, parameters, constants, observables, drivers
        )

        # Check that default values are properly passed through
        assert ib.state_values[sp.Symbol("x", real=True)] == 1.5
        assert ib.state_values[sp.Symbol("y", real=True)] == 2.3
        assert ib.parameter_values[sp.Symbol("a", real=True)] == 0.1
        assert ib.parameter_values[sp.Symbol("b", real=True)] == 0.2
        assert ib.constant_values[sp.Symbol("c", real=True)] == 3.14

    def test_getters(self):
        """Test all getter properties for equality with symbols, strings, and values."""
        # Create test data with default values
        states_dict = {"x": 1.0, "y": 2.0}
        params_dict = {"a": 0.1, "b": 0.2}
        constants_dict = {"c": 3.14, "d": 2.71}
        observables_list = ["obs1", "obs2"]
        drivers_list = ["drv1"]

        ib = IndexedBases.from_user_inputs(
            states=states_dict,
            parameters=params_dict,
            constants=constants_dict,
            observables=observables_list,
            drivers=drivers_list,
        )

        # Test state getters
        state_names = ib.state_names
        state_symbols = ib.state_symbols
        state_values = ib.state_values

        assert set(state_names) == {"x", "y"}
        assert len(state_symbols) == len(state_names)
        assert all(isinstance(sym, sp.Symbol) for sym in state_symbols)
        assert set(str(sym) for sym in state_symbols) == set(state_names)
        assert len(state_values) == len(state_names)
        assert state_values[sp.Symbol("x", real=True)] == 1.0
        assert state_values[sp.Symbol("y", real=True)] == 2.0

        # Test parameter getters
        param_names = ib.parameter_names
        param_symbols = ib.parameter_symbols
        param_values = ib.parameter_values

        assert set(param_names) == {"a", "b"}
        assert len(param_symbols) == len(param_names)
        assert all(isinstance(sym, sp.Symbol) for sym in param_symbols)
        assert set(str(sym) for sym in param_symbols) == set(param_names)
        assert len(param_values) == len(param_names)
        assert param_values[sp.Symbol("a", real=True)] == 0.1
        assert param_values[sp.Symbol("b", real=True)] == 0.2

        # Test constant getters
        const_names = ib.constant_names
        const_symbols = ib.constant_symbols
        const_values = ib.constant_values

        assert set(const_names) == {"c", "d"}
        assert len(const_symbols) == len(const_names)
        assert all(isinstance(sym, sp.Symbol) for sym in const_symbols)
        assert set(str(sym) for sym in const_symbols) == set(const_names)
        assert len(const_values) == len(const_names)
        assert const_values[sp.Symbol("c", real=True)] == 3.14
        assert const_values[sp.Symbol("d", real=True)] == 2.71

        # Test observable getters
        obs_names = ib.observable_names
        obs_symbols = ib.observable_symbols

        assert set(obs_names) == {"obs1", "obs2"}
        assert len(obs_symbols) == len(obs_names)
        assert all(isinstance(sym, sp.Symbol) for sym in obs_symbols)
        assert set(str(sym) for sym in obs_symbols) == set(obs_names)

        # Test driver getters
        drv_names = ib.driver_names
        drv_symbols = ib.driver_symbols

        assert set(drv_names) == {"drv1"}
        assert len(drv_symbols) == len(drv_names)
        assert all(isinstance(sym, sp.Symbol) for sym in drv_symbols)
        assert set(str(sym) for sym in drv_symbols) == set(drv_names)

        # Test dxdt getters
        dxdt_names = ib.dxdt_names
        dxdt_symbols = ib.dxdt_symbols

        assert set(dxdt_names) == {"dx", "dy"}
        assert len(dxdt_symbols) == len(dxdt_names)
        assert all(isinstance(sym, sp.Symbol) for sym in dxdt_symbols)
        assert set(str(sym) for sym in dxdt_symbols) == set(dxdt_names)

    def test_properties(self, sample_indexed_bases):
        """Test properties of IndexedBases."""
        ib = sample_indexed_bases

        assert "x" in ib.state_names
        assert "y" in ib.state_names
        assert "a" in ib.parameter_names
        assert "b" in ib.parameter_names
        assert "c" in ib.constant_names
        assert "o1" in ib.observable_names
        assert "d1" in ib.driver_names
        assert "dx" in ib.dxdt_names
        assert "dy" in ib.dxdt_names

    def test_getitem(self, sample_indexed_bases):
        """Test __getitem__ method."""
        ib = sample_indexed_bases

        # Should be able to access any symbol
        x_ref = ib[sp.Symbol("x", real=True)]
        assert str(x_ref) == "state[0]"

        a_ref = ib[sp.Symbol("a", real=True)]
        assert str(a_ref) == "param[0]"

    def test_all_symbols(self, sample_indexed_bases):
        """Test all_symbols property."""
        ib = sample_indexed_bases
        symbols = ib.all_symbols

        assert len(symbols) == 9
        assert "x" in symbols
        assert "a" in symbols
        assert "c" in symbols


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
        expressions, funcs = _rhs_pass(lines, symbols)

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

        expressions, funcs = _rhs_pass(lines, symbols)
        assert expressions[0] == [dx, Piecewise((a, x > 0), (b, True))]


class TestHashDxdt:
    """Test dxdt hashing function."""

    def test_hash_dxdt_string(self):
        """Test hashing string input."""
        dxdt = "dx = x + y\ndy = x - y"
        result = hash_dxdt(dxdt)
        assert isinstance(result, int)

    def test_hash_dxdt_list(self):
        """Test hashing list input."""
        dxdt = ["dx = x + y", "dy = x - y"]
        result = hash_dxdt(dxdt)
        assert isinstance(result, int)

    def test_hash_dxdt_tuple(self):
        """Test hashing tuple input."""
        dxdt = ("dx = x + y", "dy = x - y")
        result = hash_dxdt(dxdt)
        assert isinstance(result, int)

    def test_hash_dxdt_consistency(self):
        """Test that equivalent inputs produce same hash."""
        dxdt1 = "dx = x + y\ndy = x - y"
        dxdt2 = ["dx = x + y", "dy = x - y"]
        dxdt3 = "dx=x+y\ndy=x-y"  # Different whitespace

        hash1 = hash_dxdt(dxdt1)
        hash2 = hash_dxdt(dxdt2)
        hash3 = hash_dxdt(dxdt3)

        assert hash1 == hash2 == hash3

    def test_hash_dxdt_different_content(self):
        """Test that different content produces different hashes."""
        dxdt1 = "dx = x + y"
        dxdt2 = "dx = x - y"

        hash1 = hash_dxdt(dxdt1)
        hash2 = hash_dxdt(dxdt2)

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
        assert isinstance(fn_hash, int)

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
        constants = []
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
            )

        # Should have warnings about anonymous auxiliaries
        warning_messages = [str(warning.message) for warning in w]
        assert any("uninited" in msg for msg in warning_messages)
        assigned_to = [expr[0] for expr in equation_map]
        expr = [expr[1] for expr in equation_map]
        # Check that equations were parsed correctly
        assert sp.Symbol('done', real=True) in assigned_to
        assert sp.Symbol('safari', real=True)  + sp.Symbol('zoo', real=True) == expr[-1]

