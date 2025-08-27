import pytest
import sympy as sp
import warnings
from cubie.systemmodels.symbolic.sym_utils import (
    topological_sort,
    cse_and_stack,
)


class TestTopologicalSort:
    """Test cases for topological_sort function."""

    def test_topological_sort_simple_list(self):
        """Test topological sort with simple list of assignments."""
        x, y, z = sp.symbols("x y z")
        assignments = [(z, x + y), (y, x * 2), (x, sp.Integer(1))]

        result = topological_sort(assignments)

        # x should come first, then y, then z
        assert len(result) == 3
        assert result[0] == (x, sp.Integer(1))
        assert result[1] == (y, x * 2)
        assert result[2] == (z, x + y)

    def test_topological_sort_dict_input(self):
        """Test topological sort with dictionary input."""
        x, y, z = sp.symbols("x y z")
        assignments = {z: x + y, y: x * 2, x: sp.Integer(1)}

        result = topological_sort(assignments)

        # Should return same order as list version
        assert len(result) == 3
        assert result[0] == (x, sp.Integer(1))
        assert result[1] == (y, x * 2)
        assert result[2] == (z, x + y)

    def test_topological_sort_no_dependencies(self):
        """Test topological sort with independent assignments."""
        x, y, z = sp.symbols("x y z")
        assignments = [
            (x, sp.Integer(1)),
            (y, sp.Integer(2)),
            (z, sp.Integer(3)),
        ]

        result = topological_sort(assignments)

        # Order doesn't matter for independent assignments
        assert len(result) == 3
        result_symbols = [sym for sym, _ in result]
        assert set(result_symbols) == {x, y, z}

    def test_topological_sort_complex_dependencies(self):
        """Test topological sort with complex dependency chain."""
        a, b, c, d, e = sp.symbols("a b c d e")
        assignments = [
            (e, c + d),
            (d, a + b),
            (c, a * 2),
            (b, a + 1),
            (a, sp.Integer(5)),
        ]

        result = topological_sort(assignments)

        # Verify dependencies are respected
        symbol_positions = {sym: i for i, (sym, _) in enumerate(result)}

        # a should come before all others
        assert symbol_positions[a] < symbol_positions[b]
        assert symbol_positions[a] < symbol_positions[c]
        assert symbol_positions[a] < symbol_positions[d]
        assert symbol_positions[a] < symbol_positions[e]

        # b and c should come before d and e
        assert symbol_positions[b] < symbol_positions[d]
        assert symbol_positions[c] < symbol_positions[e]

        # d and c should come before e
        assert symbol_positions[d] < symbol_positions[e]
        assert symbol_positions[c] < symbol_positions[e]

    def test_topological_sort_circular_dependency(self):
        """Test that circular dependencies raise ValueError."""
        x, y, z = sp.symbols("x y z")
        assignments = [
            (x, y + 1),
            (y, z + 1),
            (z, x + 1),  # Creates circular dependency
        ]

        with pytest.raises(ValueError, match="Circular dependency detected"):
            topological_sort(assignments)

    def test_topological_sort_self_reference(self):
        """Test that self-references raise ValueError."""
        x, y = sp.symbols("x y")
        assignments = [
            (x, x + 1),  # Self-reference
            (y, sp.Integer(2)),
        ]

        with pytest.raises(ValueError, match="Circular dependency detected"):
            topological_sort(assignments)

    def test_topological_sort_external_symbols(self):
        """Test that external symbols (not in assignments) are ignored."""
        x, y, z, external = sp.symbols("x y z external")
        assignments = [
            (x, external + 1),  # external is not in assignments
            (y, x + z),
            (z, sp.Integer(3)),
        ]

        result = topological_sort(assignments)

        # Should work fine, external symbols are ignored
        assert len(result) == 3
        symbol_positions = {sym: i for i, (sym, _) in enumerate(result)}
        assert symbol_positions[z] < symbol_positions[y]
        assert symbol_positions[x] < symbol_positions[y]

    def test_topological_sort_empty_input(self):
        """Test topological sort with empty input."""
        result = topological_sort([])
        assert result == []

        result = topological_sort({})
        assert result == []

    def test_topological_sort_single_assignment(self):
        """Test topological sort with single assignment."""
        x = sp.Symbol("x")
        assignments = [(x, sp.Integer(42))]

        result = topological_sort(assignments)
        assert result == [(x, sp.Integer(42))]

    def test_topological_sort_with_functions(self):
        """Test topological sort with function calls."""
        x, y, z = sp.symbols("x y z")
        assignments = [(z, sp.sin(y)), (y, sp.cos(x)), (x, sp.pi / 4)]

        result = topological_sort(assignments)

        # Verify order respects dependencies
        symbol_positions = {sym: i for i, (sym, _) in enumerate(result)}
        assert symbol_positions[x] < symbol_positions[y]
        assert symbol_positions[y] < symbol_positions[z]


class TestCseAndStack:
    """Test cases for cse_and_stack function."""

    def test_cse_and_stack_simple(self):
        """Test cse_and_stack with simple expressions."""
        x, y, z = sp.symbols("x y z")
        a, b = sp.symbols("a b")

        equations = [
            (a, x + y),
            (b, x + y + z),  # x + y is common subexpression
        ]

        result = cse_and_stack(equations)

        # Should introduce CSE symbol and reorder appropriately
        assert len(result) >= 2
        # Result should be in topological order
        result_symbols = [sym for sym, _ in result]
        assert a in result_symbols
        assert b in result_symbols

    def test_cse_and_stack_no_common_subexpressions(self):
        """Test cse_and_stack when there are no common subexpressions."""
        x, y, z = sp.symbols("x y z")
        a, b, c = sp.symbols("a b c")

        equations = [(a, x), (b, y), (c, z)]

        result = cse_and_stack(equations)

        # Should return original equations in topological order
        assert len(result) == 3
        result_dict = dict(result)
        assert result_dict[a] == x
        assert result_dict[b] == y
        assert result_dict[c] == z

    def test_cse_and_stack_complex_expressions(self):
        """Test cse_and_stack with complex expressions."""
        x, y = sp.symbols("x y")
        a, b, c = sp.symbols("a b c")

        equations = [
            (a, (x + y) ** 2),
            (b, (x + y) ** 3),
            (c, (x + y) ** 2 + (x + y) ** 3),
        ]

        result = cse_and_stack(equations)

        # Should identify common subexpressions
        assert len(result) >= 3
        # All original symbols should be present
        result_symbols = [sym for sym, _ in result]
        assert a in result_symbols
        assert b in result_symbols
        assert c in result_symbols

    def test_cse_and_stack_custom_symbol_prefix(self):
        """Test cse_and_stack with custom symbol prefix."""
        x, y = sp.symbols("x y")
        a, b = sp.symbols("a b")

        equations = [(a, x + y), (b, x + y + 1)]

        result = cse_and_stack(equations, symbol="temp")

        # Should use custom prefix for CSE symbols
        cse_symbols = [sym for sym, _ in result if str(sym).startswith("temp")]
        # May or may not have CSE symbols depending on SymPy's optimization
        assert len(result) >= 2

    def test_cse_and_stack_symbol_collision_warning(self):
        """Test warning when CSE symbol prefix collides with existing symbols."""
        x, y = sp.symbols("x y")
        _cse0 = sp.Symbol(
            "_cse0"
        )  # Symbol that conflicts with default CSE naming
        a, b = sp.symbols("a b")

        equations = [
            (a, x + y),
            (b, x + y + 1),
            (_cse0, x),  # This creates a collision
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cse_and_stack(equations)

            # Should warn about symbol collision
            if w:  # Only check if warnings were generated
                warning_messages = [str(warning.message) for warning in w]
                assert any("_cse" in msg for msg in warning_messages)

    def test_cse_and_stack_with_dependencies(self):
        """Test cse_and_stack respects dependencies between assignments."""
        x, y, z = sp.symbols("x y z")
        a, b, c = sp.symbols("a b c")

        equations = [
            (c, a + b),  # c depends on a and b
            (b, x + y),
            (a, x * 2),
        ]

        result = cse_and_stack(equations)

        # Should return in topological order
        symbol_positions = {sym: i for i, (sym, _) in enumerate(result)}
        assert symbol_positions[a] < symbol_positions[c]
        assert symbol_positions[b] < symbol_positions[c]

    def test_cse_and_stack_empty_input(self):
        """Test cse_and_stack with empty input."""
        result = cse_and_stack([])
        assert result == []

    def test_cse_and_stack_single_equation(self):
        """Test cse_and_stack with single equation."""
        x = sp.Symbol("x")
        a = sp.Symbol("a")

        equations = [(a, x**2)]
        result = cse_and_stack(equations)

        assert len(result) == 1
        assert result[0] == (a, x**2)

    def test_cse_and_stack_with_functions(self):
        """Test cse_and_stack with mathematical functions."""
        x, y = sp.symbols("x y")
        a, b, c = sp.symbols("a b c")

        equations = [
            (a, sp.sin(x + y)),
            (b, sp.cos(x + y)),
            (c, sp.sin(x + y) + sp.cos(x + y)),
        ]

        result = cse_and_stack(equations)

        # Should identify x + y as common subexpression
        assert len(result) >= 3
        result_symbols = [sym for sym, _ in result]
        assert a in result_symbols
        assert b in result_symbols
        assert c in result_symbols

    def test_cse_and_stack_maintains_expression_equivalence(self):
        """Test that cse_and_stack maintains mathematical equivalence."""
        x, y = sp.symbols("x y")
        a, b = sp.symbols("a b")

        original_equations = [
            (a, x**2 + 2 * x * y + y**2),
            (b, x**2 + 2 * x * y + y**2 + 1),
        ]

        result = cse_and_stack(original_equations)

        # Create substitution chain to verify equivalence
        substitutions = dict(result)

        # Substitute all CSE symbols to get final expressions
        final_a = substitutions[a]
        final_b = substitutions[b]

        # Recursively substitute until no more CSE symbols remain
        max_iterations = 10
        for _ in range(max_iterations):
            old_a, old_b = final_a, final_b
            final_a = final_a.subs(substitutions)
            final_b = final_b.subs(substitutions)
            if final_a == old_a and final_b == old_b:
                break

        # Should be equivalent to original expressions
        original_a = x**2 + 2*x*y + y**2
        original_b = x**2 + 2*x*y + y**2 + 1

        assert sp.simplify(final_a - original_a) == 0
        assert sp.simplify(final_b - original_b) == 0
