import numpy as np
import pytest
import sympy as sp
from numba import cuda
from numpy.testing import assert_allclose
from sympy import Matrix

from cubie.systemmodels.symbolic.jacobian import (
    JVP_TEMPLATE,
    VJP_TEMPLATE,
    generate_analytical_jvp,
    generate_analytical_vjp,
    generate_jacobian,
    generate_jvp_code,
    generate_vjp_code,
    generate_i_minus_hj_code,
    generate_residual_plus_i_minus_hj_code,
    clear_cache,
    get_cache_counts,
)
from cubie.systemmodels.symbolic.parser import IndexedBases
from cubie.systemmodels.symbolic.symbolicODE import SymbolicODE


def substitute_jacobian_elements(results_dict, auxiliaries):
    """
    Helper function to substitute jacobian elements into results expressions.

    Args:
        results_dict: Dictionary mapping lhs symbols to rhs expressions (e.g., {jvp[0]: expr1, jvp[1]: expr2})
        auxiliaries: Dictionary of jacobian elements to substitute

    Returns:
        Dictionary mapping lhs symbols to expressions with jacobian elements substituted
    """
    subbed_dict = {}
    for lhs, expr in results_dict.items():
        expr = expr.subs(auxiliaries)
        while expr.free_symbols.intersection(auxiliaries):
            expr = expr.subs(auxiliaries)
        subbed_dict[lhs] = expr
    return subbed_dict


class TestJacobianTemplates:
    """Test the JVP and VJP template constants."""

    def test_jvp_template_format(self):
        """Test that JVP_TEMPLATE can be formatted correctly."""
        func_name = "test_jvp_factory"
        body = "    # test jvp body"

        formatted = JVP_TEMPLATE.format(func_name=func_name, body=body)

        assert func_name in formatted
        assert "test jvp body" in formatted
        assert "@cuda.jit" in formatted
        assert "def jvp(" in formatted
        assert "return jvp" in formatted

    def test_vjp_template_format(self):
        """Test that VJP_TEMPLATE can be formatted correctly."""
        func_name = "test_vjp_factory"
        body = "    # test vjp body"

        formatted = VJP_TEMPLATE.format(func_name=func_name, body=body)

        assert func_name in formatted
        assert "test vjp body" in formatted
        assert "@cuda.jit" in formatted
        assert "def vjp(" in formatted
        assert "return vjp" in formatted

    def test_template_structure(self):
        """Test the structure of both templates."""
        for template in [JVP_TEMPLATE, VJP_TEMPLATE]:
            lines = template.splitlines()
            assert any("def {func_name}" in line for line in lines)
            assert any("@cuda.jit" in line for line in lines)
            assert any("precision[:]" in line for line in lines)
            assert any("device=True" in line for line in lines)
            assert any("inline=True" in line for line in lines)


class TestGenerateJacobian:
    """Test the generate_jacobian function."""

    def test_linear_system_jacobian(self):
        """Test Jacobian for a simple linear system."""
        x, y = sp.symbols("x y")
        a, b, c, d = sp.symbols("a b c d")

        # Linear system: dx/dt = ax + by, dy/dt = cx + dy
        equations = [
            (sp.Symbol("dx"), a * x + b * y),
            (sp.Symbol("dy"), c * x + d * y),
        ]

        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0, sp.Symbol("dy"): 1}

        jacobian = generate_jacobian(equations, input_order, output_order)

        # Expected Jacobian matrix: [[a, b], [c, d]]
        expected = Matrix([[a, b], [c, d]])
        assert jacobian.equals(expected)

    def test_nonlinear_system_jacobian(self):
        """Test Jacobian for a nonlinear system."""
        x, y = sp.symbols("x y")
        a, b = sp.symbols("a b")

        # Lotka-Volterra: dx/dt = ax - bxy, dy/dt = bxy - ay
        equations = [
            (sp.Symbol("dx"), a * x - b * x * y),
            (sp.Symbol("dy"), b * x * y - a * y),
        ]

        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0, sp.Symbol("dy"): 1}

        jacobian = generate_jacobian(equations, input_order, output_order)

        # Expected Jacobian: [[(a - by), -bx], [by, (bx - a)]]
        expected = Matrix([[a - b * y, -b * x], [b * y, b * x - a]])
        assert jacobian.equals(expected)

    def test_single_variable_system(self):
        """Test Jacobian for single variable system."""
        x = sp.symbols("x")
        a = sp.symbols("a")

        equations = [(sp.Symbol("dx"), a * x**2)]
        input_order = {x: 0}
        output_order = {sp.Symbol("dx"): 0}

        jacobian = generate_jacobian(equations, input_order, output_order)

        # Expected: [[2*a*x]]
        expected = Matrix([[2 * a * x]])
        assert jacobian.equals(expected)

    def test_system_with_auxiliaries(self):
        """Test Jacobian for system with auxiliary variables."""
        x, y = sp.symbols("x y")
        a, b = sp.symbols("a b")

        # System with auxiliary variable
        equations = [
            (sp.Symbol("aux"), a * x + b * y),  # auxiliary
            (sp.Symbol("dx"), sp.Symbol("aux") - x),
            (sp.Symbol("dy"), -sp.Symbol("aux") + y),
        ]

        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0, sp.Symbol("dy"): 1}

        jacobian = generate_jacobian(equations, input_order, output_order)

        # Chain rule should give: [[(a-1), b], [-a, (-b+1)]]
        expected = Matrix([[a - 1, b], [-a, -b + 1]])
        assert jacobian.equals(expected)

    def test_empty_system(self):
        """Test Jacobian for empty system."""
        equations = []
        input_order = {}
        output_order = {}

        jacobian = generate_jacobian(equations, input_order, output_order)

        # Should be empty matrix
        assert jacobian.shape == (0, 0)

    def test_topological_order_violation(self):
        """Test error handling for topological order violations."""
        x, y = sp.symbols("x y")

        # Create equations with circular dependency
        equations = [
            (sp.Symbol("aux1"), sp.Symbol("aux2") + x),
            (sp.Symbol("aux2"), sp.Symbol("aux1") + y),
            (sp.Symbol("dx"), sp.Symbol("aux1")),
        ]

        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0}

        with pytest.raises(ValueError, match="Circular dependency"):
            generate_jacobian(equations, input_order, output_order)


class TestGenerateAnalyticalJvp:
    """Test the generate_analytical_jvp function."""

    def test_linear_system_jvp(self):
        """Test JVP generation for linear system."""
        x, y = sp.symbols("x y")
        a, b, c, d = sp.symbols("a b c d")

        equations = [
            (sp.Symbol("dx"), a * x + b * y),
            (sp.Symbol("dy"), c * x + d * y),
        ]

        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0, sp.Symbol("dy"): 1}

        jvp_expressions = generate_analytical_jvp(
            equations, input_order, output_order
        )

        assert isinstance(jvp_expressions, list)
        assert len(jvp_expressions) > 0

        # Verify the actual JVP computation
        # For linear system with Jacobian [[a, b], [c, d]] and vector v = [v[0], v[1]]
        # Expected JVP = [[a, b], [c, d]] * [v[0], v[1]] = [a*v[0] + b*v[1], c*v[0] + d*v[1]]

        # Extract Jacobian elements and JVP results
        # Extract Jacobian elements and JVP results, preserving order
        jacobian_elements = {
            lhs: rhs
            for lhs, rhs in jvp_expressions
            if str(lhs).startswith("j_")
        }
        jvp_results_dict = {
            lhs: rhs
            for lhs, rhs in jvp_expressions
            if str(lhs).startswith("jvp[")
        }

        # Substitute Jacobian elements into JVP expressions
        subbed_dict = substitute_jacobian_elements(
            jvp_results_dict, jacobian_elements
        )

        # Assemble results in index order (jvp[0], jvp[1], etc.)
        subbed_exprs = []
        for i in range(len(subbed_dict)):
            jvp_symbol = sp.Symbol(f"jvp[{i}]")
            if jvp_symbol in subbed_dict:
                subbed_exprs.append(subbed_dict[jvp_symbol])
            else:
                # Fallback: try to find by string representation
                for lhs, expr in subbed_dict.items():
                    if str(lhs) == f"jvp[{i}]":
                        subbed_exprs.append(expr)
                        break

        # Extract the vector variable from the expressions
        v = sp.IndexedBase("v", shape=(2,))
        expected_jvp = [a * v[0] + b * v[1], c * v[0] + d * v[1]]

        # Verify that the JVP expressions match the expected mathematical result
        assert len(subbed_exprs) == 2, (
            f"Expected 2 JVP results, got {len(subbed_exprs)}"
        )

        # Simplify both sides and compare
        for i, (actual, expected) in enumerate(
            zip(subbed_exprs, expected_jvp)
        ):
            simplified_actual = sp.simplify(actual)
            simplified_expected = sp.simplify(expected)
            assert simplified_actual.equals(simplified_expected), (
                f"JVP[{i}] mismatch: got {simplified_actual}, expected {simplified_expected}"
            )

    def test_nonlinear_system_jvp(self, nonlinear_equations):
        """Test JVP for nonlinear system."""
        x, y, a, b, dx, dy = sp.symbols("x y a b dx dy", real=True)
        input_order = {x: 0, y: 1}
        output_order = {dx: 0, dy: 1}

        jvp_expressions = generate_analytical_jvp(
            nonlinear_equations, input_order, output_order
        )

        assert isinstance(jvp_expressions, list)
        assert len(jvp_expressions) > 0

        # Verify the actual JVP computation for Lotka-Volterra system
        # dx/dt = a*x - b*x*y, dy/dt = b*x*y - a*y
        # Jacobian = [[a - b*y, -b*x], [b*y, b*x - a]]
        # Expected JVP = [[a - b*y, -b*x], [b*y, b*x - a]] * [v[0], v[1]]
        # = [(a - b*y)*v[0] + (-b*x)*v[1], b*y*v[0] + (b*x - a)*v[1]]

        # Extract Jacobian elements and JVP results
        jacobian_elements = {
            lhs: rhs
            for lhs, rhs in jvp_expressions
            if not str(lhs).startswith("jvp")
        }
        jvp_results_dict = {
            lhs: rhs
            for lhs, rhs in jvp_expressions
            if str(lhs).startswith("jvp[")
        }

        # Substitute Jacobian elements into JVP expressions
        subbed_dict = substitute_jacobian_elements(
            jvp_results_dict, jacobian_elements
        )

        # Assemble results in index order (jvp[0], jvp[1], etc.)
        subbed_exprs = []
        for i in range(len(subbed_dict)):
            jvp_symbol = sp.Symbol(f"jvp[{i}]")
            if jvp_symbol in subbed_dict:
                subbed_exprs.append(subbed_dict[jvp_symbol])
            else:
                # Fallback: try to find by string representation
                for lhs, expr in subbed_dict.items():
                    if str(lhs) == f"jvp[{i}]":
                        subbed_exprs.append(expr)
                        break

        # Extract the vector variable from the expressions
        v = sp.IndexedBase("v", shape=(2,))
        expected_jvp = [
            (a - b * y) * v[0] + (-b * x) * v[1],
            b * y * v[0] + (b * x - a) * v[1],
        ]

        # Verify that the JVP expressions match the expected mathematical result
        assert len(subbed_exprs) == 2, (
            f"Expected 2 JVP results, got {len(subbed_exprs)}"
        )

        # Simplify both sides and compare
        for i, (actual, expected) in enumerate(
            zip(subbed_exprs, expected_jvp)
        ):
            simplified_actual = sp.simplify(actual)
            simplified_expected = sp.simplify(expected)
            assert simplified_actual.equals(simplified_expected)

    def test_jvp_without_cse(self, simple_equations):
        """Test JVP generation without common subexpression elimination."""
        x, y = sp.symbols("x y")
        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0, sp.Symbol("dy"): 1}

        jvp_expressions = generate_analytical_jvp(
            simple_equations, input_order, output_order, cse=False
        )

        assert isinstance(jvp_expressions, list)
        assert len(jvp_expressions) > 0

    def test_single_output_jvp(self):
        """Test JVP for single output system."""
        x, y = sp.symbols("x y")
        a, b = sp.symbols("a b")

        equations = [(sp.Symbol("dx"), a * x + b * y)]
        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0}

        jvp_expressions = generate_analytical_jvp(
            equations, input_order, output_order
        )

        assert isinstance(jvp_expressions, list)
        assert len(jvp_expressions) > 0

        # For single output, Jacobian = [[a, b]], vector v = [v[0], v[1]]
        # Expected JVP = [[a, b]] * [v[0], v[1]] = [a*v[0] + b*v[1]]

        # Extract Jacobian elements and JVP results
        jacobian_elements = {
            lhs: rhs
            for lhs, rhs in jvp_expressions
            if not str(lhs).startswith("jvp[")
        }
        jvp_results_dict = {
            lhs: rhs
            for lhs, rhs in jvp_expressions
            if str(lhs).startswith("jvp[")
        }

        # Substitute Jacobian elements into JVP expressions
        subbed_dict = substitute_jacobian_elements(
            jvp_results_dict, jacobian_elements
        )

        # Assemble results in index order (jvp[0], jvp[1], etc.)
        subbed_exprs = []
        for i in range(len(subbed_dict)):
            jvp_symbol = sp.Symbol(f"jvp[{i}]")
            if jvp_symbol in subbed_dict:
                subbed_exprs.append(subbed_dict[jvp_symbol])
            else:
                # Fallback: try to find by string representation
                for lhs, expr in subbed_dict.items():
                    if str(lhs) == f"jvp[{i}]":
                        subbed_exprs.append(expr)
                        break

        v = sp.IndexedBase("v", shape=(2,))
        expected_jvp = [a * v[0] + b * v[1]]

        assert len(subbed_exprs) == 1, (
            f"Expected 1 JVP result, got {len(subbed_exprs)}"
        )

        simplified_actual = sp.simplify(subbed_exprs[0])
        simplified_expected = sp.simplify(expected_jvp[0])
        assert simplified_actual.equals(simplified_expected), (
            f"Single output JVP mismatch: got {simplified_actual}, expected {simplified_expected}"
        )

    def test_jvp_vjp_consistency(self):
        """Test that JVP and VJP are consistent (transposes of each other)."""
        x, y = sp.symbols("x y")
        a, b, c, d = sp.symbols("a b c d")

        equations = [
            (sp.Symbol("dx"), a * x + b * y),
            (sp.Symbol("dy"), c * x + d * y),
        ]

        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0, sp.Symbol("dy"): 1}

        # Generate both JVP and VJP expressions
        jvp_expressions = generate_analytical_jvp(
            equations, input_order, output_order
        )
        vjp_expressions = generate_analytical_vjp(
            equations, input_order, output_order
        )

        # Extract Jacobian elements and results, preserving order
        jvp_jacobian_elements = {
            lhs: rhs
            for lhs, rhs in jvp_expressions
            if str(lhs).startswith("j_")
        }
        vjp_jacobian_elements = {
            lhs: rhs
            for lhs, rhs in vjp_expressions
            if str(lhs).startswith("j_")
        }

        jvp_results_dict = {
            lhs: rhs
            for lhs, rhs in jvp_expressions
            if str(lhs).startswith("jvp[")
        }
        vjp_results_dict = {
            lhs: rhs
            for lhs, rhs in vjp_expressions
            if str(lhs).startswith("vjp[")
        }

        # Substitute Jacobian elements into results
        jvp_subbed_dict = substitute_jacobian_elements(
            jvp_results_dict, jvp_jacobian_elements
        )
        vjp_subbed_dict = substitute_jacobian_elements(
            vjp_results_dict, vjp_jacobian_elements
        )

        # Assemble results in index order
        jvp_subbed = []
        for i in range(len(jvp_subbed_dict)):
            jvp_symbol = sp.Symbol(f"jvp[{i}]")
            jvp_subbed.append(jvp_subbed_dict[jvp_symbol])

        vjp_subbed = []
        for i in range(len(vjp_subbed_dict)):
            jvp_symbol = sp.Symbol(f"vjp[{i}]")
            vjp_subbed.append(vjp_subbed_dict[jvp_symbol])

        # For consistency check, we verify that:
        # JVP: J * v = [a*v[0] + b*v[1], c*v[0] + d*v[1]]
        # VJP: v^T * J = [v[0]*a + v[1]*c, v[0]*b + v[1]*d]
        # The Jacobian J = [[a, b], [c, d]]

        v_jvp = sp.IndexedBase("v", shape=(2,))
        v_vjp = sp.IndexedBase("v", shape=(2,))

        expected_jvp = [
            a * v_jvp[0] + b * v_jvp[1],
            c * v_jvp[0] + d * v_jvp[1],
        ]
        expected_vjp = [
            v_vjp[0] * a + v_vjp[1] * c,
            v_vjp[0] * b + v_vjp[1] * d,
        ]

        # Verify JVP results
        assert len(jvp_subbed) == 2
        for i, (actual, expected) in enumerate(zip(jvp_subbed, expected_jvp)):
            assert sp.simplify(actual).equals(sp.simplify(expected)), (
                f"JVP consistency check failed at index {i}"
            )

        # Verify VJP results
        assert len(vjp_subbed) == 2
        for i, (actual, expected) in enumerate(zip(vjp_subbed, expected_vjp)):
            assert sp.simplify(actual).equals(sp.simplify(expected)), (
                f"VJP consistency check failed at index {i}"
            )


class TestGenerateAnalyticalVjp:
    """Test the generate_analytical_vjp function."""

    def test_linear_system_vjp(self):
        """Test VJP generation for linear system."""
        x, y = sp.symbols("x y")
        a, b, c, d = sp.symbols("a b c d")

        equations = [
            (sp.Symbol("dx"), a * x + b * y),
            (sp.Symbol("dy"), c * x + d * y),
        ]

        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0, sp.Symbol("dy"): 1}

        vjp_expressions = generate_analytical_vjp(
            equations, input_order, output_order
        )

        assert isinstance(vjp_expressions, list)
        assert len(vjp_expressions) > 0

        # Verify the actual VJP computation
        # For linear system with Jacobian [[a, b], [c, d]] and vector v = [v[0], v[1]]
        # Expected VJP = [v[0], v[1]] * [[a, b], [c, d]] = [v[0]*a + v[1]*c, v[0]*b + v[1]*d]

        # Extract Jacobian elements and VJP results
        jacobian_elements = {
            lhs: rhs
            for lhs, rhs in vjp_expressions
            if not str(lhs).startswith("vjp[")
        }
        vjp_results = {
            lhs: rhs
            for lhs, rhs in vjp_expressions
            if str(lhs).startswith("vjp[")
        }

        subbed_dict = substitute_jacobian_elements(
            vjp_results, jacobian_elements
        )
        subbed_exprs = []
        for i in range(len(vjp_results)):
            vjp_symbol = sp.Symbol(f"vjp[{i}]")
            subbed_exprs.append(subbed_dict[vjp_symbol])

        # Extract the vector variable from the expressions
        v = sp.IndexedBase("v", shape=(2,))
        expected_vjp = [v[0] * a + v[1] * c, v[0] * b + v[1] * d]

        # Verify that the VJP expressions match the expected mathematical result
        assert len(subbed_exprs) == 2, (
            f"Expected 2 VJP results, got {len(subbed_exprs)}"
        )

        # Simplify both sides and compare
        for i, (actual, expected) in enumerate(
            zip(subbed_exprs, expected_vjp)
        ):
            simplified_actual = sp.simplify(actual)
            simplified_expected = sp.simplify(expected)
            assert simplified_actual.equals(simplified_expected), (
                f"VJP[{i}] mismatch: got {simplified_actual}, expected "
                f"{simplified_expected}"
            )

    def test_vjp_without_cse(self, simple_equations):
        """Test VJP generation without CSE."""
        x, y = sp.symbols("x y")
        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0, sp.Symbol("dy"): 1}

        vjp_expressions = generate_analytical_vjp(
            simple_equations, input_order, output_order, cse=False
        )

        assert isinstance(vjp_expressions, list)
        assert len(vjp_expressions) > 0

    def test_nonlinear_system_vjp(self, nonlinear_equations):
        """Test VJP for nonlinear system."""
        x, y, a, b, dx, dy = sp.symbols("x y a b dx dy", real=True)

        input_order = {x: 0, y: 1}
        output_order = {dx: 0, dy: 1}

        vjp_expressions = generate_analytical_vjp(
            nonlinear_equations, input_order, output_order
        )

        assert isinstance(vjp_expressions, list)
        assert len(vjp_expressions) > 0

        # Verify the actual VJP computation for Lotka-Volterra system
        # dx/dt = a*x - b*x*y, dy/dt = b*x*y - a*y
        # Jacobian = [[a - b*y, -b*x], [b*y, b*x - a]]
        # Expected VJP = [v[0], v[1]] * [[a - b*y, -b*x], [b*y, b*x - a]]
        # = [v[0]*(a - b*y) + v[1]*b*y, v[0]*(-b*x) + v[1]*(b*x - a)]

        # Extract Jacobian elements and VJP results
        jacobian_elements = {
            lhs: rhs
            for lhs, rhs in vjp_expressions
            if not str(lhs).startswith("vjp[")
        }
        vjp_results = {
            lhs: rhs
            for lhs, rhs in vjp_expressions
            if str(lhs).startswith("vjp[")
        }

        subbed_dict = substitute_jacobian_elements(
            vjp_results, jacobian_elements
        )
        subbed_exprs = []
        for i in range(len(vjp_results)):
            vjp_symbol = sp.Symbol(f"vjp[{i}]")
            subbed_exprs.append(subbed_dict[vjp_symbol])

        # Extract the vector variable from the expressions
        v = sp.IndexedBase("v", shape=(2,))
        expected_vjp = [
            v[0] * (a - b * y) + v[1] * b * y,
            v[0] * (-b * x) + v[1] * (b * x - a),
        ]

        # Verify that the VJP expressions match the expected mathematical result
        assert len(subbed_exprs) == 2, (
            f"Expected 2 VJP results, got {len(subbed_exprs)}"
        )

        # Simplify both sides and compare
        for i, (actual, expected) in enumerate(
            zip(subbed_exprs, expected_vjp)
        ):
            simplified_actual = sp.simplify(actual)
            simplified_expected = sp.simplify(expected)
            assert simplified_actual.equals(simplified_expected), (
                f"Nonlinear VJP[{i}] mismatch: got {simplified_actual}, expected {simplified_expected}"
            )

    def test_single_output_vjp(self):
        """Test VJP for single output system."""
        x, y = sp.symbols("x y")
        a, b = sp.symbols("a b")

        equations = [(sp.Symbol("dx"), a * x + b * y)]
        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0}

        vjp_expressions = generate_analytical_vjp(
            equations, input_order, output_order
        )

        assert isinstance(vjp_expressions, list)
        assert len(vjp_expressions) > 0

        # For single output, Jacobian = [[a, b]], vector v = [v[0]]
        # Expected VJP = [v[0]] * [[a, b]] = [v[0]*a, v[0]*b]

        # Extract Jacobian elements and VJP results
        jacobian_elements = {
            lhs: rhs
            for lhs, rhs in vjp_expressions
            if str(lhs).startswith("j_")
        }
        vjp_results = {
            lhs: rhs
            for lhs, rhs in vjp_expressions
            if str(lhs).startswith("vjp[")
        }

        subbed_dict = substitute_jacobian_elements(
            vjp_results, jacobian_elements
        )
        subbed_exprs = []
        for i in range(len(vjp_results)):
            vjp_symbol = sp.Symbol(f"vjp[{i}]")
            subbed_exprs.append(subbed_dict[vjp_symbol])

        v = sp.IndexedBase("v", shape=(1,))
        expected_vjp = [v[0] * a, v[0] * b]

        assert len(subbed_exprs) == 2, (
            f"Expected 2 VJP results, got {len(subbed_exprs)}"
        )

        for i, (actual, expected) in enumerate(
            zip(subbed_exprs, expected_vjp)
        ):
            simplified_actual = sp.simplify(actual)
            simplified_expected = sp.simplify(expected)
            assert simplified_actual.equals(simplified_expected), (
                f"Single output VJP[{i}] mismatch: got {simplified_actual}, expected {simplified_expected}"
            )


class TestGenerateJvpCode:
    """Test the generate_jvp_code function."""

    def test_jvp_code_generation(self, linear_system_equations, indexed_bases):
        """Test JVP code generation."""
        code = generate_jvp_code(linear_system_equations, indexed_bases)

        assert isinstance(code, str)
        assert "def jvp_factory" in code
        assert "@cuda.jit" in code
        assert "def jvp(" in code
        assert "return jvp" in code

    def test_jvp_custom_function_name(self, simple_equations, indexed_bases):
        """Test JVP code generation with custom function name."""
        func_name = "custom_jvp_factory"
        code = generate_jvp_code(simple_equations, indexed_bases, func_name)

        assert f"def {func_name}" in code
        assert "def jvp_factory" not in code

    def test_jvp_code_structure(self, simple_equations, indexed_bases):
        """Test structure of generated JVP code."""
        code = generate_jvp_code(simple_equations, indexed_bases)

        assert "precision[:]" in code
        assert "device=True" in code
        assert "inline=True" in code

        # Check parameter order
        lines = code.splitlines()
        func_def_line = None
        for line in lines:
            if "def jvp(" in line:
                func_def_line = line
                break

        assert func_def_line is not None
        assert "state" in func_def_line
        assert "parameters" in func_def_line
        assert "driver" in func_def_line


class TestGenerateVjpCode:
    """Test the generate_vjp_code function."""

    def test_vjp_code_generation(self, linear_system_equations, indexed_bases):
        """Test VJP code generation."""
        code = generate_vjp_code(linear_system_equations, indexed_bases)

        assert isinstance(code, str)
        assert "def vjp_factory" in code
        assert "@cuda.jit" in code
        assert "def vjp(" in code
        assert "return vjp" in code

    def test_vjp_custom_function_name(self, simple_equations, indexed_bases):
        """Test VJP code generation with custom function name."""
        func_name = "custom_vjp_factory"
        code = generate_vjp_code(simple_equations, indexed_bases, func_name)

        assert f"def {func_name}" in code
        assert "def vjp_factory" not in code

    def test_vjp_code_structure(self, nonlinear_equations, indexed_bases):
        """Test structure of generated VJP code."""
        code = generate_vjp_code(nonlinear_equations, indexed_bases)

        assert "precision[:]" in code
        assert "device=True" in code
        assert "inline=True" in code


class TestAdditionalFactories:
    """Test additional Jacobian-level factory code generation."""

    def test_i_minus_hj_code_generation(self, linear_system_equations, indexed_bases):
        code = generate_i_minus_hj_code(linear_system_equations, indexed_bases)
        assert "def i_minus_hj_factory" in code
        assert "def i_minus_hj(" in code
        assert "stages=1" in code

    def test_residual_plus_i_minus_hj_generation(self,
                                                 linear_system_equations,
                                                 indexed_bases):
        code = generate_residual_plus_i_minus_hj_code(
            linear_system_equations, indexed_bases
        )
        assert "def residual_plus_i_minus_hj_factory" in code
        assert "def residual_plus_i_minus_hj(" in code
        assert "stages=1" in code


class TestJacobianIntegration:
    """Integration tests for Jacobian functionality."""

    def test_complete_jvp_workflow(self):
        """Test complete JVP workflow from equations to code."""
        # Van der Pol oscillator
        states = ["x", "y"]
        parameters = ["mu"]
        constants = []
        observables = []
        drivers = []

        indexed_bases = IndexedBases.from_user_inputs(
            states=states,
            parameters=parameters,
            constants=constants,
            observables=observables,
            drivers=drivers,
        )

        x, y, mu = sp.symbols("x y mu")
        equations = [
            (sp.Symbol("dx"), y),
            (sp.Symbol("dy"), mu * (1 - x**2) * y - x),
        ]

        code = generate_jvp_code(equations, indexed_bases)

        assert isinstance(code, str)
        assert "def jvp_factory" in code
        assert len(code) > 300  # Should be substantial

    def test_complete_vjp_workflow(self):
        """Test complete VJP workflow."""
        # Simple chemical reaction
        states = ["A", "B"]
        parameters = ["k1", "k2"]
        constants = []
        observables = []
        drivers = []

        indexed_bases = IndexedBases.from_user_inputs(
            states=states,
            parameters=parameters,
            constants=constants,
            observables=observables,
            drivers=drivers,
        )

        A, B, k1, k2 = sp.symbols("A B k1 k2")
        equations = [
            (sp.Symbol("dA"), -k1 * A + k2 * B),
            (sp.Symbol("dB"), k1 * A - k2 * B),
        ]

        code = generate_vjp_code(equations, indexed_bases)

        assert isinstance(code, str)
        assert "def vjp_factory" in code

    def test_higher_order_system(self):
        """Test with higher-order system."""
        states = ["x1", "x2", "x3"]
        parameters = ["a", "b", "c"]
        constants = []
        observables = []
        drivers = []

        indexed_bases = IndexedBases.from_user_inputs(
            states=states,
            parameters=parameters,
            constants=constants,
            observables=observables,
            drivers=drivers,
        )

        x1, x2, x3, a, b, c = sp.symbols("x1 x2 x3 a b c")
        equations = [
            (sp.Symbol('dx1'), a*x1 + b*x2),
            (sp.Symbol('dx2'), b*x2 + c*x3),
            (sp.Symbol('dx3'), c*x1 - a*x3)
        ]

        # Test both JVP and VJP
        jvp_code = generate_jvp_code(equations, indexed_bases)
        vjp_code = generate_vjp_code(equations, indexed_bases)

        assert isinstance(jvp_code, str)
        assert isinstance(vjp_code, str)
        assert "def jvp_factory" in jvp_code
        assert "def vjp_factory" in vjp_code



class TestCachingBehavior:
    """Tests for unified caching of Jacobian, JVP, and VJP expressions."""

    def setup_method(self):
        clear_cache()

    def test_jacobian_cache_counts(self):
        counts = get_cache_counts()
        assert counts["jac"] == 0 and counts["jvp"] == 0 and counts["vjp"] == 0

        # Simple 2x2 linear system
        x, y = sp.symbols("x y")
        a, b, c, d = sp.symbols("a b c d")
        equations = [
            (sp.Symbol("dx"), a * x + b * y),
            (sp.Symbol("dy"), c * x + d * y),
        ]
        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0, sp.Symbol("dy"): 1}

        _ = generate_jacobian(equations, input_order, output_order)
        counts = get_cache_counts()
        assert counts["jac"] == 1

        # Second call should hit cache, not increase count
        _ = generate_jacobian(equations, input_order, output_order)
        counts2 = get_cache_counts()
        assert counts2 == counts

    def test_jvp_vjp_cache_counts_and_cse_key(self):
        clear_cache()
        counts = get_cache_counts()
        assert counts["jac"] == 0 and counts["jvp"] == 0 and counts["vjp"] == 0

        x, y = sp.symbols("x y")
        a, b, c, d = sp.symbols("a b c d")
        equations = [
            (sp.Symbol("dx"), a * x + b * y),
            (sp.Symbol("dy"), c * x + d * y),
        ]
        input_order = {x: 0, y: 1}
        output_order = {sp.Symbol("dx"): 0, sp.Symbol("dy"): 1}

        # JVP cache
        _ = generate_analytical_jvp(equations, input_order, output_order, cse=True)
        counts = get_cache_counts()
        assert counts["jvp"] == 1

        # Same call doesn't increase count
        _ = generate_analytical_jvp(equations, input_order, output_order, cse=True)
        counts2 = get_cache_counts()
        assert counts2["jvp"] == 1

        # Different CSE setting should create a separate cache entry
        _ = generate_analytical_jvp(equations, input_order, output_order, cse=False)
        counts3 = get_cache_counts()
        assert counts3["jvp"] == 2

        # VJP cache
        _ = generate_analytical_vjp(equations, input_order, output_order, cse=True)
        counts4 = get_cache_counts()
        assert counts4["vjp"] == 1

        # Clearing should reset all
        clear_cache()
        counts5 = get_cache_counts()
        assert counts5["jac"] == 0 and counts5["jvp"] == 0 and counts5["vjp"] == 0

    def test_factories_use_cached_jvp(self):
        """Generating i_minus_hj and residual_plus_i_minus_hj should reuse cached JVP."""
        clear_cache()
        # Build a small system and index map
        states = ["x", "y"]
        parameters = ["a", "b"]
        indexed_bases = IndexedBases.from_user_inputs(
            states=states, parameters=parameters, constants=[], observables=[], drivers=[]
        )

        x, y, a, b = sp.symbols("x y a b")
        equations = [
            (sp.Symbol("dx", real=True), a * x + b * y),
            (sp.Symbol("dy", real=True), b * x - a * y),
        ]

        # Precompute and cache JVP with same arguments the factories will use
        _ = generate_analytical_jvp(
            equations,
            input_order=indexed_bases.states.index_map,
            output_order=indexed_bases.dxdt.index_map,
            observables=indexed_bases.observable_symbols,
            cse=True,
        )
        counts_before = get_cache_counts()
        assert counts_before["jvp"] == 1
        
        # Now generate factories; cache count should not increase
        _ = generate_i_minus_hj_code(equations, indexed_bases)
        counts_after_i_minus_hj = get_cache_counts()
        assert counts_after_i_minus_hj["jvp"] == counts_before["jvp"]
        
        _ = generate_residual_plus_i_minus_hj_code(equations, indexed_bases)
        counts_after_residual = get_cache_counts()
        assert counts_after_residual["jvp"] == counts_before["jvp"]


def test_biochemical_numerical(precision):
    """Numerically validate JVP, VJP and solver helpers."""
    states = ["Su", "En", "ES", "Pr"]
    parameters = ["k1", "k2", "k3"]
    constants = ["Et"]
    indexed_bases = IndexedBases.from_user_inputs(
        states=states,
        parameters=parameters,
        constants=constants,
        observables=[],
        drivers=[],
    )

    Su, En, ES, Pr, k1, k2, k3, Et = sp.symbols(
        "Su En ES Pr k1 k2 k3 Et", real=True
    )
    equations = [
        (sp.Symbol("dSu", real=True), -k1 * Su * En + k2 * ES),
        (sp.Symbol("dEn", real=True), -k1 * Su * En + k2 * ES + k3 * ES),
        (sp.Symbol("dES", real=True), k1 * Su * En - k2 * ES - k3 * ES),
        (sp.Symbol("dPr", real=True), k3 * ES),
    ]
    system = SymbolicODE(
        equations,
        indexed_bases,
        precision=precision,
        name="biochemical_system",
    )
    system.build()
    dxdt_fn = system.dxdt_function
    jvp_fn = system.jvp_function
    vjp_fn = system.vjp_function
    i_minus_hj_fn = system.i_minus_hj_function
    res_plus_i_minus_hj_fn = system.residual_plus_i_minus_hj_function

    initial_values = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=precision)
    parameter_vals = np.asarray([0.5, 0.5, 0.5], dtype=precision)
    driver_vals = np.zeros(1, dtype=precision)
    v = np.asarray([0.2, 0.1, 0.1, 0.2], dtype=precision)
    h = precision(0.1)

    d_dxdt = cuda.device_array(4, dtype=precision)
    d_jvp = cuda.device_array(4, dtype=precision)
    d_vjp = cuda.device_array(4, dtype=precision)
    d_i_minus_hj = cuda.device_array(4, dtype=precision)
    d_residual = cuda.device_array(4, dtype=precision)
    d_state = cuda.to_device(initial_values)
    d_params = cuda.to_device(parameter_vals)
    d_drivers = cuda.to_device(driver_vals)
    d_observables = cuda.device_array(1, dtype=precision)
    d_v = cuda.to_device(v)

    @cuda.jit()
    def kernel(
        state,
        params,
        drivers,
        observables,
        dxdt,
        vec,
        jvp,
        vjp,
        step,
        i_minus_hj,
        residual,
    ):
        dxdt_fn(state, params, drivers, observables, dxdt)
        jvp_fn(state, params, drivers, vec, jvp)
        vjp_fn(state, params, drivers, vec, vjp)
        i_minus_hj_fn(state, params, drivers, step, vec, i_minus_hj)
        res_plus_i_minus_hj_fn(state, params, drivers, step, vec, residual)

    kernel[1, 1](
        d_state,
        d_params,
        d_drivers,
        d_observables,
        d_dxdt,
        d_v,
        d_jvp,
        d_vjp,
        h,
        d_i_minus_hj,
        d_residual,
    )

    dxdt = d_dxdt.copy_to_host()
    jvp = d_jvp.copy_to_host()
    vjp = d_vjp.copy_to_host()
    i_minus_hj = d_i_minus_hj.copy_to_host()
    residual = d_residual.copy_to_host()

    p = parameter_vals
    x = initial_values
    expected_dxdt = np.zeros(4, dtype=precision)
    expected_dxdt[0] = -p[0] * x[0] * x[1] + p[1] * x[2]
    expected_dxdt[1] = -p[0] * x[0] * x[1] + (p[1] + p[2]) * x[2]
    expected_dxdt[2] = p[0] * x[0] * x[1] - (p[1] + p[2]) * x[2]
    expected_dxdt[3] = p[2] * x[2]

    expected_jvp = np.zeros(4, dtype=precision)
    expected_jvp[0] = -p[0] * x[1] * v[0] - p[0] * x[0] * v[1] + p[1] * v[2]
    expected_jvp[1] = -p[0] * x[1] * v[0] - p[0] * x[0] * v[1] + (
        p[1] + p[2]
    ) * v[2]
    expected_jvp[2] = p[0] * x[1] * v[0] + p[0] * x[0] * v[1] - (
        p[1] + p[2]
    ) * v[2]
    expected_jvp[3] = p[2] * v[2]

    expected_vjp = np.zeros(4, dtype=precision)
    expected_vjp[0] = p[0] * x[1] * (v[2] - v[1] - v[0])
    expected_vjp[1] = p[0] * x[0] * (v[2] - v[1] - v[0])
    expected_vjp[2] = p[1] * v[0] + (p[1] + p[2]) * (v[1] - v[2]) + p[2] * v[3]
    expected_vjp[3] = precision(0.0)

    expected_i_minus_hj = v - h * expected_jvp
    expected_residual = expected_dxdt + expected_i_minus_hj

    assert_allclose(dxdt, expected_dxdt, atol=1e-6, rtol=1e-6, err_msg="dxdt")
    assert_allclose(jvp, expected_jvp, atol=1e-6, rtol=1e-6, err_msg="jvp")
    assert_allclose(vjp, expected_vjp, atol=1e-6, rtol=1e-6, err_msg="vjp")
    assert_allclose(
        i_minus_hj,
        expected_i_minus_hj,
        atol=1e-6,
        rtol=1e-6,
        err_msg="i_minus_hj",
    )
    assert_allclose(
        residual,
        expected_residual,
        atol=1e-6,
        rtol=1e-6,
        err_msg="residual_plus_i_minus_hj",
    )
