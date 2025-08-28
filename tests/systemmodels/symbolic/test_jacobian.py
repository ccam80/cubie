import pytest
import sympy as sp
from sympy import Matrix

from cubie.systemmodels.symbolic.jacobian import (
    JVP_TEMPLATE,
    VJP_TEMPLATE,
    generate_analytical_jvp,
    generate_analytical_vjp,
    generate_jacobian,
    generate_jvp_code,
    generate_vjp_code,
)
from cubie.systemmodels.symbolic.parser import IndexedBases


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

        x1, x2, x3, a, b, c = sp.symbols('x1 x2 x3 a b c')
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
