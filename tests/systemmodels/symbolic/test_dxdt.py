import sympy as sp

from cubie.systemmodels.symbolic.dxdt import (
    DXDT_TEMPLATE,
    generate_dxdt_fac_code,
)
from cubie.systemmodels.symbolic.parser import IndexedBases


class TestDxdtTemplate:
    """Test the DXDT_TEMPLATE constant."""

    def test_dxdt_template_format(self):
        """Test that DXDT_TEMPLATE can be formatted correctly."""
        func_name = "test_factory"
        body = "    # test body"

        formatted = DXDT_TEMPLATE.format(func_name=func_name, body=body)

        assert func_name in formatted
        assert "test body" in formatted
        assert "@cuda.jit" in formatted
        assert "def dxdt(" in formatted
        assert "return dxdt" in formatted

    def test_dxdt_template_structure(self):
        """Test the structure of DXDT_TEMPLATE."""
        lines = DXDT_TEMPLATE.splitlines()

        # Should contain proper function definition
        assert any("def {func_name}" in line for line in lines)
        assert any("@cuda.jit" in line for line in lines)
        assert any("def dxdt(" in line for line in lines)
        assert any("return dxdt" in line for line in lines)


class TestGenerateDxdtFacCode:
    """Test the generate_dxdt_fac_code function."""

    def test_simple_equations(self, simple_equations, indexed_bases):
        """Test code generation with simple equations."""
        code = generate_dxdt_fac_code(simple_equations, indexed_bases)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code
        assert "@cuda.jit" in code
        assert "def dxdt(" in code
        assert "return dxdt" in code

    def test_custom_function_name(self, simple_equations, indexed_bases):
        """Test code generation with custom function name."""
        func_name = "custom_dxdt_factory"
        code = generate_dxdt_fac_code(
            simple_equations, indexed_bases, func_name
        )

        assert f"def {func_name}" in code
        assert "def dxdt_factory" not in code

    def test_complex_equations(self, complex_equations, indexed_bases):
        """Test code generation with complex equations containing auxiliaries."""
        code = generate_dxdt_fac_code(complex_equations, indexed_bases)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code
        assert len(code) > 100  # Should be substantial code

    def test_linear_system(self, linear_system_equations, indexed_bases):
        """Test code generation with linear system."""
        code = generate_dxdt_fac_code(linear_system_equations, indexed_bases)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_nonlinear_equations(self, nonlinear_equations, indexed_bases):
        """Test code generation with nonlinear equations."""
        code = generate_dxdt_fac_code(nonlinear_equations, indexed_bases)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_empty_equations(self, indexed_bases):
        """Test code generation with empty equations list."""
        empty_equations = []
        code = generate_dxdt_fac_code(empty_equations, indexed_bases)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_single_equation(self, indexed_bases):
        """Test code generation with single equation."""
        x = sp.symbols("x")
        a = sp.symbols("a")
        single_eq = [(sp.Symbol("dx"), a * x)]

        code = generate_dxdt_fac_code(single_eq, indexed_bases)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_equations_with_constants(self, indexed_bases):
        """Test equations that reference constants."""
        x, y = sp.symbols("x y")
        equations = [
            (sp.Symbol("dx"), sp.pi * x + sp.E * y),
            (sp.Symbol("dy"), sp.sqrt(2) * x - y),
        ]

        code = generate_dxdt_fac_code(equations, indexed_bases)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_equations_with_functions(self, indexed_bases):
        """Test equations with mathematical functions."""
        x, y = sp.symbols("x y")
        a = sp.symbols("a")
        equations = [
            (sp.Symbol("dx"), sp.sin(a * x) + sp.cos(y)),
            (sp.Symbol("dy"), sp.exp(-a * x) * y),
        ]

        code = generate_dxdt_fac_code(equations, indexed_bases)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_code_contains_cuda_signatures(
        self, simple_equations, indexed_bases
    ):
        """Test that generated code contains proper CUDA signatures."""
        code = generate_dxdt_fac_code(simple_equations, indexed_bases)

        # Should contain proper parameter types
        assert "precision[:]" in code
        assert "device=True" in code
        assert "inline=True" in code

    def test_code_parameter_order(self, simple_equations, indexed_bases):
        """Test that function parameters are in correct order."""
        code = generate_dxdt_fac_code(simple_equations, indexed_bases)

        # Find the function signature line
        lines = code.splitlines()
        func_def_line = None
        for line in lines:
            if "def dxdt(" in line:
                func_def_line = line
                break

        assert func_def_line is not None
        assert "state" in func_def_line
        assert "parameters" in func_def_line
        assert "driver" in func_def_line
        assert "observables" in func_def_line
        assert "dxdt" in func_def_line


class TestDxdtIntegration:
    """Integration tests for DXDT functionality."""

    def test_realistic_ode_system(self):
        """Test with a realistic ODE system (Lotka-Volterra)."""
        # Lotka-Volterra predator-prey model
        states = ["prey", "predator"]
        parameters = ["alpha", "beta", "gamma", "delta"]
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

        # Define equations symbolically
        prey, predator = sp.symbols("prey predator")
        alpha, beta, gamma, delta = sp.symbols("alpha beta gamma delta")

        equations = [
            (sp.Symbol("dprey"), alpha * prey - beta * prey * predator),
            (
                sp.Symbol("dpredator"),
                gamma * prey * predator - delta * predator,
            ),
        ]

        code = generate_dxdt_fac_code(equations, indexed_bases)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code
        assert len(code) > 200  # Should be substantial

    def test_with_auxiliary_variables(self):
        """Test system with auxiliary variables."""
        states = ["x", "y"]
        parameters = ["k1", "k2"]
        constants = ["c1"]
        observables = ["total"]
        drivers = []

        indexed_bases = IndexedBases.from_user_inputs(
            states=states,
            parameters=parameters,
            constants=constants,
            observables=observables,
            drivers=drivers,
        )

        x, y = sp.symbols("x y")
        k1, k2, c1 = sp.symbols("k1 k2 c1")

        equations = [
            (sp.Symbol("total"), x + y),  # auxiliary variable
            (sp.Symbol("dx"), k1 * sp.Symbol("total") - k2 * x + c1),
            (sp.Symbol("dy"), -k1 * sp.Symbol("total") + k2 * x),
        ]

        code = generate_dxdt_fac_code(equations, indexed_bases)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code
