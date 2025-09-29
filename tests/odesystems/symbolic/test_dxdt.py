import sympy as sp

import numpy as np
import pytest
from numba import cuda

from cubie.odesystems.symbolic.dxdt import (
    DXDT_TEMPLATE,
    generate_dxdt_fac_code,
    generate_observables_fac_code,
)
from cubie.odesystems.symbolic.parser import IndexedBases, parse_input
from cubie.odesystems.symbolic.symbolicODE import SymbolicODE


class TestDxdtTemplate:
    """Test the DXDT_TEMPLATE constant."""

    def test_dxdt_template_format(self):
        """Test that DXDT_TEMPLATE can be formatted correctly."""
        func_name = "test_factory"
        body = "    # test body"

        formatted = DXDT_TEMPLATE.format(
            func_name=func_name, const_lines="", body=body
        )

        assert func_name in formatted
        assert "test body" in formatted
        assert "@cuda.jit" in formatted
        assert "def dxdt(" in formatted
        assert "out, t" in formatted
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
        assert "out[" in code
        assert "dxdt[" not in code
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
        assert "out" in func_def_line
        assert "t" in func_def_line

    def test_constants_unpacked(self, indexed_bases):
        """Constants should be defined as standalone variables."""
        x, c = sp.symbols("x c", real=True)
        equations = [(sp.Symbol("dx", real=True), c * x)]
        code = generate_dxdt_fac_code(equations, indexed_bases)
        assert "c = precision(constants['c'])" in code


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

    def test_dxdt_reuses_observables_buffer(self):
        """Generated dxdt code should not reassign the observables array."""

        dxdt_lines = [
            "a1 = x1**2",
            "a2 = x1 * x2",
            "a3 = p1 * x3 * x2",
            "o1 = a3**2 + x1",
            "o2 = x2 * a2",
            "dx1 = o2 * p2",
            "dx2 = o1 * a1 * p2",
            "dx3 = x3 * a1",
        ]

        index_map, _, _, equations, _ = parse_input(
            dxdt=dxdt_lines,
            states={"x1": 0.0, "x2": 0.0, "x3": 0.0},
            observables=["o1", "o2"],
            parameters={"p1": 0.0, "p2": 0.0},
            constants={},
            drivers=[],
            strict=False,
        )

        code = generate_dxdt_fac_code(equations, index_map)

        lines = code.splitlines()
        assert any("observables[" in line for line in lines)
        assert all(
            not line.lstrip().startswith("observables[") for line in lines
        )


class TestGenerateObservablesFacCode:
    """Tests for observables-only factory generation."""

    def test_observables_factory_structure(self, indexed_bases):
        """Generated code should expose the observables factory."""

        x, y = sp.symbols("x y", real=True)
        a = sp.symbols("a", real=True)
        equations = [
            (sp.Symbol("obs1", real=True), x + a),
            (sp.Symbol("dx", real=True), y + x),
        ]

        code = generate_observables_fac_code(equations, indexed_bases)

        assert "def observables" in code
        assert "def get_observables" in code
        assert "observables[" in code
        assert "out[" not in code
        assert "observables, t" in code

    def test_observables_factory_includes_cse_dependencies(
        self, indexed_bases
    ):
        """CSE expressions shared with dxdt should still be emitted."""

        x, y = sp.symbols("x y", real=True)
        repeated = (x + y) ** 2
        equations = [
            (sp.Symbol("obs1", real=True), repeated),
            (sp.Symbol("dx", real=True), repeated + x),
        ]

        code = generate_observables_fac_code(equations, indexed_bases)

        assert "_cse" in code
        assert "out[" not in code


class TestObservablesDeviceParity:
    """Ensure observables helpers mirror dx/dt side effects."""

    @pytest.mark.parametrize(
        "state_values,param_values,driver_value",
        [
            ([1.2, -0.4], [0.3, 0.7], 0.25),
            ([-0.6, 1.5], [0.9, -0.2], -0.4),
        ],
    )
    def test_dxdt_preserves_observables(
        self,
        observables_kernel_system,
        state_values,
        param_values,
        driver_value,
        precision,
    ):
        """dxdt should consume, not overwrite, the observables buffer."""

        system = observables_kernel_system
        system.build()
        dxdt_dev = system.dxdt_function
        observables_fn = system.observables_function

        state = np.array(state_values, dtype=precision)
        parameters = np.array(param_values, dtype=precision)
        drivers = np.array([driver_value], dtype=precision)

        state_kernel = state.copy()
        parameters_kernel = parameters.copy()
        drivers_kernel = drivers.copy()

        observables_buffer = np.full(
            system.num_observables, precision(-1.0), dtype=precision
        )
        out = np.zeros(system.num_states, dtype=precision)

        @cuda.jit
        def kernel(
            state_in,
            params_in,
            drivers_in,
            obs_buf,
            out_buf,
            time_scalar,
        ):
            observables_fn(state_in, params_in, drivers_in, obs_buf, time_scalar)
            dxdt_dev(state_in, params_in, drivers_in, obs_buf, out_buf, time_scalar)

        kernel[
            1,
            1,
        ](
            state_kernel,
            parameters_kernel,
            drivers_kernel,
            observables_buffer,
            out,
            precision(0.0),
        )

        state_idx = system.states.indices_dict
        param_idx = system.parameters.indices_dict
        driver_names = system.indices.driver_names
        drive_idx = driver_names.index("drive")
        const_value = precision(system.constants.values_dict["c0"])

        x_val = state_kernel[state_idx["x"]]
        y_val = state_kernel[state_idx["y"]]
        alpha = parameters_kernel[param_idx["alpha"]]
        beta = parameters_kernel[param_idx["beta"]]
        drive_val = drivers_kernel[drive_idx]

        expected_rate = alpha * x_val + const_value
        expected_total = expected_rate + beta * y_val + drive_val
        expected_observables = np.array(
            [expected_rate, expected_total], dtype=precision
        )
        expected_dx = expected_total - y_val + alpha * drive_val
        expected_dy = expected_rate * x_val + const_value

        np.testing.assert_allclose(
            observables_buffer, expected_observables, rtol=1e-6
        )
        np.testing.assert_allclose(
            out,
            np.array([expected_dx, expected_dy], dtype=precision),
            rtol=1e-6,
        )


def test_recompile_updates_constants(precision):
    """Recompiling with new constants updates the device function."""

    system = SymbolicODE.create(
        dxdt=["dx = c * x"],
        states={"x": precision(1.0)},
        parameters={},
        constants={"c": precision(2.0)},
        drivers=[],
        observables=[],
        precision=precision,
        strict=True,
        name="recompile_constants",
    )

    def run_dxdt(current_system: SymbolicODE) -> float:
        cache = current_system.build()
        dxdt_func = cache.dxdt

        @cuda.jit
        def kernel(state, parameters, drivers, observables, out, time_scalar):
            dxdt_func(state, parameters, drivers, observables, out, time_scalar)

        state = np.array([precision(1.0)], dtype=precision)
        parameters = np.zeros(current_system.num_parameters, dtype=precision)
        drivers = np.zeros(current_system.num_drivers, dtype=precision)
        observables = np.zeros(
            current_system.num_observables, dtype=precision
        )
        out = np.zeros(current_system.num_states, dtype=precision)
        kernel[1, 1](state, parameters, drivers, observables, out, precision(0.0))
        return float(out[0])

    res1 = run_dxdt(system)
    system.set_constants({"c": precision(3.0)})
    res2 = run_dxdt(system)

    assert res1 == pytest.approx(2.0)
    assert res2 == pytest.approx(3.0)
