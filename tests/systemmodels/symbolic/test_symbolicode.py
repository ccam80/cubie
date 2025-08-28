import pytest
from numpy.testing import assert_array_equal

from cubie import is_devfunc
from cubie.systemmodels.symbolic.symbolicODE import (
    SymbolicODE,
    create_ODE_system,
)


@pytest.fixture(scope="function")
def symbolic_input_simple():
    return {
        "observables": ["obs1", "obs2"],
        "parameters": {"k1": 0.32, "k2": 0.91},
        "constants": {"c1": 2.1, "c2": 1.8},
        "drivers": {"d1": 0.9, "d2": 0.8},
        "states": {"x1": 0.5, "x2": 2.0},
        "dxdt": [
            "obs1 = k1 * x1 * d2 + d1 * c1",
            "obs2 = c2 * c2 * k2 + x1 + x2 ** 2 + obs1",
            "dx1 = obs1 + c2",
            "dx2 = c1 + obs1 + obs2",
        ],
    }


@pytest.fixture(scope="function")
def simple_ode_strict(symbolic_input_simple):
    return SymbolicODE.create(
        dxdt=symbolic_input_simple["dxdt"],
        states=symbolic_input_simple["states"],
        parameters=symbolic_input_simple["parameters"],
        constants=symbolic_input_simple["constants"],
        observables=symbolic_input_simple["observables"],
        drivers=symbolic_input_simple["drivers"],
        name="simpletest_strict",
        strict=True,
    )


@pytest.fixture(scope="function")
def simple_ode_nonstrict(symbolic_input_simple):
    return SymbolicODE.create(
        dxdt=symbolic_input_simple["dxdt"],
        strict=False,
        name="simpletest_nonstrict",
    )


def test_create_ODE_system_strict(simple_ode_strict, symbolic_input_simple):
    sys1 = create_ODE_system(
        dxdt=symbolic_input_simple["dxdt"],
        states=symbolic_input_simple["states"],
        parameters=symbolic_input_simple["parameters"],
        constants=symbolic_input_simple["constants"],
        observables=symbolic_input_simple["observables"],
        drivers=symbolic_input_simple["drivers"],
        name="simpletest_strict",
        strict=True,
    )
    sys2 = simple_ode_strict
    assert_array_equal(
        sys1.constants.values_array, sys2.constants.values_array
    )
    assert_array_equal(
        sys1.parameters.values_array, sys2.parameters.values_array
    )
    assert_array_equal(
        sys1.initial_values.values_array, sys2.initial_values.values_array
    )
    assert_array_equal(
        sys1.observables.values_array, sys2.observables.values_array
    )
    assert_array_equal(sys1.num_drivers, sys2.num_drivers)


def test_create_ODE_system_nonstrict(
    simple_ode_nonstrict, symbolic_input_simple
):
    sys1 = create_ODE_system(
        dxdt=symbolic_input_simple["dxdt"],
        name="simpletest_nonstrict",
    )
    sys2 = simple_ode_nonstrict
    assert_array_equal(
        sys1.constants.values_array, sys2.constants.values_array
    )
    assert_array_equal(
        sys1.parameters.values_array, sys2.parameters.values_array
    )
    assert_array_equal(
        sys1.initial_values.values_array, sys2.initial_values.values_array
    )
    assert_array_equal(
        sys1.observables.values_array, sys2.observables.values_array
    )
    assert_array_equal(sys1.num_drivers, sys2.num_drivers)


@pytest.fixture(scope="function")
def built_simple_strict(simple_ode_strict):
    simple_ode_strict.build()
    return simple_ode_strict


@pytest.fixture(scope="function")
def built_simple_nonstrict(simple_ode_nonstrict):
    simple_ode_nonstrict.build()
    return simple_ode_nonstrict

@pytest.mark.nocudasim
def test_simple_strict_builds(built_simple_strict):
    assert is_devfunc(built_simple_strict.jvp_function)

@pytest.mark.nocudasim
def test_simple_nonstrict_builds(built_simple_nonstrict):
    assert is_devfunc(built_simple_nonstrict.jvp_function)