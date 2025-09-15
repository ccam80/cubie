"""Tests for IVPLoop with Euler variants."""

import numpy as np
import pytest

from cubie.integrators.algorithms_ import (
    BackwardsEulerStep,
    ExplicitEulerStep,
)

from tests.integrators.loops.ODELoopTester import ODELoopTester


@pytest.mark.nocudasim
class TestExplicitEulerLoop(ODELoopTester):
    """Tests for IVPLoop with explicit Euler steps."""

    @pytest.fixture(scope="function")
    def step_object(self, system, buffer_sizes, precision):
        return ExplicitEulerStep(
            system.dxdt_function,
            precision,
            system.sizes.states,
            0.1,
        )

    @pytest.fixture(scope="function")
    def expected_state(self, system, initial_state):
        params = system.parameters.values_dict
        consts = system.constants.values_dict
        a, b = params["a"], params["b"]
        c0, c1 = consts["c0"], consts["c1"]
        h = 0.1
        x0, x1 = initial_state
        x0_sol = (x0 - c0 / a) * np.exp(-a * h) + c0 / a
        x1_sol = (x1 - c1 / b) * np.exp(-b * h) + c1 / b
        return np.array([x0_sol, x1_sol], dtype=np.float32)

    @pytest.fixture(scope="function")
    def algorithm_update_param(self):
        return {"fixed_step_size": 0.05}


@pytest.mark.nocudasim
class TestBackwardsEulerLoop(ODELoopTester):
    """Tests for IVPLoop with backward Euler steps."""

    @pytest.fixture(scope="function")
    def step_object(self, system, buffer_sizes):
        return BackwardsEulerStep(
            precision=np.float32,
            n=system.sizes.states,
            dxdt_function=system.dxdt_function,
            get_solver_helper_fn=system.get_solver_helper,
            atol=1e-6,
            rtol=1e-6,
        )

    @pytest.fixture(scope="function")
    def expected_state(self, system, initial_state):
        params = system.parameters.values_dict
        consts = system.constants.values_dict
        a, b = params["a"], params["b"]
        c0, c1 = consts["c0"], consts["c1"]
        h = 0.1
        x0, x1 = initial_state
        x0_sol = (x0 - c0 / a) * np.exp(-a * h) + c0 / a
        x1_sol = (x1 - c1 / b) * np.exp(-b * h) + c1 / b
        return np.array([x0_sol, x1_sol], dtype=np.float32)

    @pytest.fixture(scope="function")
    def algorithm_update_param(self):
        return {"atol": 1e-4}

