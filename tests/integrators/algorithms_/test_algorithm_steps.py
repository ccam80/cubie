import pytest

from cubie.integrators.algorithms_ import (
    ExplicitEulerStep,
    BackwardsEulerStep,
)

from tests.integrators.algorithms_.StepAlgorithmTester import StepAlgorithmTester


@pytest.mark.parametrize("system_type", ["linear", "nonlinear"], indirect=True)
class TestExplicitEulerStep(StepAlgorithmTester):
    """Tests for the explicit Euler step."""

    @pytest.fixture(scope="function")
    def algorithm_class(self):
        return ExplicitEulerStep


@pytest.mark.nocudasim
@pytest.mark.parametrize("system_type", ["linear", "nonlinear"], indirect=True)
class TestBackwardsEulerStep(StepAlgorithmTester):
    """Tests for the backward Euler step."""

    @pytest.fixture(scope="function")
    def algorithm_class(self):
        return BackwardsEulerStep

    @pytest.fixture(scope="function")
    def step_kwargs(self, system, precision):
        return {
            "dxdt_function": system.dxdt_function,
            "get_solver_helper_fn": system.get_solver_helper,
            "precision": precision,
            "n": system.sizes.states,
            "atol": 1e-6,
            "rtol": 1e-6,
            "preconditioner_order": 2,
        }
