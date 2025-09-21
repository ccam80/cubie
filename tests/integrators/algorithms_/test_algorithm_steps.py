"""Tests for single-step integrator algorithms against CPU references."""

import pytest

from cubie.integrators.algorithms_.backwards_euler import BackwardsEulerStep
from cubie.integrators.algorithms_.backwards_euler_predict_correct import (
    BackwardsEulerPCStep,
)
from cubie.integrators.algorithms_.crank_nicolson import CrankNicolsonStep
from cubie.integrators.algorithms_.explicit_euler import ExplicitEulerStep

from tests.integrators.algorithms_.StepAlgorithmTester import StepAlgorithmTester


@pytest.mark.parametrize(
    "system_override", ["linear", "nonlinear"], indirect=True
)
@pytest.mark.parametrize(
    ("step_size_override", "tolerances_override"),
    [
        (0.05, {"rtol": 5e-5, "atol": 5e-6}),
        (0.1, {"rtol": 1e-4, "atol": 1e-5}),
        (0.25, {"rtol": 2e-3, "atol": 5e-4}),
    ],
    indirect=["step_size_override", "tolerances_override"],
)
class TestExplicitEulerStep(StepAlgorithmTester):
    """Tests for the explicit Euler step."""

    @pytest.fixture(scope="function")
    def algorithm_class(self):
        return ExplicitEulerStep


@pytest.mark.parametrize(
    "system_override", ["linear", "nonlinear"], indirect=True
)
@pytest.mark.parametrize(
    ("step_size_override", "tolerances_override"),
    [
        (0.05, {"rtol": 1e-6, "atol": 1e-7}),
        (0.1, {"rtol": 5e-6, "atol": 5e-7}),
        (0.2, {"rtol": 5e-5, "atol": 5e-6}),
    ],
    indirect=["step_size_override", "tolerances_override"],
)
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


@pytest.mark.parametrize(
    "system_override", ["linear", "nonlinear"], indirect=True
)
@pytest.mark.parametrize(
    ("step_size_override", "tolerances_override"),
    [
        (0.05, {"rtol": 1e-6, "atol": 1e-7}),
        (0.1, {"rtol": 5e-6, "atol": 5e-7}),
        (0.2, {"rtol": 5e-5, "atol": 5e-6}),
    ],
    indirect=["step_size_override", "tolerances_override"],
)
class TestBackwardsEulerPredictCorrectStep(StepAlgorithmTester):
    """Tests for the backward Euler predictor-corrector step."""

    @pytest.fixture(scope="function")
    def algorithm_class(self):
        return BackwardsEulerPCStep

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


@pytest.mark.parametrize(
    "system_override", ["linear", "nonlinear"], indirect=True
)
@pytest.mark.parametrize(
    ("step_size_override", "tolerances_override"),
    [
        (0.05, {"rtol": 5e-6, "atol": 5e-7}),
        (0.1, {"rtol": 1e-5, "atol": 1e-6}),
        (0.2, {"rtol": 5e-5, "atol": 5e-6}),
    ],
    indirect=["step_size_override", "tolerances_override"],
)
class TestCrankNicolsonStep(StepAlgorithmTester):
    """Tests for the Crank–Nicolson step."""

    @pytest.fixture(scope="function")
    def algorithm_class(self):
        return CrankNicolsonStep

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
