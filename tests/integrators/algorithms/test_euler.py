import numpy as np
import pytest

from cubie.integrators.algorithms import Euler
from cubie.outputhandling.output_sizes import LoopBufferSizes
from integrators.cpu_reference import CPUODESystem
from tests.integrators.cpu_reference import run_reference_loop
from tests.integrators.algorithms.LoopAlgorithmTester import (
    LoopAlgorithmTester,
)


class TestEuler(LoopAlgorithmTester):
    """Testing class for the Euler algorithm. Checks the instantiation, compilation, and input/output for a range
    of cases, including incomplete inputs and random floats of different scales."""

    @pytest.mark.parametrize(
        "system_override", ["three_chamber", "stiff", "large"]
    )
    def test_loop_function_builds(self, built_loop_function):
        pass

    def test_loop_compile_settings_passed_successfully(
        self,
        loop_compile_settings_overrides,
        output_functions,
        loop_under_test,
        expected_summary_buffer_size,
    ):
        pass

    @pytest.fixture(scope="class")
    def algorithm_class(self):
        return Euler

    @pytest.fixture(scope="function")
    def expected_answer(
        self,
        system,
        loop_compile_settings,
        cpu_step_controller,
        step_controller_settings,
        solver_settings,
        inputs,
        precision,
        output_functions,
    ):

        evaluator = CPUODESystem(system)
        result = run_reference_loop(
            evaluator=evaluator,
            inputs=inputs,
            solver_settings=solver_settings,
            controller=cpu_step_controller,
            output_functions=output_functions,
            step_controller_settings=step_controller_settings,
        )

        return result["state"], result["observables"]

    @pytest.mark.nocudasim
    @pytest.mark.parametrize(
        "loop_compile_settings_overrides, inputs_override, solver_settings_override, "
        "precision_override",
        [
            (
                {
                    "output_functions": ["state", "observables", "time"],
                    "saved_state_indices": [0, 1, 2],
                    "saved_observable_indices": [0, 1],
                },
                {},
                {"duration": 0.1, "warmup": 0.0},
                {},
            ),
            (
                {
                    "output_functions": [
                        "state",
                        "observables",
                        "mean",
                        "max",
                        "rms",
                        "peaks[3]",
                    ]
                },
                {},
                {"duration": 1.0, "warmup": 1.0},
                {"precision": np.float64},
            ),
            (
                {
                    "output_functions": [
                        "state",
                        "observables",
                        "mean",
                        "max",
                        "rms",
                        "peaks[3]",
                    ]
                },
                {},
                {"duration": 5.0, "warmup": 1.0},
                {"precision": np.float32},
            ),
        ],
        ids=["no_summaries", "all_summaries_64", "all_summaries_long_32"],
        indirect=True,
    )
    def test_loop(
        self,
        loop_test_kernel,
        outputs,
        inputs,
        precision,
        output_functions,
        loop_under_test,
        expected_answer,
        expected_summaries,
        solverkernel,
    ):
        super().test_loop(
            loop_test_kernel,
            outputs,
            inputs,
            precision,
            output_functions,
            loop_under_test,
            expected_answer,
            expected_summaries,
            solverkernel,
        )

    @pytest.mark.parametrize(
        "loop_compile_settings_overrides",
        [
            {},
            {
                "dt_min": 0.002,
                "dt_max": 0.02,
                "dt_save": 0.02,
                "dt_summarise": 0.2,
                "atol": 1.0e-7,
                "rtol": 1.0e-5,
                "saved_state_indices": [0, 1, 2],
                "saved_observable_indices": [0, 2],
                "output_functions": ["state", "peaks[3]"],
            },
        ],
        ids=["no_change", "change_all"],
        indirect=True,
    )
    def test_loop_function_builds(self, built_loop_function):
        super().test_loop_function_builds(built_loop_function)

    @pytest.mark.parametrize(
        "loop_compile_settings_overrides",
        [
            {},
            {
                "dt_min": 0.002,
                "dt_max": 0.02,
                "dt_save": 0.02,
                "dt_summarise": 0.2,
                "atol": 1.0e-7,
                "rtol": 1.0e-5,
                "saved_state_indices": [0, 1, 2],
                "saved_observable_indices": [0, 2],
                "output_functions": ["state", "peaks[3]"],
            },
        ],
        ids=["no_change", "change_all"],
        indirect=True,
    )
    def test_loop_compile_settings_passed_successfully(
        self,
        loop_compile_settings_overrides,
        output_functions,
        loop_under_test,
        expected_summary_buffer_size,
    ):
        super().test_loop_compile_settings_passed_successfully(
            loop_compile_settings_overrides,
            output_functions,
            loop_under_test,
            expected_summary_buffer_size,
        )

    @pytest.fixture()
    def expected_loop_shared_memory(self, system, output_functions):
        """Calculate the expected shared memory size for the Euler algorithm."""
        sizes = LoopBufferSizes.from_system_and_output_fns(
            system, output_functions
        )
        loop_shared_memory = (
            sizes.state
            + sizes.dxdt
            + sizes.observables
            + sizes.parameters
            + sizes.drivers
            + sizes.state_summaries
            + sizes.observable_summaries
        )
        return loop_shared_memory
