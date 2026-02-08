"""Harness for inspecting instrumented CPU and CUDA step outputs."""

import pytest

from tests.integrated_numerical_tests.test_step_algorithms import (
    device_step_results  # noqa
)
from tests._utils import ALGORITHM_PARAM_SETS

from .conftest import print_comparison

@pytest.mark.parametrize(
    "solver_settings_override",
    ALGORITHM_PARAM_SETS,
    indirect=True,
)
@pytest.mark.specific_algos
@pytest.mark.sim_only
def test_instrumented_gpu_vs_cpu(
    instrumented_cpu_step_results,
    instrumented_step_results,
    device_step_results,
):
    """Print instrumented CPU and GPU arrays for two consecutive steps."""

    print_comparison(
        instrumented_cpu_step_results,
        instrumented_step_results,
        device_step_results,
    )
