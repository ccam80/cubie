"""Harness for inspecting instrumented CPU and CUDA step outputs."""

import pytest

from tests.integrators.algorithms.test_step_algorithms import (
    STEP_CASES,
    device_step_results # noqa
)

from .conftest import print_comparison


@pytest.mark.parametrize(
    "solver_settings_override2",
    [{'max_linear_iters': 3,
      'max_newton_iters': 3,
      'newton_max_backtracks': 3}],
    ids=["4iters"],
    indirect=True,
)
@pytest.mark.parametrize(
    "solver_settings_override, system_override",
    STEP_CASES,
    indirect=True,
)
@pytest.mark.specific_algos
def test_instrumented_gpu_vs_cpu(
    instrumented_cpu_step_results,
    instrumented_step_results,
    device_step_results,
):
    """Print instrumented CPU and GPU arrays for manual inspection."""

    print_comparison(
        instrumented_cpu_step_results,
        instrumented_step_results,
        device_step_results,
    )
