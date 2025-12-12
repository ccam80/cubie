"""Harness for inspecting instrumented CPU and CUDA step outputs."""

import pytest

from tests.integrators.algorithms.test_step_algorithms import (
    device_step_results  # noqa
)
from tests._utils import MID_RUN_PARAMS, merge_param

from .conftest import print_comparison

STEP_SETTINGS = MID_RUN_PARAMS.copy()
STEP_SETTINGS.update(
       {'dt': 0.01,
      'max_linear_iters': 3,
      'max_newton_iters': 3,
      'newton_max_backtracks': 2,
      'krylov_tolerance': 1e-7,
      'newton_tolerance': 1e-7,
      'correction_type': 'minimal_residual'
    })


STEP_CASES_INSTRUMENTED = [merge_param(STEP_SETTINGS, case)
                           for case in STEP_CASES]


@pytest.mark.parametrize(
    "solver_settings_override",
    STEP_CASES_INSTRUMENTED,
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
