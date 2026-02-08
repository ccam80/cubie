"""Batch integration numerical test.

Runs a batch kernel integration and checks outputs match CPU
reference. Expensive test — uses LONG_RUN_PARAMS.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba import cuda

from tests._utils import (
    LONG_RUN_PARAMS,
    LoopRunResult,
    assert_integration_outputs,
)
#
#
# @pytest.mark.parametrize(
#     "solver_settings_override",
#     (
#         {"system_type": "three_chamber",
#          **LONG_RUN_PARAMS},
#     ),
#     ids=["smoke_test"],
#     indirect=True,
# )
# def test_run(
#     solverkernel,
#     batch_input_arrays,
#     solver_settings,
#     batch_settings,
#     cpu_batch_results,
#     precision,
#     system,
#     driver_array,
#     output_functions,
#     driver_settings,
#     tolerance,
# ):
#     """Batch integration test: run batch and check outputs match
#     CPU reference."""
#     inits, params = batch_input_arrays
#
#     solverkernel.run(
#         duration=solver_settings["duration"],
#         params=params,
#         inits=inits,
#         driver_coefficients=driver_array.coefficients,
#         blocksize=solver_settings["blocksize"],
#         stream=solver_settings["stream"],
#         warmup=solver_settings["warmup"],
#     )
#     cuda.synchronize()
#     state = solverkernel.state
#     observables = solverkernel.observables
#     state_summaries = solverkernel.state_summaries
#     observable_summaries = solverkernel.observable_summaries
#     iteration_counters = solverkernel.iteration_counters
#     device = LoopRunResult(
#         state=state,
#         observables=observables,
#         state_summaries=state_summaries,
#         observable_summaries=observable_summaries,
#         counters=iteration_counters,
#         status=0,
#     )
#
#     assert_integration_outputs(
#         device=device,
#         reference=cpu_batch_results,
#         output_functions=output_functions,
#         atol=tolerance.abs_loose,
#         rtol=tolerance.rel_loose,
#     )
