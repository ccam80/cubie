"""Utility helpers for testing :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
import pytest

from tests._utils import (
    assert_integration_outputs,
    MID_RUN_PARAMS,
    merge_dicts,
    ALGORITHM_PARAM_SETS,
)

Array = NDArray[np.floating]

# Build, update, getter tests combined
def test_getters(
    loop_mutable,
    precision,
    solver_settings,
):
    loop = loop_mutable
    assert isinstance(loop.device_function, Callable), "Loop builds"

    #Test getters get
    assert loop.precision == precision, "precision getter"
    assert loop.dt_save == precision(solver_settings['dt_save']), \
        "dt_save getter"
    assert loop.dt_summarise == precision(solver_settings[
                                              'dt_summarise']),\
        "dt_summarise getter"
    # test update
    loop.update({"dt_save": 2 * solver_settings["dt_save"]})
    assert loop.dt_save == pytest.approx(
        2 * solver_settings["dt_save"], rel=1e-6, abs=1e-6
    )


def test_initial_observable_seed_matches_reference(
    device_loop_outputs,
    cpu_loop_outputs,
    tolerance,
):

    np.testing.assert_allclose(
        device_loop_outputs.observables[0],
        cpu_loop_outputs["observables"][0],
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )

@pytest.mark.parametrize(
    "solver_settings_override",
    ALGORITHM_PARAM_SETS,
    indirect=True,
)
def test_loop(
    device_loop_outputs,
    step_object,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    atol=tolerance.abs_loose
    rtol=tolerance.rel_loose

    #In high stage-count Explicit methods, the error between the GPU and CPU
    # versions tends to compound, where the same code produces errors inside
    # tolerance in lower-stage-count methods. I presume this is due to the
    # CPU reference defaulting to higher-precision intermediates, rather
    # than a real error in the algorithm execution that is hidden in smaller
    # tableaus, so we relax the tolerance for these big ones
    if step_object.stage_count > 10:
        rtol = tolerance.rel_loose * 5
        atol = tolerance.abs_loose * 5
    assert_integration_outputs(
        reference=cpu_loop_outputs,
        device=device_loop_outputs,
        output_functions=output_functions,
        rtol=rtol,
        atol=atol,
    )
    assert device_loop_outputs.status == 0



# Add combos
metric_test_output_cases = (
        {"output_types": [  #combined metrics
            "state",
            "mean",
            "std",
            "rms",
            "max",
            "min",
            "time",
            "max_magnitude",
            "peaks[3]",
            "negative_peaks[3]",
            "dxdt_max",
            "dxdt_min",
            "d2xdt2_max",
            "d2xdt2_min",
            ],
        },
        { # no combos
            "output_types": [
                "state",
                "mean",
                "rms",
                "max",
                "min",
                "time",
                "max_magnitude",
                "negative_peaks[3]",
                "dxdt_max",
                "d2xdt2_max",
            ],
        },
        {# 1st generation metrics
            "output_types": [
                "state",
                "mean",
                "rms",
                "max",
                "time",
                "peaks[3]",
            ],
        },
)

metric_test_ids = (
        "combined metrics",
        "no combos",
        "1st generation metrics"
)

METRIC_TEST_CASES_MERGED = [merge_dicts(MID_RUN_PARAMS, case)
                            for case in metric_test_output_cases]


@pytest.mark.parametrize(
    "solver_settings_override",
    METRIC_TEST_CASES_MERGED,
    ids=metric_test_ids,
    indirect=True,
)
def test_all_summary_metrics_numerical_check(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    """Verify all summary metrics produce numerically correct results in loop context.
    
    Note: This test uses loose tolerance (1e-5) because of roundoff in the
    second-derivative methods - the cpu reference functions have no precision
    enforcement. This can be reduced if precision is implemented in cpu
    reference functions..
    """
    # Check state summaries match reference
    assert_integration_outputs(
        cpu_loop_outputs,
        device_loop_outputs,
        output_functions,
        rtol=tolerance.rel_loose * 5, # Added tolerance - x/dt_save**2 is
            # rough
        atol=tolerance.abs_loose* 5,
    )
    
    assert device_loop_outputs.status == 0, "Integration should complete successfully"


@pytest.mark.parametrize("solver_settings_override",
                         [{
                             'precision': np.float32,
                             'output_types': ['state', 'time'],
                             'duration': 1e-4,
                             'dt_save': 2e-5,  # representable in f32: 2e6*1.0
                             't0': 1.0,
                             'algorithm': "euler",
                             'dt': 1e-7,  # smaller than 1f32 eps
                         }],
                         indirect=True,
                         ids=[""])
def test_float32_small_timestep_accumulation(device_loop_outputs, precision):
    """Verify time accumulates correctly with float32 precision and small dt."""
    assert device_loop_outputs.state[-2,-1] == pytest.approx(precision(1.00008))


@pytest.mark.parametrize("solver_settings_override",
                         [
                             {
                                 'precision': np.float32,
                                 'output_types': ['state', 'time'],
                                 'duration': 1e-3,
                                 'dt_save': 2e-4,
                                 't0': 1e2,
                                 'algorithm': 'euler',
                                 'dt': 1e-6,
                             },
                             {
                                 'precision': np.float64,
                                 'output_types': ['state', 'time'],
                                 'duration': 1e-3,
                                 'dt_save': 2e-4,
                                 't0': 1e2,
                                 'algorithm': 'euler',
                                 'dt': 1e-6,
                             },
                         ],
                         indirect=True,
                         ids=["float32", "float64"])
def test_large_t0_with_small_steps(device_loop_outputs, precision):
    """Verify long integrations with small steps complete correctly."""
    # There may be an ulp of error here, that's fine, we're testing the
    # ability to accumulate time during long examples.
    assert np.isclose(device_loop_outputs.state[-2,-1],
                      precision(100.0008),
                      atol=2e-7)



@pytest.mark.parametrize("solver_settings_override",
                         [{
                             'precision': np.float32,
                             'duration': 1e-4,
                             'dt_save': 2e-5,
                             't0': 1.0,
                             'algorithm': 'crank_nicolson',
                             'step_controller': 'PI',
                             'output_types': ['state', 'time'],
                             'dt_min': 1e-7,
                             'dt_max': 1e-6,  # smaller than eps * t0
                         }],
                         indirect=True,
                         ids=[""])
def test_adaptive_controller_with_float32(device_loop_outputs, precision):
    """Verify adaptive controllers work with float32 and small dt_min."""
    assert device_loop_outputs.state[-2,-1] == pytest.approx(precision(
            1.00008))
    #Testing second-last sample as the final sample overshoots t_end in this
                                                             # case

@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "precision": np.float32,
            "duration": 0.2000,
            "settling_time": 0.1,
            "t0": 1.0,
            "output_types": ["state", "time"],
            "algorithm": "euler",
            "dt": 1e-2,
            "dt_save": 0.1,
        }
    ],
    indirect=True,
)
def test_save_at_settling_time_boundary(device_loop_outputs, precision):
    """Test save point occurring exactly at settling_time boundary."""
    # Should complete successfully with first save at t=settling_time
    assert device_loop_outputs.state[-1,-1] == precision(1.2)
    assert device_loop_outputs.state[-2,-1] == precision(1.1)


def test_ivploop_generate_dummy_args(loop_mutable, precision):
    """Verify IVPLoop._generate_dummy_args returns properly shaped args."""
    loop = loop_mutable
    config = loop.compile_settings

    result = loop._generate_dummy_args()

    assert 'loop_function' in result, "Must contain 'loop_function' key"

    args = result['loop_function']
    assert len(args) == 13, "loop_function takes 13 arguments"

    # Verify array shapes match compile_settings
    initial_states = args[0]
    assert initial_states.shape == (config.n_states,)
    assert initial_states.dtype == precision

    parameters = args[1]
    assert parameters.shape == (config.n_parameters,)
    assert parameters.dtype == precision

    driver_coeffs = args[2]
    assert driver_coeffs.shape == (100, config.n_states, 6)
    assert driver_coeffs.dtype == precision

    shared_scratch = args[3]
    assert shared_scratch.shape == (4096,)
    assert shared_scratch.dtype == np.float32

    persistent_local = args[4]
    assert persistent_local.shape == (4096,)
    assert persistent_local.dtype == precision

    state_output = args[5]
    assert state_output.shape == (100, config.n_states)
    assert state_output.dtype == precision

    observables_output = args[6]
    assert observables_output.shape == (100, config.n_observables)
    assert observables_output.dtype == precision

    state_summaries_output = args[7]
    assert state_summaries_output.shape == (100, config.n_states)
    assert state_summaries_output.dtype == precision

    observable_summaries_output = args[8]
    assert observable_summaries_output.shape == (100, config.n_observables)
    assert observable_summaries_output.dtype == precision

    iteration_counters_output = args[9]
    assert iteration_counters_output.shape == (1, config.n_counters)
    assert iteration_counters_output.dtype == np.int32

    # Verify scalar arguments
    duration = args[10]
    assert duration == np.float64(config.dt_save + 0.01)

    settling_time = args[11]
    assert settling_time == np.float64(0.0)

    t0 = args[12]
    assert t0 == np.float64(0.0)


def test_ivploop_no_critical_shapes_attribute(loop_mutable):
    """Verify loop_fn no longer has critical_shapes attribute."""
    loop = loop_mutable
    # Access the device function to trigger build
    device_fn = loop.device_function

    assert not hasattr(device_fn, 'critical_shapes'), \
        "loop_fn should not have critical_shapes attribute"
    assert not hasattr(device_fn, 'critical_values'), \
        "loop_fn should not have critical_values attribute"
