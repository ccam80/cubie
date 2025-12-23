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
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    # Be a little looser for odd controller/algo changes
    atol=tolerance.abs_loose
    rtol=tolerance.rel_loose
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
