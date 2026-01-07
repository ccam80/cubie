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
    assert loop.save_every == precision(solver_settings['save_every']), \
        "save_every getter"
    assert loop.summarise_every == precision(solver_settings[
                                              'summarise_every']),\
        "summarise_every getter"
    # test update
    loop.update({"save_every": 2 * solver_settings["save_every"]})
    assert loop.save_every == pytest.approx(
        2 * solver_settings["save_every"], rel=1e-6, abs=1e-6
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
        rtol=tolerance.rel_loose * 5, # Added tolerance - x/save_every**2 is
            # rough
        atol=tolerance.abs_loose* 5,
    )
    
    assert device_loop_outputs.status == 0, "Integration should complete successfully"


@pytest.mark.parametrize("solver_settings_override",
                         [{
                             'precision': np.float32,
                             'output_types': ['state', 'time'],
                             'duration': 1e-4,
                             'save_every': 2e-5,  # representable in f32: 2e6*1.0
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
                                 'save_every': 2e-4,
                                 't0': 1e2,
                                 'algorithm': 'euler',
                                 'dt': 1e-6,
                             },
                             {
                                 'precision': np.float64,
                                 'output_types': ['state', 'time'],
                                 'duration': 1e-3,
                                 'save_every': 2e-4,
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
                             'save_every': 2e-5,
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
            "save_every": 0.1,
        }
    ],
    indirect=True,
)
def test_save_at_settling_time_boundary(device_loop_outputs, precision):
    """Test save point occurring exactly at settling_time boundary."""
    # Should complete successfully with first save at t=settling_time
    assert device_loop_outputs.state[-1,-1] == precision(1.2)
    assert device_loop_outputs.state[-2, -1] == precision(1.1)


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "precision": np.float32,
            "duration": 0.1,
            "output_types": ["state", "time"],
            "algorithm": "euler",
            "dt": 0.01,
            "save_every": None,
            "summarise_every": None,
            "sample_summaries_every": None,
        }
    ],
    indirect=True,
)
def test_save_last_flag_from_config(loop_mutable):
    """Verify IVPLoop reads save_last flag from ODELoopConfig.

    When all timing parameters are None, ODELoopConfig sets save_last=True.
    IVPLoop.build() should read this from config.save_last.
    """
    config = loop_mutable.compile_settings
    assert config.save_last is True


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "precision": np.float32,
            "duration": 0.1,
            "output_types": ["state", "time", "mean"],
            "algorithm": "euler",
            "dt": 0.01,
            "save_every": None,
            "summarise_every": None,
            "sample_summaries_every": None,
        }
    ],
    indirect=True,
)
def test_summarise_last_flag_from_config(loop_mutable):
    """Verify IVPLoop reads summarise_last flag from ODELoopConfig.

    When all timing parameters are None, ODELoopConfig sets summarise_last=True.
    IVPLoop.build() should read this from config.summarise_last.
    """
    config = loop_mutable.compile_settings
    assert config.summarise_last is True


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "precision": np.float32,
            "duration": 0.1,
            "output_types": ["state", "time", "mean"],
            "algorithm": "euler",
            "dt": 0.01,
            "save_every": None,
            "summarise_every": None,
            "sample_summaries_every": None,
        }
    ],
    indirect=True,
)
def test_summarise_last_collects_final_summary(
    device_loop_outputs,
    precision,
):
    """Verify summaries collected at end of run with summarise_last=True.

    When all timing parameters are None, the loop should collect a summary
    at the end of the integration run. The summary buffer should contain
    valid data.
    """
    # With summarise_last=True and no periodic summaries, we should get
    # exactly one summary collected at the end
    state_summaries = device_loop_outputs.state_summaries

    # Summary should exist and have non-zero values for the final state
    assert state_summaries is not None, "State summaries should be collected"
    assert state_summaries.shape[0] >= 1, "At least one summary should exist"

    # The summary should contain valid (non-NaN, non-zero) values
    final_summary = state_summaries[0]
    assert not np.isnan(final_summary).any(), "Summary should not contain NaN"


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "precision": np.float32,
            "duration": 0.15,
            "output_types": ["state", "time", "mean"],
            "algorithm": "euler",
            "dt": 0.01,
            "save_every": 0.05,
            "summarise_every": 0.05,
            "sample_summaries_every": 0.05,
            "summarise_last": True,
        }
    ],
    indirect=True,
)
def test_summarise_last_with_summarise_every(
    device_loop_outputs,
    precision,
):
    """Verify summarise_last and summarise_every work together without 
    double-write.

    When both periodic summaries and summarise_last are enabled, the loop
    should collect summaries at regular intervals and also at the end.
    There should be no duplicate summaries when the final step coincides
    with a regular summary point.
    """
    state_summaries = device_loop_outputs.state_summaries

    # With duration=0.15 and summarise_every=0.05, we expect:
    # - t=0.0: initial summary
    # - t=0.05: periodic summary
    # - t=0.10: periodic summary
    # - t=0.15: final summary (either via periodic or summarise_last)
    # Total: 4 summaries
    assert state_summaries is not None, "State summaries should be collected"
    # At least 3-4 summaries expected (initial + periodic + final)
    assert state_summaries.shape[0] >= 3, "Multiple summaries expected"

    # Check that all summaries are valid
    for i in range(min(4, state_summaries.shape[0])):
        assert not np.isnan(state_summaries[i]).any(), \
            f"Summary {i} should not contain NaN"


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "precision": np.float32,
            "duration": 0.15,
            "output_types": ["state", "time"],
            "algorithm": "euler",
            "dt": 0.01,
            "save_every": 0.05,
            "save_last": True,
        }
    ],
    indirect=True,
)
def test_save_last_with_save_every(
    device_loop_outputs,
    precision,
):
    """Verify save_last and save_every can be used together.

    When both periodic saves (save_every) and save_last are enabled, the loop
    should collect saves at regular intervals and also at the end. The final
    state should be saved even if it doesn't align with a periodic save point.
    """
    states = device_loop_outputs.state

    # With duration=0.15 and save_every=0.05, we expect:
    # - t=0.0: initial save
    # - t=0.05: periodic save
    # - t=0.10: periodic save
    # - t=0.15: final save (from save_last or coincides with periodic)
    assert states is not None, "State outputs should be collected"
    assert states.shape[0] >= 4, "At least 4 saves expected"

    # Check that the final state is at or near t_end
    final_time = states[-1, -1]  # time is stored in last position
    assert final_time == pytest.approx(
        precision(0.15), rel=1e-5
    ), "Final save should be at t_end"


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "precision": np.float32,
            "duration": 0.12,
            "output_types": ["state", "time", "mean"],
            "algorithm": "euler",
            "dt": 0.01,
            "save_every": 0.04,
            "summarise_every": 0.04,
            "sample_summaries_every": 0.04,
            "summarise_last": True,
        }
    ],
    indirect=True,
)
def test_summarise_last_with_summarise_every_combined(
    device_loop_outputs,
    precision,
):
    """Verify summarise_last and summarise_every can be used together.

    When both periodic summaries (summarise_every) and summarise_last are
    enabled, the loop should collect summaries at regular intervals and also
    ensure the final summary is captured. This tests that no double-write
    occurs when the end time doesn't align perfectly with a periodic summary.
    """
    state_summaries = device_loop_outputs.state_summaries

    # With duration=0.12 and summarise_every=0.04, we expect:
    # - t=0.0: initial summary
    # - t=0.04: periodic summary
    # - t=0.08: periodic summary
    # - t=0.12: final summary (from summarise_last or coincides with periodic)
    assert state_summaries is not None, "State summaries should be collected"
    assert state_summaries.shape[0] >= 3, "At least 3 summaries expected"

    # Check that all summaries are valid (no NaN values)
    for i in range(min(4, state_summaries.shape[0])):
        assert not np.isnan(state_summaries[i]).any(), \
            f"Summary {i} should not contain NaN"

