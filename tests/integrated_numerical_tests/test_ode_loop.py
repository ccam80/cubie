"""Numerical correctness tests for the ODE integration loop.

Compares device loop outputs against CPU reference for various
algorithm, precision, and timing configurations.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._utils import (
    assert_integration_outputs,
    MID_RUN_PARAMS,
    merge_dicts,
    ALGORITHM_PARAM_SETS,
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
    atol = tolerance.abs_loose
    rtol = tolerance.rel_loose

    # High stage-count explicit methods compound roundoff between
    # GPU and CPU; relax tolerance for these.
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
        {"output_types": [  # combined metrics
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
        {  # no combos
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
        {  # 1st generation metrics
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
    """Verify all summary metrics produce numerically correct
    results in loop context.

    Uses loose tolerance (1e-5) because of roundoff in the
    second-derivative methods - the cpu reference functions have
    no precision enforcement.
    """
    assert_integration_outputs(
        cpu_loop_outputs,
        device_loop_outputs,
        output_functions,
        rtol=tolerance.rel_loose * 5,
        atol=tolerance.abs_loose * 5,
    )

    assert device_loop_outputs.status == 0, (
        "Integration should complete successfully"
    )


@pytest.mark.parametrize("solver_settings_override",
                         [{
                             'precision': np.float32,
                             'output_types': ['state', 'time'],
                             'duration': 1e-4,
                             'save_every': 2e-5,
                             't0': 1.0,
                             'algorithm': "euler",
                             'dt': 1e-7,
                         }],
                         indirect=True,
                         ids=[""])
def test_float32_small_timestep_accumulation(
    device_loop_outputs, precision
):
    """Verify time accumulates correctly with float32 and small dt."""
    assert device_loop_outputs.state[-2, -1] == pytest.approx(
        precision(1.00008)
    )


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
    assert np.isclose(device_loop_outputs.state[-2, -1],
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
                             'dt_max': 1e-6,
                         }],
                         indirect=True,
                         ids=[""])
def test_adaptive_controller_with_float32(
    device_loop_outputs, precision
):
    """Verify adaptive controllers work with float32 and small
    dt_min."""
    assert device_loop_outputs.state[-2, -1] == pytest.approx(
        precision(1.00008)
    )


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
def test_save_at_settling_time_boundary(
    device_loop_outputs, precision
):
    """Test save point occurring exactly at settling_time boundary."""
    assert device_loop_outputs.state[-1, -1] == precision(1.2)
    assert device_loop_outputs.state[-2, -1] == precision(1.1)


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
def test_final_summary(
    device_loop_outputs,
    precision,
):
    """Verify summaries collected at end of run with summaries unset.

    When all timing parameters are None, the loop should collect a
    summary at the end of the integration run.
    """
    state_summaries = device_loop_outputs.state_summaries

    assert state_summaries is not None, (
        "State summaries should be collected"
    )
    assert state_summaries.shape[0] >= 1, (
        "At least one summary should exist"
    )

    final_summary = state_summaries[0]
    assert not np.isnan(final_summary).any(), (
        "Summary should not contain NaN"
    )


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
        }
    ],
    indirect=True,
)
def test_summarise_every(
    device_loop_outputs,
    precision,
):
    """Verify summarise_every works without double-write.

    When both periodic summaries and summarise_last are enabled,
    the loop should collect summaries at regular intervals and also
    at the end.
    """
    state_summaries = device_loop_outputs.state_summaries

    assert state_summaries is not None, (
        "State summaries should be collected"
    )
    assert state_summaries.shape[0] >= 3, (
        "Multiple summaries expected"
    )

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

    When both periodic saves and save_last are enabled, the final
    state should be saved even if it doesn't align with a periodic
    save point.
    """
    states = device_loop_outputs.state

    assert states is not None, "State outputs should be collected"
    assert states.shape[0] >= 4, "At least 4 saves expected"

    final_time = states[-1, -1]
    assert final_time == pytest.approx(
        precision(0.15), rel=1e-5
    ), "Final save should be at t_end"


def test_finish_check_no_float32_stagnation():
    """Regression: finish check must not stagnate in float32.

    When t0 is large relative to dt, float32 addition stagnates:
    float32(1000) + float32(1e-12) == float32(1000).  The finish
    check at ode_loop.py line 703 uses float64 to avoid this.

    Builds a SingleIntegratorRun with no regular outputs so that
    the no-regular-outputs branch is exercised.  Session fixtures
    cannot test this because their loop has summaries baked in.

    Under cudasim, the f32 overshoot (~6e7 extra steps) still
    completes in seconds, so this test validates the code path
    exists and produces correct status rather than detecting a
    hang.  On real CUDA hardware the overshoot would be
    proportionally more costly.
    """
    from cubie.buffer_registry import buffer_registry
    from cubie.integrators.SingleIntegratorRun import (
        SingleIntegratorRun,
    )
    from tests.system_fixtures import (
        build_three_state_constant_deriv_system,
    )
    from tests._utils import run_device_loop

    precision = np.float32
    system = build_three_state_constant_deriv_system(precision)
    buffer_registry.reset()

    sir = SingleIntegratorRun(
        system=system,
        step_control_settings={"step_controller": "fixed", "dt": 1e-8},
        algorithm_settings={"algorithm": "euler"},
        output_settings={
            "output_types": ["state", "time"],
            "saved_state_indices": [0],
            "saved_observable_indices": [],
            "summarised_state_indices": [],
            "summarised_observable_indices": [],
        },
        loop_settings={},
    )

    # Verify the no-regular-outputs branch is exercised
    loop_cfg = sir._loop.compile_settings
    assert not loop_cfg.save_regularly
    assert not loop_cfg.summarise_regularly

    solver_config = {
        "warmup": np.float64(0.0),
        "duration": np.float64(1e-6),
        "t0": np.float64(1000.0),
        "driverspline_order": 3,
    }
    result = run_device_loop(
        singleintegratorrun=sir,
        system=system,
        initial_state=system.initial_values.values_array.astype(
            precision
        ),
        solver_config=solver_config,
    )
    assert result.status == 0
