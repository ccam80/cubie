"""Utility helpers for testing :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
import pytest

from cubie import summary_metrics
from tests._utils import (
    assert_integration_outputs,
    MID_RUN_PARAMS,
    merge_dicts,
    merge_param,
)

Array = NDArray[np.floating]


DEFAULT_OVERRIDES = MID_RUN_PARAMS


LOOP_CASES = [
    pytest.param(
        {"algorithm": "euler", "step_controller": "fixed"},
        id="euler",
    ),
    pytest.param(
        {"algorithm": "backwards_euler", "step_controller": "fixed"},
        id="backwards_euler",
    ),
    pytest.param(
        {"algorithm": "backwards_euler_pc", "step_controller": "fixed"},
        id="backwards_euler_pc",
    ),
    pytest.param(
        {"algorithm": "crank_nicolson", "step_controller": "pid"},
        id="crank_nicolson_pid",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "crank_nicolson", "step_controller": "pid"},
        id="crank_nicolson_pi",
    ),
    pytest.param(
        {"algorithm": "crank_nicolson", "step_controller": "i"},
        id="crank_nicolson_i",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "crank_nicolson", "step_controller": "gustafsson"},
        id="crank_nicolson_gustafsson",
        marks=pytest.mark.specific_algos,
    ), # Gustaffson looping infintely!
    pytest.param(
        {"algorithm": "erk", "step_controller": "pid"},
        id="erk",
    ),
    pytest.param(
        {"algorithm": "firk", "step_controller": "fixed"},
        id="firk",
    ),
    pytest.param(
        {"algorithm": "dirk", "step_controller": "fixed"},
        id="dirk",
    ),
    pytest.param(
        {"algorithm": "rosenbrock", "step_controller": "i"},
        id="rosenbrock-rodas3p",
    ),
    pytest.param(
        {"algorithm": "dopri54", "step_controller": "pid"},
        id="erk-dopri54",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "cash-karp-54", "step_controller": "pid"},
        id="erk-cash-karp-54",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "fehlberg-45", "step_controller": "i"},
        id="erk-fehlberg-45",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "bogacki-shampine-32", "step_controller": "pid"},
        id="erk-bogacki-shampine-32",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "heun-21", "step_controller": "fixed"},
        id="erk-heun-21",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "ralston-33", "step_controller": "fixed"},
        id="erk-ralston-33",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "classical-rk4", "step_controller": "fixed"},
        id="erk-classical-rk4",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "implicit_midpoint", "step_controller": "fixed"},
        id="dirk-implicit-midpoint",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "trapezoidal_dirk", "step_controller": "fixed"},
        id="dirk-trapezoidal",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "sdirk_2_2", "step_controller": "i"},
        id="dirk-sdirk-2-2",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "lobatto_iiic_3", "step_controller": "fixed"},
        id="dirk-lobatto-iiic-3",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "l_stable_dirk_3", "step_controller": "fixed"},
        id="dirk-l-stable-3",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "l_stable_sdirk_4", "step_controller": "i"},
        id="dirk-l-stable-4",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "dop853", "step_controller": "i"},
        id="erk-dop853",
        marks=pytest.mark.specific_algos,
    ),
    # pytest.param(
    #     {"algorithm": "radau", "step_controller": "i"},
    #     id="firk-radau",
    #     marks=pytest.mark.specific_algos,
    # ), #FSAL caching causing drift - on hold until we have an accurate
    # reference
    pytest.param(
        {"algorithm": "ode23s", "step_controller": "i"},
        id="rosenbrock-ode23s",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "tsit5", "step_controller": "i"},
        id="erk-tsit5",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "vern7", "step_controller": "i"},
        id="erk-vern7",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "Rodas4P", "step_controller": "i"},
        id="rosenbrock-rodas4p",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "Rodas5P", "step_controller": "i"},
        id="rosenbrock-rodas5p",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "firk_gauss_legendre_2", "step_controller": "fixed"},
        id="firk-gauss-legendre-2",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "rodas3p", "step_controller": "i"},
        id="rosenbrock-rodas3p",
        marks=pytest.mark.specific_algos,
    ),
]

# Create merged LOOP_CASES with DEFAULT_OVERRIDES baked in
LOOP_CASES_MERGED = [merge_param(DEFAULT_OVERRIDES, case)
                     for case in LOOP_CASES]

# Build, update, getter tests combined
def test_getters(
    loop_mutable,
    buffer_settings,
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
    assert (
        loop.local_memory_elements
        == 2
    ), "local_memory getter"
    assert (
        loop.shared_memory_elements
        == loop.compile_settings.shared_buffer_indices.local_end
    ), "shared_memory getter"
    assert loop.compile_settings.shared_buffer_indices is not None, \
        "shared_buffer_indices getter"
    # test update
    loop.update({"dt_save": 2 * solver_settings["dt_save"]})
    assert loop.dt_save == pytest.approx(
        2 * solver_settings["dt_save"], rel=1e-6, abs=1e-6
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "duration": 0.05,
            "dt_save" : 0.05,
            "algorithm": "crank_nicolson",
            "output_types": ["state", "observables"],
        },
    ],
    indirect=True,
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


# @pytest.mark.parametrize("system_override",
#                          ["three_chamber",
#                           ],
#                          ids=["3cm"], indirect=True)
@pytest.mark.parametrize(
    "solver_settings_override",
    LOOP_CASES_MERGED,
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

# One per metric
#Awful to read, but we add [2] to peaks or negative peaks, clumsily
metric_test_output_cases = tuple({"output_types": [metric]} if metric not in ["peaks",
         "negative_peaks"] else {"output_types": [metric + "[2]"]} for
                                 metric in
         summary_metrics.implemented_metrics )
metric_test_ids = tuple(metric for metric in
                        summary_metrics.implemented_metrics)
# Add combos
metric_test_output_cases = (
        *metric_test_output_cases,
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
        *metric_test_ids,
        "combined metrics",
        "no combos",
        "1st generation metrics"
)

# Base settings for metric tests
METRIC_TEST_BASE = {
    "system_type": "linear",
    "algorithm": "euler",
    "duration": 0.2,
    "dt": 0.0025,
    "dt_save": 0.01,
    "dt_summarise": 0.1,
}

METRIC_TEST_CASES_MERGED = [merge_dicts(METRIC_TEST_BASE, case)
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
        rtol=tolerance.rel_loose * 3, # Added tolerance - x/dt_save**2 is rough
        atol=tolerance.abs_loose,
    )
    
    assert device_loop_outputs.status == 0, "Integration should complete successfully"


@pytest.mark.parametrize("solver_settings_override",
                         [{
                             'precision': np.float32,
                             'output_types': ['state', 'time'],
                             'duration': 1e-4,
                             'dt_save': 2e-5,
                             't0': 1.0,
                             'algorithm': "euler",
                             'dt': 1e-7,
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
                             'dt_max': 1e-6,
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
