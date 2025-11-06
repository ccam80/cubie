"""Utility helpers for testing :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import pytest

from cubie import summary_metrics
from tests._utils import assert_integration_outputs

Array = NDArray[np.floating]


DEFAULT_OVERRIDES = {
    'dt': 0.001,
    'dt_min': 1e-8,
    'dt_max': 0.5,
    'dt_save': 0.01953125,
    'newton_tolerance': 1e-7,
    'krylov_tolerance': 1e-7,
    'atol': 1e-5,
    'rtol': 1e-6,
    'output_types': ["state", "time"],
    'saved_state_indices': [0, 1, 2],
}

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
        {"algorithm": "crank_nicolson", "step_controller": "pi"},
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
    ),
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
        {"algorithm": "cash-karp-54", "step_controller": "pi"},
        id="erk-cash-karp-54",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "fehlberg-45", "step_controller": "i"},
        id="erk-fehlberg-45",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "bogacki-shampine-32", "step_controller": "pi"},
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
    pytest.param(
        {"algorithm": "radau", "step_controller": "i"},
        id="firk-radau",
        marks=pytest.mark.specific_algos,
    ),
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

# Build, update, getter tests combined
def test_getters(
    loop_mutable,
    loop_buffer_sizes,
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
            "algorithm": "crank_nicolson",
            "output_types": ["state", "observables"],
        },
    ],
    indirect=True,
)
def test_initial_observable_seed_matches_reference(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    """Ensure the initial observable snapshot reflects the initial state."""
    if not output_functions.compile_flags.save_observables:
        pytest.skip("Observables are not saved for this configuration.")

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
    "solver_settings_override2",
    [DEFAULT_OVERRIDES],
    indirect=True,
    ids=[""],
)
@pytest.mark.parametrize(
    "solver_settings_override",
    LOOP_CASES,
    indirect=True,
)
def  test_loop(
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

@pytest.mark.parametrize("system_override", ["linear"], ids=[""],
                         indirect=True)
@pytest.mark.parametrize("solver_settings_override2",
     [{
        "algorithm": "euler",
        "duration": 0.2,
        "dt": 0.0025,
        "dt_save": 0.01,
        "dt_summarise": 0.1,
    }],
    indirect=True,
    ids = [""]
)
@pytest.mark.parametrize(
    "solver_settings_override",
    metric_test_output_cases,
    ids = metric_test_ids,
    indirect=True,
)
@pytest.mark.parametrize(
    "tolerance_override",
    [
        SimpleNamespace(
            abs_loose=1e-5,
            abs_tight=2e-6,
            rel_loose=1e-5,
            rel_tight=2e-6,
        )
    ],
    ids=["relaxed_tolerance"],
    indirect=True,
)
def test_all_summary_metrics_numerical_check(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    """Verify all summary metrics produce numerically correct results in loop context.
    
    Note: This test uses relaxed tolerance (2e-6) instead of the default tight 
    tolerance (1e-7) because the shifted-data algorithm for std calculations, 
    while significantly more stable than the naive formula, still accumulates 
    ~1-2e-6 error with float32 due to the nature of numerical integration.
    This is acceptable as 1e-7 is tighter than float32 machine epsilon (1.19e-7).
    """
    # Check state summaries match reference
    assert_integration_outputs(
        cpu_loop_outputs,
        device_loop_outputs,
        output_functions,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )
    
    assert device_loop_outputs.status == 0, "Integration should complete successfully"
