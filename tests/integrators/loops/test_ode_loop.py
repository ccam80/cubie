"""Utility helpers for testing :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
import pytest

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
    #    Gustaffson controller will diverge due to different niters with a
    #    matrix-based solver vs matrix-free newton-krylov.
    # pytest.param(
    #     {"algorithm": "crank_nicolson", "step_controller": "gustafsson"},
    #     id="crank_nicolson_gustafsson",
    #     marks=pytest.mark.specific_algos,
    # ),
    # pytest.param( # Rosenbrock is not correct at this stage (24/10)
    #     {"algorithm": "rosenbrock", "step_controller": "pi"},
    #     id="rosenbrock",
    # ),
    pytest.param(
        {"algorithm": "erk", "step_controller": "pid"},
        id="erk",
    ),
    pytest.param(
        {"algorithm": "dirk", "step_controller": "fixed"},
        id="dirk",
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
        {"algorithm": "l_stable_dirk_3", "step_controller": "pi"},
        id="dirk-l-stable-3",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {"algorithm": "l_stable_sdirk_4", "step_controller": "pi"},
        id="dirk-l-stable-4",
        marks=pytest.mark.specific_algos,
    ),
    # pytest.param(
    #     {"algorithm": "ros3p", "step_controller": "pi"},
    #     id="rosenbrock-ros3p",
    #     marks=pytest.mark.specific_algos,
    # ),
    # pytest.param(
    #     {"algorithm": "rosenbrock_w6s4os", "step_controller": "pi"},
    #     id="rosenbrock-w6s4os",
    #     marks=pytest.mark.specific_algos,
    # ),
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
