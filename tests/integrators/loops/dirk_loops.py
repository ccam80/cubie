"""Utility helpers for testing :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations


import numpy as np
from numpy.typing import NDArray
import pytest

from tests._utils import assert_integration_outputs

Array = NDArray[np.floating]


DEFAULT_OVERRIDES = {
    'dt': 0.001953125,  # try an exactly-representable dt
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
        {"algorithm": "implicit_midpoint", "step_controller": "fixed"},
        id="dirk-implicit-midpoint",
        marks=pytest.mark.specific_algos,
    ),
    # pytest.param(
    #     {"algorithm": "trapezoidal_dirk", "step_controller": "fixed"},
    #     id="dirk-trapezoidal",
    #     marks=pytest.mark.specific_algos,
    # ),
    pytest.param(
        {"algorithm": "sdirk_2_2", "step_controller": "fixed"},
        id="dirk-sdirk-2-2",
        marks=pytest.mark.specific_algos,
    ),
    # pytest.param(
    #     {"algorithm": "lobatto_iiic_3", "step_controller": "fixed"},
    #     id="dirk-lobatto-iiic-3",
    #     marks=pytest.mark.specific_algos,
    # ),
]

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
