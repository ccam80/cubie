"""Utility helpers for testing :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations


import numpy as np
from numpy.typing import NDArray
import pytest

from tests._utils import assert_integration_outputs

Array = NDArray[np.floating]


DEFAULT_OVERRIDES = {
    # time-stepping values moved into individual loop cases below so we can
    # exercise representable / non-representable combinations explicitly.
    # 'dt', 'dt_min' and 'dt_save' are intentionally omitted here.
    'dt_max': 0.5,
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

    # sdirk_2_2 scenarios requested by the user. Each case explicitly
    # provides time-stepping values (dt, dt_save, dt_min where relevant)
    # so we can test representable vs non-representable behaviour.

    # 1) Exactly representable (single-precision) dt, integer-multiple
    # representable dt_save, fixed controller.
    pytest.param(
        {
            "algorithm": "sdirk_2_2",
            "step_controller": "fixed",
            "dt": 0.001953125,           # 1/512 (exact in binary float)
            "dt_save": 0.01953125,      # 10 * dt (exact multiple)
        },
        id="sdirk-2-2-fixed-exact-dt-integer-save",
        marks=pytest.mark.specific_algos,
    ),

    # 2) Non-representable dt, integer-multiple non-representable dt_save,
    # fixed controller.
    pytest.param(
        {
            "algorithm": "sdirk_2_2",
            "step_controller": "fixed",
            "dt": 0.001,                # decimal not exactly representable in binary
            "dt_save": 0.01,            # 10 * dt, also non-representable
        },
        id="sdirk-2-2-fixed-nonrep-dt-integer-save",
        marks=pytest.mark.specific_algos,
    ),

    # 3) Exactly representable dt, non-integer non-representable dt_save > 10x dt,
    # fixed controller.
    pytest.param(
        {
            "algorithm": "sdirk_2_2",
            "step_controller": "fixed",
            "dt": 0.001953125,         # exact
            "dt_save": 0.03,           # >10x dt, not integer multiple
        },
        id="sdirk-2-2-fixed-exact-dt-nonint-nonrep-save",
        marks=pytest.mark.specific_algos,
    ),

    # 4) Non-representable dt_min and non-representable dt_save, non-integer
    # multiples, fixed controller.
    pytest.param(
        {
            "algorithm": "sdirk_2_2",
            "step_controller": "fixed",
            "dt": 0.001953125,         # use an exact dt but vary dt_min/save
            "dt_save": 0.000123,      # non-representable, not integer multiple
        },
        id="sdirk-2-2-fixed-nonrep-dtmin-nonrep-save",
        marks=pytest.mark.specific_algos,
    ),

    # 5) i controller, non-representable dt_save
    pytest.param(
        {
            "algorithm": "sdirk_2_2",
            "step_controller": "i",
            "dt_save": 0.03,           # non-representable
        },
        id="sdirk-2-2-i-nonrep-save",
        marks=pytest.mark.specific_algos,
    ),

    # 6) i controller, representable dt_save
    pytest.param(
        {
            "algorithm": "sdirk_2_2",
            "step_controller": "i",
            "dt_save": 0.01953125,     # integer multiple, representable
        },
        id="sdirk-2-2-i-exact-save",
        marks=pytest.mark.specific_algos,
    ),
]


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
