"""Utility helpers for testing :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
import pytest

from tests._utils import assert_integration_outputs

Array = NDArray[np.floating]

# Build, update, getter tests combined into one large test to avoid paying
# setup cost multiple times. Numerical tests are done on pre-updated
# settings as the fixtures are set up at function start.
@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {"algorithm": "euler", "step_controller": "fixed", "dt_min": 0.01},
        {"algorithm": "crank_nicolson", "step_controller": "pid", 'atol': 1e-6,
         'rtol': 1e-6, 'dt_min': 0.001},
    ],
    indirect=True,
)
def test_build(loop, step_controller, step_object,
               loop_buffer_sizes, precision, solver_settings,
               device_loop_outputs, cpu_loop_outputs, output_functions):
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
        == loop_buffer_sizes.state + 2
    ), "local_memory getter"
    assert (
        loop.shared_memory_elements
        == loop.compile_settings.shared_buffer_indices.local_end
    ), "shared_memory getter"
    assert loop.compile_settings.shared_buffer_indices is not None, \
        "shared_buffer_indices getter"

    #test update
    loop.update({"dt_save": 2 * solver_settings["dt_save"]})
    assert loop.dt_save == pytest.approx(
        2 * solver_settings["dt_save"], rel=1e-6, abs=1e-6
    )

    assert device_loop_outputs.status == 0
    assert_integration_outputs(
            cpu_loop_outputs,
            device_loop_outputs,
            output_functions,
            rtol=1e-5,
            atol=1e-6)