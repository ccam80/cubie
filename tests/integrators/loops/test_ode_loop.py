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
        {"algorithm": "euler", "step_controller": "fixed", "dt_min": 0.05},
        {"algorithm": "crank_nicolson", "step_controller": "pid", 'atol': 1e-6,
         'rtol': 1e-6, 'dt_min': 0.005},
    ],
    indirect=True,
)
class TestLoop:

    def test_build(self, loop, step_controller, step_object,
                   loop_buffer_sizes, precision, solver_settings,
                   device_loop_outputs, cpu_loop_outputs, output_functions):
        assert isinstance(loop.device_function, Callable), "Loop builds"

        #Test getters get
        assert loop.is_adaptive == step_controller.is_adaptive, "is_adaptive getter"
        assert loop.precision == precision, "precision getter"
        assert loop.dt0 == step_controller.dt0, "dt0 getter"
        assert loop.dt_min == step_controller.dt_min, "dt_min getter"
        assert loop.dt_max == step_controller.dt_max, "dt_max getter"
        assert loop.dt_save == precision(solver_settings['dt_save']), \
            "dt_save getter"
        assert loop.dt_summarise == precision(solver_settings[
                                                  'dt_summarise']),\
            "dt_summarise getter"
        assert (loop.local_memory_elements ==
                step_object.persistent_local_required +
                step_controller.local_memory_elements +
                loop_buffer_sizes.state + 3), "local_memory getter"
        assert loop.shared_memory_elements == (
                step_object.shared_memory_required + loop.buffer_indices.local_end
        ), "shared_memory getter"
        assert loop.buffer_indices is not None, "buffer_indices getter"

        #test update
        if solver_settings['algorithm'].lower() == 'euler':
            updates = {'dt': 0.0001}
            loop.update(updates)
            assert step_controller.dt_min == pytest.approx(0.0001, rel=1e-6,
                                                           abs=1e-6), \
                "fixed controller dt_min update"
            assert step_object.dt == pytest.approx(0.0001, rel=1e-6,
                                                   abs=1e-6),\
                "euler step dt update"
        else:
            updates = {'dt_min': 0.0001,
                       'atol': 1e-12,
                       'ki': 2.0,
                       'max_newton_iters': 512}
            loop.update(updates)
            assert step_controller.dt_min == pytest.approx(
                    0.0001, rel=1e-6, abs=1e-6),\
                "adaptive controller dt_min update"
            assert step_controller.atol == pytest.approx(
                    1e-12, rel=1e-6, abs=1e-6), \
                "adaptive controller atol update"
            assert step_object.compile_settings.max_newton_iters == 512, \
                "CN step max_newton_iters update"
            assert loop.dt_min == pytest.approx(
                    0.0001, rel=1e-6, abs=1e-6), \
                "Loop dt_min update"

        assert device_loop_outputs.status == 0
        assert_integration_outputs(
                cpu_loop_outputs,
                device_loop_outputs,
                output_functions,
                rtol=1e-5,
                atol=1e-6)