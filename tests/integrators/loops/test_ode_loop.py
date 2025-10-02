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
@pytest.mark.parametrize("system_override", ["three_chamber"], indirect=True)
@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {"algorithm": "euler", "step_controller": "fixed", "dt_min": 0.01},
        {
            # "algorithm": "euler",
            # "step_controller": "fixed",
            "dt_min": 0.0025,
            # "dt_save": 0.02,
            "duration": 1.0,
            "output_types": [
                "state",
                "time",
                "observables",
                "mean",
                "max",
                "rms",
                "peaks[2]"],
            # "dt_summarise": 0.1,
            'saved_state_indices': [0,1,2],
            'saved_observable_indices': [0,1,2],
            'summarised_state_indices': [0,1,2],
            'summarised_observable_indices': [0,1,2],
        },
        # {
        #     "algorithm": "crank_nicolson",
        #     "step_controller": "pid",
        #     "atol": 1e-6,
        #     "rtol": 1e-6,
        #     "dt_min": 1e-6,
        #     "output_types": [
        #         "state",
        #         "time",
        #         "observables",
        #         "mean",
        #         "max",
        #         "rms",
        #         "peaks[2]",
        #     ],
        # },
        {"output_types": ["state", "observables", "time", "mean", "rms"],
          "duration": 0.6},
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


    #CPU test seems machine-dependent, this is a kludge from CI in
    # crank-nicolson
    if loop.is_adaptive:
        atol = 1e-4
        rtol = 1e-4
    else:
        atol=1e-5
        rtol=1e-5
    assert_integration_outputs(
            reference=cpu_loop_outputs,
            device=device_loop_outputs,
            output_functions=output_functions,
            rtol=rtol,
            atol=atol)
    assert device_loop_outputs.status == 0


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "algorithm": "euler",
            "step_controller": "fixed",
            "dt_min": 0.01,
            "output_types": ["state", "observables"],
        },
        {
            "algorithm": "crank_nicolson",
            "step_controller": "pid",
            "dt_min": 1e-6,
            "output_types": ["state", "observables"],
        },
    ],
    indirect=True,
)
def test_initial_observable_seed_matches_reference(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
):
    """Ensure the initial observable snapshot reflects the initial state."""
    if not output_functions.compile_flags.save_observables:
        pytest.skip("Observables are not saved for this configuration.")

    np.testing.assert_allclose(
        device_loop_outputs.observables[0],
        cpu_loop_outputs["observables"][0],
        rtol=1e-7,
        atol=1e-7,
    )
