"""Utility helpers for testing :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
import pytest

from tests._utils import assert_integration_outputs

Array = NDArray[np.floating]

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
        == loop_buffer_sizes.state + 2
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


@pytest.mark.parametrize("system_override",
                         ["three_chamber",
                          ],
                         ids=["3cm"], indirect=True)
@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "algorithm": "euler",
            "step_controller": "fixed",
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6
        },
        {
            "algorithm": "backwards_euler",
            "step_controller": "fixed",
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {
            "algorithm": "backwards_euler_pc",
            "step_controller": "fixed",
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {
            "algorithm": "crank_nicolson",
            "step_controller": "pid",
            "atol": 1e-5,
            "rtol": 1e-5,
            "dt_min": 1e-6,
            "dt_save": 0.2,
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {
            "algorithm": "crank_nicolson",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 1e-6,
            "dt_save": 0.2,
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {
            "algorithm": "crank_nicolson",
            "step_controller": "i",
            "atol": 1e-5,
            "rtol": 1e-5,
            "dt_min": 1e-6,
            "dt_save": 0.2,
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
        },
        {
            "algorithm": "crank_nicolson",
            "step_controller": "gustafsson",
            "atol": 1e-5,
            "rtol": 1e-5,
            "dt_min": 1e-6,
            "dt_save": 0.2,
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
        },
    ],
    ids=["euler", "bweuler", "bweulerpc", "cnpid", "cnpi", "cni", "cngust"],
    indirect=True,
)
def test_loop(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    # Be a little looser for odd controller/algo changes
    atol=tolerance.abs_loose * 5
    rtol=tolerance.rel_loose * 5
    assert_integration_outputs(
            reference=cpu_loop_outputs,
            device=device_loop_outputs,
            output_functions=output_functions,
            rtol=rtol,
            atol=atol)
    assert device_loop_outputs.status == 0