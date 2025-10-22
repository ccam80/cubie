"""Utility helpers for testing :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
import pytest

from tests._utils import assert_integration_outputs

Array = NDArray[np.floating]


def _loop_override(**kwargs):
    base = {
        "dt": 0.0025,
        "dt_save": 0.0025,
        "duration":0.025,
        "output_types": ["state"],
        "saved_state_indices": [0, 1, 2],
    }
    base.update(kwargs)
    return base


LOOP_CASES = [
    pytest.param(
        _loop_override(
            algorithm="euler",
            step_controller="fixed",
            newton_tolerance=5e-6,
            krylov_tolerance=1e-6,
        ),
        id="euler",
    ),
    pytest.param(
        _loop_override(
            algorithm="backwards_euler",
            step_controller="fixed",
            newton_tolerance=5e-6,
            krylov_tolerance=1e-6,
        ),
        id="backwards_euler",
    ),
    pytest.param(
        _loop_override(
            algorithm="backwards_euler_pc",
            step_controller="fixed",
            newton_tolerance=5e-6,
            krylov_tolerance=1e-6,
        ),
        id="backwards_euler_pc",
    ),
    pytest.param(
        _loop_override(
            algorithm="crank_nicolson",
            step_controller="pid",
            atol=1e-5,
            rtol=1e-5,
            dt_min=1e-6,
            newton_tolerance=5e-6,
            krylov_tolerance=1e-6,
        ),
        id="crank_nicolson_pid",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="crank_nicolson",
            step_controller="pi",
            atol=1e-6,
            rtol=1e-6,
            dt_min=1e-6,
            newton_tolerance=5e-6,
            krylov_tolerance=1e-6,
        ),
        id="crank_nicolson_pi",
    ),
    pytest.param(
        _loop_override(
            algorithm="crank_nicolson",
            step_controller="i",
            atol=1e-5,
            rtol=1e-5,
            dt_min=1e-6,
        ),
        id="crank_nicolson_i",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="crank_nicolson",
            step_controller="gustafsson",
            atol=1e-5,
            rtol=1e-5,
            dt_min=1e-6,
        ),
        id="crank_nicolson_gustafsson",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="rosenbrock",
            step_controller="pid",
            atol=1e-5,
            rtol=1e-5,
            dt_min=1e-6,
            krylov_tolerance=1e-6,
        ),
        id="rosenbrock",
    ),
    pytest.param(
        _loop_override(
            algorithm="erk",
            step_controller="pid",
            atol=1e-6,
            rtol=1e-6,
            dt_min=1e-6,
        ),
        id="erk",
    ),
    pytest.param(
        _loop_override(
            algorithm="dirk",
            step_controller="pid",
            atol=1e-6,
            rtol=1e-6,
            dt_min=1e-6,
            newton_tolerance=5e-6,
            krylov_tolerance=1e-6,
        ),
        id="dirk",
    ),
    pytest.param(
        _loop_override(
            algorithm="ros3p",
            step_controller="pid",
            atol=1e-4,
            rtol=1e-4,
            dt_min=1e-6,
            krylov_tolerance=1e-6,
        ),
        id="rosenbrock-ros3p",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="rosenbrock_w6s4os",
            step_controller="pi",
            atol=1e-4,
            rtol=1e-4,
            dt_min=1e-6,
            krylov_tolerance=1e-6,
        ),
        id="rosenbrock-w6s4os",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="dopri54",
            step_controller="pi",
            atol=1e-6,
            rtol=1e-6,
            dt_min=1e-6,
        ),
        id="erk-dopri54",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="cash-karp-54",
            step_controller="pi",
            atol=1e-6,
            rtol=1e-6,
            dt_min=1e-6,
        ),
        id="erk-cash-karp-54",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="fehlberg-45",
            step_controller="pi",
            atol=1e-6,
            rtol=1e-6,
            dt_min=1e-6,
        ),
        id="erk-fehlberg-45",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="bogacki-shampine-32",
            step_controller="pi",
            atol=1e-6,
            rtol=1e-6,
            dt_min=1e-6,
        ),
        id="erk-bogacki-shampine-32",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="heun-21",
            step_controller="fixed",
        ),
        id="erk-heun-21",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="ralston-33",
            step_controller="fixed",
        ),
        id="erk-ralston-33",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="classical-rk4",
            step_controller="fixed",
        ),
        id="erk-classical-rk4",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="implicit_midpoint",
            step_controller="fixed",
            newton_tolerance=5e-6,
            krylov_tolerance=1e-6,
        ),
        id="dirk-implicit-midpoint",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="trapezoidal_dirk",
            step_controller="fixed",
            newton_tolerance=5e-6,
            krylov_tolerance=1e-6,
        ),
        id="dirk-trapezoidal",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="sdirk_2_2",
            step_controller="pi",
            atol=1e-6,
            rtol=1e-6,
            dt_min=1e-6,
            newton_tolerance=5e-6,
            krylov_tolerance=1e-6,
        ),
        id="dirk-sdirk-2-2",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _loop_override(
            algorithm="lobatto_iiic_3",
            step_controller="fixed",
            newton_tolerance=5e-6,
            krylov_tolerance=1e-6,
        ),
        id="dirk-lobatto-iiic-3",
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


@pytest.mark.parametrize("system_override",
                         ["three_chamber",
                          ],
                         ids=["3cm"], indirect=True)
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
    atol=tolerance.abs_tight
    rtol=tolerance.rel_tight
    assert_integration_outputs(
        reference=cpu_loop_outputs,
        device=device_loop_outputs,
        output_functions=output_functions,
        rtol=rtol,
        atol=atol,
    )
    assert device_loop_outputs.status == 0
