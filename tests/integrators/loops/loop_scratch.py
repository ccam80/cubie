from __future__ import annotations


import pytest
import numpy as np
from tests._utils import assert_integration_outputs
from cubie.time_logger import _default_timelogger

_default_timelogger.set_verbosity('verbose')
np.set_printoptions(threshold=np.inf, precision=14, linewidth=120)

DEFAULT_OVERRIDES = {
    'algorithm': 'tsit5',
    'step_controller': 'PID',
    'dt_min': 1e-8,
    'dt_max': 1e-3,
    'newton_tolerance': 1e-7,
    'krylov_tolerance': 1e-7,
    'atol': 1e-5,
    'rtol': 1e-6,
    'output_types': ["state", "time", "counters"],
    'saved_state_indices': [0, 1, 2],
}

@pytest.mark.parametrize(
    "solver_settings_override2",
    [DEFAULT_OVERRIDES],
    indirect=True,
    ids=[""],
)
@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {"dt_save": 0.0025, "duration": 0.0025},
        {"dt_save": 0.033, "duration": 0.0333},
        {"dt_save": 0.001, "duration": 0.01},
        {"dt_save": 0.033, "duration": 0.3},
        {"dt_save": 0.0025, "duration": 0.025},
        {"dt_save": 0.1},
    ],
    ids=[
        "save_0.0025_quick",
        "save_0.033_quick",
        "save_0.001",
        "save_0.033",
        "save_0.0025",
        "save_0.1",
    ],
    indirect=True,
)
def  test_loop(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
        solver
):
    solver.set_verbosity('verbose')
    # Be a little looser for odd controller/algo changes
    atol = tolerance.abs_loose
    rtol = tolerance.rel_loose

    def _prepend_index(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[:, None]
        nrows = arr.shape[0]
        idx = np.arange(nrows, dtype=int)[:, None]
        return np.concatenate([idx, arr], axis=1)

    print("\n Delta")
    print(
        _prepend_index(device_loop_outputs.state - cpu_loop_outputs["state"])
    )
    print("\n Device:")
    print(_prepend_index(device_loop_outputs.state))
    print("\n CPU:")
    print(_prepend_index(cpu_loop_outputs["state"]))
    assert_integration_outputs(
        reference=cpu_loop_outputs,
        device=device_loop_outputs,
        output_functions=output_functions,
        rtol=rtol,
        atol=atol,
    )
    assert device_loop_outputs.status == 0
