import pytest

from tests._utils import assert_integration_outputs


# Build, update, getter tests combined into one large test to avoid paying
# setup cost multiple times. Numerical tests are done on pre-updated
# settings as the fixtures are set up at function start.
@pytest.mark.parametrize("system_override",
                         ["three_chamber",
                          ],
                         ids=["3cm"], indirect=True)
# @pytest.mark.parametrize("precision_override",
#                          [np.float64,
#                           ],
#                          ids=["64"], indirect=True)
@pytest.mark.parametrize("implicit_step_settings_override",
                         [{"nonlinear_tolerance": 1e-5,
                           "linear_tolerance": 1e-6},
                          ],
                         ids=["tol"],
                         indirect=True)
@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "algorithm": "euler",
            "step_controller": "fixed",
            "dt_min": 0.0025,
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
        },
        {
            "algorithm": "backwards_euler",
            "step_controller": "fixed",
            "dt_min": 0.0025,
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
        },
        {
            "algorithm": "backwards_euler_pc",
            "step_controller": "fixed",
            "dt_min": 0.0025,
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
        },
        {
            "algorithm": "crank_nicolson",
            "step_controller": "pid",
            "atol": 1e-5,
            "rtol": 1e-5,
            "dt_min": 1e-6,
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
        },
        {
            "algorithm": "crank_nicolson",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 1e-6,
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
        },
        {
            "algorithm": "crank_nicolson",
            "step_controller": "i",
            "atol": 1e-5,
            "rtol": 1e-5,
            "dt_min": 1e-6,
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
            "output_types": [
                "state",
            ],
            "saved_state_indices": [0, 1, 2],
        },
    ],
    ids=["euler","bweuler", "bweulerpc", "cnpid", "cnpi", "cni", "cngust"],
    indirect=True,
)
def test_loop(
    loop,
    step_controller,
    step_object,
    loop_buffer_sizes,
    precision,
    solver_settings,
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