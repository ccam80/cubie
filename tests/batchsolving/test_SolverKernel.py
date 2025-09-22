import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cubie.batchsolving._utils import ensure_nonzero_size
from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.outputhandling.output_sizes import BatchOutputSizes
from tests._utils import calculate_expected_summaries
from tests.integrators.cpu_reference import run_reference_loop


@pytest.fixture(scope="function")
def batchconfig_instance(system):
    return BatchGridBuilder.from_system(system)


@pytest.fixture(scope="function")
def square_drive(system, solver_settings, precision, request):
    """amplitude 1 square wave, request "cycles" to change default cycles per simulation from 5"""
    if hasattr(request, "param"):
        if "cycles" in request.param:
            cycles = request.getattr("cycles", 5)
    else:
        cycles = 5
    numvecs = system.sizes.drivers
    length = int(solver_settings["duration"] // solver_settings["dt_min"])
    driver = np.zeros((length, numvecs), dtype=precision)
    half_period = length // (2 * cycles)

    for i in range(cycles):
        driver[i * half_period : (i + 1) * half_period, :] = 1.0

    return driver


@pytest.fixture(scope="function")
def batch_settings_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def batch_settings(batch_settings_override):
    """Fixture providing default batch settings."""
    defaults = {
        "num_state_vals_0": 2,
        "num_state_vals_1": 0,
        "num_param_vals_0": 2,
        "num_param_vals_1": 0,
        "kind": "combinatorial",
    }

    if batch_settings_override:
        for key, value in batch_settings_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults


@pytest.fixture(scope="function")
def batch_request(system, batch_settings):
    """Parametrized batch settings."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    return {
        state_names[0]: np.linspace(
            0.1, 1.0, batch_settings["num_state_vals_0"]
        ),
        # system)
        state_names[1]: np.linspace(
            0.1, 1.0, batch_settings["num_state_vals_1"]
        ),
        # system)
        param_names[0]: np.linspace(
            0.1, 1.0, batch_settings["num_param_vals_0"]
        ),
        param_names[1]: np.linspace(
            0.1, 1.0, batch_settings["num_param_vals_1"]
        ),
    }


@pytest.fixture(scope="function")
def batch_input_arrays(batch_request, batch_settings, batchconfig_instance):
    return batchconfig_instance.grid_arrays(
        batch_request, kind=batch_settings["kind"]
    )


def test_kernel_builds(solverkernel):
    """Test that the solver builds without errors."""
    kernelfunc = solverkernel.kernel


@pytest.mark.parametrize(
    "system_override, solver_settings_override, batch_settings_override",
    (
        ({}, {}, {}),
        pytest.param(
            "three_chamber",
            {
                "duration": 1.0,
                "output_types": [
                    "state",
                    "observables",
                    "mean",
                    "max",
                    "rms",
                    "peaks[2]",
                ],
            },
            {},
        marks=pytest.mark.nocudasim
        ),
        # ("three_chamber",
        #  {'duration': 10.0, 'output_types':["state", "observables", "mean", "max"]},
        #  {'num_state_vals_0': 10, 'num_state_vals_1': 10, 'num_param_vals_0': 10, 'num_param_vals_1': 10})
    ),
    ids=["smoke_test", "fire_test (all outputs)"],
    # "10s threeCM runs"],
    indirect=True,
)
def test_run(
    solverkernel,
    batch_input_arrays,
    solver_settings,
    square_drive,
    batch_settings,
    expected_batch_answers_euler,
    expected_batch_summaries,
    precision,
):
    """Big integration tet. Runs a batch integratino and checks outputs match expected. Expensive, don't run
    "scorcher" in CI."""
    inits, params = batch_input_arrays

    solverkernel.run(
        duration=solver_settings["duration"],
        params=params,
        inits=inits,  # debug: inits has no varied parameters
        forcing_vectors=square_drive,
        blocksize=solver_settings["blocksize"],
        stream=solver_settings["stream"],
        warmup=solver_settings["warmup"],
    )

    # Check that outputs are as expected
    output_length = int(
        np.ceil(solver_settings["duration"] / solver_settings["dt_save"])
    )
    summaries_length = int(
        np.ceil(solver_settings["duration"] / solver_settings["dt_summarise"])
    )
    numruns = (
        (
            batch_settings["num_state_vals_0"]
            if batch_settings["num_state_vals_0"] != 0
            else 1
        )
        * (
            batch_settings["num_state_vals_1"]
            if batch_settings["num_state_vals_1"] != 0
            else 1
        )
        * (
            batch_settings["num_param_vals_0"]
            if batch_settings["num_param_vals_0"] != 0
            else 1
        )
        * (
            batch_settings["num_param_vals_1"]
            if batch_settings["num_param_vals_1"] != 0
            else 1
        )
    )

    active_output_arrays = solverkernel.active_output_arrays
    expected_state_output_shape = (
        output_length,
        numruns,
        len(solverkernel.saved_state_indices),
    )
    expected_observables_output_shape = (
        output_length,
        numruns,
        len(solverkernel.saved_observable_indices),
    )
    output_summaries_height = solverkernel.single_integrator._output_functions.summaries_output_height_per_var
    expected_state_summaries_shape = (
        summaries_length,
        numruns,
        len(solverkernel.summarised_state_indices) * output_summaries_height,
    )
    expected_observable_summaries_shape = (
        summaries_length,
        numruns,
        len(solverkernel.summarised_observable_indices)
        * output_summaries_height,
    )

    expected_state_output_shape = ensure_nonzero_size(
        expected_state_output_shape
    )
    expected_observables_output_shape = ensure_nonzero_size(
        expected_observables_output_shape
    )
    expected_state_summaries_shape = ensure_nonzero_size(
        expected_state_summaries_shape
    )
    expected_observable_summaries_shape = ensure_nonzero_size(
        expected_observable_summaries_shape
    )

    if active_output_arrays.state is False:
        expected_state_output_shape = (1, 1, 1)
    if active_output_arrays.observables is False:
        expected_observables_output_shape = (1, 1, 1)
    if active_output_arrays.state_summaries is False:
        expected_state_summaries_shape = (1, 1, 1)
    if active_output_arrays.observable_summaries is False:
        expected_observable_summaries_shape = (1, 1, 1)

    state = solverkernel.state
    observables = solverkernel.observables
    state_summaries = solverkernel.state_summaries
    observable_summaries = solverkernel.observable_summaries

    # Check sizes match
    assert state.shape == expected_state_output_shape
    assert observables.shape == expected_observables_output_shape
    assert state_summaries.shape == expected_state_summaries_shape
    assert observable_summaries.shape == expected_observable_summaries_shape

    # Check that the arrays are not empty
    if active_output_arrays.state is True:
        with pytest.raises(AssertionError):
            assert_array_equal(
                state,
                np.zeros(state.shape, dtype=precision),
                err_msg="No output found",
            )
    if active_output_arrays.observables is True:
        with pytest.raises(AssertionError):
            assert_array_equal(
                observables,
                np.zeros(observables.shape, dtype=precision),
                err_msg="No observables output found",
            )
    if active_output_arrays.state_summaries is True:
        with pytest.raises(AssertionError):
            assert_array_equal(
                state_summaries,
                np.zeros(state_summaries.shape, dtype=precision),
                err_msg="No state summaries_array output found",
            )
    if active_output_arrays.observable_summaries is True:
        with pytest.raises(AssertionError):
            assert_array_equal(
                observable_summaries,
                np.zeros(observable_summaries.shape, dtype=precision),
                err_msg=("No observable summaries_array output found"),
            )

    # Set tolerance based on precision
    if precision == np.float32:
        atol = 1e-5
        rtol = 1e-5
    else:
        atol = 1e-12
        rtol = 1e-12

    for i, (expected_state_output, expected_observables_output) in enumerate(
        expected_batch_answers_euler
    ):
        expected_state_summaries = expected_batch_summaries[i][0]
        expected_obs_summaries = expected_batch_summaries[i][1]
        if active_output_arrays.state:
            assert_allclose(
                expected_state_output,
                state[:, i, :],
                atol=atol,
                rtol=rtol,
                err_msg="Output does not match expected.",
            )
        if active_output_arrays.observables:
            assert_allclose(
                expected_observables_output,
                observables[:, i, :],
                atol=atol,
                rtol=rtol,
                err_msg="Observables do not match expected.",
            )
        if active_output_arrays.state_summaries:
            assert_allclose(
                expected_state_summaries,
                state_summaries[:, i, :],
                atol=atol,
                rtol=rtol,
                err_msg="Summary states do not match expected.",
            )
        if active_output_arrays.observable_summaries:
            assert_allclose(
                expected_obs_summaries,
                observable_summaries[:, i, :],
                atol=atol,
                rtol=rtol,
                err_msg="Summary observables do not match expected.",
            )


def test_algorithm_change(solverkernel):
    solverkernel.update({"algorithm": "generic"})
    assert (
        solverkernel.single_integrator._integrator_instance.shared_memory_required
        == 0
    )


def test_getters_get(solverkernel):
    """Check for dead getters"""
    assert solverkernel.shared_memory_bytes_per_run is not None, (
        "BatchSolverKernel.shared_memory_bytes_per_run returning None"
    )
    assert solverkernel.shared_memory_elements_per_run is not None, (
        "BatchSolverKernel.shared_memory_elements_per_run returning None"
    )
    assert solverkernel.precision is not None, (
        "BatchSolverKernel.precision returning None"
    )
    assert solverkernel.threads_per_loop is not None, (
        "BatchSolverKernel.threads_per_loop returning None"
    )
    assert solverkernel.output_heights is not None, (
        "BatchSolverKernel.output_heights returning None"
    )
    assert solverkernel.output_length is not None, (
        "BatchSolverKernel.output_length returning None"
    )
    assert solverkernel.summaries_length is not None, (
        "BatchSolverKernel.summaries_length returning None"
    )
    assert solverkernel.num_runs is not None, (
        "BatchSolverKernel.num_runs returning None"
    )
    assert solverkernel.system is not None, (
        "BatchSolverKernel.system returning None"
    )
    assert solverkernel.duration is not None, (
        "BatchSolverKernel.duration returning None"
    )
    assert solverkernel.warmup is not None, (
        "BatchSolverKernel.warmup returning None"
    )
    assert solverkernel.dt_save is not None, (
        "BatchSolverKernel.dt_save returning None"
    )
    assert solverkernel.dt_summarise is not None, (
        "BatchSolverKernel.dt_summarise returning None"
    )
    assert solverkernel.system_sizes is not None, (
        "BatchSolverKernel.system_sizes returning None"
    )
    assert solverkernel.ouput_array_sizes_2d is not None, (
        "BatchSolverKernel.ouput_array_sizes_2d returning None"
    )
    assert solverkernel.output_array_sizes_3d is not None, (
        "BatchSolverKernel.output_array_sizes_3d returning None"
    )
    assert solverkernel.summary_legend_per_variable is not None, (
        "BatchSolverKernel.summary_legend_per_variable returning None"
    )
    assert solverkernel.saved_state_indices is not None, (
        "BatchSolverKernel.saved_state_indices returning None"
    )
    assert solverkernel.saved_observable_indices is not None, (
        "BatchSolverKernel.saved_observable_indices returning None"
    )
    assert solverkernel.summarised_state_indices is not None, (
        "BatchSolverKernel.summarised_state_indices returning None"
    )
    assert solverkernel.summarised_observable_indices is not None, (
        "BatchSolverKernel.summarised_observable_indices returning None"
    )
    assert solverkernel.active_output_arrays is not None, (
        "BatchSolverKernel.active_output_arrays returning None"
    )
    # device arrays SHOULD be None.


def test_all_lower_plumbing(system, solverkernel):
    """Big plumbing integration check - check that config classes match exactly between an updated solver and one
    instantiated with the update settings."""
    new_settings = {
        "duration": 1.0,
        "dt_min": 0.0001,
        "dt_max": 0.01,
        "dt_save": 0.01,
        "dt_summarise": 0.1,
        "atol": 1e-2,
        "rtol": 1e-1,
        "saved_state_indices": [0, 1, 2],
        "saved_observable_indices": [0, 1, 2],
        "summarised_state_indices": [
            0,
        ],
        "summarised_observable_indices": [
            0,
        ],
        "output_types": [
            "state",
            "observables",
            "mean",
            "max",
            "rms",
            "peaks[3]",
        ],
        "precision": np.float64,
    }
    solverkernel.update(new_settings)
    freshsolver = BatchSolverKernel(system, algorithm="euler", **new_settings)

    assert freshsolver.compile_settings == solverkernel.compile_settings, (
        "BatchSolverConfig mismatch"
    )
    assert (
        freshsolver.single_integrator.config
        == solverkernel.single_integrator.config
    ), "IntegratorRunSettings mismatch"
    assert (
        freshsolver.single_integrator._output_functions.compile_settings
        == solverkernel.single_integrator._output_functions.compile_settings
    ), "OutputFunctions mismatch"
    assert (
        freshsolver.single_integrator._system.compile_settings
        == solverkernel.single_integrator._system.compile_settings
    ), "SystemCompileSettings mismatch"
    assert BatchOutputSizes.from_solver(
        freshsolver
    ) == BatchOutputSizes.from_solver(solverkernel), (
        "BatchOutputSizes mismatch"
    )


def test_bogus_update_fails(solverkernel):
    solverkernel.update(dt_min=0.0001)
    with pytest.raises(KeyError):
        solverkernel.update(obviously_bogus_key="this should not work")


@pytest.fixture(scope="function")
def expected_batch_answers_euler(
    system,
    solver_settings,
    implicit_step_settings,
    batch_input_arrays,
    square_drive,
    cpu_step_controller,
    output_functions,
    cpu_system,
    step_controller_settings,
):
    inits, params = batch_input_arrays
    driver_vec = square_drive.T if square_drive.ndim == 2 else square_drive
    precision = system.precision
    driver_matrix = driver_vec.astype(precision)
    solver_config = {
        "dt_min": float(solver_settings["dt_min"]),
        "dt_max": float(solver_settings["dt_max"]),
        "dt_save": float(solver_settings["dt_save"]),
        "dt_summarise": float(solver_settings["dt_summarise"]),
        "warmup": float(solver_settings["warmup"]),
        "duration": float(solver_settings["duration"]),
        "atol": float(solver_settings["atol"]),
        "rtol": float(solver_settings["rtol"]),
    }
    output_sets = []
    for i in range(inits.shape[0]):
        run_inputs = {
            "initial_values": inits[i, :].astype(precision),
            "parameters": params[i, :].astype(precision),
            "forcing_vectors": driver_matrix,
        }
        result = run_reference_loop(
            evaluator=cpu_system,
            inputs=run_inputs,
            solver_settings=solver_config,
            implicit_step_settings=implicit_step_settings,
            controller=cpu_step_controller,
            output_functions=output_functions,
            step_controller_settings=step_controller_settings,
        )
        output_sets.append((result["state"], result["observables"]))
    return output_sets


@pytest.fixture(scope="function")
def expected_batch_summaries(
    expected_batch_answers_euler,
    solver_settings,
    output_functions,
    precision,
):
    """
    Calculate the expected summaries_array for the loop algorithm.

    Usage example:
    @pytest.mark.parametrize("summarise_every", [10], indirect=True)
    def test_expected_summaries(expected_summaries):
        ...
    """
    saves_per_summary = max(
        int(
            np.ceil(
                float(solver_settings["dt_summarise"])
                / float(solver_settings["dt_save"])
            )
        ),
        1,
    )
    expected_summaries = []
    for expected_state, expected_output in expected_batch_answers_euler:
        expected_summaries.append(
            calculate_expected_summaries(
                expected_state,
                expected_output,
                saves_per_summary,
                solver_settings["output_types"],
                output_functions.summaries_output_height_per_var,
                precision,
            )
        )
    return expected_summaries
