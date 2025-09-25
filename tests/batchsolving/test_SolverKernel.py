import numpy as np
import pytest
from numpy.testing import assert_allclose

from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.outputhandling.output_sizes import BatchOutputSizes
from tests._utils import _driver_sequence




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
    batch_settings,
    cpu_batch_results,
    precision,
    system,
):
    """Big integration tet. Runs a batch integration and checks outputs
    match expected. Expensive, don't run
    "scorcher" in CI."""
    inits, params = batch_input_arrays

    samples = int(max(1,
            np.ceil(solver_settings["duration"] / solver_settings["dt_save"])))

    drivers = _driver_sequence(
        samples=samples,
        total_time=solver_settings["duration"],
        n_drivers=system.num_drivers,
        precision=precision,
    )

    solverkernel.run(
        duration=solver_settings["duration"],
        params=params,
        inits=inits,  # debug: inits has no varied parameters
        forcing_vectors=drivers,
        blocksize=solver_settings["blocksize"],
        stream=solver_settings["stream"],
        warmup=solver_settings["warmup"],
    )

    active_output_arrays = solverkernel.active_output_arrays

    state = solverkernel.state
    observables = solverkernel.observables
    state_summaries = solverkernel.state_summaries
    observable_summaries = solverkernel.observable_summaries

    # Set tolerance based on precision
    if precision == np.float32:
        atol = 1e-5
        rtol = 1e-5
    else:
        atol = 1e-12
        rtol = 1e-12

    if active_output_arrays.state:
        assert_allclose(
            cpu_batch_results.state,
            state,
            atol=atol,
            rtol=rtol,
            err_msg="Output does not match expected.",
        )
    if active_output_arrays.observables:
        assert_allclose(
            cpu_batch_results.observables,
            observables,
            atol=atol,
            rtol=rtol,
            err_msg="Observables do not match expected.",
        )
    if active_output_arrays.state_summaries:
        assert_allclose(
            cpu_batch_results.state_summaries,
            state_summaries,
            atol=atol,
            rtol=rtol,
            err_msg="Summary states do not match expected.",
        )
    if active_output_arrays.observable_summaries:
        assert_allclose(
            cpu_batch_results.observable_summaries,
            observable_summaries,
            atol=atol,
            rtol=rtol,
            err_msg="Summary observables do not match expected.",
        )


def test_algorithm_change(solverkernel):
    solverkernel.update({"algorithm": "backwards_euler",
                         "step_controller_kind": "pid"})
    assert (
        solverkernel.single_integrator._step_controller.atol is not None)


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
        "summarised_state_indices": [0],
        "summarised_observable_indices": [0],
        "output_types": [
            "state",
            "observables",
            "mean",
            "max",
            "rms",
            "peaks[3]",
        ],
    }
    solverkernel.update(new_settings)
    freshsolver = BatchSolverKernel(system, algorithm="euler", **new_settings)

    assert freshsolver.compile_settings == solverkernel.compile_settings, (
        "BatchSolverConfig mismatch"
    )
    assert (
        freshsolver.single_integrator.compile_settings
        == solverkernel.single_integrator.compile_settings
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


