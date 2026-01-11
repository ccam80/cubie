import numpy as np
import pytest

from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.outputhandling.output_sizes import BatchOutputSizes
from cubie.outputhandling.output_config import OutputCompileFlags
from cubie.batchsolving.BatchSolverConfig import ActiveOutputs


def test_kernel_builds(solverkernel):
    """Test that the solver builds without errors."""
    kernelfunc = solverkernel.kernel


# @pytest.mark.parametrize(
#     "solver_settings_override",
#     (
#         {"system_type": "three_chamber",
#          **LONG_RUN_PARAMS},
#     ),
#     ids=["smoke_test"],
#     indirect=True,
# )
# def test_run(
#     solverkernel,
#     batch_input_arrays,
#     solver_settings,
#     batch_settings,
#     cpu_batch_results,
#     precision,
#     system,
#     driver_array,
#     output_functions,
#     driver_settings,
#         tolerance
# ):
#     """Big integration test. Runs a batch integration and checks outputs
#     match expected. Expensive."""
#     inits, params = batch_input_arrays
#
#     solverkernel.run(
#         duration=solver_settings["duration"],
#         params=params,
#         inits=inits,
#         driver_coefficients=driver_array.coefficients,
#         blocksize=solver_settings["blocksize"],
#         stream=solver_settings["stream"],
#         warmup=solver_settings["warmup"],
#     )
#     cuda.synchronize()
#     state = solverkernel.state
#     observables = solverkernel.observables
#     state_summaries = solverkernel.state_summaries
#     observable_summaries = solverkernel.observable_summaries
#     iteration_counters = solverkernel.iteration_counters
#     device = LoopRunResult(state=state,
#                            observables=observables,
#                            state_summaries=state_summaries,
#                            observable_summaries=observable_summaries,
#                            counters=iteration_counters,
#                            status=0)
#
#     assert_integration_outputs(device=device,
#                                reference=cpu_batch_results,
#                                output_functions=output_functions,
#                                atol=tolerance.abs_loose,
#                                rtol=tolerance.rel_loose)


def test_algorithm_change(solverkernel_mutable):
    solverkernel = solverkernel_mutable
    solverkernel.update(
        {"algorithm": "crank_nicolson", "step_controller": "pid"}
    )
    assert solverkernel.single_integrator._step_controller.atol is not None


def test_getters_get(solverkernel):
    """Check for dead getters"""
    assert solverkernel.shared_memory_bytes is not None, (
        "BatchSolverKernel.shared_memory_bytes returning None"
    )
    assert solverkernel.shared_memory_elements is not None, (
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
    assert solverkernel.save_every is not None, (
        "BatchSolverKernel.save_every returning None"
    )
    assert solverkernel.summarise_every is not None, (
        "BatchSolverKernel.summarise_every returning None"
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
    assert solverkernel.active_outputs is not None, (
        "BatchSolverKernel.active_outputs returning None"
    )
    # device arrays SHOULD be None.


def test_all_lower_plumbing(
    system,
    solverkernel_mutable,
    step_controller_settings,
    algorithm_settings,
    precision,
    driver_array,
):
    """Big plumbing integration check - check that config classes match exactly between an updated solver and one
    instantiated with the update settings."""
    solverkernel = solverkernel_mutable
    new_settings = {
        # "duration": 1.0,
        "dt_min": 0.0001,
        "dt_max": 0.01,
        "save_every": 0.01,
        "summarise_every": 0.1,
        "sample_summaries_every": 0.05,
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
    updated_controller_settings = step_controller_settings.copy()
    updated_controller_settings.update(
        {
            "dt_min": 0.0001,
            "dt_max": 0.01,
            "atol": 1e-2,
            "rtol": 1e-1,
        }
    )
    output_settings = {
        "saved_state_indices": np.asarray([0, 1, 2]),
        "saved_observable_indices": np.asarray([0, 1, 2]),
        "summarised_state_indices": np.asarray([0]),
        "summarised_observable_indices": np.asarray([0]),
        "output_types": [
            "state",
            "observables",
            "mean",
            "max",
            "rms",
            "peaks[3]",
        ],
    }
    freshsolver = BatchSolverKernel(
        system,
        step_control_settings=updated_controller_settings,
        algorithm_settings=algorithm_settings,
        output_settings=output_settings,
        loop_settings={
            "save_every": 0.01,
            "summarise_every": 0.1,
            "sample_summaries_every": 0.05,
        },
    )
    inits = np.ones((3, 1), dtype=precision)
    params = np.ones((3, 1), dtype=precision)
    driver_coefficients = driver_array.coefficients
    freshsolver.run(
        inits=inits,
        params=params,
        driver_coefficients=driver_coefficients,
        duration=0.1,
    )
    solverkernel.run(
        inits=inits,
        params=params,
        driver_coefficients=driver_coefficients,
        duration=0.1,
    )
    assert (
        freshsolver.compile_settings.local_memory_elements
        == solverkernel.compile_settings.local_memory_elements
    ), "Local memory mismatch mismatch"
    assert (
        freshsolver.single_integrator.compile_settings
        == solverkernel.single_integrator.compile_settings
    ), "IntegratorRunSettings mismatch"
    assert (
        freshsolver.single_integrator._step_controller.compile_settings
        == solverkernel.single_integrator._step_controller.compile_settings
    )
    assert (
        freshsolver.single_integrator._algo_step.compile_settings
        == solverkernel.single_integrator._algo_step.compile_settings
    )
    assert (
        freshsolver.single_integrator._loop.compile_settings
        == solverkernel.single_integrator._loop.compile_settings
    )
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


def test_bogus_update_fails(solverkernel_mutable):
    solverkernel = solverkernel_mutable
    solverkernel.update(dt_min=0.0001)
    with pytest.raises(KeyError):
        solverkernel.update(obviously_bogus_key="this should not work")


class TestTimingParameterValidation:
    """Tests for timing parameter validation in BatchSolverKernel.run()."""

    def test_save_every_greater_than_duration_no_save_last_raises(
        self, system, precision, driver_array
    ):
        """Test that save_every > duration without save_last raises."""
        kernel = BatchSolverKernel(
            system,
            loop_settings={"save_every": 1.0},
            output_settings={
                "output_types": ["state"],
                "saved_state_indices": np.array([0, 1, 2]),
            },
            algorithm_settings={"algorithm": "euler"},
        )
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        with pytest.raises(
            ValueError, match=r"save_every.*>.*duration.*no outputs"
        ):
            kernel.run(
                inits=inits,
                params=params,
                driver_coefficients=driver_array.coefficients,
                duration=0.5,
            )

    def test_save_every_greater_than_duration_with_save_last_succeeds(
        self, system, precision, driver_array
    ):
        """Test that save_every >= duration with save_last=True is valid."""
        kernel = BatchSolverKernel(
            system,
            loop_settings={"save_every": None},
            output_settings={
                "output_types": ["state"],
                "saved_state_indices": np.array([0, 1, 2]),
            },
            algorithm_settings={"algorithm": "euler"},
        )
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        # Should not raise when save_last is True (default when save_every=None)
        kernel.run(
            inits=inits,
            params=params,
            driver_coefficients=driver_array.coefficients,
            duration=0.5,
        )

    def test_summarise_every_greater_than_duration_raises(
        self, system, precision, driver_array
    ):
        """Test that summarise_every > duration raises."""
        kernel = BatchSolverKernel(
            system,
            loop_settings={
                "summarise_every": 1.0,
                "sample_summaries_every": 0.1,
            },
            output_settings={
                "output_types": ["mean"],
                "summarised_state_indices": np.array([0, 1, 2]),
            },
            algorithm_settings={"algorithm": "euler"},
        )
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        with pytest.raises(
            ValueError,
            match=r"summarise_every.*>.*duration.*no summary outputs",
        ):
            kernel.run(
                inits=inits,
                params=params,
                driver_coefficients=driver_array.coefficients,
                duration=0.5,
            )

    def test_sample_summaries_every_gte_summarise_every_raises(
        self, system, precision, driver_array
    ):
        """Test that sample_summaries_every >= summarise_every raises."""
        kernel = BatchSolverKernel(
            system,
            loop_settings={
                "summarise_every": 0.1,
                "sample_summaries_every": 0.2,
            },
            output_settings={
                "output_types": ["mean"],
                "summarised_state_indices": np.array([0, 1, 2]),
            },
            algorithm_settings={"algorithm": "euler"},
        )
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        with pytest.raises(
            ValueError, match=r"sample_summaries_every.*>=.*summarise_every"
        ):
            kernel.run(
                inits=inits,
                params=params,
                driver_coefficients=driver_array.coefficients,
                duration=1.0,
            )

    def test_valid_timing_parameters_succeeds(
        self, system, precision, driver_array
    ):
        """Test that valid timing parameters allow run to proceed."""
        kernel = BatchSolverKernel(
            system,
            loop_settings={
                "save_every": 0.1,
                "summarise_every": 0.5,
                "sample_summaries_every": 0.05,
            },
            output_settings={
                "output_types": ["state", "mean"],
                "saved_state_indices": np.array([0, 1, 2]),
                "summarised_state_indices": np.array([0, 1, 2]),
            },
            algorithm_settings={"algorithm": "euler"},
        )
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        # Should not raise with valid parameters
        kernel.run(
            inits=inits,
            params=params,
            driver_coefficients=driver_array.coefficients,
            duration=1.0,
        )


class TestActiveOutputsFromCompileFlags:
    """Tests for ActiveOutputs.from_compile_flags factory method."""

    def test_all_flags_true(self, precision):
        """Test mapping when all compile flags are enabled."""
        # Use specific flags (summarise_state, summarise_observables) which are
        # what ActiveOutputs.from_compile_flags() reads; the general 'summarise'
        # flag is redundant here but included for completeness
        flags = OutputCompileFlags(
            save_state=True,
            save_observables=True,
            summarise_observables=True,
            summarise_state=True,
            save_counters=True,
        )
        active = ActiveOutputs.from_compile_flags(flags)

        assert active.state is True
        assert active.observables is True
        assert active.state_summaries is True
        assert active.observable_summaries is True
        assert active.iteration_counters is True
        assert active.status_codes is True

    def test_all_flags_false(self, precision):
        """Test mapping when all compile flags are disabled."""
        flags = OutputCompileFlags(
            save_state=False,
            save_observables=False,
            summarise_observables=False,
            summarise_state=False,
            save_counters=False,
        )
        active = ActiveOutputs.from_compile_flags(flags)

        assert active.state is False
        assert active.observables is False
        assert active.state_summaries is False
        assert active.observable_summaries is False
        assert active.iteration_counters is False
        # status_codes is ALWAYS True
        assert active.status_codes is True

    def test_status_codes_always_true(self, precision):
        """Verify status_codes is always True regardless of flags."""
        flags = OutputCompileFlags()  # All defaults (False)
        active = ActiveOutputs.from_compile_flags(flags)
        assert active.status_codes is True

    def test_partial_flags(self, precision):
        """Test with only some flags enabled."""
        flags = OutputCompileFlags(
            save_state=True,
            save_observables=False,
            summarise=True,
            summarise_observables=False,
            summarise_state=True,
            save_counters=False,
        )
        active = ActiveOutputs.from_compile_flags(flags)

        assert active.state is True
        assert active.observables is False
        assert active.state_summaries is True
        assert active.observable_summaries is False
        assert active.iteration_counters is False
        assert active.status_codes is True
