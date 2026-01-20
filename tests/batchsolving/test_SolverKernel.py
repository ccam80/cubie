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
    """Big plumbing integration check - check that config classes match exactly
    between an updated solver and one instantiated with the update settings."""
    solverkernel = solverkernel_mutable

    # Limit indices to actual system sizes to prevent IndexError
    n_states = system.sizes.states
    n_obs = system.sizes.observables

    saved_state_idx = list(range(min(3, n_states)))
    saved_obs_idx = list(range(min(3, n_obs)))
    summarised_state_idx = [0] if n_states > 0 else []
    summarised_obs_idx = [0] if n_obs > 0 else []

    new_settings = {
        # "duration": 1.0,
        "dt_min": 0.0001,
        "dt_max": 0.01,
        "save_every": 0.01,
        "summarise_every": 0.1,
        "sample_summaries_every": 0.05,
        "atol": 1e-2,
        "rtol": 1e-1,
        "saved_state_indices": saved_state_idx,
        "saved_observable_indices": saved_obs_idx,
        "summarised_state_indices": summarised_state_idx,
        "summarised_observable_indices": summarised_obs_idx,
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
        "saved_state_indices": np.asarray(saved_state_idx),
        "saved_observable_indices": np.asarray(saved_obs_idx),
        "summarised_state_indices": np.asarray(summarised_state_idx),
        "summarised_observable_indices": np.asarray(summarised_obs_idx),
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
    inits = np.ones((n_states, 1), dtype=precision)
    params = np.ones((system.sizes.parameters, 1), dtype=precision)
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
    # Note: compile_settings.local_memory_elements was removed in refactoring
    # Memory elements are now accessed via properties that query buffer_registry
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
        self, system, precision, driver_array, solver, driver_settings
    ):
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        with pytest.raises(
            ValueError, match=r"save_every.*>.*duration.*no outputs"
        ):
            solver.solve(
                inits,
                params,
                driver_settings,
                save_every=1.0,
                duration=0.5,
            )

    def test_save_every_greater_than_duration_with_save_last_succeeds(
        self, system, precision, driver_array, solver, driver_settings
    ):
        """Test that save_every >= duration with save_last=True is valid."""
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        # Should not raise when save_last is True (default when save_every=None)
        solver.solve(
            inits,
            params,
            drivers=driver_settings,
            save_every=None,
            duration=0.05,
            dt=0.02,
        )

    def test_summarise_every_greater_than_duration_raises(
        self, system, precision, driver_array, solver, driver_settings
    ):
        """Test that summarise_every > duration raises."""
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        with pytest.raises(
            ValueError,
            match=r"summarise_every.*>.*duration.*no summary outputs",
        ):
            solver.solve(
                inits,
                params,
                driver_settings,
                summarise_every=0.6,
                duration=0.5,
            )

    def test_sample_summaries_every_gte_summarise_every_raises(
        self, system, precision, driver_array, solver, driver_settings
    ):
        """Test that sample_summaries_every >= summarise_every raises."""
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        with pytest.raises(
            ValueError, match=r"sample_summaries_every.*>=.*summarise_every"
        ):
            solver.solve(
                inits,
                params,
                drivers=driver_settings,
                summarise_every=0.01,
                sample_summaries_every=0.01,
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


class TestRunParamsIntegration:
    """Tests for RunParams integration into BatchSolverKernel."""

    def test_runparams_initialized_on_construction(self, solverkernel_mutable):
        """Verify BatchSolverKernel initializes run_params with defaults."""
        solverkernel = solverkernel_mutable  # A used solverkernel might be
        # updated

        assert hasattr(solverkernel, "run_params")
        assert solverkernel.run_params.duration == 0.0
        assert solverkernel.run_params.warmup == 0.0
        assert solverkernel.run_params.t0 == 0.0
        assert solverkernel.run_params.runs == 1
        assert solverkernel.run_params.num_chunks == 1
        assert solverkernel.run_params.chunk_length == 0


def test_batch_solver_kernel_init_without_memory_elements(solverkernel):
    """Verify BatchSolverKernel initializes without memory elements in config.
    
    After the refactoring in Task Group 2, BatchSolverKernel.__init__ should
    no longer pass local_memory_elements or shared_memory_elements to
    BatchSolverConfig during initialization. This test verifies that the
    kernel initializes successfully without these parameters.
    """
    # Verify that the solverkernel was initialized successfully
    assert solverkernel is not None
    
    # Verify that compile_settings exists and has the expected fields
    assert hasattr(solverkernel, 'compile_settings')
    assert solverkernel.compile_settings is not None
    
    # BatchSolverConfig should have precision, loop_fn, and compile_flags,
    # but NOT local_memory_elements or shared_memory_elements
    config = solverkernel.compile_settings
    assert hasattr(config, 'precision')
    assert hasattr(config, 'loop_fn')
    assert hasattr(config, 'compile_flags')
    
    # These fields should NOT exist in the config anymore
    assert not hasattr(config, 'local_memory_elements'), (
        "local_memory_elements should not exist in BatchSolverConfig"
    )
    assert not hasattr(config, 'shared_memory_elements'), (
        "shared_memory_elements should not exist in BatchSolverConfig"
    )
    
    # Verify that the kernel still has access to memory element counts
    # through its properties (which will query buffer_registry in Task Group 3)
    assert hasattr(solverkernel, 'local_memory_elements')
    assert hasattr(solverkernel, 'shared_memory_elements')
    assert hasattr(solverkernel, 'shared_memory_bytes')
    
    # Properties should return integer values (even if 0)
    local_elems = solverkernel.local_memory_elements
    shared_elems = solverkernel.shared_memory_elements
    shared_bytes = solverkernel.shared_memory_bytes
    
    assert isinstance(local_elems, int), (
        "local_memory_elements property should return int"
    )
    assert isinstance(shared_elems, int), (
        "shared_memory_elements property should return int"
    )
    assert isinstance(shared_bytes, int), (
        "shared_memory_bytes property should return int"
    )
    assert local_elems >= 0
    assert shared_elems >= 0
    assert shared_bytes >= 0


def test_batch_solver_kernel_properties_query_buffer_registry(solverkernel):
    """Verify kernel properties query buffer_registry correctly.
    
    After Task Group 3 refactoring, the memory element properties should
    query buffer_registry instead of reading from compile_settings. This
    test verifies:
    1. local_memory_elements returns buffer_registry.persistent_local_buffer_size()
    2. shared_memory_elements returns buffer_registry.shared_buffer_size()
    3. shared_memory_bytes is computed from shared_memory_elements * itemsize
    """
    from cubie.buffer_registry import buffer_registry
    
    # Get the loop object that owns the buffer registrations
    loop = solverkernel.single_integrator._loop
    
    # Verify local_memory_elements queries buffer_registry
    local_from_property = solverkernel.local_memory_elements
    local_from_registry = buffer_registry.persistent_local_buffer_size(loop)
    assert local_from_property == local_from_registry, (
        f"local_memory_elements property returned {local_from_property}, "
        f"but buffer_registry reports {local_from_registry}"
    )
    
    # Verify shared_memory_elements queries buffer_registry
    shared_from_property = solverkernel.shared_memory_elements
    shared_from_registry = buffer_registry.shared_buffer_size(loop)
    assert shared_from_property == shared_from_registry, (
        f"shared_memory_elements property returned {shared_from_property}, "
        f"but buffer_registry reports {shared_from_registry}"
    )
    
    # Verify shared_memory_bytes is computed correctly from shared_memory_elements
    shared_bytes = solverkernel.shared_memory_bytes
    precision = solverkernel.precision
    itemsize = np.dtype(precision).itemsize
    expected_bytes = shared_from_property * itemsize
    assert shared_bytes == expected_bytes, (
        f"shared_memory_bytes returned {shared_bytes}, "
        f"but expected {shared_from_property} * {itemsize} = {expected_bytes}"
    )
    
    # Verify all values are non-negative integers
    assert isinstance(local_from_property, int)
    assert isinstance(shared_from_property, int)
    assert isinstance(shared_bytes, int)
    assert local_from_property >= 0
    assert shared_from_property >= 0
    assert shared_bytes >= 0
    
    # Verify bytes computation makes sense (if shared elements > 0, bytes > 0)
    if shared_from_property > 0:
        assert shared_bytes > 0, (
            "shared_memory_bytes should be > 0 when shared_memory_elements > 0"
        )
    else:
        assert shared_bytes == 0, (
            "shared_memory_bytes should be 0 when shared_memory_elements is 0"
        )


def test_batch_solver_kernel_run_updates_without_memory_elements(
    solverkernel_mutable, system, precision, driver_array
):
    """Verify run method updates compile settings without memory elements.
    
    After Task Group 4 refactoring, the BatchSolverKernel.run() method should
    no longer attempt to update local_memory_elements or shared_memory_elements
    in compile_settings. This test verifies that:
    1. run() can be called without errors
    2. compile_settings are updated with loop_fn and precision only
    3. Memory element properties remain accessible via buffer_registry queries
    """
    solverkernel = solverkernel_mutable
    
    # Prepare test inputs
    n_states = system.sizes.states
    n_params = system.sizes.parameters
    inits = np.ones((n_states, 1), dtype=precision)
    params = np.ones((n_params, 1), dtype=precision)
    driver_coefficients = driver_array.coefficients
    
    # Capture initial config state
    initial_loop_fn = solverkernel.compile_settings.loop_fn
    initial_precision = solverkernel.compile_settings.precision
    
    # Run the kernel - this should update compile_settings without errors
    solverkernel.run(
        inits=inits,
        params=params,
        driver_coefficients=driver_coefficients,
        duration=0.1,
        warmup=0.0,
        t0=0.0,
    )
    
    # Verify compile_settings were updated
    updated_config = solverkernel.compile_settings
    assert updated_config is not None
    
    # Verify loop_fn and precision are updated/present
    assert hasattr(updated_config, 'loop_fn')
    assert hasattr(updated_config, 'precision')
    assert updated_config.precision == solverkernel.single_integrator.precision
    
    # Verify that memory elements are NOT in compile_settings
    assert not hasattr(updated_config, 'local_memory_elements'), (
        "local_memory_elements should not be in compile_settings after run()"
    )
    assert not hasattr(updated_config, 'shared_memory_elements'), (
        "shared_memory_elements should not be in compile_settings after run()"
    )
    
    # Verify memory element properties still work (querying buffer_registry)
    local_elems = solverkernel.local_memory_elements
    shared_elems = solverkernel.shared_memory_elements
    shared_bytes = solverkernel.shared_memory_bytes
    
    assert isinstance(local_elems, int), (
        "local_memory_elements property should return int"
    )
    assert isinstance(shared_elems, int), (
        "shared_memory_elements property should return int"
    )
    assert isinstance(shared_bytes, int), (
        "shared_memory_bytes property should return int"
    )
    assert local_elems >= 0
    assert shared_elems >= 0
    assert shared_bytes >= 0
    
    # Verify kernel execution was successful by checking outputs exist
    assert solverkernel.status_codes is not None
    assert solverkernel.state is not None


def test_batch_solver_kernel_build_uses_current_buffer_sizes(
    solverkernel_mutable, system, precision, driver_array
):
    """Verify build_kernel uses current buffer_registry state via properties.
    
    After Task Group 5 refactoring, build_kernel should query the
    shared_memory_elements property (which queries buffer_registry) instead
    of reading from config.shared_memory_elements. This test verifies that:
    1. build_kernel can be called successfully
    2. The kernel uses the current buffer_registry state for shared memory size
    3. Kernel compilation succeeds with buffer_registry-derived sizes
    """
    from cubie.buffer_registry import buffer_registry
    
    solverkernel = solverkernel_mutable
    
    # Prepare test inputs for run (which triggers build if needed)
    n_states = system.sizes.states
    n_params = system.sizes.parameters
    inits = np.ones((n_states, 1), dtype=precision)
    params = np.ones((n_params, 1), dtype=precision)
    driver_coefficients = driver_array.coefficients
    
    # Get shared memory size from buffer_registry before building
    loop = solverkernel.single_integrator._loop
    shared_size_from_registry = buffer_registry.shared_buffer_size(loop)
    shared_size_from_property = solverkernel.shared_memory_elements
    
    # These should match since the property queries buffer_registry
    assert shared_size_from_property == shared_size_from_registry, (
        f"Property returned {shared_size_from_property}, "
        f"but buffer_registry reports {shared_size_from_registry}"
    )
    
    # Trigger build_kernel by calling run
    # build_kernel should use self.shared_memory_elements (querying registry)
    # rather than config.shared_memory_elements (which no longer exists)
    solverkernel.run(
        inits=inits,
        params=params,
        driver_coefficients=driver_coefficients,
        duration=0.05,
        warmup=0.0,
        t0=0.0,
    )
    
    # Verify the kernel was built successfully
    assert solverkernel.kernel is not None, (
        "Kernel should be built after run() call"
    )
    
    # Verify the kernel executed successfully
    assert solverkernel.status_codes is not None
    assert solverkernel.state is not None
    
    # Verify shared memory bytes calculation is consistent
    # The kernel should have used shared_memory_elements from property/registry
    shared_bytes = solverkernel.shared_memory_bytes
    itemsize = np.dtype(precision).itemsize
    expected_bytes = shared_size_from_property * itemsize
    
    assert shared_bytes == expected_bytes, (
        f"shared_memory_bytes is {shared_bytes}, "
        f"but expected {shared_size_from_property} * {itemsize} = "
        f"{expected_bytes}"
    )
    
    # Verify that buffer_registry is the source of truth for memory sizes
    # by checking the property still matches registry after build
    post_build_property = solverkernel.shared_memory_elements
    post_build_registry = buffer_registry.shared_buffer_size(loop)
    
    assert post_build_property == post_build_registry, (
        f"After build, property returned {post_build_property}, "
        f"but buffer_registry reports {post_build_registry}"
    )
    
    # Both should match the pre-build value (registry state unchanged)
    assert post_build_property == shared_size_from_property, (
        "shared_memory_elements should be consistent before and after build"
    )


def test_batch_solver_kernel_update_recognizes_buffer_locations(
    solverkernel_mutable, system
):
    """Verify update method recognizes buffer location parameters.
    
    After Task Group 6 refactoring, BatchSolverKernel.update() should
    delegate buffer location parameter recognition to buffer_registry.update().
    This test verifies that:
    1. update() recognizes buffer location parameters (e.g., 'state_location')
    2. buffer_registry.update() is called and returns recognized parameters
    3. Unrecognized parameters still raise KeyError when silent=False
    4. Memory element parameters are NOT in compile_settings updates
    """
    from cubie.buffer_registry import buffer_registry
    
    solverkernel = solverkernel_mutable
    loop = solverkernel.single_integrator._loop
    
    # Get the list of registered buffers for this loop
    # This allows us to construct valid location parameter names
    registered_buffers = set()
    if loop in buffer_registry._groups:
        # BufferGroup.entries is a Dict[str, CUDABuffer]
        for buffer_name in buffer_registry._groups[loop].entries.keys():
            registered_buffers.add(buffer_name)
    
    # Test 1: Verify that valid buffer location parameters are recognized
    if registered_buffers:
        # Pick the first registered buffer to test with
        test_buffer = list(registered_buffers)[0]
        location_param = f"{test_buffer}_location"
        
        # Update with a valid location parameter - should be recognized
        recognized = solverkernel.update(
            {location_param: 'shared'}, silent=True
        )
        
        # The location parameter should be recognized (in the returned set)
        assert location_param in recognized, (
            f"Location parameter '{location_param}' should be recognized "
            f"by buffer_registry.update()"
        )
    
    # Test 2: Verify that invalid buffer location values raise ValueError
    if registered_buffers:
        test_buffer = list(registered_buffers)[0]
        location_param = f"{test_buffer}_location"
        
        # Invalid location value should raise ValueError from attrs validator
        # The attrs in_() validator raises ValueError with message like:
        # "'location' must be in ['shared', 'local'] (got 'invalid_location')"
        with pytest.raises(
            ValueError, 
            match=r"must be in.*\['shared', 'local'\].*got 'invalid_location'"
        ):
            solverkernel.update({location_param: 'invalid_location'})
    
    # Test 3: Verify that completely bogus parameters are not recognized
    bogus_update = {'completely_bogus_parameter': 42}
    recognized = solverkernel.update(bogus_update, silent=True)
    
    assert 'completely_bogus_parameter' not in recognized, (
        "Bogus parameter should not be recognized"
    )
    
    # With silent=False, bogus parameter should raise KeyError
    with pytest.raises(KeyError, match="Unrecognized parameters"):
        solverkernel.update({'another_bogus_param': 'value'}, silent=False)
    
    # Test 4: Verify that compile_settings no longer tracks memory elements
    # (they were removed in Task Group 1)
    config = solverkernel.compile_settings
    assert not hasattr(config, 'local_memory_elements'), (
        "local_memory_elements should not exist in BatchSolverConfig"
    )
    assert not hasattr(config, 'shared_memory_elements'), (
        "shared_memory_elements should not exist in BatchSolverConfig"
    )
    
    # Test 5: Verify that update() still updates loop_fn and compile_flags
    # even when location parameters are present
    if registered_buffers:
        test_buffer = list(registered_buffers)[0]
        location_param = f"{test_buffer}_location"
        
        # Capture initial state
        initial_loop_fn = solverkernel.compile_settings.loop_fn
        
        # Update with both location parameter and other valid parameters
        # dt0 should be recognized by single_integrator.update()
        recognized = solverkernel.update(
            {location_param: 'local', 'dt0': 0.002}, silent=True
        )
        
        # Both parameters should be recognized
        assert location_param in recognized, (
            "Location parameter should be recognized"
        )
        assert 'dt0' in recognized, (
            "dt0 should be recognized by single_integrator.update()"
        )
        
        # compile_settings should still have loop_fn and compile_flags
        assert hasattr(solverkernel.compile_settings, 'loop_fn')
        assert hasattr(solverkernel.compile_settings, 'compile_flags')
    
    # Test 6: Verify memory element properties still work (via buffer_registry)
    local_elems = solverkernel.local_memory_elements
    shared_elems = solverkernel.shared_memory_elements
    shared_bytes = solverkernel.shared_memory_bytes
    
    assert isinstance(local_elems, int)
    assert isinstance(shared_elems, int)
    assert isinstance(shared_bytes, int)
    assert local_elems >= 0
    assert shared_elems >= 0
    assert shared_bytes >= 0
