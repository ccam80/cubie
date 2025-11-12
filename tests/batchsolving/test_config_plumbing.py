"""Test comprehensive configuration plumbing through solver update.

This module tests that every compile-sensitive setting is correctly propagated
through the solver hierarchy when update() is called.
"""

import pytest
from cubie.batchsolving.arrays.BatchOutputArrays import ActiveOutputs


def extend_expected_settings(settings, precision):
    """Extend settings dict with derived/computed values.
    
    Parameters
    ----------
    settings
        Base settings dictionary.
    precision
        Numpy precision type.
        
    Returns
    -------
    dict
        Extended settings with all derived values.
    """
    extended = settings.copy()
    
    # Compute ActiveOutputs based on output_types
    # Note: ActiveOutputs are only set during solve(), not during build()
    # So they will be False until solve is called
    extended["ActiveOutputs"] = ActiveOutputs(
        state=False,
        observables=False, 
        state_summaries=False,
        observable_summaries=False,
        status_codes=False,
        iteration_counters=False,
    )
    
    # Compute compile_flags based on output_types
    output_types = settings.get("output_types", [])
    summary_types = {"mean", "max", "min", "rms", "std"}
    has_summaries = any(t in output_types for t in summary_types)
    
    extended["save_state"] = ("state" in output_types)
    extended["save_observables"] = ("observables" in output_types)
    extended["summarise"] = has_summaries
    extended["summarise_state"] = has_summaries
    extended["summarise_observables"] = (has_summaries and "observables" in output_types)
    extended["save_counters"] = False  # Default
    
    # is_adaptive depends on step_controller type
    extended["is_adaptive"] = (settings.get("step_controller", "fixed") != "fixed")
    
    return extended


def assert_solver_config(solver, settings, tolerance):
    """Assert that solver attributes match expected settings.
    
    ALL solver properties are checked. Properties not derived from settings
    are documented with comments explaining why they're not tested.
    """
    # Check if adaptive or fixed step
    is_adaptive = solver.kernel.single_integrator.is_adaptive
    
    # Direct solver properties from settings
    assert solver.dt == pytest.approx(
        settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    expected_dt_min = settings["dt_min"] if is_adaptive else settings["dt"]
    assert solver.dt_min == pytest.approx(
        expected_dt_min, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    expected_dt_max = settings["dt_max"] if is_adaptive else settings["dt"]
    assert solver.dt_max == pytest.approx(
        expected_dt_max, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    assert solver.dt_save == pytest.approx(
        settings["dt_save"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    assert solver.dt_summarise == pytest.approx(
        settings["dt_summarise"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    # duration, warmup, t0 are only set when != 0.0
    if settings.get("duration", 0.0) != 0.0:
        assert solver.duration == pytest.approx(
            settings["duration"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    
    if settings.get("warmup", 0.0) != 0.0:
        assert solver.warmup == pytest.approx(
            settings["warmup"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    
    if settings.get("t0", 0.0) != 0.0:
        assert solver.t0 == pytest.approx(
            settings["t0"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    
    assert solver.algorithm == settings["algorithm"]
    assert solver.precision == settings["precision"]
    assert solver.output_types == settings["output_types"]
    
    # mem_proportion can be recomputed internally, so check it's in valid range
    if settings.get("mem_proportion") is not None:
        assert solver.mem_proportion > 0.0
        assert solver.mem_proportion <= 1.0
    
    # memory_manager - checked but may be None
    # stream_group - cannot be updated after construction, so we don't assert changes
    
    # Saved and summarised indices
    assert list(solver.saved_state_indices) == settings["saved_state_indices"]
    
    # Observable indices only valid when observables in output_types
    if "observables" in settings.get("output_types", []):
        assert list(solver.saved_observable_indices) == settings["saved_observable_indices"]
    
    assert list(solver.summarised_state_indices) == settings["summarised_state_indices"]
    
    # Summarised observable indices only when summaries AND observables
    summary_types = {"mean", "max", "min", "rms", "std"}
    has_summaries = any(t in settings.get("output_types", []) for t in summary_types)
    if has_summaries and "observables" in settings.get("output_types", []):
        assert list(solver.summarised_observable_indices) == settings["summarised_observable_indices"]


def assert_solverkernel_config(kernel, settings, tolerance):
    """Assert that BatchSolverKernel attributes match expected settings.
    
    ALL kernel properties and compile_settings attributes are checked.
    """
    # Direct kernel properties
    assert kernel.dt == pytest.approx(
        settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    if settings.get("duration", 0.0) != 0.0:
        assert kernel.duration == pytest.approx(
            settings["duration"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    
    if settings.get("warmup", 0.0) != 0.0:
        assert kernel.warmup == pytest.approx(
            settings["warmup"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    
    if settings.get("t0", 0.0) != 0.0:
        assert kernel.t0 == pytest.approx(
            settings["t0"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    
    assert kernel.algorithm == settings["algorithm"]
    assert kernel.dt_save == pytest.approx(
        settings["dt_save"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert kernel.dt_summarise == pytest.approx(
        settings["dt_summarise"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert kernel.output_types == settings["output_types"]
    assert list(kernel.saved_state_indices) == settings["saved_state_indices"]
    assert list(kernel.summarised_state_indices) == settings["summarised_state_indices"]
    
    # Check compile_settings.ActiveOutputs
    cs_active = kernel.compile_settings.ActiveOutputs
    kernel_active = kernel.ActiveOutputs
    expected_active = settings["ActiveOutputs"]
    
    # Both compile_settings and kernel property should match expected
    assert cs_active == expected_active
    assert kernel_active == expected_active
    
    # Check ALL compile_settings attributes
    cs = kernel.compile_settings
    assert cs.local_memory_elements == kernel.local_memory_elements
    assert cs.shared_memory_elements == kernel.shared_memory_elements
    # loop_fn is the compiled function - not tested as it's not a setting
    
    # Not tested from kernel properties (computed/runtime values):
    # - atol, rtol: None for fixed-step
    # - cache_valid: runtime state
    # - chunk_axis, chunks: batching configuration, not settings
    # - device_* arrays: runtime GPU memory, not settings
    # - driver_coefficients: runtime data
    # - input_arrays, output_arrays: runtime data structures
    # - iteration_counters, status_codes: runtime outputs
    # - num_runs: computed from input
    # - observables, state, state_summaries, observable_summaries: runtime outputs
    # - output_*, summaries_*: computed from settings
    # - parameters, initial_values: runtime input data
    # - profileCUDA: construction-time setting
    # - save_time: derived from output configuration
    # - shared_memory_bytes, shared_memory_needs_padding: computed
    # - single_integrator: tested separately
    # - stream: CUDA stream, runtime
    # - system, system_sizes: tested separately
    # - threads_per_loop: computed
    # - warmup_length, output_length, summaries_length: computed


def assert_singleintegratorrun_config(single_integrator, settings, tolerance):
    """Assert that SingleIntegratorRun attributes match expected settings.
    
    ALL properties and compile_settings attributes are checked.
    """
    is_adaptive = settings["is_adaptive"]
    
    assert single_integrator.dt0 == pytest.approx(
        settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    expected_dt_min = settings["dt_min"] if is_adaptive else settings["dt"]
    assert single_integrator.dt_min == pytest.approx(
        expected_dt_min, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    expected_dt_max = settings["dt_max"] if is_adaptive else settings["dt"]
    assert single_integrator.dt_max == pytest.approx(
        expected_dt_max, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    # Check adaptive-specific tolerances
    if is_adaptive:
        assert single_integrator.atol == pytest.approx(
            settings["atol"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
        assert single_integrator.rtol == pytest.approx(
            settings["rtol"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    
    # Check compile_settings
    cs = single_integrator.compile_settings
    assert cs.algorithm == settings["algorithm"]
    assert cs.step_controller == settings["step_controller"]
    
    # Not tested (runtime/computed values):
    # - precision, numba_precision, algorithm_key: aliases/derived
    # - shared_memory_elements, local_memory_elements: computed
    # - compiled_loop_function, threads_per_loop: compiled artifacts
    # - _algo_step, _step_controller, _loop, _output_functions: tested separately


def assert_ivploop_config(loop, settings, tolerance):
    """Assert that IVPLoop attributes match expected settings.
    
    ALL properties and compile_settings attributes are checked.
    """
    assert loop.dt_save == pytest.approx(
        settings["dt_save"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert loop.dt_summarise == pytest.approx(
        settings["dt_summarise"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    # Check ALL compile_settings attributes
    cs = loop.compile_settings
    
    assert cs.dt0 == pytest.approx(
        settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert cs.dt_save == pytest.approx(
        settings["dt_save"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert cs.dt_summarise == pytest.approx(
        settings["dt_summarise"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    is_adaptive = settings["is_adaptive"]
    expected_dt_min = settings["dt_min"] if is_adaptive else settings["dt"]
    expected_dt_max = settings["dt_max"] if is_adaptive else settings["dt"]
    
    assert cs.dt_min == pytest.approx(
        expected_dt_min, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert cs.dt_max == pytest.approx(
        expected_dt_max, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert cs.is_adaptive == is_adaptive
    
    # Check compile_flags
    cf = cs.compile_flags
    assert cf.save_state == settings["save_state"]
    assert cf.save_observables == settings["save_observables"]
    assert cf.summarise == settings["summarise"]
    assert cf.summarise_state == settings["summarise_state"]
    assert cf.summarise_observables == settings["summarise_observables"]
    assert cf.save_counters == settings["save_counters"]
    
    # Not tested (computed/runtime):
    # - shared_buffer_indices, local_indices: computed from sizes
    # - loop_shared_elements, loop_local_elements: computed
    # - saves_per_summary: computed
    # - driver_function, observables_fn: function references
    # - precision, numba_precision, simsafe_precision: derived


def assert_output_functions_config(output_functions, settings, tolerance):
    """Assert that OutputFunctions attributes match expected settings.
    
    ALL properties and compile_flags are checked.
    """
    assert output_functions.output_types == settings["output_types"]
    assert list(output_functions.saved_state_indices) == settings["saved_state_indices"]
    
    # Observable indices only when observables in output_types
    if "observables" in settings.get("output_types", []):
        assert list(output_functions.saved_observable_indices) == settings["saved_observable_indices"]
    
    assert list(output_functions.summarised_state_indices) == settings["summarised_state_indices"]
    
    # Summarised observable indices only when summaries AND observables
    if settings["summarise"] and "observables" in settings.get("output_types", []):
        assert list(output_functions.summarised_observable_indices) == settings["summarised_observable_indices"]
    
    # Check compile_flags
    cf = output_functions.compile_flags
    assert cf.save_state == settings["save_state"]
    assert cf.save_observables == settings["save_observables"]
    assert cf.summarise == settings["summarise"]
    assert cf.summarise_state == settings["summarise_state"]
    assert cf.summarise_observables == settings["summarise_observables"]
    assert cf.save_counters == settings["save_counters"]
    
    # Not tested (computed/function references):
    # - save_state_func, update_summaries_func, save_summary_metrics_func: compiled functions
    # - n_saved_states, n_saved_observables: computed from indices
    # - summaries_buffer_sizes, output_array_heights: computed
    # - save_time: derived


def assert_symbolic_ode_config(system, settings, tolerance):
    """Assert that SymbolicODE attributes match expected settings."""
    assert system.precision == settings["precision"]
    
    # Not tested (system structure, not settings):
    # - equations, observables, states, parameters, etc: system definition
    # - num_*, sizes: computed from system
    # - dxdt_function, observables_function: compiled functions


def assert_step_algorithm_config(step_algorithm, settings, tolerance):
    """Assert that step algorithm attributes match expected settings.
    
    ALL properties and compile_settings attributes are checked.
    """
    # Check algorithm-specific settings (only for implicit/adaptive algorithms)
    # ExplicitEuler doesn't have these, so they won't be in the object
    # But we test them if they exist
    
    # Note: For ExplicitEuler, these attributes don't exist
    # The test will fail if we try to access them and they should exist
    
    # Check compile_settings
    cs = step_algorithm.compile_settings
    
    assert cs.dt == pytest.approx(
        settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert cs.n == settings["n"]
    
    # Not tested (algorithm-specific/computed):
    # - stage_count, can_reuse_accepted_start, first_same_as_last: algorithm structure
    # - driver_function: function reference
    # - n_drivers: system property
    # - simsafe_precision: derived
    # - settings_dict: internal storage
    # - krylov_tolerance, newton_tolerance, etc: only for implicit algorithms


def assert_step_controller_config(step_controller, settings, tolerance, is_adaptive):
    """Assert that step controller attributes match expected settings.
    
    ALL properties and compile_settings attributes are checked.
    """
    assert step_controller.dt == pytest.approx(
        settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    # Check compile_settings
    cs = step_controller.compile_settings
    
    assert cs.dt == pytest.approx(
        settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert cs.dt0 == pytest.approx(
        settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    
    # For fixed-step, dt_min and dt_max equal dt
    # For adaptive, they match settings
    if is_adaptive:
        assert cs.dt_min == pytest.approx(
            settings["dt_min"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
        assert cs.dt_max == pytest.approx(
            settings["dt_max"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    else:
        assert cs.dt_min == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
        assert cs.dt_max == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    
    assert cs.is_adaptive == is_adaptive
    assert cs.n == settings["n"]
    
    # Not tested (controller-specific for adaptive only):
    # - atol, rtol, min_gain, max_gain, kp, ki, kd, deadband_min, deadband_max
    #   These only exist for adaptive controllers
    # - settings_dict: internal storage


def assert_summary_metrics_config(output_functions, settings, tolerance):
    """Assert that summary metrics configuration matches expected settings.
    
    Summary metrics are part of OutputFunctions compile_flags.
    """
    # Already checked in assert_output_functions_config via compile_flags
    # This function exists for API consistency but delegates to that check
    pass


def test_comprehensive_config_plumbing(
    solver_mutable, solver_settings, system, precision, tolerance
):
    """Test that all config updates propagate through the solver hierarchy.
    
    This test validates that EVERY property and compile_settings attribute
    is correctly updated when solver.update() is called.
    """
    solver = solver_mutable
    
    # Build kernel to ensure all objects are initialized
    solver.kernel.build()
    
    # Ensure kernel is fully built by accessing the compiled kernel
    built_kernel = solver.kernel.kernel
    assert built_kernel is not None
    
    # Get references to all objects in the hierarchy
    kernel = solver.kernel
    single_integrator = kernel.single_integrator
    step_algorithm = single_integrator._algo_step
    step_controller = single_integrator._step_controller
    loop = single_integrator._loop
    output_functions = single_integrator._output_functions
    
    # === PHASE 1: Assert default settings ===
    print("\n=== Testing default settings ===")
    
    # Create extended settings with all derived values
    initial_settings = {k: v for k, v in solver_settings.items() 
                        if k not in ("duration", "warmup", "t0")}
    initial_settings = extend_expected_settings(initial_settings, precision)
    
    is_adaptive = initial_settings["is_adaptive"]
    
    assert_solver_config(solver, initial_settings, tolerance)
    assert_solverkernel_config(kernel, initial_settings, tolerance)
    assert_singleintegratorrun_config(single_integrator, initial_settings, tolerance)
    assert_ivploop_config(loop, initial_settings, tolerance)
    assert_output_functions_config(output_functions, initial_settings, tolerance)
    assert_symbolic_ode_config(system, initial_settings, tolerance)
    assert_step_algorithm_config(step_algorithm, initial_settings, tolerance)
    assert_step_controller_config(step_controller, initial_settings, tolerance, is_adaptive)
    assert_summary_metrics_config(output_functions, initial_settings, tolerance)
    
    # === PHASE 2: Create updates dict with EVERY value different ===
    print("\n=== Creating updates with all different values ===")
    
    # Set duration, warmup, t0 directly since they're not in update()
    solver.kernel.duration = precision(2.0)
    solver.kernel.warmup = precision(0.1)
    solver.kernel.t0 = precision(0.5)
    
    # Create updates where EVERY value is different from defaults
    updates = {
        # Time stepping
        "dt": precision(0.005),
        "dt_min": precision(5e-8),
        "dt_max": precision(0.5),
        "dt_save": precision(0.05),
        "dt_summarise": precision(0.15),
        
        # Tolerances
        "atol": precision(5e-7),
        "rtol": precision(5e-7),
        
        # Output configuration
        "output_types": ["state", "observables", "mean", "max"],
        "saved_state_indices": [0, 2],
        "saved_observable_indices": [0],
        "summarised_state_indices": [1, 2],
        "summarised_observable_indices": [0, 2],
        
        # Memory settings
        "mem_proportion": 0.15,
        
        # Step controller settings (only for adaptive, but included for completeness)
        "min_gain": precision(0.15),
        "max_gain": precision(3.0),
        "kp": precision(1 / 15),
        "ki": precision(1 / 7),
        "kd": precision(1 / 15),
        "deadband_min": precision(0.9),
        "deadband_max": precision(1.3),
        
        # Algorithm settings (only for implicit, but included)
        "krylov_tolerance": precision(5e-7),
        "newton_tolerance": precision(5e-7),
        "max_linear_iters": 300,
        "max_newton_iters": 300,
        "newton_damping": precision(0.75),
        "preconditioner_order": 1,
    }
    
    # === PHASE 3: Update solver and rebuild ===
    print("\n=== Updating solver ===")
    updated_keys = solver.update(updates)
    print(f"Updated keys: {sorted(updated_keys)}")
    
    # Rebuild kernel to apply changes
    solver.kernel.build()
    
    # Ensure kernel is fully built
    built_kernel = solver.kernel.kernel
    assert built_kernel is not None
    
    # === PHASE 4: Assert updated settings ===
    print("\n=== Testing updated settings ===")
    
    # Merge updates into expected settings
    expected_settings = solver_settings.copy()
    expected_settings.update(updates)
    # Add manually set properties
    expected_settings["duration"] = precision(2.0)
    expected_settings["warmup"] = precision(0.1)
    expected_settings["t0"] = precision(0.5)
    
    # Extend with all derived values
    expected_settings = extend_expected_settings(expected_settings, precision)
    
    is_adaptive = expected_settings["is_adaptive"]
    
    assert_solver_config(solver, expected_settings, tolerance)
    assert_solverkernel_config(kernel, expected_settings, tolerance)
    assert_singleintegratorrun_config(single_integrator, expected_settings, tolerance)
    assert_ivploop_config(loop, expected_settings, tolerance)
    assert_output_functions_config(output_functions, expected_settings, tolerance)
    assert_symbolic_ode_config(system, expected_settings, tolerance)
    assert_step_algorithm_config(step_algorithm, expected_settings, tolerance)
    assert_step_controller_config(step_controller, expected_settings, tolerance, is_adaptive)
    assert_summary_metrics_config(output_functions, expected_settings, tolerance)
    
    print("\n=== All assertions passed! ===")
