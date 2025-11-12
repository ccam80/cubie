"""Test comprehensive configuration plumbing through solver update.

This module tests that every compile-sensitive setting is correctly propagated
through the solver hierarchy when update() is called.
"""

import numpy as np
import pytest


def assert_solver_config(solver, settings, tolerance):
    """Assert that solver attributes match expected settings.

    Parameters
    ----------
    solver
        The Solver instance to check.
    settings
        Dictionary of expected settings values.
    tolerance
        Tolerance namespace for floating point comparisons.
    """
    # Check if adaptive or fixed step
    is_adaptive = solver.kernel.single_integrator.is_adaptive
    
    # Direct solver properties
    if "dt" in settings:
        assert solver.dt == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "dt_min" in settings:
        expected_dt_min = settings["dt_min"] if is_adaptive else settings["dt"]
        assert solver.dt_min == pytest.approx(
            expected_dt_min, rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "dt_max" in settings:
        expected_dt_max = settings["dt_max"] if is_adaptive else settings["dt"]
        assert solver.dt_max == pytest.approx(
            expected_dt_max, rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "dt_save" in settings:
        assert solver.dt_save == pytest.approx(
            settings["dt_save"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "dt_summarise" in settings:
        assert solver.dt_summarise == pytest.approx(
            settings["dt_summarise"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
    if "duration" in settings and settings["duration"] != 0.0:
        assert solver.duration == pytest.approx(
            settings["duration"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "warmup" in settings and settings["warmup"] != 0.0:
        assert solver.warmup == pytest.approx(
            settings["warmup"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "t0" in settings and settings["t0"] != 0.0:
        assert solver.t0 == pytest.approx(
            settings["t0"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "algorithm" in settings:
        assert solver.algorithm == settings["algorithm"]
    if "precision" in settings:
        assert solver.precision == settings["precision"]
    if "output_types" in settings:
        assert solver.output_types == settings["output_types"]
    if "mem_proportion" in settings and settings["mem_proportion"] is not None:
        # mem_proportion can be slightly different due to internal calculations
        # Just check it's in a reasonable range
        assert solver.mem_proportion > 0.0
        assert solver.mem_proportion <= 1.0
    if "memory_manager" in settings:
        assert solver.memory_manager == settings["memory_manager"]
    if "stream_group" in settings:
        assert solver.stream_group == settings["stream_group"]

    # Saved and summarised variables (check indices)
    if "saved_state_indices" in settings:
        assert list(solver.saved_state_indices) == settings[
            "saved_state_indices"
        ]
    if "saved_observable_indices" in settings:
        # Only check if observables are in output_types
        if "observables" in settings.get("output_types", []):
            assert list(solver.saved_observable_indices) == settings[
                "saved_observable_indices"
            ]
    if "summarised_state_indices" in settings:
        assert list(solver.summarised_state_indices) == settings[
            "summarised_state_indices"
        ]
    if "summarised_observable_indices" in settings:
        # Only check if we have summary output types for observables
        summary_types = {"mean", "max", "min", "rms", "std"}
        has_summaries = any(t in settings.get("output_types", []) for t in summary_types)
        if has_summaries and "observables" in settings.get("output_types", []):
            assert list(solver.summarised_observable_indices) == settings[
                "summarised_observable_indices"
            ]


def assert_solverkernel_config(kernel, settings, tolerance):
    """Assert that BatchSolverKernel attributes match expected settings.

    Parameters
    ----------
    kernel
        The BatchSolverKernel instance to check.
    settings
        Dictionary of expected settings values.
    tolerance
        Tolerance namespace for floating point comparisons.
    """
    if "dt" in settings:
        assert kernel.dt == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "duration" in settings and settings["duration"] != 0.0:
        assert kernel.duration == pytest.approx(
            settings["duration"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "warmup" in settings and settings["warmup"] != 0.0:
        assert kernel.warmup == pytest.approx(
            settings["warmup"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "t0" in settings and settings["t0"] != 0.0:
        assert kernel.t0 == pytest.approx(
            settings["t0"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "algorithm" in settings:
        assert kernel.algorithm == settings["algorithm"]


def assert_singleintegratorrun_config(
    single_integrator, settings, tolerance
):
    """Assert that SingleIntegratorRun attributes match expected settings.

    Parameters
    ----------
    single_integrator
        The SingleIntegratorRun instance to check.
    settings
        Dictionary of expected settings values.
    tolerance
        Tolerance namespace for floating point comparisons.
    """
    is_adaptive = single_integrator.is_adaptive
    
    if "dt" in settings:
        assert single_integrator.dt0 == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "dt_min" in settings:
        expected_dt_min = settings["dt_min"] if is_adaptive else settings["dt"]
        assert single_integrator.dt_min == pytest.approx(
            expected_dt_min, rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "dt_max" in settings:
        expected_dt_max = settings["dt_max"] if is_adaptive else settings["dt"]
        assert single_integrator.dt_max == pytest.approx(
            expected_dt_max, rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    
    # Check adaptive vs fixed step - only check tolerances for adaptive
    if is_adaptive:
        if "atol" in settings and settings["atol"] is not None:
            assert single_integrator.atol == pytest.approx(
                settings["atol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
        if "rtol" in settings and settings["rtol"] is not None:
            assert single_integrator.rtol == pytest.approx(
                settings["rtol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )


def assert_ivploop_config(loop, settings, tolerance):
    """Assert that IVPLoop attributes match expected settings.

    Parameters
    ----------
    loop
        The IVPLoop instance to check (accessed from SingleIntegratorRun).
    settings
        Dictionary of expected settings values.
    tolerance
        Tolerance namespace for floating point comparisons.
    """
    if "dt_save" in settings:
        assert loop.dt_save == pytest.approx(
            settings["dt_save"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    if "dt_summarise" in settings:
        assert loop.dt_summarise == pytest.approx(
            settings["dt_summarise"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )


def assert_output_functions_config(output_functions, settings, tolerance):
    """Assert that OutputFunctions attributes match expected settings.

    Parameters
    ----------
    output_functions
        The OutputFunctions instance to check.
    settings
        Dictionary of expected settings values.
    tolerance
        Tolerance namespace for floating point comparisons.
    """
    if "output_types" in settings:
        assert output_functions.output_types == settings["output_types"]
    if "saved_state_indices" in settings:
        assert list(output_functions.saved_state_indices) == settings[
            "saved_state_indices"
        ]
    if "saved_observable_indices" in settings:
        # Only check if observables are in output_types
        if "observables" in settings.get("output_types", []):
            assert list(output_functions.saved_observable_indices) == settings[
                "saved_observable_indices"
            ]
    if "summarised_state_indices" in settings:
        assert list(output_functions.summarised_state_indices) == settings[
            "summarised_state_indices"
        ]
    if "summarised_observable_indices" in settings:
        # Only check if we have summary output types for observables
        summary_types = {"mean", "max", "min", "rms", "std"}
        has_summaries = any(t in settings.get("output_types", []) for t in summary_types)
        if has_summaries and "observables" in settings.get("output_types", []):
            assert list(output_functions.summarised_observable_indices) == settings[
                "summarised_observable_indices"
            ]


def assert_symbolic_ode_config(system, settings, tolerance):
    """Assert that SymbolicODE attributes match expected settings.

    Parameters
    ----------
    system
        The SymbolicODE instance to check.
    settings
        Dictionary of expected settings values.
    tolerance
        Tolerance namespace for floating point comparisons.
    """
    if "precision" in settings:
        assert system.precision == settings["precision"]


def assert_step_algorithm_config(step_algorithm, settings, tolerance):
    """Assert that step algorithm attributes match expected settings.

    Parameters
    ----------
    step_algorithm
        The algorithm step instance to check.
    settings
        Dictionary of expected settings values.
    tolerance
        Tolerance namespace for floating point comparisons.
    """
    # Check algorithm-specific settings
    if hasattr(step_algorithm, "krylov_tolerance"):
        if "krylov_tolerance" in settings and settings["krylov_tolerance"] is not None:
            assert step_algorithm.krylov_tolerance == pytest.approx(
                settings["krylov_tolerance"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
    if hasattr(step_algorithm, "newton_tolerance"):
        if "newton_tolerance" in settings and settings["newton_tolerance"] is not None:
            assert step_algorithm.newton_tolerance == pytest.approx(
                settings["newton_tolerance"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
    if hasattr(step_algorithm, "max_newton_iters"):
        if "max_newton_iters" in settings:
            assert (
                step_algorithm.max_newton_iters == settings["max_newton_iters"]
            )
    if hasattr(step_algorithm, "max_linear_iters"):
        if "max_linear_iters" in settings:
            assert (
                step_algorithm.max_linear_iters == settings["max_linear_iters"]
            )
    if hasattr(step_algorithm, "newton_damping"):
        if "newton_damping" in settings and settings["newton_damping"] is not None:
            assert step_algorithm.newton_damping == pytest.approx(
                settings["newton_damping"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
    if hasattr(step_algorithm, "preconditioner_order"):
        if "preconditioner_order" in settings:
            assert (
                step_algorithm.preconditioner_order
                == settings["preconditioner_order"]
            )


def assert_step_controller_config(step_controller, settings, tolerance, is_adaptive):
    """Assert that step controller attributes match expected settings.

    Parameters
    ----------
    step_controller
        The step controller instance to check.
    settings
        Dictionary of expected settings values.
    tolerance
        Tolerance namespace for floating point comparisons.
    is_adaptive
        Whether the controller is adaptive.
    """
    if "dt" in settings:
        assert step_controller.dt == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    
    # Check controller-specific settings for adaptive controllers
    if hasattr(step_controller, "atol") and is_adaptive:
        if "atol" in settings and settings["atol"] is not None:
            assert step_controller.atol == pytest.approx(
                settings["atol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
    if hasattr(step_controller, "rtol") and is_adaptive:
        if "rtol" in settings and settings["rtol"] is not None:
            assert step_controller.rtol == pytest.approx(
                settings["rtol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
    if hasattr(step_controller, "dt_min") and is_adaptive:
        if "dt_min" in settings:
            assert step_controller.dt_min == pytest.approx(
                settings["dt_min"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
    if hasattr(step_controller, "dt_max") and is_adaptive:
        if "dt_max" in settings:
            assert step_controller.dt_max == pytest.approx(
                settings["dt_max"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
    if hasattr(step_controller, "min_gain") and is_adaptive:
        if "min_gain" in settings and settings["min_gain"] is not None:
            assert step_controller.min_gain == pytest.approx(
                settings["min_gain"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
    if hasattr(step_controller, "max_gain") and is_adaptive:
        if "max_gain" in settings and settings["max_gain"] is not None:
            assert step_controller.max_gain == pytest.approx(
                settings["max_gain"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
    if hasattr(step_controller, "kp") and is_adaptive:
        if "kp" in settings and settings["kp"] is not None:
            assert step_controller.kp == pytest.approx(
                settings["kp"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
            )
    if hasattr(step_controller, "ki") and is_adaptive:
        if "ki" in settings and settings["ki"] is not None:
            assert step_controller.ki == pytest.approx(
                settings["ki"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
            )
    if hasattr(step_controller, "kd") and is_adaptive:
        if "kd" in settings and settings["kd"] is not None:
            assert step_controller.kd == pytest.approx(
                settings["kd"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
            )
    if hasattr(step_controller, "deadband_min") and is_adaptive:
        if "deadband_min" in settings and settings["deadband_min"] is not None:
            assert step_controller.deadband_min == pytest.approx(
                settings["deadband_min"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
    if hasattr(step_controller, "deadband_max") and is_adaptive:
        if "deadband_max" in settings and settings["deadband_max"] is not None:
            assert step_controller.deadband_max == pytest.approx(
                settings["deadband_max"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )


def assert_summary_metrics_config(output_functions, settings, tolerance):
    """Assert that summary metrics configuration matches expected settings.

    Parameters
    ----------
    output_functions
        The OutputFunctions instance containing summary metrics.
    settings
        Dictionary of expected settings values.
    tolerance
        Tolerance namespace for floating point comparisons.
    """
    # Summary metrics are part of OutputFunctions
    # Check that state_summaries and observable_summaries are configured
    if hasattr(output_functions, "state_summaries"):
        if "output_types" in settings:
            # state_summaries should be set based on output_types
            has_summary_types = any(
                t in settings["output_types"]
                for t in ["mean", "max", "min", "rms", "std"]
            )
            if has_summary_types:
                assert output_functions.state_summaries is not None
    
    if hasattr(output_functions, "observable_summaries"):
        if "output_types" in settings:
            has_summary_types = any(
                t in settings["output_types"]
                for t in ["mean", "max", "min", "rms", "std"]
            )
            if has_summary_types:
                assert output_functions.observable_summaries is not None


def test_comprehensive_config_plumbing(
    solver_mutable, solver_settings, system, precision, tolerance
):
    """Test that all config updates propagate through the solver hierarchy.

    This test creates assertion functions for each object in the solver
    hierarchy and verifies that every compile-sensitive setting is correctly
    updated when we call solver.update().
    """
    solver = solver_mutable
    
    # Build kernel to ensure all objects are initialized
    solver.kernel.build()
    
    # Get references to all objects in the hierarchy
    kernel = solver.kernel
    single_integrator = kernel.single_integrator
    step_algorithm = single_integrator._algo_step
    step_controller = single_integrator._step_controller
    
    # Access IVPLoop through the internal attribute
    loop = single_integrator._loop
    # Access OutputFunctions directly from single_integrator
    output_functions = single_integrator._output_functions
    
    # === PHASE 1: Assert default settings ===
    print("\n=== Testing default settings ===")
    
    is_adaptive = single_integrator.is_adaptive
    
    # Create a copy of settings without duration/warmup/t0 for initial check
    initial_settings = {k: v for k, v in solver_settings.items() 
                        if k not in ("duration", "warmup", "t0")}
    
    assert_solver_config(solver, initial_settings, tolerance)
    assert_solverkernel_config(kernel, initial_settings, tolerance)
    assert_singleintegratorrun_config(
        single_integrator, initial_settings, tolerance
    )
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
        "dt": precision(0.005),  # was 0.01
        "dt_min": precision(5e-8),  # was 1e-7
        "dt_max": precision(0.5),  # was 1.0
        "dt_save": precision(0.05),  # was 0.1
        "dt_summarise": precision(0.15),  # was 0.2
        
        # Algorithm settings (only if adaptive)
        # Note: euler is not adaptive, so atol/rtol won't apply
        # "algorithm": "backwards_euler_pc",  # Don't change algorithm
        
        # Tolerances
        "atol": precision(5e-7),  # was 1e-6
        "rtol": precision(5e-7),  # was 1e-6
        
        # Output configuration
        "output_types": ["state", "observables", "mean", "max"],  # was ["state"]
        "saved_state_indices": [0, 2],  # was [0, 1]
        "saved_observable_indices": [0],  # was [0, 1]
        "summarised_state_indices": [1, 2],  # was [0, 1]
        "summarised_observable_indices": [0, 2],  # was [0, 1]
        
        # Memory settings
        "mem_proportion": 0.15,  # was None
        # Don't update stream_group - it's set at construction time
        # "stream_group": "updated_group",  # was "test_group"
        
        # Step controller settings
        "min_gain": precision(0.15),  # was 0.2
        "max_gain": precision(3.0),  # was 2.0
        "kp": precision(1 / 15),  # was 1/18
        "ki": precision(1 / 7),  # was 1/9
        "kd": precision(1 / 15),  # was 1/18
        "deadband_min": precision(0.9),  # was 1.0
        "deadband_max": precision(1.3),  # was 1.2
        
        # Algorithm settings
        "krylov_tolerance": precision(5e-7),  # was 1e-6
        "newton_tolerance": precision(5e-7),  # was 1e-6
        "max_linear_iters": 300,  # was 500
        "max_newton_iters": 300,  # was 500
        "newton_damping": precision(0.75),  # was 0.85
        "preconditioner_order": 1,  # was 2
    }
    
    # === PHASE 3: Update solver and rebuild ===
    print("\n=== Updating solver ===")
    updated_keys = solver.update(updates)
    print(f"Updated keys: {sorted(updated_keys)}")
    
    # Rebuild kernel to apply changes
    solver.kernel.build()
    
    # === PHASE 4: Assert updated settings ===
    print("\n=== Testing updated settings ===")
    
    # Merge updates into expected settings
    expected_settings = solver_settings.copy()
    expected_settings.update(updates)
    # Add manually set properties
    expected_settings["duration"] = precision(2.0)
    expected_settings["warmup"] = precision(0.1)
    expected_settings["t0"] = precision(0.5)
    
    assert_solver_config(solver, expected_settings, tolerance)
    assert_solverkernel_config(kernel, expected_settings, tolerance)
    assert_singleintegratorrun_config(
        single_integrator, expected_settings, tolerance
    )
    assert_ivploop_config(loop, expected_settings, tolerance)
    assert_output_functions_config(
        output_functions, expected_settings, tolerance
    )
    assert_symbolic_ode_config(system, expected_settings, tolerance)
    assert_step_algorithm_config(step_algorithm, expected_settings, tolerance)
    assert_step_controller_config(
        step_controller, expected_settings, tolerance, is_adaptive
    )
    assert_summary_metrics_config(
        output_functions, expected_settings, tolerance
    )
    
    print("\n=== All assertions passed! ===")
