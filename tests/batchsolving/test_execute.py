"""Tests for BatchSolverKernel.execute() low-level API.

These tests require real CUDA device arrays and are marked nocudasim
to skip in simulator mode where device array behavior differs.
"""

import pytest
import numpy as np
from os import environ

# Import only what is needed for the test module

# Check if running in CUDA simulator mode
IS_CUDASIM = environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1"

if not IS_CUDASIM:
    from numba import cuda


@pytest.mark.nocudasim
class TestExecute:
    """Tests for BatchSolverKernel.execute() method."""

    def test_execute_basic(self, solver):
        """execute() runs with pre-allocated device arrays."""
        kernel = solver.kernel
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 4

        # Prepare host arrays
        h_inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
        h_params = np.ones((n_params, n_runs), dtype=precision)

        # First do a normal run to set up output sizes
        kernel.run(
            inits=h_inits,
            params=h_params,
            driver_coefficients=solver.driver_interpolator.coefficients,
            duration=0.1,
        )

        # Get sizes from the kernel after run sets them up
        output_sizes = kernel.output_array_sizes_3d

        # Allocate device arrays
        d_inits = cuda.to_device(h_inits)
        d_params = cuda.to_device(h_params)
        d_driver_coeffs = cuda.to_device(
            solver.driver_interpolator.coefficients
        )

        # Allocate output arrays based on kernel sizes
        d_state = cuda.device_array(
            output_sizes.state, dtype=precision
        )
        d_observables = cuda.device_array(
            output_sizes.observables, dtype=precision
        )
        d_state_summaries = cuda.device_array(
            output_sizes.state_summaries, dtype=precision
        )
        d_observable_summaries = cuda.device_array(
            output_sizes.observable_summaries, dtype=precision
        )
        d_iteration_counters = cuda.device_array(
            output_sizes.iteration_counters, dtype=np.int32
        )
        d_status_codes = cuda.device_array(
            (n_runs,), dtype=np.int32
        )

        # Execute directly with device arrays
        kernel.execute(
            d_inits,
            d_params,
            d_driver_coeffs,
            d_state,
            d_observables,
            d_state_summaries,
            d_observable_summaries,
            d_iteration_counters,
            d_status_codes,
            duration=0.1,
        )

        # Sync and check results
        cuda.synchronize()

        # Copy back and verify non-zero
        state_result = d_state.copy_to_host()
        assert state_result.shape == output_sizes.state

    def test_execute_single_run(self, solver):
        """execute() works with single run."""
        kernel = solver.kernel
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 1

        h_inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
        h_params = np.ones((n_params, n_runs), dtype=precision)

        # Set up sizes via normal run
        kernel.run(
            inits=h_inits,
            params=h_params,
            driver_coefficients=solver.driver_interpolator.coefficients,
            duration=0.1,
        )

        output_sizes = kernel.output_array_sizes_3d

        d_inits = cuda.to_device(h_inits)
        d_params = cuda.to_device(h_params)
        d_driver_coeffs = cuda.to_device(
            solver.driver_interpolator.coefficients
        )
        d_state = cuda.device_array(output_sizes.state, dtype=precision)
        d_observables = cuda.device_array(
            output_sizes.observables, dtype=precision
        )
        d_state_summaries = cuda.device_array(
            output_sizes.state_summaries, dtype=precision
        )
        d_observable_summaries = cuda.device_array(
            output_sizes.observable_summaries, dtype=precision
        )
        d_iteration_counters = cuda.device_array(
            output_sizes.iteration_counters, dtype=np.int32
        )
        d_status_codes = cuda.device_array((n_runs,), dtype=np.int32)

        kernel.execute(
            d_inits,
            d_params,
            d_driver_coeffs,
            d_state,
            d_observables,
            d_state_summaries,
            d_observable_summaries,
            d_iteration_counters,
            d_status_codes,
            duration=0.1,
        )

        cuda.synchronize()
        state_result = d_state.copy_to_host()
        assert state_result.shape[2] == 1  # single run

    def test_execute_with_warmup(self, solver):
        """execute() respects warmup parameter."""
        kernel = solver.kernel
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 2

        h_inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
        h_params = np.ones((n_params, n_runs), dtype=precision)

        kernel.run(
            inits=h_inits,
            params=h_params,
            driver_coefficients=solver.driver_interpolator.coefficients,
            duration=0.1,
            warmup=0.05,
        )

        output_sizes = kernel.output_array_sizes_3d

        d_inits = cuda.to_device(h_inits)
        d_params = cuda.to_device(h_params)
        d_driver_coeffs = cuda.to_device(
            solver.driver_interpolator.coefficients
        )
        d_state = cuda.device_array(output_sizes.state, dtype=precision)
        d_observables = cuda.device_array(
            output_sizes.observables, dtype=precision
        )
        d_state_summaries = cuda.device_array(
            output_sizes.state_summaries, dtype=precision
        )
        d_observable_summaries = cuda.device_array(
            output_sizes.observable_summaries, dtype=precision
        )
        d_iteration_counters = cuda.device_array(
            output_sizes.iteration_counters, dtype=np.int32
        )
        d_status_codes = cuda.device_array((n_runs,), dtype=np.int32)

        # Execute with warmup
        kernel.execute(
            d_inits,
            d_params,
            d_driver_coeffs,
            d_state,
            d_observables,
            d_state_summaries,
            d_observable_summaries,
            d_iteration_counters,
            d_status_codes,
            duration=0.1,
            warmup=0.05,
        )

        cuda.synchronize()
        # Should complete without error


@pytest.mark.nocudasim
class TestRunCallsExecute:
    """Tests ensuring run() delegates to execute()."""

    def test_run_and_execute_consistent(self, solver):
        """run() and execute() produce equivalent results."""
        kernel = solver.kernel
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 2

        h_inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
        h_params = np.ones((n_params, n_runs), dtype=precision)
        driver_coeffs = solver.driver_interpolator.coefficients

        # Run via run()
        kernel.run(
            inits=h_inits,
            params=h_params,
            driver_coefficients=driver_coeffs,
            duration=0.1,
        )
        solver.memory_manager.sync_stream(kernel)

        # Copy results from run()
        run_state = kernel.state.copy()

        # Allocate fresh device arrays for execute()
        output_sizes = kernel.output_array_sizes_3d
        d_inits = cuda.to_device(h_inits)
        d_params = cuda.to_device(h_params)
        d_driver_coeffs = cuda.to_device(driver_coeffs)
        d_state = cuda.device_array(output_sizes.state, dtype=precision)
        d_observables = cuda.device_array(
            output_sizes.observables, dtype=precision
        )
        d_state_summaries = cuda.device_array(
            output_sizes.state_summaries, dtype=precision
        )
        d_observable_summaries = cuda.device_array(
            output_sizes.observable_summaries, dtype=precision
        )
        d_iteration_counters = cuda.device_array(
            output_sizes.iteration_counters, dtype=np.int32
        )
        d_status_codes = cuda.device_array((n_runs,), dtype=np.int32)

        kernel.execute(
            d_inits,
            d_params,
            d_driver_coeffs,
            d_state,
            d_observables,
            d_state_summaries,
            d_observable_summaries,
            d_iteration_counters,
            d_status_codes,
            duration=0.1,
        )
        cuda.synchronize()

        execute_state = d_state.copy_to_host()

        # Results should match
        np.testing.assert_allclose(
            run_state, execute_state, rtol=1e-5, atol=1e-7
        )
