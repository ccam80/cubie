"""Tests for BatchSolverKernel._generate_dummy_args method."""

import numpy as np
import pytest


class TestBatchSolverGenerateDummyArgs:
    """Test _generate_dummy_args method of BatchSolverKernel."""

    def test_batch_solver_generate_dummy_args(self, solverkernel):
        """Verify BatchSolverKernel returns properly shaped args."""
        result = solverkernel._generate_dummy_args()

        # Check that result has the expected key
        assert 'solver_kernel' in result
        args = result['solver_kernel']

        # Should have 13 arguments matching integration_kernel signature
        assert len(args) == 13

        system_sizes = solverkernel.system_sizes
        n_states = int(system_sizes.states)
        n_params = int(system_sizes.parameters)
        n_obs = int(system_sizes.observables)
        n_drivers = int(system_sizes.drivers)
        precision = solverkernel.precision

        # Verify array shapes and dtypes
        # inits (n_states, 1)
        assert args[0].shape == (n_states, 1)
        assert args[0].dtype == precision

        # params (n_params, 1)
        assert args[1].shape == (n_params, 1)
        assert args[1].dtype == precision

        # d_coefficients (100, n_drivers, 6)
        assert args[2].shape == (100, n_drivers, 6)
        assert args[2].dtype == precision

        # state_output (100, n_states, 1)
        assert args[3].shape == (100, n_states, 1)
        assert args[3].dtype == precision

        # observables_output (100, n_obs, 1)
        assert args[4].shape == (100, n_obs, 1)
        assert args[4].dtype == precision

        # state_summaries (100, n_states, 1)
        assert args[5].shape == (100, n_states, 1)
        assert args[5].dtype == precision

        # observable_summaries (100, n_obs, 1)
        assert args[6].shape == (100, n_obs, 1)
        assert args[6].dtype == precision

        # iteration_counters (100, 4, 1)
        assert args[7].shape == (100, 4, 1)
        assert args[7].dtype == np.int32

        # status_codes (1,)
        assert args[8].shape == (1,)
        assert args[8].dtype == np.int32

        # duration (float64)
        assert args[9] == np.float64(0.001)

        # warmup (float64)
        assert args[10] == np.float64(0.0)

        # t0 (float64)
        assert args[11] == np.float64(0.0)

        # n_runs (int32)
        assert args[12] == np.int32(1)

    def test_batch_solver_no_critical_shapes_attribute(self, solverkernel):
        """Verify kernel no longer has critical_shapes attribute."""
        kernel = solverkernel.kernel

        # The kernel should NOT have critical_shapes attribute
        assert not hasattr(kernel, 'critical_shapes'), (
            "kernel should not have critical_shapes attribute after refactor"
        )

    def test_batch_solver_no_critical_values_attribute(self, solverkernel):
        """Verify kernel no longer has critical_values attribute."""
        kernel = solverkernel.kernel

        # The kernel should NOT have critical_values attribute
        assert not hasattr(kernel, 'critical_values'), (
            "kernel should not have critical_values attribute after refactor"
        )
