"""Tests for buffer location argument filtering in Solver."""

import pytest

from cubie.batchsolving.solver import Solver
from cubie.integrators.loops.ode_loop import ALL_BUFFER_LOCATION_PARAMETERS
from cubie.integrators.algorithms import ALL_ALGORITHM_BUFFER_LOCATION_PARAMETERS


class TestBufferLocationFiltering:
    """Test buffer location parameter handling in Solver."""

    def test_buffer_location_params_recognized_at_init(self, system, precision):
        """Buffer location params should be recognized in Solver.__init__."""
        solver = Solver(
            system,
            state_buffer_location='shared',
            parameters_location='local',
        )
        # Verify the buffer locations are actually set correctly
        loop = solver.kernel.single_integrator._loop
        buffer_settings = loop.compile_settings.buffer_settings
        assert buffer_settings.state_buffer_location == 'shared'
        assert buffer_settings.parameters_location == 'local'

    def test_buffer_location_all_params_defined(self):
        """Verify all expected buffer location params are in the constant."""
        expected_params = {
            "state_buffer_location",
            "state_proposal_location",
            "parameters_location",
            "drivers_location",
            "drivers_proposal_location",
            "observables_location",
            "observables_proposal_location",
            "error_location",
            "counters_location",
            "state_summary_location",
            "observable_summary_location",
            "scratch_location",
        }
        assert ALL_BUFFER_LOCATION_PARAMETERS == expected_params

    def test_buffer_location_invalid_value_rejected(self, system, precision):
        """Invalid buffer location values should be rejected."""
        with pytest.raises(ValueError):
            Solver(
                system,
                state_buffer_location='invalid_location',
            )

    def test_buffer_location_update_recognized(self, system, precision):
        """Buffer location params should be recognized in solver.update()."""
        solver = Solver(system, state_buffer_location='shared')
        # Verify initial value
        loop = solver.kernel.single_integrator._loop
        buffer_settings = loop.compile_settings.buffer_settings
        assert buffer_settings.state_buffer_location == 'shared'
        
        # Update and verify the recognized set includes the parameter
        recognized = solver.update(state_buffer_location='local', silent=True)
        assert 'state_buffer_location' in recognized
        
        # Verify the buffer location actually changed
        buffer_settings = loop.compile_settings.buffer_settings
        assert buffer_settings.state_buffer_location == 'local'

    def test_buffer_size_update_changes_indices(self, system, precision):
        """Changing buffer size should update shared indices accordingly."""
        solver = Solver(system, state_buffer_location='shared')
        loop = solver.kernel.single_integrator._loop
        buffer_settings = loop.compile_settings.buffer_settings
        
        # Record initial n_states and shared memory elements
        initial_n_states = buffer_settings.n_states
        initial_shared_elements = buffer_settings.shared_memory_elements
        
        # Update n_states (which affects shared indices when in shared memory)
        new_n_states = initial_n_states + 2
        buffer_settings.n_states = new_n_states
        
        # Verify shared memory elements changed
        new_shared_elements = buffer_settings.shared_memory_elements
        assert new_shared_elements != initial_shared_elements
        assert buffer_settings.n_states == new_n_states

    def test_buffer_location_preserved_on_unrelated_update(
        self, system, precision
    ):
        """Buffer locations should be preserved when updating other params."""
        solver = Solver(
            system,
            state_buffer_location='local',
            parameters_location='local',
        )
        # Verify initial values
        loop = solver.kernel.single_integrator._loop
        buffer_settings = loop.compile_settings.buffer_settings
        assert buffer_settings.state_buffer_location == 'local'
        assert buffer_settings.parameters_location == 'local'
        
        # Update an unrelated parameter
        solver.update(dt=0.005, silent=True)
        
        # Verify buffer locations are preserved
        buffer_settings = loop.compile_settings.buffer_settings
        assert buffer_settings.state_buffer_location == 'local'
        assert buffer_settings.parameters_location == 'local'
 
    def test_buffer_location_cache_invalidation(self, system, precision):
        """Changing buffer location should invalidate the cache."""
        solver = Solver(system, state_buffer_location='shared')
        _ = solver.kernel.kernel
        loop = solver.kernel.single_integrator._loop
        # Force compilation by accessing the device function
        _ = loop.device_function
        
        # Cache should be valid after compilation
        assert loop.cache_valid
        
        # Update buffer location directly on the loop to test cache invalidation
        loop.update_compile_settings(
            {'state_buffer_location': 'local'}, silent=True
        )
        
        # Cache should be invalid after buffer location change
        assert not loop.cache_valid


class TestAlgorithmBufferLocationFiltering:
    """Test algorithm buffer location parameter handling in Solver."""

    def test_erk_buffer_location_from_init(self, system, precision):
        """ERK buffer locations should be settable from Solver init."""
        solver = Solver(
            system,
            algorithm='erk',
            stage_rhs_location='shared',
            stage_accumulator_location='shared',
        )
        algo_step = solver.kernel.single_integrator._algo_step
        buffer_settings = algo_step.compile_settings.buffer_settings
        assert buffer_settings.stage_rhs_location == 'shared'
        assert buffer_settings.stage_accumulator_location == 'shared'

    def test_erk_buffer_locations_local(self, system, precision):
        """ERK with local buffers should report zero shared memory."""
        solver = Solver(
            system,
            algorithm='erk',
            stage_rhs_location='local',
            stage_accumulator_location='local',
        )
        algo_step = solver.kernel.single_integrator._algo_step
        buffer_settings = algo_step.compile_settings.buffer_settings
        assert buffer_settings.shared_memory_elements == 0
        assert buffer_settings.local_memory_elements > 0

    def test_erk_buffer_locations_shared(self, system, precision):
        """ERK with shared buffers should report nonzero shared memory."""
        solver = Solver(
            system,
            algorithm='erk',
            stage_rhs_location='shared',
            stage_accumulator_location='shared',
        )
        algo_step = solver.kernel.single_integrator._algo_step
        buffer_settings = algo_step.compile_settings.buffer_settings
        assert buffer_settings.shared_memory_elements > 0

    def test_algorithm_buffer_location_params_combined(self):
        """Verify combined algorithm buffer location parameters."""
        # Check that expected params are in the combined set
        assert 'stage_rhs_location' in ALL_ALGORITHM_BUFFER_LOCATION_PARAMETERS
        assert 'stage_accumulator_location' in ALL_ALGORITHM_BUFFER_LOCATION_PARAMETERS
        assert 'stage_increment_location' in ALL_ALGORITHM_BUFFER_LOCATION_PARAMETERS
        assert 'solver_scratch_location' in ALL_ALGORITHM_BUFFER_LOCATION_PARAMETERS
