"""Integration tests for refactored memory allocation patterns.

Verifies that BatchSolverKernel correctly queries buffer_registry
for memory sizes and that behavior is identical to pre-refactoring.
"""

import pytest
import numpy as np

from cubie.batchsolving.solver import Solver
from cubie.buffer_registry import buffer_registry
from tests.system_fixtures import build_three_state_linear_system


@pytest.fixture(scope="function")
def three_state_linear(precision):
    """Return a three-state linear system for testing.
    
    Uses function scope to ensure buffer_registry cleanup between tests.
    """
    return build_three_state_linear_system(precision)


def test_memory_allocation_via_buffer_registry(three_state_linear):
    """Verify BatchSolverKernel queries buffer_registry for memory sizes.
    
    This test confirms that the memory element properties correctly delegate
    to buffer_registry rather than reading from compile_settings (which no
    longer stores memory sizes after the refactoring).
    """
    solver = Solver(
        system=three_state_linear,
        algorithm='euler',
        dt=0.001,
    )
    
    # Memory sizes should be queryable via properties
    shared_elements = solver.kernel.shared_memory_elements
    local_elements = solver.kernel.local_memory_elements
    shared_bytes = solver.kernel.shared_memory_bytes
    
    # These should match buffer_registry queries directly
    # The single_integrator._loop is the object that registers buffers
    loop = solver.kernel.single_integrator._loop
    
    # Get sizes directly from buffer_registry
    shared_from_registry = buffer_registry.shared_buffer_size(loop)
    local_from_registry = buffer_registry.persistent_local_buffer_size(loop)
    
    # Properties should delegate to buffer_registry methods
    assert shared_elements == shared_from_registry, (
        f"shared_memory_elements property returned {shared_elements}, "
        f"but buffer_registry.shared_buffer_size() returned "
        f"{shared_from_registry}"
    )
    assert local_elements == local_from_registry, (
        f"local_memory_elements property returned {local_elements}, "
        f"but buffer_registry.persistent_local_buffer_size() returned "
        f"{local_from_registry}"
    )
    
    # shared_bytes should be elements * itemsize
    expected_bytes = shared_elements * np.dtype(
        solver.kernel.precision
    ).itemsize
    assert shared_bytes == expected_bytes


def test_memory_sizes_update_with_buffer_changes(three_state_linear):
    """Verify memory sizes reflect buffer_registry state after updates.
    
    This test confirms that after updating integrator parameters, the memory
    properties still correctly query buffer_registry and return valid values.
    """
    solver = Solver(
        system=three_state_linear,
        algorithm='euler',
        dt=0.001,
    )
    
    initial_shared = solver.kernel.shared_memory_elements
    
    # Update a parameter (this should not break buffer_registry queries)
    # The dt0 update is a safe operation that doesn't re-register buffers
    solver.kernel.single_integrator.update({'dt0': 0.002})
    
    # Memory sizes should still be accessible via properties
    updated_shared = solver.kernel.shared_memory_elements
    
    # Size should be consistent (explicit_euler has fixed buffer requirements)
    # Verify the properties return valid integers >= 0
    assert isinstance(updated_shared, int)
    assert updated_shared >= 0
    # For euler with same system, shared memory should be identical
    assert updated_shared == initial_shared


def test_solver_run_with_refactored_allocation(three_state_linear):
    """Verify solver.solve() works correctly with refactored allocation.
    
    This is the key end-to-end integration test that confirms the entire
    refactoring maintains correct behavior. The solver should compile the
    kernel using memory sizes from buffer_registry and successfully execute
    batch solves.
    """
    solver = Solver(
        system=three_state_linear,
        algorithm='euler',
        dt=0.001,
    )
    
    # Prepare initial values and parameters for the solve
    # The three_state_linear system has 3 states and 3 parameters
    initial_values = {
        'x0': [1.0] * 10,
        'x1': [1.0] * 10,
        'x2': [1.0] * 10,
    }
    parameters = {
        'p0': [1.0] * 10,
        'p1': [2.0] * 10,
        'p2': [3.0] * 10,
    }
    
    # Run a solve to verify everything works end-to-end
    # This exercises the full pipeline: kernel build → memory allocation → run
    result = solver.solve(
        initial_values=initial_values,
        parameters=parameters,
        duration=1.0,
    )
    
    # Verify result is valid
    assert result.state is not None
    assert result.state.shape[0] == 10  # 10 runs
    assert result.status_codes is not None
    
    # Verify all runs completed successfully (status_code 0 = success)
    assert np.all(result.status_codes == 0), (
        f"Some runs failed: {result.status_codes}"
    )
    
    # Verify state has expected shape (10 runs, 3 states, 1 final time point)
    # For fixed-step solver with save_every > duration, only final state saved
    assert result.state.shape[1] == 3  # 3 state variables
    
    # Verify memory properties remain accessible after solve
    assert solver.kernel.shared_memory_elements >= 0
    assert solver.kernel.local_memory_elements >= 0
    assert solver.kernel.shared_memory_bytes >= 0
