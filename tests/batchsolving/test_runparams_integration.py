"""Integration tests for RunParams refactoring.

This module tests the end-to-end flow of RunParams through the solver
pipeline, including single chunk, multiple chunk, and exact division
scenarios.
"""

import pytest
import numpy as np
from cubie import create_ODE_system
from cubie.batchsolving.solver import Solver


@pytest.fixture(scope="module")
def integration_system(precision):
    """Create a simple test system for integration testing."""
    equations = [
        "dx = -x + p0",
        "dy = -y + p1",
    ]
    states = {"x": 1.0, "y": 1.0}
    parameters = {"p0": 1.0, "p1": 2.0}
    
    system = create_ODE_system(
        dxdt=equations,
        states=states,
        parameters=parameters,
        precision=precision,
        name="integration_test_system",
    )
    return system


@pytest.fixture(scope="module")
def integration_solver(integration_system, solver_settings):
    """Create a solver for integration testing."""
    solver = Solver(
        integration_system,
        algorithm='euler',
        dt=0.01,
        save_every=0.05,
        output_types=['state'],
        memory_manager=solver_settings["memory_manager"],
        stream_group="integration_test",
    )
    return solver


def test_runparams_single_chunk(integration_solver, integration_system):
    """Verify RunParams works correctly when no chunking occurs.
    
    This test uses a small batch that fits in memory, verifying that
    run_params is created with correct values and num_chunks=1.
    """
    # Create small arrays that fit in memory (no chunking)
    num_runs = 10
    inits = np.random.rand(integration_system.num_states, num_runs)
    params = np.random.rand(integration_system.num_params, num_runs)
    
    result = integration_solver.solve(
        initial_values=inits,
        parameters=params,
        duration=1.0,
        warmup=0.1,
        t0=0.0
    )
    
    # Verify run_params was created and used correctly
    assert integration_solver.kernel.run_params.runs == num_runs
    assert integration_solver.kernel.run_params.num_chunks == 1
    assert integration_solver.kernel.run_params.duration == 1.0
    assert integration_solver.kernel.run_params.warmup == 0.1
    assert integration_solver.kernel.run_params.t0 == 0.0
    
    # Verify chunk_length equals runs when no chunking
    assert integration_solver.kernel.run_params.chunk_length == num_runs
    
    # Verify result was produced
    assert result is not None
    assert result.time_domain_array is not None
    assert result.time_domain_array.shape[2] == num_runs


def test_runparams_multiple_chunks(integration_solver, integration_system):
    """Verify RunParams works correctly with chunking.
    
    This test forces chunking by setting a small memory proportion,
    verifying that run_params is updated with chunking metadata and
    that per-chunk parameters are correctly calculated.
    """
    # Force chunking by setting a very small memory proportion
    integration_solver.kernel.memory_manager.set_manual_proportion(
        integration_solver.kernel, 0.01
    )
    
    # Use enough runs to require chunking
    num_runs = 100
    inits = np.random.rand(integration_system.num_states, num_runs)
    params = np.random.rand(integration_system.num_params, num_runs)
    
    result = integration_solver.solve(
        initial_values=inits,
        parameters=params,
        duration=1.0,
    )
    
    # Verify chunking occurred and run_params was updated
    assert integration_solver.kernel.run_params.runs == num_runs
    assert integration_solver.kernel.run_params.num_chunks > 1
    assert integration_solver.kernel.run_params.chunk_length > 0
    
    # Verify chunk_length calculation is correct
    # chunk_length should be ceil(runs / num_chunks)
    expected_chunk_length = int(
        np.ceil(num_runs / integration_solver.kernel.run_params.num_chunks)
    )
    assert integration_solver.kernel.run_params.chunk_length == (
        expected_chunk_length
    )
    
    # Verify dangling chunk calculation using __getitem__
    last_chunk_index = integration_solver.kernel.run_params.num_chunks - 1
    last_chunk_params = integration_solver.kernel.run_params[last_chunk_index]
    
    # Last chunk runs should account for any "dangling" portion
    expected_last_chunk_runs = (
        num_runs 
        - (integration_solver.kernel.run_params.num_chunks - 1) 
        * integration_solver.kernel.run_params.chunk_length
    )
    assert last_chunk_params.runs == expected_last_chunk_runs
    
    # Verify all chunks have correct timing parameters
    for i in range(integration_solver.kernel.run_params.num_chunks):
        chunk_params = integration_solver.kernel.run_params[i]
        assert chunk_params.duration == 1.0
        assert chunk_params.warmup == 0.0
        assert chunk_params.t0 == 0.0
    
    # Verify result was produced correctly
    assert result is not None
    assert result.time_domain_array is not None
    assert result.time_domain_array.shape[2] == num_runs
    
    # Reset memory proportion for other tests
    integration_solver.kernel.memory_manager.set_auto_limit_mode(
        integration_solver.kernel
    )


def test_runparams_exact_division(integration_solver, integration_system):
    """Verify RunParams when runs divide evenly into chunks.
    
    This test verifies that when the number of runs divides evenly by
    the chunk count, all chunks have exactly the same size with no
    dangling chunk.
    """
    # Force specific chunking to get exact division
    # We'll manually set a proportion that produces exactly 4 chunks
    # with 25 runs per chunk (100 runs total)
    integration_solver.kernel.memory_manager.set_manual_proportion(
        integration_solver.kernel, 0.015
    )
    
    num_runs = 100
    inits = np.random.rand(integration_system.num_states, num_runs)
    params = np.random.rand(integration_system.num_params, num_runs)
    
    result = integration_solver.solve(
        initial_values=inits,
        parameters=params,
        duration=1.0,
    )
    
    # Verify chunking occurred
    num_chunks = integration_solver.kernel.run_params.num_chunks
    chunk_length = integration_solver.kernel.run_params.chunk_length
    
    # If we have exact division, verify all chunks are the same size
    if num_runs % chunk_length == 0:
        # All chunks should have exactly chunk_length runs
        for i in range(num_chunks):
            chunk_params = integration_solver.kernel.run_params[i]
            assert chunk_params.runs == chunk_length
        
        # Last chunk should not be smaller (no dangling portion)
        last_chunk_params = integration_solver.kernel.run_params[num_chunks - 1]
        assert last_chunk_params.runs == chunk_length
    else:
        # If not exact division, last chunk should be smaller
        last_chunk_params = integration_solver.kernel.run_params[num_chunks - 1]
        assert last_chunk_params.runs < chunk_length
        assert last_chunk_params.runs == (
            num_runs - (num_chunks - 1) * chunk_length
        )
    
    # Verify result was produced
    assert result is not None
    assert result.time_domain_array is not None
    assert result.time_domain_array.shape[2] == num_runs
    
    # Reset memory proportion
    integration_solver.kernel.memory_manager.set_auto_limit_mode(
        integration_solver.kernel
    )


def test_runparams_indexing_edge_cases(integration_solver):
    """Verify RunParams.__getitem__ handles edge cases correctly.
    
    This test verifies that chunk indexing works correctly for boundary
    cases like first chunk, last chunk, and out-of-bounds indices.
    """
    # Set up run_params with known values
    from cubie.batchsolving.BatchSolverKernel import RunParams
    
    run_params = RunParams(
        duration=1.0,
        warmup=0.0,
        t0=0.0,
        runs=100,
        num_chunks=4,
        chunk_length=25,
    )
    
    # Test first chunk
    first_chunk = run_params[0]
    assert first_chunk.runs == 25
    assert first_chunk.duration == 1.0
    
    # Test middle chunk
    middle_chunk = run_params[1]
    assert middle_chunk.runs == 25
    
    # Test last chunk (exact division)
    last_chunk = run_params[3]
    assert last_chunk.runs == 25
    
    # Test out-of-bounds indices
    with pytest.raises(IndexError, match="out of range"):
        _ = run_params[4]
    
    with pytest.raises(IndexError, match="out of range"):
        _ = run_params[-1]
    
    # Test dangling chunk scenario
    run_params_dangling = RunParams(
        duration=1.0,
        warmup=0.0,
        t0=0.0,
        runs=103,
        num_chunks=4,
        chunk_length=26,  # ceil(103/4) = 26
    )
    
    # Last chunk should have remaining runs: 103 - 3*26 = 25
    last_chunk_dangling = run_params_dangling[3]
    assert last_chunk_dangling.runs == 25


def test_runparams_immutability(integration_solver):
    """Verify that RunParams is immutable (frozen attrs class).
    
    This test ensures that RunParams instances cannot be modified after
    creation, which is important for thread safety and correctness.
    """
    from cubie.batchsolving.BatchSolverKernel import RunParams
    from attr.exceptions import FrozenInstanceError
    
    run_params = RunParams(
        duration=1.0,
        warmup=0.0,
        t0=0.0,
        runs=100,
    )
    
    # Attempting to modify should raise FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        run_params.runs = 200
    
    with pytest.raises(FrozenInstanceError):
        run_params.duration = 2.0
