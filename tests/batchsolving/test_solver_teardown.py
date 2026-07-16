"""Solver resource cleanup tests."""

import gc

import numpy as np
import pytest

from cubie.batchsolving.solver import Solver, solve_ivp
from cubie.cuda_simsafe import cuda, CUDA_SIMULATION
from cubie.memory.mem_manager import MemoryManager
from tests._utils import _build_solver_instance


if not CUDA_SIMULATION:
    @cuda.jit
    def _busy_kernel(out):
        # Sized to outlast a solver close by a wide margin on a warm
        # GPU (~5 s) so the not-yet-done canary assertions hold.
        value = 0.0
        for _ in range(100_000_000):
            value += 1.0
        out[0] = value


    def _start_cuda_work():
        stream = cuda.stream()
        out = cuda.device_array(1, dtype=np.float32)
        done = cuda.event()
        _busy_kernel[1, 1, stream](out)
        done.record(stream)
        return out, stream, done


    def _finish_cuda_work(out, stream, done):
        stream.synchronize()
        assert done.query()


def _instance_ids(solver):
    """Return the ids of the three memory-manager clients of a solver."""
    kernel = solver.kernel
    return (
        id(kernel),
        id(kernel.input_arrays),
        id(kernel.output_arrays),
    )


def _still_registered(manager, ids):
    """Return the subset of ``ids`` still present in the registry."""
    return [instance_id for instance_id in ids if instance_id in
            manager.registry]


def _registered_bytes(manager, ids):
    """Total device bytes the registry keeps alive for ``ids``."""
    total = 0
    for instance_id in ids:
        settings = manager.registry.get(instance_id)
        if settings is not None:
            total += settings.allocated_bytes
    return total


def test_solver_releases_registry_on_gc(
    system, batch_input_arrays, thread_mem_manager
):
    """Collection releases registry entries and buffers."""
    manager = thread_mem_manager
    solver = Solver(system, algorithm="euler", dt=0.01, memory_manager=manager)
    y0, params = batch_input_arrays
    solver.solve(y0, params, duration=0.1)

    ids = _instance_ids(solver)
    assert _still_registered(manager, ids) == list(ids)
    assert _registered_bytes(manager, ids) > 0

    del solver
    gc.collect()

    assert _still_registered(manager, ids) == []


def test_close_releases_registry_immediately(
    solver_mutable, batch_input_arrays, driver_settings, thread_mem_manager
):
    """``close`` deregisters without waiting for garbage collection."""
    manager = thread_mem_manager
    solver = solver_mutable
    y0, params = batch_input_arrays
    solver.solve(y0, params, drivers=driver_settings, duration=0.1)

    ids = _instance_ids(solver)
    assert _still_registered(manager, ids) == list(ids)

    solver.close()

    assert _still_registered(manager, ids) == []
    solver.close()
    assert _still_registered(manager, ids) == []


def test_closed_solver_raises_on_solve(
    solver_mutable, batch_input_arrays, driver_settings
):
    """A closed solver rejects another solve."""
    solver = solver_mutable
    y0, params = batch_input_arrays
    solver.solve(y0, params, drivers=driver_settings, duration=0.1)
    solver.close()

    with pytest.raises(RuntimeError, match="closed"):
        solver.solve(y0, params, duration=0.1)


def test_context_manager_releases_on_exit(
    solver_mutable, batch_input_arrays, driver_settings, thread_mem_manager
):
    """Context exit releases the solver."""
    manager = thread_mem_manager
    y0, params = batch_input_arrays
    with solver_mutable as solver:
        solver.solve(y0, params, drivers=driver_settings, duration=0.1)
        ids = _instance_ids(solver)
        assert _still_registered(manager, ids) == list(ids)

    assert _still_registered(manager, ids) == []


@pytest.mark.nocudasim
def test_close_timeout_is_retryable(
    solver_mutable, batch_input_arrays, driver_settings, thread_mem_manager
):
    """A timed-out close can be retried."""
    manager = thread_mem_manager
    solver = solver_mutable
    y0, params = batch_input_arrays
    solver.solve(y0, params, drivers=driver_settings, duration=0.1)
    input_arrays = solver.kernel.input_arrays
    instance_id = id(input_arrays)
    settings = manager.registry[instance_id]
    group = manager.get_stream_group(input_arrays)
    work_output, work_stream, work_done = _start_cuda_work()
    buffer = input_arrays._buffer_pool.acquire(
        "close_gate", (1,), np.dtype(np.float32)
    )
    input_arrays._transfer_watcher.submit_release(
        work_done,
        buffer,
        input_arrays._buffer_pool,
        "close_gate",
    )

    try:
        with pytest.raises(TimeoutError, match="wait_all timed out"):
            solver.close(shutdown_timeout=0.0)

        assert manager.registry[instance_id] is settings
        assert instance_id in manager._auto_pool
        assert instance_id in manager.stream_groups.get_instances_in_group(
            group
        )
    finally:
        _finish_cuda_work(work_output, work_stream, work_done)

    solver.close()
    assert _still_registered(manager, _instance_ids(solver)) == []


def test_solve_ivp_releases_temporary_solver(
    system, batch_input_arrays, thread_mem_manager
):
    """solve_ivp releases its temporary solver."""
    manager = thread_mem_manager
    baseline = set(manager.registry)
    y0, params = batch_input_arrays

    solve_ivp(
        system,
        y0,
        params,
        duration=0.1,
        grid_type="verbatim",
        dt=0.01,
        memory_manager=manager,
    )

    assert set(manager.registry) == baseline


@pytest.mark.nocudasim
def test_custom_stream_close_does_not_wait_for_unrelated_stream(
    solver_mutable,
    batch_input_arrays,
    driver_array,
    solver_settings,
    system,
    thread_mem_manager,
):
    """Close waits only for the run stream."""
    manager = thread_mem_manager
    target_solver = solver_mutable
    y0, params = batch_input_arrays
    ids = _instance_ids(target_solver)
    assert _registered_bytes(manager, ids) == 0

    run_stream = cuda.stream()
    target_solver.kernel.run(
        y0,
        params,
        target_solver.driver_interpolator.coefficients,
        duration=0.1,
        stream=run_stream,
    )
    custom_stream_state_view = target_solver.kernel.state
    assert _registered_bytes(manager, ids) > 0
    # Drain Numba's deferred-deallocation queue before launching the
    # canary: a flush of unrelated queued driver frees (events,
    # streams, modules from earlier tests) synchronizes the device,
    # which this test would misread as close() waiting on the canary.
    gc.collect()
    cuda.current_context().deallocations.clear()
    work_output, unrelated_stream, unrelated_done = _start_cuda_work()
    try:
        target_solver.close()

        assert _still_registered(manager, ids) == []
        custom_stream_state = custom_stream_state_view.copy()
        assert not unrelated_done.query()
    finally:
        _finish_cuda_work(work_output, unrelated_stream, unrelated_done)

    reference_settings = solver_settings.copy()
    reference_settings["stream_group"] = "close_reference"
    reference_solver = _build_solver_instance(
        system=system,
        solver_settings=reference_settings,
        driver_array=driver_array,
        memory_manager=MemoryManager(),
    )
    try:
        reference_solver.kernel.run(
            y0,
            params,
            reference_solver.driver_interpolator.coefficients,
            duration=0.1,
        )
        reference_solver.kernel.synchronize()
        reference_solver.kernel.wait_for_writeback()
        expected_state = reference_solver.kernel.state.copy()
    finally:
        reference_solver.close()
    np.testing.assert_array_equal(custom_stream_state, expected_state)


def test_repeated_solvers_do_not_grow_registry(
    system, batch_input_arrays, thread_mem_manager
):
    """Repeated solvers do not grow the registry."""
    manager = thread_mem_manager
    y0, params = batch_input_arrays

    gc.collect()
    baseline = len(manager.registry)

    for _ in range(6):
        solver = Solver(
            system, algorithm="euler", dt=0.01, memory_manager=manager
        )
        solver.solve(y0, params, duration=0.1)
        del solver
        gc.collect()

    assert len(manager.registry) <= baseline
