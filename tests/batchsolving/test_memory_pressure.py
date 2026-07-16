"""Memory pressure, eviction, spill, and asynchronous transfer tests."""

from pathlib import Path

import numpy as np
import pytest

from cubie.batchsolving.solver import Solver
from cubie.cuda_simsafe import cuda, CUDA_SIMULATION
from cubie.memory import MemoryManager
from cubie.memory.mem_manager import HOST_STAGING_BYTES
from cubie.memory.array_requests import ArrayResponse


if not CUDA_SIMULATION:
    @cuda.jit
    def _busy_kernel(out):
        value = 0.0
        for _ in range(20_000_000):
            value += 1.0
        out[0] = value


    def _start_cuda_work():
        stream = cuda.stream()
        out = cuda.device_array(1, dtype=np.float32)
        done = cuda.event()
        _busy_kernel[1, 1, stream](out)
        done.record(stream)
        return out, stream, done


@pytest.mark.parametrize("forced_free_mem", [700], indirect=True)
def test_idle_solver_evicted_under_pressure_and_self_heals(
    system, batch_input_arrays, low_memory, forced_free_mem
):
    """A competing solve evicts an idle solver that later recovers."""
    manager = low_memory
    y0, params = batch_input_arrays

    solver_a = Solver(
        system,
        algorithm="euler",
        dt=0.01,
        memory_manager=manager,
        allow_memory_eviction=True,
    )
    solver_b = Solver(
        system,
        algorithm="euler",
        dt=0.01,
        memory_manager=manager,
        allow_memory_eviction=True,
    )

    solver_a.solve(y0, params, duration=0.1)
    a_output_id = id(solver_a.kernel.output_arrays)
    assert manager.registry[a_output_id].allocated_bytes > 0

    solver_b.solve(y0, params, duration=0.1)
    assert manager.registry[a_output_id].allocated_bytes == 0

    result = solver_a.solve(y0, params, duration=0.1)
    assert manager.registry[a_output_id].allocated_bytes > 0
    assert np.isfinite(result.as_numpy["time_domain_array"]).all()


def test_host_arrays_spill_to_disk_and_results_match(
    system, batch_input_arrays, tmp_path
):
    """Spilled outputs match ordinary host outputs exactly."""
    y0, params = batch_input_arrays

    reference_solver = Solver(
        system,
        algorithm="euler",
        dt=0.01,
        save_every=0.05,
        memory_manager=MemoryManager(),
    )
    reference = reference_solver.solve(y0, params, duration=0.2)

    spill_solver = Solver(
        system,
        algorithm="euler",
        dt=0.01,
        save_every=0.05,
        memory_manager=MemoryManager(),
        host_spill_threshold=512,
        spill_directory=str(tmp_path),
    )
    spilled = spill_solver.solve(y0, params, duration=0.2)

    state_host = spill_solver.kernel.output_arrays.state
    assert isinstance(state_host, np.memmap)
    assert len(list(tmp_path.iterdir())) > 0

    np.testing.assert_array_equal(
        spilled.as_numpy["time_domain_array"],
        reference.as_numpy["time_domain_array"],
    )
    assert isinstance(spilled.time_domain_array, np.memmap)
    assert type(spilled.as_numpy["time_domain_array"]) is np.ndarray


def test_solver_spill_policies_are_independent(
    system, batch_input_arrays, tmp_path
):
    """Solvers sharing a manager keep separate spill policies."""
    manager = MemoryManager()
    y0, params = batch_input_arrays
    spill_solver = Solver(
        system,
        algorithm="euler",
        dt=0.01,
        memory_manager=manager,
        host_spill_threshold=1,
        spill_directory=tmp_path,
    )
    memory_solver = Solver(
        system,
        algorithm="euler",
        dt=0.01,
        memory_manager=manager,
        host_spill_threshold=2**30,
        spill_directory=tmp_path,
    )

    spill_solver.solve(y0, params, duration=0.1)
    memory_solver.solve(y0, params, duration=0.1)

    assert isinstance(spill_solver.kernel.output_arrays.state, np.memmap)
    assert not isinstance(memory_solver.kernel.output_arrays.state, np.memmap)


def test_empty_peer_response_changes_nothing(solver_mutable):
    """An empty peer response leaves array state unchanged."""
    arrays = solver_mutable.kernel.output_arrays
    chunks = arrays._chunks
    memory_types = {
        name: slot.memory_type
        for name, slot in arrays.host.iter_managed_arrays()
    }
    arrays._on_allocation_complete(
        ArrayResponse(
            arr={}, chunks=99, chunk_length=1, chunked_shapes={}
        )
    )
    assert arrays._chunks == chunks
    assert {
        name: slot.memory_type
        for name, slot in arrays.host.iter_managed_arrays()
    } == memory_types


def test_shape_change_updates_memmap_metadata(
    system, batch_input_arrays, tmp_path
):
    """Host backing metadata follows each replacement array."""
    solver = Solver(
        system,
        algorithm="euler",
        dt=0.01,
        memory_manager=MemoryManager(),
        host_spill_threshold=500,
        spill_directory=tmp_path,
    )
    y0, params = batch_input_arrays
    large_y0 = np.tile(y0, 10)
    large_params = np.tile(params, 10)
    solver.solve(large_y0, large_params, duration=0.1)
    slot = solver.kernel.output_arrays.host.state
    assert isinstance(slot.array, np.memmap)
    assert slot.memory_type == "memmap"
    old_path = Path(slot.array._cubie_spill_path)

    solver.solve(y0, params, duration=0.1)
    assert not isinstance(slot.array, np.memmap)
    assert slot.memory_type == "pinned"
    assert not old_path.exists()


@pytest.mark.nocudasim
def test_host_spill_does_not_wait_for_unrelated_stream(tmp_path):
    """Host spill setup leaves unrelated CUDA work running."""
    work, stream, done = _start_cuda_work()
    manager = MemoryManager(host_spill_threshold=1, spill_directory=tmp_path)
    try:
        array = manager.create_host_array((32,), np.float32, "host")
        assert isinstance(array, np.memmap)
        assert not done.query()
    finally:
        stream.synchronize()
        assert work.copy_to_host()[0] > 0


@pytest.mark.nocudasim
def test_legacy_default_stream_sync_is_stream_local():
    """Legacy stream zero does not wait for unrelated CUDA work."""
    manager = MemoryManager()
    owner = object()
    manager.register(owner)
    ready = cuda.event()
    ready.record(manager.get_stream(owner))
    ready.synchronize()
    work, stream, done = _start_cuda_work()
    try:
        manager.sync_stream(owner, stream=0)
        assert not done.query()
    finally:
        stream.synchronize()
        assert work.copy_to_host()[0] > 0


@pytest.mark.nocudasim
def test_spill_kernel_run_is_async(
    system, batch_input_arrays, tmp_path
):
    """Spill transfers do not synchronize unrelated streams."""
    y0, params = batch_input_arrays
    solver = Solver(
        system,
        algorithm="euler",
        dt=0.01,
        memory_manager=MemoryManager(),
        host_spill_threshold=1,
        spill_directory=tmp_path,
    )
    solver.solve(y0, params, duration=0.1)
    work, stream, done = _start_cuda_work()
    try:
        solver.kernel.run(y0, params, None, duration=0.1)
        assert not done.query()
        for buffers in solver.kernel.output_arrays._buffer_pool._buffers.values():
            assert all(
                buffer.array.nbytes <= HOST_STAGING_BYTES
                for buffer in buffers
            )
    finally:
        solver.kernel.synchronize()
        solver.kernel.wait_for_writeback()
        solver.close()
        stream.synchronize()
        assert work.copy_to_host()[0] > 0
