"""Solver-level eviction, spill, and staging integration tests.

Memory-manager unit behaviour (thresholds, spill files, stream-local
waits) is covered in ``tests/memory/test_memmgmt.py``; these tests
exercise the same machinery through real solves.
"""

from pathlib import Path

import numpy as np
import pytest

from cubie.cuda_simsafe import cuda
from cubie.memory import MemoryManager
from cubie.memory.array_requests import ArrayResponse
from cubie.memory.mem_manager import HOST_STAGING_BYTES
from tests._utils import _build_solver_instance


@pytest.mark.parametrize("forced_free_mem", [700], indirect=True)
def test_idle_solver_evicted_under_pressure_and_self_heals(
    low_mem_solver,
    second_low_mem_solver,
    low_memory,
    batch_input_arrays,
    driver_settings,
):
    """A competing solve evicts an idle solver that later recovers."""
    y0, params = batch_input_arrays
    solve_kwargs = dict(drivers=driver_settings, duration=0.1)

    low_mem_solver.solve(y0, params, **solve_kwargs)
    idle_outputs_id = id(low_mem_solver.kernel.output_arrays)
    assert low_memory.registry[idle_outputs_id].allocated_bytes > 0

    second_low_mem_solver.solve(y0, params, **solve_kwargs)
    assert low_memory.registry[idle_outputs_id].allocated_bytes == 0

    result = low_mem_solver.solve(y0, params, **solve_kwargs)
    assert low_memory.registry[idle_outputs_id].allocated_bytes > 0
    assert np.isfinite(result.as_numpy["time_domain_array"]).all()


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"host_spill_threshold": 512}],
    indirect=True,
)
def test_host_arrays_spill_to_disk_and_results_match(
    solver_mutable,
    system,
    solver_settings,
    driver_array,
    driver_settings,
    batch_input_arrays,
):
    """Spilled outputs match in-RAM outputs exactly."""
    y0, params = batch_input_arrays
    solve_kwargs = dict(drivers=driver_settings, duration=0.2)

    spilled = solver_mutable.solve(y0, params, **solve_kwargs)
    state_host = solver_mutable.kernel.output_arrays.state
    assert isinstance(state_host, np.memmap)
    assert Path(state_host._cubie_spill_path).exists()
    assert isinstance(spilled.time_domain_array, np.memmap)

    reference_settings = solver_settings.copy()
    reference_settings["host_spill_threshold"] = None
    reference_settings["stream_group"] = "spill_reference"
    reference_solver = _build_solver_instance(
        system=system,
        solver_settings=reference_settings,
        driver_array=driver_array,
        memory_manager=MemoryManager(),
    )
    try:
        reference = reference_solver.solve(y0, params, **solve_kwargs)
        spilled_numpy = spilled.as_numpy["time_domain_array"]
        assert type(spilled_numpy) is np.ndarray
        np.testing.assert_array_equal(
            spilled_numpy, reference.as_numpy["time_domain_array"]
        )
    finally:
        reference_solver.close()


def test_solver_spill_policies_are_independent(
    system,
    solver_settings,
    driver_array,
    driver_settings,
    batch_input_arrays,
    tmp_path,
):
    """Solvers sharing one manager keep separate spill policies."""
    manager = MemoryManager()
    y0, params = batch_input_arrays
    solve_kwargs = dict(drivers=driver_settings, duration=0.1)

    spill_settings = solver_settings.copy()
    spill_settings["host_spill_threshold"] = 1
    spill_settings["spill_directory"] = str(tmp_path)
    spill_settings["stream_group"] = "spill_policy_spill"
    ram_settings = solver_settings.copy()
    ram_settings["host_spill_threshold"] = 2**30
    ram_settings["stream_group"] = "spill_policy_ram"

    spill_solver = _build_solver_instance(
        system=system,
        solver_settings=spill_settings,
        driver_array=driver_array,
        memory_manager=manager,
    )
    ram_solver = _build_solver_instance(
        system=system,
        solver_settings=ram_settings,
        driver_array=driver_array,
        memory_manager=manager,
    )
    try:
        spill_solver.solve(y0, params, **solve_kwargs)
        ram_solver.solve(y0, params, **solve_kwargs)

        assert isinstance(
            spill_solver.kernel.output_arrays.state, np.memmap
        )
        assert len(list(tmp_path.iterdir())) > 0
        assert not isinstance(
            ram_solver.kernel.output_arrays.state, np.memmap
        )
    finally:
        spill_solver.close()
        ram_solver.close()


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


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"host_spill_threshold": 500}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_settings_override",
    [{"num_state_vals_0": 9, "num_param_vals_0": 9}],
    indirect=True,
)
def test_shape_change_updates_memmap_metadata(
    solver_mutable, batch_input_arrays, driver_settings
):
    """Host backing metadata follows each replacement array."""
    y0, params = batch_input_arrays
    solve_kwargs = dict(drivers=driver_settings, duration=0.1)

    solver_mutable.solve(y0, params, **solve_kwargs)
    slot = solver_mutable.kernel.output_arrays.host.state
    assert isinstance(slot.array, np.memmap)
    assert slot.memory_type == "memmap"
    old_path = Path(slot.array._cubie_spill_path)

    solver_mutable.solve(y0[:, :3], params[:, :3], **solve_kwargs)
    assert not isinstance(slot.array, np.memmap)
    assert slot.memory_type == "pinned"
    assert not old_path.exists()


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override",
    [{"host_spill_threshold": 1}],
    indirect=True,
)
def test_spill_solve_is_async(
    solver_mutable,
    batch_input_arrays,
    driver_settings,
    start_cuda_busy_work,
):
    """Spill staging transfers leave unrelated CUDA work running."""
    y0, params = batch_input_arrays
    # Warm solve: compile the kernel, set the driver coefficients, and
    # allocate the spill-backed host arrays.
    solver_mutable.solve(
        y0, params, drivers=driver_settings, duration=0.1
    )

    # Numba's deferred-deallocation queue flushes onto the legacy
    # default stream, which serializes every blocking stream —
    # including the canary against the solver's run stream. Hold the
    # queue from before the canary launches until the assertion, so
    # the canary sees only waits the solve itself performs. The
    # bracketed solve omits drivers: re-supplying them rebuilds the
    # driver function and recompiles, which is not the staging path
    # under test.
    with cuda.defer_cleanup():
        work, stream, done = start_cuda_busy_work()
        try:
            solver_mutable.solve(y0, params, duration=0.1)
            assert not done.query()
            pool = solver_mutable.kernel.output_arrays._buffer_pool
            for buffers in pool._buffers.values():
                assert all(
                    buffer.array.nbytes <= HOST_STAGING_BYTES
                    for buffer in buffers
                )
        finally:
            stream.synchronize()
            assert work.copy_to_host()[0] > 0
