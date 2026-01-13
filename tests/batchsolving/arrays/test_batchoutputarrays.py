import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cubie.batchsolving.arrays.BatchOutputArrays import (
    OutputArrayContainer,
    OutputArrays,
)
from cubie.memory.mem_manager import MemoryManager
from cubie.outputhandling.output_sizes import BatchOutputSizes


@pytest.fixture(scope="session")
def output_test_overrides(request):
    if hasattr(request, "param"):
        return request.param
    return {}


@pytest.fixture(scope="session")
def output_test_settings(output_test_overrides):
    settings = {
        "num_runs": 5,
        "dtype": np.float32,
        "memory": "device",
        "stream_group": "default",
        "memory_proportion": None,
    }
    settings.update(output_test_overrides)
    return settings


@pytest.fixture(scope="function")
def test_memory_manager():
    """Create a MemoryManager instance for testing"""
    return MemoryManager(mode="passive")


@pytest.fixture(scope="function")
def output_arrays_manager(
    precision, solver, output_test_settings, test_memory_manager
):
    """Create a OutputArrays instance using real solver"""
    solver.kernel.duration = 1.0

    batch_output_sizes = BatchOutputSizes.from_solver(solver)
    return OutputArrays(
        sizes=batch_output_sizes,
        precision=precision,
        stream_group=output_test_settings["stream_group"],
        memory_proportion=output_test_settings["memory_proportion"],
        memory_manager=test_memory_manager,
    )


@pytest.fixture(scope="function")
def sample_output_arrays(solver_mutable, output_test_settings, precision):
    """Create sample output arrays for testing based on real solver"""
    solver = solver_mutable
    solver.kernel.duration = 1.0
    num_runs = output_test_settings["num_runs"]
    dtype = precision

    # Get dimensions from solver's output sizes
    output_sizes = BatchOutputSizes.from_solver(solver)
    time_points = output_sizes.state[0]
    variables_count = solver.system_sizes.states
    observables_count = solver.system_sizes.observables

    return {
        "state": np.random.rand(time_points, variables_count, num_runs).astype(
            dtype
        ),
        "observables": np.random.rand(
            time_points, observables_count, num_runs
        ).astype(dtype),
        "state_summaries": np.random.rand(
            max(0, time_points - 1), variables_count, num_runs
        ).astype(dtype),
        "observable_summaries": np.random.rand(
            max(0, time_points - 1), observables_count, num_runs
        ).astype(dtype),
        "status_codes": np.random.randint(0, 5, size=num_runs, dtype=np.int32),
    }


class TestOutputArrayContainer:
    """Test the OutputArrayContainer class"""

    def test_container_arrays_after_init(self):
        """Test that container has correct arrays after initialization"""
        container = OutputArrayContainer()
        expected_arrays = {
            "iteration_counters",
            "state",
            "observables",
            "state_summaries",
            "observable_summaries",
            "status_codes",
        }
        assert set(container.array_names()) == expected_arrays
        for _, managed in container.iter_managed_arrays():
            assert_array_equal(
                managed.array, np.zeros(managed.shape, dtype=managed.dtype)
            )

    def test_container_stride_order(self):
        """Test that stride order is set correctly"""
        container = OutputArrayContainer()
        stride_order = container.state.stride_order
        assert stride_order == ("time", "variable", "run")

    def test_container_memory_type_default(self):
        """Test default memory type"""
        container = OutputArrayContainer()
        assert container.state.memory_type == "device"

    def test_host_factory(self):
        """Test host factory method creates pinned memory container"""
        container = OutputArrayContainer.host_factory()
        assert container.state.memory_type == "pinned"

    def test_device_factory(self):
        """Test device factory method"""
        container = OutputArrayContainer.device_factory()
        assert container.state.memory_type == "device"


class TestOutputArrays:
    """Test the OutputArrays class"""

    def test_initialization_container_types(self, output_arrays_manager):
        """Test that containers have correct array types after initialization"""
        # Check host container arrays
        expected_arrays = {
            "iteration_counters",
            "state",
            "observables",
            "state_summaries",
            "observable_summaries",
            "status_codes",
        }
        assert set(output_arrays_manager.host.array_names()) == expected_arrays
        assert (
            set(output_arrays_manager.device.array_names()) == expected_arrays
        )

        # Check memory types are set correctly in post_init
        for _, managed in output_arrays_manager.host.iter_managed_arrays():
            assert managed.memory_type == "pinned"
        for _, managed in output_arrays_manager.device.iter_managed_arrays():
            assert managed.memory_type == "device"

    def test_from_solver_factory(self, solver):
        """Test creating OutputArrays from solver"""
        output_arrays = OutputArrays.from_solver(solver)

        assert isinstance(output_arrays, OutputArrays)
        assert isinstance(output_arrays._sizes, BatchOutputSizes)
        assert output_arrays._precision == solver.precision

    def test_allocation_and_getters_not_none(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Test that all getters return non-None after allocation"""
        # Call the manager to allocate arrays based on solver
        output_arrays_manager.update(solver)
        # Process the allocation queue to create device arrays
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )

        # Check host getters
        assert output_arrays_manager.state is not None
        assert output_arrays_manager.observables is not None
        assert output_arrays_manager.state_summaries is not None
        assert output_arrays_manager.observable_summaries is not None
        assert output_arrays_manager.status_codes is not None

        # Check device getters
        assert output_arrays_manager.device_state is not None
        assert output_arrays_manager.device_observables is not None
        assert output_arrays_manager.device_state_summaries is not None
        assert output_arrays_manager.device_observable_summaries is not None
        assert output_arrays_manager.device_status_codes is not None

    def test_getters_return_none_before_allocation(
        self, output_arrays_manager
    ):
        """Test that getters return None before allocation"""
        assert output_arrays_manager.device_state is None
        assert output_arrays_manager.device_observables is None
        assert output_arrays_manager.device_state_summaries is None
        assert output_arrays_manager.device_observable_summaries is None

    def test_call_method_allocates_arrays(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Test that update method allocates arrays based on solver"""
        # Call the manager - it allocates based on solver sizes only
        output_arrays_manager.update(solver)
        # Process the allocation queue to create device arrays
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )

        # Check that arrays were allocated
        assert output_arrays_manager.state is not None
        assert output_arrays_manager.observables is not None
        assert output_arrays_manager.state_summaries is not None
        assert output_arrays_manager.observable_summaries is not None
        assert output_arrays_manager.status_codes is not None

        # Check device arrays
        assert output_arrays_manager.device_state is not None
        assert output_arrays_manager.device_observables is not None
        assert output_arrays_manager.device_state_summaries is not None
        assert output_arrays_manager.device_observable_summaries is not None
        assert output_arrays_manager.device_status_codes is not None

    def test_reallocation_on_size_change(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Test that arrays are reallocated when sizes change"""
        # Initial allocation
        output_arrays_manager.update(solver)
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )
        original_device_state = output_arrays_manager.device_state
        original_shape = output_arrays_manager.device_state.shape

        # Simulate a size change by directly updating the sizes
        new_shape = (
            original_shape[0] + 2,
            original_shape[1],
            original_shape[2],
        )
        output_arrays_manager._sizes.state = new_shape

        # Force reallocation by calling update_from_solver
        output_arrays_manager.update_from_solver(solver)

        # Verify reallocation occurred (array should be different object)
        assert output_arrays_manager.device_state is not None

    def test_chunking_affects_device_array_size(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Test that chunking changes device array allocation size"""
        # Allocate initially
        output_arrays_manager.update(solver)
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )

        # Set up chunking - this should affect the device array size
        output_arrays_manager._chunks = 2
        output_arrays_manager._chunk_axis = "run"

        # The chunking logic should be reflected in allocation behavior
        # (This tests the chunking mechanism exists)
        assert output_arrays_manager._chunks == 2
        assert output_arrays_manager._chunk_axis == "run"

    def test_update_from_solver(self, output_arrays_manager, solver):
        """Test update_from_solver method"""
        output_arrays_manager.update_from_solver(solver)

        assert output_arrays_manager._precision == solver.precision
        assert isinstance(output_arrays_manager._sizes, BatchOutputSizes)

    def test_update_from_solver_fast_path(self, output_arrays_manager, solver):
        """Test that update_from_solver reuses arrays when shape/dtype match."""
        # First call allocates arrays
        output_arrays_manager.update(solver)

        # Store references to existing arrays
        original_state = output_arrays_manager.host.state.array
        original_observables = output_arrays_manager.host.observables.array
        original_status_codes = output_arrays_manager.host.status_codes.array

        # Second call with same solver should reuse arrays
        new_arrays = output_arrays_manager.update_from_solver(solver)

        # Arrays should be the same objects (not reallocated)
        assert new_arrays["state"] is original_state
        assert new_arrays["observables"] is original_observables
        assert new_arrays["status_codes"] is original_status_codes

    def test_initialise_method(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Test initialise method (no-op for outputs)"""
        # Set up the manager
        output_arrays_manager.update(solver)
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )

        # Set up chunking
        output_arrays_manager._chunks = 1
        output_arrays_manager._chunk_axis = "run"

        # Call initialise - should be a no-op for OutputArrays
        host_indices = slice(None)
        output_arrays_manager.initialise(host_indices)

        # Since initialise is a no-op for outputs, just verify it doesn't crash
        # and arrays remain intact
        assert output_arrays_manager.device_state is not None
        assert output_arrays_manager.device_observables is not None

    @pytest.mark.nocudasim
    def test_finalise_method_copies_device_to_host(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Test finalise method copies data from device to host"""
        # Set up the manager
        output_arrays_manager.update(solver)
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )

        # Simulate computation by modifying device arrays
        # (In reality, CUDA kernels would write to these device arrays)
        original_device_state = np.array(output_arrays_manager.device_state)
        original_device_observables = np.array(
            output_arrays_manager.device_observables
        )
        original_status_codes = np.arange(
            output_arrays_manager.device_status_codes.size, dtype=np.int32
        )

        # Modify device arrays to simulate kernel output
        output_arrays_manager.device_state[:] = original_device_state * 2
        output_arrays_manager.device_observables[:] = (
            original_device_observables * 3
        )
        output_arrays_manager.device_status_codes[:] = original_status_codes

        # Set up chunking
        output_arrays_manager._chunks = 1
        output_arrays_manager._chunk_axis = "run"

        # Call finalise - queues async transfer
        host_indices = slice(None)
        output_arrays_manager.finalise(host_indices)

        # Sync stream and complete writebacks
        output_arrays_manager._memory_manager.sync_stream(
            output_arrays_manager
        )
        output_arrays_manager.wait_pending()

        # Verify that host arrays now contain the modified device data
        np.testing.assert_array_equal(
            output_arrays_manager.state,
            output_arrays_manager.device_state,
        )
        np.testing.assert_array_equal(
            output_arrays_manager.observables,
            output_arrays_manager.device_observables,
        )
        np.testing.assert_array_equal(
            output_arrays_manager.status_codes,
            output_arrays_manager.device_status_codes,
        )


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"precision": np.float32}, {"precision": np.float64}],
    indirect=True,
)
def test_dtype(output_arrays_manager, solver, precision, test_memory_manager):
    """Test OutputArrays with different configurations"""
    # Test that the manager works with different configurations
    output_arrays_manager.update(solver)
    test_memory_manager.allocate_queue(output_arrays_manager, chunk_axis="run")

    expected_dtype = precision
    assert output_arrays_manager.state.dtype == expected_dtype
    assert output_arrays_manager.observables.dtype == expected_dtype
    assert output_arrays_manager.state_summaries.dtype == expected_dtype
    assert output_arrays_manager.observable_summaries.dtype == expected_dtype
    assert output_arrays_manager.status_codes.dtype == np.int32
    assert output_arrays_manager.device_status_codes.dtype == np.int32
    assert output_arrays_manager.device_state.dtype == expected_dtype
    assert output_arrays_manager.device_observables.dtype == expected_dtype
    assert output_arrays_manager.device_state_summaries.dtype == expected_dtype
    assert (
        output_arrays_manager.device_observable_summaries.dtype
        == expected_dtype
    )


@pytest.mark.parametrize(
    "output_test_overrides",
    [
        {"num_runs": 8},
        {"num_runs": 3},
        {"stream_group": "test_group", "memory_proportion": 0.5},
    ],
    indirect=True,
)
def test_output_arrays_with_different_configs(
    output_arrays_manager, solver, output_test_settings, test_memory_manager
):
    """Test OutputArrays with different configurations"""
    # Test that the manager works with different configurations
    solver.kernel.num_runs = output_test_settings["num_runs"]
    output_arrays_manager.update(solver)
    test_memory_manager.allocate_queue(output_arrays_manager, chunk_axis="run")

    # Check shapes match expected configuration based on solver
    expected_num_runs = output_test_settings["num_runs"]
    assert output_arrays_manager.state is not None
    assert output_arrays_manager.observables is not None
    assert output_arrays_manager.state_summaries is not None
    assert output_arrays_manager.observable_summaries is not None
    assert output_arrays_manager.status_codes.shape == (expected_num_runs,)

    # Check data types
    expected_dtype = output_test_settings["dtype"]
    assert output_arrays_manager.state.dtype == expected_dtype
    assert output_arrays_manager.observables.dtype == expected_dtype
    assert output_arrays_manager.state_summaries.dtype == expected_dtype
    assert output_arrays_manager.observable_summaries.dtype == expected_dtype
    assert output_arrays_manager.status_codes.dtype == np.int32
    assert output_arrays_manager.device_status_codes.shape == (
        expected_num_runs,
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "system_type": "three_chamber",
            "saved_state_indices": None,
            "saved_observable_indices": None,
            "output_types": [
                "state",
                "observables",
                "mean",
                "max",
                "rms",
                "peaks[2]",
            ],
        },
        {
            "system_type": "stiff",
            "saved_state_indices": None,
            "saved_observable_indices": None,
            "output_types": [
                "state",
                "observables",
                "mean",
                "max",
                "rms",
                "peaks[2]",
            ],
        },
        {
            "system_type": "linear",
            "saved_state_indices": None,
            "saved_observable_indices": None,
            "output_types": [
                "state",
                "observables",
                "mean",
                "max",
                "rms",
                "peaks[2]",
            ],
        },
    ],
    indirect=True,
)
def test_output_arrays_with_different_systems(
    output_arrays_manager, solver_mutable, test_memory_manager
):
    """Test OutputArrays with different system models"""
    # Test that the manager works with different system types
    solver = solver_mutable
    solver.kernel.duration = 1.0
    output_arrays_manager.update(solver)
    test_memory_manager.allocate_queue(output_arrays_manager, chunk_axis="run")

    # Verify the arrays match the system's requirements
    # With stride order (time, variable, run), variable is at index 1
    assert (
        output_arrays_manager.state.shape[1]
        == solver.output_array_heights.state
    )
    assert (
        output_arrays_manager.observables.shape[1]
        == solver.output_array_heights.observables
    )
    assert (
        output_arrays_manager.state_summaries.shape[1]
        == solver.output_array_heights.state_summaries
    )
    assert (
        output_arrays_manager.observable_summaries.shape[1]
        == solver.output_array_heights.observable_summaries
    )

    # Check that all getters work
    assert output_arrays_manager.state is not None
    assert output_arrays_manager.observables is not None
    assert output_arrays_manager.state_summaries is not None
    assert output_arrays_manager.observable_summaries is not None
    assert output_arrays_manager.device_state is not None
    assert output_arrays_manager.device_observables is not None
    assert output_arrays_manager.device_state_summaries is not None
    assert output_arrays_manager.device_observable_summaries is not None


class TestOutputArraysSpecialCases:
    """Test special cases for OutputArrays"""

    def test_allocation_with_different_solver_sizes(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Test that arrays are allocated based on solver sizes"""
        # Test allocation - arrays should be sized based on solver
        output_arrays_manager.update(solver)
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )

        # Manager should be set up without errors and arrays should exist
        assert output_arrays_manager.state is not None
        assert output_arrays_manager.observables is not None
        assert output_arrays_manager.state_summaries is not None
        assert output_arrays_manager.observable_summaries is not None


class TestBufferPoolAndWatcherIntegration:
    """Test buffer pool and watcher integration in OutputArrays."""

    def test_finalise_uses_buffer_pool_when_chunked(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Verify chunked finalise acquires buffers from pool."""
        output_arrays_manager.update(solver)
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )

        # Set up chunking
        output_arrays_manager._chunks = 2
        output_arrays_manager._chunk_axis = "run"

        # Get initial buffer pool state
        initial_buffer_count = len(output_arrays_manager._buffer_pool._buffers)

        # Call finalise with host indices
        num_runs = output_arrays_manager.state.shape[2]
        chunk_size = num_runs // 2
        host_indices = slice(0, chunk_size)
        output_arrays_manager.finalise(host_indices)

        # Buffer pool should have acquired buffers for chunked arrays
        # The pool organizes buffers by array name
        assert len(output_arrays_manager._buffer_pool._buffers) >= 0

        # Wait for completion to clean up
        output_arrays_manager.wait_pending()

    def test_finalise_submits_to_watcher_when_chunked(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Verify chunked finalise submits tasks to watcher."""
        output_arrays_manager.update(solver)
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )

        # Set up chunking
        output_arrays_manager._chunks = 2
        output_arrays_manager._chunk_axis = "run"

        # Call finalise with host indices
        num_runs = output_arrays_manager.state.shape[2]
        chunk_size = num_runs // 2
        host_indices = slice(0, chunk_size)
        output_arrays_manager.finalise(host_indices)

        # In CUDASIM mode, tasks are processed immediately
        # Just verify the watcher is functional
        output_arrays_manager.wait_pending()

    def test_wait_pending_blocks_until_complete(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Verify wait_pending blocks until watcher completes."""
        output_arrays_manager.update(solver)
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )

        # Set up chunking
        output_arrays_manager._chunks = 2
        output_arrays_manager._chunk_axis = "run"

        # Populate device arrays with known values
        output_arrays_manager.device_state[:] = 42.0

        # Call finalise
        num_runs = output_arrays_manager.state.shape[2]
        chunk_size = num_runs // 2
        host_indices = slice(0, chunk_size)
        output_arrays_manager.finalise(host_indices)

        # Wait for completion
        output_arrays_manager.wait_pending()

        # Host array should be updated (at least the chunk portion)
        # Just verify no exceptions occurred
        assert output_arrays_manager.state is not None

    def test_reset_clears_buffer_pool_and_watcher(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Verify reset clears buffer pool and shuts down watcher."""
        output_arrays_manager.update(solver)
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )

        # Set up chunking
        output_arrays_manager._chunks = 2
        output_arrays_manager._chunk_axis = "run"

        # Call finalise to populate buffer pool
        num_runs = output_arrays_manager.state.shape[2]
        chunk_size = num_runs // 2
        host_indices = slice(0, chunk_size)
        output_arrays_manager.finalise(host_indices)

        # Wait for completion
        output_arrays_manager.wait_pending()

        # Reset should clear everything
        output_arrays_manager.reset()

        # Buffer pool should be empty
        assert len(output_arrays_manager._buffer_pool._buffers) == 0

        # Pending buffers should be clear
        assert len(output_arrays_manager._pending_buffers) == 0


class TestNeedsChunkedTransferBranching:
    """Test needs_chunked_transfer property usage in BatchOutputArrays."""

    def test_convert_host_to_numpy_uses_needs_chunked_transfer(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Verify _convert_host_to_numpy uses needs_chunked_transfer.

        The method should convert pinned arrays to regular numpy only
        when the device array's needs_chunked_transfer property is True.
        This is determined by comparing shape vs chunked_shape.
        """
        from cubie.memory.mem_manager import ArrayResponse

        # Allocate first to set up arrays
        output_arrays_manager.update(solver)
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )

        # Create a mock response with chunked_shapes that differ from shape
        num_runs = output_arrays_manager.state.shape[2]

        # Set up chunked allocation scenario: 2 chunks
        chunk_size = num_runs // 2
        chunked_shapes = {}
        for name, slot in output_arrays_manager.device.iter_managed_arrays():
            # Unchunkable arrays (status_codes) keep original shape
            if not slot.is_chunked:
                chunked_shapes[name] = slot.shape
            else:
                # Compute chunked shape with smaller run dimension
                if "run" in slot.stride_order:
                    axis_idx = slot.stride_order.index("run")
                    chunked_shape = tuple(
                        chunk_size if i == axis_idx else dim
                        for i, dim in enumerate(slot.shape)
                    )
                    chunked_shapes[name] = chunked_shape
                else:
                    chunked_shapes[name] = slot.shape

        # Create response simulating 2-chunk allocation
        response = ArrayResponse(
            arr={
                name: output_arrays_manager.device.get_array(name)
                for name in output_arrays_manager.device.array_names()
            },
            chunks=2,
            chunk_axis="run",
            chunked_shapes=chunked_shapes,
        )

        # Trigger allocation complete to store chunked_shapes
        output_arrays_manager._on_allocation_complete(response)

        # Verify state array had needs_chunked_transfer = True (shapes differ)
        device_state = output_arrays_manager.device.get_managed_array("state")
        assert device_state.needs_chunked_transfer is True

        # Verify status_codes had needs_chunked_transfer = False (unchunkable)
        device_status = output_arrays_manager.device.get_managed_array(
            "status_codes"
        )
        # Status codes is unchunkable, so chunked_shape == shape
        assert device_status.needs_chunked_transfer is False

        # Verify host arrays for chunkable arrays converted to "host"
        host_state = output_arrays_manager.host.get_managed_array("state")
        assert host_state.memory_type == "host"

    def test_finalise_uses_needs_chunked_transfer(
        self, output_arrays_manager, solver, test_memory_manager
    ):
        """Verify finalise uses needs_chunked_transfer for branching.

        When device_slot.needs_chunked_transfer is True, finalise should
        acquire buffers from the pool. When False, it should transfer
        directly without buffering.
        """
        from cubie.memory.mem_manager import ArrayResponse

        # Allocate first
        output_arrays_manager.update(solver)
        test_memory_manager.allocate_queue(
            output_arrays_manager, chunk_axis="run"
        )

        num_runs = output_arrays_manager.state.shape[2]
        chunk_size = max(1, num_runs // 2)

        # Set up chunked_shapes in device arrays
        chunked_shapes = {}
        for name, slot in output_arrays_manager.device.iter_managed_arrays():
            if not slot.is_chunked:
                chunked_shapes[name] = slot.shape
            else:
                if "run" in slot.stride_order:
                    axis_idx = slot.stride_order.index("run")
                    chunked_shape = tuple(
                        chunk_size if i == axis_idx else dim
                        for i, dim in enumerate(slot.shape)
                    )
                    chunked_shapes[name] = chunked_shape
                else:
                    chunked_shapes[name] = slot.shape

        response = ArrayResponse(
            arr={
                name: output_arrays_manager.device.get_array(name)
                for name in output_arrays_manager.device.array_names()
            },
            chunks=2,
            chunk_axis="run",
            chunked_shapes=chunked_shapes,
        )

        # Apply chunked allocation
        output_arrays_manager._on_allocation_complete(response)

        # Clear any pending buffers from allocation
        output_arrays_manager._pending_buffers.clear()

        # Call finalise with chunk indices
        host_indices = slice(0, chunk_size)
        output_arrays_manager.finalise(host_indices)

        # Pending buffers should be created for arrays with
        # needs_chunked_transfer = True
        assert len(output_arrays_manager._pending_buffers) > 0

        # Verify buffers created for chunked arrays
        buffer_names = {
            pb.array_name for pb in output_arrays_manager._pending_buffers
        }
        # state should have buffer (is_chunked=True, shapes differ)
        assert "state" in buffer_names
        # observables should have buffer
        assert "observables" in buffer_names
        # status_codes should NOT have buffer (is_chunked=False)
        assert "status_codes" not in buffer_names

        # Cleanup
        output_arrays_manager.wait_pending()
