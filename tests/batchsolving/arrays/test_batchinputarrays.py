
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cubie.batchsolving.arrays.BatchInputArrays import (
    InputArrayContainer,
    InputArrays,
)
from cubie.memory import default_memmgr
from cubie.memory.chunk_buffer_pool import ChunkBufferPool, PinnedBuffer
from cubie.outputhandling.output_sizes import BatchInputSizes


@pytest.fixture(scope="session")
def input_test_overrides(request):
    if hasattr(request, "param"):
        return request.param
    return {}


@pytest.fixture(scope="session")
def input_test_settings(input_test_overrides):
    settings = {
        "num_runs": 5,
        "dtype": np.float32,
        "memory": "device",
        "stream_group": "default",
        "memory_proportion": None,
    }
    settings.update(input_test_overrides)
    return settings


@pytest.fixture(scope="session")
def input_arrays_manager(precision, solver, input_test_settings):
    """Create a InputArrays instance using real solver"""
    batch_input_sizes = BatchInputSizes.from_solver(solver)
    return InputArrays(
        sizes=batch_input_sizes,
        precision=precision,
        stream_group=input_test_settings["stream_group"],
        memory_proportion=input_test_settings["memory_proportion"],
        memory_manager=default_memmgr,
    )


@pytest.fixture(scope="session")
def sample_input_arrays(solver, input_test_settings, precision):
    """Create sample input arrays for testing based on real solver.
    
    Arrays are created in native (variable, run) format matching the internal
    representation used by the solver. This format has run in the rightmost
    dimension for CUDA memory coalescing.
    """
    num_runs = input_test_settings["num_runs"]
    dtype = precision

    variables_count = solver.system_sizes.states
    parameters_count = solver.system_sizes.parameters
    forcing_count = solver.system_sizes.drivers

    # Native format: (variable, run) - run in rightmost dimension
    return {
        "initial_values": np.random.rand(variables_count, num_runs).astype(
            dtype
        ),
        "parameters": np.random.rand(parameters_count, num_runs).astype(dtype),
        "driver_coefficients": np.random.rand(forcing_count, num_runs).astype(
            dtype
        ),
    }


class TestInputArrayContainer:
    """Test the InputArrayContainer class"""

    def test_container_arrays_after_init(self):
        """Test that container has correct arrays after initialization"""
        container = InputArrayContainer()
        expected_arrays = {"driver_coefficients", "parameters",
                           "initial_values"}
        assert set(container.array_names()) == expected_arrays

        # Check that all arrays are size-1 zeros initially
        for _, managed in container.iter_managed_arrays():
            assert_array_equal(
                    managed.array,
                    np.zeros(managed.shape, dtype=managed.dtype)
            )

    def test_container_stride_order(self):
        """Test that stride order is set correctly"""
        container = InputArrayContainer()
        stride_order = container.parameters.stride_order
        assert stride_order == ("variable", "run")

    def test_host_factory(self):
        """Test host factory method creates pinned memory container"""
        container = InputArrayContainer.host_factory()
        assert container.get_managed_array("initial_values").memory_type == "pinned"

    def test_device_factory(self):
        """Test device factory method"""
        container = InputArrayContainer.device_factory()
        assert container.get_managed_array("initial_values").memory_type == "device"


class TestInputArrays:
    """Test the InputArrays class"""

    def test_initialization_container_types(self, input_arrays_manager):
        """Test that containers have correct array types after initialization"""
        # Check host container arrays
        expected_arrays = {"initial_values", "parameters", "driver_coefficients"}
        assert set(input_arrays_manager.host.array_names()) == expected_arrays
        assert set(input_arrays_manager.device.array_names()) == expected_arrays

        # Check memory types are set correctly in post_init
        for _, managed in input_arrays_manager.host.iter_managed_arrays():
            assert managed.memory_type == "pinned"
        for _, managed in input_arrays_manager.device.iter_managed_arrays():
            assert managed.memory_type == "device"

    def test_from_solver_factory(self, solver):
        """Test creating InputArrays from solver"""
        input_arrays = InputArrays.from_solver(solver)

        assert isinstance(input_arrays, InputArrays)
        assert isinstance(input_arrays._sizes, BatchInputSizes)
        assert input_arrays._precision == solver.precision

    def test_allocation_and_getters_not_none(
        self, input_arrays_manager, solver, sample_input_arrays
    ):
        """Test that all getters return non-None after allocation"""
        # Call the manager to set up arrays and allocate
        # solver.numruns=sample_input_arrays['initial_values'].shape[1]
        input_arrays_manager.update(
            solver,
            sample_input_arrays["initial_values"],
            sample_input_arrays["parameters"],
            sample_input_arrays["driver_coefficients"],
        )
        # Process the allocation queue to create device arrays
        default_memmgr.allocate_queue(input_arrays_manager, chunk_axis="run")

        # Check host getters
        assert input_arrays_manager.initial_values is not None
        assert input_arrays_manager.parameters is not None
        assert input_arrays_manager.driver_coefficients is not None

        # Check device getters
        assert input_arrays_manager.device_initial_values is not None
        assert input_arrays_manager.device_parameters is not None
        assert input_arrays_manager.device_driver_coefficients is not None

    def test_call_method_updates_host_arrays(
        self, input_arrays_manager, solver, sample_input_arrays
    ):
        """Test that update method updates host arrays"""
        input_arrays_manager.update(
            solver,
            sample_input_arrays["initial_values"],
            sample_input_arrays["parameters"],
            sample_input_arrays["driver_coefficients"],
        )

        # Check that host arrays were updated
        # Arrays are in native (variable, run) format - no transpose needed
        assert_array_equal(
            input_arrays_manager.initial_values,
            sample_input_arrays["initial_values"],
        )
        assert_array_equal(
            input_arrays_manager.parameters, sample_input_arrays["parameters"]
        )
        assert_array_equal(
            input_arrays_manager.driver_coefficients,
            sample_input_arrays["driver_coefficients"],
        )

    def test_call_method_size_change_triggers_reallocation(
        self, input_arrays_manager, solver, input_test_settings
    ):
        """Test that update method triggers reallocation when size changes"""
        dtype = input_test_settings["dtype"]
        num_runs = input_test_settings["num_runs"]

        variables_count = solver.system_sizes.states
        parameters_count = solver.system_sizes.parameters
        forcing_count = solver.system_sizes.drivers

        # Initial call with original sizes
        # Native format: (variable, run)
        initial_arrays = {
            "initial_values": np.random.rand(variables_count, num_runs).astype(
                dtype
            ),
            "parameters": np.random.rand(parameters_count, num_runs).astype(
                dtype
            ),
            "driver_coefficients": np.random.rand(forcing_count, num_runs).astype(
                dtype
            ),
        }

        input_arrays_manager.update(
            solver,
            initial_arrays["initial_values"],
            initial_arrays["parameters"],
            initial_arrays["driver_coefficients"],
        )
        # Process the allocation queue to create device arrays
        default_memmgr.allocate_queue(input_arrays_manager, chunk_axis="run")

        original_device_initial_values = (
            input_arrays_manager.device_initial_values
        )

        # Call with different sized arrays (more runs)
        new_num_runs = num_runs + 2
        # Native format: (variable, run)
        new_arrays = {
            "initial_values": np.random.rand(
                variables_count, new_num_runs
            ).astype(dtype),
            "parameters": np.random.rand(
                parameters_count, new_num_runs
            ).astype(dtype),
            "driver_coefficients": np.random.rand(
                forcing_count, new_num_runs
            ).astype(dtype),
        }

        input_arrays_manager.update(
            solver,
            new_arrays["initial_values"],
            new_arrays["parameters"],
            new_arrays["driver_coefficients"],
        )
        # Process the allocation queue after size change
        default_memmgr.allocate_queue(input_arrays_manager, chunk_axis="run")

        # Should have triggered reallocation for all arrays
        assert (
            input_arrays_manager.device_initial_values
            is not original_device_initial_values
        )
        # Native format is (variable, run)
        assert input_arrays_manager.device_initial_values.shape == (
            variables_count,
            new_num_runs,
        )

    def test_update_from_solver(self, input_arrays_manager, solver):
        """Test update_from_solver method"""
        input_arrays_manager.update_from_solver(solver)

        assert input_arrays_manager._precision == solver.precision
        assert isinstance(input_arrays_manager._sizes, BatchInputSizes)

    def test_initialise_method(
        self, input_arrays_manager, solver, sample_input_arrays
    ):
        """Test initialise method copies data to device"""
        # Set up the manager
        input_arrays_manager.update(
            solver,
            sample_input_arrays["initial_values"],
            sample_input_arrays["parameters"],
            sample_input_arrays["driver_coefficients"],
        )
        # Process the allocation queue to create device arrays
        default_memmgr.allocate_queue(input_arrays_manager, chunk_axis="run")

        # Clear device arrays to test initialise
        input_arrays_manager.device.initial_values.array[:, :] = 0.0
        input_arrays_manager.device.parameters.array[:, :] = 0.0
        input_arrays_manager.device.driver_coefficients.array[:, :] = 0.0

        # Set up chunking
        input_arrays_manager._chunks = 1
        input_arrays_manager._chunk_axis = "run"

        # Call initialise with host indices (all data)
        host_indices = slice(None)
        input_arrays_manager.initialise(host_indices)

        # Check that device arrays now match host arrays
        # Arrays are in native (variable, run) format - no transpose needed
        np.testing.assert_array_equal(
            np.array(input_arrays_manager.device.initial_values.array),
            sample_input_arrays["initial_values"],
        )
        np.testing.assert_array_equal(
            np.array(input_arrays_manager.device.parameters.array),
            sample_input_arrays["parameters"],
        )
        np.testing.assert_array_equal(
            np.array(input_arrays_manager.device.driver_coefficients.array),
            sample_input_arrays["driver_coefficients"],
        )

    # Implementation removed while issue #76 incomplete
    # def test_finalise_method(self, solver, sample_input_arrays):
    #     """Test finalise method copies data from device"""
    #     # Set up the manager
    #     input_arrays_manager = InputArrays.from_solver(solver)
    #     input_arrays_manager.update(
    #         solver,
    #         sample_input_arrays["initial_values"],
    #         sample_input_arrays["parameters"],
    #         sample_input_arrays["driver_coefficients"],
    #     )
    #     solver.memory_manager.allocate_queue(input_arrays_manager)
    #     # Modify device initial_values (simulate computation results)
    #     modified_values = (
    #         np.array(input_arrays_manager.device.initial_values.array.copy_to_host())
    #         * 2
    #     )
    #     cuda.to_device(
    #         modified_values, to=input_arrays_manager.device.initial_values.array
    #     )
    #
    #     # Set up chunking
    #     input_arrays_manager._chunks = 1
    #     input_arrays_manager._chunk_axis = "run"
    #
    #     # Store original host values
    #     original_host_values = input_arrays_manager.host.initial_values.array.copy()
    #
    #     # Call finalise with host indices (all data)
    #     host_indices = slice(None)
    #     input_arrays_manager.finalise(host_indices)
    #
    #     # Check that host initial_values were updated with device values
    #     # np.testing.assert_array_equal(
    #     #     input_arrays_manager.host.initial_values.array, modified_values
    #     # )
    #
    #     # Verify it actually changed from original
    #     assert not np.array_equal(
    #         input_arrays_manager.host.initial_values.array, original_host_values
    #     )

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"precision": np.float32},
         {"precision": np.float64}],
    indirect=True
    )
    def test_dtype(
        self, input_arrays_manager, solver, sample_input_arrays, precision
    ):
        """Test finalise method copies data from device"""
        # Set up the manager
        input_arrays_manager.update(
            solver,
            sample_input_arrays["initial_values"],
            sample_input_arrays["parameters"],
            sample_input_arrays["driver_coefficients"],
        )
        # Process the allocation queue to create device arrays
        default_memmgr.allocate_queue(input_arrays_manager, chunk_axis="run")

        expected_dtype = precision
        assert input_arrays_manager.initial_values.dtype.type == expected_dtype
        assert input_arrays_manager.parameters.dtype.type == expected_dtype
        assert (
            input_arrays_manager.driver_coefficients.dtype.type == expected_dtype
        )
        assert (
            input_arrays_manager.device_initial_values.dtype.type == expected_dtype
        )
        assert (
            input_arrays_manager.device_parameters.dtype.type == expected_dtype
        )
        assert (
            input_arrays_manager.device_driver_coefficients.dtype.type
            == expected_dtype
        )


# Parametrized tests for different configurations
@pytest.mark.parametrize(
    "input_test_overrides",
    [
        {"num_runs": 10},
        {"num_runs": 3},
        {"stream_group": "test_group", "memory_proportion": 0.5},
    ],
    indirect=True,
)
def test_input_arrays_with_different_configs(
    input_arrays_manager, solver, sample_input_arrays, input_test_settings
):
    """Test InputArrays with different configurations"""
    # Test that the manager works with different configurations
    input_arrays_manager.update(
        solver,
        sample_input_arrays["initial_values"],
        sample_input_arrays["parameters"],
        sample_input_arrays["driver_coefficients"],
    )

    # Check shapes match expected configuration
    expected_num_runs = input_test_settings["num_runs"]
    assert input_arrays_manager.initial_values.shape[1] == expected_num_runs
    assert input_arrays_manager.parameters.shape[1] == expected_num_runs
    assert input_arrays_manager.driver_coefficients.shape[1] == expected_num_runs

    # Check data types
    expected_dtype = input_test_settings["dtype"]
    assert input_arrays_manager.initial_values.dtype.type == expected_dtype
    assert input_arrays_manager.parameters.dtype.type == expected_dtype
    assert (
        input_arrays_manager.driver_coefficients.dtype.type == expected_dtype
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "three_chamber"},
     {"system_type":"stiff"},
     {"system_type":"linear"}],
    indirect=True,
)
def test_input_arrays_with_different_systems(
    input_arrays_manager, solver, sample_input_arrays
):
    """Test InputArrays with different system models"""
    # Test that the manager works with different system types
    input_arrays_manager.update(
        solver,
        sample_input_arrays["initial_values"],
        sample_input_arrays["parameters"],
        sample_input_arrays["driver_coefficients"],
    )
    # Process the allocation queue to create device arrays
    default_memmgr.allocate_queue(input_arrays_manager, chunk_axis="run")

    # Verify the arrays match the system's requirements
    assert (
        input_arrays_manager.initial_values.shape[0]
        == solver.system_sizes.states
    )
    assert (
        input_arrays_manager.parameters.shape[0]
        == solver.system_sizes.parameters
    )
    assert (
        input_arrays_manager.driver_coefficients.shape[0]
        == solver.system_sizes.drivers
    )

    # Check that all getters work
    assert input_arrays_manager.initial_values is not None
    assert input_arrays_manager.parameters is not None
    assert input_arrays_manager.driver_coefficients is not None
    assert input_arrays_manager.device_initial_values is not None
    assert input_arrays_manager.device_parameters is not None
    assert input_arrays_manager.device_driver_coefficients is not None


class TestBufferPoolIntegration:
    """Test buffer pool integration for chunked mode."""

    def test_input_arrays_has_buffer_pool(self, input_arrays_manager):
        """Verify InputArrays has a ChunkBufferPool attribute."""
        assert hasattr(input_arrays_manager, '_buffer_pool')
        assert isinstance(input_arrays_manager._buffer_pool, ChunkBufferPool)

    def test_input_arrays_has_active_buffers(self, input_arrays_manager):
        """Verify InputArrays has an _active_buffers list."""
        assert hasattr(input_arrays_manager, '_active_buffers')
        assert isinstance(input_arrays_manager._active_buffers, list)

    def test_initialise_uses_buffer_pool_when_chunked(
        self, solver, sample_input_arrays, precision
    ):
        """Verify chunked initialise acquires buffers from pool."""
        input_arrays = InputArrays.from_solver(solver)
        input_arrays.update(
            solver,
            sample_input_arrays["initial_values"],
            sample_input_arrays["parameters"],
            sample_input_arrays["driver_coefficients"],
        )
        # Allocate device arrays via memory manager
        default_memmgr.allocate_queue(input_arrays, chunk_axis="run")

        # Configure for chunked mode (multiple chunks)
        input_arrays._chunks = 3
        input_arrays._chunk_axis = "run"

        # Set chunked_shape and chunked_slice_fn on BOTH host and device
        # arrays to trigger needs_chunked_transfer
        num_runs = sample_input_arrays["initial_values"].shape[1]
        chunk_size = max(1, num_runs // 3)

        def make_slice_fn(run_axis_idx, chunk_sz, ndim):
            def slice_fn(chunk_idx):
                slices = [slice(None)] * ndim
                start = chunk_idx * chunk_sz
                end = start + chunk_sz
                slices[run_axis_idx] = slice(start, end)
                return tuple(slices)
            return slice_fn

        for name, device_slot in input_arrays.device.iter_managed_arrays():
            if "run" in device_slot.stride_order:
                run_idx = device_slot.stride_order.index("run")
                chunked = list(device_slot.shape)
                chunked[run_idx] = chunk_size
                chunked_shape = tuple(chunked)
                ndim = len(device_slot.shape)
                slice_fn = make_slice_fn(run_idx, chunk_size, ndim)
                device_slot.chunked_shape = chunked_shape
                device_slot.chunked_slice_fn = slice_fn
                # Also set on corresponding host array
                host_slot = input_arrays.host.get_managed_array(name)
                host_slot.chunked_shape = chunked_shape
                host_slot.chunked_slice_fn = slice_fn

        # Clear any existing active buffers
        input_arrays._active_buffers.clear()

        # Call initialise with a chunk index
        input_arrays.initialise(0)

        # Verify buffers were acquired and stored in _active_buffers
        assert len(input_arrays._active_buffers) > 0
        for buffer in input_arrays._active_buffers:
            assert isinstance(buffer, PinnedBuffer)
            assert buffer.in_use is True

    def test_release_buffers_returns_to_pool(
        self, solver, sample_input_arrays, precision
    ):
        """Verify release_buffers returns buffers to pool."""
        input_arrays = InputArrays.from_solver(solver)
        input_arrays.update(
            solver,
            sample_input_arrays["initial_values"],
            sample_input_arrays["parameters"],
            sample_input_arrays["driver_coefficients"],
        )
        # Allocate device arrays via memory manager
        default_memmgr.allocate_queue(input_arrays, chunk_axis="run")

        # Configure for chunked mode
        input_arrays._chunks = 3
        input_arrays._chunk_axis = "run"

        # Set chunked_shape and chunked_slice_fn on BOTH host and device
        num_runs = sample_input_arrays["initial_values"].shape[1]
        chunk_size = max(1, num_runs // 3)

        def make_slice_fn(run_axis_idx, chunk_sz, ndim):
            def slice_fn(chunk_idx):
                slices = [slice(None)] * ndim
                start = chunk_idx * chunk_sz
                end = start + chunk_sz
                slices[run_axis_idx] = slice(start, end)
                return tuple(slices)
            return slice_fn

        for name, device_slot in input_arrays.device.iter_managed_arrays():
            if "run" in device_slot.stride_order:
                run_idx = device_slot.stride_order.index("run")
                chunked = list(device_slot.shape)
                chunked[run_idx] = chunk_size
                chunked_shape = tuple(chunked)
                ndim = len(device_slot.shape)
                slice_fn = make_slice_fn(run_idx, chunk_size, ndim)
                device_slot.chunked_shape = chunked_shape
                device_slot.chunked_slice_fn = slice_fn
                # Also set on corresponding host array
                host_slot = input_arrays.host.get_managed_array(name)
                host_slot.chunked_shape = chunked_shape
                host_slot.chunked_slice_fn = slice_fn

        # Call initialise to acquire buffers
        input_arrays.initialise(0)

        # Store reference to buffers before release
        buffers_before = list(input_arrays._active_buffers)
        assert len(buffers_before) > 0

        # Release buffers
        input_arrays.release_buffers()

        # Verify _active_buffers is cleared
        assert len(input_arrays._active_buffers) == 0

        # Verify buffers are marked as not in use
        for buffer in buffers_before:
            assert buffer.in_use is False

    def test_non_chunked_uses_direct_pinned(
        self, solver, sample_input_arrays, precision
    ):
        """Verify non-chunked mode does not use buffer pool."""
        input_arrays = InputArrays.from_solver(solver)
        input_arrays.update(
            solver,
            sample_input_arrays["initial_values"],
            sample_input_arrays["parameters"],
            sample_input_arrays["driver_coefficients"],
        )

        # Configure for non-chunked mode (single chunk)
        input_arrays._chunks = 1
        input_arrays._chunk_axis = "run"

        # Clear any existing active buffers
        input_arrays._active_buffers.clear()

        # Call initialise with chunk index 0 (non-chunked mode)
        input_arrays.initialise(0)

        # Verify no buffers were acquired from pool
        assert len(input_arrays._active_buffers) == 0

    def test_reset_clears_buffer_pool_and_active_buffers(
        self, solver, sample_input_arrays, precision
    ):
        """Verify reset clears buffer pool and active buffers."""
        input_arrays = InputArrays.from_solver(solver)
        input_arrays.update(
            solver,
            sample_input_arrays["initial_values"],
            sample_input_arrays["parameters"],
            sample_input_arrays["driver_coefficients"],
        )
        # Allocate device arrays via memory manager
        default_memmgr.allocate_queue(input_arrays, chunk_axis="run")

        # Configure for chunked mode and run initialise
        input_arrays._chunks = 3
        input_arrays._chunk_axis = "run"
        num_runs = sample_input_arrays["initial_values"].shape[1]
        chunk_size = max(1, num_runs // 3)

        def make_slice_fn(run_axis_idx, chunk_sz, ndim):
            def slice_fn(chunk_idx):
                slices = [slice(None)] * ndim
                start = chunk_idx * chunk_sz
                end = start + chunk_sz
                slices[run_axis_idx] = slice(start, end)
                return tuple(slices)
            return slice_fn

        # Set chunked_shape and chunked_slice_fn on BOTH host and device
        for name, device_slot in input_arrays.device.iter_managed_arrays():
            if "run" in device_slot.stride_order:
                run_idx = device_slot.stride_order.index("run")
                chunked = list(device_slot.shape)
                chunked[run_idx] = chunk_size
                chunked_shape = tuple(chunked)
                ndim = len(device_slot.shape)
                slice_fn = make_slice_fn(run_idx, chunk_size, ndim)
                device_slot.chunked_shape = chunked_shape
                device_slot.chunked_slice_fn = slice_fn
                # Also set on corresponding host array
                host_slot = input_arrays.host.get_managed_array(name)
                host_slot.chunked_shape = chunked_shape
                host_slot.chunked_slice_fn = slice_fn

        input_arrays.initialise(0)

        # Verify there are active buffers
        assert len(input_arrays._active_buffers) > 0

        # Call reset
        input_arrays.reset()

        # Verify both active buffers and pool are cleared
        assert len(input_arrays._active_buffers) == 0
        # Verify pool has no buffers after clear
        assert len(input_arrays._buffer_pool._buffers) == 0

    def test_buffers_reused_across_chunks(
        self, solver, sample_input_arrays, precision
    ):
        """Verify buffers are reused when released between chunks."""
        input_arrays = InputArrays.from_solver(solver)
        input_arrays.update(
            solver,
            sample_input_arrays["initial_values"],
            sample_input_arrays["parameters"],
            sample_input_arrays["driver_coefficients"],
        )

        # Configure for chunked mode
        input_arrays._chunks = 3
        input_arrays._chunk_axis = "run"
        num_runs = sample_input_arrays["initial_values"].shape[1]
        chunk_size = max(1, num_runs // 3)

        def make_slice_fn(run_axis_idx, chunk_sz, ndim):
            def slice_fn(chunk_idx):
                slices = [slice(None)] * ndim
                start = chunk_idx * chunk_sz
                end = start + chunk_sz
                slices[run_axis_idx] = slice(start, end)
                return tuple(slices)
            return slice_fn

        # Set chunked_shape and chunked_slice_fn on BOTH host and device
        for name, device_slot in input_arrays.device.iter_managed_arrays():
            if "run" in device_slot.stride_order:
                run_idx = device_slot.stride_order.index("run")
                chunked = list(device_slot.shape)
                chunked[run_idx] = chunk_size
                chunked_shape = tuple(chunked)
                ndim = len(device_slot.shape)
                slice_fn = make_slice_fn(run_idx, chunk_size, ndim)
                device_slot.chunked_shape = chunked_shape
                device_slot.chunked_slice_fn = slice_fn
                # Also set on corresponding host array
                host_slot = input_arrays.host.get_managed_array(name)
                host_slot.chunked_shape = chunked_shape
                host_slot.chunked_slice_fn = slice_fn

        # First chunk
        input_arrays.initialise(0)
        first_buffers = list(input_arrays._active_buffers)
        first_buffer_ids = [b.buffer_id for b in first_buffers]
        input_arrays.release_buffers()

        # Second chunk - should reuse buffers from pool
        input_arrays.initialise(1)
        second_buffers = list(input_arrays._active_buffers)
        second_buffer_ids = [b.buffer_id for b in second_buffers]

        # Buffer IDs should match, indicating reuse
        assert first_buffer_ids == second_buffer_ids


class TestNeedsChunkedTransferBranching:
    """Test that initialise uses needs_chunked_transfer for branching."""

    def test_initialise_uses_needs_chunked_transfer(
        self, solver, sample_input_arrays, precision
    ):
        """Verify initialise uses needs_chunked_transfer for branching.

        When needs_chunked_transfer is False, the array is copied directly.
        When needs_chunked_transfer is True, buffer pool is used for staging.
        """
        input_arrays = InputArrays.from_solver(solver)
        input_arrays.update(
            solver,
            sample_input_arrays["initial_values"],
            sample_input_arrays["parameters"],
            sample_input_arrays["driver_coefficients"],
        )

        # Configure for chunked mode
        input_arrays._chunks = 3
        input_arrays._chunk_axis = "run"

        # Set up chunked_shape so needs_chunked_transfer returns True
        # for arrays with run axis in stride_order
        num_runs = sample_input_arrays["initial_values"].shape[1]
        chunk_size = num_runs // 3

        for name, device_slot in input_arrays.device.iter_managed_arrays():
            # Full shape stored in shape, chunked shape set differently
            original_shape = device_slot.shape
            # If array has 'run' in stride_order, set chunked_shape smaller
            if "run" in device_slot.stride_order:
                run_idx = device_slot.stride_order.index("run")
                chunked = list(original_shape)
                chunked[run_idx] = chunk_size
                device_slot.chunked_shape = tuple(chunked)
            else:
                # Array not chunked - chunked_shape equals shape
                device_slot.chunked_shape = original_shape

        # Clear any existing active buffers
        input_arrays._active_buffers.clear()

        # Call initialise with a chunk slice
        host_indices = slice(0, chunk_size)
        input_arrays.initialise(host_indices)

        # Arrays with needs_chunked_transfer=True should have used buffers
        buffer_names = [b.name for b in input_arrays._active_buffers]

        # Check which arrays used buffer pool (needs_chunked_transfer=True)
        for name, device_slot in input_arrays.device.iter_managed_arrays():
            if device_slot.needs_chunked_transfer:
                if input_arrays._chunk_axis in device_slot.stride_order:
                    # Should have used buffer pool
                    assert name in buffer_names, (
                        f"Array {name} with needs_chunked_transfer=True "
                        "should have used buffer pool"
                    )
            else:
                # Should NOT have used buffer pool
                assert name not in buffer_names, (
                    f"Array {name} with needs_chunked_transfer=False "
                    "should not have used buffer pool"
                )

    def test_initialise_no_buffers_when_needs_chunked_transfer_false(
        self, solver, sample_input_arrays, precision
    ):
        """Verify no buffers used when all arrays have needs_chunked_transfer=False.

        When chunked_shape equals shape, needs_chunked_transfer returns False
        and no buffer pool staging is needed.
        """
        input_arrays = InputArrays.from_solver(solver)
        input_arrays.update(
            solver,
            sample_input_arrays["initial_values"],
            sample_input_arrays["parameters"],
            sample_input_arrays["driver_coefficients"],
        )

        # Configure for chunked mode but set chunked_shape = shape
        # so needs_chunked_transfer returns False
        input_arrays._chunks = 3
        input_arrays._chunk_axis = "run"

        for name, device_slot in input_arrays.device.iter_managed_arrays():
            # Set chunked_shape equal to shape
            device_slot.chunked_shape = device_slot.shape

        # Clear any existing active buffers
        input_arrays._active_buffers.clear()

        # Call initialise
        num_runs = sample_input_arrays["initial_values"].shape[1]
        chunk_size = num_runs // 3
        host_indices = slice(0, chunk_size)
        input_arrays.initialise(host_indices)

        # No buffers should be used since needs_chunked_transfer is False
        assert len(input_arrays._active_buffers) == 0, (
            "No buffers should be used when needs_chunked_transfer is False"
        )
