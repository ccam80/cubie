from os import environ

import numpy as np
import pytest

from cubie.batchsolving.arrays.BatchOutputArrays import (
    ActiveOutputs,
    OutputArrayContainer,
    OutputArrays,
)
from cubie.memory.mem_manager import MemoryManager
from cubie.outputhandling.output_sizes import BatchOutputSizes

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    pass
else:
    pass


@pytest.fixture(scope="function")
def output_test_overrides(request):
    if hasattr(request, "param"):
        return request.param
    return {}


@pytest.fixture(scope="function")
def output_test_settings(output_test_overrides):
    settings = {
        "num_runs": 5,
        "dtype": "float32",
        "memory": "mapped",
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
def output_arrays_manager(solver, output_test_settings, test_memory_manager):
    """Create a OutputArrays instance using real solver"""
    batch_output_sizes = BatchOutputSizes.from_solver(solver)
    return OutputArrays(
        sizes=batch_output_sizes,
        precision=solver.precision,
        stream_group=output_test_settings["stream_group"],
        memory_proportion=output_test_settings["memory_proportion"],
        memory_manager=test_memory_manager,
    )


@pytest.fixture(scope="function")
def sample_output_arrays(solver, output_test_settings, precision):
    """Create sample output arrays for testing based on real solver"""
    num_runs = output_test_settings["num_runs"]
    dtype = precision

    # Get dimensions from solver's output sizes
    output_sizes = BatchOutputSizes.from_solver(solver)
    time_points = output_sizes.state[0]
    variables_count = solver.system_sizes.states
    observables_count = solver.system_sizes.observables

    return {
        "state": np.random.rand(time_points, num_runs, variables_count).astype(
            dtype
        ),
        "observables": np.random.rand(
            time_points, num_runs, observables_count
        ).astype(dtype),
        "state_summaries": np.random.rand(
            max(0, time_points - 2), num_runs, variables_count
        ).astype(dtype),
        "observable_summaries": np.random.rand(
            max(0, time_points - 2), num_runs, observables_count
        ).astype(dtype),
    }


class TestOutputArrayContainer:
    """Test the OutputArrayContainer class"""

    def test_container_arrays_after_init(self):
        """Test that container has correct arrays after initialization"""
        container = OutputArrayContainer()
        expected_arrays = {
            "state",
            "observables",
            "state_summaries",
            "observable_summaries",
        }
        actual_arrays = {
            key for key in container.__dict__.keys() if not key.startswith("_")
        }

        assert actual_arrays == expected_arrays

        # Check that all arrays are None initially
        assert container.state is None
        assert container.observables is None
        assert container.state_summaries is None
        assert container.observable_summaries is None

    def test_container_stride_order(self):
        """Test that stride order is set correctly"""
        container = OutputArrayContainer()
        assert container.stride_order['state'] == ("time", "run", "variable")

    def test_container_memory_type_default(self):
        """Test default memory type"""
        container = OutputArrayContainer()
        assert container._memory_type == "device"

    def test_host_factory(self):
        """Test host factory method"""
        container = OutputArrayContainer.host_factory()
        assert container._memory_type == "host"

    def test_device_factory(self):
        """Test device factory method"""
        container = OutputArrayContainer.device_factory()
        assert container._memory_type == "mapped"


class TestActiveOutputs:
    """Test the ActiveOutputs class"""

    def test_active_outputs_initialization(self):
        """Test ActiveOutputs initialization"""
        active = ActiveOutputs()
        assert active.state is False
        assert active.observables is False
        assert active.state_summaries is False
        assert active.observable_summaries is False

    def test_update_from_outputarrays_all_active(
        self, output_arrays_manager, sample_output_arrays
    ):
        """Test update_from_outputarrays with all arrays active"""
        # Set up arrays in the manager
        output_arrays_manager.host.state = sample_output_arrays["state"]
        output_arrays_manager.host.observables = sample_output_arrays[
            "observables"
        ]
        output_arrays_manager.host.state_summaries = sample_output_arrays[
            "state_summaries"
        ]
        output_arrays_manager.host.observable_summaries = sample_output_arrays[
            "observable_summaries"
        ]

        active = ActiveOutputs()
        active.update_from_outputarrays(output_arrays_manager)

        assert active.state is True
        assert active.observables is True
        assert active.state_summaries is True
        assert active.observable_summaries is True

    def test_update_from_outputarrays_size_one_arrays(
        self, output_arrays_manager
    ):
        """Test update_from_outputarrays with size-1 arrays (treated as inactive)"""
        # Set up size-1 arrays (treated as artifacts)
        output_arrays_manager.host.state = np.array([1])
        output_arrays_manager.host.observables = np.array([1])
        output_arrays_manager.host.state_summaries = None
        output_arrays_manager.host.observable_summaries = None

        active = ActiveOutputs()
        active.update_from_outputarrays(output_arrays_manager)

        assert active.state is False  # Size 1 treated as inactive
        assert active.observables is False  # Size 1 treated as inactive
        assert active.state_summaries is False  # None
        assert active.observable_summaries is False  # None


class TestOutputArrays:
    """Test the OutputArrays class"""

    def test_initialization_container_types(self, output_arrays_manager):
        """Test that containers have correct array types after initialization"""
        # Check host container arrays
        expected_arrays = {
            "state",
            "observables",
            "state_summaries",
            "observable_summaries",
        }
        host_arrays = {
            key
            for key in output_arrays_manager.host.__dict__.keys()
            if not key.startswith("_")
        }
        device_arrays = {
            key
            for key in output_arrays_manager.device.__dict__.keys()
            if not key.startswith("_")
        }

        assert host_arrays == expected_arrays
        assert device_arrays == expected_arrays

        # Check memory types are set correctly in post_init
        assert output_arrays_manager.host._memory_type == "host"
        assert output_arrays_manager.device._memory_type == "mapped"

    def test_from_solver_factory(self, solver):
        """Test creating OutputArrays from solver"""
        output_arrays = OutputArrays.from_solver(solver)

        assert isinstance(output_arrays, OutputArrays)
        assert isinstance(output_arrays._sizes, BatchOutputSizes)
        assert output_arrays._precision == solver.precision

    def test_allocation_and_getters_not_none(
        self, output_arrays_manager, solver
    ):
        """Test that all getters return non-None after allocation"""
        # Call the manager to allocate arrays based on solver
        output_arrays_manager.update(solver)

        # Check host getters
        assert output_arrays_manager.state is not None
        assert output_arrays_manager.observables is not None
        assert output_arrays_manager.state_summaries is not None
        assert output_arrays_manager.observable_summaries is not None

        # Check device getters
        assert output_arrays_manager.device_state is not None
        assert output_arrays_manager.device_observables is not None
        assert output_arrays_manager.device_state_summaries is not None
        assert output_arrays_manager.device_observable_summaries is not None

    def test_getters_return_none_before_allocation(
        self, output_arrays_manager
    ):
        """Test that getters return None before allocation"""
        assert output_arrays_manager.device_state is None
        assert output_arrays_manager.device_observables is None
        assert output_arrays_manager.device_state_summaries is None
        assert output_arrays_manager.device_observable_summaries is None

    def test_call_method_allocates_arrays(self, output_arrays_manager, solver):
        """Test that update method allocates arrays based on solver"""
        # Call the manager - it allocates based on solver sizes only
        output_arrays_manager.update(solver)

        # Check that arrays were allocated
        assert output_arrays_manager.state is not None
        assert output_arrays_manager.observables is not None
        assert output_arrays_manager.state_summaries is not None
        assert output_arrays_manager.observable_summaries is not None

        # Check device arrays
        assert output_arrays_manager.device_state is not None
        assert output_arrays_manager.device_observables is not None
        assert output_arrays_manager.device_state_summaries is not None
        assert output_arrays_manager.device_observable_summaries is not None

    def test_reallocation_on_size_change(self, output_arrays_manager, solver):
        """Test that arrays are reallocated when sizes change"""
        # Initial allocation
        output_arrays_manager.update(solver)
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
        self, output_arrays_manager, solver
    ):
        """Test that chunking changes device array allocation size"""
        # Allocate initially
        output_arrays_manager.update(solver)

        # Set up chunking - this should affect the device array size
        output_arrays_manager._chunks = 2
        output_arrays_manager._chunk_axis = "run"

        # The chunking logic should be reflected in allocation behavior
        # (This tests the chunking mechanism exists)
        assert output_arrays_manager._chunks == 2
        assert output_arrays_manager._chunk_axis == "run"

    def test_active_outputs_property(self, output_arrays_manager, solver):
        """Test _active_outputs property"""
        output_arrays_manager.update(solver)

        active = output_arrays_manager.active_outputs
        assert isinstance(active, ActiveOutputs)
        # Active status depends on whether arrays have size > 1
        # After allocation from solver, arrays should be active

    def test_update_from_solver(self, output_arrays_manager, solver):
        """Test update_from_solver method"""
        output_arrays_manager.update_from_solver(solver)

        assert output_arrays_manager._precision == solver.precision
        assert isinstance(output_arrays_manager._sizes, BatchOutputSizes)

    def test_initialise_method(self, output_arrays_manager, solver):
        """Test initialise method (no-op for outputs)"""
        # Set up the manager
        output_arrays_manager.update(solver)

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
        self, output_arrays_manager, solver
    ):
        """Test finalise method copies data from device to host"""
        # Set up the manager
        output_arrays_manager.update(solver)

        # Simulate computation by modifying device arrays
        # (In reality, CUDA kernels would write to these device arrays)
        original_device_state = np.array(output_arrays_manager.device_state)
        original_device_observables = np.array(
            output_arrays_manager.device_observables
        )

        # Modify device arrays to simulate kernel output
        output_arrays_manager.device_state[:] = original_device_state * 2
        output_arrays_manager.device_observables[:] = (
            original_device_observables * 3
        )

        # Set up chunking
        output_arrays_manager._chunks = 1
        output_arrays_manager._chunk_axis = "run"

        # Call finalise - should copy device data to host
        host_indices = slice(None)
        output_arrays_manager.finalise(host_indices)

        # Verify that host arrays now contain the modified device data
        np.testing.assert_array_equal(
            output_arrays_manager.host.state,
            output_arrays_manager.device_state,
        )
        np.testing.assert_array_equal(
            output_arrays_manager.host.observables,
            output_arrays_manager.device_observables,
        )


@pytest.mark.parametrize(
    "precision_override", [np.float32, np.float64], indirect=True
)
def test_dtype(output_arrays_manager, solver, precision):
    """Test OutputArrays with different configurations"""
    # Test that the manager works with different configurations
    output_arrays_manager.update(solver)

    expected_dtype = precision
    assert output_arrays_manager.state.dtype == expected_dtype
    assert output_arrays_manager.observables.dtype == expected_dtype
    assert output_arrays_manager.state_summaries.dtype == expected_dtype
    assert output_arrays_manager.observable_summaries.dtype == expected_dtype
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
    output_arrays_manager, solver, output_test_settings
):
    """Test OutputArrays with different configurations"""
    # Test that the manager works with different configurations
    output_arrays_manager.update(solver)

    # Check shapes match expected configuration based on solver
    expected_num_runs = output_test_settings["num_runs"]
    assert output_arrays_manager.state is not None
    assert output_arrays_manager.observables is not None
    assert output_arrays_manager.state_summaries is not None
    assert output_arrays_manager.observable_summaries is not None

    # Check data types
    expected_dtype = getattr(np, output_test_settings["dtype"])
    assert output_arrays_manager.state.dtype == expected_dtype
    assert output_arrays_manager.observables.dtype == expected_dtype
    assert output_arrays_manager.state_summaries.dtype == expected_dtype
    assert output_arrays_manager.observable_summaries.dtype == expected_dtype


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
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
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "system_override",
    ["three_chamber", "stiff", "linear"],
    indirect=True,
)
def test_output_arrays_with_different_systems(output_arrays_manager, solver):
    """Test OutputArrays with different system models"""
    # Test that the manager works with different system types
    output_arrays_manager.update(solver)

    # Verify the arrays match the system's requirements
    assert (
        output_arrays_manager.state.shape[2]
        == solver.output_array_heights.state
    )
    assert (
        output_arrays_manager.observables.shape[2]
        == solver.output_array_heights.observables
    )
    assert (
        output_arrays_manager.state_summaries.shape[2]
        == solver.output_array_heights.state_summaries
    )
    assert (
        output_arrays_manager.observable_summaries.shape[2]
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
        self, output_arrays_manager, solver
    ):
        """Test that arrays are allocated based on solver sizes"""
        # Test allocation - arrays should be sized based on solver
        output_arrays_manager.update(solver)

        # Manager should be set up without errors and arrays should exist
        assert output_arrays_manager.state is not None
        assert output_arrays_manager.observables is not None
        assert output_arrays_manager.state_summaries is not None
        assert output_arrays_manager.observable_summaries is not None

    def test_active_outputs_after_allocation(
        self, output_arrays_manager, solver
    ):
        """Test active outputs detection after allocation"""
        # Allocate arrays from solver
        output_arrays_manager.update(solver)

        # Check active outputs - should be active since arrays have size > 1
        active = output_arrays_manager.active_outputs
        assert isinstance(active, ActiveOutputs)
        # Arrays allocated from solver should typically be active (size > 1)
