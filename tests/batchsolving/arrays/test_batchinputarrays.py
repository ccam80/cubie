from os import environ

import numpy as np
import pytest
from numba import cuda
from numpy.testing import assert_array_equal

from cubie.batchsolving.arrays.BatchInputArrays import (
    InputArrayContainer,
    InputArrays,
)
from cubie.memory import default_memmgr
from cubie.outputhandling.output_sizes import BatchInputSizes

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    pass
else:
    pass


@pytest.fixture(scope="function")
def input_test_overrides(request):
    if hasattr(request, "param"):
        return request.param
    return {}


@pytest.fixture(scope="function")
def input_test_settings(input_test_overrides):
    settings = {
        "num_runs": 5,
        "dtype": "float32",
        "memory": "device",
        "stream_group": "default",
        "memory_proportion": None,
    }
    settings.update(input_test_overrides)
    return settings


@pytest.fixture(scope="function")
def input_arrays_manager(solver, input_test_settings):
    """Create a InputArrays instance using real solver"""
    batch_input_sizes = BatchInputSizes.from_solver(solver)
    return InputArrays(
        sizes=batch_input_sizes,
        precision=solver.precision,
        stream_group=input_test_settings["stream_group"],
        memory_proportion=input_test_settings["memory_proportion"],
        memory_manager=default_memmgr,
    )


@pytest.fixture(scope="function")
def sample_input_arrays(solver, input_test_settings, precision):
    """Create sample input arrays for testing based on real solver"""
    num_runs = input_test_settings["num_runs"]
    dtype = precision

    variables_count = solver.system_sizes.states
    parameters_count = solver.system_sizes.parameters
    forcing_count = solver.system_sizes.drivers

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
        actual_arrays = {
            key for key in container.__dict__.keys() if not key.startswith("_")
        }

        assert actual_arrays == expected_arrays

        # Check that all arrays are None initially
        assert container.initial_values is None
        assert container.parameters is None
        assert container.driver_coefficients is None

    def test_container_stride_order(self):
        """Test that stride order is set correctly"""
        container = InputArrayContainer()
        assert container.stride_order['parameters'] == ("run", "variable")

    def test_host_factory(self):
        """Test host factory method"""
        container = InputArrayContainer.host_factory()
        assert container._memory_type == "host"

    def test_device_factory(self):
        """Test device factory method"""
        container = InputArrayContainer.device_factory()
        assert container._memory_type == "device"


class TestInputArrays:
    """Test the InputArrays class"""

    def test_initialization_container_types(self, input_arrays_manager):
        """Test that containers have correct array types after initialization"""
        # Check host container arrays
        expected_arrays = {"initial_values", "parameters", "driver_coefficients"}
        host_arrays = {
            key
            for key in input_arrays_manager.host.__dict__.keys()
            if not key.startswith("_")
        }
        device_arrays = {
            key
            for key in input_arrays_manager.device.__dict__.keys()
            if not key.startswith("_")
        }

        assert host_arrays == expected_arrays
        assert device_arrays == expected_arrays

        # Check memory types are set correctly in post_init
        assert input_arrays_manager.host._memory_type == "host"
        assert input_arrays_manager.device._memory_type == "device"

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
        dtype = getattr(np, input_test_settings["dtype"])
        num_runs = input_test_settings["num_runs"]

        variables_count = solver.system_sizes.states
        parameters_count = solver.system_sizes.parameters
        forcing_count = solver.system_sizes.drivers

        # Initial call with original sizes
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

        original_device_initial_values = (
            input_arrays_manager.device_initial_values
        )

        # Call with different sized arrays (more runs)
        new_num_runs = num_runs + 2
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

        # Should have triggered reallocation for all arrays
        assert (
            input_arrays_manager.device_initial_values
            is not original_device_initial_values
        )
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

        # Clear device arrays to test initialise
        input_arrays_manager.device.initial_values[:, :] = 0.0
        input_arrays_manager.device.parameters[:, :] = 0.0
        input_arrays_manager.device.driver_coefficients[:, :] = 0.0

        # Set up chunking
        input_arrays_manager._chunks = 1
        input_arrays_manager._chunk_axis = "run"

        # Call initialise with host indices (all data)
        host_indices = slice(None)
        input_arrays_manager.initialise(host_indices)

        # Check that device arrays now match host arrays
        np.testing.assert_array_equal(
            np.array(input_arrays_manager.device.initial_values),
            sample_input_arrays["initial_values"],
        )
        np.testing.assert_array_equal(
            np.array(input_arrays_manager.device.parameters),
            sample_input_arrays["parameters"],
        )
        np.testing.assert_array_equal(
            np.array(input_arrays_manager.device.driver_coefficients),
            sample_input_arrays["driver_coefficients"],
        )

    def test_finalise_method(self, solver, sample_input_arrays):
        """Test finalise method copies data from device"""
        # Set up the manager
        input_arrays_manager = InputArrays.from_solver(solver)
        input_arrays_manager.update(
            solver,
            sample_input_arrays["initial_values"],
            sample_input_arrays["parameters"],
            sample_input_arrays["driver_coefficients"],
        )
        solver.memory_manager.allocate_queue(input_arrays_manager)
        # Modify device initial_values (simulate computation results)
        modified_values = (
            np.array(input_arrays_manager.device.initial_values.copy_to_host())
            * 2
        )
        cuda.to_device(
            modified_values, to=input_arrays_manager.device.initial_values
        )

        # Set up chunking
        input_arrays_manager._chunks = 1
        input_arrays_manager._chunk_axis = "run"

        # Store original host values
        original_host_values = input_arrays_manager.host.initial_values.copy()

        # Call finalise with host indices (all data)
        host_indices = slice(None)
        input_arrays_manager.finalise(host_indices)

        # Check that host initial_values were updated with device values
        np.testing.assert_array_equal(
            input_arrays_manager.host.initial_values, modified_values
        )

        # Verify it actually changed from original
        assert not np.array_equal(
            input_arrays_manager.host.initial_values, original_host_values
        )

    @pytest.mark.parametrize(
        "precision_override", [np.float32, np.float64], indirect=True
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

        expected_dtype = precision
        assert input_arrays_manager.initial_values.dtype == expected_dtype
        assert input_arrays_manager.parameters.dtype == expected_dtype
        assert input_arrays_manager.driver_coefficients.dtype == expected_dtype
        assert (
            input_arrays_manager.device_initial_values.dtype == expected_dtype
        )
        assert input_arrays_manager.device_parameters.dtype == expected_dtype
        assert (
            input_arrays_manager.device_driver_coefficients.dtype == expected_dtype
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
    expected_dtype = getattr(np, input_test_settings["dtype"])
    assert input_arrays_manager.initial_values.dtype == expected_dtype
    assert input_arrays_manager.parameters.dtype == expected_dtype
    assert input_arrays_manager.driver_coefficients.dtype == expected_dtype


@pytest.mark.parametrize(
    "system_override",
    ["three_chamber", "stiff", "linear"],
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
