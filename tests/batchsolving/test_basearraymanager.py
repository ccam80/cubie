from os import environ
import pytest
from cubie.batchsolving.BaseArrayManager import BaseArrayManager, ArrayContainer
import attrs
import numpy as np

from cubie.memory.array_requests import ArrayResponse, ArrayRequest
from cubie.memory.mem_manager import MemoryManager

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from numba.cuda.simulator.cudadrv.devicearray import \
        FakeCUDAArray as DeviceNDArray
    from numba.cuda.simulator.cudadrv.devicearray import (FakeCUDAArray as
                                                          MappedNDArray)
    from numpy import zeros as mapped_array
    from numpy import zeros as device_array
else:
    from numba.cuda.cudadrv.devicearray import DeviceNDArray, MappedNDArray
    from numba.cuda import mapped_array, device_array

@attrs.define
class ConcreteArrayManager(BaseArrayManager):
    """Concrete implementation of BaseArrayManager"""
    def finalise_chunk(self, indices, axis):
        return indices, axis

    def initialise_chunk(self, indices, axis):
        return indices, axis

    def check_sizes(self, arrays, system_sizes):
        pass

@attrs.define(slots=False)
class TestArrays(ArrayContainer):
    arr1 = attrs.field(default=None)
    arr2 = attrs.field(default=None)

@pytest.fixture(scope='function')
def arraytest_overrides(request):
    if hasattr(request, 'param'):
        return request.param
    return {}

@pytest.fixture(scope="function")
def arraytest_settings(arraytest_overrides):
    settings = {
        'hostshape1': (2, 3, 4),
        'hostshape2': (2, 3, 4),
        'devshape1': (2, 3, 4),
        'devshape2': (2, 3, 4),
        'dtype': 'float32',
        'memory': 'device',
        'stride_order': ('time', 'run', 'variable'),
        'stream_group': 'default',
        'memory_proportion': None,
    }
    settings.update(arraytest_overrides)
    return settings

@pytest.fixture(scope='function')
def hostarrays(arraytest_settings):
    if arraytest_settings['memory'] == 'mapped':
        empty = mapped_array
    else:
        empty = np.zeros
    return TestArrays(
        arr1=empty(arraytest_settings['hostshape1'],
                   dtype=arraytest_settings['dtype']),
        arr2=empty(arraytest_settings['hostshape2'],
                   dtype=arraytest_settings['dtype'])
    )

@pytest.fixture(scope='function')
def devarrays(arraytest_settings, hostarrays):
    if arraytest_settings['memory'] == 'mapped':
        #Not clear how I should manage the mapped functionality, do not test
        # yet
        devarr1 = MappedNDArray(hostarrays.arr1, arraytest_settings['dtype'])
        devarr2 = MappedNDArray(hostarrays.arr2, arraytest_settings['dtype'])
    else:
        devarr1 = device_array(arraytest_settings['devshape1'],
                               arraytest_settings['dtype'])
        devarr2 = device_array(arraytest_settings['devshape2'],
                               arraytest_settings['dtype'])
    return TestArrays(arr1=devarr1, arr2=devarr2)

@pytest.fixture(scope='function')
def test_memory_manager():
    """Create an actual MemoryManager instance for testing"""
    # Create a memory manager in passive mode to avoid CUDA context issues
    mem_mgr = MemoryManager(mode="passive")
    return mem_mgr

@pytest.fixture(scope='function')
def test_arrmgr(hostarrays, devarrays, precision, arraytest_settings, test_memory_manager):
    return ConcreteArrayManager(
            precision=precision,
            host=hostarrays,
            device=devarrays,
            stream_group=arraytest_settings['stream_group'],
            memory_proportion=arraytest_settings['memory_proportion'],
            memory_manager=test_memory_manager)

@pytest.fixture(scope='function')
def allocation_response(arraytest_settings, precision):
    """Create a test ArrayResponse based on arraytest_settings for consistent parametrized testing"""
    # Create arrays that match the settings
    if arraytest_settings['memory'] == 'mapped':
        create_func = mapped_array
    else:
        create_func = device_array

    mock_arrays = {
        'arr1': create_func(arraytest_settings['devshape1'], dtype=precision),
        'arr2': create_func(arraytest_settings['devshape2'], dtype=precision)
    }
    return ArrayResponse(arr=mock_arrays, chunks=1)

@pytest.fixture(scope='function')
def array_requests(arraytest_settings, precision):
    """Create ArrayRequest objects based on arraytest_settings"""
    return {
        'arr1': ArrayRequest(
            shape=arraytest_settings['devshape1'],
            dtype=precision,
            memory=arraytest_settings['memory'],
            stride_order=arraytest_settings['stride_order']
        ),
        'arr2': ArrayRequest(
            shape=arraytest_settings['devshape2'],
            dtype=precision,
            memory=arraytest_settings['memory'],
            stride_order=arraytest_settings['stride_order']
        )
    }

@pytest.fixture(scope='function')
def second_arrmgr(test_memory_manager, precision, arraytest_settings):
    """Create a second array manager for testing grouping behavior"""
    # Create simple arrays for the second manager
    host_arrays = TestArrays(
        arr1=np.zeros(arraytest_settings['hostshape1'], dtype=arraytest_settings['dtype']),
        arr2=np.zeros(arraytest_settings['hostshape2'], dtype=arraytest_settings['dtype'])
    )
    dev_arrays = TestArrays(
        arr1=device_array(arraytest_settings['devshape1'], arraytest_settings['dtype']),
        arr2=device_array(arraytest_settings['devshape2'], arraytest_settings['dtype'])
    )

    return ConcreteArrayManager(
        precision=precision,
        host=host_arrays,
        device=dev_arrays,
        stream_group=arraytest_settings['stream_group'],
        memory_proportion=arraytest_settings['memory_proportion'],
        memory_manager=test_memory_manager
    )


class TestArrayContainer:
    """Test the ArrayContainer base class"""

    def test_attach_existing_label(self):
        """Test attaching an array to an existing label"""
        container = TestArrays(arr1=None, arr2=None)
        test_array = np.array([1, 2, 3])

        container.attach('arr1', test_array)
        assert container.arr1 is test_array

    def test_attach_nonexistent_label(self):
        """Test attaching an array to a non-existent label raises warning"""
        container = TestArrays(arr1=None, arr2=None)
        test_array = np.array([1, 2, 3])

        with pytest.warns(UserWarning, match="Device array with label 'nonexistent' does not exist"):
            container.attach('nonexistent', test_array)

    def test_delete_existing_label(self):
        """Test deleting an existing array label"""
        test_array = np.array([1, 2, 3])
        container = TestArrays(arr1=test_array, arr2=None)

        container.delete('arr1')
        assert container.arr1 is None

    def test_delete_nonexistent_label(self):
        """Test deleting a non-existent label raises warning"""
        container = TestArrays(arr1=None, arr2=None)

        with pytest.warns(UserWarning, match="Host array with label 'nonexistent' does not exist"):
            container.delete('nonexistent')

    def test_delete_all(self):
        """Test deleting all arrays"""
        test_array1 = np.array([1, 2, 3])
        test_array2 = np.array([4, 5, 6])
        container = TestArrays(arr1=test_array1, arr2=test_array2)

        container.delete_all()
        assert container.arr1 is None
        assert container.arr2 is None


class TestBaseArrayManager:
    """Test the BaseArrayManager class"""

    def test_initialization(self, test_arrmgr, precision, arraytest_settings, test_memory_manager):
        """Test proper initialization of BaseArrayManager"""
        assert test_arrmgr._precision == precision
        assert isinstance(test_arrmgr.device, ArrayContainer)
        assert isinstance(test_arrmgr.host, ArrayContainer)
        assert test_arrmgr._stream_group == arraytest_settings['stream_group']
        assert test_arrmgr._memory_proportion == arraytest_settings['memory_proportion']
        assert test_arrmgr._needs_reallocation == []
        assert test_arrmgr._needs_overwrite == []
        assert test_arrmgr._memory_manager is test_memory_manager

        # Check that the instance was registered with the memory manager
        instance_id = id(test_arrmgr)
        assert instance_id in test_memory_manager.registry

    def test_register_with_memory_manager(self, test_arrmgr, test_memory_manager):
        """Test registration with memory manager using the test_arrmgr fixture"""
        instance_id = id(test_arrmgr)
        assert instance_id in test_memory_manager.registry

        settings = test_memory_manager.registry[instance_id]
        assert settings.invalidate_hook == test_arrmgr._invalidate_hook
        assert settings.allocation_ready_hook == test_arrmgr._on_allocation_complete

        # Check stream group assignment
        assert test_memory_manager.get_stream_group(test_arrmgr) == test_arrmgr._stream_group

    def test_on_allocation_complete(self, test_arrmgr, allocation_response):
        """Test allocation completion callback"""
        # Setup test data - arrays need to be in reallocation list to be attached
        test_arrmgr._needs_reallocation = ['arr1', 'arr2']
        test_arrmgr._needs_overwrite = ['arr1']

        test_arrmgr._on_allocation_complete(allocation_response)

        # Check that arrays were attached (should be different from originals)
        assert test_arrmgr.device.arr1 is allocation_response.arr['arr1']
        assert test_arrmgr.device.arr2 is allocation_response.arr['arr2']

        # Check that lists were cleared
        assert test_arrmgr._needs_reallocation == []
        assert test_arrmgr._needs_overwrite == []

    def test_request_allocation_single_force(self, test_arrmgr, array_requests):
        """Test allocation request forced to single mode"""
        # Test that single request can be called without error
        test_arrmgr.request_allocation(array_requests, force_type="single")

        # Verify that arrays were allocated and attached to the device container
        # The memory manager should have called the allocation_ready_hook
        assert test_arrmgr.device.arr1 is not None
        assert test_arrmgr.device.arr2 is not None

    def test_request_allocation_group_force(self, test_arrmgr, array_requests):
        """Test allocation request forced to group mode"""
        # Test that group request can be called without error
        test_arrmgr.request_allocation(array_requests, force_type="group")

        # In group mode, requests get queued until allocation is triggered
        # We need to trigger the allocation
        test_arrmgr._memory_manager.allocate_queue(test_arrmgr)

        # Verify that arrays were allocated
        assert test_arrmgr.device.arr1 is not None
        assert test_arrmgr.device.arr2 is not None

    def test_request_allocation_auto_single(self, test_arrmgr, array_requests):
        """Test automatic allocation request when not grouped"""
        # Single instance should automatically use single allocation
        test_arrmgr.request_allocation(array_requests)

        # Should behave like single request
        assert test_arrmgr.device.arr1 is not None
        assert test_arrmgr.device.arr2 is not None

    def test_request_allocation_auto_group(self, test_arrmgr, second_arrmgr, array_requests):
        """Test automatic allocation request when grouped"""
        # Change both managers to the same non-default group to make them grouped
        test_arrmgr._memory_manager.change_stream_group(test_arrmgr, "test_group")
        test_arrmgr._memory_manager.change_stream_group(second_arrmgr, "test_group")

        # Now they should be detected as grouped
        assert test_arrmgr._memory_manager.is_grouped(test_arrmgr) is True

        # Request allocation - should be queued
        test_arrmgr.request_allocation(array_requests)

        # Trigger group allocation
        test_arrmgr._memory_manager.allocate_queue(test_arrmgr)

        # Verify allocation occurred
        assert test_arrmgr.device.arr1 is not None
        assert test_arrmgr.device.arr2 is not None

    def test_invalidate_hook_functionality(self, test_arrmgr):
        """Test cache invalidation hook functionality"""

        test_arrmgr._needs_reallocation = ['old_item']
        test_arrmgr._needs_overwrite = ['old_overwrite']
        test_arrmgr.device.arr1 = np.array([1, 2, 3])
        test_arrmgr.device.arr2 = np.array([4, 5, 6])

        # Call the invalidate hook
        test_arrmgr._invalidate_hook()

        # Check that lists were cleared and arrays were set for reallocation
        assert test_arrmgr._needs_reallocation == ['arr1', 'arr2']
        assert test_arrmgr._needs_overwrite == []
        # Arrays should be None after delete_all
        assert test_arrmgr.device.arr1 is None
        assert test_arrmgr.device.arr2 is None

    def test_arrays_equal_both_none(self, test_arrmgr):
        """Test arrays_equal with both arrays None"""
        assert test_arrmgr._arrays_equal(None, None) is True

    def test_arrays_equal_one_none(self, test_arrmgr):
        """Test arrays_equal with one array None"""
        arr = np.array([1, 2, 3])
        assert test_arrmgr._arrays_equal(None, arr) is False
        assert test_arrmgr._arrays_equal(arr, None) is False

    def test_arrays_equal_same_arrays(self, test_arrmgr):
        """Test arrays_equal with identical arrays"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        assert test_arrmgr._arrays_equal(arr1, arr2) is True

    def test_arrays_equal_different_arrays(self, test_arrmgr):
        """Test arrays_equal with different arrays"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        assert test_arrmgr._arrays_equal(arr1, arr2) is False

    def test_update_host_array_no_change(self, test_arrmgr):
        """Test update_host_array when arrays are equal"""
        current = np.array([1, 2, 3])
        new = np.array([1, 2, 3])

        result = test_arrmgr._update_host_array(new, current, 'test_label')

        assert result is current
        assert 'test_label' not in test_arrmgr._needs_reallocation
        assert 'test_label' not in test_arrmgr._needs_overwrite

    def test_update_host_array_shape_change(self, test_arrmgr):
        """Test update_host_array when array shape changes"""
        current = np.array([[1, 2], [3, 4]])
        new = np.array([1, 2, 3])

        result = test_arrmgr._update_host_array(new, current, 'test_label')

        assert np.array_equal(result, new)
        assert 'test_label' in test_arrmgr._needs_reallocation
        assert 'test_label' not in test_arrmgr._needs_overwrite

    def test_update_host_array_zero_shape(self, test_arrmgr):
        """Test update_host_array when new array has zero in shape"""
        current = np.array([[1, 2], [3, 4]])
        new = np.zeros((0, 5))

        result = test_arrmgr._update_host_array(new, current, 'test_label')

        # Should return a minimal (1,1,1) array when zero in shape
        assert result.shape == (1, 1, 1)
        assert result.dtype == test_arrmgr._precision
        assert 'test_label' in test_arrmgr._needs_reallocation

    def test_update_host_array_value_change(self, test_arrmgr):
        """Test update_host_array when array values change but shape stays same"""
        current = np.array([1, 2, 3])
        new = np.array([4, 5, 6])

        result = test_arrmgr._update_host_array(new, current, 'test_label')

        assert np.array_equal(result, new)
        assert 'test_label' not in test_arrmgr._needs_reallocation
        assert 'test_label' in test_arrmgr._needs_overwrite

    def test_chunk_placeholders(self, test_arrmgr):
        """Test next_chunk method (placeholder implementation)"""
        # This is a placeholder method, so just ensure it exists and is callable
        assert test_arrmgr.initialise_chunk('test1', 'test2') == ('test1', 'test2')
        assert test_arrmgr.finalise_chunk('test1', 'test2') == ('test1', 'test2')



class TestMemoryManagerIntegration:
    """Test integration between BaseArrayManager and MemoryManager"""

    def test_memory_manager_registration(self, test_arrmgr, test_memory_manager, arraytest_settings):
        """Test that ArrayManager properly registers with MemoryManager using parametrized fixture"""
        instance_id = id(test_arrmgr)
        assert instance_id in test_memory_manager.registry

        # Check registration details
        settings = test_memory_manager.registry[instance_id]
        assert settings.invalidate_hook == test_arrmgr._invalidate_hook
        assert settings.allocation_ready_hook == test_arrmgr._on_allocation_complete

        # Check stream group assignment from arraytest_settings
        assert test_memory_manager.get_stream_group(test_arrmgr) == arraytest_settings['stream_group']

    @pytest.mark.parametrize("arraytest_overrides", [
        {'memory_proportion': 0.5},
        {'memory_proportion': 0.3},
        {'memory_proportion': 0.7},
    ], indirect=True)
    def test_memory_manager_proportion_handling(self, test_arrmgr, test_memory_manager, arraytest_settings):
        """Test that memory proportion is handled correctly using parametrized proportions"""
        expected_proportion = arraytest_settings['memory_proportion']

        # The proportion should be registered with the memory manager
        if expected_proportion is not None:
            assert test_memory_manager.proportion(test_arrmgr) == expected_proportion

    @pytest.mark.parametrize("arraytest_overrides", [
        {'stream_group': 'group1'},
        {'stream_group': 'group2'},
    ], indirect=True)
    def test_stream_group_management(self, test_arrmgr, second_arrmgr, test_memory_manager, arraytest_settings):
        """Test stream group functionality using parametrized groups"""
        # Both managers should be in the same group from arraytest_settings
        group_name = arraytest_settings['stream_group']

        # Check that managers in same group are detected as grouped
        if group_name != 'default':
            assert test_memory_manager.is_grouped(test_arrmgr) is True
            assert test_memory_manager.is_grouped(second_arrmgr) is True

            # Get instances in group
            group_instances = test_memory_manager.stream_groups.get_instances_in_group(group_name)
            assert len(group_instances) == 2
            assert id(test_arrmgr) in group_instances
            assert id(second_arrmgr) in group_instances
        else:
            # Default group behavior - not grouped unless multiple instances
            assert test_memory_manager.get_stream_group(test_arrmgr) == group_name
            assert test_memory_manager.get_stream_group(second_arrmgr) == group_name

    def test_allocation_with_settings(self, test_arrmgr, array_requests, arraytest_settings):
        """Test that allocation respects arraytest_settings parameters"""
        # Request allocation
        test_arrmgr.request_allocation(array_requests)

        # Check that allocated arrays match the settings
        assert test_arrmgr.device.arr1.shape == arraytest_settings['devshape1']
        assert test_arrmgr.device.arr2.shape == arraytest_settings['devshape2']
        assert str(test_arrmgr.device.arr1.dtype) == arraytest_settings['dtype']
        assert str(test_arrmgr.device.arr2.dtype) == arraytest_settings['dtype']


# Parametrized tests for different array configurations
@pytest.mark.parametrize("arraytest_overrides", [
    {'memory': 'device', 'stream_group': 'test_group'},
    {'memory': 'device', 'memory_proportion': 0.5},
    {'hostshape1': (5, 5), 'devshape1': (5, 5)},
    {'dtype': 'float64'},
], indirect=True)
def test_array_manager_with_different_configs(test_arrmgr, arraytest_overrides, arraytest_settings):
    """Test BaseArrayManager with different configurations"""
    # Check that overrides were applied correctly
    if 'stream_group' in arraytest_overrides:
        assert test_arrmgr._stream_group == arraytest_settings['stream_group']
    if 'memory_proportion' in arraytest_overrides:
        assert test_arrmgr._memory_proportion == arraytest_settings['memory_proportion']
    if 'hostshape1' in arraytest_overrides:
        assert test_arrmgr.host.arr1.shape == arraytest_settings['hostshape1']
    if 'dtype' in arraytest_overrides:
        assert str(test_arrmgr.host.arr1.dtype) == arraytest_settings['dtype']

    # Should still be properly initialized
    assert isinstance(test_arrmgr.device, ArrayContainer)
    assert isinstance(test_arrmgr.host, ArrayContainer)

@pytest.mark.parametrize("arraytest_overrides", [
    {'memory': 'device', 'devshape1': (10, 20), 'devshape2': (15, 25)},
    {'memory': 'device', 'devshape1': (3, 4, 5), 'devshape2': (6, 7, 8)},
], indirect=True)
def test_allocation_response_matches_settings(allocation_response, arraytest_settings):
    """Test that allocation_response fixture uses arraytest_settings correctly"""
    assert allocation_response.arr['arr1'].shape == arraytest_settings['devshape1']
    assert allocation_response.arr['arr2'].shape == arraytest_settings['devshape2']
    assert allocation_response.chunks == 1

@pytest.mark.parametrize("arraytest_overrides", [
    {'memory': 'device', 'stream_group': 'custom_group', 'memory_proportion': 0.3},
    {'memory': 'device', 'stream_group': 'another_group', 'memory_proportion': 0.7},
], indirect=True)
def test_memory_integration_with_settings(test_arrmgr, test_memory_manager, arraytest_settings):
    """Test that memory manager integration respects arraytest_settings"""
    # Check stream group from settings
    assert test_memory_manager.get_stream_group(test_arrmgr) == arraytest_settings['stream_group']

    # Check proportion from settings
    if arraytest_settings['memory_proportion'] is not None:
        assert test_memory_manager.proportion(test_arrmgr) == arraytest_settings['memory_proportion']
