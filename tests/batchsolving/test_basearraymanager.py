from os import environ
import pytest
from cupy import zeros_like

from cubie.batchsolving.BaseArrayManager import BaseArrayManager, ArrayContainer
import attrs
import numpy as np

from cubie.memory.array_requests import ArrayResponse, ArrayRequest
from cubie.memory.mem_manager import MemoryManager
from cubie.outputhandling.output_sizes import BatchOutputSizes
from numpy.testing import assert_array_equal

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
    def finalise(self, indices, axis):
        return indices, axis

    def initialise(self, indices, axis):
        return indices, axis

@attrs.define(slots=False)
class TestArrays(ArrayContainer):
    state = attrs.field(default=None)
    observables = attrs.field(default=None)
    state_summaries = attrs.field(default=None)
    observable_summaries = attrs.field(default=None)

@attrs.define(slots=False)
class TestArraysSimple(ArrayContainer):
    arr1 = attrs.field(default=None)
    arr2 = attrs.field(default=None)
    stride_order = ("time", "run", "variable")

@pytest.fixture(scope='function')
def arraytest_overrides(request):
    if hasattr(request, 'param'):
        return request.param
    return {}

@pytest.fixture(scope="function")
def arraytest_settings(arraytest_overrides):
    settings = {
        'hostshape1': (4, 3, 4),
        'hostshape2': (4, 3, 4),
        'hostshape3': (2, 3, 4),
        'hostshape4': (2, 3, 4),
        'devshape1': (4, 3, 4),
        'devshape2': (4, 3, 4),
        'devshape3': (2, 3, 4),
        'devshape4': (2, 3, 4),
        'chunks': 2,
        'chunk_axis': 'run',
        'dtype': 'float32',
        'memory': 'device',
        '_stride_order': ('time', 'run', 'variable'),
        'stream_group': 'default',
        'memory_proportion': None,
    }
    settings.update(arraytest_overrides)

    # Calculate derived shapes for summary arrays (time_dim - 2)
    settings['hostshape3'] = (max(0, settings['hostshape1'][0] - 2),
                              settings['hostshape1'][1],
                              settings['hostshape1'][2])
    settings['hostshape4'] = (max(0, settings['hostshape2'][0] - 2),
                              settings['hostshape2'][1],
                              settings['hostshape2'][2])
    settings['devshape3'] = (max(0, settings['devshape1'][0] - 2),
                             settings['devshape1'][1],
                             settings['devshape1'][2])
    settings['devshape4'] = (max(0, settings['devshape2'][0] - 2),
                             settings['devshape2'][1],
                             settings['devshape2'][2])

    return settings

@pytest.fixture(scope='function')
def hostarrays(arraytest_settings):
    if arraytest_settings['memory'] == 'mapped':
        empty = mapped_array
    else:
        empty = np.zeros
    return TestArraysSimple(
        arr1=empty(arraytest_settings['hostshape1'],
                   dtype=arraytest_settings['dtype']),
        arr2=empty(arraytest_settings['hostshape2'],
                   dtype=arraytest_settings['dtype'])
    )

@pytest.fixture(scope='function')
def devarrays(arraytest_settings, hostarrays):
    if arraytest_settings['memory'] == 'mapped':
        devarr1 = MappedNDArray(hostarrays.arr1, arraytest_settings['dtype'])
        devarr2 = MappedNDArray(hostarrays.arr2, arraytest_settings['dtype'])
    else:
        devarr1 = device_array(arraytest_settings['devshape1'],
                               arraytest_settings['dtype'])
        devarr2 = device_array(arraytest_settings['devshape2'],
                               arraytest_settings['dtype'])
    return TestArraysSimple(arr1=devarr1, arr2=devarr2)

@pytest.fixture(scope='function')
def test_memory_manager():
    """Create an actual MemoryManager instance for testing"""
    # Create a memory manager in passive mode to avoid CUDA context issues
    mem_mgr = MemoryManager(mode="passive")
    return mem_mgr

@pytest.fixture(scope='function')
def test_arrmgr(hostarrays, devarrays, precision, arraytest_settings, test_memory_manager, batch_output_sizes):
    return ConcreteArrayManager(
            precision=precision,
            sizes=batch_output_sizes,
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
            stride_order=arraytest_settings['_stride_order']
        ),
        'arr2': ArrayRequest(
            shape=arraytest_settings['devshape2'],
            dtype=precision,
            memory=arraytest_settings['memory'],
            stride_order=arraytest_settings['_stride_order']
        )
    }

@pytest.fixture(scope='function')
def second_arrmgr(test_memory_manager, precision, arraytest_settings, batch_output_sizes):
    """Create a second array manager for testing grouping behavior"""
    # Create simple arrays for the second manager
    host_arrays = TestArraysSimple(
        arr1=np.zeros(arraytest_settings['hostshape1'], dtype=arraytest_settings['dtype']),
        arr2=np.zeros(arraytest_settings['hostshape2'], dtype=arraytest_settings['dtype'])
    )
    dev_arrays = TestArraysSimple(
        arr1=device_array(arraytest_settings['devshape1'], arraytest_settings['dtype']),
        arr2=device_array(arraytest_settings['devshape2'], arraytest_settings['dtype'])
    )

    return ConcreteArrayManager(
        precision=precision,
        sizes=batch_output_sizes,
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
        container = TestArraysSimple(arr1=None, arr2=None)
        test_array = np.array([1, 2, 3])

        container.attach('arr1', test_array)
        assert container.arr1 is test_array

    def test_attach_nonexistent_label(self):
        """Test attaching an array to a non-existent label raises warning"""
        container = TestArraysSimple(arr1=None, arr2=None)
        test_array = np.array([1, 2, 3])

        with pytest.warns(UserWarning, match="Device array with label 'nonexistent' does not exist"):
            container.attach('nonexistent', test_array)

    def test_delete_existing_label(self):
        """Test deleting an existing array label"""
        test_array = np.array([1, 2, 3])
        container = TestArraysSimple(arr1=test_array, arr2=None)

        container.delete('arr1')
        assert container.arr1 is None

    def test_delete_nonexistent_label(self):
        """Test deleting a non-existent label raises warning"""
        container = TestArraysSimple(arr1=None, arr2=None)

        with pytest.warns(UserWarning, match="Host array with label 'nonexistent' does not exist"):
            container.delete('nonexistent')

    def test_delete_all(self):
        """Test deleting all arrays"""
        test_array1 = np.array([1, 2, 3])
        test_array2 = np.array([4, 5, 6])
        container = TestArraysSimple(arr1=test_array1, arr2=test_array2)

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
        assert test_arrmgr._needs_reallocation
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

        test_arrmgr._update_host_array(new, current, 'arr1')
        test_arrmgr.allocate()
        assert 'arr1' not in test_arrmgr._needs_reallocation
        assert 'arr1' not in test_arrmgr._needs_overwrite

    def test_update_host_array_shape_change(self, test_arrmgr):
        """Test update_host_array when array shape changes"""
        current = np.array([[1, 2], [3, 4]])
        new = np.array([1, 2, 3])
        test_arrmgr._update_host_array(new, current, 'arr1')

        assert 'arr1' in test_arrmgr._needs_reallocation
        assert 'arr1' in test_arrmgr._needs_overwrite
        # Check that the array was attached to the host container
        assert test_arrmgr.host.arr1 is new

    def test_update_host_array_zero_shape(self, test_arrmgr):
        """Test update_host_array when new array has zero in shape"""
        current = np.array([[1, 2], [3, 4]])
        new = np.zeros((0, 5))

        test_arrmgr._update_host_array(new, current, 'arr1')

        # Should attach a minimal (1,1,1) array when zero in shape
        assert test_arrmgr.host.arr1.shape == (1, 1, 1)
        assert test_arrmgr.host.arr1.dtype == test_arrmgr._precision
        assert 'arr1' in test_arrmgr._needs_reallocation

    def test_update_host_array_value_change(self, test_arrmgr):
        """Test update_host_array when array values change but shape stays same"""
        current = np.array([1, 2, 3])
        new = np.array([4, 5, 6])
        test_arrmgr.allocate()
        test_arrmgr._update_host_array(new, current, 'arr1')

        assert 'arr1' not in test_arrmgr._needs_reallocation
        assert 'arr1' in test_arrmgr._needs_overwrite
        # Check that the array was attached to the host container
        assert test_arrmgr.host.arr1 is new

    def test_chunk_placeholders(self, test_arrmgr):
        """Test next_chunk method (placeholder implementation)"""
        # This is a placeholder method, so just ensure it exists and is callable
        assert test_arrmgr.initialise('test1', 'test2') == ('test1', 'test2')
        assert test_arrmgr.finalise('test1', 'test2') == ('test1', 'test2')


@pytest.fixture(scope='function')
def batch_output_sizes(arraytest_settings):
    """Create a real BatchOutputSizes instance using arraytest_settings"""
    return BatchOutputSizes(
        state=arraytest_settings['hostshape1'],
        observables=arraytest_settings['hostshape2'],
        state_summaries=arraytest_settings['hostshape3'],
        observable_summaries=arraytest_settings['hostshape4'],
        stride_order=arraytest_settings['_stride_order']
    )

@pytest.fixture(scope='function')
def test_arrays_with_stride_order(arraytest_settings):
    """Create test arrays with proper stride order set using arraytest_settings shapes"""
    host_arrays = TestArrays(
        state=np.zeros(arraytest_settings['hostshape1'], dtype=np.float32),
        observables=np.zeros(arraytest_settings['hostshape2'], dtype=np.float32),
        state_summaries=np.zeros(arraytest_settings['hostshape3'], dtype=np.float32),
        observable_summaries=np.zeros(arraytest_settings['hostshape4'], dtype=np.float32)
    )
    host_arrays.stride_order = arraytest_settings['_stride_order']

    device_arrays = TestArrays(
        state=device_array(arraytest_settings['devshape1'], dtype=np.float32),
        observables=device_array(arraytest_settings['devshape2'], dtype=np.float32),
        state_summaries=device_array(arraytest_settings['devshape3'], dtype=np.float32),
        observable_summaries=device_array(arraytest_settings['devshape4'],
                                          dtype=np.float32),
        memory_type="device"
    )
    device_arrays.stride_order = arraytest_settings['_stride_order']

    return host_arrays, device_arrays

@pytest.fixture(scope='function')
def test_manager_with_sizing(test_memory_manager, test_arrays_with_stride_order, batch_output_sizes):
    """Create array manager with proper arrays for size testing"""
    host_arrays, device_arrays = test_arrays_with_stride_order
    return ConcreteArrayManager(
        precision=np.float32,
        sizes=batch_output_sizes,
        host=host_arrays,
        device=device_arrays,
        stream_group='default',
        memory_proportion=None,
        memory_manager=test_memory_manager)


class TestCheckSizesAndTypes:
    """Test the updated check_sizes and check_type methods that return dictionaries"""

    def test_check_type_returns_dict(self, test_manager_with_sizing):
        """Test that check_type returns a dictionary of results"""
        arrays = {
            'state': np.zeros((10, 5, 3), dtype=np.float32),
            'observables': np.zeros((10, 5, 2), dtype=np.float64)  # Wrong dtype
        }

        result = test_manager_with_sizing.check_type(arrays)

        assert isinstance(result, dict)
        assert result['state'] is True
        assert result['observables'] is False

    def test_check_type_all_correct(self, test_manager_with_sizing):
        """Test check_type when all arrays have correct dtype"""
        arrays = {
            'state': np.zeros((10, 5, 3), dtype=np.float32),
            'observables': np.zeros((10, 5, 2), dtype=np.float32)
        }

        result = test_manager_with_sizing.check_type(arrays)

        assert all(result.values())
        assert result['state'] is True
        assert result['observables'] is True

    def test_check_type_with_none_arrays(self, test_manager_with_sizing):
        """Test check_type with None arrays"""
        arrays = {
            'state': None,
            'observables': np.zeros((10, 5, 2), dtype=np.float32)
        }

        result = test_manager_with_sizing.check_type(arrays)

        assert result['state'] is True  # None arrays should pass type check
        assert result['observables'] is True

    def test_check_sizes_returns_dict(self, arraytest_settings,
        test_manager_with_sizing):
        """Test that check_sizes returns a dictionary of results"""
        arrays = {
            "state": np.zeros(arraytest_settings['hostshape1'], dtype=np.float32),
            "observables": np.zeros(arraytest_settings['hostshape2'], dtype=np.float32),
            "state_summaries": np.zeros(arraytest_settings['hostshape3'], dtype=np.float32),
            "observable_summaries": np.zeros(arraytest_settings['hostshape4'], dtype=np.float32),
        }

        result = test_manager_with_sizing.check_sizes(arrays, location="host")
        assert isinstance(result, dict)
        assert all([key in result for key in arrays.keys()])

    def test_check_sizes_correct_shapes(self, test_manager_with_sizing, arraytest_settings):
        """Test check_sizes with correctly shaped arrays"""
        arrays = {
            'state': np.zeros(arraytest_settings['hostshape1'], dtype=np.float32),
            'observables': np.zeros(arraytest_settings['hostshape2'], dtype=np.float32)
        }

        result = test_manager_with_sizing.check_sizes(arrays, location="host")

        assert result['state'] is True
        assert result['observables'] is True

    def test_check_sizes_wrong_shapes(self, test_manager_with_sizing, arraytest_settings):
        """Test check_sizes with incorrectly shaped arrays"""
        # Create wrong shapes by adding 1 to each dimension
        wrong_shape1 = tuple(dim + 1 for dim in arraytest_settings['hostshape1'])
        wrong_shape2 = tuple(dim + 1 for dim in arraytest_settings['hostshape2'])

        arrays = {
            'state': np.zeros(wrong_shape1, dtype=np.float32),
            'observables': np.zeros(wrong_shape2, dtype=np.float32)
        }

        result = test_manager_with_sizing.check_sizes(arrays, location="host")

        assert result['state'] is False
        assert result['observables'] is False

    def test_check_sizes_with_chunking(self, test_manager_with_sizing, 
                                       arraytest_settings):
        """Test check_sizes with device location and chunking"""
        stride_order = test_manager_with_sizing.device.stride_order
        chunk_axis = arraytest_settings['chunk_axis']
        chunks = arraytest_settings['chunks']
        test_manager_with_sizing._chunks = arraytest_settings['chunks']
        test_manager_with_sizing._chunk_axis = chunk_axis
        chunk_index = stride_order.index(chunk_axis)
        expected_shape1 = tuple(int(np.ceil(val / chunks)) if i == chunk_index
                           else val for i, val in enumerate(arraytest_settings[
                                                            'hostshape1']))

        expected_shape2 = tuple(int(np.ceil(val / chunks)) if i == chunk_index
                           else val for i, val in enumerate(arraytest_settings[
                                                            'hostshape2']))
        # For chunked device arrays, the run dimension should be divided by chunks
        arrays = {
            'state': device_array(expected_shape1, dtype=np.float32),  # 5 runs / 2 chunks = 3 (ceil)
            'observables': device_array(expected_shape2, dtype=np.float32)
        }

        result = test_manager_with_sizing.check_sizes(arrays, location="device")

        assert result['state'] is True
        assert result['observables'] is True

    def test_check_sizes_invalid_location(self, test_manager_with_sizing):
        """Test check_sizes with invalid location raises AttributeError"""
        arrays = {'state': np.zeros((10, 5, 3), dtype=np.float32)}

        with pytest.raises(AttributeError, match="Invalid location: invalid"):
            test_manager_with_sizing.check_sizes(arrays, location="invalid")


class TestCheckIncomingArrays:
    """Test the check_incoming_arrays method"""

    def test_check_incoming_arrays_all_pass(self, test_manager_with_sizing, arraytest_settings):
        """Test check_incoming_arrays when all arrays pass both checks"""
        arrays = {
            'state': np.zeros(arraytest_settings['hostshape1'], dtype=np.float32),
            'observables': np.zeros(arraytest_settings['hostshape2'], dtype=np.float32)
        }

        result = test_manager_with_sizing.check_incoming_arrays(arrays, location="host")

        assert isinstance(result, dict)
        assert result['state'] is True
        assert result['observables'] is True

    def test_check_incoming_arrays_size_fail(self, test_manager_with_sizing, arraytest_settings):
        """Test check_incoming_arrays when size check fails"""
        # Create wrong shapes by adding 1 to each dimension
        wrong_shape1 = tuple(dim + 1 for dim in arraytest_settings['hostshape1'])
        wrong_shape2 = tuple(dim + 1 for dim in arraytest_settings['hostshape2'])
        
        arrays = {
            'state': np.zeros(wrong_shape1, dtype=np.float32),  # Wrong shape
            'observables': np.zeros(arraytest_settings['hostshape2'], dtype=np.float32)
        }

        result = test_manager_with_sizing.check_incoming_arrays(arrays, location="host")

        assert result['state'] is False  # Should fail due to size
        assert result['observables'] is True

    def test_check_incoming_arrays_type_fail(self, test_manager_with_sizing, arraytest_settings):
        """Test check_incoming_arrays when type check fails"""
        arrays = {
            'state': np.zeros(arraytest_settings['hostshape1'], dtype=np.float64),  # Wrong dtype
            'observables': np.zeros(arraytest_settings['hostshape2'], dtype=np.float32)
        }

        result = test_manager_with_sizing.check_incoming_arrays(arrays, location="host")

        assert result['state'] is False  # Should fail due to type
        assert result['observables'] is True


class TestAttachExternalArrays:
    """Test the attach_external_arrays method"""

    def test_attach_external_arrays_success(self, test_manager_with_sizing, arraytest_settings):
        """Test successfully attaching external arrays"""
        arrays = {
            'state': np.zeros(arraytest_settings['hostshape1'], dtype=np.float32),
            'observables': np.zeros(arraytest_settings['hostshape2'], dtype=np.float32)
        }

        result = test_manager_with_sizing.attach_external_arrays(
            arrays, location="host"
        )

        assert result is True
        assert test_manager_with_sizing.host.state is arrays['state']
        assert test_manager_with_sizing.host.observables is arrays['observables']

    def test_attach_external_arrays_with_failures(self, test_manager_with_sizing, arraytest_settings):
        """Test attaching external arrays when some fail validation"""
        # Create wrong shapes by adding 1 to each dimension
        wrong_shape1 = tuple(dim + 1 for dim in arraytest_settings['hostshape1'])

        arrays = {
            'state': np.zeros(wrong_shape1, dtype=np.float32),  # Wrong shape
            'observables': np.zeros(arraytest_settings['hostshape2'], dtype=np.float32)  # Correct
        }

        with pytest.warns(UserWarning, match="The following arrays did not match"):
            result = test_manager_with_sizing.attach_external_arrays(
                arrays, location="host"
            )

        assert result is True
        # Only the valid array should be attached
        assert test_manager_with_sizing.host.observables is arrays['observables']


class TestUpdateHostArrays:
    """Test the new update_host_arrays method"""

    def test_update_host_arrays_success(self, test_manager_with_sizing, arraytest_settings):
        """Test update_host_arrays with valid arrays"""
        # Set up initial arrays in the host container
        test_manager_with_sizing.host.state = np.zeros(arraytest_settings['hostshape1'], dtype=np.float32)
        test_manager_with_sizing.host.observables = np.zeros(arraytest_settings['hostshape2'], dtype=np.float32)

        new_arrays = {
            'state': np.ones(arraytest_settings['hostshape1'], dtype=np.float32),
            'observables': np.ones(arraytest_settings['hostshape2'], dtype=np.float32)
        }

        test_manager_with_sizing.update_host_arrays(new_arrays)

        # Arrays should be updated and marked for overwrite
        assert test_manager_with_sizing.host.state is new_arrays['state']
        assert test_manager_with_sizing.host.observables is new_arrays['observables']
        assert 'state' in test_manager_with_sizing._needs_overwrite
        assert 'observables' in test_manager_with_sizing._needs_overwrite

    def test_update_host_arrays_shape_change(self, test_manager_with_sizing, arraytest_settings):
        """Test update_host_arrays when array shapes change"""
        # Set up initial arrays in the host container
        test_manager_with_sizing.host.state = np.zeros(arraytest_settings['hostshape1'], dtype=np.float32)

        # Create a different shape by modifying the first dimension
        new_shape = (arraytest_settings['hostshape1'][0] - 1,) + arraytest_settings['hostshape1'][1:]
        new_arrays = {
            'state': np.ones(new_shape, dtype=np.float32),  # Different shape
        }

        test_manager_with_sizing.update_host_arrays(new_arrays)

        # Array should be updated and marked for reallocation
        assert test_manager_with_sizing.host.state is new_arrays['state']
        assert 'state' in test_manager_with_sizing._needs_reallocation

    def test_update_host_arrays_size_mismatch(self, test_manager_with_sizing, arraytest_settings):
        """Test update_host_arrays with size mismatch"""
        test_manager_with_sizing.host.state = np.zeros(arraytest_settings['hostshape1'], dtype=np.float32)

        # Create wrong shape by adding 1 to each dimension
        wrong_shape = tuple(dim + 1 for dim in arraytest_settings['hostshape1'])
        new_arrays = {
            'state': np.ones(wrong_shape, dtype=np.float32),  # Wrong shape
        }

        with pytest.warns(UserWarning, match="do not match the expected system sizes"):
            test_manager_with_sizing.update_host_arrays(new_arrays)

    def test_update_host_arrays_nonexistent_array(self,
                                                  test_manager_with_sizing, arraytest_settings):
        """Test update_host_arrays with non-existent array"""
        new_arrays = {
            'nonexistent': np.zeros(arraytest_settings['hostshape1'],
            dtype=np.float32),
        }

        with pytest.warns(UserWarning, match="does not exist"):
            test_manager_with_sizing.update_host_arrays(new_arrays)

    def test_update_host_arrays_no_change(self, test_manager_with_sizing, arraytest_settings):
        """Test update_host_arrays when arrays are unchanged"""
        initial_array = np.ones(arraytest_settings['hostshape1'], dtype=np.float32)
        test_manager_with_sizing.host.state = initial_array

        new_arrays = {
            'state': np.ones(arraytest_settings['hostshape1'], dtype=np.float32),  # Same values
        }

        initial_needs_reallocation = test_manager_with_sizing._needs_reallocation.copy()
        initial_needs_overwrite = test_manager_with_sizing._needs_overwrite.copy()

        test_manager_with_sizing.update_host_arrays(new_arrays)

        # Should not add to reallocation or overwrite lists since arrays are equal
        assert test_manager_with_sizing._needs_reallocation == initial_needs_reallocation
        assert test_manager_with_sizing._needs_overwrite == initial_needs_overwrite


class TestUpdateSizes:
    """Test the update_sizes method"""

    def test_update_sizes_success(self, test_manager_with_sizing, arraytest_settings):
        """Test successful update of sizes"""
        # Create new sizes with different dimensions
        new_sizes = BatchOutputSizes(
            state=(12, 6, 4),  # Different from original
            observables=(12, 6, 3),
            state_summaries=(10, 6, 4),
            observable_summaries=(10, 6, 3),
            stride_order=arraytest_settings['_stride_order']
        )

        test_manager_with_sizing.update_sizes(new_sizes)

        assert test_manager_with_sizing._sizes is new_sizes

    def test_update_sizes_wrong_type(self, test_manager_with_sizing):
        """Test update_sizes with wrong type raises TypeError"""
        # Try to update with a different type
        wrong_type_sizes = {"state": (12, 6, 4)}

        with pytest.raises(TypeError, match="Expected the new sizes object to be the same size"):
            test_manager_with_sizing.update_sizes(wrong_type_sizes)


# Add missing precision fixture
@pytest.fixture(scope='function')
def precision():
    """Provide precision for tests"""
    return np.float32


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
    {'hostshape1': (5, 5, 2), 'devshape1': (5, 5, 2)},
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
    {'memory': 'device', 'devshape1': (10, 20, 3), 'devshape2': (15, 25, 2)},
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


