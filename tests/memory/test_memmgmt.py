from os import environ

import pytest

from cubie.memory.cupyemm import CuPyAsyncNumbaManager, CuPySyncNumbaManager

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from cubie.cudasim_utils import (FakeNumbaCUDAMemoryManager as
                                     NumbaCUDAMemoryManager)
    from cubie.cudasim_utils import FakeStream as Stream
    from numba.cuda.simulator.cudadrv.devicearray import (FakeCUDAArray as
                                               DeviceNDArrayBase)
    from numba.cuda.simulator.cudadrv.devicearray import (FakeCUDAArray as
                                               MappedNDArray)

else:
    from numba.cuda.cudadrv.driver import NumbaCUDAMemoryManager
    from numba.cuda.cudadrv.driver import Stream
    from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase, MappedNDArray


from cubie.memory.memmgmt import (
    MemoryManager,
    ArrayRequest,
    StreamGroups,
    ArrayResponse,
    InstanceMemorySettings,
)



import warnings
from numba import cuda
import numpy as np


# ========================== dataclasses and helpers ======================== #
class DummyClass:
    def __init__(self, proportion=None, invalidate_all_hook=None):
        self.proportion = proportion
        self.invalidate_all_hook = invalidate_all_hook

@pytest.fixture(scope="function")
def array_request_override(request):
    return request.param if hasattr(request, 'param') else {}

@pytest.fixture(scope="function")
def array_request_settings(array_request_override):
    """Fixture to provide settings for ArrayRequest."""
    defaults = {'shape': (1,1,1),
                'dtype': np.float32,
                'memory': 'device',
                'stride_order': ("time", "run", "variable")}
    if array_request_override:
        for key, value in array_request_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults

@pytest.fixture(scope="function")
def array_request(array_request_settings):
    return ArrayRequest(**array_request_settings)

@pytest.fixture(scope="function")
def expected_single_array(array_request_settings):
    arr_request = array_request_settings
    if arr_request['memory'] == 'device':
        arr = cuda.device_array(array_request_settings['shape'],
                                dtype=array_request_settings['dtype'])
    elif arr_request['memory'] == 'pinned':
        arr = cuda.pinned_array(array_request_settings['shape'],
                                dtype=array_request_settings['dtype'])
    elif arr_request['memory'] == 'mapped':
        arr = cuda.mapped_array(array_request_settings['shape'],
                                dtype=array_request_settings['dtype'])
    elif arr_request['memory'] == 'managed':
        raise NotImplementedError("Managed memory not implemented")
    else:
        raise ValueError(f"Invalid memory type: {arr_request['memory']}")
    return arr

class TestArrayRequests:
    @pytest.mark.parametrize("array_request_override",
                             [{'shape': (20000,), 'dtype': np.float64},
                              {'memory': 'pinned'},
                              {'shape': (10, 10, 10, 10, 10),
                               'dtype': np.float32}],
                              indirect=True)
    def test_size(self, array_request):
        assert (array_request.size == np.prod(array_request.shape) *
                array_request.dtype().itemsize), "Incorrect size calculated"

    @pytest.mark.parametrize("array_request_override",
                             [{'shape': (20000,), 'dtype': np.float64}],
                             indirect=True)
    def test_instantiation(self, array_request):
        assert array_request.shape == (20000,)
        assert array_request.dtype == np.float64
        assert array_request.memory == 'device'
        assert array_request.stride_order == ("time", "run", "variable")

    def test_default_stride_ordering(self):
        # 3D shape default stride_order
        req3 = ArrayRequest(shape=(2,3,4), dtype=np.float32, memory='device', stride_order=None)
        assert req3.stride_order == ('time', 'run', 'variable')
        # 2D shape default stride_order
        req2 = ArrayRequest(shape=(5,6), dtype=np.float32, memory='device', stride_order=None)
        assert req2.stride_order == ('variable', 'run')
        # 1D shape leaves stride_order None
        req1 = ArrayRequest(
            shape=(10,), dtype=np.float32, memory="device", stride_order=None
        )
        assert req1.stride_order is None

    @pytest.mark.nocudasim
    @pytest.mark.parametrize("array_request_override",
                             [{'shape': (20000,), 'dtype': np.float64},
                              {'memory': 'pinned'},
                              {'memory': 'mapped'}], indirect=True)
    def test_array_response(self, array_request,
                            array_request_settings,
                            expected_single_array):
        mgr = MemoryManager()
        instance = DummyClass()
        mgr.register(instance)
        resp = mgr.allocate_all(instance, {'test':array_request})
        arr = resp['test']

        # Can't directly check for equality as they'll be at different addresses
        assert arr.shape == expected_single_array.shape
        assert type(arr) == type(expected_single_array)
        assert arr.nbytes == expected_single_array.nbytes
        assert arr.strides == expected_single_array.strides
        assert arr.dtype == expected_single_array.dtype

@pytest.fixture(scope="function")
def stream_groups(array_request_settings):
    return StreamGroups()

class TestStreamGroups:
    @pytest.mark.nocudasim
    def test_add_instance(self, stream_groups):
        inst = DummyClass()
        stream_groups.add_instance(inst, "group1")
        assert "group1" in stream_groups.groups
        assert id(inst) in stream_groups.groups["group1"]
        assert isinstance(stream_groups.streams["group1"], Stream)
        with pytest.raises(ValueError):
            stream_groups.add_instance(inst, "group2")

        inst2 = DummyClass()
        stream_groups.add_instance(inst2, "group2")
        assert id(inst2) in stream_groups.groups["group2"]
        assert id(inst) not in stream_groups.groups["group2"]
        assert id(inst2) not in stream_groups.groups["group1"]

    def test_get_group(self, stream_groups):
        """Test that get_group returns the correct group for an instance"""
        inst = DummyClass()
        stream_groups.add_instance(inst, "group1")
        assert stream_groups.get_group(inst) == "group1"
        with pytest.raises(ValueError):
            assert stream_groups.get_group(DummyClass()) is None
        inst1 = DummyClass()
        stream_groups.add_instance(inst1, "group2")
        assert stream_groups.get_group(inst1) == "group2"
        assert stream_groups.get_group(inst) != "group2"

    @pytest.mark.nocudasim
    def test_change_group(self, stream_groups):
        """Test that change_group removes the instance from the old group,
        adds it to the new one, and that the instances stream has changed"""
        inst = DummyClass()
        stream_groups.add_instance(inst, 'group1')
        old_stream = stream_groups.get_stream(inst)
        # move to new group
        stream_groups.change_group(inst, 'group2')
        assert id(inst) not in stream_groups.groups['group1']
        assert id(inst) in stream_groups.groups['group2']
        new_stream = stream_groups.get_stream(inst)
        assert int(new_stream.handle.value) != int(old_stream.handle.value)
        # error when instance not in any group
        with pytest.raises(ValueError):
            stream_groups.change_group(DummyClass(), 'groupX')


    @pytest.mark.nocudasim
    def test_get_stream(self, stream_groups):
        inst = DummyClass()
        stream_groups.add_instance(inst, "group1")
        stream1 = int(stream_groups.get_stream(inst).handle.value)
        stream2 = int(stream_groups.get_stream(inst).handle.value)
        assert stream1 == stream2
        assert stream1 != stream_groups.streams["group1"]

        inst2 = DummyClass()
        stream_groups.add_instance(inst2, "group2")
        stream3 = int(stream_groups.get_stream(inst2).handle.value)
        assert stream3 != stream1
        assert stream3 != stream2
        assert stream3 != stream_groups.streams["group1"]
        assert stream3 != stream_groups.streams["group2"]

    @pytest.mark.nocudasim
    def test_reinit_streams(self, stream_groups):
        """ test that two instances have different streams in different
        groups, then reinit, and check that streams don't match old ones or
        each other."""
        inst1 = DummyClass()
        inst2 = DummyClass()
        stream_groups.add_instance(inst1, 'g1')
        stream_groups.add_instance(inst2, 'g2')
        # ensure different initial streams
        s1_old = stream_groups.get_stream(inst1)
        s2_old = stream_groups.get_stream(inst2)
        assert s1_old != s2_old
        # reinitialize streams
        stream_groups.reinit_streams()
        s1_new = stream_groups.get_stream(inst1)
        s2_new = stream_groups.get_stream(inst2)
        assert s1_new != s1_old
        assert s2_new != s2_old
        assert s1_new != s2_new

@pytest.fixture(scope="function")
def instance_settings_override(request):
    return request.param if hasattr(request, 'param') else {}

@pytest.fixture(scope="function")
def instance_settings(instance_settings_override):
    defaults = {
        'proportion': 0.5,
        'invalidate_hook': lambda: None,
    }
    if instance_settings_override:
        for key, value in instance_settings_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults

@pytest.fixture(scope="function")
def instance_settings_obj(instance_settings):
    return InstanceMemorySettings(**instance_settings)

class TestInstanceMemorySettings:
    def test_instantiation(self, instance_settings_obj, instance_settings):
        # Test that the settings in the object match the settings fixture
        assert instance_settings_obj.proportion == instance_settings['proportion']
        assert callable(instance_settings_obj.invalidate_hook)
        assert isinstance(instance_settings_obj.allocations, dict)

    def test_add_allocation(self, instance_settings_obj):
        # test that add_allocation adds the reference to the allocations dict
        arr = np.ndarray((10,), dtype=np.float64)
        instance_settings_obj.add_allocation('foo', arr)
        assert 'foo' in instance_settings_obj.allocations
        assert instance_settings_obj.allocations['foo'] is arr

    def test_free(self, instance_settings_obj):
        # test that free removes the reference to the allocations dict
        arr = np.ndarray((10,), dtype=np.float64)
        instance_settings_obj.add_allocation('foo', arr)
        instance_settings_obj.free('foo')
        assert 'foo' not in instance_settings_obj.allocations

    def test_free_all(self, instance_settings_obj):
        # test that free_all removes all references to the allocations dict
        arr1 = np.ndarray((10,), dtype=np.float64)
        arr2 = np.ndarray((20,), dtype=np.float64)
        instance_settings_obj.add_allocation('foo', arr1)
        instance_settings_obj.add_allocation('bar', arr2)
        instance_settings_obj.free_all()
        assert instance_settings_obj.allocations == {}

    @pytest.mark.nocudasim
    def test_allocated_bytes(self, instance_settings_obj):
        # test that the allocated_bytes property returns the correct value
        arr1 = np.ndarray((100,), dtype=np.float64)
        arr2 = np.ndarray((25,), dtype=np.float64)
        instance_settings_obj.add_allocation('foo', arr1)
        instance_settings_obj.add_allocation('bar', arr2)
        expected_bytes = arr1.nbytes + arr2.nbytes
        assert instance_settings_obj.allocated_bytes == expected_bytes
        instance_settings_obj.free('foo')
        assert instance_settings_obj.allocated_bytes == arr2.nbytes


# ========================= Memory Manager Class =========================== #
@pytest.fixture
def fixed_mem_override(request):
    return request.param if hasattr(request, 'param') else {}

@pytest.fixture(scope="function")
def fixed_mem_settings(fixed_mem_override):
    defaults = {
        'free': 1*1024**3,
        'total': 8*1024**3}
    if fixed_mem_override:
        for key, value in fixed_mem_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults

@pytest.fixture(scope="function")
def mem_manager_override(request):
    return request.param if hasattr(request, 'param') else {}

@pytest.fixture(scope="function")
def mem_manager_settings(mem_manager_override):
    defaults = {
        'mode': 'passive',
        'stride_order': ("time", "run", "variable")
    }
    if mem_manager_override:
        for key, value in mem_manager_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults


# Override memory info for consistent tests
@pytest.fixture(scope="function")
def mgr(fixed_mem_settings, mem_manager_settings):
    class TestMemoryManager(MemoryManager):
        def get_memory_info(self):
            free = fixed_mem_settings['free']
            total = fixed_mem_settings['total']
            return free, total

        if environ.get('NUMBA_ENABLE_CUDASIM', '0') == '1':
            def set_allocator(self, name: str=None):
                pass
            _allocator = None

    # Create an instance of the TestMemoryManager with the provided settings
    return TestMemoryManager(**mem_manager_settings)

@pytest.fixture(scope="function")
def registered_instance_override(request):
    return request.param if hasattr(request, 'param') else {}

@pytest.fixture(scope="function")
def registered_instance_settings(registered_instance_override):
    defaults = {
        'proportion': 0.5,
        'invalidate_all_hook': lambda: None,
    }
    if registered_instance_override:
        for key, value in registered_instance_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults

@pytest.fixture(scope="function")
def registered_instance(registered_instance_settings):
    return DummyClass(**registered_instance_settings)

@pytest.fixture
def registered_mgr(mgr, registered_instance):
    mgr.register(registered_instance,
                 proportion=registered_instance.proportion,
                 invalidate_all_hook=registered_instance.invalidate_all_hook,
                 )
    return mgr

class TestMemoryManager:
    @pytest.mark.nocudasim
    def test_instantiation(self, mgr):
        """ Test that the settings in the object match the settings fixture"""
        assert mgr.totalmem == 8*1024**3
        assert mgr.registry == {}
        assert mgr._mode in ["passive", "active"]
        assert mgr._allocator is not None
        assert isinstance(mgr.stream_groups, type(mgr.stream_groups))

    def test_register(self, registered_mgr, registered_instance):
        """ Test that the register method adds the instance to the
        appropriate pool, the proportion and cap are set correctly,
        an entry is added to registry, that the instance gets a stream"""
        mgr = registered_mgr
        instance = registered_instance
        assert id(instance) in mgr.registry
        settings = mgr.registry[id(instance)]
        assert settings.proportion == instance.proportion
        assert settings.cap == int(mgr.totalmem * instance.proportion)
        group = mgr.stream_groups.get_group(instance)
        assert group == "default"
        assert id(instance) in mgr.stream_groups.groups[group]
        assert mgr.stream_groups.get_stream(instance) is not None

    @pytest.mark.cupy
    def test_set_allocator(self, mgr):
        """ Test that set_allocator sets the allocator correctly
        test each of "cupy_async", "cupy", "default", checking that
        (self._allocator is CuPyAsyncNumbaManager, CuPySyncNumbaManager.
         NumbaCudaMemoryManager, respectively)"""
        mgr.set_allocator("cupy_async")
        assert mgr._allocator == CuPyAsyncNumbaManager
        mgr.set_allocator("cupy")
        assert mgr._allocator == CuPySyncNumbaManager
        mgr.set_allocator("default")
        assert mgr._allocator == NumbaCUDAMemoryManager
        with pytest.raises(ValueError):
            mgr.set_allocator("invalid")

    def test_set_limit_mode(self, mgr):
        """ Test that set_limit_mode assigns the mode correctly,
        and that it raises ValueError if an invalid mode is passed"""
        mgr.set_limit_mode("active")
        assert mgr._mode == "active"
        mgr.set_limit_mode("passive")
        assert mgr._mode == "passive"
        with pytest.raises(ValueError):
            mgr.set_limit_mode("invalid")

    @pytest.mark.nocudasim
    def test_get_stream(self, registered_mgr, registered_instance):
        """test that get_stream successfully passes a different stream for
        instances registered in different stream groups"""
        mgr = registered_mgr
        instance = registered_instance
        stream = mgr.get_stream(instance)
        assert isinstance(stream, Stream)
        # Register another instance in a new group
        inst2 = DummyClass()
        mgr.register(inst2, stream_group="other")
        stream2 = mgr.get_stream(inst2)
        assert stream2 is not None
        assert int(stream.handle.value) != int(stream2.handle.value)

    def test_change_stream_group(self, registered_mgr, registered_instance):
        """test that change_stream_group changes the stream group of an
        instance, and that it raises ValueError if the instance wasn't
        already in a group"""
        mgr = registered_mgr
        instance = registered_instance
        mgr.change_stream_group(instance, "other")
        assert id(instance) in mgr.stream_groups.groups["other"]
        dummy = DummyClass()
        with pytest.raises(ValueError):
            mgr.change_stream_group(dummy, "newgroup")

    @pytest.mark.nocudasim
    def test_reinit_streams(self, registered_mgr, registered_instance):
        """test that reinit_streams causess a different stream to be
        returned from get_stream"""
        mgr = registered_mgr
        instance = registered_instance
        stream1 = mgr.get_stream(instance)
        mgr.reinit_streams()
        stream2 = mgr.get_stream(instance)
        assert int(stream1.handle.value) != int(stream2.handle.value)

    def test_invalidate_all(self, registered_mgr, registered_instance):
        """Add a new instance with a measurable invalidate hook, and check
        that it is called when invalidate all is called"""
        mgr = registered_mgr
        called = {"flag": False}
        def hook():
            called["flag"] = True
        inst = DummyClass()
        mgr.register(inst, invalidate_all_hook=hook)
        mgr.invalidate_all()
        assert called["flag"] is True

    @pytest.mark.parametrize("registered_instance_override", [{"proportion":
                                                               None}],
                             indirect=True)
    def test_set_manual_limit_mode(self, registered_mgr, registered_instance):
        """Test that set_manual_limit_mode sets the instance to manual mode,
        and that it raises ValueError if the instance is already in manual mode"""
        mgr = registered_mgr
        instance = registered_instance
        instance_id = id(instance)
        mgr.set_manual_limit_mode(instance, 0.3)
        assert instance_id in mgr._manual_pool
        with pytest.raises(ValueError):
            mgr.set_manual_limit_mode(instance, 0.2)

    def test_set_auto_limit_mode(self, registered_mgr, registered_instance):
        """Test that set_auto_limit_mode sets the instance to auto mode"""
        mgr = registered_mgr
        instance = registered_instance
        mgr.set_auto_limit_mode(instance)
        instance_id = id(instance)
        assert instance_id not in mgr._manual_pool

    @pytest.mark.parametrize("registered_instance_override", [{"proportion":
                                                               0.5}],
                             indirect=True)
    def test_proportion(self, registered_mgr, registered_instance):
        """Test that proportion returns the requested proportion if set,
        and 1.0 if not set (auto)"""
        mgr = registered_mgr
        instance = registered_instance
        assert mgr.proportion(instance) == instance.proportion
        # Register auto instance
        inst2 = DummyClass(proportion=None)
        mgr.register(inst2, proportion=None)
        assert mgr.proportion(inst2) == 0.5

    def test_cap(self, registered_mgr, registered_instance):
        """Test that cap returns the correct value"""
        proportion = registered_instance.proportion if registered_instance.proportion else 1.0
        testcap = registered_mgr.cap(registered_instance)
        free, total = registered_mgr.get_memory_info()
        solvercap = total * proportion
        assert testcap == solvercap

    @pytest.mark.parametrize("registered_instance_override", 
                             [{"proportion": None}],
                             indirect=True)
    def test_pool_proportions(self, registered_mgr, registered_instance):
        """Add a few auto and manual instances, check that the total manual
        and auto pool proportions match expected. Test add and rebalance
        methods along the way"""
        mgr = registered_mgr
        instance1 = registered_instance
        instance2 = DummyClass()
        instance3 = DummyClass()
        instance4 = DummyClass()
        instance5 = DummyClass()

        # already registered
        with pytest.raises(ValueError):
            mgr.register(instance1)
        mgr.set_limit_mode("active")

        assert mgr.proportion(instance1) == 1.0
        assert mgr.manual_pool_proportion == 0.0
        assert mgr.auto_pool_proportion == 1.0

        mgr.register(instance2)
        assert abs(mgr.proportion(instance2) - 0.5) < 1e-6
        assert abs(mgr.manual_pool_proportion - 0.0) < 1e-6
        assert abs(mgr.auto_pool_proportion - 1.0) < 1e-6

        mgr.register(instance3, proportion=0.5)
        assert abs(mgr.proportion(instance3) - 0.5) < 1e-6
        assert abs(mgr.manual_pool_proportion - 0.5) < 1e-6
        assert abs(mgr.auto_pool_proportion - 0.5) < 1e-6

        mgr.register(instance4, proportion=0.4)
        assert abs(mgr.proportion(instance4) - 0.4) < 1e-6
        assert abs(mgr.manual_pool_proportion - 0.9) < 1e-6
        assert abs(mgr.auto_pool_proportion - 0.1) < 1e-6

        mgr.set_auto_limit_mode(instance3)
        assert abs(mgr.proportion(instance3) - 0.2) < 1e-6
        assert abs(mgr.manual_pool_proportion - 0.4) < 1e-6
        assert abs(mgr.auto_pool_proportion - 0.6) < 1e-6

        mgr.set_manual_limit_mode(instance3, 0.3)
        assert abs(mgr.proportion(instance3) - 0.3) < 1e-6
        assert abs(mgr.manual_pool_proportion - 0.7) < 1e-6
        assert abs(mgr.auto_pool_proportion - 0.3) < 1e-6

        with pytest.raises(ValueError):
            mgr.register(instance5, proportion=0.3)


    def test_set_strides(self, mgr):
        """ test that the strides are set correctly"""
        # Default stride order, should be None
        req = ArrayRequest(shape=(2,3,4), dtype=np.float32, memory="device",
                           stride_order=("time","run","variable"))
        assert mgr.get_strides(req) is None
        # Custom stride order, should return a tuple
        req2 = ArrayRequest(shape=(2,3,4), dtype=np.float32, memory="device",
                            stride_order=("run","variable","time"))
        strides = mgr.get_strides(req2)
        # Manually computed strides for shape (2,3,4) and order (run, variable, time)
        # Should match what MemoryManager.get_strides returns
        itemsize = req2.dtype().itemsize
        expected = (12, 4, 24)
        assert strides == expected

    def test_set_global_stride_ordering(self, mgr):
        """Test that set_global_stride_ordering sets the stride order and invalidates all."""
        # Valid ordering
        mgr.set_global_stride_ordering(("run", "variable", "time"))
        assert mgr._stride_order == ("run", "variable", "time")
        # Invalid ordering
        with pytest.raises(ValueError):
            mgr.set_global_stride_ordering(("foo", "bar", "baz"))

    def test_process_request(self, registered_mgr, registered_instance):
        """Test process_request returns ArrayResponse with correct arrays and chunk count."""
        mgr = registered_mgr
        instance = registered_instance
        requests = {
            "arr1": ArrayRequest(shape=(8, 8, 8), dtype=np.float32, memory="device"),
            "arr2": ArrayRequest(shape=(4, 4, 4), dtype=np.float32, memory="device"),
        }
        response = mgr.request(instance, requests)
        assert isinstance(response, ArrayResponse)
        assert set(response.arr.keys()) == set(requests.keys())
        for key, arr in response.arr.items():
            assert arr.shape == requests[key].shape
            assert arr.dtype == requests[key].dtype
        assert isinstance(response.chunks, int)

    def test_allocate_all(self, registered_mgr, registered_instance):
        """Test allocate_all allocates arrays for all requests and stores them in allocations."""
        mgr = registered_mgr
        instance = registered_instance
        requests = {
            "arr1": ArrayRequest(shape=(2, 2, 2), dtype=np.float32, memory="device"),
            "arr2": ArrayRequest(shape=(2, 2, 2), dtype=np.float32, memory="device"),
        }
        arrays = mgr.allocate_all(instance, requests)
        assert set(arrays.keys()) == set(requests.keys())
        for key, arr in arrays.items():
            assert arr.shape == requests[key].shape
            assert arr.dtype == requests[key].dtype
            assert key in mgr.registry[id(instance)].allocations

    @pytest.mark.nocudasim
    def test_allocate(self, mgr):
        """Test allocate returns correct array type and shape for each memory type."""
        for mem_type in ["device", "mapped", "pinned"]:
            arr = mgr.allocate(shape=(2, 2), dtype=np.float32, memory_type=mem_type)
            if mem_type in ["device", "mapped"]:
                assert hasattr(arr, "__cuda_array_interface__")
            else:
                assert isinstance(arr, np.ndarray)
            assert arr.shape == (2, 2)
            assert arr.dtype == np.float32
        with pytest.raises(NotImplementedError):
            mgr.allocate(shape=(1, 1), dtype=np.float32, memory_type="managed")
        with pytest.raises(ValueError):
            mgr.allocate(shape=(1, 1), dtype=np.float32, memory_type="invalid")

    def test_free(self, registered_mgr, registered_instance):
        """Test free removes allocation by key from all instances."""
        mgr = registered_mgr
        instance = registered_instance
        arr = np.zeros((2, 2), dtype=np.float32)
        mgr.registry[id(instance)].add_allocation("foo", arr)
        mgr.free("foo")
        assert "foo" not in mgr.registry[id(instance)].allocations

    def test_free_all(self, registered_mgr, registered_instance):
        """Test free_all removes all allocations from all instances."""
        mgr = registered_mgr
        instance = registered_instance
        arr = np.zeros((2, 2), dtype=np.float32)
        mgr.registry[id(instance)].add_allocation("foo", arr)
        mgr.registry[id(instance)].add_allocation("bar", arr)
        mgr.free_all()
        assert mgr.registry[id(instance)].allocations == {}

    def test_get_chunks(self, registered_mgr, registered_instance):
        """Test get_chunks returns correct chunk count based on available memory."""
        mgr = registered_mgr
        instance = registered_instance
        # Passive mode
        mgr.set_limit_mode("passive")
        chunks = mgr.get_chunks(instance, 100)
        assert isinstance(chunks, int)
        assert chunks == 1

        chunks = mgr.get_chunks(instance, int(2 * 1024**3))
        assert isinstance(chunks, int)
        assert chunks == 2

        chunks = mgr.get_chunks(instance, int(2.5 * 1024**3))
        assert chunks == 3

        # Active mode
        mgr.set_limit_mode("active")
        mgr.set_manual_proportion(instance, 0.1)  # 800MB
        chunks = mgr.get_chunks(instance, int(2 * 1024**3))
        assert chunks == 3

        chunks = mgr.get_chunks(instance, int(2.5 * 1024**3))
        assert chunks == 4

    def test_get_memory_info(self, mgr):
        """Test get_memory_info returns tuple of free and total memory."""
        free, total = mgr.get_memory_info()
        assert free == 1 * 1024**3
        assert total == 8 * 1024**3

    def test_get_strides(self, mgr):
        """Test get_strides returns correct strides for 3D arrays and None for default/correct ordering or 2D arrays."""
        # 3D array, default stride order
        req_default = ArrayRequest(
            shape=(2, 3, 4),
            dtype=np.float32,
            memory="device",
            stride_order=("time", "run", "variable"),
        )
        assert mgr.get_strides(req_default) is None
        # 3D array, custom stride order
        req_custom = ArrayRequest(
            shape=(2, 3, 4),
            dtype=np.float32,
            memory="device",
            stride_order=("run", "variable", "time"),
        )
        strides = mgr.get_strides(req_custom)
        # Should match manual calculation
        itemsize = req_custom.dtype().itemsize
        expected = (12, 4, 24)
        assert strides == expected
        # 2D array, should always be None
        req_2d = ArrayRequest(
            shape=(3, 4),
            dtype=np.float32,
            memory="device",
            stride_order=("variable", "run"),
        )
        assert mgr.get_strides(req_2d) is None

    @pytest.mark.parametrize("registered_instance_override", 
                             [{"proportion": None}],
                             indirect=True)
    def test_manual_pool_proportion(self, registered_mgr, registered_instance):
        """Test manual_pool_proportion property returns correct sum."""
        mgr = registered_mgr
        instance = registered_instance
        mgr.set_manual_limit_mode(instance, 0.3)
        assert abs(mgr.manual_pool_proportion - 0.3) < 1e-6

    def test_auto_pool_proportion(self, registered_mgr, registered_instance):
        """Test auto_pool_proportion property returns correct sum."""
        mgr = registered_mgr
        instance = registered_instance
        # Add an auto instance
        inst2 = DummyClass()
        mgr.register(inst2)
        expected = 1.0 - mgr.registry[id(instance)].proportion
        assert abs(mgr.auto_pool_proportion - expected) < 1e-6

    def test_set_manual_proportion(self, registered_mgr, registered_instance):
        """Test set_manual_proportion sets manual proportion and updates registry."""
        mgr = registered_mgr
        instance = registered_instance
        instance_id = id(instance)
        mgr.set_manual_proportion(instance, 0.4)
        assert mgr.registry[id(instance)].proportion == 0.4
        assert instance_id in mgr._manual_pool

    def test_rebalance_auto_pool(self, mgr):
        """Test _rebalance_auto_pool splits available proportion among auto pool."""
        inst1 = DummyClass()
        inst2 = DummyClass()
        mgr.register(inst1)
        mgr.register(inst2)
        assert mgr.registry[id(inst1)].proportion == 0.5
        assert mgr.auto_pool_proportion == 1.0
        mgr.set_manual_limit_mode(inst2, 0.75)
        mgr._rebalance_auto_pool()
        assert abs(mgr.registry[id(inst1)].proportion - 0.25) < 1e-6
        assert abs(mgr.auto_pool_proportion - 0.25) < 1e-6


    def test_get_chunks_warnings(self, registered_mgr, registered_instance):
        registered_mgr._mode = "passive"
        # large request to trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chunks = registered_mgr.get_chunks(registered_instance, 25 * 1024**3)
            assert chunks == 25
            assert any("exceeds available VRAM" in str(wi.message) for wi in w)

    def test_get_chunks_active_without_register(self, registered_mgr,
                                                registered_instance):
        # active mode without registering should error
        instance = DummyClass()
        registered_mgr._mode = "active"
        with pytest.raises(AttributeError):
            registered_mgr.get_chunks(instance, 100)

    def test_full_request(self, registered_mgr, registered_instance):
        """Run through a full request cycle, matching a likely input with
        the expected output from the array as a whole"""
        # Registered instance has 1GB available
        instance_id = id(registered_instance)

        arr1 = ArrayRequest(
            shape=(1000, 1000, 1000),
            dtype=np.float32,
            memory="device",
            stride_order=("time", "run", "variable"),
        )
        arr2 = ArrayRequest(
            shape=(100, 100, 100),
            dtype=np.float32,
            memory="mapped",
            stride_order=("time", "variable", "run"),
        )
        arr3 = ArrayRequest(shape=(100,100,100),
                            dtype=np.float32,
                            memory="pinned",
                            stride_order=("run","variable","time")
        )
        arr4 = ArrayRequest(shape=(100,100),
                            dtype=np.float32,
                            memory="device",
                            stride_order=("variable","run"))
        arr5 = ArrayRequest(shape=(100,100),
                            dtype=np.float32,
                            memory="device",
                            stride_order=("variable","run"))
        arr6 = ArrayRequest(
            shape=(100, 100),
            dtype=np.float32,
            memory="device",
            stride_order=("variable", "run"),
        )
        requests = [arr1, arr2, arr3, arr4, arr5, arr6]
        request_names = ["arr1", "arr2", "arr3", "arr4", "arr5", "arr6"]
        request_dict = dict(zip(request_names, requests))
        response = registered_mgr.request(registered_instance, request_dict)

        # This was very un-pythonically overflowing, hence the explicit int64
        request_size = np.sum(tuple(arr.size for arr in requests),
                              dtype=np.int64)



        #Did we get one response for each request?
        assert len(response.arr) == len(requests), "response dict"
        assert set(response.arr.keys()) == set(request_names)

        # Did we chunk correctly?
        expected_chunks = int(np.ceil(request_size / 1024**3))
        assert response.chunks == expected_chunks, "chunks"

        total_allocation = 0
        for key, request in request_dict.items():
            allocated_array = response.arr[key]

            # are the arrays the correct memory type?
            if request.memory == "device":
                assert isinstance(allocated_array, DeviceNDArrayBase), "memtype"
            if request.memory == "pinned":
                assert isinstance(allocated_array, np.ndarray), "memtype"
            if request.memory == "mapped":
                assert isinstance(allocated_array, MappedNDArray), "memtype"

            # Of the correct shape?
            expected_shape = tuple(
                int(dim / expected_chunks) if request.stride_order[i] == "run"
                else dim for i, dim in enumerate(request.shape)
            )
            assert allocated_array.shape == expected_shape, "chunked shape"

            # With correct strides?
            array_native_order = request.stride_order
            desired_order = ("time", "run", "variable")
            shape = expected_shape
            itemsize = request.dtype().itemsize

            if len(shape) != 3:
                expected_strides = (itemsize * shape[1], itemsize)
            else:
                dims = {name: size for name, size in zip(array_native_order, shape)}
                strides = {}
                current_stride = itemsize

                for name in reversed(desired_order):
                    strides[name] = current_stride
                    current_stride *= dims[name]
                expected_strides = tuple(strides[dim] for dim in array_native_order)

            assert allocated_array.strides == expected_strides, "strides"

            # Have we recorded the allocation correctly?
            total_allocation += allocated_array.nbytes
            assert instance_id in registered_mgr.registry
            assert key in registered_mgr.registry[instance_id].allocations, \
                "allocation recorded"

        #Does the allocated size match expected?
        assert total_allocation == \
                registered_mgr.registry[instance_id].allocated_bytes, \
                "allocated size"

        # What happens when we free it
        for name, array in response.arr.items():
            registered_mgr.free(name)

        assert registered_mgr.registry[instance_id].allocated_bytes == 0, \
                "all freed"
        assert registered_mgr.registry[instance_id].allocations == {}, \
                "all freed"
