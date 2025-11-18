import warnings

import pytest
from cubie.memory.cupy_emm import CuPyAsyncNumbaManager, CuPySyncNumbaManager
from cubie.cuda_simsafe import (
    NumbaCUDAMemoryManager,
    Stream,
)

from cubie.memory.mem_manager import (
    MemoryManager,
    ArrayRequest,
    ArrayResponse,
    InstanceMemorySettings,
)

import numpy as np


# ========================== dataclasses and helpers ======================== #
class DummyClass:
    def __init__(self, proportion=None, invalidate_all_hook=None):
        self.proportion = proportion
        self.invalidate_all_hook = invalidate_all_hook


@pytest.fixture(scope="function")
def instance_settings_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def instance_settings(instance_settings_override):
    defaults = {
        "proportion": 0.5,
        "invalidate_hook": lambda: None,
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
        assert (
            instance_settings_obj.proportion == instance_settings["proportion"]
        )
        assert callable(instance_settings_obj.invalidate_hook)
        assert isinstance(instance_settings_obj.allocations, dict)

    def test_add_allocation(self, instance_settings_obj):
        # test that add_allocation adds the reference to the allocations dict
        arr = np.ndarray((10,), dtype=np.float64)
        instance_settings_obj.add_allocation("foo", arr)
        assert "foo" in instance_settings_obj.allocations
        assert instance_settings_obj.allocations["foo"] is arr

    def test_free(self, instance_settings_obj):
        # test that free removes the reference to the allocations dict
        arr = np.ndarray((10,), dtype=np.float64)
        instance_settings_obj.add_allocation("foo", arr)
        instance_settings_obj.free("foo")
        assert "foo" not in instance_settings_obj.allocations

    def test_free_all(self, instance_settings_obj):
        # test that free_all removes all references to the allocations dict
        arr1 = np.ndarray((10,), dtype=np.float64)
        arr2 = np.ndarray((20,), dtype=np.float64)
        instance_settings_obj.add_allocation("foo", arr1)
        instance_settings_obj.add_allocation("bar", arr2)
        instance_settings_obj.free_all()
        assert instance_settings_obj.allocations == {}

    @pytest.mark.nocudasim
    def test_allocated_bytes(self, instance_settings_obj):
        # test that the allocated_bytes property returns the correct value
        arr1 = np.ndarray((100,), dtype=np.float64)
        arr2 = np.ndarray((25,), dtype=np.float64)
        instance_settings_obj.add_allocation("foo", arr1)
        instance_settings_obj.add_allocation("bar", arr2)
        expected_bytes = arr1.nbytes + arr2.nbytes
        assert instance_settings_obj.allocated_bytes == expected_bytes
        instance_settings_obj.free("foo")
        assert instance_settings_obj.allocated_bytes == arr2.nbytes


# ========================= Memory Manager Class =========================== #
@pytest.fixture
def fixed_mem_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def fixed_mem_settings(fixed_mem_override):
    defaults = {"free": 1 * 1024**3, "total": 8 * 1024**3}
    if fixed_mem_override:
        for key, value in fixed_mem_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults


@pytest.fixture(scope="function")
def mem_manager_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def mem_manager_settings(mem_manager_override):
    defaults = {"mode": "passive", "stride_order": ("time", "run", "variable")}
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
            free = fixed_mem_settings["free"]
            total = fixed_mem_settings["total"]
            return free, total

    # Create an instance of the TestMemoryManager with the provided settings
    return TestMemoryManager(**mem_manager_settings)


@pytest.fixture(scope="function")
def registered_instance_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def registered_instance_settings(registered_instance_override):
    defaults = {
        "proportion": 0.5,
        "invalidate_all_hook": lambda: None,
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
    mgr.register(
        registered_instance,
        proportion=registered_instance.proportion,
        invalidate_cache_hook=registered_instance.invalidate_all_hook,
    )
    return mgr


class TestMemoryManager:
    @pytest.mark.nocudasim
    def test_instantiation(self, mgr):
        """Test that the settings in the object match the settings fixture"""
        assert mgr.totalmem == 8 * 1024**3
        assert mgr.registry == {}
        assert mgr._mode in ["passive", "active"]
        assert mgr._allocator is not None
        assert isinstance(mgr.stream_groups, type(mgr.stream_groups))

    def test_register(self, registered_mgr, registered_instance):
        """Test that the register method adds the instance to the
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
        """Test that set_allocator sets the allocator correctly
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
        """Test that set_limit_mode assigns the mode correctly,
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
        mgr.register(inst, invalidate_cache_hook=hook)
        mgr.invalidate_all()
        assert called["flag"] is True

    @pytest.mark.parametrize(
        "registered_instance_override", [{"proportion": None}], indirect=True
    )
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

    @pytest.mark.parametrize(
        "registered_instance_override", [{"proportion": 0.5}], indirect=True
    )
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
        proportion = (
            registered_instance.proportion
            if registered_instance.proportion
            else 1.0
        )
        testcap = registered_mgr.cap(registered_instance)
        free, total = registered_mgr.get_memory_info()
        solvercap = total * proportion
        assert testcap == solvercap

    @pytest.mark.parametrize(
        "registered_instance_override", [{"proportion": None}], indirect=True
    )
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
        """test that the strides are set correctly"""
        # Default stride order, should be None
        req = ArrayRequest(
            shape=(2, 3, 4),
            dtype=np.float32,
            memory="device",
            stride_order=("time", "run", "variable"),
        )
        assert mgr.get_strides(req) is None
        # Custom stride order, should return a tuple
        req2 = ArrayRequest(
            shape=(2, 3, 4),
            dtype=np.float32,
            memory="device",
            stride_order=("run", "variable", "time"),
        )
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

    def test_process_request(self, mgr):
        """Test single_request calls allocation hook with correct ArrayResponse."""
        # Create instance with callback to capture response
        instance = DummyClass()
        callback_called = {"flag": False, "response": None}

        def allocation_hook(response):
            callback_called["flag"] = True
            callback_called["response"] = response

        mgr.register(instance, allocation_ready_hook=allocation_hook)

        requests = {
            "arr1": ArrayRequest(
                shape=(8, 8, 8), dtype=np.float32, memory="device"
            ),
            "arr2": ArrayRequest(
                shape=(4, 4, 4), dtype=np.float32, memory="device"
            ),
        }
        mgr.single_request(instance, requests)

        # Check that callback was called with correct response
        assert callback_called["flag"] is True
        response = callback_called["response"]
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
            "arr1": ArrayRequest(
                shape=(2, 2, 2), dtype=np.float32, memory="device"
            ),
            "arr2": ArrayRequest(
                shape=(2, 2, 2), dtype=np.float32, memory="device"
            ),
        }
        stream = mgr.get_stream(instance)
        arrays = mgr.allocate_all(requests, id(instance), stream)
        assert set(arrays.keys()) == set(requests.keys())
        for key, arr in arrays.items():
            assert arr.shape == requests[key].shape
            assert arr.dtype == requests[key].dtype
            assert key in mgr.registry[id(instance)].allocations

    @pytest.mark.nocudasim
    def test_allocate(self, mgr):
        """Test allocate returns correct array type and shape for each memory type."""
        for mem_type in ["device", "mapped", "pinned"]:
            arr = mgr.allocate(
                shape=(2, 2), dtype=np.float32, memory_type=mem_type
            )
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
        mgr.registry[id(instance)].add_alocation("foo", arr)
        mgr.registry[id(instance)].add_allocation("bar", arr)
        mgr.free_all()
        assert mgr.registry[id(instance)].allocations == {}

    def test_get_chunks(self, mgr):
        """Test get_chunks returns correct chunk count based on available memory."""
        # Test with specific request size and available memory
        request_size = 100
        available = 200
        chunks = mgr.get_chunks(request_size, available)
        assert isinstance(chunks, int)
        assert chunks == 1

        request_size = int(2 * 1024**3)
        available = int(1 * 1024**3)
        chunks = mgr.get_chunks(request_size, available)
        assert chunks == 2

        request_size = int(2.5 * 1024**3)
        available = int(1 * 1024**3)
        chunks = mgr.get_chunks(request_size, available)
        assert chunks == 3

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

    @pytest.mark.parametrize(
        "registered_instance_override", [{"proportion": None}], indirect=True
    )
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

    def test_get_available_single(self, registered_mgr, registered_instance):
        """Test get_available_single returns correct available memory for an instance."""
        mgr = registered_mgr
        instance = registered_instance
        instance_id = id(instance)

        # Passive mode should return free memory
        mgr.set_limit_mode("passive")
        available = mgr.get_available_single(instance_id)
        free, total = mgr.get_memory_info()
        assert available == free

        # Active mode should consider instance cap and allocated bytes
        mgr.set_limit_mode("active")
        available = mgr.get_available_single(instance_id)
        settings = mgr.registry[instance_id]
        expected = min(settings.cap - settings.allocated_bytes, free)
        assert available == expected

    def test_get_available_group(self, mgr):
        """Test get_available_group returns correct available memory for a stream group."""
        inst1 = DummyClass()
        inst2 = DummyClass()
        mgr.register(inst1, stream_group="test_group")
        mgr.register(inst2, stream_group="test_group")

        # Passive mode should return free memory
        mgr.set_limit_mode("passive")
        available = mgr.get_available_group("test_group")
        free, total = mgr.get_memory_info()
        assert available == free

        # Active mode should consider group totals
        mgr.set_limit_mode("active")
        available = mgr.get_available_group("test_group")
        total_cap = mgr.registry[id(inst1)].cap + mgr.registry[id(inst2)].cap
        total_allocated = (
            mgr.registry[id(inst1)].allocated_bytes
            + mgr.registry[id(inst2)].allocated_bytes
        )
        expected = min(total_cap - total_allocated, free)
        assert available == expected

    def test_get_stream_group(self, registered_mgr, registered_instance):
        """Test get_stream_group returns the correct group name."""
        mgr = registered_mgr
        instance = registered_instance
        group = mgr.get_stream_group(instance)
        assert group == "default"

        # Test with custom group
        inst2 = DummyClass()
        mgr.register(inst2, stream_group="custom")
        assert mgr.get_stream_group(inst2) == "custom"

    def test_check_requests(self, mgr):
        """Test _check_requests validates request format correctly."""
        # Valid requests should pass
        valid_requests = {
            "arr1": ArrayRequest(
                shape=(2, 2), dtype=np.float32, memory="device"
            ),
            "arr2": ArrayRequest(
                shape=(3, 3), dtype=np.float64, memory="mapped"
            ),
        }
        mgr._check_requests(valid_requests)  # Should not raise

        # Invalid dict type should raise TypeError
        with pytest.raises(TypeError):
            mgr._check_requests("not a dict")

        # Invalid request values should raise TypeError
        invalid_requests = {
            "arr1": ArrayRequest(
                shape=(2, 2), dtype=np.float32, memory="device"
            ),
            "arr2": "not an ArrayRequest",
        }
        with pytest.raises(TypeError):
            mgr._check_requests(invalid_requests)

    def test_queue_request(self, registered_mgr, registered_instance):
        """Test queue_request adds requests to the queue correctly."""
        mgr = registered_mgr
        instance = registered_instance

        requests = {
            "arr1": ArrayRequest(
                shape=(2, 2), dtype=np.float32, memory="device"
            ),
        }

        mgr.queue_request(instance, requests)
        stream_group = mgr.get_stream_group(instance)
        assert stream_group in mgr._queued_allocations
        assert id(instance) in mgr._queued_allocations[stream_group]
        assert mgr._queued_allocations[stream_group][id(instance)] == requests

        # Test that subsequent requests from same instance overwrite (not append)
        requests2 = {
            "arr2": ArrayRequest(
                shape=(3, 3), dtype=np.float32, memory="device"
            ),
        }
        mgr.queue_request(instance, requests2)
        # Should only have the latest request
        assert mgr._queued_allocations[stream_group][id(instance)] == requests2

    def test_chunk_arrays(self, mgr):
        """Test chunk_arrays divides array shapes correctly along specified axis."""
        requests = {
            "arr1": ArrayRequest(
                shape=(100, 200, 50),
                dtype=np.float32,
                memory="device",
                stride_order=("time", "run", "variable"),
            ),
            "arr2": ArrayRequest(
                shape=(50, 400, 25),
                dtype=np.float32,
                memory="device",
                stride_order=("time", "run", "variable"),
            ),
        }

        # Chunk by run dimension (index 1)
        chunked = mgr.chunk_arrays(requests, numchunks=4, axis="run")

        # arr1: (100, 200, 50) -> (100, 50, 50) since 200/4 = 50
        assert chunked["arr1"].shape == (100, 50, 50)  # 200/4 = 50
        assert chunked["arr2"].shape == (50, 100, 25)  # 400/4 = 100

        # Chunk by time dimension (index 0)
        chunked_time = mgr.chunk_arrays(requests, numchunks=2, axis="time")
        assert chunked_time["arr1"].shape == (50, 200, 50)  # 100/2 = 50
        assert chunked_time["arr2"].shape == (25, 400, 25)  # 50/2 = 25

    def test_single_request(self, registered_mgr, registered_instance):
        """Test single_request processes individual requests correctly."""
        # Create instance with callback to capture response
        instance = DummyClass()
        callback_called = {"flag": False, "response": None}

        def allocation_hook(response):
            callback_called["flag"] = True
            callback_called["response"] = response

        registered_mgr.register(
            instance, allocation_ready_hook=allocation_hook
        )

        requests = {
            "arr1": ArrayRequest(
                shape=(4, 4, 4), dtype=np.float32, memory="device"
            ),
            "arr2": ArrayRequest(
                shape=(2, 2, 2), dtype=np.float32, memory="mapped"
            ),
        }

        registered_mgr.single_request(instance, requests)

        # Check that callback was called with correct response
        assert callback_called["flag"] is True
        response = callback_called["response"]
        assert isinstance(response, ArrayResponse)
        assert set(response.arr.keys()) == set(requests.keys())
        assert isinstance(response.chunks, int)

        # Arrays should be allocated and registered
        for key in requests.keys():
            assert key in registered_mgr.registry[id(instance)].allocations

    def test_allocate_queue_single_instance(self, mgr):
        """Test allocate_queue with single instance in queue."""
        instance = DummyClass()
        callback_called = {"flag": False, "response": ArrayResponse()}

        def allocation_hook(response):
            callback_called["flag"] = True
            callback_called["response"] = response

        mgr.register(instance, allocation_ready_hook=allocation_hook)

        requests = {
            "arr1": ArrayRequest(
                shape=(2, 2, 2), dtype=np.float32, memory="device"
            ),
        }

        mgr.queue_request(instance, requests)
        mgr.allocate_queue(instance)

        # Check that callback was called
        assert callback_called["flag"] is True
        assert isinstance(callback_called["response"], ArrayResponse)
        response = callback_called["response"]

        assert response.arr["arr1"].shape == (2, 2, 2)
        assert response.arr["arr1"].dtype == np.float32
        assert response.chunks == 1

    def test_allocate_queue_multiple_instances_group_limit(self, mgr):
        """Test allocate_queue with multiple instances using group limit."""
        inst1 = DummyClass()
        inst2 = DummyClass()
        callbacks_called = {
            "inst1": {"flag": False, "response": ArrayResponse()},
            "inst2": {"flag": False, "response": ArrayResponse()},
        }

        def hook1(response):
            callbacks_called["inst1"]["flag"] = True
            callbacks_called["inst1"]["response"] = response

        def hook2(response):
            callbacks_called["inst2"]["flag"] = True
            callbacks_called["inst2"]["response"] = response

        mgr.register(inst1, allocation_ready_hook=hook1, stream_group="test")
        mgr.register(inst2, allocation_ready_hook=hook2, stream_group="test")

        requests1 = {
            "arr1": ArrayRequest(
                shape=(2, 2, 2), dtype=np.float32, memory="device"
            )
        }
        requests2 = {
            "arr2": ArrayRequest(
                shape=(3, 3, 3), dtype=np.float32, memory="device"
            )
        }

        mgr.queue_request(inst1, requests1)
        mgr.queue_request(inst2, requests2)
        mgr.allocate_queue(inst1, limit_type="group")

        # Both callbacks should be called
        assert callbacks_called["inst1"]["flag"] is True
        assert callbacks_called["inst2"]["flag"] is True

    @pytest.mark.parametrize(
        "fixed_mem_override, limit_type, chunk_axis",
        [
            [{"total": 512 * 1024**2}, "instance", "run"],
            [{"total": 512 * 1024**2}, "group", "run"],
            [{"total": 512 * 1024**2}, "instance", "time"],
            [{"total": 512 * 1024**2}, "group", "time"],
        ],
        indirect=["fixed_mem_override"],
    )
    def test_allocate_queue_multiple_instances_instance_limit(
        self, mgr, limit_type, chunk_axis
    ):
        """Test allocate_queue with multiple instances using instance limit."""
        inst1 = DummyClass()
        inst2 = DummyClass()
        callbacks_called = {
            "inst1": {"flag": False, "response": ArrayRequest()},
            "inst2": {"flag": False, "response": ArrayRequest()},
        }

        def hook1(response):
            callbacks_called["inst1"]["flag"] = True
            callbacks_called["inst1"]["response"] = response

        def hook2(response):
            callbacks_called["inst2"]["flag"] = True
            callbacks_called["inst2"]["response"] = response

        mgr.register(inst1, allocation_ready_hook=hook1, stream_group="test")
        mgr.register(inst2, allocation_ready_hook=hook2, stream_group="test")
        mgr.set_limit_mode("active")
        requests1 = {
            "arr": ArrayRequest(
                shape=(4, 4, 4), dtype=np.float32, memory="device"
            )
        }
        requests2 = {
            "arr": ArrayRequest(
                shape=(1000, 256, 1024),
                # ~4x instance
                dtype=np.float32,
                memory="device",
            )
        }

        mgr.queue_request(inst1, requests1)
        mgr.queue_request(inst2, requests2)
        mgr.allocate_queue(inst1, limit_type=limit_type, chunk_axis=chunk_axis)

        # Both callbacks should be called
        assert callbacks_called["inst1"]["flag"] is True
        assert callbacks_called["inst2"]["flag"] is True
        response1 = callbacks_called["inst1"]["response"]
        response2 = callbacks_called["inst2"]["response"]

        if limit_type == "instance":
            expected_chunks = 4
        else:
            expected_chunks = 2
        if chunk_axis == "run":
            chunk_index = 1
        elif chunk_axis == "time":
            chunk_index = 0

        shape_1 = np.asarray([4, 4, 4])
        shape_2 = np.asarray([1000, 256, 1024])
        shape_1[chunk_index] = shape_1[chunk_index] // expected_chunks
        shape_2[chunk_index] = shape_2[chunk_index] // expected_chunks

        assert response1.arr["arr"].shape == tuple(shape_1)
        assert response1.arr["arr"].dtype == np.float32
        assert response1.chunks == expected_chunks

        assert response2.arr["arr"].shape == tuple(shape_2)
        assert response2.arr["arr"].dtype == np.float32
        assert response2.chunks == expected_chunks

    def test_allocate_queue_empty_queue(
        self, registered_mgr, registered_instance
    ):
        """Test allocate_queue with empty queue returns None."""
        mgr = registered_mgr
        instance = registered_instance

        result = mgr.allocate_queue(instance)
        assert result is None

    def test_is_grouped(self, mgr):
        """Test is_grouped returns correct grouping status for instances."""
        # Test instance in default group (should return False)
        inst_default = DummyClass()
        mgr.register(inst_default, stream_group="default")
        assert mgr.is_grouped(inst_default) is False

        # Test single instance in named group (should return False)
        inst_single = DummyClass()
        mgr.register(inst_single, stream_group="single_group")
        assert mgr.is_grouped(inst_single) is False

        # Test multiple instances in named group (should return True)
        inst_group1 = DummyClass()
        inst_group2 = DummyClass()
        mgr.register(inst_group1, stream_group="multi_group")
        mgr.register(inst_group2, stream_group="multi_group")
        assert mgr.is_grouped(inst_group1) is True
        assert mgr.is_grouped(inst_group2) is True

        # Test three instances in named group (should return True)
        inst_group3 = DummyClass()
        mgr.register(inst_group3, stream_group="multi_group")
        assert mgr.is_grouped(inst_group1) is True
        assert mgr.is_grouped(inst_group2) is True
        assert mgr.is_grouped(inst_group3) is True

        # Test after moving instance to different group
        mgr.change_stream_group(inst_group3, "new_group")
        # inst_group3 should now be alone in "new_group"
        assert mgr.is_grouped(inst_group3) is False
        # inst_group1 and inst_group2 should still be grouped in "multi_group"
        assert mgr.is_grouped(inst_group1) is True
        assert mgr.is_grouped(inst_group2) is True

    @pytest.mark.parametrize(
        "fixed_mem_override", [{"total": 512 * 1024**2}], indirect=True
    )
    def test_get_available_single_low_memory_warning(
        self, registered_mgr, registered_instance
    ):
        """Test get_available_single issues warning when memory usage is high."""
        mgr = registered_mgr
        instance = registered_instance  # initializes with 0.5 of VRAM assigned
        instance_id = id(instance)

        mgr.set_limit_mode("active")
        requests1 = {
            "arr1": ArrayRequest(
                shape=(250, 1024, 256),  # >95%
                dtype=np.float32,
                memory="device",
            )
        }

        # Add some fake allocation to trigger low memory warning
        # allocation
        mgr.single_request(instance, requests1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mgr.get_available_single(instance_id)
            # Should warn about high memory usage
            assert any(
                "95% of it's allotted memory" in str(wi.message) for wi in w
            )

    @pytest.mark.parametrize(
        "fixed_mem_override", [{"total": 512 * 1024**2}], indirect=True
    )
    def test_get_available_group_low_memory_warning(self, mgr):
        """Test get_available_group issues warning when group memory usage is high."""
        inst1 = DummyClass()
        inst2 = DummyClass()
        mgr.register(inst1, stream_group="test", proportion=0.25)
        mgr.register(inst2, stream_group="test", proportion=0.25)
        mgr.set_limit_mode("passive")
        requests1 = {
            "arr1": ArrayRequest(
                shape=(250, 1024, 256),  # >95%
                dtype=np.float32,
                memory="device",
            )
        }
        mgr.single_request(inst1, requests1)
        # Add fake allocations to trigger warning

        mgr.set_limit_mode("active")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mgr.get_available_group("test")
            # Should warn about high group memory usage
            assert any("has used more than 95%" in str(wi.message) for wi in w)

    @pytest.mark.nocudasim
    def test_to_device(self, registered_mgr, registered_instance):
        """Test to_device copies values to allocated device arrays correctly."""

        mgr = registered_mgr
        instance = registered_instance

        # Allocate device arrays through the memory manager
        requests = {
            "arr1": ArrayRequest(
                shape=(3, 4), dtype=np.float32, memory="device"
            ),
            "arr2": ArrayRequest(
                shape=(2, 3), dtype=np.float64, memory="device"
            ),
        }

        stream = mgr.get_stream(instance)
        device_arrays = mgr.allocate_all(requests, id(instance), stream)

        # Create host arrays with test data
        host_arr1 = np.arange(12, dtype=np.float32).reshape(3, 4)
        host_arr2 = np.arange(6, dtype=np.float64).reshape(2, 3) * 2.5

        # Copy to device using to_device method
        from_arrays = [host_arr1, host_arr2]
        to_arrays = [device_arrays["arr1"], device_arrays["arr2"]]

        mgr.to_device(instance, from_arrays, to_arrays)

        # Synchronize stream to ensure copy is complete
        stream.synchronize()

        # Copy back to host and verify values
        result_arr1 = device_arrays["arr1"].copy_to_host()
        result_arr2 = device_arrays["arr2"].copy_to_host()

        np.testing.assert_array_equal(result_arr1, host_arr1)
        np.testing.assert_array_equal(result_arr2, host_arr2)

    @pytest.mark.nocudasim
    def test_from_device(self, registered_mgr, registered_instance):
        """Test from_device copies values from allocated device arrays correctly."""
        from numba import cuda

        mgr = registered_mgr
        instance = registered_instance

        # Allocate device arrays through the memory manager
        requests = {
            "arr1": ArrayRequest(
                shape=(2, 5), dtype=np.float32, memory="device"
            ),
            "arr2": ArrayRequest(
                shape=(3, 2), dtype=np.float64, memory="device"
            ),
        }

        stream = mgr.get_stream(instance)
        device_arrays = mgr.allocate_all(requests, id(instance), stream)

        # Create host arrays with test data and copy to device first
        host_source1 = np.arange(10, dtype=np.float32).reshape(2, 5) * 3.0
        host_source2 = np.arange(6, dtype=np.float64).reshape(3, 2) + 10.0

        # Copy test data to device arrays using cuda.to_device directly
        cuda.to_device(host_source1, stream=stream, to=device_arrays["arr1"])
        cuda.to_device(host_source2, stream=stream, to=device_arrays["arr2"])
        stream.synchronize()

        # Create empty host arrays to receive the data
        host_dest1 = np.zeros_like(host_source1)
        host_dest2 = np.zeros_like(host_source2)

        # Copy from device using from_device method
        from_arrays = [device_arrays["arr1"], device_arrays["arr2"]]
        to_arrays = [host_dest1, host_dest2]

        mgr.from_device(instance, from_arrays, to_arrays)

        # Synchronize stream to ensure copy is complete
        stream.synchronize()

        # Verify that the data was correctly copied from device to host
        np.testing.assert_array_equal(host_dest1, host_source1)
        np.testing.assert_array_equal(host_dest2, host_source2)


def test_get_total_request_size():
    """Test get_total_request_size calculates correct total size."""
    from cubie.memory.mem_manager import get_total_request_size

    requests = {
        "arr1": ArrayRequest(
            shape=(10, 10), dtype=np.float32, memory="device"
        ),  # 400 bytes
        "arr2": ArrayRequest(
            shape=(5, 5), dtype=np.float64, memory="device"
        ),  # 200 bytes
    }

    total_size = get_total_request_size(requests)
    expected_size = (10 * 10 * 4) + (5 * 5 * 8)  # 400 + 200 = 600
    assert total_size == expected_size


@pytest.mark.nocudasim
def test_ensure_cuda_context():
    """Test _ensure_cuda_context validates CUDA is available."""
    from cubie.memory.mem_manager import _ensure_cuda_context
    
    # This test should not crash when CUDA is available
    # It will raise RuntimeError if CUDA context cannot be initialized
    try:
        _ensure_cuda_context()
    except RuntimeError as e:
        pytest.fail(f"CUDA context validation failed: {e}")


def test_ensure_cuda_context_simulation():
    """Test _ensure_cuda_context is no-op in simulation mode."""
    from cubie.memory.mem_manager import _ensure_cuda_context
    from cubie.cuda_simsafe import CUDA_SIMULATION
    
    # In simulation mode, the function should do nothing and not raise
    if CUDA_SIMULATION:
        _ensure_cuda_context()  # Should not raise
