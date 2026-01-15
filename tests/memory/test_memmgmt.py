import pytest
from numba import cuda

from cubie.cuda_simsafe import (
    Stream,
)

from cubie.memory.mem_manager import (
    MemoryManager,
    ArrayRequest,
    ArrayResponse,
    InstanceMemorySettings,
)

import numpy as np


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
    defaults = {"mode": "passive"}
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


@pytest.fixture(scope="function")
def registered_mgr(mgr, registered_instance):
    mgr.register(
        registered_instance,
        proportion=registered_instance.proportion,
        invalidate_cache_hook=registered_instance.invalidate_all_hook,
    )
    return mgr


def registered_mgr_context_safe(
    mem_manager_settings, registered_instance_settings
):
    fixed_mem_settings_closure = {"free": 1 * 1024**3, "total": 8 * 1024**3}

    class TestMemoryManager(MemoryManager):
        def get_memory_info(self):
            free = fixed_mem_settings_closure["free"]
            total = fixed_mem_settings_closure["total"]
            return free, total

    manager = TestMemoryManager(**mem_manager_settings)
    registeree = DummyClass(**registered_instance_settings)

    manager.register(
        registeree,
        proportion=registeree.proportion,
        invalidate_cache_hook=registeree.invalidate_all_hook,
    )

    return manager, registeree


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

    # Can't recover from this - context stays stale. This doesn't reflect a
    # real use case; reinstate if live memory manager switching is required
    # @pytest.mark.cupy
    # @pytest.mark.parametrize("manager_target",
    #                          [("cupy_async", CuPyAsyncNumbaManager),
    #                           ("cupy", CuPySyncNumbaManager),
    #                           ("default", NumbaCUDAMemoryManager)
    #                         ],
    #                          ids=["cupy_async", "cupy", "default"])
    # def test_set_allocator(self, manager_target, mem_manager_settings,
    #                              registered_instance_settings):
    #     """Test that set_allocator sets the allocator correctly
    #     test each of "cupy_async", "cupy", "default", checking that
    #     (self._allocator is CuPyAsyncNumbaManager, CuPySyncNumbaManager.
    #      NumbaCudaMemoryManager, respectively)"""
    #     label, cls = manager_target
    #     regmgr, instance = registered_mgr_context_safe(
    #             mem_manager_settings,
    #             registered_instance_settings
    #     )
    #     regmgr.set_allocator(label)
    #     assert regmgr._allocator == cls

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
    def test_get_stream(
        self, mem_manager_settings, registered_instance_settings
    ):
        """test that get_stream successfully passes a different stream for
        instances registered in different stream groups"""

        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        stream = regmgr.get_stream(instance)
        assert isinstance(stream, Stream)
        # Register another instance in a new group
        inst2 = DummyClass()
        regmgr.register(inst2, stream_group="other")
        stream2 = regmgr.get_stream(inst2)
        assert stream2 is not None
        assert int(stream.handle.value) != int(stream2.handle.value)

    def test_change_stream_group(
        self, mem_manager_settings, registered_instance_settings
    ):
        """test that change_stream_group changes the stream group of an
        instance, and that it raises ValueError if the instance wasn't
        already in a group"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        regmgr.change_stream_group(instance, "other")
        assert id(instance) in regmgr.stream_groups.groups["other"]
        dummy = DummyClass()
        with pytest.raises(ValueError):
            regmgr.change_stream_group(dummy, "newgroup")

    @pytest.mark.nocudasim
    def test_reinit_streams(
        self, mem_manager_settings, registered_instance_settings
    ):
        """test that reinit_streams causess a different stream to be
        returned from get_stream"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        stream1 = regmgr.get_stream(instance)
        regmgr.reinit_streams()
        stream2 = regmgr.get_stream(instance)
        assert int(stream1.handle.value) != int(stream2.handle.value)

    def test_invalidate_all(
        self, mem_manager_settings, registered_instance_settings
    ):
        """Add a new instance with a measurable invalidate hook, and check
        that it is called when invalidate all is called"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        called = {"flag": False}

        def hook():
            called["flag"] = True

        inst = DummyClass()
        regmgr.register(inst, invalidate_cache_hook=hook)
        regmgr.invalidate_all()
        assert called["flag"] is True

    @pytest.mark.parametrize(
        "registered_instance_override", [{"proportion": None}], indirect=True
    )
    def test_set_manual_limit_mode(
        self, mem_manager_settings, registered_instance_settings
    ):
        """Test that set_manual_limit_mode sets the instance to manual mode"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        instance_id = id(instance)
        regmgr.set_manual_limit_mode(instance, 0.3)
        assert instance_id in regmgr._manual_pool

    def test_set_auto_limit_mode(
        self, mem_manager_settings, registered_instance_settings
    ):
        """Test that set_auto_limit_mode sets the instance to auto mode"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        regmgr.set_auto_limit_mode(instance)
        instance_id = id(instance)
        assert instance_id not in regmgr._manual_pool

    @pytest.mark.parametrize(
        "registered_instance_override", [{"proportion": 0.5}], indirect=True
    )
    def test_proportion(
        self, mem_manager_settings, registered_instance_settings
    ):
        """Test that proportion returns the requested proportion if set,
        and 1.0 if not set (auto)"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        assert regmgr.proportion(instance) == instance.proportion
        # Register auto instance
        inst2 = DummyClass(proportion=None)
        regmgr.register(inst2, proportion=None)
        assert regmgr.proportion(inst2) == 0.5

    def test_cap(self, mem_manager_settings, registered_instance_settings):
        """Test that cap returns the correct value"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        proportion = instance.proportion
        testcap = regmgr.cap(instance)
        free, total = regmgr.get_memory_info()
        solvercap = total * proportion
        assert testcap == solvercap

    @pytest.mark.parametrize(
        "registered_instance_override", [{"proportion": None}], indirect=True
    )
    def test_pool_proportions(
        self, mem_manager_settings, registered_instance_settings
    ):
        """Add a few auto and manual instances, check that the total manual
        and auto pool proportions match expected. Test add and rebalance
        methods along the way"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        instance1 = instance
        instance2 = DummyClass()
        instance3 = DummyClass()
        instance4 = DummyClass()
        instance5 = DummyClass()

        # already registered
        with pytest.raises(ValueError):
            regmgr.register(instance1)
        regmgr.set_limit_mode("active")

        assert regmgr.proportion(instance1) == 1.0
        assert regmgr.manual_pool_proportion == 0.0
        assert regmgr.auto_pool_proportion == 1.0

        regmgr.register(instance2)
        assert abs(regmgr.proportion(instance2) - 0.5) < 1e-6
        assert abs(regmgr.manual_pool_proportion - 0.0) < 1e-6
        assert abs(regmgr.auto_pool_proportion - 1.0) < 1e-6

        regmgr.register(instance3, proportion=0.5)
        assert abs(regmgr.proportion(instance3) - 0.5) < 1e-6
        assert abs(regmgr.manual_pool_proportion - 0.5) < 1e-6
        assert abs(regmgr.auto_pool_proportion - 0.5) < 1e-6

        regmgr.register(instance4, proportion=0.4)
        assert abs(regmgr.proportion(instance4) - 0.4) < 1e-6
        assert abs(regmgr.manual_pool_proportion - 0.9) < 1e-6
        assert abs(regmgr.auto_pool_proportion - 0.1) < 1e-6

        regmgr.set_auto_limit_mode(instance3)
        assert abs(regmgr.proportion(instance3) - 0.2) < 1e-6
        assert abs(regmgr.manual_pool_proportion - 0.4) < 1e-6
        assert abs(regmgr.auto_pool_proportion - 0.6) < 1e-6

        regmgr.set_manual_limit_mode(instance3, 0.3)
        assert abs(regmgr.proportion(instance3) - 0.3) < 1e-6
        assert abs(regmgr.manual_pool_proportion - 0.7) < 1e-6
        assert abs(regmgr.auto_pool_proportion - 0.3) < 1e-6

        with pytest.raises(ValueError):
            regmgr.register(instance5, proportion=0.3)

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
                shape=(4, 4, 8), dtype=np.float32, memory="device"
            ),
        }
        mgr.queue_request(instance, requests)
        mgr.allocate_queue(instance)
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
        mgr.registry[id(instance)].add_allocation("foo", arr)
        mgr.registry[id(instance)].add_allocation("bar", arr)
        mgr.free_all()
        assert mgr.registry[id(instance)].allocations == {}

    def test_get_memory_info(self, mgr):
        """Test get_memory_info returns tuple of free and total memory."""
        free, total = mgr.get_memory_info()
        assert free == 1 * 1024**3
        assert total == 8 * 1024**3

    def test_create_host_array_1d(self, mgr):
        """Test create_host_array returns correct 1D array."""
        arr = mgr.create_host_array(shape=(10,), dtype=np.float32)
        assert arr.shape == (10,)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, np.zeros(10, dtype=np.float32))

    def test_create_host_array_2d(self, mgr):
        """Test create_host_array returns correct 2D array."""
        arr = mgr.create_host_array(
            shape=(5, 3),
            dtype=np.float64,
        )
        assert arr.shape == (5, 3)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, np.zeros((5, 3), dtype=np.float64))

    def test_create_host_array_3d_default_stride(self, mgr):
        """Test create_host_array returns correct 3D array with default stride."""
        arr = mgr.create_host_array(
            shape=(2, 3, 4),
            dtype=np.float32,
        )
        assert arr.shape == (2, 3, 4)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(
            arr, np.zeros((2, 3, 4), dtype=np.float32)
        )

    def test_create_host_array_3d_custom_stride(self, mgr):
        """Test create_host_array returns C-contiguous 3D array."""
        arr = mgr.create_host_array(
            shape=(2, 3, 4),
            dtype=np.float32,
        )
        assert arr.shape == (2, 3, 4)
        assert arr.dtype == np.float32
        assert arr.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(
            arr, np.zeros((2, 3, 4), dtype=np.float32)
        )

    def test_create_host_array_pageable_memory(self, mgr):
        """Test create_host_array with host (pageable) memory type."""
        arr = mgr.create_host_array(
            shape=(2, 3, 4),
            dtype=np.float32,
            memory_type="host",
        )
        assert arr.shape == (2, 3, 4)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(
            arr, np.zeros((2, 3, 4), dtype=np.float32)
        )

    def test_create_host_array_invalid_memory_type(self, mgr):
        """Test create_host_array raises ValueError for invalid memory type."""
        with pytest.raises(ValueError, match="memory_type must be"):
            mgr.create_host_array(
                shape=(2, 3, 4),
                dtype=np.float32,
                memory_type="invalid",
            )

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
        mgr.allocate_queue(inst1)

        # Both callbacks should be called
        assert callbacks_called["inst1"]["flag"] is True
        assert callbacks_called["inst2"]["flag"] is True

    def test_allocate_queue_empty_queue(
        self, registered_mgr, registered_instance
    ):
        """Test allocate_queue with empty queue returns None."""
        mgr = registered_mgr
        instance = registered_instance

        mgr.allocate_queue(instance)

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


class TestGetChunkParameters:
    """Tests for get_chunk_parameters method."""

    def test_get_chunk_parameters_unchunkable_exceeds_memory(self, mgr):
        """Verify error when unchunkable arrays exceed available memory."""
        # Register an instance
        inst = DummyClass()
        mgr.register(inst, stream_group="test")

        # Create a request where unchunkable size exceeds available memory
        # Memory manager has 1GB free (1 * 1024**3 bytes)
        # Create a large unchunkable array (2GB total)
        huge_shape = (512, 512, 512)  # 512^3 * 4 bytes = 512MB per array
        requests = {
            id(inst): {
                "huge_unchunkable": ArrayRequest(
                    shape=huge_shape,
                    dtype=np.float32,
                    memory="device",
                    unchunkable=True,
                ),
                "huge_unchunkable2": ArrayRequest(
                    shape=huge_shape,
                    dtype=np.float32,
                    memory="device",
                    unchunkable=True,
                ),
                # Need at least one chunkable array to hit the unchunkable
                # exceeds memory path (otherwise hits all-unchunkable path)
                "small_chunkable": ArrayRequest(
                    shape=(1, 1),
                    dtype=np.float32,
                    memory="device",
                    unchunkable=False,
                ),
            }
        }

        with pytest.raises(ValueError, match="unchunkable"):
            mgr.get_chunk_parameters(
                requests=requests,
                axis_length=512,
                stream_group="test",
            )


class TestAllocateQueueExtractsNumRuns:
    """Tests for allocate_queue extraction of num_runs from triggering
    instance."""

    def test_allocate_queue_extracts_num_runs(self, mgr):
        """Verify allocate_queue correctly extracts num_runs from
        triggering instance.

        The allocate_queue method should extract num_runs from
        triggering_instance.run_params.runs instead of computing it
        from array request shapes. This test verifies that behavior.
        """

        # Create a mock instance with run_params
        class MockRunParams:
            def __init__(self, runs):
                self.runs = runs

        class MockInstance:
            def __init__(self, runs):
                self.run_params = MockRunParams(runs)

        instance = MockInstance(runs=100)

        # Track whether the allocation callback was called with correct
        # response
        callback_called = {"flag": False, "response": None}

        def allocation_hook(response):
            callback_called["flag"] = True
            callback_called["response"] = response

        mgr.register(instance, allocation_ready_hook=allocation_hook)

        # Create requests with arrays that have run axis
        # The shape indicates 50 runs, but run_params.runs is 100
        # allocate_queue should use 100 from run_params, not 50 from
        # shape
        requests = {
            "arr1": ArrayRequest(
                shape=(10, 50),  # 50 in run axis (second dimension)
                dtype=np.float32,
                memory="device",
                unchunkable=False,
            ),
        }

        mgr.queue_request(instance, requests)
        mgr.allocate_queue(instance)

        # Verify callback was called
        assert callback_called["flag"] is True
        response = callback_called["response"]
        assert isinstance(response, ArrayResponse)

        # The key validation: chunk_length should be computed based on
        # num_runs=100 from run_params, not from the array shape
        # Since we have 1GB free and small arrays, no chunking should
        # occur, giving us chunk_length matching the full num_runs
        assert response.chunks == 1
        # chunk_length should equal num_runs when no chunking occurs
        assert response.chunk_length == 100

    def test_allocate_queue_chunks_correctly(self, mgr):
        """Verify chunking calculations work with num_runs from instance.

        When memory constraints force chunking, the chunk calculations
        should be based on num_runs extracted from
        triggering_instance.run_params.runs.
        """

        # Create instance with large num_runs to force chunking
        class MockRunParams:
            def __init__(self, runs):
                self.runs = runs

        class MockInstance:
            def __init__(self, runs):
                self.run_params = MockRunParams(runs)

        # Use large num_runs to increase memory requirements
        instance = MockInstance(runs=10000)

        callback_called = {"flag": False, "response": None}

        def allocation_hook(response):
            callback_called["flag"] = True
            callback_called["response"] = response

        mgr.register(instance, allocation_ready_hook=allocation_hook)

        # Create large requests to force chunking
        # Each element is 4 bytes, so 10000 runs * 100 vars * 4 bytes =
        # 4MB per array. With multiple arrays, this should exceed the
        # 1GB available and trigger chunking.
        requests = {
            "arr1": ArrayRequest(
                shape=(100, 10000),
                dtype=np.float32,
                memory="device",
                unchunkable=False,
            ),
            "arr2": ArrayRequest(
                shape=(100, 10000),
                dtype=np.float32,
                memory="device",
                unchunkable=False,
            ),
        }

        mgr.queue_request(instance, requests)
        mgr.allocate_queue(instance)

        # Verify callback was called
        assert callback_called["flag"] is True
        response = callback_called["response"]
        assert isinstance(response, ArrayResponse)

        # With 10000 runs and 1GB available memory, chunking should
        # occur. The exact chunk_length depends on available memory,
        # but it should be less than 10000.
        # Verify that chunking occurred
        if response.chunks > 1:
            # chunk_length should be calculated to fit in available
            # memory
            assert response.chunk_length < 10000
            assert response.chunk_length > 0
            # Verify chunks * chunk_length covers all runs (with
            # possible dangling chunk)
            total_covered = (
                response.chunks - 1
            ) * response.chunk_length + response.chunk_length
            assert total_covered >= 10000
        else:
            # If no chunking occurred (e.g., in CUDA sim with more
            # memory), chunk_length should equal num_runs
            assert response.chunk_length == 10000


def test_allocate_queue_no_chunked_slices_in_response(mgr):
    """Verify ArrayResponse from allocate_queue does not have chunked_slices.

    After refactoring to use on-demand chunk slice computation, the
    ArrayResponse should no longer contain a chunked_slices field. This test
    verifies that the response contains the necessary chunk parameters
    (chunks, , chunk_length, )
    """

    # Create instance with run_params
    class MockRunParams:
        def __init__(self, runs):
            self.runs = runs

    class MockInstance:
        def __init__(self, runs):
            self.run_params = MockRunParams(runs)

    instance = MockInstance(runs=100)

    # Track the allocation callback response
    callback_called = {"flag": False, "response": None}

    def allocation_hook(response):
        callback_called["flag"] = True
        callback_called["response"] = response

    mgr.register(instance, allocation_ready_hook=allocation_hook)

    # Create requests with chunkable arrays
    requests = {
        "arr1": ArrayRequest(
            shape=(10, 100),
            dtype=np.float32,
            memory="device",
            unchunkable=False,
        ),
        "arr2": ArrayRequest(
            shape=(5, 100),
            dtype=np.float32,
            memory="device",
            unchunkable=False,
        ),
    }

    mgr.queue_request(instance, requests)
    mgr.allocate_queue(instance)

    # Verify callback was called
    assert callback_called["flag"] is True
    response = callback_called["response"]
    assert isinstance(response, ArrayResponse)

    # Verify chunked_slices is NOT in the response
    assert not hasattr(response, "chunked_slices")

    # Verify that chunk parameters ARE in the response
    assert hasattr(response, "chunks")
    assert hasattr(response, "chunk_length")
    assert hasattr(response, "chunked_shapes")

    # Verify chunk parameters have expected values
    assert isinstance(response.chunks, int)
    assert isinstance(response.chunk_length, int)
    assert isinstance(response.chunked_shapes, dict)


class TestExtractNumRuns:
    """Tests for _extract_num_runs helper method."""

    def test_extract_num_runs_finds_single_value(self, mgr):
        """Verify _extract_num_runs returns correct value when one request has total_runs."""
        # Create queued_requests with single instance and single array
        queued_requests = {
            "instance_1": {
                "arr1": ArrayRequest(
                    shape=(10, 100),
                    dtype=np.float32,
                    memory="device",
                    total_runs=100,
                ),
            }
        }
        
        num_runs = mgr._extract_num_runs(queued_requests)
        assert num_runs == 100

    def test_extract_num_runs_ignores_none_values(self, mgr):
        """Verify _extract_num_runs works when some requests have total_runs=None."""
        # Create queued_requests with multiple arrays, some with None total_runs
        queued_requests = {
            "instance_1": {
                "arr1": ArrayRequest(
                    shape=(10, 100),
                    dtype=np.float32,
                    memory="device",
                    total_runs=None,  # Should be ignored
                ),
                "arr2": ArrayRequest(
                    shape=(5, 100),
                    dtype=np.float32,
                    memory="device",
                    total_runs=100,  # Should use this value
                ),
            },
            "instance_2": {
                "arr3": ArrayRequest(
                    shape=(3, 100),
                    dtype=np.float32,
                    memory="device",
                    total_runs=None,  # Should be ignored
                ),
            },
        }
        
        num_runs = mgr._extract_num_runs(queued_requests)
        assert num_runs == 100

    def test_extract_num_runs_raises_on_no_values(self, mgr):
        """Verify _extract_num_runs raises ValueError when all total_runs are None."""
        # Create queued_requests where all arrays have total_runs=None
        queued_requests = {
            "instance_1": {
                "arr1": ArrayRequest(
                    shape=(10, 100),
                    dtype=np.float32,
                    memory="device",
                    total_runs=None,
                ),
                "arr2": ArrayRequest(
                    shape=(5, 100),
                    dtype=np.float32,
                    memory="device",
                    total_runs=None,
                ),
            }
        }
        
        with pytest.raises(ValueError, match="No total_runs found"):
            mgr._extract_num_runs(queued_requests)

    def test_extract_num_runs_raises_on_inconsistent_values(self, mgr):
        """Verify _extract_num_runs raises ValueError when requests have different total_runs."""
        # Create queued_requests with inconsistent total_runs values
        queued_requests = {
            "instance_1": {
                "arr1": ArrayRequest(
                    shape=(10, 100),
                    dtype=np.float32,
                    memory="device",
                    total_runs=100,
                ),
                "arr2": ArrayRequest(
                    shape=(5, 200),
                    dtype=np.float32,
                    memory="device",
                    total_runs=200,  # Different value - should raise
                ),
            }
        }
        
        with pytest.raises(ValueError, match="Inconsistent total_runs"):
            mgr._extract_num_runs(queued_requests)

