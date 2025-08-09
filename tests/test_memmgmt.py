import pytest
from cubie.memmgmt import MemoryManager, ArrayRequest, StreamGroups
from cubie.memmgmt import ArrayResponse
from cupy.cuda import MemoryAsyncPool, MemoryPool
from numba import cuda
import numpy as np

@pytest.fixture(scope="function")
def mgr():
    return MemoryManager()

@pytest.fixture(scope="function")
def arrayy_request(request):
    defaults = {'shape': (10000),
                'dtype': np.float32,
                'memory': 'device'
    }
    if hasattr(request, 'param'):
        for key, value in request.param.items():
            defaults[key] = value
    return ArrayRequest(**defaults)

@pytest.fixture(scope="function")
def stream_groups():
    return StreamGroups()

class DummyClass:
    def __init__(self,
                 proportion=None,
                 stream_group=None,
                 native_stride_order=None):
        self.proportion = proportion
        self.stream_group = stream_group
        self.native_stride_order = native_stride_order

def test_assign_allocator(mgr):
    mgr.set_allocator("cupy_stream")
    assert isinstance(mgr._allocator, MemoryAsyncPool)
    mgr.set_allocator("default")
    assert mgr.allocator == "default" # replace with actual one
    with pytest.raises(ValueError):
        mgr.set_allocator("unknown")

def test_upper_limit_per_process(mgr, solver):
    proportion = 0.3
    mgr.register(solver, 0.3)
    testcap = mgr.cap(solver)
    free, total = cuda.current_context().get_memory_info()
    solvercap = total * proportion
    assert testcap == solvercap

def test_chunking(mgr, solver):
    request_size = 7*1024**3
    proportion = 0.5
    mgr.register(solver, 0.5)
    chunks = mgr.get_chunks(solver, request_size)

    free, total = cuda.current_context().get_memory_info() # ~6GB, 8GB
    cap = total * proportion
    available_mem = min(total, free)
    expected_chunks = np.ceil(request_size / available_mem)

    assert chunks == expected_chunks



def test_load_balance(mgr):
    instance1 = DummyClass()
    instance2 = DummyClass()
    instance3 = DummyClass()
    instance4 = DummyClass()
    instance5 = DummyClass()

    mgr.register(instance1)
    mgr.set_limit_mode("active")

    assert mgr.proportion(instance1) == 1.0

    mgr.register(instance2)
    assert mgr.proportion(instance2) == 0.5

    mgr.register(instance3, 0.5)
    assert mgr.proportion(instance3) == 0.5
    assert mgr.proportion(instance1) == 0.25
    assert mgr.proportion(instance2) == 0.25

    mgr.register(instance4,0.3)
    assert mgr.proportion(instance4) == 0.3
    assert mgr.proportion(instance1) == 0.1
    assert mgr.proportion(instance2) == 0.1
    assert mgr.proportion(instance3) == 0.5

    with pytest.raises(ValueError):
        mgr.register(instance5, 0.3)

    mgr.set_limit_mode("passive")
    assert mgr.proportion(instance1) == 1.0
    assert mgr.proportion(instance2) == 1.0
    assert mgr.proportion(instance3) == 1.0
    assert mgr.proportion(instance4) == 1.0
    assert mgr.proportion(instance5) == 1.0

def test_stride_ordering(mgr):
    instance1 = DummyClass()
    instance2 = DummyClass()

    # Register instances in different stream groups
    mgr.register(instance1, stream_group="group1")
    mgr.register(instance2, stream_group="group2")

    stream1 = mgr.stream_groups.streams["group1"]
    stream2 = mgr.stream_groups.streams["group2"]

    assert stream1 != stream2
    assert instance1 in mgr.stream_groups.groups["group1"]
    assert instance2 in mgr.stream_groups.groups["group2"]

def test_streamgroups_add_and_change():
    sg = StreamGroups()
    inst1 = DummyClass()
    sg.add_instance(inst1, "group1")
    assert "group1" in sg.groups
    assert inst1 in sg.groups["group1"]
    from numba.cuda.cudadrv.driver import Stream
    assert isinstance(sg.streams["group1"], Stream)
    # change group
    sg.change_group(inst1, "group2")
    assert inst1 in sg.groups["group2"]
    assert inst1 not in sg.groups["group1"]

def test_manual_auto_allocation_switch(mgr):
    inst1 = DummyClass()
    inst2 = DummyClass()
    mgr.register(inst1)
    mgr.register(inst2)
    # initial auto allocation
    assert pytest.approx(mgr.proportion(inst1), rel=1e-6) == 0.5
    assert pytest.approx(mgr.proportion(inst2), rel=1e-6) == 0.5

    # set manual allocation
    mgr.set_manual_allocation(inst1, 0.7)
    assert mgr.registered_instances[inst1]._manual_proportion is True
    assert pytest.approx(mgr.proportion(inst1), rel=1e-6) == 0.7
    # auto pool reproportioned for inst2
    assert mgr.registered_instances[inst2]._manual_proportion is False
    assert pytest.approx(mgr.proportion(inst2), rel=1e-6) == 0.3

    # switch back to auto
    mgr.set_auto_allocation(inst1)
    assert mgr.registered_instances[inst1]._manual_proportion is False
    # both should have equal proportions again
    assert pytest.approx(mgr.proportion(inst1), rel=1e-6) == pytest.approx(mgr.proportion(inst2), rel=1e-6)

def test_manual_allocation_errors(mgr):
    inst = DummyClass()
    mgr.register(inst)
    # set manual allocation first time
    mgr.set_manual_allocation(inst, 0.3)
    # setting manual again should raise
    with pytest.raises(ValueError):
        mgr.set_manual_allocation(inst, 0.2)

def test_auto_allocation_errors(mgr):
    inst = DummyClass()
    mgr.register(inst, proportion=0.3)
    # already manual, setting auto allocation should work first
    mgr.set_auto_allocation(inst)
    # setting auto again should raise
    with pytest.raises(ValueError):
        mgr.set_auto_allocation(inst)

def test_combined_management(mgr):
    from numba.cuda.cudadrv.driver import Stream
    inst1 = DummyClass()
    inst2 = DummyClass()
    inst3 = DummyClass()

    # register two instances auto
    mgr.register(inst1)
    mgr.register(inst2)
    assert inst1 in mgr._auto_pool and inst2 in mgr._auto_pool
    # proportions equal initially
    p1 = mgr.proportion(inst1)
    assert pytest.approx(p1, rel=1e-6) == pytest.approx(mgr.proportion(inst2), rel=1e-6)

    # register manual instance
    mgr.register(inst3, proportion=0.2)
    assert inst3 in mgr._manual_pool
    auto_remain = 1.0 - 0.2
    expected_auto = auto_remain / 2
    assert pytest.approx(mgr.proportion(inst1), rel=1e-6) == expected_auto
    assert pytest.approx(mgr.proportion(inst2), rel=1e-6) == expected_auto

    # change stream group of inst1
    mgr.stream_groups.change_group(inst1, "other")
    assert inst1 in mgr.stream_groups.groups["other"]
    assert inst1 not in mgr.stream_groups.groups["default"]
    # new group has a stream
    assert isinstance(mgr.stream_groups.streams["other"], Stream)

    # check pool sizes
    assert len(mgr._manual_pool) == 1
    assert len(mgr._auto_pool) == 2
