from numba import cuda
import pytest
import attrs
import attrs.validators as val
import cupy
from numba.cuda.cudadrv.driver import Stream
import numpy as np

@pytest.fixture(scope="function")
def mgr():
    return MemoryManager()

def test_assign_allocator(mgr):
    mgr.select_allocator("cupy_stream")
    assert isinstance(mgr._allocator, cupy.cuda.MemoryAsyncPool)
    mgr.select_allocator("default")
    assert mgr.allocator == "default"
    with pytest.raises(ValueError):
        mgr.select_allocator("unknown")

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

class DummyClass:
    def __init__(self, request_size=None):
        self.request_size = request_size

def test_load_balance(mgr):
    instance1 = DummyClass()
    instance2 = DummyClass()
    instance3 = DummyClass()
    instance4 = DummyClass()
    instance5 = DummyClass()

    mgr.register(instance1)
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

def test_stream_allocation(mgr):
    instance1 = DummyClass(request_size=2*1024**3)
    instance2 = DummyClass(request_size=3*1024**3)

    stream1 = mgr.register(instance1)
    stream2 = mgr.register(instance2)

    assert stream1 != stream2
    assert isinstance(stream1, Stream
                      )

@attrs.define
class InstanceMemorySettings:
    proportion: float = attrs.field(default=1.0,
                                  validator=val.instance_of(float))
    max_size: int = attrs.field(default=None,
                               validator=val.optional(
                                       val.instance_of(int)))
    current_allocation: int = attrs.field(default=0,
                                         validator=val.instance_of(int))
    _manual_proportion: bool = attrs.field(default=False,
                                           validator=val.instance_of(bool))

@attrs.define
class MemoryManager:
    proportions: dict[object, float] = attrs.field(default=None,
                                                  validator=val.optional(
                                                          val.instance_of(dict)))
    totalmem: int = attrs.field(default=None,
                               validator=val.optional(
                                       val.instance_of(int)))

    registered_instances: dict[object, InstanceMemorySettings] = attrs.field(
            default=None, validator=val.optional(val.instance_of(dict))
    )
    def __attrs_post_init__(self):
        _, self.totalmem = cuda.current_context().get_memory_info()

    def register(self, cls_, proportion=None):
        pass

    def cap(self, cls_):
        pass

    def get_chunks(self, cls_, request_size):
        pass

    def proportion(self, cls_):
        pass

