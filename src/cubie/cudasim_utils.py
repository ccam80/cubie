from contextlib import contextmanager
from ctypes import c_void_p

class FakeBaseCUDAMemoryManager:
    """Placeholder for CUDA simulator environments"""
    def __init__(self, context=None):
        self.context = context
    def initialize(self):
        pass
    def reset(self):
        pass
    def defer_cleanup(self):
        return contextmanager(lambda: (yield))()

class FakeNumbaCUDAMemoryManager(FakeBaseCUDAMemoryManager):
    """Placeholder for CUDA simulator environments"""
    handle: int = 0
    ptr: int = 0
    free: int = 0
    total: int = 0
    def __init__(self):
        pass
    def initialize(self):
        pass
    def reset(self):
        pass
    def defer_cleanup(self):
        return contextmanager(lambda: (yield))()

class FakeGetIpcHandleMixin:
    """Placeholder for CUDA simulator environments"""
    def get_ipc_handle(self):
        class FakeIpcHandle:
            def __init__(self):
                pass
        return FakeIpcHandle()

class FakeStream:
    """Placeholder for CUDA simulator environments"""
    handle = c_void_p(0)

class FakeHostOnlyCUDAManager(FakeBaseCUDAMemoryManager):
    """Placeholder for CUDA simulator environments"""
    def __init__(self, context=None):
        self.context = context
    def initialize(self):
        pass
    def reset(self):
        pass
    def defer_cleanup(self):
        return contextmanager(lambda: (yield))()

class FakeMemoryPointer:
    """Placeholder for CUDA simulator environments"""
    def __init__(self, context, device_pointer, size, finalizer=None):
        self.context = context
        self.device_pointer = device_pointer
        self.size = size
        self._cuda_memsize = size
        self.handle = self.device_pointer

def fake_get_memory_info():
    fakemem = FakeMemoryInfo()
    return fakemem.free, fakemem.total

class FakeMemoryInfo:
    free = 1024**3
    total = 8*1024**3

def fake_set_manager(manager):
    pass