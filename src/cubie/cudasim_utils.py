"""Fake CUDA interfaces for simulator environments.

This module provides lightweight stand-ins for CUDA memory managers and
streams so that code depending on Numba's CUDA API can run on systems
without a CUDA driver.
"""
# no cover: start
from contextlib import contextmanager
from ctypes import c_void_p
import os

import numba
import numpy as np


class FakeBaseCUDAMemoryManager:
    """Minimal stub of a CUDA memory manager."""

    def __init__(self, context=None):
        self.context = context

    def initialize(self):
        """Placeholder initialize  method."""
        pass

    def reset(self):
        """Placeholder reset method."""
        pass

    def defer_cleanup(self):
        """Return a no-op context manager."""
        return contextmanager(lambda: (yield))()


class FakeNumbaCUDAMemoryManager(FakeBaseCUDAMemoryManager):
    """Minimal fake of a CUDA memory manager."""

    handle: int = 0
    ptr: int = 0
    free: int = 0
    total: int = 0

    def __init__(self):
        pass


class FakeGetIpcHandleMixin:
    """Return a fake IPC handle object."""

    def get_ipc_handle(self):
        class FakeIpcHandle:
            """Trivial stand-in for an IPC handle."""

            def __init__(self):
                pass

        return FakeIpcHandle()


class FakeStream:
    """Placeholder CUDA stream."""

    handle = c_void_p(0)


class FakeHostOnlyCUDAManager(FakeBaseCUDAMemoryManager):
    """Host-only manager used in simulation environments."""


class FakeMemoryPointer:
    """Lightweight pointer-like object used in simulation."""

    def __init__(self, context, device_pointer, size, finalizer=None):
        self.context = context
        self.device_pointer = device_pointer
        self.size = size
        self._cuda_memsize = size
        self.handle = self.device_pointer


def fake_get_memory_info():
    """Return fake free and total memory values."""
    fakemem = FakeMemoryInfo()
    return fakemem.free, fakemem.total


class FakeMemoryInfo:
    """Container for fake memory statistics."""

    free = 1024**3
    total = 8 * 1024**3


def fake_set_manager(manager):
    """Stub for setting a memory manager."""
    pass


def from_dtype(dtype: np.dtype):
    if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1":
        return numba.from_dtype(dtype)
    else:
        return dtype

# no cover: end