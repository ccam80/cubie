"""
GPU memory management subsystem for cubie.

This module provides GPU memory management capabilities including:
- CuPy memory pool integration for efficient GPU memory allocation
- Stream group management for asynchronous CUDA operations
- Array request/response system for structured memory allocation
- Manual or automatic allocation of VRAM to different processes
- Automatic chunking for large allocations that exceed available memory

The main components are:

- :class:`MemoryManager`: Singleton interface for managing all memory operations
- :class:`ArrayRequest`: Specification for array allocation requests
- :class:`ArrayResponse`: Results of array allocation operations
- :class:`StreamGroups`: Management of CUDA stream groups for coordination
- :func:`current_cupy_stream`: Context manager for CuPy stream integration
- :class:`CuPyAsyncNumbaManager`: Async CuPy memory pool integration
- :class:`CuPySyncNumbaManager`: Sync CuPy memory pool integration

The default memory manager instance is available as `default_memmgr`.
"""

from cubie.memory.cupy_emm import (
    current_cupy_stream,
    CuPySyncNumbaManager,
    CuPyAsyncNumbaManager,
)
from cubie.memory.mem_manager import MemoryManager

default_memmgr = MemoryManager()

__all__ = [
    "current_cupy_stream",
    "CuPySyncNumbaManager",
    "CuPyAsyncNumbaManager",
    "default_memmgr",
    "MemoryManager"
]
