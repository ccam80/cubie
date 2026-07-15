"""CuPy async External Memory Manager plugin for Numba CUDA contexts.

Backs Numba's device allocations with CuPy's asynchronous (stream-ordered)
memory pool via the EMM plugin interface, so ``cuda.device_array`` returns a
**native** ``DeviceNDArray`` drawn from a pooled allocator. Native arrays keep
the fast kernel-launch path (no per-launch ``__cuda_array_interface__``
re-parse) and let transfers use Numba's pinned + streamed async copies.

See Also
--------
:class:`~cubie.memory.mem_manager.MemoryManager`
    Coordinates allocation through this plugin.
"""

import ctypes
import logging
from typing import Any, Callable, Optional

from cubie.cuda_simsafe import cuda, cupy, CUDA_SIMULATION

logger = logging.getLogger(__name__)


if not CUDA_SIMULATION:

    class CuPyAsyncNumbaManager(
        cuda.GetIpcHandleMixin, cuda.HostOnlyCUDAMemoryManager
    ):
        """EMM plugin allocating native Numba arrays from CuPy's async pool.

        Adapted from the numba cupy-EMM tutorial, using
        ``cupy.cuda.MemoryAsyncPool`` so allocations are stream-ordered
        (cudaMallocAsync) against whichever stream is current at allocation.
        """

        def __init__(self, context) -> None:
            super().__init__(context=context)
            # Kept alive so CuPy returns the block to the pool on finalize.
            self._allocations: dict[int, Any] = {}
            self._mp = None
            self.is_cupy = True

        def initialize(self) -> None:
            super().initialize()
            # Context.prepare_for_use calls initialize() on every context
            # activation; the pool must persist across calls because live
            # allocations hold blocks from it.
            if self._mp is None:
                self._mp = cupy.cuda.MemoryAsyncPool()

        def memalloc(self, nbytes: int) -> "cuda.MemoryPointer":
            cp_mp = self._mp.malloc(nbytes)
            self._allocations[cp_mp.ptr] = cp_mp
            return cuda.MemoryPointer(
                cuda.current_context(),
                ctypes.c_void_p(int(cp_mp.ptr)),
                nbytes,
                finalizer=self._make_finalizer(cp_mp.ptr),
            )

        def _make_finalizer(self, ptr: int) -> Callable[[], None]:
            allocations = self._allocations

            def finalizer() -> None:
                # Dropping the last reference returns the block to the pool.
                allocations.pop(ptr, None)

            return finalizer

        def get_memory_info(self) -> "cuda.MemoryInfo":
            # Real device free/total (cuMemGetInfo), not pool bytes: the
            # manager uses this to size chunks against the device, and the
            # pool's free_bytes() is 0 until it has cached blocks.
            free, total = cupy.cuda.runtime.memGetInfo()
            return cuda.MemoryInfo(free=free, total=total)

        def reset(self, stream: Optional[Any] = None) -> None:
            super().reset()
            if self._mp:
                self._mp.free_all_blocks(stream=stream)

        def free_pool_blocks(self) -> None:
            """Return the pool's unused cached blocks to the driver.

            Trims the async pool so freed allocations become visible
            to ``cuMemGetInfo``; live allocations are unaffected.
            """
            if self._mp is not None:
                self._mp.free_all_blocks()

        @property
        def interface_version(self) -> int:
            return 1

    def install_async_emm() -> None:
        """Install the CuPy async pool as Numba's device memory manager.

        Must run before the CUDA context is created; the manager takes effect
        on first context creation.
        """
        cuda.set_memory_manager(CuPyAsyncNumbaManager)

else:  # pragma: no cover - simulated: no device, no EMM
    CuPyAsyncNumbaManager = None

    def install_async_emm() -> None:
        return None
