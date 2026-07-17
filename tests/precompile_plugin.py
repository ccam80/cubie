"""Headless precompile pass over the test suite.

Loaded with ``pytest -p tests.precompile_plugin`` on a machine with no
GPU. Pins the compilation target to the compute capability named by
``CUBIE_TARGET_CC`` (e.g. ``"8.9"``), stubs the handful of places
cubie touches the device outside kernel launch, and turns every kernel
launch into compile-and-cache with no execution: outputs stay
zero-filled and tests run on to their remaining launches. Test
outcomes are meaningless in this mode; the product of the run is the
populated ``CUBIE_KERNEL_CACHE_DIR``, which a GPU runner then
consumes. Only the compiled-kernel cache is relocated there — the
codegen source caches stay in the session's per-worker temporaries,
which keeps xdist workers from racing on shared generated files.

Kernels whose dispatcher has no configured disk cache (one-off test
kernels without an :func:`tests._utils.attach_kernel_cache` key) are
not compiled: caching them by source location would collide across
closures, so their launches are simply no-ops here and they compile on
the GPU runner as before.
"""
import os
from types import SimpleNamespace

import numpy as np

_raw_cc = os.environ.get("CUBIE_TARGET_CC", "")
if not _raw_cc.strip():
    raise RuntimeError(
        "tests.precompile_plugin requires CUBIE_TARGET_CC (e.g. '8.9') "
        "naming the GPU runner's compute capability."
    )
TARGET_CC = tuple(int(p) for p in _raw_cc.replace(",", ".").split("."))

if not os.environ.get("CUBIE_KERNEL_CACHE_DIR", "").strip():
    raise RuntimeError(
        "tests.precompile_plugin requires CUBIE_KERNEL_CACHE_DIR: "
        "without it the compiled-kernel cache lands in the session's "
        "temporary directories and is deleted at teardown."
    )

# The artifact must keep every precompiled kernel.
os.environ.setdefault("CUBIE_MAX_CACHE_ENTRIES", "0")

# ---- pin the compilation target to TARGET_CC -------------------------
import numba.cuda as nb_cuda  # noqa: E402
import numba.cuda.dispatcher as nb_dispatcher  # noqa: E402
from numba.cuda.cudadrv import devices as nb_devices  # noqa: E402

_fake_device = SimpleNamespace(
    compute_capability=TARGET_CC,
    id=0,
    name=b"cubie-precompile",
    MAX_SHARED_MEMORY_PER_BLOCK=49152,
    WARP_SIZE=32,
)
_fake_context = SimpleNamespace(device=_fake_device)
nb_dispatcher.get_current_device = lambda: _fake_device
nb_devices.get_context = lambda *args, **kwargs: _fake_context
nb_cuda.get_current_device = lambda: _fake_device


# ---- host-side stand-ins for device arrays and streams ---------------
class _FakeStream:
    handle = None

    def synchronize(self):
        pass


class _FakeDeviceArray(np.ndarray):
    """numpy stand-in for DeviceNDArray: identical typing signature,
    plus the host-copy API used by build-time diagnostics."""

    def copy_to_host(self, ary=None, stream=0):
        if ary is None:
            return np.array(self)
        np.copyto(ary, self)
        return ary

    def copy_to_device(self, ary, stream=0):
        np.copyto(self, np.asarray(ary).reshape(self.shape))


def _fake_dev(arr):
    return arr.view(_FakeDeviceArray)


def _fake_to_device(ary, stream=0, copy=True, to=None):
    arr = np.array(ary, copy=True)
    if to is not None:
        np.copyto(to, arr.reshape(np.shape(to)))
        return to
    return _fake_dev(arr)


class _FakeEvent:
    def record(self, stream=0):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 0.0


_fake_stream = _FakeStream()
nb_cuda.to_device = _fake_to_device
nb_cuda.device_array = (
    lambda shape, dtype=np.float64, *a, **k: _fake_dev(
        np.zeros(shape, dtype=dtype)
    )
)
nb_cuda.device_array_like = (
    lambda ary, stream=0: _fake_dev(np.zeros_like(ary))
)
nb_cuda.pinned_array = (
    lambda shape, dtype=np.float64, *a, **k: np.zeros(shape, dtype=dtype)
)
nb_cuda.event = lambda timing=True: _FakeEvent()
nb_cuda.synchronize = lambda: None
nb_cuda.stream = lambda: _fake_stream
nb_cuda.default_stream = lambda: _fake_stream

# ---- stub cubie's eager device touches --------------------------------
import importlib  # noqa: E402

import cubie.memory.stream_groups as stream_groups  # noqa: E402
import cubie.memory.mem_manager as mem_manager  # noqa: E402

_bsk = importlib.import_module("cubie.batchsolving.BatchSolverKernel")

stream_groups.cuda = SimpleNamespace(stream=lambda: _fake_stream)
mem_manager._ensure_cuda_context = lambda: None
mem_manager._pinned_host_array = (
    lambda shape, dtype: np.empty(shape, dtype=dtype)
)
_bsk.max_shared_memory_per_block = lambda: 49152

_MemoryManager = mem_manager.MemoryManager
_MemoryManager.allocate = (
    lambda self, shape, dtype, memory_type, stream=0: _fake_dev(
        np.zeros(shape, dtype=dtype)
    )
)


def _host_copy(self, instance, from_arrays, to_arrays, stream=None):
    for src, dst in zip(from_arrays, to_arrays):
        if getattr(src, "size", 0):
            np.copyto(dst, np.asarray(src).reshape(np.shape(dst)))


_MemoryManager.to_device = _host_copy
_MemoryManager.from_device = _host_copy
_MemoryManager.get_available_memory = lambda self, group: 8 << 30
_MemoryManager.get_memory_info = lambda self: (8 << 30, 24 << 30)

# ---- fake-launch: compile and cache instead of executing --------------
from numba.cuda.dispatcher import CUDADispatcher, _Kernel  # noqa: E402
from numba.cuda.core.caching import NullCache  # noqa: E402

STATS = {"compiled": 0, "cache_hit": 0, "uncached_noop": 0}


def _precompile(dispatcher, args):
    if isinstance(dispatcher._cache, NullCache):
        STATS["uncached_noop"] += 1
        return None

    argtypes = tuple(dispatcher.typeof_pyval(a) for a in args)
    if argtypes not in dispatcher.overloads:
        kernel = dispatcher._cache.load_overload(
            argtypes, dispatcher.targetctx
        )
        if kernel is not None:
            STATS["cache_hit"] += 1
        else:
            kernel = _Kernel(
                dispatcher.py_func, argtypes, **dispatcher.targetoptions
            )
            kernel._codelibrary.get_cubin(cc=TARGET_CC)
            dispatcher._cache.save_overload(argtypes, kernel)
            STATS["compiled"] += 1
        dispatcher.add_overload(kernel, argtypes)
    return None


def _getitem(self, config):
    dispatcher = self

    class _Shim:
        def __call__(self, *args):
            return _precompile(dispatcher, args)

    return _Shim()


CUDADispatcher.__getitem__ = _getitem
CUDADispatcher.call = (
    lambda self, args, launch_config: _precompile(self, args)
)


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(
        f"PRECOMPILE_STATS compiled={STATS['compiled']} "
        f"cache_hit={STATS['cache_hit']} "
        f"uncached_noop={STATS['uncached_noop']}"
    )
