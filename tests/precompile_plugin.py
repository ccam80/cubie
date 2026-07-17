"""Compile test kernels into a shared cache without a GPU."""
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _parse_target_cc():
    raw_cc = os.environ.get("CUBIE_TARGET_CC", "").strip()
    if not raw_cc:
        raise RuntimeError(
            "tests.precompile_plugin requires CUBIE_TARGET_CC, "
            "for example '8.9'."
        )
    try:
        target_cc = tuple(
            int(part) for part in raw_cc.replace(",", ".").split(".")
        )
    except ValueError as exc:
        raise RuntimeError(
            f"CUBIE_TARGET_CC must look like '8.9', got {raw_cc!r}."
        ) from exc
    if len(target_cc) != 2 or any(part < 0 for part in target_cc):
        raise RuntimeError(
            f"CUBIE_TARGET_CC must look like '8.9', got {raw_cc!r}."
        )
    return target_cc


TARGET_CC = _parse_target_cc()
TARGET_CC_TEXT = ".".join(str(part) for part in TARGET_CC)

if not os.environ.get("CUBIE_KERNEL_CACHE_DIR", "").strip():
    raise RuntimeError(
        "tests.precompile_plugin requires CUBIE_KERNEL_CACHE_DIR."
    )

# Keep every kernel in the uploaded artifact.
os.environ.setdefault("CUBIE_MAX_CACHE_ENTRIES", "0")


def _select_backend():
    requested = os.environ.get("CUBIE_CUDA_BACKEND", "").strip().lower()
    if requested == "mlir":
        return "mlir"
    if requested == "numba-cuda":
        return "numba-cuda"
    if requested:
        raise RuntimeError(f"Unsupported CUBIE_CUDA_BACKEND: {requested!r}")
    try:
        import numba.cuda  # noqa: F401
    except ImportError:
        return "mlir"
    return "numba-cuda"


BACKEND = _select_backend()

if BACKEND == "mlir":
    from cuda.core._device import ComputeCapability  # noqa: E402
    from numba_cuda_mlir import cuda as backend_cuda  # noqa: E402
    from numba_cuda_mlir import tools as mlir_tools  # noqa: E402

    mlir_tools._cached_cc = ComputeCapability(*TARGET_CC)
else:
    import numba.cuda as backend_cuda  # noqa: E402


class _FakeStream:
    handle = None

    def synchronize(self):
        pass


class _FakeDeviceArray(np.ndarray):
    """Host array with the device-copy methods cubie uses."""

    def copy_to_host(self, ary=None, stream=0):
        if ary is None:
            return np.array(self)
        np.copyto(ary, self)
        return ary

    def copy_to_device(self, ary, stream=0):
        np.copyto(self, np.asarray(ary).reshape(self.shape))


def _fake_device_array(array):
    return array.view(_FakeDeviceArray)


def _fake_to_device(ary, stream=0, copy=True, to=None):
    array = np.array(ary, copy=True)
    if to is not None:
        np.copyto(to, array.reshape(np.shape(to)))
        return to
    return _fake_device_array(array)


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
backend_cuda.to_device = _fake_to_device
backend_cuda.device_array = (
    lambda shape, dtype=np.float64, *args, **kwargs: _fake_device_array(
        np.zeros(shape, dtype=dtype)
    )
)
backend_cuda.device_array_like = (
    lambda ary, stream=0: _fake_device_array(np.zeros_like(ary))
)
backend_cuda.pinned_array = (
    lambda shape, dtype=np.float64, *args, **kwargs: np.zeros(
        shape, dtype=dtype
    )
)
backend_cuda.event = lambda timing=True: _FakeEvent()
backend_cuda.synchronize = lambda: None
backend_cuda.stream = lambda: _fake_stream
backend_cuda.default_stream = lambda: _fake_stream

if BACKEND == "numba-cuda":
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
    backend_cuda.get_current_device = lambda: _fake_device

# Replace cubie's eager allocations with zero-filled host arrays.
import cubie.memory.mem_manager as mem_manager  # noqa: E402
import cubie.memory.stream_groups as stream_groups  # noqa: E402

_batch_solver_kernel = importlib.import_module(
    "cubie.batchsolving.BatchSolverKernel"
)
stream_groups.cuda = SimpleNamespace(stream=lambda: _fake_stream)
mem_manager._ensure_cuda_context = lambda: None
mem_manager._pinned_host_array = (
    lambda shape, dtype: np.zeros(shape, dtype=dtype)
)
_batch_solver_kernel.max_shared_memory_per_block = lambda: 49152

_MemoryManager = mem_manager.MemoryManager
_MemoryManager.allocate = (
    lambda self, shape, dtype, memory_type, stream=0: _fake_device_array(
        np.zeros(shape, dtype=dtype)
    )
)


def _host_copy(self, instance, from_arrays, to_arrays, stream=None):
    for source, destination in zip(from_arrays, to_arrays):
        if getattr(source, "size", 0):
            np.copyto(
                destination,
                np.asarray(source).reshape(np.shape(destination)),
            )


_MemoryManager.to_device = _host_copy
_MemoryManager.from_device = _host_copy
_MemoryManager.get_available_memory = lambda self, group: 8 << 30
_MemoryManager.get_memory_info = lambda self: (8 << 30, 24 << 30)

_STAT_KEYS = (
    "compiled",
    "cache_hit",
    "uncached_noop",
    "production_compiled",
    "production_cache_hit",
)
STATS = {key: 0 for key in _STAT_KEYS}
_WORKER_STATS = []


def _is_production_kernel(dispatcher):
    cache_name = getattr(dispatcher._cache, "_system_name", "")
    function_name = getattr(dispatcher.py_func, "__name__", "")
    return not cache_name.startswith("harness_") and (
        function_name != "evaluate_rhs"
    )


def _record(dispatcher, key):
    STATS[key] += 1
    if _is_production_kernel(dispatcher):
        production_key = f"production_{key}"
        if production_key in STATS:
            STATS[production_key] += 1


def _combined_stats():
    combined = STATS.copy()
    for worker_stats in _WORKER_STATS:
        for key in _STAT_KEYS:
            combined[key] += worker_stats.get(key, 0)
    return combined


if BACKEND == "numba-cuda":
    from numba.cuda.cext import _dispatcher  # noqa: E402
    from numba.cuda.core.caching import NullCache  # noqa: E402
    from numba.cuda.dispatcher import (  # noqa: E402
        CUDADispatcher,
        _LAUNCH_CONFIG_KW,
        _Kernel,
    )

    _Kernel.bind = (
        lambda self: self._codelibrary.get_cubin(cc=TARGET_CC)
    )

    def _precompile_numba(dispatcher, args, launch_config):
        previous_args = launch_config._push_args(args)
        try:
            selected = dispatcher._select_launch_config_dispatcher(
                launch_config
            )
            if selected is not dispatcher:
                selected._cache = dispatcher._cache

            if isinstance(selected._cache, NullCache):
                _record(selected, "uncached_noop")
                return None

            cache_hits = sum(selected._cache_hits.values())
            cache_misses = sum(selected._cache_misses.values())
            _dispatcher.Dispatcher._cuda_call(
                selected,
                *args,
                **{_LAUNCH_CONFIG_KW: launch_config},
            )
            if sum(selected._cache_hits.values()) > cache_hits:
                _record(selected, "cache_hit")
            if sum(selected._cache_misses.values()) > cache_misses:
                _record(selected, "compiled")
            return None
        finally:
            launch_config._pop_args(previous_args)

    def _numba_getitem(self, config):
        dispatcher = self
        config = list(config)
        if len(config) >= 3 and isinstance(config[2], _FakeStream):
            config[2] = 0
        launch_config = dispatcher.configure(*config)

        class _Shim:
            def __call__(self, *args):
                return _precompile_numba(
                    dispatcher, args, launch_config
                )

        return _Shim()

    CUDADispatcher.__getitem__ = _numba_getitem
    CUDADispatcher.call = _precompile_numba
else:
    from numba_cuda_mlir.descriptor import MLIRDispatcher  # noqa: E402
    from numba_cuda_mlir.numba_cuda import types, typing  # noqa: E402
    from numba_cuda_mlir.numba_cuda.core.caching import (  # noqa: E402
        NullCache,
    )

    def _precompile_mlir(dispatcher, args):
        if isinstance(dispatcher._cache, NullCache):
            _record(dispatcher, "uncached_noop")
            return None

        argtypes = tuple(dispatcher.typeof_pyval(arg) for arg in args)
        if argtypes in dispatcher.overloads:
            return None

        dispatcher.targetoptions["chip"] = (
            f"sm_{TARGET_CC[0]}{TARGET_CC[1]}"
        )
        cache_hits = sum(dispatcher._cache_hits.values())
        signature = typing.signature(types.none, *argtypes)
        dispatcher.compile(signature)
        if sum(dispatcher._cache_hits.values()) > cache_hits:
            _record(dispatcher, "cache_hit")
        else:
            _record(dispatcher, "compiled")
        return None

    def _mlir_getitem(self, config):
        dispatcher = self

        class _Shim:
            def __call__(self, *args):
                return _precompile_mlir(dispatcher, args)

        return _Shim()

    # Compilation stays on MLIR's public path; only execution is suppressed.
    MLIRDispatcher.__getitem__ = _mlir_getitem


def pytest_testnodedown(node, error):
    worker_stats = getattr(node, "workeroutput", {}).get(
        "cubie_precompile_stats"
    )
    if worker_stats:
        _WORKER_STATS.append(worker_stats)


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workeroutput"):
        session.config.workeroutput["cubie_precompile_stats"] = STATS.copy()
        return

    report_path = os.environ.get("CUBIE_PRECOMPILE_REPORT", "").strip()
    if not report_path:
        return

    report = _combined_stats()
    report.update(
        backend=BACKEND,
        target_cc=TARGET_CC_TEXT,
        exitstatus=int(exitstatus),
    )
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")


def pytest_terminal_summary(terminalreporter):
    stats = _combined_stats()
    terminalreporter.write_line(
        "PRECOMPILE_STATS "
        + " ".join(f"{key}={stats[key]}" for key in _STAT_KEYS)
    )
