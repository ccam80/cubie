<!-- Parent: ../AGENTS.md -->

# memory

## Purpose
GPU memory management for CuBIE. `MemoryManager` is a proportion-based VRAM allocator that
registers caller instances, enforces per-instance memory caps, chunks batches along the run
axis when they exceed available VRAM, and routes host/device transfers to the right CUDA
stream. It runs as a process-wide instance (`default_memmgr`, created in `__init__.py` at
import) — a singleton **by convention**, not enforced via `__new__`. Supporting pieces:
`StreamGroups` (CUDA stream grouping), `ArrayRequest`/`ArrayResponse` (allocation metadata),
`ChunkBufferPool` (reusable pinned staging buffers), and optional CuPy memory-pool plugins for
Numba's External Memory Manager interface.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Instantiates `default_memmgr = MemoryManager()`; re-exports `MemoryManager`, `current_cupy_stream`, `CuPySyncNumbaManager`, `CuPyAsyncNumbaManager`. |
| `mem_manager.py` | `MemoryManager` (central allocator); `InstanceMemorySettings` (per-instance registry entry); `ALL_MEMORY_MANAGER_PARAMETERS`; `MIN_AUTOPOOL_SIZE`. |
| `array_requests.py` | `ArrayRequest` (shape/dtype/placement spec) and `ArrayResponse` (allocated arrays + chunk metadata). |
| `stream_groups.py` | `StreamGroups` — maps instance ids to named groups, each backed by a CUDA stream. |
| `cupy_emm.py` | `CuPyNumbaManager` (internal base), `CuPyAsyncNumbaManager` (async pool), `CuPySyncNumbaManager` (sync pool), and the `current_cupy_stream` context-manager class. |
| `chunk_buffer_pool.py` | `PinnedBuffer` + `ChunkBufferPool` — reusable pinned staging buffers. Not exported from `__init__.py`. |

## For AI Agents

### Registration & pools
- `register(instance, proportion=None, invalidate_cache_hook=…, allocation_ready_hook=…, stream_group="default")` once per object. `proportion=None` → auto pool (equal share of remaining VRAM); a float → manual pool. `MIN_AUTOPOOL_SIZE = 0.05` reserves 5% for the auto pool; a manual proportion that would crowd it below 5% raises `ValueError` if auto instances exist (else warns).
- The registry is keyed by `id(instance)`. Keep a live reference to every registered object — GC frees the id and a new object can silently claim the slot.
- Two limit modes (`_mode`, default `"passive"`): `"passive"` computes caps but doesn't enforce (returns raw free VRAM); `"active"` enforces per-instance caps. Switch via `set_limit_mode()`.

### Allocator switching destroys everything
`set_allocator("default" | "cupy" | "cupy_async")` swaps the Numba EMM plugin, then
`cuda.close()` → `invalidate_all()` (calls every registered `invalidate_hook`) →
`reinit_streams()`. **Every kernel and device array from before the call is invalid.** Register
with an `invalidate_cache_hook` so owning `CUDAFactory`s rebuild. An unknown name raises
`ValueError`.

### Queued / chunked allocation
- `queue_request(instance, {label: ArrayRequest(...)})` per participating instance, then
  `allocate_queue(triggering_instance)` once. The manager computes chunk parameters across all
  queued requests in the stream group and calls each instance's
  `allocation_ready_hook(ArrayResponse)`.
- **Notary instances** — same stream group, no queued requests — still get an
  `allocation_ready_hook` with an empty `arr` dict but correct `chunks`/`chunk_length`; hooks
  must handle empty `arr`.
- Chunking replaces `shape[chunk_axis_index]` with `chunk_length`; `unchunkable=True` keeps the
  full shape.

### ArrayRequest / ArrayResponse
- `ArrayRequest.dtype` is validated to exactly `float64`/`float32`/`int32`; `memory` ∈
  `{device, mapped, pinned, managed}`; `chunk_axis_index` defaults to `2` (the run axis in the
  3-D output layout — callers with other layouts pass their own index); `total_runs ≥ 1` (the
  manager reads it from the first chunkable request to size chunks).
- **`managed` allocation raises `NotImplementedError`** — the type validates but `allocate()`
  doesn't implement it.
- `ArrayResponse` carries `arr` (label→device array), `chunks`, `chunk_length`, `chunked_shapes`.

### Stream groups
- Groups map instance ids → a shared `cuda.stream()`. The `"default"` group is created
  **lazily** on the first `register(..., stream_group="default")` (via `add_instance`, with a
  fresh `cuda.stream()`), not at construction. `reinit_streams()` (run after the context reset
  in `set_allocator`) replaces every group's stream with a new `cuda.stream()`.
- `add_instance`/`get_group`/`get_stream`/`change_group` accept either a plain `int` (used as
  the id directly) or any object (uses `id()`).

### CuPy EMM (optional)
- `current_cupy_stream` is a **class** context manager that forwards a Numba stream as a CuPy
  `ExternalStream`; it's a no-op when the active allocator isn't CuPy. Constructing any CuPy
  manager or `current_cupy_stream` **raises `ImportError` if `cupy` isn't installed — there is
  no silent fallback** (cupy is imported lazily inside the methods, so the module still loads
  without it).
- `CuPyAsyncNumbaManager` (async pool) / `CuPySyncNumbaManager` (sync pool) wrap CuPy pools;
  `CuPyNumbaManager` (base) is internal. `get_memory_info()` returns the CuPy **pool**'s
  free/total, not whole-device memory — this changes chunk sizing when the pool is smaller than
  VRAM.

### ChunkBufferPool (internal, not exported)
Reusable pinned staging buffers for chunked transfers, keyed by `(array_name, shape, dtype)`.
`acquire` reuses a free matching buffer or allocates one; `release` marks it free (doesn't
free); `clear` frees all (call on error paths). Thread-safe via a `Lock`. Under
`CUDA_SIMULATION` it allocates plain numpy arrays instead of `cuda.pinned_array`. Consumers:
`InputArrays`/`OutputArrays`.

### Testing
`tests/memory/` (`test_memmgmt.py`, `test_array_requests.py`, `test_stream_groups.py`,
`test_chunk_buffer_pool.py`, `test_cupyemm.py` — needs the `cupy` marker + a real GPU with
cupy installed).

## Dependencies
### Internal
- `cubie.cuda_simsafe` (base memory-manager classes, `Stream`, `MemoryPointer`, `MemoryInfo`,
  `DeviceNDArrayBase`, `set_cuda_memory_manager`, `CUDA_SIMULATION`); `cubie._utils`
  (validators in `array_requests.py`).
### External
- `numba`/`numba.cuda` (context/stream management, `device_array`/`pinned_array`/`mapped_array`);
  `attrs`; `numpy`; `cupy` (optional, imported lazily in `cupy_emm.py`).
