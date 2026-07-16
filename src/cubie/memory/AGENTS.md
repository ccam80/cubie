<!-- Parent: ../AGENTS.md -->

# memory

## Purpose
GPU memory management for CuBIE. `MemoryManager` is a proportion-based VRAM allocator that
registers caller instances, enforces per-instance memory caps, chunks batches along the run
axis when they exceed available VRAM, and routes host/device transfers to the right CUDA
stream. It runs as a process-wide instance (`default_memmgr`, created in `__init__.py` at
import) — a singleton **by convention**, not enforced via `__new__`. CuPy's async memory pool
is the single device allocation provider on a real GPU, plugged into Numba as an External
Memory Manager (`cupy_emm.py`), so `cuda.device_array` returns **native** `DeviceNDArray`
objects backed by pooled, stream-ordered allocations. Pinned host buffers come from Numba
(`cuda.pinned_array`) and the chunk staging pool from CuPy (`cupyx.empty_pinned`). The CUDA
simulator never touches CuPy — it keeps its own numpy-backed fakes. Supporting pieces:
`StreamGroups` (CUDA stream grouping), `ArrayRequest`/`ArrayResponse` (allocation metadata),
`ChunkBufferPool` (reusable pinned staging buffers).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Installs the CuPy async EMM (`install_async_emm()`, before any CUDA context exists), instantiates `default_memmgr = MemoryManager()`; re-exports `MemoryManager`, `current_cupy_stream`, `CuPyAsyncNumbaManager`. |
| `cupy_emm.py` | `CuPyAsyncNumbaManager` — Numba EMM plugin drawing device memory from `cupy.cuda.MemoryAsyncPool`; `install_async_emm()`. |
| `mem_manager.py` | `MemoryManager` (central allocator); `InstanceMemorySettings` (per-instance registry entry); `ALL_MEMORY_MANAGER_PARAMETERS`; `MIN_AUTOPOOL_SIZE`; `current_cupy_stream` (Numba→CuPy stream forwarding); `_pinned_host_array`. |
| `array_requests.py` | `ArrayRequest` (shape/dtype/placement spec) and `ArrayResponse` (allocated arrays + chunk metadata). |
| `stream_groups.py` | `StreamGroups` — maps instance ids to named groups, each backed by a CUDA stream. |
| `chunk_buffer_pool.py` | `PinnedBuffer` + `ChunkBufferPool` — reusable pinned staging buffers. Not exported from `__init__.py`. |

## For AI Agents

### Registration & pools
- `register(instance, proportion=None, invalidate_cache_hook=…, allocation_ready_hook=…, stream_group="default")` once per object. `proportion=None` → auto pool (equal share of remaining VRAM); a float → manual pool. `MIN_AUTOPOOL_SIZE = 0.05` reserves 5% for the auto pool; a manual proportion that would crowd it below 5% raises `ValueError` if auto instances exist (else warns).
- The registry is keyed by `id(instance)`. Keep a live reference to every registered object — GC frees the id and a new object can silently claim the slot.
- Two limit modes (`_mode`, default `"passive"`): `"passive"` computes caps but doesn't enforce (returns raw free VRAM); `"active"` enforces per-instance caps. Switch via `set_limit_mode()`.

### Deregistration & teardown
- Registry allocations keep device arrays alive until deregistration.
- `release_instance` removes one exact registry entry. The identity check
  protects against reused object IDs.
- Explicit close reports cleanup failures and can be retried. Finalizers are
  best effort and do not raise during interpreter shutdown.
- Allocation, copies, launch, and release use the run's stream. Memory caps
  cause chunking without device-wide synchronization or garbage collection.
- Physical pressure may evict opted-in owners after their completion event.
  Manual and external registrations are protected.

### Host spill
- Host arrays above their owner's spill threshold use a temporary memmap.
- Memmap transfers use bounded pinned staging buffers.
- Spill settings belong to the solver owner, not the shared manager.

### Single allocation provider
CuPy's async pool is the only device allocator, reached through the EMM plugin; `cupy`/`cupyx`
come from `cubie.cuda_simsafe`, which imports them at package import on a real GPU.
`allocate()` routes `"device"` requests through `cuda.device_array` (inside
`current_cupy_stream`, so the pool allocation is stream-ordered) and `"pinned"` requests
through `cuda.pinned_array`; any other placement raises `ValueError`. `to_device`/`from_device`
issue streamed `cuda.cudadrv.driver.host_to_device`/`device_to_host` copies between pinned
host buffers and native device arrays, sized by the pinned buffer's `nbytes`. Device arrays
must be allocated (via `allocate_queue`) before `to_device` copies into them.

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
  `{device, pinned}` (`ManagedArray.memory_type` allows `device`/`pinned`/`host`); any other
  placement raises `ValueError` at construction;
  `chunk_axis_index` defaults to `2` (the run axis in the 3-D output layout — callers with
  other layouts pass their own index); `total_runs ≥ 1` (the manager reads it from the first
  chunkable request to size chunks).
- `ArrayResponse` carries `arr` (label→device array), `chunks`, `chunk_length`, `chunked_shapes`.

### Stream groups
- Groups map instance ids → a shared `cuda.stream()`. The `"default"` group is created
  **lazily** on the first `register(..., stream_group="default")` (via `add_instance`, with a
  fresh `cuda.stream()`), not at construction. `reinit_streams()` replaces every group's stream
  with a new `cuda.stream()`.
- `add_instance`/`get_group`/`get_stream`/`change_group` accept either a plain `int` (used as
  the id directly) or any object (uses `id()`).

### CuPy stream forwarding
- `current_cupy_stream` (defined in `mem_manager.py`) is a **class** context manager that
  always forwards the given Numba stream into CuPy via `cupy.cuda.Stream.from_external`
  (Numba's default stream, handle `0`, is left as CuPy's ambient current stream instead of
  wrapped).
- Allocation and release enter the same external CuPy stream. Transfers use
  that Numba stream directly. `get_memory_info()` reports device-wide memory.

### ChunkBufferPool (internal, not exported)
Reusable pinned staging buffers for chunked transfers, keyed by `(array_name, shape, dtype)`.
`acquire` reuses a free matching buffer or allocates one; `release` marks it free (doesn't
free); `clear` frees all (call on error paths). Thread-safe via a `Lock`. Under
`CUDA_SIMULATION` it allocates plain numpy arrays; otherwise it uses CuPy's pinned memory pool
(`cupyx.empty_pinned`). Consumers: `InputArrays`/`OutputArrays`.

### Testing
`tests/memory/` (`test_memmgmt.py`, `test_array_requests.py`, `test_stream_groups.py`,
`test_chunk_buffer_pool.py`, `test_memmgmt.py` — needs the `cupy` marker + a real GPU with
cupy installed). Native-device-array assertions and the CuPy-stream-forwarding test are marked
`nocudasim`; the `ValueError` on unsupported placements is exercised at both request
construction and direct `allocate()` calls.

## Dependencies
### Internal
- `cubie.cuda_simsafe` (`Stream`, `DeviceNDArrayBase`, `CUDA_SIMULATION`, `current_mem_info`);
  `cubie._utils` (validators in `array_requests.py`).
### External
- `numba`/`numba.cuda` (context/stream management, kernel launch, pinned arrays, driver
  copies); `attrs`; `numpy`; `cupy` (required on a real GPU — its async pool backs all device
  allocation through the EMM plugin, imported once through `cubie.cuda_simsafe`, which
  supplies `None` stand-ins under the CUDA simulator; `cupyx.empty_pinned` for the chunk
  staging pool).
