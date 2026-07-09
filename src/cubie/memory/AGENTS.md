<!-- Parent: ../AGENTS.md -->

# memory

## Purpose
GPU memory management for CuBIE. `MemoryManager` is a proportion-based VRAM allocator that
registers caller instances, enforces per-instance memory caps, chunks batches along the run
axis when they exceed available VRAM, and routes host/device transfers to the right CUDA
stream. It runs as a process-wide instance (`default_memmgr`, created in `__init__.py` at
import) — a singleton **by convention**, not enforced via `__new__`. CuPy is the single device
allocation provider on a real GPU (no Numba allocator fallback): device arrays come from CuPy's
memory pool and host staging buffers from CuPy's pinned memory pool. The CUDA simulator never
touches CuPy — it keeps its own numpy-backed fakes. Supporting pieces: `StreamGroups` (CUDA
stream grouping), `ArrayRequest`/`ArrayResponse` (allocation metadata), `ChunkBufferPool`
(reusable pinned staging buffers).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Instantiates `default_memmgr = MemoryManager()`; re-exports `MemoryManager`, `current_cupy_stream`. |
| `mem_manager.py` | `MemoryManager` (central allocator); `InstanceMemorySettings` (per-instance registry entry); `ALL_MEMORY_MANAGER_PARAMETERS`; `MIN_AUTOPOOL_SIZE`; `current_cupy_stream` (Numba→CuPy stream forwarding); `_pinned_host_array`. |
| `array_requests.py` | `ArrayRequest` (shape/dtype/placement spec) and `ArrayResponse` (allocated arrays + chunk metadata). |
| `stream_groups.py` | `StreamGroups` — maps instance ids to named groups, each backed by a CUDA stream. |
| `chunk_buffer_pool.py` | `PinnedBuffer` + `ChunkBufferPool` — reusable pinned staging buffers. Not exported from `__init__.py`. |

## For AI Agents

### Registration & pools
- `register(instance, proportion=None, invalidate_cache_hook=…, allocation_ready_hook=…, stream_group="default")` once per object. `proportion=None` → auto pool (equal share of remaining VRAM); a float → manual pool. `MIN_AUTOPOOL_SIZE = 0.05` reserves 5% for the auto pool; a manual proportion that would crowd it below 5% raises `ValueError` if auto instances exist (else warns).
- The registry is keyed by `id(instance)`. Keep a live reference to every registered object — GC frees the id and a new object can silently claim the slot.
- Two limit modes (`_mode`, default `"passive"`): `"passive"` computes caps but doesn't enforce (returns raw free VRAM); `"active"` enforces per-instance caps. Switch via `set_limit_mode()`.

### Single allocation provider
CuPy is the only device allocator; there is no `set_allocator`/allocator-selection API and no
`"allocator"` key in `ALL_MEMORY_MANAGER_PARAMETERS`. `MemoryManager.__attrs_post_init__` raises
a clear `ImportError` if CuPy is not importable on a real GPU (the CUDA simulator never needs
it). `allocate()` routes `"device"` requests through `cupy.empty`, `"pinned"` requests through
`cupyx.empty_pinned`, and raises `NotImplementedError` for `"mapped"`/`"managed"` (unsupported,
matching Numba's own upcoming memory-API deprecation for `device_array`/`to_device`/pinned/mapped
allocation). `to_device`/`from_device` use CuPy's `ndarray.set`/`ndarray.get`, wrapped in
`current_cupy_stream` so the copy stays ordered on the instance's Numba stream.

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
  `{device, pinned}` — `mapped`/`managed` are rejected at construction with `ValueError`
  (`ManagedArray.memory_type` likewise allows only `device`/`pinned`/`host`);
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
  always forwards the given Numba stream into CuPy as an `ExternalStream` (Numba's default
  stream, handle `0`, is left as CuPy's ambient current stream instead of wrapped).
- `MemoryManager.allocate`/`to_device`/`from_device` wrap their CuPy calls in
  `current_cupy_stream(stream)` so allocation and copies stay ordered on the instance's stream.
  `get_memory_info()` still queries whole-device free/total via the Numba context, not a CuPy
  pool's own accounting.

### ChunkBufferPool (internal, not exported)
Reusable pinned staging buffers for chunked transfers, keyed by `(array_name, shape, dtype)`.
`acquire` reuses a free matching buffer or allocates one; `release` marks it free (doesn't
free); `clear` frees all (call on error paths). Thread-safe via a `Lock`. Under
`CUDA_SIMULATION` it allocates plain numpy arrays; otherwise it uses CuPy's pinned memory pool
(`cupyx.empty_pinned`). Consumers: `InputArrays`/`OutputArrays`.

### Testing
`tests/memory/` (`test_memmgmt.py`, `test_array_requests.py`, `test_stream_groups.py`,
`test_chunk_buffer_pool.py`, `test_cupyemm.py` — needs the `cupy` marker + a real GPU with
cupy installed). CuPy-returning-array assertions and the CuPy-stream-forwarding test are marked
`nocudasim`; construction-time CuPy requirement, the `ValueError` on
`"mapped"`/`"managed"` requests, and `allocate()`'s `NotImplementedError` backstop for
direct calls are exercised directly.

## Dependencies
### Internal
- `cubie.cuda_simsafe` (`Stream`, `DeviceNDArrayBase`, `CUDA_SIMULATION`, `current_mem_info`);
  `cubie._utils` (validators in `array_requests.py`).
### External
- `numba`/`numba.cuda` (context/stream management, kernel launch); `attrs`; `numpy`; `cupy`
  (required on a real GPU — single device allocation provider, imported once through
  `cubie.cuda_simsafe`, which supplies `None` stand-ins under the CUDA simulator;
  `cupyx.empty_pinned` for pinned host buffers).
