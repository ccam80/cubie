<!-- Parent: ../AGENTS.md -->

# arrays

## Purpose
Host/device array coordination for batch solves. Owns the NumPy host arrays and Numba CUDA
device arrays the batch kernel reads and writes, brokering their sizing, allocation,
host↔device transfer, and run-axis chunking through the shared `MemoryManager`.
`BatchSolverKernel` owns one `InputArrays` and one `OutputArrays` (both `BaseArrayManager`
subclasses); each holds a `host` and a `device` `ArrayContainer` of `ManagedArray` metadata
wrappers — one per logical array, matched across containers by field name.

See `CUDAFactory` (root) for the `ArraySizingClass`/`.nonzero` pattern and attrs conventions.
Allocation, streams, and chunk math live in `cubie.memory`.

## Key Files
| File | Description |
|------|-------------|
| `BaseArrayManager.py` | `ManagedArray` (per-array metadata: dtype, `stride_order`, shapes, chunk axis/length, backing array), `ArrayContainer` (ABC), `BaseArrayManager` (ABC: registration, size/dtype checks, host updates, allocation, chunk-aware transfer, pinned↔numpy conversion). |
| `BatchInputArrays.py` | `InputArrayContainer` (`initial_values`, `parameters`, `driver_coefficients`) + `InputArrays` — sizes from `BatchInputSizes`; `initialise` stages H2D. |
| `BatchOutputArrays.py` | `OutputArrayContainer` (`state`, `observables`, `state_summaries`, `observable_summaries`, `status_codes`, `iteration_counters`) + `OutputArrays` — sizes from `BatchOutputSizes`; `finalise` does D2H + async writeback. |
| `__init__.py` | Empty — managers are imported from their modules. |

## For AI Agents

### Two containers per manager
Each manager owns a `host` and a `device` container of the same type; arrays are matched by
field name. Iterate both via `_iter_managed_arrays` (device then host), one container via
`container.iter_managed_arrays()`. Host is pinned by default; device is `"device"`.

### ManagedArray & chunking
- A `ManagedArray` always holds a real backing array — `__attrs_post_init__` allocates a
  `np_zeros` default, so `.array` is never `None`.
- Chunking is always along the `"run"` axis: the chunk axis index is `stride_order.index("run")`.
  Arrays without `"run"` in `stride_order` or with `is_chunked=False` (e.g.
  `driver_coefficients`) are never chunked; `needs_chunked_transfer` is true only when the full
  `shape` differs from `chunked_shape`.
- `status_codes` is `int32`/`("run",)`; `iteration_counters` is `int32`/`(time,variable,run)`
  default `(1,4,1)`. Float arrays get their dtype rebound to the solver precision in
  `update_from_solver`; integer arrays keep their dtype.

### Lifecycle
`from_solver(...)` builds a manager with sizes only (no allocation). `update(...)` refreshes
sizes/precision/run-count, sets host arrays (`update_host_arrays` — same-shape updates copy
values into the existing host buffer in place and queue `_needs_overwrite`; shape changes
stage into a buffer of the slot's memory type and queue reallocation), and calls
`allocate()`, which queues `ArrayRequest`s with the memory manager. The memory manager later drives
`_on_allocation_complete(response)`: attach device arrays, record
`chunked_shape`/`chunk_length`/`num_chunks`, set `_chunks`, and convert host arrays to pinned
(non-chunked) or plain numpy (chunked). `_invalidate_hook` drops device refs and re-marks
everything for reallocation.

### Teardown
Each manager registers a `weakref.finalize` (in `register_with_memory_manager`) that
deregisters it from the memory manager and runs the callables from `_teardown_cleanups()`
(`InputArrays`: pool `clear`; `OutputArrays`: pool `clear` + watcher `shutdown`) when the
manager is garbage collected — so the registry's keepalive on the device buffers, the pinned
staging pool, and the watcher thread are all released without waiting for the next
registration to purge. The finalizer callback must never close over the manager (that would
keep it alive); it holds only the memory manager, the instance id, the `settings` entry, and
the detached cleanup callables. `close()` runs that finalizer early and then `reset()`s, for
deterministic release; it is idempotent, and a closed manager should not be reused. Overriding
`_teardown_cleanups()` is how a subclass adds its own resources to the finalizer.

### Per-chunk hooks (called by `BatchSolverKernel.run` around each launch)
- `initialise(chunk_index)` — pre-launch. `InputArrays`: H2D (non-chunked copies the
  `_needs_overwrite` arrays; chunked stages each run-axis slice through a `ChunkBufferPool`
  pinned buffer). `OutputArrays`: no-op.
- `finalise(chunk_index)` — post-launch. `OutputArrays`: D2H for the outputs the compile
  flags enable (placeholder arrays are skipped); chunked copies into a pooled pinned buffer
  and submits a `PendingBuffer` to the `WritebackWatcher`; non-chunked transfers immediately
  (still async). `InputArrays`: releases its staging buffers.

### Memory types
Host arrays are `"pinned"` (page-locked → async transfer) for non-chunked runs, converted to
plain `"host"` numpy for chunked runs with per-chunk pinned staging from `ChunkBufferPool`
(`_convert_host_to_pinned`/`_convert_host_to_numpy`, run in `_on_allocation_complete`).

### Async writeback
Chunked outputs write back on a background `WritebackWatcher` thread: `finalise` records a CUDA
event and submits `PendingBuffer`s; `wait_pending()` blocks until the watcher drains; `reset()`
shuts the watcher down and clears the pool. Under `CUDA_SIMULATION` the event is `None` (treated
as immediately complete). `OutputArrays.finalise` reassigns `event.handle = event.handle.value`
to work around a numba event-handle issue — it is load-bearing, do not remove.

### Container mechanics
Containers use `@define(slots=False)` and discover their arrays by scanning `__dict__` for
`ManagedArray` instances (`_iter_field_items`), so fields are picked up dynamically. `attach`
warns (doesn't raise) on an unknown label.

### Sizes
`_sizes` is a `BatchInputSizes`/`BatchOutputSizes` (`ArraySizingClass`); `update_sizes` raises
`TypeError` if the replacement isn't the same subtype. `.nonzero` (floor empty/disabled dims to
1) is applied to `_sizes` before allocation, in `update_from_solver`. See root for the
`ArraySizingClass`/`.nonzero` pattern.

### Testing
`tests/batchsolving/arrays/`. Pinned/async-writeback paths short-circuit under `CUDA_SIMULATION`
(events treated as complete). Prefer building via `InputArrays.from_solver`/
`OutputArrays.from_solver` against a `BatchSolverKernel` fixture.

## Dependencies
### Internal
- `cubie.memory` (`default_memmgr`, `MemoryManager`, `ArrayRequest`/`ArrayResponse`,
  `chunk_buffer_pool.ChunkBufferPool`/`PinnedBuffer`); `cubie.outputhandling.output_sizes`
  (`ArraySizingClass`, `BatchInputSizes`, `BatchOutputSizes`); `cubie.batchsolving`
  (`ArrayTypes`); `cubie.batchsolving.writeback_watcher` (`WritebackWatcher`, `PendingBuffer`);
  `cubie.cuda_simsafe` (`DeviceNDArrayBase`, `CUDA_SIMULATION`); `cubie._utils` (validators).
### External
- `numpy`; `attrs`; `numba.cuda` (events in `OutputArrays.finalise`).
