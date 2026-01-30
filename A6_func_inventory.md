# A6 Functionality Inventory — Memory & Batch Solving

---

## `memory/mem_manager.py`

### Module-Level Functions

#### `placeholder_invalidate`

| # | Functionality |
|---|--------------|
| 1 | No-op function returns None |

#### `placeholder_dataready`

| # | Functionality |
|---|--------------|
| 2 | No-op function accepts ArrayResponse and returns None |

#### `_ensure_cuda_context`

| # | Functionality |
|---|--------------|
| 3 | Skips entirely in CUDA_SIMULATION mode |
| 4 | Calls `cuda.current_context()` and succeeds when context valid |
| 5 | Raises RuntimeError when context is None |
| 6 | Wraps non-None exceptions with helpful RuntimeError message |

#### `get_portioned_request_size`

| # | Functionality |
|---|--------------|
| 7 | Returns (chunkable_bytes, unchunkable_bytes) summed across all instance requests |
| 8 | Chunkable portion includes only requests where `is_request_chunkable` is True |
| 9 | Unchunkable portion includes only requests where `is_request_chunkable` is False |
| 10 | Byte size computed as `prod(shape) * dtype().itemsize` |

#### `is_request_chunkable`

| # | Functionality |
|---|--------------|
| 11 | Returns False when `request.unchunkable` is True |
| 12 | Returns False when shape is empty (len == 0) |
| 13 | Returns False when `chunk_axis_index` is None |
| 14 | Returns False when `chunk_axis_index >= len(shape)` |
| 15 | Returns False when run axis dimension == 1 |
| 16 | Returns True when all above checks pass |

#### `replace_with_chunked_size`

| # | Functionality |
|---|--------------|
| 17 | Replaces dimension at `axis_index` with `chunked_size` |
| 18 | Other dimensions unchanged |

### `InstanceMemorySettings`

#### Construction

| # | Functionality |
|---|--------------|
| 19 | Defaults: proportion=1.0, allocations={}, invalidate_hook=placeholder, allocation_ready_hook=placeholder, cap=None |
| 20 | Validators enforce types (float, dict, Callable, Optional[int]) |

#### `add_allocation`

| # | Functionality |
|---|--------------|
| 21 | Frees old allocation when key already exists before adding new |
| 22 | Adds new allocation to dict |

#### `free`

| # | Functionality |
|---|--------------|
| 23 | Deletes allocation by key when present |
| 24 | Emits warning when key not found |

#### `free_all`

| # | Functionality |
|---|--------------|
| 25 | Frees all allocations via copy-then-iterate |

#### `allocated_bytes` property

| # | Functionality |
|---|--------------|
| 26 | Returns sum of `.nbytes` across all tracked allocations |
| 27 | Returns 0 when no allocations |

### `MemoryManager`

#### `__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 28 | Sets totalmem from `get_memory_info()` on success |
| 29 | Sets totalmem=1 on ValueError (cuda-less env), warns |
| 30 | Sets totalmem=1 on unexpected Exception, warns |
| 31 | Resets registry to empty dict |

#### `register`

| # | Functionality |
|---|--------------|
| 32 | Raises ValueError if instance already registered |
| 33 | Adds instance to stream_groups |
| 34 | Creates InstanceMemorySettings with hooks |
| 35 | Adds to manual pool when proportion provided and valid |
| 36 | Raises ValueError when proportion outside [0,1] |
| 37 | Adds to auto pool when proportion is None/falsy |

#### `set_allocator`

| # | Functionality |
|---|--------------|
| 38 | Sets CuPyAsyncNumbaManager for "cupy_async" |
| 39 | Sets CuPySyncNumbaManager for "cupy" |
| 40 | Sets NumbaCUDAMemoryManager for "default" |
| 41 | Raises ValueError for unknown name |
| 42 | Calls set_cuda_memory_manager, cuda.close(), invalidate_all, reinit_streams |

#### `set_limit_mode`

| # | Functionality |
|---|--------------|
| 43 | Sets mode to "passive" or "active" |
| 44 | Raises ValueError for unknown mode |

#### `get_stream`

| # | Functionality |
|---|--------------|
| 45 | Forwards to `stream_groups.get_stream(instance)` |

#### `change_stream_group`

| # | Functionality |
|---|--------------|
| 46 | Forwards to `stream_groups.change_group(instance, new_group)` |

#### `reinit_streams`

| # | Functionality |
|---|--------------|
| 47 | Forwards to `stream_groups.reinit_streams()` |

#### `invalidate_all`

| # | Functionality |
|---|--------------|
| 48 | Calls free_all then each instance's invalidate_hook |

#### `set_manual_proportion`

| # | Functionality |
|---|--------------|
| 49 | Raises ValueError when proportion < 0 or > 1 |
| 50 | Moves from auto pool to manual pool when instance in auto pool |
| 51 | Re-adds to manual pool when already in manual pool |

#### `set_manual_limit_mode`

| # | Functionality |
|---|--------------|
| 52 | No-op when instance already in manual pool |
| 53 | Removes from auto pool and adds to manual pool with new proportion |

#### `set_auto_limit_mode`

| # | Functionality |
|---|--------------|
| 54 | No-op when instance already in auto pool |
| 55 | Removes from manual pool and adds to auto pool |

#### `proportion`

| # | Functionality |
|---|--------------|
| 56 | Returns `registry[id(instance)].proportion` |

#### `cap`

| # | Functionality |
|---|--------------|
| 57 | Returns `registry[id(instance)].cap` |

#### `manual_pool_proportion` property

| # | Functionality |
|---|--------------|
| 58 | Returns sum of proportions for all manually-allocated instances |

#### `auto_pool_proportion` property

| # | Functionality |
|---|--------------|
| 59 | Returns sum of proportions for all auto-allocated instances |

#### `_add_manual_proportion`

| # | Functionality |
|---|--------------|
| 60 | Raises ValueError when new total > 1.0 |
| 61 | Raises ValueError when new total > 1.0 - MIN_AUTOPOOL_SIZE and auto pool non-empty |
| 62 | Warns when new total > 1.0 - MIN_AUTOPOOL_SIZE and auto pool empty |
| 63 | Appends to manual pool, sets proportion and cap |
| 64 | Calls _rebalance_auto_pool |

#### `_add_auto_proportion`

| # | Functionality |
|---|--------------|
| 65 | Raises ValueError when autopool available <= MIN_AUTOPOOL_SIZE |
| 66 | Appends to auto pool and rebalances |
| 67 | Returns assigned proportion |

#### `_rebalance_auto_pool`

| # | Functionality |
|---|--------------|
| 68 | No-op when auto pool empty |
| 69 | Distributes remaining proportion equally among auto-pool instances |
| 70 | Sets cap = int(each_proportion * totalmem) for each |

#### `free`

| # | Functionality |
|---|--------------|
| 71 | Frees allocation by label across all instances that have it |

#### `free_all`

| # | Functionality |
|---|--------------|
| 72 | Calls free_all on every registered instance |

#### `_check_requests`

| # | Functionality |
|---|--------------|
| 73 | Raises TypeError if requests not a dict |
| 74 | Raises TypeError if any value not an ArrayRequest |

#### `create_host_array`

| # | Functionality |
|---|--------------|
| 75 | Calls _ensure_cuda_context |
| 76 | Raises ValueError if memory_type not "pinned" or "host" |
| 77 | Creates pinned_array for "pinned" |
| 78 | Creates np_empty for "host" |
| 79 | Copies from `like` when provided |
| 80 | Fills with zeros when `like` is None |

#### `get_available_memory`

| # | Functionality |
|---|--------------|
| 81 | Returns free memory in passive mode |
| 82 | Returns min(headroom, free) in active mode |
| 83 | Warns when headroom/cap < 0.05 in active mode |

#### `get_memory_info`

| # | Functionality |
|---|--------------|
| 84 | Returns `current_mem_info()` |

#### `get_stream_group`

| # | Functionality |
|---|--------------|
| 85 | Forwards to `stream_groups.get_group(instance)` |

#### `is_grouped`

| # | Functionality |
|---|--------------|
| 86 | Returns False when group is "default" |
| 87 | Returns False when only 1 peer in group |
| 88 | Returns True when multiple peers in named group |

#### `allocate_all`

| # | Functionality |
|---|--------------|
| 89 | Allocates each request and registers with instance settings |
| 90 | Returns dict of label->array |

#### `allocate`

| # | Functionality |
|---|--------------|
| 91 | Calls _ensure_cuda_context |
| 92 | Uses cupy stream context when allocator is CuPyAsyncNumbaManager |
| 93 | Returns device_array for "device" |
| 94 | Returns mapped_array for "mapped" |
| 95 | Returns pinned_array for "pinned" |
| 96 | Raises NotImplementedError for "managed" |
| 97 | Raises ValueError for unknown memory_type |

#### `queue_request`

| # | Functionality |
|---|--------------|
| 98 | Checks requests via _check_requests |
| 99 | Creates new stream_group entry when not present |
| 100 | Stores instance requests keyed by stream_group and instance_id |

#### `to_device`

| # | Functionality |
|---|--------------|
| 101 | Calls _ensure_cuda_context |
| 102 | Uses cupy stream context when allocator is CuPyAsync |
| 103 | Copies each from_array to to_array via cuda.to_device |

#### `from_device`

| # | Functionality |
|---|--------------|
| 104 | Calls _ensure_cuda_context |
| 105 | Uses cupy stream context when allocator is CuPyAsync |
| 106 | Calls copy_to_host for each from_array |

#### `sync_stream`

| # | Functionality |
|---|--------------|
| 107 | Calls _ensure_cuda_context then stream.synchronize() |

#### `allocate_queue`

| # | Functionality |
|---|--------------|
| 108 | Pops queued requests for triggering instance's stream group |
| 109 | Extracts num_runs from first request with > 1 runs |
| 110 | Computes chunk parameters via get_chunk_parameters |
| 111 | Computes chunked shapes and allocates for each queued instance |
| 112 | Calls allocation_ready_hook for each instance with ArrayResponse |
| 113 | Notifies non-requesting peers in same stream group with empty ArrayResponse |

#### `get_chunk_parameters`

| # | Functionality |
|---|--------------|
| 114 | Returns (axis_length, 1) when request fits in memory |
| 115 | Warns when request exceeds VRAM by 20x |
| 116 | Raises ValueError when all arrays unchunkable but exceed memory |
| 117 | Raises ValueError when unchunkable alone >= available memory |
| 118 | Raises ValueError when max_chunk_size == 0 |
| 119 | Computes max_chunk_size and num_chunks from chunkable/unchunkable ratio |

#### `compute_chunked_shapes`

| # | Functionality |
|---|--------------|
| 120 | Replaces run axis with chunk_size for chunkable requests |
| 121 | Retains original shape for unchunkable requests |

---

## `memory/stream_groups.py`

### `StreamGroups`

#### `__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 1 | Initializes default group when groups is None |
| 2 | Initializes default stream when streams is None |

#### `add_instance`

| # | Functionality |
|---|--------------|
| 3 | Accepts int or object (uses id()) for instance_id |
| 4 | Raises ValueError if instance already in any group |
| 5 | Creates new group with new stream when group doesn't exist |
| 6 | Appends instance_id to group |

#### `get_group`

| # | Functionality |
|---|--------------|
| 7 | Returns group name containing instance |
| 8 | Raises ValueError if instance not in any group |
| 9 | Handles both int and object instances |

#### `get_stream`

| # | Functionality |
|---|--------------|
| 10 | Returns stream for instance's group via get_group |

#### `get_instances_in_group`

| # | Functionality |
|---|--------------|
| 11 | Returns list of instance ids for existing group |
| 12 | Returns empty list for non-existent group |

#### `change_group`

| # | Functionality |
|---|--------------|
| 13 | Removes instance from current group |
| 14 | Creates new group with stream if target doesn't exist |
| 15 | Appends instance to new group |
| 16 | Handles both int and object instances |

#### `reinit_streams`

| # | Functionality |
|---|--------------|
| 17 | Replaces all streams with new `cuda.stream()` |

---

## `memory/cupy_emm.py`

### Module-Level Functions

#### `_numba_stream_ptr`

| # | Functionality |
|---|--------------|
| 1 | Returns None when nb_stream is None |
| 2 | Returns None when handle attribute is None |
| 3 | Extracts int from ctypes.c_void_p handle |
| 4 | Returns None when c_void_p.value is None |
| 5 | Falls back to `int(getattr(h, "value", h))` |
| 6 | Returns None on extraction exception |

### `current_cupy_stream`

#### `__init__`

| # | Functionality |
|---|--------------|
| 7 | Raises ImportError when cupy not installed |
| 8 | Stores nb_stream and sets cupy_ext_stream to None |
| 9 | Detects if memory manager is cupy via `is_cupy` attribute |
| 10 | Sets `_mgr_is_cupy = False` on AttributeError |

#### `__enter__`

| # | Functionality |
|---|--------------|
| 11 | Raises ImportError when cupy not installed |
| 12 | Creates and enters CuPy ExternalStream when mgr is cupy and ptr valid |
| 13 | Returns self without creating stream when mgr not cupy |
| 14 | Returns self without creating stream when ptr is falsy |

#### `__exit__`

| # | Functionality |
|---|--------------|
| 15 | Exits and clears cupy_ext_stream when mgr is cupy and stream exists |
| 16 | No-op when mgr not cupy |
| 17 | No-op when cupy_ext_stream is None |

### `CuPyNumbaManager`

#### `__init__`

| # | Functionality |
|---|--------------|
| 18 | Raises ImportError when cupy not installed |
| 19 | Initializes _allocations dict, _mp=None, is_cupy=True |

#### `memalloc`

| # | Functionality |
|---|--------------|
| 20 | Allocates from CuPy pool via `_mp.malloc(nbytes)` |
| 21 | Records allocation in _allocations dict |
| 22 | Returns MemoryPointer wrapping CuPy allocation with finalizer |

#### `_make_finalizer`

| # | Functionality |
|---|--------------|
| 23 | Returns closure that removes allocation from _allocations dict |

#### `get_memory_info`

| # | Functionality |
|---|--------------|
| 24 | Returns MemoryInfo from pool's free_bytes/total_bytes |

#### `reset`

| # | Functionality |
|---|--------------|
| 25 | Calls super().reset() |
| 26 | Calls _mp.free_all_blocks(stream) when _mp exists |
| 27 | No-op on _mp when _mp is None |

#### `defer_cleanup`

| # | Functionality |
|---|--------------|
| 28 | Context manager wrapping super().defer_cleanup() |

#### `interface_version` property

| # | Functionality |
|---|--------------|
| 29 | Returns 1 |

### `CuPyAsyncNumbaManager`

#### `initialize`

| # | Functionality |
|---|--------------|
| 30 | Sets _mp to CuPy MemoryAsyncPool |

#### `memalloc`

| # | Functionality |
|---|--------------|
| 31 | Sets _testout="async" when _testing is True |
| 32 | Delegates to super().memalloc |

### `CuPySyncNumbaManager`

#### `initialize`

| # | Functionality |
|---|--------------|
| 33 | Sets _mp to CuPy default memory pool |

#### `memalloc`

| # | Functionality |
|---|--------------|
| 34 | Sets _testout="sync" when _testing is True |
| 35 | Delegates to super().memalloc |

---

## `memory/chunk_buffer_pool.py`

### `PinnedBuffer`

| # | Functionality |
|---|--------------|
| 1 | Attrs dataclass with buffer_id (int), array (ndarray), in_use (bool, default False) |

### `ChunkBufferPool`

#### `acquire`

| # | Functionality |
|---|--------------|
| 2 | Returns existing matching buffer (shape+dtype) when available and not in_use |
| 3 | Marks returned buffer as in_use |
| 4 | Allocates new buffer when no suitable one found |
| 5 | Creates new array_name entry when first buffer for that name |
| 6 | Thread-safe via _lock |

#### `release`

| # | Functionality |
|---|--------------|
| 7 | Sets buffer.in_use = False under lock |

#### `clear`

| # | Functionality |
|---|--------------|
| 8 | Clears _buffers dict and resets _next_id to 0 under lock |

#### `_allocate_buffer`

| # | Functionality |
|---|--------------|
| 9 | Uses np_zeros in CUDA_SIMULATION mode |
| 10 | Uses cuda.pinned_array in non-simulation mode |
| 11 | Assigns incrementing buffer_id |

---

## `memory/array_requests.py`

### `ArrayRequest`

| # | Functionality |
|---|--------------|
| 1 | Validates dtype in [float64, float32, int32] |
| 2 | Default shape (1,1,1), memory "device", chunk_axis_index 2, unchunkable False, total_runs 1 |
| 3 | deep_iterable validator on shape |
| 4 | memory validated against ["device", "mapped", "pinned", "managed"] |

### `ArrayResponse`

| # | Functionality |
|---|--------------|
| 5 | Default arr={}, chunks=1, chunk_length=1, chunked_shapes={} |

---

## `batchsolving/_utils.py`

| # | Functionality |
|---|--------------|
| 1 | Empty module — no public exports remain |

---

## `batchsolving/SystemInterface.py`

### `SystemInterface`

#### `__init__`

| # | Functionality |
|---|--------------|
| 1 | Stores parameters, states, observables as SystemValues |

#### `from_system` classmethod

| # | Functionality |
|---|--------------|
| 2 | Creates interface from system.parameters, system.initial_values, system.observables |

#### `update`

| # | Functionality |
|---|--------------|
| 3 | Returns None when updates is None and no kwargs |
| 4 | Merges kwargs into updates dict |
| 5 | Attempts update on both parameters and states |
| 6 | Raises KeyError when unrecognized keys and silent=False |
| 7 | Returns recognized keys set when silent=True |

#### `state_indices`

| # | Functionality |
|---|--------------|
| 8 | Returns all state indices when keys_or_indices is None |
| 9 | Delegates to states.get_indices otherwise |

#### `observable_indices`

| # | Functionality |
|---|--------------|
| 10 | Returns all observable indices when keys_or_indices is None |
| 11 | Delegates to observables.get_indices otherwise |

#### `parameter_indices`

| # | Functionality |
|---|--------------|
| 12 | Delegates to parameters.get_indices |

#### `get_labels`

| # | Functionality |
|---|--------------|
| 13 | Delegates to values_object.get_labels(indices) |

#### `state_labels`

| # | Functionality |
|---|--------------|
| 14 | Returns all state names when indices is None |
| 15 | Delegates to get_labels(states, indices) otherwise |

#### `observable_labels`

| # | Functionality |
|---|--------------|
| 16 | Returns all observable names when indices is None |
| 17 | Delegates to get_labels(observables, indices) otherwise |

#### `parameter_labels`

| # | Functionality |
|---|--------------|
| 18 | Returns all parameter names when indices is None |
| 19 | Delegates to get_labels(parameters, indices) otherwise |

#### `resolve_variable_labels`

| # | Functionality |
|---|--------------|
| 20 | Returns (None, None) when labels is None |
| 21 | Returns (empty, empty) int32 arrays when labels is empty list |
| 22 | Resolves labels to state and observable indices |
| 23 | Raises ValueError for unresolved labels when silent=False |
| 24 | Returns (state_idxs, obs_idxs) as int32 |

#### `merge_variable_inputs`

| # | Functionality |
|---|--------------|
| 25 | Returns full range (all states, all obs) when all three inputs None |
| 26 | Replaces None inputs with empty arrays |
| 27 | Computes union of label-resolved and directly-provided indices |

#### `merge_variable_labels_and_idxs`

| # | Functionality |
|---|--------------|
| 28 | Pops save_variables and summarise_variables from dict |
| 29 | Calls merge_variable_inputs for save variables |
| 30 | Defaults summarise indices to saved indices when all summarise inputs None |
| 31 | Calls merge_variable_inputs for summarise when not all None |
| 32 | Updates dict in-place with final index arrays |

#### Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 33 | `all_input_labels` | `state_labels() + parameter_labels()` |
| 34 | `all_output_labels` | `state_labels() + observable_labels()` |

---

## `batchsolving/BatchInputHandler.py`

### Module-Level Functions

#### `unique_cartesian_product`

| # | Functionality |
|---|--------------|
| 1 | Deduplicates each input array preserving order |
| 2 | Returns Cartesian product in (variable, run) format |

#### `combinatorial_grid`

| # | Functionality |
|---|--------------|
| 3 | Filters empty values from request |
| 4 | Resolves indices via values_instance |
| 5 | Returns (indices, combos) from unique_cartesian_product |

#### `verbatim_grid`

| # | Functionality |
|---|--------------|
| 6 | Filters empty values from request |
| 7 | Resolves indices via values_instance |
| 8 | Stacks values as rows (variable, run) format without expansion |

#### `generate_grid`

| # | Functionality |
|---|--------------|
| 9 | Dispatches to combinatorial_grid for kind="combinatorial" |
| 10 | Dispatches to verbatim_grid for kind="verbatim" |
| 11 | Raises ValueError for unknown kind |

#### `combine_grids`

| # | Functionality |
|---|--------------|
| 12 | Combinatorial: repeats grid1 columns and tiles grid2 columns |
| 13 | Verbatim: broadcasts single-run grid1 to match grid2 |
| 14 | Verbatim: broadcasts single-run grid2 to match grid1 |
| 15 | Verbatim: raises ValueError when run counts differ after broadcast |
| 16 | Raises ValueError for unknown kind |

#### `extend_grid_to_array`

| # | Functionality |
|---|--------------|
| 17 | Returns tiled defaults when indices empty |
| 18 | Returns single-column defaults when grid is 1D |
| 19 | Raises ValueError when grid rows != indices length |
| 20 | Returns grid directly when all indices swept |
| 21 | Creates default array and overwrites swept indices |

### `BatchInputHandler`

#### `__init__`

| # | Functionality |
|---|--------------|
| 22 | Stores parameters, states, precision from interface |

#### `from_system` classmethod

| # | Functionality |
|---|--------------|
| 23 | Creates handler via SystemInterface.from_system |

#### `__call__`

| # | Functionality |
|---|--------------|
| 24 | Updates precision from current system state |
| 25 | Attempts _fast_return_arrays first |
| 26 | Falls through to _process_single_input for states and params |
| 27 | Aligns run counts via _align_run_counts |
| 28 | Casts to precision |

#### `_trim_or_extend`

| # | Functionality |
|---|--------------|
| 29 | Extends with default values when arr has fewer rows |
| 30 | Trims extra rows when arr has more rows |
| 31 | Returns unchanged when row count matches |

#### `_sanitise_arraylike`

| # | Functionality |
|---|--------------|
| 32 | Returns None passthrough when arr is None |
| 33 | Coerces non-ndarray to ndarray |
| 34 | Raises ValueError for >2D input |
| 35 | Converts 1D to single-column 2D |
| 36 | Warns and adjusts when row count mismatches values_object.n |
| 37 | Returns None when array is empty after processing |

#### `_process_single_input`

| # | Functionality |
|---|--------------|
| 38 | Returns empty (0,1) array when values_object is empty and input is None |
| 39 | Raises ValueError when values_object empty but non-empty input provided |
| 40 | Returns single-column defaults when input is None |
| 41 | Processes dict: wraps scalars, generates grid, extends with defaults |
| 42 | Processes array-like: sanitises to 2D |
| 43 | Returns defaults when sanitised result is None |
| 44 | Raises TypeError for unsupported input type |

#### `_align_run_counts`

| # | Functionality |
|---|--------------|
| 45 | Delegates to combine_grids |

#### `_cast_to_precision`

| # | Functionality |
|---|--------------|
| 46 | Returns C-contiguous arrays cast to self.precision |

#### `_is_right_sized_array`

| # | Functionality |
|---|--------------|
| 47 | Returns True for None when values_object empty |
| 48 | Returns True for 2D empty array when values_object empty |
| 49 | Returns False for non-ndarray |
| 50 | Returns False for non-2D ndarray |
| 51 | Returns True when shape[0] == values_object.n |

#### `_is_1d_or_none`

| # | Functionality |
|---|--------------|
| 52 | Returns True for None |
| 53 | Returns False for dict |
| 54 | Returns True for 1D ndarray |
| 55 | Returns True for flat list/tuple (no nested iterables) |
| 56 | Returns False otherwise |

#### `_to_defaults_column`

| # | Functionality |
|---|--------------|
| 57 | Returns tiled defaults with n_runs columns |

#### `_fast_return_arrays`

| # | Functionality |
|---|--------------|
| 58 | Returns (states, params) immediately when both are device arrays with matching runs |
| 59 | Returns None when device arrays have mismatched runs |
| 60 | Returns cast arrays when both are right-sized with matching runs |
| 61 | Handles empty parameters (creates empty array) |
| 62 | Fast path: states_ok + params_small → broadcast params to match |
| 63 | Fast path: params_ok + states_small → broadcast states to match |
| 64 | Returns None when no fast path applies |

#### `_get_run_count`

| # | Functionality |
|---|--------------|
| 65 | Returns shape[1] for 2D ndarray |
| 66 | Returns None for non-2D ndarray |
| 67 | Extracts shape[1] from __cuda_array_interface__ |
| 68 | Returns None otherwise |

---

## `batchsolving/BatchSolverConfig.py`

### `ActiveOutputs`

| # | Functionality |
|---|--------------|
| 1 | Six boolean fields: state, observables, state_summaries, observable_summaries, status_codes, iteration_counters |
| 2 | All default to False |

#### `from_compile_flags` classmethod

| # | Functionality |
|---|--------------|
| 3 | Maps save_state → state, save_observables → observables |
| 4 | Maps summarise_state → state_summaries, summarise_observables → observable_summaries |
| 5 | Always sets status_codes = True |
| 6 | Maps save_counters → iteration_counters |

### `BatchSolverConfig`

| # | Functionality |
|---|--------------|
| 7 | Extends CUDAFactoryConfig with loop_fn and compile_flags |
| 8 | loop_fn default None, compile_flags default OutputCompileFlags() |

#### `active_outputs` property

| # | Functionality |
|---|--------------|
| 9 | Derives ActiveOutputs from compile_flags via from_compile_flags |

---

## `batchsolving/BatchSolverKernel.py`

### `RunParams`

#### Construction (frozen attrs)

| # | Functionality |
|---|--------------|
| 1 | duration, warmup, t0 validated >= 0.0 |
| 2 | runs validated >= 1 |
| 3 | num_chunks default 1, chunk_length default 0 |

#### `__getitem__`

| # | Functionality |
|---|--------------|
| 4 | Raises IndexError for out-of-range chunk index |
| 5 | Last chunk: calculates remaining runs as runs - (num_chunks-1)*chunk_length |
| 6 | Non-last chunk: returns chunk_length runs |
| 7 | Returns evolved RunParams with updated runs |

#### `update_from_allocation`

| # | Functionality |
|---|--------------|
| 8 | Returns evolved RunParams with num_chunks and chunk_length from response |

### `BatchSolverCache`

| # | Functionality |
|---|--------------|
| 9 | Extends CUDADispatcherCache with solver_kernel field (default -1) |

### `BatchSolverKernel`

#### `__init__`

| # | Functionality |
|---|--------------|
| 10 | Defaults None dicts to empty dicts |
| 11 | Creates RunParams with zeros and 1 run |
| 12 | Sets up memory manager via _setup_memory_manager |
| 13 | Creates SingleIntegratorRun |
| 14 | Creates system_name with unnamed fallback when name==hash |
| 15 | Creates CubieCacheHandler |
| 16 | Creates BatchSolverConfig with initial precision and compile flags |
| 17 | Creates InputArrays and OutputArrays from solver |

#### `_setup_memory_manager`

| # | Functionality |
|---|--------------|
| 18 | Merges user settings with DEFAULT_MEMORY_SETTINGS |
| 19 | Registers self with memory manager using allocation_ready_hook |

#### `_setup_cuda_events`

| # | Functionality |
|---|--------------|
| 20 | Creates 1 GPU workload event + 3 events per chunk |

#### `_get_chunk_events`

| # | Functionality |
|---|--------------|
| 21 | Returns (h2d, kernel, d2h) events for chunk by index |

#### `_validate_timing_parameters`

| # | Functionality |
|---|--------------|
| 22 | Raises ValueError when save_every > duration and save_last is False |
| 23 | Raises ValueError when sample_summaries_every is None with summary outputs |
| 24 | Raises ValueError when summarise_every is None with summary outputs |
| 25 | Raises ValueError when sample_summaries_every >= summarise_every |
| 26 | Raises ValueError when summarise_every > duration |

#### `run`

| # | Functionality |
|---|--------------|
| 27 | Uses solver's stream when stream is None |
| 28 | Casts duration to float64 |
| 29 | Creates RunParams from actual run values |
| 30 | Calls set_summary_timing_from_duration |
| 31 | Validates timing parameters |
| 32 | Updates compile settings with loop_fn and precision |
| 33 | Updates input and output arrays |
| 34 | Processes allocation queue |
| 35 | Calculates dynamic shared memory with optional padding |
| 36 | Calls limit_blocksize |
| 37 | Ensures minimum 4 bytes dynamic shared memory |
| 38 | Iterates chunks: h2d transfer, kernel launch, d2h transfer with events |

#### `limit_blocksize`

| # | Functionality |
|---|--------------|
| 39 | Halves blocksize while dynamic_sharedmem >= 32768 |
| 40 | Warns when blocksize reduced below 32 |
| 41 | Returns adjusted (blocksize, dynamic_sharedmem) |

#### `build_kernel`

| # | Functionality |
|---|--------------|
| 42 | Extracts precision, output flags, shared memory params |
| 43 | Gets buffer allocators from buffer_registry |
| 44 | JIT-compiles integration_kernel with @cuda.jit |
| 45 | Attaches cache via cache_handler |

#### `update`

| # | Functionality |
|---|--------------|
| 46 | Returns empty set for empty updates |
| 47 | Flattens nested dicts via unpack_dict_values |
| 48 | Forwards to single_integrator.update |
| 49 | Forwards to buffer_registry.update |
| 50 | Injects loop_fn and compile_flags into updates |
| 51 | Forwards to update_compile_settings |
| 52 | Forwards to cache_handler.update |
| 53 | Raises KeyError for unrecognized params when silent=False |
| 54 | Returns recognised keys including unpacked dict keys |

#### `wait_for_writeback`

| # | Functionality |
|---|--------------|
| 55 | Delegates to output_arrays.wait_pending() |

#### Forwarding Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 56 | `local_memory_elements` | `single_integrator.local_memory_elements` |
| 57 | `shared_memory_elements` | `single_integrator.shared_memory_elements` |
| 58 | `compile_flags` | `compile_settings.compile_flags` |
| 59 | `active_outputs` | `compile_settings.active_outputs` |
| 60 | `cache_config` | `cache_handler.config` |
| 61 | `memory_manager` | `_memory_manager` |
| 62 | `stream_group` | `memory_manager.get_stream_group(self)` |
| 63 | `stream` | `memory_manager.get_stream(self)` |
| 64 | `mem_proportion` | `memory_manager.proportion(self)` |
| 65 | `shared_memory_bytes` | `single_integrator.shared_memory_bytes` |
| 66 | `threads_per_loop` | `single_integrator.threads_per_step` |
| 67 | `duration` | `run_params.duration` (cast to float64) |
| 68 | `warmup` | `run_params.warmup` (cast to float64) |
| 69 | `t0` | `run_params.t0` (cast to float64) |
| 70 | `num_runs` | `run_params.runs` |
| 71 | `chunks` | `run_params.num_chunks` |
| 72 | `output_length` | `single_integrator.output_length(duration)` |
| 73 | `summaries_length` | `single_integrator.summaries_length(duration)` |
| 74 | `system` | `single_integrator.system` |
| 75 | `algorithm` | `single_integrator.algorithm` |
| 76 | `dt_min` | `single_integrator.dt_min` |
| 77 | `dt_max` | `single_integrator.dt_max` |
| 78 | `atol` | `single_integrator.atol` |
| 79 | `rtol` | `single_integrator.rtol` |
| 80 | `save_every` | `single_integrator.save_every` |
| 81 | `summarise_every` | `single_integrator.summarise_every` |
| 82 | `sample_summaries_every` | `single_integrator.sample_summaries_every` |
| 83 | `system_sizes` | `single_integrator.system_sizes` |
| 84 | `output_array_heights` | `single_integrator.output_array_heights` |
| 85 | `summary_legend_per_variable` | `single_integrator.summary_legend_per_variable` |
| 86 | `summary_unit_modifications` | `single_integrator.summary_unit_modifications` |
| 87 | `saved_state_indices` | `single_integrator.saved_state_indices` |
| 88 | `saved_observable_indices` | `single_integrator.saved_observable_indices` |
| 89 | `summarised_state_indices` | `single_integrator.summarised_state_indices` |
| 90 | `summarised_observable_indices` | `single_integrator.summarised_observable_indices` |
| 91 | `state` | `output_arrays.state` |
| 92 | `observables` | `output_arrays.observables` |
| 93 | `state_summaries` | `output_arrays.state_summaries` |
| 94 | `status_codes` | `output_arrays.status_codes` |
| 95 | `observable_summaries` | `output_arrays.observable_summaries` |
| 96 | `iteration_counters` | `output_arrays.iteration_counters` |
| 97 | `initial_values` | `input_arrays.initial_values` |
| 98 | `parameters` | `input_arrays.parameters` |
| 99 | `driver_coefficients` | `input_arrays.driver_coefficients` |
| 100 | `device_driver_coefficients` | `input_arrays.device_driver_coefficients` |
| 101 | `save_time` | `single_integrator.save_time` |
| 102 | `output_types` | `single_integrator.output_types` |
| 103 | `output_heights` | `single_integrator.output_array_heights` |
| 104 | `dt` | `single_integrator.dt or None` |
| 105 | `kernel` | `device_function` (via get_cached_output) |

#### Setters

| # | Functionality |
|---|--------------|
| 106 | `duration.setter` evolves run_params with new float64 value |
| 107 | `warmup.setter` evolves run_params with new float64 value |
| 108 | `t0.setter` evolves run_params with new float64 value |
| 109 | `num_runs.setter` evolves run_params with new runs value |

#### `shared_memory_needs_padding` property

| # | Functionality |
|---|--------------|
| 110 | Returns False for float64 precision |
| 111 | Returns False when shared_memory_elements == 0 |
| 112 | Returns True for float32 with even element count |
| 113 | Returns False for float32 with odd element count |

#### `_on_allocation`

| # | Functionality |
|---|--------------|
| 114 | Updates run_params from allocation response |

#### `_invalidate_cache`

| # | Functionality |
|---|--------------|
| 115 | Calls super()._invalidate_cache() then cache_handler.invalidate() |

#### `build`

| # | Functionality |
|---|--------------|
| 116 | Returns BatchSolverCache wrapping build_kernel() result |

#### `profileCUDA` property

| # | Functionality |
|---|--------------|
| 117 | Returns _profileCUDA and not is_cudasim_enabled() |

#### `set_cache_dir`

| # | Functionality |
|---|--------------|
| 118 | Delegates to cache_handler.update with Path |

#### `enable_profiling` / `disable_profiling`

| # | Functionality |
|---|--------------|
| 119 | Sets _profileCUDA = True |
| 120 | Sets _profileCUDA = False |

---

## `batchsolving/solveresult.py`

### Module-Level Functions

#### `_format_time_domain_label`

| # | Functionality |
|---|--------------|
| 1 | Returns "label [unit]" when unit is not "dimensionless" |
| 2 | Returns just label when unit is "dimensionless" |

### `SolveSpec`

| # | Functionality |
|---|--------------|
| 3 | Frozen attrs with validated dt, dt_min, dt_max, save_every, summarise_every, sample_summaries_every, atol, rtol, duration, warmup, t0, algorithm, saved_states, saved_observables, summarised_states, summarised_observables, output_types, precision |

### `SolveResult`

#### Construction

| # | Functionality |
|---|--------------|
| 4 | Defaults: empty arrays, None status_codes, empty dicts for legends, default stride_order ("time","variable","run") |

#### `from_solver` classmethod

| # | Functionality |
|---|--------------|
| 5 | Returns raw dict of arrays when results_type="raw" |
| 6 | Cleaves time from state array |
| 7 | Combines time domain arrays (state + observables) |
| 8 | Combines summaries arrays |
| 9 | NaN-fills error trajectories when nan_error_trajectories=True and status codes nonzero |
| 10 | Handles run_index at position 0, 1, or 2 in stride order for NaN fill |
| 11 | Generates time_domain_legend and summaries_legend |
| 12 | Returns SolveResult for results_type="full" |
| 13 | Returns as_numpy for results_type="numpy" |
| 14 | Returns as_numpy_per_summary for results_type="numpy_per_summary" |
| 15 | Returns as_pandas for results_type="pandas" |
| 16 | Returns SolveResult for unknown results_type |

#### `as_pandas` property

| # | Functionality |
|---|--------------|
| 17 | Raises ImportError when pandas not available |
| 18 | Creates per-run DataFrames with MultiIndex columns |
| 19 | Uses time as index when available (handles 1D and 2D time) |
| 20 | Concatenates summary DataFrames when summaries active |
| 21 | Returns dict with "time_domain" and "summaries" keys |

#### `as_numpy` property

| # | Functionality |
|---|--------------|
| 22 | Returns dict with copies of time, arrays, legends, and counters |
| 23 | Returns None for time when self.time is None |

#### `as_numpy_per_summary` property

| # | Functionality |
|---|--------------|
| 24 | Returns base arrays plus per_summary_arrays |

#### `per_summary_arrays` property

| # | Functionality |
|---|--------------|
| 25 | Returns empty dict when no summaries active |
| 26 | Splits summaries_array by type using singlevar_legend |
| 27 | Includes "summary_legend" key |

#### `active_outputs` property

| # | Functionality |
|---|--------------|
| 28 | Returns _active_outputs |

#### `cleave_time` static method

| # | Functionality |
|---|--------------|
| 29 | Returns (None, state) when time_saved is False |
| 30 | Uses default stride_order when None provided |
| 31 | Extracts time as last variable slice and returns state without it |

#### `combine_time_domain_arrays` static method

| # | Functionality |
|---|--------------|
| 32 | Concatenates state and observables when both active |
| 33 | Returns state.copy() when only state active |
| 34 | Returns observables.copy() when only observables active |
| 35 | Returns empty array when neither active |

#### `combine_summaries_array` static method

| # | Functionality |
|---|--------------|
| 36 | Concatenates when both active |
| 37 | Returns state_summaries.copy() when only states |
| 38 | Returns observable_summaries.copy() when only observables |
| 39 | Returns empty array when neither active |

#### `summary_legend_from_solver` static method

| # | Functionality |
|---|--------------|
| 40 | Generates legend entries for state summaries with units |
| 41 | Generates legend entries for observable summaries with units |
| 42 | Uses "dimensionless" fallback when no units |
| 43 | Applies unit_modifications replacing "unit" placeholder |

#### `time_domain_legend_from_solver` static method

| # | Functionality |
|---|--------------|
| 44 | Generates legend for state labels with units via _format_time_domain_label |
| 45 | Generates legend for observable labels with offset |

---

## `batchsolving/solver.py`

### `solve_ivp` function

| # | Functionality |
|---|--------------|
| 1 | Pops loop_settings from kwargs |
| 2 | Sets save_variables/summarise_variables defaults in kwargs |
| 3 | Creates Solver with method, loop_settings, and kwargs |
| 4 | Starts/stops solve_ivp timing event |
| 5 | Calls solver.solve with all parameters |

### `Solver`

#### `__init__`

| # | Functionality |
|---|--------------|
| 6 | Defaults all None settings dicts to empty dicts |
| 7 | Sets global time logging verbosity |
| 8 | Creates SystemInterface from system |
| 9 | Creates ArrayInterpolator with placeholder |
| 10 | Creates BatchInputHandler from interface |
| 11 | Merges kwargs into output, memory, step, algorithm, loop, cache settings |
| 12 | Converts output labels via convert_output_labels |
| 13 | Sets algorithm in algorithm_settings |
| 14 | Creates BatchSolverKernel |
| 15 | Raises KeyError in strict mode for unrecognized kwargs |

#### `convert_output_labels`

| # | Functionality |
|---|--------------|
| 16 | Delegates to system_interface.merge_variable_labels_and_idxs |

#### `solve`

| # | Functionality |
|---|--------------|
| 17 | Updates settings from kwargs when provided |
| 18 | Builds input grids via input_handler |
| 19 | Checks drivers against system when provided |
| 20 | Updates driver interpolator and pushes evaluate_driver_at_t when fn changed |
| 21 | Calls kernel.run with all parameters |
| 22 | Synchronizes stream and waits for writeback |
| 23 | Returns SolveResult.from_solver |
| 24 | Starts/stops solver_solve timing events |

#### `build_grid`

| # | Functionality |
|---|--------------|
| 25 | Delegates to input_handler(states, params, kind) |

#### `update`

| # | Functionality |
|---|--------------|
| 26 | Returns empty set for empty updates |
| 27 | Converts output labels when variable keys present |
| 28 | Forwards to driver_interpolator.update |
| 29 | Injects evaluate_driver_at_t/driver_del_t when driver recognised |
| 30 | Forwards to update_memory_settings |
| 31 | Forwards to system_interface.update |
| 32 | Forwards to kernel.update |
| 33 | Handles profileCUDA enable/disable |
| 34 | Raises KeyError for unrecognized params when silent=False |

#### `update_memory_settings`

| # | Functionality |
|---|--------------|
| 35 | Returns empty set for empty updates |
| 36 | Calls set_auto_limit_mode when mem_proportion is None |
| 37 | Calls set_manual_proportion when mem_proportion has value |
| 38 | Calls set_allocator for "allocator" key |
| 39 | Raises KeyError for unrecognized params when silent=False |

#### `enable_profiling` / `disable_profiling`

| # | Functionality |
|---|--------------|
| 40 | Delegates to kernel.enable_profiling() |
| 41 | Delegates to kernel.disable_profiling() |

#### `get_state_indices` / `get_observable_indices`

| # | Functionality |
|---|--------------|
| 42 | Delegates to system_interface.state_indices |
| 43 | Delegates to system_interface.observable_indices |

#### Forwarding Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 44 | `precision` | `kernel.precision` |
| 45 | `compile_flags` | `kernel.compile_flags` |
| 46 | `active_outputs` | `kernel.active_outputs` |
| 47 | `system_sizes` | `kernel.system_sizes` |
| 48 | `output_array_heights` | `kernel.output_array_heights` |
| 49 | `num_runs` | `kernel.num_runs` |
| 50 | `output_length` | `kernel.output_length` |
| 51 | `summaries_length` | `kernel.summaries_length` |
| 52 | `summary_legend_per_variable` | `kernel.summary_legend_per_variable` |
| 53 | `summary_unit_modifications` | `kernel.summary_unit_modifications` |
| 54 | `saved_state_indices` | `kernel.saved_state_indices` |
| 55 | `saved_observable_indices` | `kernel.saved_observable_indices` |
| 56 | `summarised_state_indices` | `kernel.summarised_state_indices` |
| 57 | `summarised_observable_indices` | `kernel.summarised_observable_indices` |
| 58 | `state` | `kernel.state` |
| 59 | `observables` | `kernel.observables` |
| 60 | `state_summaries` | `kernel.state_summaries` |
| 61 | `observable_summaries` | `kernel.observable_summaries` |
| 62 | `iteration_counters` | `kernel.iteration_counters` |
| 63 | `status_codes` | `kernel.status_codes` |
| 64 | `parameters` | `kernel.parameters` |
| 65 | `initial_values` | `kernel.initial_values` |
| 66 | `driver_coefficients` | `kernel.driver_coefficients` |
| 67 | `save_time` | `kernel.save_time` |
| 68 | `output_types` | `kernel.output_types` |
| 69 | `chunks` | `kernel.chunks` |
| 70 | `memory_manager` | `kernel.memory_manager` |
| 71 | `stream_group` | `kernel.stream_group` |
| 72 | `stream` | `kernel.stream` |
| 73 | `mem_proportion` | `kernel.mem_proportion` |
| 74 | `system` | `kernel.system` |
| 75 | `dt` | `kernel.dt` |
| 76 | `dt_min` | `kernel.dt_min` |
| 77 | `dt_max` | `kernel.dt_max` |
| 78 | `save_every` | `kernel.save_every` |
| 79 | `summarise_every` | `kernel.summarise_every` |
| 80 | `sample_summaries_every` | `kernel.sample_summaries_every` |
| 81 | `duration` | `kernel.duration` |
| 82 | `warmup` | `kernel.warmup` |
| 83 | `t0` | `kernel.t0` |
| 84 | `atol` | `kernel.atol` |
| 85 | `rtol` | `kernel.rtol` |
| 86 | `algorithm` | `kernel.algorithm` |
| 87 | `input_variables` | `system_interface.all_input_labels` |
| 88 | `output_variables` | `system_interface.all_output_labels` |
| 89 | `cache_enabled` | `kernel.cache_config.cache_enabled` |
| 90 | `cache_mode` | `kernel.cache_config.cache_mode` |
| 91 | `cache_dir` | `kernel.cache_config.cache_dir` |

#### Computed Properties

| # | Functionality |
|---|--------------|
| 92 | `saved_states` resolves state labels from saved_state_indices |
| 93 | `saved_observables` resolves observable labels from saved_observable_indices |
| 94 | `summarised_states` resolves state labels from summarised_state_indices |
| 95 | `summarised_observables` resolves observable labels from summarised_observable_indices |

#### `set_cache_dir`

| # | Functionality |
|---|--------------|
| 96 | Delegates to kernel.set_cache_dir |

#### `set_verbosity`

| # | Functionality |
|---|--------------|
| 97 | Delegates to default_timelogger.set_verbosity |

#### `solve_info` property

| # | Functionality |
|---|--------------|
| 98 | Constructs SolveSpec from all current solver properties |

---

## `batchsolving/writeback_watcher.py`

### `PendingBuffer`

| # | Functionality |
|---|--------------|
| 1 | Attrs dataclass: buffer, target_array, array_name, data_shape, buffer_pool |

### `WritebackTask`

| # | Functionality |
|---|--------------|
| 2 | Attrs dataclass: event, buffer, target_array, buffer_pool, array_name, data_shape (default None) |

#### `from_pending_buffer` classmethod

| # | Functionality |
|---|--------------|
| 3 | Creates WritebackTask from PendingBuffer fields + event |

### `WritebackWatcher`

#### `__init__`

| # | Functionality |
|---|--------------|
| 4 | Initializes Queue, thread=None, stop_event, poll_interval, pending_count=0, lock |

#### `start`

| # | Functionality |
|---|--------------|
| 5 | No-op when thread already alive |
| 6 | Creates and starts daemon poll thread |

#### `_submit_task`

| # | Functionality |
|---|--------------|
| 7 | Increments _pending_count under lock |
| 8 | Puts task on queue |
| 9 | Starts thread if not running |

#### `submit_from_pending_buffer`

| # | Functionality |
|---|--------------|
| 10 | Creates WritebackTask from PendingBuffer and submits |

#### `submit`

| # | Functionality |
|---|--------------|
| 11 | Creates WritebackTask from individual args and submits |

#### `wait_all`

| # | Functionality |
|---|--------------|
| 12 | Returns immediately when pending_count == 0 |
| 13 | Polls until pending_count == 0 |
| 14 | Raises TimeoutError when timeout expires |

#### `shutdown`

| # | Functionality |
|---|--------------|
| 15 | Sets stop_event |
| 16 | Joins thread with 1s timeout |
| 17 | Sets _thread = None |

#### `_poll_loop`

| # | Functionality |
|---|--------------|
| 18 | Processes pending tasks, re-queues incomplete ones |
| 19 | Gets new tasks from queue (non-blocking) |
| 20 | On shutdown: drains queue and completes all remaining tasks |

#### `_process_task`

| # | Functionality |
|---|--------------|
| 21 | Treats as immediately complete in CUDA_SIMULATION or event is None |
| 22 | Queries event.query() for completion otherwise |
| 23 | Copies buffer data with data_shape slice when data_shape provided |
| 24 | Copies full buffer when data_shape is None |
| 25 | Releases buffer back to pool |
| 26 | Returns False when task not yet complete |

---

## `batchsolving/arrays/BaseArrayManager.py`

### `ManagedArray`

#### `__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 1 | Initializes _array as zeros with shape from default_shape or (1,)*len(stride_order) |
| 2 | Sets _chunk_axis_index from stride_order.index("run") when "run" present |

#### `shape` property

| # | Functionality |
|---|--------------|
| 3 | Returns _array.shape when array exists |
| 4 | Returns default_shape when array is None |

#### `needs_chunked_transfer` property

| # | Functionality |
|---|--------------|
| 5 | Returns False when chunked_shape is None |
| 6 | Returns True when shape != chunked_shape |

#### `chunk_slice`

| # | Functionality |
|---|--------------|
| 7 | Raises TypeError for non-int chunk_index |
| 8 | Returns full array when no chunking (axis None, is_chunked False, chunk_length None) |
| 9 | Last chunk: slice from start to end (None) |
| 10 | Non-last chunk: slice from start to start+chunk_length |
| 11 | Builds multi-dimensional slice tuple with slice(None) on non-chunk axes |

#### `array` property + setter

| # | Functionality |
|---|--------------|
| 12 | Returns _array |
| 13 | Setter stores value as _array |

### `ArrayContainer`

#### `_iter_field_items`

| # | Functionality |
|---|--------------|
| 14 | Yields (name, value) for all ManagedArray fields in __dict__ |

#### `iter_managed_arrays`

| # | Functionality |
|---|--------------|
| 15 | Delegates to _iter_field_items |

#### `array_names`

| # | Functionality |
|---|--------------|
| 16 | Returns list of labels from iter_managed_arrays |

#### `get_managed_array`

| # | Functionality |
|---|--------------|
| 17 | Returns ManagedArray for matching label |
| 18 | Raises AttributeError for non-existent label |

#### `get_array`

| # | Functionality |
|---|--------------|
| 19 | Returns stored array for label |

#### `set_array`

| # | Functionality |
|---|--------------|
| 20 | Attaches array reference to label |

#### `set_memory_type`

| # | Functionality |
|---|--------------|
| 21 | Sets memory_type on all managed arrays |

#### `memory_type` property

| # | Functionality |
|---|--------------|
| 22 | Returns memory_type of first managed array |
| 23 | Returns "No arrays managed" when empty |

#### `delete_all`

| # | Functionality |
|---|--------------|
| 24 | Sets array=None on all managed arrays |

#### `attach`

| # | Functionality |
|---|--------------|
| 25 | Calls set_array for label |
| 26 | Warns when label doesn't exist (catches AttributeError) |

### `BaseArrayManager`

#### `__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 27 | Registers with memory manager |
| 28 | Calls _invalidate_hook |

#### `is_chunked` property

| # | Functionality |
|---|--------------|
| 29 | Returns True when _chunks > 1 |

#### `set_array_runs`

| # | Functionality |
|---|--------------|
| 30 | Updates num_runs on self and all managed arrays |

#### `_iter_managed_arrays` property

| # | Functionality |
|---|--------------|
| 31 | Yields arrays from both device and host containers |

#### `_on_allocation_complete`

| # | Functionality |
|---|--------------|
| 32 | Attaches allocated arrays to device container |
| 33 | Stores chunked_shape, chunk_length, num_chunks on both host and device managed arrays |
| 34 | Warns when array label not found in response |
| 35 | Updates _chunks from response |
| 36 | Converts host to numpy for chunked mode |
| 37 | Converts host to pinned for non-chunked mode |
| 38 | Clears _needs_reallocation |

#### `register_with_memory_manager`

| # | Functionality |
|---|--------------|
| 39 | Registers self with hooks and stream_group |

#### `request_allocation`

| # | Functionality |
|---|--------------|
| 40 | Delegates to _memory_manager.queue_request |

#### `_invalidate_hook`

| # | Functionality |
|---|--------------|
| 41 | Clears reallocation and overwrite lists |
| 42 | Deletes all device arrays |
| 43 | Marks all device array names for reallocation |

#### `_arrays_equal`

| # | Functionality |
|---|--------------|
| 44 | Returns True when both None |
| 45 | Returns False when only one is None |
| 46 | Returns False when shapes differ |
| 47 | Returns False when dtypes differ and check_type=True |
| 48 | Returns True for shape_only=True when shapes match |
| 49 | Falls through to np_array_equal for content check |

#### `update_sizes`

| # | Functionality |
|---|--------------|
| 50 | Raises TypeError when new sizes type differs from existing |
| 51 | Updates _sizes |

#### `check_type`

| # | Functionality |
|---|--------------|
| 52 | Returns dict of bool indicating dtype match per array |

#### `check_sizes`

| # | Functionality |
|---|--------------|
| 53 | Raises AttributeError for invalid location |
| 54 | Returns False for arrays not in container |
| 55 | Skips arrays with None expected sizes |
| 56 | Returns False for dimension count mismatch |
| 57 | Returns True when all non-None expected dimensions match |

#### `check_incoming_arrays`

| # | Functionality |
|---|--------------|
| 58 | Combines check_sizes and check_type results |

#### `_update_host_array`

| # | Functionality |
|---|--------------|
| 59 | Raises ValueError when new_array is None |
| 60 | Skips when arrays equal (shape and optionally content) |
| 61 | Marks for reallocation+overwrite when current is None |
| 62 | Marks for reallocation+overwrite when shape differs |
| 63 | Creates zeros array when new_array has zero in shape |
| 64 | Marks for overwrite only when shapes match but content differs |

#### `update_host_arrays`

| # | Functionality |
|---|--------------|
| 65 | Warns for array names not in host container |
| 66 | Warns when arrays don't match expected sizes |
| 67 | Calls _update_host_array for each valid array |

#### `allocate`

| # | Functionality |
|---|--------------|
| 68 | Builds ArrayRequest for each array needing reallocation |
| 69 | Sets unchunkable from host metadata |
| 70 | Queues request via request_allocation |
| 71 | Skips None host arrays |

#### `reset`

| # | Functionality |
|---|--------------|
| 72 | Deletes all host and device arrays |
| 73 | Clears reallocation and overwrite lists |

#### `to_device` / `from_device`

| # | Functionality |
|---|--------------|
| 74 | Delegates to _memory_manager.to_device |
| 75 | Delegates to _memory_manager.from_device |

#### `_convert_host_to_pinned`

| # | Functionality |
|---|--------------|
| 76 | Converts "host" memory_type arrays to pinned via memory_manager |
| 77 | Sets host container memory_type to "pinned" |

#### `_convert_host_to_numpy`

| # | Functionality |
|---|--------------|
| 78 | Converts "pinned" arrays with chunked transfers to "host" memory |
| 79 | Updates slot memory_type to "host" |

---

## `batchsolving/arrays/BatchInputArrays.py`

### `InputArrayContainer`

| # | Functionality |
|---|--------------|
| 1 | Three ManagedArray fields: initial_values, parameters, driver_coefficients |
| 2 | driver_coefficients has is_chunked=False |

#### `host_factory` classmethod

| # | Functionality |
|---|--------------|
| 3 | Creates container with specified memory_type |

#### `device_factory` classmethod

| # | Functionality |
|---|--------------|
| 4 | Creates container with "device" memory_type |

### `InputArrays`

#### `__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 5 | Calls super().__attrs_post_init__() |
| 6 | Sets host memory_type to "pinned", device to "device" |

#### `update`

| # | Functionality |
|---|--------------|
| 7 | Builds updates dict from initial_values, parameters, optional driver_coefficients |
| 8 | Calls update_from_solver, update_host_arrays, allocate |

#### Forwarding Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 9 | `initial_values` | `host.initial_values.array` |
| 10 | `parameters` | `host.parameters.array` |
| 11 | `driver_coefficients` | `host.driver_coefficients.array` |
| 12 | `device_initial_values` | `device.initial_values.array` |
| 13 | `device_parameters` | `device.parameters.array` |
| 14 | `device_driver_coefficients` | `device.driver_coefficients.array` |

#### `from_solver` classmethod

| # | Functionality |
|---|--------------|
| 15 | Creates InputArrays with sizes, precision, memory_manager, stream_group from solver |

#### `update_from_solver`

| # | Functionality |
|---|--------------|
| 16 | Updates _sizes from BatchInputSizes.from_solver |
| 17 | Updates _precision from solver |
| 18 | Calls set_array_runs |
| 19 | Updates floating-point array dtypes to current precision |

#### `finalise`

| # | Functionality |
|---|--------------|
| 20 | Releases buffers via release_buffers |

#### `initialise`

| # | Functionality |
|---|--------------|
| 21 | Non-chunked: copies only _needs_overwrite arrays, then clears list |
| 22 | Chunked: copies all device arrays |
| 23 | Direct transfer when shapes match (no chunked transfer needed) |
| 24 | Chunked: acquires pinned buffer, copies host slice into buffer, stages for H2D |
| 25 | Calls to_device with from/to lists |

#### `release_buffers`

| # | Functionality |
|---|--------------|
| 26 | Releases all active buffers to pool and clears list |

#### `reset`

| # | Functionality |
|---|--------------|
| 27 | Calls super().reset() |
| 28 | Clears buffer pool and active buffers |

---

## `batchsolving/arrays/BatchOutputArrays.py`

### `OutputArrayContainer`

| # | Functionality |
|---|--------------|
| 1 | Six ManagedArray fields: state, observables, state_summaries, observable_summaries, status_codes, iteration_counters |
| 2 | status_codes has stride_order=("run",), dtype=int32 |
| 3 | iteration_counters has dtype=int32 |

#### `host_factory` classmethod

| # | Functionality |
|---|--------------|
| 4 | Creates container with specified memory_type |

#### `device_factory` classmethod

| # | Functionality |
|---|--------------|
| 5 | Creates container with "device" memory_type |

### `OutputArrays`

#### `__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 6 | Calls super().__attrs_post_init__() |
| 7 | Sets host to "pinned", device to "device" |

#### `update`

| # | Functionality |
|---|--------------|
| 8 | Calls update_from_solver, update_host_arrays (shape_only=True), allocate |

#### Forwarding Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 9 | `state` | `host.state.array` |
| 10 | `observables` | `host.observables.array` |
| 11 | `state_summaries` | `host.state_summaries.array` |
| 12 | `observable_summaries` | `host.observable_summaries.array` |
| 13 | `device_state` | `device.state.array` |
| 14 | `device_observables` | `device.observables.array` |
| 15 | `device_state_summaries` | `device.state_summaries.array` |
| 16 | `device_observable_summaries` | `device.observable_summaries.array` |
| 17 | `status_codes` | `host.status_codes.array` |
| 18 | `device_status_codes` | `device.status_codes.array` |
| 19 | `iteration_counters` | `host.iteration_counters.array` |
| 20 | `device_iteration_counters` | `device.iteration_counters.array` |

#### `from_solver` classmethod

| # | Functionality |
|---|--------------|
| 21 | Creates OutputArrays with sizes, precision, memory_manager, stream_group from solver |

#### `update_from_solver`

| # | Functionality |
|---|--------------|
| 22 | Updates _sizes from BatchOutputSizes.from_solver |
| 23 | Updates _precision from solver |
| 24 | Calls set_array_runs |
| 25 | Skips allocation when existing host array matches shape/dtype |
| 26 | Creates new host array via memory_manager.create_host_array when different |
| 27 | Updates floating-point device array dtypes to current precision |

#### `finalise`

| # | Functionality |
|---|--------------|
| 28 | Iterates host arrays, builds from/to lists for D2H transfer |
| 29 | Chunked: acquires buffer, registers PendingBuffer for watcher writeback |
| 30 | Non-chunked: uses host array directly as D2H target |
| 31 | Calls from_device |
| 32 | Records CUDA event and submits pending buffers to watcher when chunked |
| 33 | In CUDA_SIMULATION: event=None |

#### `wait_pending`

| # | Functionality |
|---|--------------|
| 34 | Delegates to _watcher.wait_all(timeout) |

#### `initialise`

| # | Functionality |
|---|--------------|
| 35 | No-op (pass) |

#### `reset`

| # | Functionality |
|---|--------------|
| 36 | Calls super().reset() |
| 37 | Clears buffer pool |
| 38 | Shuts down watcher |
| 39 | Clears pending buffers |
