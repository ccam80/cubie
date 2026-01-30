# CuBIE Test Sweep — Master Inventory

## Rules

1. **Phase 1 inventories functionality from source only.** No reference
   to existing tests. Existing tests are irrelevant until Phase 3.
2. Simple pass-through properties → table-driven forwarding checks
   against child object properties directly.
3. Calculated properties → dedicated tests.
4. Every conditional branch in source → inventory item.

## Legend

- [ ] = inventory not started
- [~] = inventory in progress
- [x] = inventory complete

---

## Completed Inventories

### `integrators/SingleIntegratorRunCore.py` [x]

#### `__init__` — Construction

| # | Functionality |
|---|--------------|
| 1 | Construction succeeds with all-`None` optional args |
| 2 | Construction succeeds with explicit values (no fallback) |
| 3 | Algorithm controller defaults flow through when no user `step_control_settings` |
| 4 | User `step_control_settings` override algorithm defaults |
| 5 | Precision popped from `output_settings`; system's precision used instead |
| 6 | `dt` from `step_control_settings` threaded into `algorithm_settings` |

#### `_process_loop_timing` — Timing Derivation

| # | Functionality |
|---|--------------|
| 7 | `_user_timing` dict updated from incoming settings |
| 8 | `save_last = True` when time-domain outputs requested + no `save_every` |
| 9 | `is_duration_dependent = True` when summary outputs requested + no `summarise_every` |
| 10 | `sample_summaries_every` auto-derived as `summarise_every / 10` when not provided |
| 11 | `summarise_every` and `sample_summaries_every` forced to `None` when no summary outputs |
| 12 | `save_regularly` / `summarise_regularly` booleans computed correctly |
| 13 | Derived values forwarded to loop and output_functions |
| 14 | Warning emitted (from Solver level) when `is_duration_dependent` is True |

#### `set_summary_timing_from_duration`

| # | Functionality |
|---|--------------|
| 15 | Duration-dependent path: `summarise_every = duration` propagated to loop |
| 16 | Duration-dependent path: `sample_summaries_every = duration / 100` propagated to loop and output_functions |
| 17 | Non-dependent path: existing values unchanged |

#### `n_error` Property

| # | Functionality |
|---|--------------|
| 18 | Returns `system.sizes.states` for adaptive algorithm (parametrized) |
| 19 | Returns 0 for non-adaptive algorithm (parametrized) |

#### `check_compatibility`

| # | Functionality |
|---|--------------|
| 20 | Incompatible pair (non-adaptive algo + adaptive controller) triggers replacement with fixed controller |
| 21 | Replacement uses `dt0` from original adaptive controller |
| 22 | Warning emitted naming algorithm, controller, mentioning "error estimate" and "fixed" |
| 23 | `None` fallback paths for `algorithm_name`, `controller_name`, `precision` (called from `update()`) |
| 24 | Replacement controller created with `warn_on_unused=False` |
| 25 | `is_controller_fixed` set on `_algo_step` based on final controller state |
| 26 | Compatible: adaptive algo + adaptive controller passes without warning |
| 27 | Compatible: non-adaptive algo + fixed controller passes without warning |
| 28 | Compatible: adaptive algo + fixed controller passes without warning |

#### `instantiate_loop`

| # | Functionality |
|---|--------------|
| 29 | All named parameters forwarded to IVPLoop constructor |
| 30 | `n_counters = 4` when `compile_flags.save_counters` is True |
| 31 | `n_counters = 0` when `compile_flags.save_counters` is False |
| 32 | `evaluate_driver_at_t` not overwritten if already in loop_settings |
| 33 | `evaluate_driver_at_t` set from argument if absent from loop_settings |
| 34 | `n_error` forwarded from `self.n_error` |

#### `update` — Parameter Routing

| # | Functionality |
|---|--------------|
| 35 | `updates_dict` and `kwargs` merged |
| 36 | Empty dict → early return with empty set |
| 37 | Nested dicts flattened via `unpack_dict_values` |
| 38 | Forwarded to `_system.update(silent=True)`, recognised keys captured |
| 39 | `n` injected from `_system.sizes.states` |
| 40 | Forwarded to `_output_functions.update(silent=True)`, `buffer_sizes_dict` captured if recognised |
| 41 | `_switch_algos` called, may swap algorithm |
| 42 | Forwarded to `_algo_step.update(silent=True)`, `threads_per_step` captured |
| 43 | `algorithm_order` injected |
| 44 | `_switch_controllers` called, may swap controller |
| 45 | Forwarded to `_step_controller.update(silent=True)`, `is_adaptive`/`dt_min`/`dt_max`/`dt0` captured |
| 46 | Algo and controller buffers re-registered with loop |
| 47 | Forwarded to `_loop.update(silent=True)` |
| 48 | `_process_loop_timing` called with updates |
| 49 | `update_compile_settings` called |
| 50 | Unrecognised params raise `KeyError` when `silent=False` |
| 51 | No `KeyError` when `silent=True` |
| 52 | Cache invalidated if any keys recognised |
| 53 | `check_compatibility` called after updates |
| 54 | Unpacked dict keys included in returned recognised set |
| 55 | Downstream components see upstream changes (system → algo → controller → loop ordering) |

#### `_switch_algos`

| # | Functionality |
|---|--------------|
| 56 | No-op when `"algorithm"` not in updates_dict (returns empty set) |
| 57 | Buffer registry reset on any algorithm change attempt |
| 58 | New algorithm step created when name differs from current |
| 59 | Old settings carried forward to new algorithm |
| 60 | `compile_settings.algorithm` updated |
| 61 | Algorithm name normalised to lowercase |
| 62 | Algorithm controller defaults merged into updates_dict (only for missing keys) |
| 63 | `algorithm_order` injected |

#### `_switch_controllers`

| # | Functionality |
|---|--------------|
| 64 | No-op when `"step_controller"` not in updates_dict (returns empty set) |
| 65 | Buffer registry reset |
| 66 | Controller name normalised to lowercase |
| 67 | New controller created when name differs, old settings carried forward |
| 68 | `algorithm_order` injected from updates_dict if present |
| 69 | `algorithm_order` falls back to `self._algo_step.order` if not in updates_dict |
| 70 | `compile_settings.step_controller` updated |

#### `build`

| # | Functionality |
|---|--------------|
| 71 | Fetches `evaluate_f`, `evaluate_observables`, `get_solver_helper` from system |
| 72 | Conditionally updates algo_step only when functions differ from current |
| 73 | Does NOT update algo_step when functions unchanged |
| 74 | Changed `evaluate_f` propagates through to algo_step |
| 75 | Algo and controller buffers re-registered with loop |
| 76 | Compiled functions dict assembled from output_functions, step_controller, algo_step |
| 77 | Loop updated with compiled functions dict |
| 78 | Returns `SingleIntegratorRunCache` wrapping loop's `device_function` |

#### Computed Properties

| # | Functionality |
|---|--------------|
| 79 | `time_domain_outputs_requested` forwards to `_output_functions.has_time_domain_outputs` |
| 80 | `summary_outputs_requested` forwards to `_output_functions.has_summary_outputs` |
| 81 | `has_time_domain_outputs`: True when types requested + `save_every` set |
| 82 | `has_time_domain_outputs`: True when types requested + `save_last` True |
| 83 | `has_time_domain_outputs`: False when types requested + neither timing set |
| 84 | `has_time_domain_outputs`: False when no types requested + timing set |
| 85 | `has_summary_outputs`: True when types requested + `summarise_every` set |
| 86 | `has_summary_outputs`: False when types requested + no `summarise_every` |
| 87 | `has_summary_outputs`: False when no types requested + `summarise_every` set |
| 88 | `has_summary_outputs`: False when no types + no `summarise_every` |

---

### `integrators/SingleIntegratorRun.py` [x]

#### Forwarding Properties (table-driven)

All compared against child object property directly.

| # | Property | Delegates to |
|---|----------|-------------|
| 1 | `algorithm` | `compile_settings.algorithm` |
| 2 | `step_controller` | `compile_settings.step_controller` |
| 3 | `shared_memory_elements` | `_loop.shared_buffer_size` |
| 4 | `shared_memory_bytes` | **dedicated test** (see below) |
| 5 | `local_memory_elements` | `_loop.local_buffer_size` |
| 6 | `persistent_local_elements` | `_loop.persistent_local_buffer_size` |
| 7 | `dt0` | `_step_controller.dt0` |
| 8 | `dt_min` | `_step_controller.dt_min` |
| 9 | `dt_max` | `_step_controller.dt_max` |
| 10 | `is_adaptive` | `_step_controller.is_adaptive` |
| 11 | `system` | `_system` |
| 12 | `system_sizes` | `_system.sizes` |
| 13 | `save_summaries_func` | `save_summary_metrics_func` (chain) |
| 14 | `evaluate_f` | `_algo_step.evaluate_f` |
| 15 | `save_every` | `_loop.save_every` |
| 16 | `summarise_every` | `_loop.summarise_every` |
| 17 | `sample_summaries_every` | `_loop.sample_summaries_every` |
| 18 | `save_last` | `_loop.compile_settings.save_last` |
| 19 | `compile_flags` | `_loop.compile_flags` |
| 20 | `save_state_fn` | `_loop.save_state_fn` |
| 21 | `update_summaries_fn` | `_loop.update_summaries_fn` |
| 22 | `save_summaries_fn` | `_loop.save_summaries_fn` |
| 23 | `atol` | `_step_controller.atol` (with hasattr guard) |
| 24 | `rtol` | `_step_controller.rtol` (with hasattr guard) |
| 25 | `dt` | `_step_controller.dt` (with hasattr guard) |
| 26 | `threads_per_step` | `_algo_step.threads_per_step` |
| 27 | `save_state_func` | `_output_functions.save_state_func` |
| 28 | `update_summaries_func` | `_output_functions.update_summaries_func` |
| 29 | `save_summary_metrics_func` | `_output_functions.save_summary_metrics_func` |
| 30 | `output_types` | `_output_functions.output_types` |
| 31 | `output_compile_flags` | `_output_functions.compile_flags` |
| 32 | `save_time` | `_output_functions.save_time` |
| 33 | `saved_state_indices` | `_output_functions.saved_state_indices` |
| 34 | `saved_observable_indices` | `_output_functions.saved_observable_indices` |
| 35 | `summarised_state_indices` | `_output_functions.summarised_state_indices` |
| 36 | `summarised_observable_indices` | `_output_functions.summarised_observable_indices` |
| 37 | `output_array_heights` | `_output_functions.output_array_heights` |
| 38 | `summary_legend_per_variable` | `_output_functions.summary_legend_per_variable` |
| 39 | `summary_unit_modifications` | `_output_functions.summary_unit_modifications` |

#### `shared_memory_bytes` — Dedicated Test

| # | Functionality |
|---|--------------|
| 40 | Returns `elements * 4` for float32 precision |
| 41 | Returns `elements * 8` for float64 precision |

#### `output_length` — Calculation

| # | Functionality |
|---|--------------|
| 42 | Always includes 1 initial sample |
| 43 | Adds `floor(duration / save_every)` regular samples when `save_every` set |
| 44 | Adds 1 final sample when `save_last` is True |
| 45 | Returns 1 when `save_every` is None and `save_last` is False |
| 46 | Returns 2 when `save_every` is None and `save_last` is True |

#### `summaries_length` — Calculation

| # | Functionality |
|---|--------------|
| 47 | Returns `int(duration / summarise_every)` when set |
| 48 | Returns 0 when `summarise_every` is None |

---

## Inventoried Files

### Core Infrastructure

| # | Source File | Inventory |
|---|------------|-----------|
| 1 | `_utils.py` | [x] |
| 2 | `cuda_simsafe.py` | [x] |
| 3 | `buffer_registry.py` | [x] |
| 4 | `CUDAFactory.py` | [x] |
| 5 | `time_logger.py` | [x] |
| 6 | `cubie_cache.py` | [x] |
| 7 | `vendored/numba_cuda_cache.py` | [x] |
| 8 | `__init__.py` | [x] |


### `_utils.py` [x]

#### `precision_converter` — Convert dtype to canonical scalar type

| # | Functionality |
|---|--------------|
| 1 | Returns `dtype_.type` for valid float16/float32/float64 input |
| 2 | Raises `ValueError` for unsupported precision (e.g. int32) |
| 3 | Accepts both `np.dtype` objects and bare type classes |

#### `precision_validator` — Validate precision attribute

| # | Functionality |
|---|--------------|
| 4 | Passes silently for allowed precisions (float16/32/64) |
| 5 | Raises `ValueError` for unsupported precision |

#### `buffer_dtype_validator` — Validate buffer dtype attribute

| # | Functionality |
|---|--------------|
| 6 | Passes silently for allowed buffer dtypes (float16/32/64, int32/64) |
| 7 | Raises `ValueError` for unsupported dtype |

#### `slice_variable_dimension` — Create combined slice

| # | Functionality |
|---|--------------|
| 8 | Wraps single `slice` input into a list |
| 9 | Wraps single `int` indices input into a list |
| 10 | Raises `ValueError` when slices and indices differ in length |
| 11 | Raises `ValueError` when max(indices) >= ndim |
| 12 | Returns correct tuple with slices placed at specified indices, rest `slice(None)` |

#### `in_attr` — Check field existence on attrs instance

| # | Functionality |
|---|--------------|
| 13 | Returns `True` when `name` is a field of the attrs class |
| 14 | Returns `True` when `_name` (underscore-prefixed) is a field |
| 15 | Returns `False` when neither `name` nor `_name` is a field |

#### `merge_kwargs_into_settings` — Merge component settings

| # | Functionality |
|---|--------------|
| 16 | Filters kwargs to only keys in `valid_keys` |
| 17 | Returns `user_settings` defaults when no kwargs match |
| 18 | kwargs override `user_settings` values |
| 19 | Warns `UserWarning` when duplicate keys exist in both filtered and user_settings |
| 20 | Returns empty user_settings dict when `user_settings` is None |
| 21 | Returns correct recognized set of consumed keys |

#### `clamp_factory` — Compile CUDA clamp device function

| # | Functionality |
|---|--------------|
| 22 | Returns a CUDA device function |
| 23 | Returned function clamps value between minimum and maximum |

#### `is_device_validator` — Validate CUDA device function

| # | Functionality |
|---|--------------|
| 24 | Passes silently for valid CUDA device functions |
| 25 | Raises `TypeError` for non-device-function values |

#### `float_array_validator` — Validate float array

| # | Functionality |
|---|--------------|
| 26 | Raises `TypeError` when value is not ndarray |
| 27 | Raises `TypeError` when array dtype is not float |
| 28 | Raises `ValueError` when array contains NaN or infinity |
| 29 | Passes silently for valid finite float array |

#### `inrangetype_validator` — Composite type+range validator

| # | Functionality |
|---|--------------|
| 30 | Validates instance_of, ge(min_), le(max_) combined |
| 31 | Expands `float` to accept `np_floating` |
| 32 | Expands `int` to accept `np_integer` |

#### `_expand_dtype` — Expand Python types for NumPy

| # | Functionality |
|---|--------------|
| 33 | `float` -> `(float, np_floating)` |
| 34 | `int` -> `(int, np_integer)` |
| 35 | Unknown types returned unchanged |

#### `gttype_validator` — Type + greater-than validator

| # | Functionality |
|---|--------------|
| 36 | Validates instance_of and gt(min_) combined |

#### `getype_validator` — Type + greater-or-equal validator

| # | Functionality |
|---|--------------|
| 37 | Validates instance_of and ge(min_) combined |

#### `tol_converter` — Convert tolerance to array

| # | Functionality |
|---|--------------|
| 38 | Returns value unchanged when `_n_changing` is True |
| 39 | Scalar value broadcast to shape (n,) array with correct precision |
| 40 | Array value converted to correct precision |
| 41 | Single-element array broadcast to shape (n,) when n > 1 |
| 42 | Raises `ValueError` when array shape doesn't match n |

#### `opt_gttype_validator` / `opt_getype_validator` — Optional validators

| # | Functionality |
|---|--------------|
| 43 | `opt_gttype_validator` accepts None |
| 44 | `opt_getype_validator` accepts None |
| 45 | Both delegate to inner validator for non-None values |

#### `ensure_nonzero_size` — Replace zero-size shapes

| # | Functionality |
|---|--------------|
| 46 | Integer input: returns `max(1, value)` |
| 47 | Tuple with any zero/None element: returns all-ones tuple |
| 48 | Tuple with no zeros: returns original tuple |
| 49 | Other types: passes through unchanged |

#### `unpack_dict_values` — Flatten nested dicts

| # | Functionality |
|---|--------------|
| 50 | Dict values are unpacked into result; original key tracked |
| 51 | Non-dict values preserved as-is |
| 52 | Raises `ValueError` on key collision between unpacked dict and existing keys |
| 53 | Raises `ValueError` on duplicate non-dict keys |
| 54 | Returns empty dict and empty set for empty input |

#### `build_config` — Build attrs config from dicts

| # | Functionality |
|---|--------------|
| 55 | Raises `TypeError` if config_class is not attrs |
| 56 | Merges required and optional kwargs |
| 57 | Filters to only valid field names |
| 58 | Handles underscore-prefixed field aliases |
| 59 | With `instance_label`: adds to merged, transforms prefixed keys |
| 60 | With `instance_label`: raises `ValueError` if class has no `instance_label` attr |
| 61 | Without `instance_label`: no prefix transformation |

---

### `cuda_simsafe.py` [x]

#### Module-level constants

| # | Functionality |
|---|--------------|
| 62 | `CUDA_SIMULATION` reflects `NUMBA_ENABLE_CUDASIM` env var |
| 63 | `compile_kwargs` is empty dict in CUDASIM, contains fastmath settings otherwise |

#### `FakeBaseCUDAMemoryManager` — Stub memory manager

| # | Functionality |
|---|--------------|
| 64 | `__init__` stores context |
| 65 | `initialize` is no-op |
| 66 | `reset` is no-op |
| 67 | `defer_cleanup` returns no-op context manager |

#### `FakeNumbaCUDAMemoryManager` — Fake memory manager

| # | Functionality |
|---|--------------|
| 68 | Inherits from `FakeBaseCUDAMemoryManager` |
| 69 | Has class-level `handle`, `ptr`, `free`, `total` attributes |

#### `FakeGetIpcHandleMixin` — Fake IPC handle

| # | Functionality |
|---|--------------|
| 70 | `get_ipc_handle` returns a `FakeIpcHandle` instance |

#### `FakeStream` — Placeholder CUDA stream

| # | Functionality |
|---|--------------|
| 71 | Has `handle` attribute as `c_void_p(0)` |

#### `FakeMemoryPointer` — Lightweight pointer stub

| # | Functionality |
|---|--------------|
| 72 | Stores context, device_pointer, size, finalizer |
| 73 | Sets `handle` to `device_pointer` |
| 74 | Sets `_cuda_memsize` to `size` |

#### `FakeMemoryInfo` — Fake memory stats

| # | Functionality |
|---|--------------|
| 75 | `free` = 1 GB, `total` = 8 GB |

#### `is_cuda_array` — Check CUDA array

| # | Functionality |
|---|--------------|
| 76 | In CUDASIM: returns `hasattr(value, "shape")` |
| 77 | In real CUDA: delegates to `_is_cuda_array` |

#### `from_dtype` — CUDA-ready dtype

| # | Functionality |
|---|--------------|
| 78 | In CUDASIM: returns dt unchanged |
| 79 | In real CUDA: returns `numba_from_dtype(dt)` |

#### `is_devfunc` — Test CUDA device function

| # | Functionality |
|---|--------------|
| 80 | In CUDASIM: checks `_device` attribute |
| 81 | In real CUDA: checks `targetoptions["device"]` |
| 82 | Returns False when targetoptions is not a dict |
| 83 | Returns False when no targetoptions attribute |

#### `is_cudasim_enabled` — Check simulator state

| # | Functionality |
|---|--------------|
| 84 | Returns `CUDA_SIMULATION` constant |

#### CUDASIM device functions

| # | Functionality |
|---|--------------|
| 85 | `selp` returns true_value when pred is True, false_value otherwise |
| 86 | `activemask` returns 0xFFFFFFFF |
| 87 | `all_sync` returns predicate directly |
| 88 | `any_sync` returns predicate directly |
| 89 | `syncwarp` is no-op |
| 90 | `stwt` writes value to array[index] |

---

### `buffer_registry.py` [x]

#### `CUDABuffer` — Buffer record

| # | Functionality |
|---|--------------|
| 91 | Construction with all required fields succeeds |
| 92 | `is_shared` returns True when location='shared' |
| 93 | `is_shared` returns False when location='local' |
| 94 | `is_persistent_local` returns True when location='local' and persistent=True |
| 95 | `is_persistent_local` returns False when persistent=False |
| 96 | `is_local` returns True when location='local' and persistent=False |
| 97 | `is_local` returns False when persistent=True |

#### `CUDABuffer.build_allocator` — Compile allocator device function

| # | Functionality |
|---|--------------|
| 98 | With shared_slice: allocator returns shared[slice] |
| 99 | With persistent_slice (no shared): allocator returns persistent[slice] |
| 100 | With local_size (no shared/persistent): allocator creates local array |
| 101 | With zero=True: allocator zeroes all elements after allocation |
| 102 | With zero=False: no zeroing performed |

#### `BufferGroup` — Buffer group management

| # | Functionality |
|---|--------------|
| 103 | `invalidate_layouts` sets all layouts to None and clears alias consumption |
| 104 | `build_layouts` short-circuits when all layouts already built |
| 105 | `build_layouts` builds shared layout with sequential offsets |
| 106 | `build_layouts` builds persistent layout with sequential offsets |
| 107 | `build_layouts` builds local sizes with max(size, 1) |
| 108 | `build_layouts` calls `layout_aliases` for aliased entries |

#### `BufferGroup.layout_aliases` — Alias resolution

| # | Functionality |
|---|--------------|
| 109 | Aliased buffer with space in shared parent: overlaps within parent |
| 110 | Aliased buffer exceeding shared parent: falls back to entry's own type |
| 111 | Fallback is_shared: allocates in shared layout |
| 112 | Fallback is_persistent_local: allocates in persistent layout |
| 113 | Fallback is_local: added to local pile |
| 114 | Local pile entries added to _local_sizes |

#### `BufferGroup.register` — Register buffer

| # | Functionality |
|---|--------------|
| 115 | Raises `ValueError` for empty name |
| 116 | Raises `ValueError` for self-aliasing |
| 117 | Raises `ValueError` for alias target not registered |
| 118 | Successful registration adds entry and invalidates layouts |

#### `BufferGroup.update_buffer` — Update existing buffer

| # | Functionality |
|---|--------------|
| 119 | Returns (False, False) for unregistered buffer |
| 120 | Returns (True, False) when values unchanged |
| 121 | Returns (True, True) when values changed; invalidates layouts |

#### `BufferGroup` — Layout properties (lazy build)

| # | Functionality |
|---|--------------|
| 122 | `shared_layout` triggers build when None |
| 123 | `persistent_layout` triggers build when None |
| 124 | `local_sizes` triggers build when None |

#### `BufferGroup` — Size methods

| # | Functionality |
|---|--------------|
| 125 | `shared_buffer_size` returns 0 for empty layout |
| 126 | `shared_buffer_size` returns max stop for non-empty layout |
| 127 | `local_buffer_size` returns sum of local sizes |
| 128 | `persistent_local_buffer_size` returns 0 for empty layout |
| 129 | `persistent_local_buffer_size` returns max stop for non-empty |

#### `BufferGroup.get_allocator` — Generate allocator

| # | Functionality |
|---|--------------|
| 130 | Raises `KeyError` for unregistered buffer |
| 131 | Returns allocator for registered buffer with correct slice |

#### `BufferRegistry` — Central registry

| # | Functionality |
|---|--------------|
| 132 | `register` creates new BufferGroup for unknown parent |
| 133 | `register` reuses existing group for known parent |
| 134 | `update_buffer` returns (False, False) for unknown parent |
| 135 | `update_buffer` delegates to group for known parent |
| 136 | `clear_layout` invalidates layouts for known parent |
| 137 | `clear_layout` no-op for unknown parent |
| 138 | `clear_parent` removes group for known parent |
| 139 | `clear_parent` no-op for unknown parent |
| 140 | `reset` clears all groups |

#### `BufferRegistry.update` — Update buffer locations

| # | Functionality |
|---|--------------|
| 141 | Returns empty set for empty updates |
| 142 | Returns empty set for unknown parent |
| 143 | Recognizes keys ending in `_location` |
| 144 | Raises `ValueError` for invalid location value |
| 145 | Updates buffer location and clears layout on change |
| 146 | Returns set of recognized keys |

#### `BufferRegistry` — Size delegation methods

| # | Functionality |
|---|--------------|
| 147 | `shared_buffer_size` returns 0 for unknown parent |
| 148 | `local_buffer_size` returns 0 for unknown parent |
| 149 | `persistent_local_buffer_size` returns 0 for unknown parent |
| 150 | All three delegate to group methods for known parent |

#### `BufferRegistry.get_allocator`

| # | Functionality |
|---|--------------|
| 151 | Raises `KeyError` for unknown parent |
| 152 | Delegates to group for known parent |

#### `BufferRegistry.get_child_allocators`

| # | Functionality |
|---|--------------|
| 153 | Registers child shared and persistent buffers with parent |
| 154 | Uses `child_{id}` name when name is None |
| 155 | Uses provided name when given |
| 156 | Returns shared and persistent allocators |

#### `BufferRegistry.get_toplevel_allocators`

| # | Functionality |
|---|--------------|
| 157 | `alloc_shared` returns `cuda.shared.array(0, float32)` |
| 158 | `alloc_persistent` returns local array of `max(1, kernel.local_memory_elements)` |
| 159 | Uses kernel's precision for persistent allocator |

---

### `CUDAFactory.py` [x]

#### `hash_tuple` — Hash tuple to SHA256

| # | Functionality |
|---|--------------|
| 160 | None values serialized as "None" |
| 161 | ndarray values hashed by bytes with shape/dtype |
| 162 | Other values serialized via str() |
| 163 | Returns 64-character hex digest |

#### `attribute_is_hashable` — Check if attribute is hashable

| # | Functionality |
|---|--------------|
| 164 | Returns False when attribute.eq is False |
| 165 | Returns True otherwise |

#### `_CubieConfigBase` — Base config class

| # | Functionality |
|---|--------------|
| 166 | `__attrs_post_init__` builds field_map from name and alias |
| 167 | `__attrs_post_init__` identifies nested attrs fields |
| 168 | `__attrs_post_init__` identifies unhashable (eq=False) fields |
| 169 | `__attrs_post_init__` generates initial values_hash |
| 170 | `__attrs_post_init__` raises `TypeError` for dict-type fields not marked eq=False |

#### `_CubieConfigBase.update` — Update fields

| # | Functionality |
|---|--------------|
| 171 | Returns (empty, empty) for empty updates |
| 172 | Recognizes fields by name or alias |
| 173 | Handles ndarray comparison via `array_equal` |
| 174 | Handles scalar comparison via `!=` |
| 175 | Only updates when value changed |
| 176 | Delegates to nested attrs objects |
| 177 | Regenerates hash after changes |
| 178 | Returns recognized and changed sets |

#### `_CubieConfigBase` — Properties

| # | Functionality |
|---|--------------|
| 179 | `cache_dict` returns dict without eq=False fields |
| 180 | `values_tuple` returns tuple without eq=False fields |
| 181 | `values_hash` returns stored hash string |

#### `CUDAFactoryConfig` — Config with precision

| # | Functionality |
|---|--------------|
| 182 | Construction validates and converts precision |
| 183 | `numba_precision` returns `from_dtype(np_dtype(self.precision))` |
| 184 | `simsafe_precision` returns `simsafe_dtype(np_dtype(self.precision))` |

#### `CUDAFactory` — Factory base class

| # | Functionality |
|---|--------------|
| 185 | `__init__` sets settings=None, cache_valid=True, cache=None |
| 186 | `setup_compile_settings` raises `TypeError` for non-attrs |
| 187 | `setup_compile_settings` stores settings and invalidates cache |
| 188 | `cache_valid` returns `_cache_valid` |
| 189 | `device_function` calls `get_cached_output("device_function")` |
| 190 | `compile_settings` returns `_compile_settings` |

#### `CUDAFactory.update_compile_settings`

| # | Functionality |
|---|--------------|
| 191 | Returns empty set for empty updates |
| 192 | Raises `ValueError` when settings not set up |
| 193 | Delegates to compile_settings.update() |
| 194 | Raises `KeyError` for unrecognized params when silent=False |
| 195 | Suppresses errors when silent=True |
| 196 | Invalidates cache when any field changed |
| 197 | Returns recognized set |

#### `CUDAFactory._build` / `_invalidate_cache`

| # | Functionality |
|---|--------------|
| 198 | `_invalidate_cache` sets `_cache_valid` to False |
| 199 | `_build` calls build(), raises `TypeError` if not CUDADispatcherCache |
| 200 | `_build` stores result and sets cache_valid=True |

#### `CUDAFactory.get_cached_output`

| # | Functionality |
|---|--------------|
| 201 | Triggers `_build` when cache invalid |
| 202 | Raises `RuntimeError` when cache is None after build |
| 203 | Raises `KeyError` when output_name not in cache |
| 204 | Raises `NotImplementedError` when cached value is int(-1) |
| 205 | Returns cached value for valid output |

#### `CUDAFactory.config_hash`

| # | Functionality |
|---|--------------|
| 206 | Returns own hash when no child factories |
| 207 | Combines own hash with child hashes when children exist |

#### `CUDAFactory._iter_child_factories`

| # | Functionality |
|---|--------------|
| 208 | Yields CUDAFactory instances from direct attributes |
| 209 | Alphabetical ordering by attribute name |
| 210 | Deduplicates by id |

#### Forwarding Properties (table-driven)

| # | Property | Delegates to |
|---|----------|-------------|
| 211 | `precision` | `compile_settings.precision` |
| 212 | `numba_precision` | `compile_settings.numba_precision` |
| 213 | `simsafe_precision` | `compile_settings.simsafe_precision` |
| 214 | `shared_buffer_size` | `buffer_registry.shared_buffer_size(self)` |
| 215 | `local_buffer_size` | `buffer_registry.local_buffer_size(self)` |
| 216 | `persistent_local_buffer_size` | `buffer_registry.persistent_local_buffer_size(self)` |

#### `MultipleInstanceCUDAFactoryConfig`

| # | Functionality |
|---|--------------|
| 217 | `get_prefixed_attributes(aliases=False)` returns field names with metadata["prefixed"] |
| 218 | `get_prefixed_attributes(aliases=True)` returns aliases instead of names |
| 219 | `prefix` property returns `instance_label` |
| 220 | `__attrs_post_init__` sets `prefixed_attributes` when label non-empty |

#### `MultipleInstanceCUDAFactoryConfig.update`

| # | Functionality |
|---|--------------|
| 221 | Removes non-prefixed keys for prefixed attributes |
| 222 | Maps prefixed keys (e.g. `newton_atol`) to unprefixed (`atol`) |
| 223 | Returns recognized/changed with prefixed key names restored |

#### `MultipleInstanceCUDAFactory`

| # | Functionality |
|---|--------------|
| 224 | `__init__` stores `_instance_label` and calls super().__init__ |
| 225 | `instance_label` property returns `_instance_label` |

---

### `time_logger.py` [x]

#### `TimingEvent` — Frozen timing record

| # | Functionality |
|---|--------------|
| 226 | Construction with valid event_type ('start'/'stop'/'progress') succeeds |
| 227 | Construction with invalid event_type raises error |

#### `CUDAEvent` — CUDA event pair

| # | Functionality |
|---|--------------|
| 228 | Uses default_timelogger when timelogger is None |
| 229 | In real CUDA: creates start/end event objects |
| 230 | In CUDASIM: uses wall-clock timestamps |
| 231 | Registers with timelogger on construction |

#### `CUDAEvent.record_start` / `record_end`

| # | Functionality |
|---|--------------|
| 232 | No-op when verbosity is None |
| 233 | In real CUDA: records event on stream |
| 234 | In CUDASIM: stores perf_counter timestamp |

#### `CUDAEvent.elapsed_time_ms`

| # | Functionality |
|---|--------------|
| 235 | Returns 0.0 when verbosity is None |
| 236 | Returns 0.0 when start/end not recorded |
| 237 | In real CUDA: uses `cuda.event_elapsed_time` |
| 238 | In CUDASIM: computes `(end - start) * 1000` |

#### `TimeLogger.__init__`

| # | Functionality |
|---|--------------|
| 239 | Raises `ValueError` for invalid verbosity |
| 240 | Normalizes string "None" to None |
| 241 | Initializes empty events, _active_starts, _event_registry, _cuda_events |

#### `TimeLogger.start_event`

| # | Functionality |
|---|--------------|
| 242 | No-op when verbosity is None |
| 243 | Raises `ValueError` for empty event_name |
| 244 | Raises `ValueError` for unregistered event |
| 245 | Silently skips if event already has active start (duplicate start) |
| 246 | Records TimingEvent with skipped flag in metadata |
| 247 | Prints debug/verbose/default messages based on verbosity level |
| 248 | Skipped events get different message formatting |

#### `TimeLogger.stop_event`

| # | Functionality |
|---|--------------|
| 249 | No-op when verbosity is None |
| 250 | Raises `ValueError` for empty event_name |
| 251 | Raises `ValueError` for unregistered event |
| 252 | Silently skips if no active start (duplicate stop) |
| 253 | Computes duration from start timestamp |
| 254 | Removes event from _active_starts |
| 255 | Prints debug/verbose/default messages based on verbosity |
| 256 | Verbose mode skips printing for was_skipped events |

#### `TimeLogger.progress`

| # | Functionality |
|---|--------------|
| 257 | No-op when verbosity is None |
| 258 | Raises `ValueError` for empty/unregistered event |
| 259 | Records progress TimingEvent with message in metadata |
| 260 | Only prints in debug mode |

#### `TimeLogger.get_event_duration`

| # | Functionality |
|---|--------------|
| 261 | Returns duration for most recent completed start/stop pair |
| 262 | Returns None when no matching pair found |

#### `TimeLogger.get_aggregate_durations`

| # | Functionality |
|---|--------------|
| 263 | Sums durations for events with same name |
| 264 | Filters by category when provided |
| 265 | Returns empty dict when no matching events |

#### `TimeLogger._get_category_total`

| # | Functionality |
|---|--------------|
| 266 | Sums standard durations from get_aggregate_durations |
| 267 | Adds CUDA event durations from metadata (converted ms -> s) |

#### `TimeLogger.print_summary`

| # | Functionality |
|---|--------------|
| 268 | No-op when verbosity is None |
| 269 | Retrieves CUDA events for runtime/None category |
| 270 | Default mode: one line per category |
| 271 | Verbose mode: category totals |
| 272 | Debug mode: per-event breakdown with category summaries |
| 273 | Clears events after printing |

#### `TimeLogger.set_verbosity`

| # | Functionality |
|---|--------------|
| 274 | Raises `ValueError` for invalid verbosity |
| 275 | Normalizes "None" string to None |

#### `TimeLogger.register_event`

| # | Functionality |
|---|--------------|
| 276 | Raises `ValueError` for invalid category |
| 277 | Registers event with metadata (category, description, messages) |
| 278 | Skips duplicate registration (idempotent) |

#### `TimeLogger.print_message`

| # | Functionality |
|---|--------------|
| 279 | No-op when verbosity is None |
| 280 | Prints when current level >= required level |
| 281 | Does not print when current level < required level |

#### `TimeLogger._register_cuda_event`

| # | Functionality |
|---|--------------|
| 282 | No-op when verbosity is None |
| 283 | Stores CUDAEvent and registers with event registry |

#### `TimeLogger._retrieve_cuda_events`

| # | Functionality |
|---|--------------|
| 284 | No-op when verbosity is None or no events |
| 285 | Creates TimingEvent with duration_ms in metadata for each CUDAEvent |
| 286 | Clears _cuda_events after retrieval |

#### `TimeLogger._clear_events`

| # | Functionality |
|---|--------------|
| 287 | Clears events, _active_starts, and _cuda_events |

---

### `cubie_cache.py` [x]

#### `CUBIECacheLocator` — Cache file locator

| # | Functionality |
|---|--------------|
| 288 | Default path: `GENERATED_DIR / system_name / CUDA_cache_{hash[:8]}` |
| 289 | Custom path: `custom_cache_dir / CUDA_cache_{hash[:8]}` |
| 290 | `get_cache_path` returns absolute path string |
| 291 | `get_source_stamp` returns system_hash |
| 292 | `get_disambiguator` returns first 16 chars of compile_settings_hash |
| 293 | `set_compile_settings_hash` updates stored hash |
| 294 | `set_system_hash` updates hash and refreshes cache path |
| 295 | `from_function` raises `NotImplementedError` |

#### `CUBIECacheImpl` — Serialization logic

| # | Functionality |
|---|--------------|
| 296 | Construction creates CUBIECacheLocator and sets filename_base |
| 297 | `locator` property returns _locator |
| 298 | `filename_base` property regenerates from locator and returns |
| 299 | `set_hashes` updates system_hash and/or compile_settings_hash |
| 300 | `reduce` in real CUDA: returns `kernel._reduce_states()` |
| 301 | `reduce` in CUDASIM: raises `RuntimeError` |
| 302 | `rebuild` in real CUDA: returns `_Kernel._rebuild(**payload)` |
| 303 | `rebuild` in CUDASIM: raises `RuntimeError` |
| 304 | `check_cachable` always returns True |

#### `CUBIECache` — File-based kernel cache

| # | Functionality |
|---|--------------|
| 305 | `__init__` creates CUBIECacheImpl, IndexDataCacheFile, enables cache |
| 306 | `_index_key` includes sig, codegen.magic_tuple, system_hash, compile_settings_hash |
| 307 | `load_overload` on cache hit: prints message, returns result |
| 308 | `load_overload` on cache miss: prints message, starts compile timer, returns None |
| 309 | `save_overload` stops compile timer and enforces cache limit |
| 310 | `enforce_cache_limit` no-op when max_entries=0 |
| 311 | `enforce_cache_limit` no-op when under limit |
| 312 | `enforce_cache_limit` evicts oldest .nbi/.nbc pairs when over limit |
| 313 | `enforce_cache_limit` warns on OSError for .nbi removal |
| 314 | `enforce_cache_limit` silently continues on .nbc removal failure |
| 315 | `flush_cache` removes directory and recreates empty |
| 316 | `cache_path` property returns Path |
| 317 | `update_from_config` updates max_entries, mode, system_hash |
| 318 | `update_from_config` recreates impl when cache dir changes |
| 319 | `update_from_config` preserves impl when cache dir unchanged |
| 320 | `set_hashes` updates hashes and rebuilds cache_file when filename changes |
| 321 | `set_hashes` preserves cache_file when filename unchanged |

#### `CacheConfig` — Configuration attrs class

| # | Functionality |
|---|--------------|
| 322 | `params_from_user_kwarg(True)` returns enabled=True, mode='hash', path=None |
| 323 | `params_from_user_kwarg(False)` returns enabled=False |
| 324 | `params_from_user_kwarg(None)` returns enabled=False |
| 325 | `params_from_user_kwarg("flush_on_change")` returns mode='flush_on_change' |
| 326 | `params_from_user_kwarg("/path/str")` returns path=Path("/path/str") |
| 327 | `params_from_user_kwarg(Path(...))` returns path=that Path |

#### `CubieCacheHandler` — Cache handler

| # | Functionality |
|---|--------------|
| 328 | `__init__` with cache_arg=True creates CUBIECache |
| 329 | `__init__` with cache_arg=False/None sets _cache=None |
| 330 | `cache` property returns _cache |
| 331 | `flush` delegates to cache when present, no-op when None |

#### `CubieCacheHandler.update`

| # | Functionality |
|---|--------------|
| 332 | Returns empty set for empty updates |
| 333 | Delegates to config.update |
| 334 | Creates cache when cache_enabled changed to True |
| 335 | Removes cache when cache_enabled changed to False |
| 336 | Updates existing cache from config when settings changed |
| 337 | Raises `KeyError` for unrecognized params when silent=False |

#### `CubieCacheHandler.configured_cache`

| # | Functionality |
|---|--------------|
| 338 | Returns None when _cache is None |
| 339 | Updates system_hash in config if changed |
| 340 | Sets hashes on cache and returns it |

#### `CubieCacheHandler.invalidate`

| # | Functionality |
|---|--------------|
| 341 | No-op when _cache is None |
| 342 | No-op when mode != 'flush_on_change' |
| 343 | Flushes cache when mode is 'flush_on_change' |

#### `CubieCacheHandler.cache_enabled`

| # | Functionality |
|---|--------------|
| 344 | Returns `config.cache_enabled` |

---

### `odesystems/ODEData.py` [x]

#### `SystemSizes` — Frozen counts

| # | Functionality |
|---|--------------|
| 345 | Construction with all int fields succeeds |
| 346 | All fields validated as int |

#### `ODEData` — ODE data container

| # | Functionality |
|---|--------------|
| 347 | Construction with SystemValues instances succeeds |
| 348 | `update_precisions` updates precision on all SystemValues when key present |
| 349 | `update_precisions` no-op when key absent |

#### ODEData Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 350 | `num_states` | `initial_states.n` |
| 351 | `num_observables` | `observables.n` |
| 352 | `num_parameters` | `parameters.n` |
| 353 | `num_constants` | `constants.n` |

| # | Functionality |
|---|--------------|
| 354 | `sizes` returns SystemSizes with all counts |
| 355 | `beta` returns `precision(self._beta)` |
| 356 | `gamma` returns `precision(self._gamma)` |
| 357 | `mass` returns `_mass` |

#### `ODEData.from_BaseODE_initargs` — Factory method

| # | Functionality |
|---|--------------|
| 358 | Creates SystemValues for each component from defaults and overrides |
| 359 | Returns ODEData with all components |
| 360 | Handles None for optional arguments |

---

### `odesystems/SystemValues.py` [x]

#### `SystemValues.__init__` — Construction

| # | Functionality |
|---|--------------|
| 361 | Raises `TypeError` for non-numeric precision |
| 362 | None values_dict treated as empty dict |
| 363 | None defaults treated as empty dict |
| 364 | List/tuple values_dict expanded to `{k: 0.0}` dict |
| 365 | List/tuple defaults expanded to `{k: 0.0}` dict |
| 366 | Symbol keys converted to strings via `_convert_symbol_keys` |
| 367 | Merge order: defaults, values_dict, kwargs (kwargs win) |
| 368 | Initializes values_array, indices_dict, keys_by_index |
| 369 | Sets `n` to length of values_array |

#### `SystemValues.__repr__`

| # | Functionality |
|---|--------------|
| 370 | Uses "System Values" when name is None |
| 371 | Shows variable names when all values are 0.0 |
| 372 | Shows full dict when values are non-zero |

#### `SystemValues._convert_symbol_keys`

| # | Functionality |
|---|--------------|
| 373 | Converts Symbol keys to strings |
| 374 | Passes through string keys |
| 375 | Returns non-dict input unchanged |

#### `SystemValues.update_param_array_and_indices`

| # | Functionality |
|---|--------------|
| 376 | Creates values_array from values_dict with correct precision |
| 377 | Creates indices_dict and keys_by_index mappings |

#### `SystemValues.get_index_of_key`

| # | Functionality |
|---|--------------|
| 378 | Returns correct index for existing key |
| 379 | Raises `KeyError` for missing key when silent=False |
| 380 | Returns None for missing key when silent=True |
| 381 | Raises `TypeError` for non-string key |

#### `SystemValues.get_indices`

| # | Functionality |
|---|--------------|
| 382 | List of strings: returns array of indices |
| 383 | List of strings with silent=True: filters out None results |
| 384 | List of ints: returns array directly |
| 385 | Mixed list: raises `TypeError` |
| 386 | List with unsupported types: raises `TypeError` |
| 387 | Single string: returns single-element array |
| 388 | Single int: returns single-element array |
| 389 | Slice: returns range of indices |
| 390 | ndarray: returns as int32 |
| 391 | Other type: raises `TypeError` |
| 392 | Out-of-bounds index: raises `IndexError` |

#### `SystemValues.get_values`

| # | Functionality |
|---|--------------|
| 393 | Single index: returns scalar array |
| 394 | Multiple indices: returns array of values |

#### `SystemValues.set_values`

| # | Functionality |
|---|--------------|
| 395 | Single index, single value: updates correctly |
| 396 | Single index, multiple values: raises `ValueError` |
| 397 | Multiple indices, single scalar value: raises `ValueError` |
| 398 | Mismatched lengths: raises `ValueError` |
| 399 | Matching lengths: updates all values |

#### `SystemValues.update_from_dict`

| # | Functionality |
|---|--------------|
| 400 | Returns empty set for None/empty input |
| 401 | kwargs merged into values_dict |
| 402 | Unrecognised keys raise `KeyError` when silent=False |
| 403 | Unrecognised keys ignored when silent=True |
| 404 | Raises `TypeError` for non-numeric values |
| 405 | Updates both values_dict and values_array for recognised keys |
| 406 | Returns set of recognised keys |

#### `SystemValues` — Properties

| # | Functionality |
|---|--------------|
| 407 | `names` returns list of keys |
| 408 | `empty` returns True when n==0 |

#### `SystemValues.get_labels`

| # | Functionality |
|---|--------------|
| 409 | Returns labels for list/ndarray of indices |
| 410 | Raises `TypeError` for non-list/non-ndarray input |

#### `SystemValues.__getitem__` / `__setitem__`

| # | Functionality |
|---|--------------|
| 411 | `__getitem__` delegates to `get_values` |
| 412 | `__setitem__` delegates to `set_values` |

#### `SystemValues.add_entry` / `remove_entry`

| # | Functionality |
|---|--------------|
| 413 | `add_entry` raises `ValueError` for duplicate name |
| 414 | `add_entry` adds entry, updates array and n |
| 415 | `remove_entry` raises `KeyError` for missing name |
| 416 | `remove_entry` removes entry, updates array and n, returns value |

---

### `odesystems/baseODE.py` [x]

#### `ODECache` — Cache container

| # | Functionality |
|---|--------------|
| 417 | Construction with dxdt succeeds; optional fields default to -1 |

#### `BaseODE.__init__` — Construction

| # | Functionality |
|---|--------------|
| 418 | Creates ODEData from init args and sets up compile settings |
| 419 | Stores name |

#### `BaseODE.__repr__`

| # | Functionality |
|---|--------------|
| 420 | Uses "ODE System" when name is None |
| 421 | Uses provided name |

#### `BaseODE.update` — Parameter routing

| # | Functionality |
|---|--------------|
| 422 | Returns empty set for None/empty input |
| 423 | Merges updates_dict and kwargs |
| 424 | Delegates to `update_compile_settings(silent=True)` |
| 425 | Delegates to `set_constants(silent=True)` |
| 426 | Calls `update_precisions` on compile_settings |
| 427 | Combines recognised sets |
| 428 | Raises `KeyError` for unrecognised params when silent=False |
| 429 | No error when silent=True |

#### `BaseODE.set_constants`

| # | Functionality |
|---|--------------|
| 430 | Returns empty set for empty input |
| 431 | Updates constants that match values_dict keys |
| 432 | Raises `KeyError` for unrecognised when silent=False |
| 433 | Calls `update_compile_settings(constants=const, silent=True)` |

#### Forwarding Properties (table-driven)

| # | Property | Delegates to |
|---|----------|-------------|
| 434 | `parameters` | `compile_settings.parameters` |
| 435 | `states` | `compile_settings.initial_states` |
| 436 | `initial_values` | `compile_settings.initial_states` |
| 437 | `observables` | `compile_settings.observables` |
| 438 | `constants` | `compile_settings.constants` |
| 439 | `num_states` | `compile_settings.num_states` |
| 440 | `num_observables` | `compile_settings.num_observables` |
| 441 | `num_parameters` | `compile_settings.num_parameters` |
| 442 | `num_constants` | `compile_settings.num_constants` |
| 443 | `num_drivers` | `compile_settings.num_drivers` |
| 444 | `sizes` | `compile_settings.sizes` |

| # | Functionality |
|---|--------------|
| 445 | `evaluate_f` returns `get_cached_output("dxdt")` |
| 446 | `evaluate_observables` returns `get_cached_output("observables")` |

#### `BaseODE.config_hash`

| # | Functionality |
|---|--------------|
| 447 | Combines super().config_hash with constants values hash |
| 448 | Handles None constants |

#### `BaseODE.get_solver_helper`

| # | Functionality |
|---|--------------|
| 449 | Delegates to `get_cached_output(func_name)` |

---

### `odesystems/symbolic/symbolicODE.py` [x]

#### `create_ODE_system` — Convenience wrapper

| # | Functionality |
|---|--------------|
| 450 | Delegates to `SymbolicODE.create` with all args |

#### `SymbolicODE.__init__`

| # | Functionality |
|---|--------------|
| 451 | Uses `all_indexed_bases.all_symbols` when all_symbols is None |
| 452 | Computes fn_hash from equations when fn_hash is None |
| 453 | Uses fn_hash as name when name is None |
| 454 | Creates ODEFile with name and hash |
| 455 | Stores equations, indices, user_functions, driver_defaults |
| 456 | Calls super().__init__ with extracted values |

#### `SymbolicODE.create` — Class method factory

| # | Functionality |
|---|--------------|
| 457 | Creates ArrayInterpolator when drivers dict has "time" or "dt" |
| 458 | Registers and starts timing event |
| 459 | Calls `parse_input` with all args |
| 460 | Creates SymbolicODE instance from parsed components |
| 461 | Stops timing event |

#### SymbolicODE Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 462 | `jacobian_aux_count` | `_jacobian_aux_count` |
| 463 | `state_units` | `indices.states.units` |
| 464 | `parameter_units` | `indices.parameters.units` |
| 465 | `constant_units` | `indices.constants.units` |
| 466 | `observable_units` | `indices.observables.units` |
| 467 | `driver_units` | `indices.drivers.units` |

#### `SymbolicODE._get_jvp_exprs`

| # | Functionality |
|---|--------------|
| 468 | Caches result on first call |
| 469 | Returns cached result on subsequent calls |

#### `SymbolicODE.build`

| # | Functionality |
|---|--------------|
| 470 | Resets `_jacobian_aux_count` to None |
| 471 | Recomputes hash; updates gen_file if hash changed |
| 472 | Generates dxdt_code only when not cached |
| 473 | Imports and calls dxdt_factory |
| 474 | Generates observables code only when not cached |
| 475 | Returns ODECache with dxdt and observables |

#### `SymbolicODE.set_constants`

| # | Functionality |
|---|--------------|
| 476 | Updates indices.constants first |
| 477 | Delegates to super().set_constants |

#### `SymbolicODE.make_parameter`

| # | Functionality |
|---|--------------|
| 478 | Gets current constant value |
| 479 | Calls `indices.constant_to_parameter` |
| 480 | Removes from compile_settings.constants |
| 481 | Adds to compile_settings.parameters |
| 482 | Invalidates cache |

#### `SymbolicODE.make_constant`

| # | Functionality |
|---|--------------|
| 483 | Gets current parameter value |
| 484 | Calls `indices.parameter_to_constant` |
| 485 | Removes from compile_settings.parameters |
| 486 | Adds to compile_settings.constants |
| 487 | Invalidates cache |

#### `SymbolicODE.set_constant_value` / `set_parameter_value` / `set_initial_value`

| # | Functionality |
|---|--------------|
| 488 | `set_constant_value` delegates to `set_constants({name: value})` |
| 489 | `set_parameter_value` updates parameters and indices |
| 490 | `set_initial_value` updates initial_values and indices |

#### `SymbolicODE.get_constants_info` / `get_parameters_info` / `get_states_info`

| # | Functionality |
|---|--------------|
| 491 | Returns list of dicts with name, value, unit for each constant |
| 492 | Returns list of dicts with name, value, unit for each parameter |
| 493 | Returns list of dicts with name, value, unit for each state |

#### `SymbolicODE.constants_gui` / `states_gui`

| # | Functionality |
|---|--------------|
| 494 | `constants_gui` imports and calls `show_constants_editor` |
| 495 | `states_gui` imports and calls `show_states_editor` |

#### `SymbolicODE.get_solver_helper`

| # | Functionality |
|---|--------------|
| 496 | Updates solver parameters (beta, gamma, etc.) |
| 497 | Registers timing event if not already registered |
| 498 | Returns cached output if available |
| 499 | For `cached_aux_count`: triggers `prepare_jac` if aux_count is None |
| 500 | Checks gen_file cache; marks as skipped if cached |
| 501 | Generates code for each func_type (linear_operator, prepare_jac, etc.) |
| 502 | Raises `NotImplementedError` for unknown func_type |
| 503 | Adds beta/gamma/order kwargs for preconditioner-related types |
| 504 | Imports factory and calls with kwargs |
| 505 | For `prepare_jac`: sets `_jacobian_aux_count` from factory attr if needed |
| 506 | Stores result in cache and stops timing |

---

### `odesystems/symbolic/sym_utils.py` [x]

#### `topological_sort` — Kahn's algorithm

| # | Functionality |
|---|--------------|
| 507 | Accepts list of (symbol, expr) pairs |
| 508 | Accepts dict of symbol -> expr |
| 509 | Returns assignments in dependency order |
| 510 | Raises `ValueError` for circular dependencies |

#### `cse_and_stack` — CSE + topological sort

| # | Functionality |
|---|--------------|
| 511 | Uses "_cse" prefix when symbol is None |
| 512 | Continues numbering from highest existing prefix index |
| 513 | Applies SymPy CSE and topologically sorts combined result |

#### `hash_system_definition` — Deterministic system hash

| # | Functionality |
|---|--------------|
| 514 | Extracts equations from ParsedEquations via `.ordered` |
| 515 | Falls back to list conversion for raw tuples |
| 516 | Sorts equations by LHS symbol name (order-independent) |
| 517 | Normalizes whitespace |
| 518 | Appends sorted constant labels (keys only, not values) |
| 519 | Appends sorted observable labels |
| 520 | Returns SHA256 hex digest |
| 521 | Handles None constants and None observable_labels |

#### `render_constant_assignments` — Code generation helper

| # | Functionality |
|---|--------------|
| 522 | Generates assignment lines with correct indent |
| 523 | Returns empty string for empty constant_names |

#### `prune_unused_assignments` — Dead code elimination

| # | Functionality |
|---|--------------|
| 524 | Returns empty list for empty input |
| 525 | Detects output symbols by name convention (`prefix[`) |
| 526 | Uses explicit `output_symbols` when provided |
| 527 | Returns all expressions when no outputs detected |
| 528 | Prunes unreachable assignments by reverse traversal |
| 529 | Preserves relative order of kept assignments |

---

### `odesystems/symbolic/indexedbasemaps.py` [x]

#### `IndexedBaseMap.__init__`

| # | Functionality |
|---|--------------|
| 530 | Uses len(labels) when length=0 |
| 531 | Creates IndexedBase with shape and real |
| 532 | Builds index_map, ref_map, symbol_map from labels |
| 533 | None defaults: creates 0.0 defaults dict |
| 534 | Dict defaults: stores as passthrough |
| 535 | List defaults: validates length and creates dict |
| 536 | Raises `ValueError` when defaults length mismatches |
| 537 | None units: defaults all to "dimensionless" |
| 538 | Dict units: fills missing with "dimensionless" |
| 539 | List units: validates length and creates dict |
| 540 | Raises `ValueError` when units length mismatches |

#### `IndexedBaseMap.pop`

| # | Functionality |
|---|--------------|
| 541 | Removes symbol from ref_map, index_map, symbol_map |
| 542 | Removes from default_values/defaults when not passthrough |
| 543 | Removes from units |
| 544 | Rebuilds base with new shape and updates length |

#### `IndexedBaseMap.push`

| # | Functionality |
|---|--------------|
| 545 | Appends symbol to all maps at next index |
| 546 | Rebuilds base with incremented shape |
| 547 | Adds default value and unit when not passthrough |

#### `IndexedBaseMap.update_values`

| # | Functionality |
|---|--------------|
| 548 | No-op when `_passthrough_defaults` is True |
| 549 | Handles Symbol keys by looking up ref_map |
| 550 | Handles string keys by looking up symbol_map |
| 551 | Updates both default_values and defaults |
| 552 | Silently ignores unknown keys |

#### `IndexedBaseMap.set_passthrough_defaults`

| # | Functionality |
|---|--------------|
| 553 | Sets `_passthrough_defaults` to True |
| 554 | Replaces both default_values and defaults with provided dict |

#### `IndexedBases.__init__`

| # | Functionality |
|---|--------------|
| 555 | Stores all component IndexedBaseMaps |
| 556 | Builds `all_indices` from ref_maps (excludes constants) |

#### `IndexedBases.from_user_inputs`

| # | Functionality |
|---|--------------|
| 557 | Dict states: extracts names and defaults separately |
| 558 | Non-dict states: uses as name list, defaults=None |
| 559 | Same pattern for parameters and constants |
| 560 | Creates dxdt base with "d{state_name}" labels |
| 561 | Passes units to each IndexedBaseMap |

#### `IndexedBases.update_constants`

| # | Functionality |
|---|--------------|
| 562 | Delegates to `constants.update_values` |

#### IndexedBases Forwarding Properties (table-driven)

| # | Property | Delegates to |
|---|----------|-------------|
| 563 | `state_names` | `states.symbol_map.keys()` |
| 564 | `state_values` | `states.default_values` |
| 565 | `parameter_names` | `parameters.symbol_map.keys()` |
| 566 | `parameter_symbols` | `parameters.ref_map.keys()` |
| 567 | `parameter_values` | `parameters.default_values` |
| 568 | `constant_names` | `constants.symbol_map.keys()` |
| 569 | `constant_values` | `constants.default_values` |
| 570 | `observable_names` | `observables.symbol_map.keys()` |
| 571 | `observable_symbols` | `observables.ref_map.keys()` |
| 572 | `driver_names` | `drivers.symbol_map.keys()` |
| 573 | `dxdt_names` | `dxdt.symbol_map.keys()` |
| 574 | `all_arrayrefs` | combined ref_maps |
| 575 | `all_symbols` | combined symbol_maps |

#### `IndexedBases.__getitem__`

| # | Functionality |
|---|--------------|
| 576 | Returns `all_indices[item]` |

#### `IndexedBases._refresh_all_indices`

| # | Functionality |
|---|--------------|
| 577 | Rebuilds all_indices from current ref_maps |

#### `IndexedBases.constant_to_parameter`

| # | Functionality |
|---|--------------|
| 578 | Raises `KeyError` when name not in constants |
| 579 | Pops from constants and pushes to parameters with value and unit |
| 580 | Refreshes all_indices |

#### `IndexedBases.parameter_to_constant`

| # | Functionality |
|---|--------------|
| 581 | Raises `KeyError` when name not in parameters |
| 582 | Pops from parameters and pushes to constants with value and unit |
| 583 | Refreshes all_indices |

---

### `odesystems/symbolic/odefile.py` [x]

#### `ODEFile.__init__`

| # | Functionality |
|---|--------------|
| 584 | Creates system directory if not exists |
| 585 | Sets file_path to `system_dir / {name}.py` |
| 586 | Calls `_init_file` |

#### `ODEFile._init_file`

| # | Functionality |
|---|--------------|
| 587 | Creates new file with hash + header when cache invalid |
| 588 | Returns True when file was created |
| 589 | Returns False when cache is valid (no-op) |

#### `ODEFile.cached_file_valid`

| # | Functionality |
|---|--------------|
| 590 | Returns False when file doesn't exist |
| 591 | Returns True when stored hash matches fn_hash |
| 592 | Returns False when stored hash differs |

#### `ODEFile.function_is_cached`

| # | Functionality |
|---|--------------|
| 593 | Returns False when file doesn't exist |
| 594 | Returns False when function def not found |
| 595 | Returns True when function def found with return statement at correct indent |
| 596 | Returns False when function def found but no return statement (incomplete) |

#### `ODEFile._import_function`

| # | Functionality |
|---|--------------|
| 597 | Imports function from generated module file |

#### `ODEFile.import_function`

| # | Functionality |
|---|--------------|
| 598 | Reinitializes file if cache invalid |
| 599 | Returns (function, True) when cached |
| 600 | Prints one-time cache notification on first cached hit |
| 601 | Raises `ValueError` when not cached and code_lines is None |
| 602 | Calls `add_function` and imports when not cached |
| 603 | Returns (function, False) when generated |

#### `ODEFile.add_function`

| # | Functionality |
|---|--------------|
| 604 | Appends printed_code to cache file |

---

### `odesystems/symbolic/parsing/auxiliary_caching.py` [x]

#### `CacheGroup` — Frozen attrs container

| # | Functionality |
|---|--------------|
| 605 | Construction stores seed, leaves, removal, prepare, saved, fill_cost |
| 606 | `leaves` and `removal` and `prepare` converted to tuples |

#### `CacheSelection` — Frozen attrs container

| # | Functionality |
|---|--------------|
| 607 | Construction stores all fields; tuple converters on sequence fields |

#### `_reachable_leaves` — DFS leaf discovery

| # | Functionality |
|---|--------------|
| 608 | Returns symbols reachable from seed with jvp_usage > 0 |
| 609 | Returns symbols reachable from seed with total_cost >= min_internal_cost |
| 610 | Always includes seed itself |
| 611 | Avoids revisiting nodes |

#### `_prepare_nodes_for_leaves` — Dependency closure

| # | Functionality |
|---|--------------|
| 612 | Returns all transitive dependencies of leaves |

#### `_simulate_cached_leaves` — Simulation

| # | Functionality |
|---|--------------|
| 613 | Skips `_cse` prefixed nodes during removal |
| 614 | Removes nodes where all references consumed |
| 615 | Returns None when removal would leave orphaned children |
| 616 | Returns (saved, removal, prepare, fill_cost) when valid |

#### `_collect_candidates` — Candidate enumeration

| # | Functionality |
|---|--------------|
| 617 | Returns empty list when slot_limit <= 0 |
| 618 | Skips seeds with zero jvp_closure_usage |
| 619 | Generates combinations up to slot_limit size |
| 620 | Deduplicates by (leaves, removal) key |
| 621 | Prefers higher saved, then lower fill_cost |
| 622 | Returns sorted by savings descending |

#### `_evaluate_leaves` — Memoized evaluation

| # | Functionality |
|---|--------------|
| 623 | Returns cached result from memo |
| 624 | Returns (0, set(), set(), 0) for empty leaves_key |
| 625 | Caches simulation result |
| 626 | Deduplicates by removal_key |

#### `_search_group_combinations` — Optimal combination search

| # | Functionality |
|---|--------------|
| 627 | Returns empty selection when no candidates or slot_limit <= 0 |
| 628 | Branch-and-bound search over group combinations |
| 629 | Selects best by: saved > improvement threshold, fewer leaves, lower fill_cost |
| 630 | Returns CacheSelection with runtime_nodes = non_jvp_order minus removal |

#### `plan_auxiliary_cache` — Entry point

| # | Functionality |
|---|--------------|
| 631 | Collects candidates and searches combinations |
| 632 | Calls `equations.update_cache_selection` to persist |
| 633 | Returns CacheSelection |

---

### `odesystems/symbolic/parsing/jvp_equations.py` [x]

#### `JVPEquations.__init__` / `__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 634 | Separates assignments into non-JVP and JVP terms by `jvp[` prefix |
| 635 | Default cache_slot_limit = 2 * len(jvp_terms) when max_cached_terms is None |
| 636 | Uses max_cached_terms when provided |
| 637 | Calls `_initialise_expression_metadata` |

#### `JVPEquations._initialise_expression_metadata`

| # | Functionality |
|---|--------------|
| 638 | Builds forward dependency graph (dependencies) |
| 639 | Builds reverse dependency graph (dependents) |
| 640 | Computes per-assignment ops_cost via sp.count_ops |
| 641 | Computes jvp_usage (direct usage in JVP terms) |
| 642 | Computes jvp_closure_usage (transitive closure) |
| 643 | Computes dependency_levels (BFS from each symbol) |
| 644 | Computes total_ops_cost (recursive cumulative) |
| 645 | Computes reference_counts = dependents count + jvp_usage |
| 646 | Builds order_index for evaluation order |

#### JVPEquations Properties (table-driven)

| # | Property | Delegates to |
|---|----------|-------------|
| 647 | `ordered_assignments` | `_ordered_assignments` |
| 648 | `non_jvp_order` | `_non_jvp_order` |
| 649 | `non_jvp_exprs` | `_non_jvp_exprs` |
| 650 | `jvp_terms` | `_jvp_terms` |
| 651 | `dependencies` | `_dependencies` |
| 652 | `dependents` | `_dependents` |
| 653 | `ops_cost` | `_ops_cost` |
| 654 | `jvp_usage` | `_jvp_usage` |
| 655 | `jvp_closure_usage` | `_jvp_closure_usage` |
| 656 | `cache_slot_limit` | `_cache_slot_limit` |
| 657 | `reference_counts` | `_reference_counts` |
| 658 | `order_index` | `_order_index` |
| 659 | `dependency_levels` | `_dependency_levels` |
| 660 | `total_ops_cost` | `_total_ops_cost` |

#### `JVPEquations.update_cache_selection`

| # | Functionality |
|---|--------------|
| 661 | Stores selection in `_cache_selection` |

#### `JVPEquations.ensure_cache_selection`

| # | Functionality |
|---|--------------|
| 662 | Calls `plan_auxiliary_cache` when _cache_selection is None |
| 663 | No-op when _cache_selection already set |

#### `JVPEquations.cache_selection` property

| # | Functionality |
|---|--------------|
| 664 | Calls `ensure_cache_selection` and returns `_cache_selection` |

#### `JVPEquations.cached_partition`

| # | Functionality |
|---|--------------|
| 665 | Partitions non-JVP assignments into cached, runtime, prepare lists |
| 666 | Assignments can appear in both prepare and cached lists |


### ODE Systems

| # | Source File | Inventory |
|---|------------|-----------|
| 9 | `odesystems/ODEData.py` | [x] |
| 10 | `odesystems/SystemValues.py` | [x] |
| 11 | `odesystems/baseODE.py` | [x] |
| 12 | `odesystems/__init__.py` | [x] |


### Symbolic ODE & Parsing

| # | Source File | Inventory |
|---|------------|-----------|
| 13 | `odesystems/symbolic/symbolicODE.py` | [x] |
| 14 | `odesystems/symbolic/sym_utils.py` | [x] |
| 15 | `odesystems/symbolic/indexedbasemaps.py` | [x] |
| 16 | `odesystems/symbolic/odefile.py` | [x] |
| 17 | `odesystems/symbolic/__init__.py` | [x] |
| 18 | `odesystems/symbolic/parsing/parser.py` | [x] |
| 19 | `odesystems/symbolic/parsing/auxiliary_caching.py` | [x] |
| 20 | `odesystems/symbolic/parsing/jvp_equations.py` | [x] |
| 21 | `odesystems/symbolic/parsing/cellml.py` | [x] |
| 22 | `odesystems/symbolic/parsing/cellml_cache.py` | [x] |
| 23 | `odesystems/symbolic/parsing/__init__.py` | [x] |


### Codegen

| # | Source File | Inventory |
|---|------------|-----------|
| 24 | `codegen/numba_cuda_printer.py` | [x] |
| 25 | `codegen/dxdt.py` | [x] |
| 26 | `codegen/time_derivative.py` | [x] |
| 27 | `codegen/jacobian.py` | [x] |
| 28 | `codegen/linear_operators.py` | [x] |
| 29 | `codegen/preconditioners.py` | [x] |
| 30 | `codegen/nonlinear_residuals.py` | [x] |
| 31 | `codegen/_stage_utils.py` | [x] |
| 32 | `codegen/__init__.py` | [x] |


## `odesystems/symbolic/parsing/parser.py`

### `_detect_input_type`

| # | Functionality |
|---|--------------|
| 1 | Returns `"function"` when dxdt is callable and not str/list/tuple |
| 2 | Returns `"string"` when dxdt is a str |
| 3 | Returns `"string"` when first element of iterable is a str |
| 4 | Returns `"sympy"` when first element is `sp.Expr` or `sp.Equality` |
| 5 | Returns `"sympy"` when first element is a 2-tuple of (Symbol/Derivative, Expr) |
| 6 | Raises `TypeError` when dxdt is None |
| 7 | Raises `TypeError` when dxdt is non-iterable non-string non-callable |
| 8 | Raises `ValueError` when dxdt is an empty iterable |
| 9 | Raises `TypeError` when first element is unrecognised type |

### `_normalize_sympy_equations`

| # | Functionality |
|---|--------------|
| 10 | Converts `sp.Equality` with Symbol LHS to (Symbol, Expr) tuple |
| 11 | Converts `sp.Equality` with Derivative LHS to (dX Symbol, Expr) tuple |
| 12 | Raises `ValueError` when Derivative has no arguments |
| 13 | Raises `ValueError` when Derivative arg is not Symbol |
| 14 | Raises `ValueError` when Equality LHS is neither Symbol nor Derivative |
| 15 | Converts tuple with Symbol LHS and Expr RHS |
| 16 | Converts tuple with Derivative LHS to (dX Symbol, Expr) |
| 17 | Raises `TypeError` when tuple has wrong length |
| 18 | Raises `TypeError` when tuple LHS is not Symbol/Derivative |
| 19 | Raises `TypeError` when tuple RHS is not sp.Expr |
| 20 | Raises `TypeError` for bare sp.Expr (no LHS) |
| 21 | Raises `TypeError` for invalid element type |
| 22 | Raises `TypeError` when equations not iterable |

### `ParsedEquations` (frozen attrs class)

| # | Functionality |
|---|--------------|
| 23 | `__iter__` returns iterator over `ordered` |
| 24 | `__len__` returns length of `ordered` |
| 25 | `__getitem__` returns equation at index from `ordered` |
| 26 | `copy` returns dict mapping lhs->rhs |
| 27 | `to_equation_list` returns mutable list of ordered equations |
| 28 | `state_symbols` property forwards `_state_symbols` |
| 29 | `observable_symbols` property forwards `_observable_symbols` |
| 30 | `auxiliary_symbols` property forwards `_auxiliary_symbols` |
| 31 | `non_observable_equations` filters out equations with observable LHS |
| 32 | `dxdt_equations` returns tuple of non-observable equations |
| 33 | `observable_system` returns all ordered equations |
| 34 | `from_equations` classmethod: partitions equations into state/observable/auxiliary by index_map |
| 35 | `from_equations` handles dict input (converts to items) |
| 36 | `from_equations` handles iterable input |

### `_sanitise_input_math`

| # | Functionality |
|---|--------------|
| 37 | Delegates to `_replace_if` for ternary-to-Piecewise conversion |

### `_replace_if`

| # | Functionality |
|---|--------------|
| 38 | Converts `X if COND else Y` to `Piecewise((X, COND), (Y, True))` |
| 39 | Recursively handles nested ternaries |
| 40 | Returns unchanged string when no ternary found |

### `_normalise_indexed_tokens`

| # | Functionality |
|---|--------------|
| 41 | Rewrites `name[index]` to `nameindex` for integer literals |
| 42 | Leaves non-matching tokens unchanged |

### `_rename_user_calls`

| # | Functionality |
|---|--------------|
| 43 | Returns original lines and empty dict when no user_functions |
| 44 | Appends `_` suffix to function call tokens in lines |
| 45 | Returns mapping from original names to suffixed names |

### `_build_sympy_user_functions`

| # | Functionality |
|---|--------------|
| 46 | Creates dynamic Function subclass with `fdiff` for device functions |
| 47 | Creates dynamic Function subclass with `fdiff` when derivative helper provided |
| 48 | `fdiff` uses derivative print name when available |
| 49 | `fdiff` falls back to `d_<orig_name>` when no derivative helper |
| 50 | Creates plain `sp.Function` for non-device functions without derivatives |
| 51 | Returns parse_locals, alias_map, and is_device_map |
| 52 | Handles None user_functions (empty iteration) |

### `_inline_nondevice_calls`

| # | Functionality |
|---|--------------|
| 53 | Returns expr unchanged when no user_functions |
| 54 | Inlines non-device user function calls that return SymPy expressions |
| 55 | Skips device functions (leaves symbolic) |
| 56 | Skips functions not found in user_functions |
| 57 | Keeps symbolic call when inline evaluation fails |
| 58 | Keeps symbolic call when return value is not SymPy type |

### `_process_calls`

| # | Functionality |
|---|--------------|
| 59 | Detects function call tokens via regex in equation lines |
| 60 | Resolves user function names to callables |
| 61 | Resolves known SymPy function names |
| 62 | Raises `ValueError` for unknown function names |

### `_process_parameters`

| # | Functionality |
|---|--------------|
| 63 | Delegates to `IndexedBases.from_user_inputs` with all arguments |

### `_lhs_pass`

| # | Functionality |
|---|--------------|
| 64 | Detects `d(name, t)` function notation as derivative of known state |
| 65 | `d(name, t)` where name is observable: converts observable to state with warning |
| 66 | `d(name, t)` where name is unknown + strict: raises ValueError |
| 67 | `d(name, t)` where name is unknown + non-strict: infers new state |
| 68 | Detects `dX` prefix as derivative when X is a known state |
| 69 | `dX` prefix where X is observable: converts to state with warning |
| 70 | `dX` prefix where X is unknown + non-strict + no initial states: infers new state |
| 71 | `dX` prefix where X is unknown + strict or had initial states: treated as auxiliary |
| 72 | `dX` prefix where X is unknown observable name: tracked as assigned observable |
| 73 | Direct state assignment raises ValueError |
| 74 | Assignment to parameter/constant/driver raises ValueError |
| 75 | Unrecognised LHS not in observables: treated as anonymous auxiliary |
| 76 | LHS matching observable name: tracked as assigned observable |
| 77 | Missing observable assignments raise ValueError |
| 78 | States with no derivative: converted to observables with warning |
| 79 | State-to-observable conversion raises ValueError if already observable |

### `_lhs_pass_sympy`

| # | Functionality |
|---|--------------|
| 80 | Mirrors `_lhs_pass` logic for SymPy equation objects |
| 81 | `d`-prefix detection using SymPy Symbol names |
| 82 | Observable-to-state conversion with warning |
| 83 | Non-strict state inference (no initial states) |
| 84 | Strict mode: d-prefix unknown treated as auxiliary |
| 85 | Direct state assignment raises ValueError |
| 86 | Immutable input assignment raises ValueError |
| 87 | Anonymous auxiliary detection |
| 88 | Missing observable validation |
| 89 | Underived state-to-observable conversion with warning |

### `_process_user_functions_for_rhs`

| # | Functionality |
|---|--------------|
| 90 | Builds SymPy wrappers via `_build_sympy_user_functions` |
| 91 | Returns callable mapping with original user function names |
| 92 | Returns empty dict when user_funcs is None |

### `_rhs_pass_sympy`

| # | Functionality |
|---|--------------|
| 93 | Validates all RHS free_symbols are declared (strict mode) |
| 94 | Raises ValueError for undeclared symbols in strict mode |
| 95 | Infers undeclared symbols as parameters in non-strict mode |
| 96 | Returns validated equations, funcs, and new_symbols |

### `_rhs_pass`

| # | Functionality |
|---|--------------|
| 97 | Calls `_process_calls` to validate function references |
| 98 | Renames user function calls via `_rename_user_calls` |
| 99 | Builds SymPy function wrappers via `_build_sympy_user_functions` |
| 100 | Parses RHS with transforms in strict mode |
| 101 | Raises ValueError (from NameError/TypeError) for undefined symbols in strict mode |
| 102 | Parses RHS without transforms in non-strict mode, infers new symbols |
| 103 | Inlines non-device function calls via `_inline_nondevice_calls` |
| 104 | Uses `raw_lines` in error messages when provided |
| 105 | Falls back to `lines` for error messages when raw_lines is None |
| 106 | Raises ValueError for unresolved symbols after all passes |

### `parse_input`

| # | Functionality |
|---|--------------|
| 107 | Defaults states to `{}` when None |
| 108 | Raises ValueError when states=None and strict=True |
| 109 | Defaults observables/parameters/constants/drivers to empty |
| 110 | Extracts driver names from dict, filtering out setting keys |
| 111 | Raises ValueError when driver dict has no driver symbols |
| 112 | Calls `_process_parameters` to build IndexedBases |
| 113 | Detects input type via `_detect_input_type` |
| 114 | String path: splits multiline string into lines |
| 115 | String path: accepts list/tuple of strings |
| 116 | String path: raises ValueError for other string-like types |
| 117 | String path: normalises indexed tokens |
| 118 | String path: runs `_lhs_pass` and `_rhs_pass` |
| 119 | SymPy path: normalises equations, substitutes canonical symbols |
| 120 | SymPy path: runs `_lhs_pass_sympy` and `_rhs_pass_sympy` |
| 121 | SymPy path: second substitution pass after LHS changes |
| 122 | Function path: delegates to `parse_function_input` |
| 123 | Raises RuntimeError for invalid input_type |
| 124 | Pushes inferred parameters into index_map |
| 125 | Sets driver passthrough defaults when driver_dict provided |
| 126 | Exposes user_functions in all_symbols dict |
| 127 | Exposes user_function_derivatives in all_symbols |
| 128 | Builds `__function_aliases__` for string pathway with renaming |
| 129 | Constructs `ParsedEquations` via `from_equations` |
| 130 | Computes system hash via `hash_system_definition` |
| 131 | Returns 5-tuple: (index_map, all_symbols, funcs, parsed_equations, fn_hash) |

---

## `odesystems/symbolic/parsing/function_inspector.py`

### `FunctionInspection.__init__`

| # | Functionality |
|---|--------------|
| 132 | Stores all 9 parameters as instance attributes |

### `_OdeAstVisitor.__init__`

| # | Functionality |
|---|--------------|
| 133 | Initialises state_param, constant_params, and empty collection attributes |

### `_OdeAstVisitor.visit_Subscript`

| # | Functionality |
|---|--------------|
| 134 | Records int subscript access on state param to state_accesses |
| 135 | Records string subscript access on state param to state_accesses |
| 136 | Records int subscript access on constant param to constant_accesses |
| 137 | Records string subscript access on constant param to constant_accesses |
| 138 | Handles `ast.Index` wrapper for Python 3.8 compat |
| 139 | Records `ast.Name` slice as "name" pattern type |
| 140 | Records complex slice expression as "expr" pattern type |
| 141 | Ignores subscripts on non-state/non-constant bases |

### `_OdeAstVisitor.visit_Attribute`

| # | Functionality |
|---|--------------|
| 142 | Records attribute access on state param to state_accesses |
| 143 | Records attribute access on constant param to constant_accesses |
| 144 | Ignores attributes on non-state/non-constant bases |

### `_OdeAstVisitor.visit_Assign`

| # | Functionality |
|---|--------------|
| 145 | Records single Name target assignment |
| 146 | Records Tuple target assignments (unpacking) |

### `_OdeAstVisitor.visit_Return`

| # | Functionality |
|---|--------------|
| 147 | Appends Return node to return_nodes list |

### `_OdeAstVisitor.visit_Call`

| # | Functionality |
|---|--------------|
| 148 | Extracts function name via `_call_name` and adds to function_calls |
| 149 | Skips when `_call_name` returns None |

### `_call_name`

| # | Functionality |
|---|--------------|
| 150 | Returns function name for `ast.Name` func node |
| 151 | Returns `module.attr` for `ast.Attribute` func node |
| 152 | Returns None for unsupported func node types |

### `_resolve_func_name`

| # | Functionality |
|---|--------------|
| 153 | Strips known module prefix (math, np, numpy, cmath) |
| 154 | Returns name unchanged when no module prefix |

### `AstToSympyConverter.__init__`

| # | Functionality |
|---|--------------|
| 155 | Stores symbol_map |

### `AstToSympyConverter.convert`

| # | Functionality |
|---|--------------|
| 156 | Dispatches to `_convert_constant` for ast.Constant |
| 157 | Dispatches to `_convert_name` for ast.Name |
| 158 | Dispatches to `_convert_binop` for ast.BinOp |
| 159 | Dispatches to `_convert_unaryop` for ast.UnaryOp |
| 160 | Dispatches to `_convert_call` for ast.Call |
| 161 | Dispatches to `_convert_subscript` for ast.Subscript |
| 162 | Dispatches to `_convert_attribute` for ast.Attribute |
| 163 | Dispatches to `_convert_compare` for ast.Compare |
| 164 | Dispatches to `_convert_ifexp` for ast.IfExp |
| 165 | Dispatches to `_convert_boolop` for ast.BoolOp |
| 166 | Raises NotImplementedError for ast.Tuple |
| 167 | Raises NotImplementedError for ast.List |
| 168 | Raises NotImplementedError for unsupported node types |

### `AstToSympyConverter._convert_constant`

| # | Functionality |
|---|--------------|
| 169 | Converts int to sp.Integer |
| 170 | Converts float to sp.Float |
| 171 | Converts bool to sp.true/sp.false |
| 172 | Raises NotImplementedError for unsupported constant types |

### `AstToSympyConverter._convert_name`

| # | Functionality |
|---|--------------|
| 173 | Returns symbol from symbol_map when present |
| 174 | Creates new real Symbol and caches in symbol_map when absent |

### `AstToSympyConverter._convert_binop`

| # | Functionality |
|---|--------------|
| 175 | Handles Add, Sub, Mult, Div, FloorDiv, Pow, Mod operations |
| 176 | Raises NotImplementedError for unsupported binary ops |

### `AstToSympyConverter._convert_unaryop`

| # | Functionality |
|---|--------------|
| 177 | Handles USub (negation), UAdd (identity), Not |
| 178 | Raises NotImplementedError for unsupported unary ops |

### `AstToSympyConverter._convert_call`

| # | Functionality |
|---|--------------|
| 179 | Raises NotImplementedError for unnamed function calls |
| 180 | Resolves module-qualified names via `_resolve_func_name` |
| 181 | Raises NotImplementedError for unknown function names |
| 182 | Converts known function call with SymPy equivalent |

### `AstToSympyConverter._convert_subscript`

| # | Functionality |
|---|--------------|
| 183 | Looks up `base[key]` or `base['key']` in symbol_map |
| 184 | Handles ast.Index wrapper for Python 3.8 compat |
| 185 | Raises NotImplementedError for non-constant subscripts |
| 186 | Raises NotImplementedError when lookup not in symbol_map |
| 187 | Raises NotImplementedError for complex subscript targets |

### `AstToSympyConverter._convert_attribute`

| # | Functionality |
|---|--------------|
| 188 | Looks up `base.attr` in symbol_map |
| 189 | Raises NotImplementedError when lookup not in symbol_map |
| 190 | Raises NotImplementedError for complex attribute targets |

### `AstToSympyConverter._convert_compare`

| # | Functionality |
|---|--------------|
| 191 | Handles chained comparisons with And |
| 192 | Single comparison returns relation directly |

### `AstToSympyConverter._comparison_op` (static)

| # | Functionality |
|---|--------------|
| 193 | Maps Gt, GtE, Lt, LtE, Eq, NotEq to SymPy relational operators |
| 194 | Raises NotImplementedError for unsupported comparison ops |

### `AstToSympyConverter._convert_ifexp`

| # | Functionality |
|---|--------------|
| 195 | Converts ternary to `sp.Piecewise((body, test), (orelse, True))` |

### `AstToSympyConverter._convert_boolop`

| # | Functionality |
|---|--------------|
| 196 | Converts `and` to `sp.And` |
| 197 | Converts `or` to `sp.Or` |
| 198 | Raises NotImplementedError for unsupported bool ops |

### `inspect_ode_function`

| # | Functionality |
|---|--------------|
| 199 | Raises TypeError when func is not callable |
| 200 | Raises TypeError for lambda functions |
| 201 | Raises TypeError for builtins without inspectable source |
| 202 | Parses source and finds FunctionDef via ast.walk |
| 203 | Raises ValueError when no FunctionDef found |
| 204 | Raises ValueError when fewer than 2 parameters |
| 205 | Warns when first param is not 't' |
| 206 | Warns when second param is not in conventional names |
| 207 | Visits function body with `_OdeAstVisitor` |
| 208 | Raises ValueError when no return statement found |
| 209 | Raises ValueError when multiple return statements found |
| 210 | Validates access consistency per base variable |
| 211 | Returns populated FunctionInspection |

### `_validate_access_consistency`

| # | Functionality |
|---|--------------|
| 212 | Raises ValueError when both int and string patterns on same base |
| 213 | Passes when single pattern type (discarding expr and name) |

---

## `odesystems/symbolic/parsing/function_parser.py`

### `parse_function_input`

| # | Functionality |
|---|--------------|
| 214 | Defaults observables to empty list when None |
| 215 | Inspects function via `inspect_ode_function` |
| 216 | Builds symbol map via `_build_symbol_map` |
| 217 | Unpacks return value via `_unpack_return` |
| 218 | Raises ValueError when return element count != state count |
| 219 | Skips auxiliary assignments for observables, dxdt, states, constant params, state param |
| 220 | Skips auxiliary assignments that are direct access aliases |
| 221 | Converts remaining local assignments to auxiliary equations |
| 222 | Converts observable assignments using index_map symbol lookup |
| 223 | Dict return: maps keys to state derivative equations |
| 224 | Dict return: raises ValueError for non-string-literal keys |
| 225 | Dict return: raises ValueError for non-state keys |
| 226 | List/tuple return: positional mapping to dxdt equations |
| 227 | Returns (equation_map, empty funcs, empty new_params) |

### `_build_symbol_map`

| # | Functionality |
|---|--------------|
| 228 | Maps time parameter to TIME_SYMBOL |
| 229 | Maps integer-indexed state accesses to state symbols |
| 230 | Maps string-indexed state accesses to state symbols |
| 231 | Maps attribute state accesses to state symbols |
| 232 | Maps integer-indexed constant/parameter accesses |
| 233 | Maps string-indexed constant/parameter accesses |
| 234 | Maps attribute constant/parameter accesses |
| 235 | Resolves assignment aliases to state/constant symbols |
| 236 | Does NOT add dxdt symbols to symbol map (avoids circular refs) |
| 237 | Maps observable symbols from index_map |

### `_resolve_alias`

| # | Functionality |
|---|--------------|
| 238 | Returns symbol for subscript access on known base |
| 239 | Returns symbol for attribute access on known base |
| 240 | Returns None for non-matching nodes |

### `_is_access_alias`

| # | Functionality |
|---|--------------|
| 241 | Returns True for subscript on state_param or constant_params |
| 242 | Returns True for attribute on state_param or constant_params |
| 243 | Returns False otherwise |

### `_unpack_return`

| # | Functionality |
|---|--------------|
| 244 | Unpacks ast.List/ast.Tuple into list of expressions |
| 245 | Unpacks ast.Dict values into list of expressions |
| 246 | Wraps single expression as single-element list |
| 247 | Inlines local assignment when return element is Name matching an assignment |
| 248 | Defaults assignments to empty dict when None |

---

## `odesystems/symbolic/codegen/numba_cuda_printer.py`

### `CUDAPrinter.__init__`

| # | Functionality |
|---|--------------|
| 249 | Initialises symbol_map, cuda_functions, func_aliases |
| 250 | Extracts `__function_aliases__` from symbol_map when present |
| 251 | Initialises `_in_index` and `_in_pow` context flags to False |

### `CUDAPrinter.doprint`

| # | Functionality |
|---|--------------|
| 252 | Forces outer assignment for Piecewise when assign_to provided |
| 253 | Delegates to super().doprint for non-Piecewise |
| 254 | Applies `_replace_powers_with_multiplication` post-processing |

### `CUDAPrinter._print_Symbol`

| # | Functionality |
|---|--------------|
| 255 | Returns array-substituted print when symbol in symbol_map |
| 256 | Falls back to default Symbol printing |

### `CUDAPrinter._print_Indexed`

| # | Functionality |
|---|--------------|
| 257 | Sets `_in_index=True` while printing indices |
| 258 | Prints `base[indices]` format |

### `CUDAPrinter._print_Integer`

| # | Functionality |
|---|--------------|
| 259 | Returns unwrapped integer when `_in_index` or `_in_pow` is True |
| 260 | Returns `precision(N)` wrapped integer otherwise |

### `CUDAPrinter._print_Pow`

| # | Functionality |
|---|--------------|
| 261 | Parenthesizes base for compound expressions |
| 262 | Sets `_in_pow=True` while printing exponent |
| 263 | Returns `base**exponent` format |

### `CUDAPrinter._print_Piecewise`

| # | Functionality |
|---|--------------|
| 264 | Builds nested ternary from Piecewise pieces in reverse |
| 265 | Last piece used as fallback expression |

### `CUDAPrinter._replace_powers_with_multiplication`

| # | Functionality |
|---|--------------|
| 266 | Delegates to `_replace_square_powers` then `_replace_cube_powers` |

### `CUDAPrinter._replace_square_powers`

| # | Functionality |
|---|--------------|
| 267 | Replaces `x**2` with `x*x` for simple identifiers |
| 268 | Replaces `(expr)**2` with `(expr)*(expr)` for parenthesized expressions |
| 269 | Handles `x**2.0` variant |

### `CUDAPrinter._replace_cube_powers`

| # | Functionality |
|---|--------------|
| 270 | Replaces `x**3` with `x*x*x` for simple identifiers |
| 271 | Replaces `(expr)**3` with `(expr)*(expr)*(expr)` for parenthesized expressions |
| 272 | Handles `x**3.0` variant |

### `CUDAPrinter._print_Function`

| # | Functionality |
|---|--------------|
| 273 | Maps CUDA-known functions to `math.*` equivalents |
| 274 | Maps aliased user functions to original names |
| 275 | Prints derivative functions (`d_*`) as-is |
| 276 | Falls back to plain function name for unknown functions |

### `CUDAPrinter._print_Float`

| # | Functionality |
|---|--------------|
| 277 | Returns unwrapped float for 2.0/3.0 when in pow context |
| 278 | Returns `precision(value)` wrapped float otherwise |

### `CUDAPrinter._print_Rational`

| # | Functionality |
|---|--------------|
| 279 | Returns `precision(p/q)` wrapped rational |

### `print_cuda`

| # | Functionality |
|---|--------------|
| 280 | Creates CUDAPrinter and prints single expression |

### `print_cuda_multiple`

| # | Functionality |
|---|--------------|
| 281 | Creates CUDAPrinter and prints each (assign_to, expr) pair |

---

## `odesystems/symbolic/codegen/dxdt.py`

### `generate_dxdt_lines`

| # | Functionality |
|---|--------------|
| 282 | Extracts non-observable equations from ParsedEquations |
| 283 | Applies CSE when cse=True |
| 284 | Applies topological sort when cse=False |
| 285 | Filters out observable symbols when index_map provided |
| 286 | Prunes unused assignments when index_map provided |
| 287 | Uses index_map.all_arrayrefs as symbol_map |
| 288 | Returns `["pass"]` when no lines generated |

### `generate_observables_lines`

| # | Functionality |
|---|--------------|
| 289 | Returns `["pass"]` early when no observables in index_map |
| 290 | Applies CSE or topological sort |
| 291 | Substitutes dxdt symbols with numbered `dxout_` symbols |
| 292 | Substitutes arrayrefs into equations |
| 293 | Prunes unused assignments for observables |
| 294 | Returns `["pass"]` when no lines generated |

### `generate_dxdt_fac_code`

| # | Functionality |
|---|--------------|
| 295 | Generates dxdt lines via `generate_dxdt_lines` |
| 296 | Renders constant assignments block |
| 297 | Formats DXDT_TEMPLATE with func_name, const_lines, body |
| 298 | Logs timing via default_timelogger |

### `generate_observables_fac_code`

| # | Functionality |
|---|--------------|
| 299 | Generates observable lines via `generate_observables_lines` |
| 300 | Renders constant assignments block |
| 301 | Formats OBSERVABLES_TEMPLATE with func_name, const_lines, body |
| 302 | Logs timing via default_timelogger |

---

## `odesystems/symbolic/codegen/time_derivative.py`

### `_build_time_derivative_assignments`

| # | Functionality |
|---|--------------|
| 303 | Topologically sorts non-observable equations |
| 304 | Creates driver_dt IndexedBase when drivers present |
| 305 | Computes direct time derivative via `sp.diff(rhs, TIME_SYMBOL)` per equation |
| 306 | Computes driver partial contribution when driver in free_symbols |
| 307 | Computes chain rule term from previously processed auxiliaries |
| 308 | Creates `time_<lhs>` derivative symbols |
| 309 | Appends output mapping `time_rhs[i]` for each dxdt symbol |
| 310 | Returns (assignments, final_symbol_map) |

### `generate_time_derivative_lines`

| # | Functionality |
|---|--------------|
| 311 | Builds assignments via `_build_time_derivative_assignments` |
| 312 | Applies CSE or topological sort |
| 313 | Prunes unused assignments for `time_rhs` outputs |
| 314 | Prints CUDA lines with combined symbol maps |
| 315 | Returns `["pass"]` when no lines |

### `generate_time_derivative_fac_code`

| # | Functionality |
|---|--------------|
| 316 | Generates time derivative lines |
| 317 | Renders constant assignments block |
| 318 | Formats TIME_DERIVATIVE_TEMPLATE |
| 319 | Logs timing |

---

## `odesystems/symbolic/codegen/jacobian.py`

### `get_cache_key`

| # | Functionality |
|---|--------------|
| 320 | Converts dict equations to tuple of items |
| 321 | Converts iterable equations to tuple of tuples |
| 322 | Returns 4-tuple of (eq_tuple, input_tuple, output_tuple, cse) |

### `generate_jacobian`

| # | Functionality |
|---|--------------|
| 323 | Returns cached Jacobian when available and use_cache=True |
| 324 | Topologically sorts equations |
| 325 | Separates auxiliary and output equations |
| 326 | Computes chain-rule gradients for auxiliary equations |
| 327 | Raises ValueError for topological order violation |
| 328 | Computes Jacobian rows for output equations with chain rule |
| 329 | Caches result: adds to existing cache entry or creates new |
| 330 | Skips cache when use_cache=False |

### `generate_analytical_jvp`

| # | Functionality |
|---|--------------|
| 331 | Substitutes observable symbols with numbered auxiliaries |
| 332 | Returns cached JVP when available |
| 333 | Constructs ParsedEquations for substituted system |
| 334 | Generates Jacobian via `generate_jacobian` |
| 335 | Flattens Jacobian, dropping zero entries, to (j_ij, expr) pairs |
| 336 | Builds JVP sum `sum(j_ij * v[j])` per output |
| 337 | Removes output equations (not needed for JVP) |
| 338 | Applies CSE or topological sort |
| 339 | Prunes unused assignments |
| 340 | Caches and returns JVPEquations |

---

## `odesystems/symbolic/codegen/linear_operators.py`

### `_partition_cached_assignments`

| # | Functionality |
|---|--------------|
| 341 | Delegates to `equations.cached_partition()` |

### `_inline_aux_assignments`

| # | Functionality |
|---|--------------|
| 342 | Returns auxiliary expressions in non_jvp_order |

### `_build_operator_body`

| # | Functionality |
|---|--------------|
| 343 | Computes mass matrix-vector product `M @ v` terms |
| 344 | Converts integer mass matrix entries to Float |
| 345 | Builds `beta*M*v - gamma*a_ij*h*J*v` output updates |
| 346 | Non-cached path: applies state substitution `base_state + a_ij*state` |
| 347 | Cached path: reads auxiliaries from `cached_aux[idx]` |
| 348 | Non-cached path: deduplicates combined assignments |
| 349 | Prunes unused assignments for `out` outputs |
| 350 | Returns `"        pass"` when no lines |

### `_build_cached_jvp_body`

| # | Functionality |
|---|--------------|
| 351 | Reads cached auxiliaries from indexed base |
| 352 | Builds `J*v` output updates from jvp_terms |
| 353 | Prunes unused assignments |
| 354 | Returns `"        pass"` when no lines |

### `_build_prepare_body`

| # | Functionality |
|---|--------------|
| 355 | Assigns preparation expressions |
| 356 | Writes cached values to `cached_aux[idx]` |
| 357 | Prunes unused assignments for `cached_aux` |
| 358 | Returns `"        pass"` when no lines |

### `generate_operator_apply_code_from_jvp`

| # | Functionality |
|---|--------------|
| 359 | Gets inline aux assignments and builds operator body |
| 360 | Renders constants and formats OPERATOR_APPLY_TEMPLATE |

### `generate_cached_operator_apply_code_from_jvp`

| # | Functionality |
|---|--------------|
| 361 | Partitions cached/runtime assignments and builds operator body |
| 362 | Renders constants and formats CACHED_OPERATOR_APPLY_TEMPLATE |

### `generate_prepare_jac_code_from_jvp`

| # | Functionality |
|---|--------------|
| 363 | Partitions assignments, builds prepare body |
| 364 | Returns (code, aux_count) tuple |

### `generate_cached_jvp_code_from_jvp`

| # | Functionality |
|---|--------------|
| 365 | Partitions assignments, builds cached JVP body |
| 366 | Formats CACHED_JVP_TEMPLATE |

### `generate_operator_apply_code`

| # | Functionality |
|---|--------------|
| 367 | Defaults M to identity matrix when None |
| 368 | Generates JVP equations when not provided |
| 369 | Delegates to `generate_operator_apply_code_from_jvp` |
| 370 | Logs timing |

### `generate_cached_operator_apply_code`

| # | Functionality |
|---|--------------|
| 371 | Defaults M to identity when None |
| 372 | Generates JVP equations when not provided |
| 373 | Delegates to `generate_cached_operator_apply_code_from_jvp` |
| 374 | Logs timing |

### `generate_prepare_jac_code`

| # | Functionality |
|---|--------------|
| 375 | Generates JVP equations when not provided |
| 376 | Delegates to `generate_prepare_jac_code_from_jvp` |
| 377 | Logs timing |

### `generate_cached_jvp_code`

| # | Functionality |
|---|--------------|
| 378 | Generates JVP equations when not provided |
| 379 | Delegates to `generate_cached_jvp_code_from_jvp` |
| 380 | Logs timing |

### `_build_n_stage_operator_lines`

| # | Functionality |
|---|--------------|
| 381 | Builds stage metadata via `build_stage_metadata` |
| 382 | Per stage: substitutes dx/observable/time/driver symbols |
| 383 | Per stage: computes state evaluation points with coefficient sums |
| 384 | Per stage: builds direction vector combinations for v substitution |
| 385 | Per stage: builds auxiliary assignments with stage-indexed symbols |
| 386 | Per stage: builds JVP terms with substitutions |
| 387 | Per stage: builds output `beta*M*v - gamma*h*jvp` updates |
| 388 | Applies CSE or topological sort to combined expressions |
| 389 | Prunes unused assignments for `out` |
| 390 | Returns `"        pass"` when no lines |

### `generate_n_stage_linear_operator_code`

| # | Functionality |
|---|--------------|
| 391 | Prepares stage data via `prepare_stage_data` |
| 392 | Defaults M to identity when None |
| 393 | Generates JVP equations when not provided |
| 394 | Builds body via `_build_n_stage_operator_lines` |
| 395 | Formats N_STAGE_OPERATOR_TEMPLATE |
| 396 | Logs timing |

---

## `odesystems/symbolic/codegen/preconditioners.py`

### `_build_neumann_body_with_state_subs`

| # | Functionality |
|---|--------------|
| 397 | Builds state substitution `base_state[i] + a_ij * state[i]` |
| 398 | Applies state substitution to JVP assignments |
| 399 | Prints CUDA lines and replaces `v[` with `out[` |
| 400 | Prunes unused assignments for `out` |
| 401 | Returns `["pass"]` when no lines |

### `_build_cached_neumann_body`

| # | Functionality |
|---|--------------|
| 402 | Partitions cached/runtime from equations |
| 403 | Reads cached auxiliaries from `cached_aux[idx]` |
| 404 | Builds JVP output assignments |
| 405 | Prunes unused assignments for `v` |
| 406 | Replaces `v[` with `out[` in printed lines |
| 407 | Returns `"            pass"` when no lines |

### `_build_n_stage_neumann_lines`

| # | Functionality |
|---|--------------|
| 408 | Builds stage metadata |
| 409 | Per stage: substitutes symbols and computes state evaluation points |
| 410 | Per stage: builds direction combos and v-substitution |
| 411 | Per stage: builds stage aux assignments with substitutions |
| 412 | Per stage: builds JVP terms and writes to `jvp[offset+i]` |
| 413 | Applies CSE or topological sort |
| 414 | Prunes unused assignments for `jvp` |
| 415 | Returns `"            pass"` when no lines |

### `generate_neumann_preconditioner_code`

| # | Functionality |
|---|--------------|
| 416 | Generates JVP equations when not provided |
| 417 | Builds body via `_build_neumann_body_with_state_subs` |
| 418 | Formats NEUMANN_TEMPLATE with n_out, jv_body, const_lines |
| 419 | Logs timing |

### `generate_neumann_preconditioner_cached_code`

| # | Functionality |
|---|--------------|
| 420 | Generates JVP equations when not provided |
| 421 | Builds body via `_build_cached_neumann_body` |
| 422 | Formats NEUMANN_CACHED_TEMPLATE |
| 423 | Logs timing |

### `generate_n_stage_neumann_preconditioner_code`

| # | Functionality |
|---|--------------|
| 424 | Prepares stage data |
| 425 | Generates JVP equations when not provided |
| 426 | Builds body via `_build_n_stage_neumann_lines` |
| 427 | Formats N_STAGE_NEUMANN_TEMPLATE with total_states and state_count |
| 428 | Logs timing |

---

## `odesystems/symbolic/codegen/nonlinear_residuals.py`

### `_build_residual_lines`

| # | Functionality |
|---|--------------|
| 429 | Substitutes dxdt symbols with `dx_i` intermediates |
| 430 | Substitutes observable symbols with numbered `aux_` symbols |
| 431 | Applies state evaluation `base_state[i] + a_ij * u[i]` |
| 432 | Computes `beta * M * u - gamma * h * dx_i` per output |
| 433 | Applies CSE or topological sort |
| 434 | Prunes unused assignments for `out` |
| 435 | Returns `"        pass"` when no lines |

### `_build_n_stage_residual_lines`

| # | Functionality |
|---|--------------|
| 436 | Builds stage metadata |
| 437 | Per stage: substitutes dx/observable/time/driver symbols |
| 438 | Per stage: computes state evaluation points with coefficient sums |
| 439 | Per stage: builds `beta*M*u - gamma*h*dx` output updates |
| 440 | Applies CSE or topological sort |
| 441 | Prunes unused assignments for `out` |
| 442 | Returns `"        pass"` when no lines |

### `generate_residual_code`

| # | Functionality |
|---|--------------|
| 443 | Defaults M to identity when None |
| 444 | Builds residual lines via `_build_residual_lines` |
| 445 | Renders constants and formats RESIDUAL_TEMPLATE |

### `generate_stage_residual_code`

| # | Functionality |
|---|--------------|
| 446 | Delegates to `generate_residual_code` |
| 447 | Logs timing |

### `generate_n_stage_residual_code`

| # | Functionality |
|---|--------------|
| 448 | Prepares stage data |
| 449 | Defaults M to identity when None |
| 450 | Builds body via `_build_n_stage_residual_lines` |
| 451 | Formats N_STAGE_RESIDUAL_TEMPLATE |
| 452 | Logs timing |

---

## `odesystems/symbolic/codegen/_stage_utils.py`

### `prepare_stage_data`

| # | Functionality |
|---|--------------|
| 453 | Sympifies coefficient matrix via `sp.Matrix.applyfunc(sp.S)` |
| 454 | Sympifies node expressions via `sp.S` |
| 455 | Returns (coeff_matrix, node_exprs, stage_count) |

### `build_stage_metadata`

| # | Functionality |
|---|--------------|
| 456 | Creates `c_<stage>` node symbols and assigns node values |
| 457 | Creates `a_<stage>_<col>` coefficient symbols and assigns values |
| 458 | Returns (metadata_exprs, coeff_symbols, node_symbols) |

---

## `odesystems/symbolic/parsing/cellml.py`

### `_sanitize_symbol_name`

| # | Functionality |
|---|--------------|
| 459 | Replaces `$` with `_` |
| 460 | Replaces `.` with `_` |
| 461 | Prepends `var` when name starts with `_` followed by digit |
| 462 | Prepends `var_` when name starts with digit |
| 463 | Replaces remaining invalid characters with `_` |

### `load_cellml_model`

| # | Functionality |
|---|--------------|
| 464 | Raises ImportError when cellmlmanip not installed |
| 465 | Raises TypeError when path is not string |
| 466 | Raises FileNotFoundError when path does not exist |
| 467 | Raises ValueError when file lacks .cellml extension |
| 468 | Defaults name to filename stem when None |
| 469 | Non-GUI path: checks cache early, returns cached ODE on hit |
| 470 | Loads model via `cellmlmanip.load_model` |
| 471 | Converts Dummy symbols to regular Symbols with sanitized names |
| 472 | Extracts initial values from state variables |
| 473 | Extracts units from state variables |
| 474 | Identifies time variable from derivative independent variables |
| 475 | Raises ValueError for multiple independent variables |
| 476 | Maps time variable to standard `t` symbol |
| 477 | Converts numeric Dummy symbols (`_0.5`, `_1.0`) to Integer/Float |
| 478 | Separates differential and algebraic equations |
| 479 | Substitutes Dummy-to-Symbol mapping in all equations |
| 480 | Builds dxdt equations from differential equations |
| 481 | Classifies algebraic equations as constants vs parameters vs auxiliaries |
| 482 | Parameters classification: checks against parameters_set |
| 483 | Collects units for parameters and observables |
| 484 | Handles parameters as dict: merges with CellML-extracted values (CellML takes precedence) |
| 485 | GUI path: launches `edit_pre_parse_dicts` for user editing |
| 486 | Post-GUI cache check with effective parameters |
| 487 | Cache miss: calls `parse_input` with all extracted data |
| 488 | Saves parsed result to cache |
| 489 | Constructs and returns SymbolicODE |

---

## `odesystems/symbolic/parsing/cellml_cache.py`

### `CellMLCache.__init__`

| # | Functionality |
|---|--------------|
| 490 | Raises TypeError when model_name is not string |
| 491 | Raises TypeError when cellml_path is not string |
| 492 | Raises ValueError when model_name is empty |
| 493 | Raises FileNotFoundError when cellml_path does not exist |
| 494 | Sets cache_dir relative to CWD/generated/model_name |
| 495 | Sets max_entries to 5 |

### `CellMLCache.get_cellml_hash`

| # | Functionality |
|---|--------------|
| 496 | Reads file in binary mode and returns SHA256 hex digest |

### `CellMLCache._serialize_args`

| # | Functionality |
|---|--------------|
| 497 | Sorts parameter and observable lists for order-independence |
| 498 | Converts precision to string via `__name__` or `str()` |
| 499 | Handles None precision |
| 500 | Returns deterministic JSON string |

### `CellMLCache.compute_cache_key`

| # | Functionality |
|---|--------------|
| 501 | Combines file hash and serialized args hash |
| 502 | Returns first 16 characters of SHA256 |

### `CellMLCache._load_manifest`

| # | Functionality |
|---|--------------|
| 503 | Returns empty manifest when file doesn't exist |
| 504 | Returns empty manifest on JSON decode error |
| 505 | Returns parsed manifest on success |

### `CellMLCache._save_manifest`

| # | Functionality |
|---|--------------|
| 506 | Creates cache directory if needed |
| 507 | Writes manifest as indented JSON |
| 508 | Logs failure message without raising |

### `CellMLCache._update_lru_order`

| # | Functionality |
|---|--------------|
| 509 | Removes existing entry for args_hash |
| 510 | Appends new entry with updated timestamp at end |

### `CellMLCache._evict_lru`

| # | Functionality |
|---|--------------|
| 511 | Removes oldest entries when over max_entries |
| 512 | Deletes cache pickle files for evicted entries |
| 513 | Ignores FileNotFoundError during deletion |

### `CellMLCache.cache_valid`

| # | Functionality |
|---|--------------|
| 514 | Returns False when file hash has changed |
| 515 | Returns False when args_hash not in entries |
| 516 | Returns False when cache pickle file missing |
| 517 | Returns True when file hash matches and pickle exists |

### `CellMLCache.load_from_cache`

| # | Functionality |
|---|--------------|
| 518 | Returns None when cache_valid is False |
| 519 | Loads pickle file and returns cached data dict |
| 520 | Updates LRU order on successful load |
| 521 | Returns None and logs error on load failure |

### `CellMLCache.save_to_cache`

| # | Functionality |
|---|--------------|
| 522 | Creates cache directory if needed |
| 523 | Serializes cache data as pickle with HIGHEST_PROTOCOL |
| 524 | Updates manifest file_hash |
| 525 | Updates LRU order |
| 526 | Evicts oldest entries if over limit |
| 527 | Saves updated manifest |
| 528 | Logs failure message without raising |


### Integrators — Algorithms

| # | Source File | Inventory |
|---|------------|-----------|
| 33 | `algorithms/base_algorithm_step.py` | [x] |
| 34 | `algorithms/ode_explicitstep.py` | [x] |
| 35 | `algorithms/ode_implicitstep.py` | [x] |
| 36 | `algorithms/explicit_euler.py` | [x] |
| 37 | `algorithms/generic_erk.py` | [x] |
| 38 | `algorithms/generic_erk_tableaus.py` | [x] |
| 39 | `algorithms/generic_dirk.py` | [x] |
| 40 | `algorithms/generic_dirk_tableaus.py` | [x] |
| 41 | `algorithms/generic_firk.py` | [x] |
| 42 | `algorithms/generic_firk_tableaus.py` | [x] |
| 43 | `algorithms/generic_rosenbrock_w.py` | [x] |
| 44 | `algorithms/generic_rosenbrockw_tableaus.py` | [x] |
| 45 | `algorithms/backwards_euler.py` | [x] |
| 46 | `algorithms/backwards_euler_predict_correct.py` | [x] |
| 47 | `algorithms/crank_nicolson.py` | [x] |
| 48 | `algorithms/__init__.py` | [x] |


## 1. `base_algorithm_step.py`

### `ButcherTableau` — Attrs Class

#### `__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 1 | Validates `b_hat` length matches `stage_count` when `b_hat` is not None |
| 2 | Raises `ValueError` when `b_hat` length != stage_count |
| 3 | Raises `ValueError` when `b_hat` does not sum to one (tolerance 1e-8) |
| 4 | Raises `ValueError` when `b` does not sum to one (tolerance 1e-8) |
| 5 | Passes validation when `b_hat` is None |

#### Properties

| # | Functionality |
|---|--------------|
| 6 | `d` returns `None` when `b_hat` is None |
| 7 | `d` returns tuple of `(b_i - b_hat_i)` differences when `b_hat` is set |
| 8 | `stage_count` returns `len(self.b)` |
| 9 | `has_error_estimate` returns `False` when `d` is None |
| 10 | `has_error_estimate` returns `False` when all `d` weights are 0.0 |
| 11 | `has_error_estimate` returns `True` when any `d` weight != 0.0 |
| 12 | `first_same_as_last` returns `True` when `c[0]==0.0`, `c[-1]==1.0`, and `a[-1]==b` |
| 13 | `first_same_as_last` returns `False` when `c` is empty |
| 14 | `first_same_as_last` returns `False` when conditions not met |
| 15 | `can_reuse_accepted_start` returns `True` when `c[0]==0.0` |
| 16 | `can_reuse_accepted_start` returns `False` when `c` is empty |
| 17 | `can_reuse_accepted_start` returns `False` when `c[0]!=0.0` |
| 18 | `accumulates_output` returns `True` when `b_matches_a_row` is None |
| 19 | `accumulates_output` returns `False` when `b_matches_a_row` is not None |
| 20 | `accumulates_error` returns `True` when `b_hat_matches_a_row` is None |
| 21 | `accumulates_error` returns `False` when `b_hat_matches_a_row` is not None |
| 22 | `b_matches_a_row` returns row index where `a[row]` matches `b` within 1e-15 |
| 23 | `b_matches_a_row` returns None when no row matches `b` |
| 24 | `b_matches_a_row` prefers the last matching row when multiple match |
| 25 | `b_hat_matches_a_row` returns row index where `a[row]` matches `b_hat` within 1e-15 |
| 26 | `b_hat_matches_a_row` returns None when `b_hat` is None |
| 27 | `b_hat_matches_a_row` returns None when no row matches |

#### `_find_matching_row`

| # | Functionality |
|---|--------------|
| 28 | Returns None immediately when `target_weights` is None |
| 29 | Returns None when no row in `a` matches `target_weights` within tolerance 1e-15 |
| 30 | Returns last matching row index when multiple rows match |
| 31 | Compares only up to `stage_count` elements from each row |

#### `typed_rows`

| # | Functionality |
|---|--------------|
| 32 | Pads rows shorter than `stage_count` with zeros |
| 33 | Converts each entry using `numba_precision` |
| 34 | Returns tuple of tuples |

#### `typed_columns`

| # | Functionality |
|---|--------------|
| 35 | Returns column-major transposition of `typed_rows` output |

#### `a_flat`

| # | Functionality |
|---|--------------|
| 36 | Returns flattened 1D row-major tuple of `a` matrix, precision-typed |

#### `explicit_terms`

| # | Functionality |
|---|--------------|
| 37 | Returns column-major `a` matrix with diagonal and upper elements zeroed |

#### `typed_vector`

| # | Functionality |
|---|--------------|
| 38 | Returns precision-typed tuple from input vector |

#### `error_weights`

| # | Functionality |
|---|--------------|
| 39 | Returns None when `has_error_estimate` is False |
| 40 | Returns precision-typed `d` vector when `has_error_estimate` is True |

### `StepControlDefaults` — Attrs Class

| # | Functionality |
|---|--------------|
| 41 | `copy()` returns a deep copy with a new `step_controller` dict |

### `BaseStepConfig` — Attrs Class

#### Properties

| # | Functionality |
|---|--------------|
| 42 | `settings_dict` returns dict with `n`, `n_drivers`, `precision` |
| 43 | `first_same_as_last` returns `False` when no `tableau` attribute |
| 44 | `first_same_as_last` delegates to `tableau.first_same_as_last` when tableau present |
| 45 | `can_reuse_accepted_start` returns `False` when no `tableau` attribute |
| 46 | `can_reuse_accepted_start` delegates to `tableau.can_reuse_accepted_start` when tableau present |
| 47 | `stage_count` returns 1 when no `tableau` attribute |
| 48 | `stage_count` delegates to `tableau.stage_count` when tableau present |

### `StepCache` — Attrs Class

| # | Functionality |
|---|--------------|
| 49 | Stores `step` device function (required) |
| 50 | Stores optional `nonlinear_solver` device function (default None) |

### `BaseAlgorithmStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 51 | Deep-copies `_controller_defaults` |
| 52 | Calls `setup_compile_settings(config)` |
| 53 | Sets `is_controller_fixed = False` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 54 | Default implementation is a no-op (pass) |

#### `update`

| # | Functionality |
|---|--------------|
| 55 | Returns empty set when `updates_dict` is None and no kwargs |
| 56 | Returns empty set when `updates_dict` is empty dict and no kwargs |
| 57 | Merges `kwargs` into `updates_dict` |
| 58 | Forwards to `update_compile_settings(silent=True)` |
| 59 | Forwards to `buffer_registry.update(self, silent=True)` |
| 60 | Calls `register_buffers()` after buffer update |
| 61 | Valid-but-inapplicable params (in `ALL_ALGORITHM_STEP_PARAMETERS` but not recognized) marked as recognized |
| 62 | Warning emitted for valid-but-inapplicable params when `silent=False` |
| 63 | No warning for valid-but-inapplicable when `silent=True` |
| 64 | `KeyError` raised for truly invalid params when `silent=False` |
| 65 | No `KeyError` for truly invalid when `silent=True` |
| 66 | Returns union of all recognized keys |

#### Forwarding Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 67 | `n_drivers` | `compile_settings.n_drivers` (cast to int) |
| 68 | `n` | `compile_settings.n` |
| 69 | `controller_defaults` | `_controller_defaults.copy()` |
| 70 | `tableau` | `getattr(compile_settings, 'tableau', None)` |
| 71 | `first_same_as_last` | `compile_settings.first_same_as_last` |
| 72 | `can_reuse_accepted_start` | `compile_settings.can_reuse_accepted_start` |
| 73 | `step_function` | `get_cached_output('step')` |
| 74 | `settings_dict` | `compile_settings.settings_dict` |
| 75 | `evaluate_f` | `compile_settings.evaluate_f` |
| 76 | `evaluate_observables` | `compile_settings.evaluate_observables` |
| 77 | `get_solver_helper_fn` | `compile_settings.get_solver_helper_fn` |
| 78 | `stage_count` | `compile_settings.stage_count` |

#### Abstract Properties

| # | Functionality |
|---|--------------|
| 79 | `threads_per_step` — abstract, raises NotImplementedError |
| 80 | `is_multistage` — abstract, raises NotImplementedError |
| 81 | `is_adaptive` — abstract, raises NotImplementedError |
| 82 | `is_implicit` — abstract, raises NotImplementedError |
| 83 | `order` — abstract, raises NotImplementedError |

---

## 2. `ode_explicitstep.py`

### `ExplicitStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 1 | Subclass of `BaseStepConfig` with no additional fields |

### `ODEExplicitStep` — Class

#### `build`

| # | Functionality |
|---|--------------|
| 2 | Unpacks `evaluate_f`, `numba_precision`, `n`, `evaluate_observables`, `evaluate_driver_at_t`, `n_drivers` from config |
| 3 | Delegates to `build_step()` with those arguments |

#### `build_step`

| # | Functionality |
|---|--------------|
| 4 | Abstract method, raises NotImplementedError |

#### Properties

| # | Functionality |
|---|--------------|
| 5 | `is_implicit` returns `False` |

---

## 3. `ode_implicitstep.py`

### `ImplicitStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 1 | `beta` property returns `self.precision(self._beta)` |
| 2 | `gamma` property returns `self.precision(self._gamma)` |
| 3 | `settings_dict` extends parent with `beta`, `gamma`, `M`, `preconditioner_order`, `get_solver_helper_fn` |

### `ODEImplicitStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 4 | Raises `ValueError` when `solver_type` is not 'newton' or 'linear' |
| 5 | Filters `kwargs` into linear solver params (ignoring None values) |
| 6 | Filters `kwargs` into Newton-Krylov params (ignoring None values) |
| 7 | Creates `LinearSolver` with precision, n, and linear kwargs |
| 8 | Creates `NewtonKrylov` wrapping `LinearSolver` when `solver_type='newton'` |
| 9 | Assigns `LinearSolver` directly when `solver_type='linear'` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 10 | Default implementation is a no-op (pass) |

#### `update`

| # | Functionality |
|---|--------------|
| 11 | Returns empty set when no updates |
| 12 | Delegates to `self.solver.update(silent=True)` |
| 13 | Injects `solver_function` from `self.solver.device_function` into updates |
| 14 | Delegates remaining to `super().update(silent=True)` |
| 15 | Returns union of recognized keys from solver and parent |

#### `build`

| # | Functionality |
|---|--------------|
| 16 | Calls `build_implicit_helpers()` first |
| 17 | Unpacks config and delegates to `build_step()` with `solver_function` included |

#### `build_step`

| # | Functionality |
|---|--------------|
| 18 | Abstract method, raises NotImplementedError |

#### `build_implicit_helpers`

| # | Functionality |
|---|--------------|
| 19 | Calls `get_solver_helper_fn("neumann_preconditioner", ...)` |
| 20 | Calls `get_solver_helper_fn("stage_residual", ...)` |
| 21 | Calls `get_solver_helper_fn("linear_operator", ...)` |
| 22 | Updates solver with operator, preconditioner, residual |
| 23 | Stores compiled `solver.device_function` in compile settings |

#### Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 24 | `is_implicit` | returns `True` |
| 25 | `beta` | `compile_settings.beta` |
| 26 | `gamma` | `compile_settings.gamma` |
| 27 | `mass_matrix` | `compile_settings.M` |
| 28 | `preconditioner_order` | `compile_settings.preconditioner_order` (cast to int) |
| 29 | `krylov_atol` | `solver.krylov_atol` |
| 30 | `krylov_rtol` | `solver.krylov_rtol` |
| 31 | `krylov_max_iters` | `solver.krylov_max_iters` (cast to int) |
| 32 | `linear_correction_type` | `solver.linear_correction_type` |
| 33 | `newton_atol` | `getattr(solver, 'newton_atol', None)` |
| 34 | `newton_rtol` | `getattr(solver, 'newton_rtol', None)` |
| 35 | `newton_max_iters` | `getattr(solver, 'newton_max_iters', None)` with int cast |
| 36 | `newton_damping` | `getattr(solver, 'newton_damping', None)` |
| 37 | `newton_max_backtracks` | `getattr(solver, 'newton_max_backtracks', None)` with int cast |
| 38 | `settings_dict` | merges `super().settings_dict` with `solver.settings_dict` |

---

## 4. `explicit_euler.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `EE_DEFAULTS` sets `step_controller='fixed'` and `dt=1e-3` |

### `ExplicitEulerStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 2 | Builds `ExplicitStepConfig` via `build_config` with required params |
| 3 | Passes `EE_DEFAULTS.copy()` as controller defaults |

#### `build_step`

| # | Functionality |
|---|--------------|
| 4 | Device step evaluates `evaluate_f` to get `dxdt` |
| 5 | Computes `proposed_state[i] = state[i] + dt * dxdt[i]` |
| 6 | Branch: calls `evaluate_driver_at_t` when available (`has_evaluate_driver_at_t`) |
| 7 | Branch: skips driver evaluation when not available |
| 8 | Calls `evaluate_observables` on proposed state |
| 9 | Returns `StepCache(step=step, nonlinear_solver=None)` |

#### Properties

| # | Functionality |
|---|--------------|
| 10 | `threads_per_step` returns 1 |
| 11 | `is_multistage` returns `False` |
| 12 | `is_adaptive` returns `False` |
| 13 | `order` returns 1 |

---

## 5. `generic_erk.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `ERK_ADAPTIVE_DEFAULTS` sets PID controller with adaptive settings |
| 2 | `ERK_FIXED_DEFAULTS` sets fixed controller with `dt=1e-3` |

### `ERKStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 3 | `tableau` field defaults to `DEFAULT_ERK_TABLEAU` |
| 4 | `stage_rhs_location` field defaults to `'local'`, validated in `['local','shared']` |
| 5 | `stage_accumulator_location` field defaults to `'local'`, validated in `['local','shared']` |
| 6 | `first_same_as_last` property delegates to `self.tableau.first_same_as_last` |

### `ERKStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 7 | Builds `ERKStepConfig` via `build_config` |
| 8 | Selects `ERK_ADAPTIVE_DEFAULTS` when `tableau.has_error_estimate` is True |
| 9 | Selects `ERK_FIXED_DEFAULTS` when `tableau.has_error_estimate` is False |
| 10 | Calls `register_buffers()` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 11 | Registers `stage_rhs` buffer with size `n`, persistent=True |
| 12 | Registers `stage_accumulator` buffer with size `max(stage_count-1,0) * n` |

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 13 | Stage 0: skips RHS evaluation when FSAL cache usable (`first_same_as_last`, all threads accepted) |
| 14 | Stage 0: evaluates RHS when cache not usable |
| 15 | FSAL warp-sync: uses `activemask()` + `all_sync` to check warp-wide acceptance |
| 16 | Accumulates output via weighted sum when `accumulates_output` is True |
| 17 | Assigns output directly from accumulator when `b_row` matches stage |
| 18 | Accumulates error via weighted sum when `accumulates_error` is True |
| 19 | Assigns error directly from accumulator when `b_hat_row` matches stage |
| 20 | Scales accumulated output by `dt_scalar` and adds `state` |
| 21 | Scales accumulated error by `dt_scalar` |
| 22 | Forms error as `proposed_state - error` when `not accumulates_error` |
| 23 | Evaluates drivers at stage times when `has_evaluate_driver_at_t` |
| 24 | Evaluates drivers at end time when `has_evaluate_driver_at_t` |
| 25 | Evaluates observables at each stage and at end time |
| 26 | Returns `int32(0)` status code |

#### Properties

| # | Functionality |
|---|--------------|
| 27 | `is_multistage` returns `tableau.stage_count > 1` |
| 28 | `is_adaptive` returns `tableau.has_error_estimate` |
| 29 | `order` returns `tableau.order` |
| 30 | `threads_per_step` returns 1 |

---

## 6. `generic_erk_tableaus.py`

### `ERKTableau` — Attrs Class

| # | Functionality |
|---|--------------|
| 1 | Subclass of `ButcherTableau` with no additional fields (type tag) |

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 2 | `HEUN_21_TABLEAU` — 2-stage, order 2, no error estimate |
| 3 | `RALSTON_33_TABLEAU` — 3-stage, order 3, no error estimate |
| 4 | `BOGACKI_SHAMPINE_32_TABLEAU` — 4-stage, order 3, with `b_hat` |
| 5 | `DORMAND_PRINCE_54_TABLEAU` — 7-stage, order 5, with `b_hat` |
| 6 | `CLASSICAL_RK4_TABLEAU` — 4-stage, order 4, no error estimate |
| 7 | `CASH_KARP_54_TABLEAU` — 6-stage, order 5, with `b_hat` |
| 8 | `FEHLBERG_45_TABLEAU` — 6-stage, order 5, with `b_hat` |
| 9 | `DORMAND_PRINCE_853_TABLEAU` — 12-stage, order 8, with `b_hat` |
| 10 | `TSITOURAS_54_TABLEAU` — 7-stage, order 5, with `b_hat` |
| 11 | `VERNER_76_TABLEAU` — 10-stage, order 7, with `b_hat` |
| 12 | `DEFAULT_ERK_TABLEAU` aliases `DORMAND_PRINCE_54_TABLEAU` |
| 13 | `ERK_TABLEAU_REGISTRY` maps string aliases to tableau instances |

---

## 7. `generic_dirk.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `DIRK_ADAPTIVE_DEFAULTS` sets PID controller with adaptive settings |
| 2 | `DIRK_FIXED_DEFAULTS` sets fixed controller with `dt=1e-3` |

### `DIRKStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 3 | `tableau` defaults to `DEFAULT_DIRK_TABLEAU` |
| 4 | `stage_increment_location` defaults `'local'`, validated `['local','shared']` |
| 5 | `stage_base_location` defaults `'local'`, validated `['local','shared']` |
| 6 | `accumulator_location` defaults `'local'`, validated `['local','shared']` |
| 7 | `stage_rhs_location` defaults `'local'`, validated `['local','shared']` |

### `DIRKStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 8 | Creates identity mass matrix `eye(n, dtype=precision)` |
| 9 | Builds `DIRKStepConfig` via `build_config` with fixed `beta=1.0`, `gamma=1.0` |
| 10 | Selects `DIRK_ADAPTIVE_DEFAULTS` when `tableau.has_error_estimate` is True |
| 11 | Selects `DIRK_FIXED_DEFAULTS` when `tableau.has_error_estimate` is False |
| 12 | Passes `**kwargs` to `super().__init__` (solver params) |
| 13 | Calls `register_buffers()` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 14 | Clears existing buffer registrations via `buffer_registry.clear_parent(self)` |
| 15 | Registers solver child allocators |
| 16 | Registers `stage_increment` (size n, persistent=True) |
| 17 | Registers `accumulator` (size `max(stage_count-1,0)*n`) |
| 18 | Registers `stage_base` (size n, aliases `accumulator`) |
| 19 | Registers `stage_rhs` (size n, persistent=True) |

#### `build_implicit_helpers` (override)

| # | Functionality |
|---|--------------|
| 20 | Calls `get_solver_helper_fn` for preconditioner, residual, operator |
| 21 | Updates solver with operator, preconditioner, residual (no `n` param unlike parent) |
| 22 | Stores `solver.device_function` in compile settings |

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 23 | Stage 0 FSAL: reuses cached RHS when `first_same_as_last`, multistage, all warp threads accepted |
| 24 | Stage 0: copies accepted drivers when `can_reuse_accepted_start` and not FSAL |
| 25 | Stage 0: evaluates drivers at stage time when neither FSAL nor reuse applies |
| 26 | Stage 0 implicit: calls nonlinear solver when `stage_implicit[0]` is True |
| 27 | Stage 0 explicit: skips solver when `stage_implicit[0]` is False |
| 28 | Stages 1..s: streams previous stage RHS into accumulators |
| 29 | Stages 1..s: calls nonlinear solver for implicit stages |
| 30 | Stages 1..s: skips solver for explicit stages |
| 31 | Accumulates output or assigns directly from `b_row` |
| 32 | Accumulates error or assigns directly from `b_hat_row` |
| 33 | Scales accumulated output by `dt_scalar` + adds `state` |
| 34 | Scales accumulated error by `dt_scalar` |
| 35 | Forms error as `proposed_state - error` when `not accumulates_error` |
| 36 | Evaluates drivers and observables at end time |
| 37 | Returns `status_code` encoding solver results |

#### Properties

| # | Functionality |
|---|--------------|
| 38 | `is_multistage` returns `tableau.stage_count > 1` |
| 39 | `is_adaptive` returns `tableau.has_error_estimate` |
| 40 | `is_implicit` returns `True` |
| 41 | `order` returns `tableau.order` |
| 42 | `threads_per_step` returns 1 |

---

## 8. `generic_dirk_tableaus.py`

### `DIRKTableau` — Attrs Class

#### `diagonal`

| # | Functionality |
|---|--------------|
| 1 | Extracts diagonal entries `a[idx][idx]` for each stage |
| 2 | Returns precision-typed tuple via `typed_vector` |

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 3 | `IMPLICIT_MIDPOINT_TABLEAU` — 1-stage, order 2 |
| 4 | `TRAPEZOIDAL_DIRK_TABLEAU` — 2-stage, order 2 (ESDIRK) |
| 5 | `LOBATTO_IIIC_3_TABLEAU` — 3-stage, order 4 |
| 6 | `SDIRK_2_2_TABLEAU` — 2-stage, order 2, L-stable |
| 7 | `L_STABLE_DIRK3_TABLEAU` — 3-stage, order 3, L-stable |
| 8 | `L_STABLE_SDIRK4_TABLEAU` — 5-stage, order 4, with `b_hat` |
| 9 | `DIRK_TABLEAU_REGISTRY` maps string aliases to tableau instances |
| 10 | `DEFAULT_DIRK_TABLEAU` aliases `LOBATTO_IIIC_3_TABLEAU` |

---

## 9. `generic_firk.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `FIRK_ADAPTIVE_DEFAULTS` sets PID controller with adaptive settings |
| 2 | `FIRK_FIXED_DEFAULTS` sets fixed controller with `dt=1e-3` |

### `FIRKStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 3 | `tableau` defaults to `DEFAULT_FIRK_TABLEAU` |
| 4 | `stage_increment_location` defaults `'local'`, validated `['local','shared']` |
| 5 | `stage_driver_stack_location` defaults `'local'`, validated `['local','shared']` |
| 6 | `stage_state_location` defaults `'local'`, validated `['local','shared']` |
| 7 | `stage_count` property delegates to `tableau.stage_count` |
| 8 | `all_stages_n` property returns `stage_count * n` |

### `FIRKStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 9 | Creates identity mass matrix |
| 10 | Builds `FIRKStepConfig` via `build_config` |
| 11 | Selects `FIRK_ADAPTIVE_DEFAULTS` when `tableau.has_error_estimate` is True |
| 12 | Selects `FIRK_FIXED_DEFAULTS` when `tableau.has_error_estimate` is False |
| 13 | Updates solver `n` to `config.all_stages_n` (coupled system) |
| 14 | Calls `register_buffers()` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 15 | Registers solver child allocators |
| 16 | Registers `stage_increment` (size `all_stages_n`, persistent=True) |
| 17 | Registers `stage_driver_stack` (size `stage_count * n_drivers`) |
| 18 | Registers `stage_state` (size `n`) |

#### `build_implicit_helpers` (override)

| # | Functionality |
|---|--------------|
| 19 | Calls `get_solver_helper_fn("n_stage_residual", ...)` with `stage_coefficients` and `stage_nodes` |
| 20 | Calls `get_solver_helper_fn("n_stage_linear_operator", ...)` |
| 21 | Calls `get_solver_helper_fn("n_stage_neumann_preconditioner", ...)` |
| 22 | Updates solver with `n=config.all_stages_n` |
| 23 | Stores `solver.device_function` in compile settings |

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 24 | Initializes `proposed_state` from `state` when `accumulates_output` |
| 25 | Pre-fills `stage_driver_stack` with driver evaluations for all stage times |
| 26 | Calls single coupled `nonlinear_solver` for all stages |
| 27 | Reconstructs stage states from increments and coupling matrix |
| 28 | Direct assignment of `proposed_state` when `b_row` matches stage |
| 29 | Direct assignment of `error` when `b_hat_row` matches stage |
| 30 | Kahan summation for accumulated output when `accumulates_output` |
| 31 | Kahan summation for accumulated error when `accumulates_error` |
| 32 | Skips end-time driver evaluation when `ends_at_one` is True |
| 33 | Evaluates drivers at end time when `ends_at_one` is False |
| 34 | Forms error as `proposed_state - error` when `not accumulates_error` |
| 35 | Evaluates observables at end time |

#### Properties

| # | Functionality |
|---|--------------|
| 36 | `is_multistage` returns `stage_count > 1` |
| 37 | `is_adaptive` returns `tableau.has_error_estimate` |
| 38 | `stage_count` returns `compile_settings.stage_count` |
| 39 | `is_implicit` returns `True` |
| 40 | `order` returns `tableau.order` |
| 41 | `threads_per_step` returns 1 |

---

## 10. `generic_firk_tableaus.py`

### `FIRKTableau` — Attrs Class

| # | Functionality |
|---|--------------|
| 1 | Subclass of `ButcherTableau` with no additional fields (type tag) |

### `compute_embedded_weights_radauIIA`

| # | Functionality |
|---|--------------|
| 2 | Defaults `order` to `s` (number of stages) when None |
| 3 | Raises `ValueError` when `order > s` |
| 4 | Uses `linalg.solve` when `order == s` (square system) |
| 5 | Uses `linalg.lstsq` when `order < s` (underdetermined) |
| 6 | Builds Vandermonde-like system from collocation nodes |

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 7 | `GAUSS_LEGENDRE_2_TABLEAU` — 2-stage, order 4, no `b_hat` |
| 8 | `RADAU_IIA_5_TABLEAU` — 3-stage, order 5, with computed `b_hat` |
| 9 | `DEFAULT_FIRK_TABLEAU` aliases `GAUSS_LEGENDRE_2_TABLEAU` |
| 10 | `FIRK_TABLEAU_REGISTRY` maps string aliases to tableau instances |

---

## 11. `generic_rosenbrock_w.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `ROSENBROCK_ADAPTIVE_DEFAULTS` sets PID controller with adaptive settings |
| 2 | `ROSENBROCK_FIXED_DEFAULTS` sets fixed controller with `dt=1e-3` |

### `RosenbrockWStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 3 | `tableau` defaults to `DEFAULT_ROSENBROCK_TABLEAU` |
| 4 | `time_derivative_function` optional device function field |
| 5 | `prepare_jacobian_function` optional device function field |
| 6 | `driver_del_t` optional device function field |
| 7 | `stage_rhs_location` defaults `'local'`, validated `['local','shared']` |
| 8 | `stage_store_location` defaults `'local'`, validated `['local','shared']` |
| 9 | `cached_auxiliaries_location` defaults `'local'`, validated `['local','shared']` |
| 10 | `base_state_placeholder_location` defaults `'local'`, validated `['local','shared']` |
| 11 | `krylov_iters_out_location` defaults `'local'`, validated `['local','shared']` |

### `GenericRosenbrockWStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 12 | Creates identity mass matrix |
| 13 | Sets `gamma` from `tableau.gamma` |
| 14 | Builds `RosenbrockWStepConfig` via `build_config` |
| 15 | Initializes `_cached_auxiliary_count = None` |
| 16 | Selects `ROSENBROCK_ADAPTIVE_DEFAULTS` when `tableau.has_error_estimate` is True |
| 17 | Selects `ROSENBROCK_FIXED_DEFAULTS` when `tableau.has_error_estimate` is False |
| 18 | Passes `solver_type='linear'` to parent (not Newton) |
| 19 | Calls `register_buffers()` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 20 | Registers `stage_rhs` (size `n`) |
| 21 | Registers `stage_store` (size `stage_count * n`) |
| 22 | Registers `cached_auxiliaries` initially with size 0 (updated later) |
| 23 | Registers `stage_increment` (size `n`, persistent=True, aliases `stage_store`) |
| 24 | Registers `base_state_placeholder` (size 1, precision=int32) |
| 25 | Registers `krylov_iters_out` (size 1, precision=int32) |

#### `build_implicit_helpers` (override)

| # | Functionality |
|---|--------------|
| 26 | Calls `get_solver_helper_fn("neumann_preconditioner_cached", ...)` |
| 27 | Calls `get_solver_helper_fn("linear_operator_cached", ...)` |
| 28 | Calls `get_solver_helper_fn("prepare_jac", ...)` |
| 29 | Calls `get_solver_helper_fn("cached_aux_count")` to get count |
| 30 | Updates `cached_auxiliaries` buffer size via `buffer_registry.update_buffer` |
| 31 | Calls `get_solver_helper_fn("time_derivative_rhs")` |
| 32 | Updates linear solver with `use_cached_auxiliaries=True` |
| 33 | Stores `solver_function`, `time_derivative_function`, `prepare_jacobian_function` in compile settings |

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 34 | Calls `prepare_jacobian` to cache Jacobian info |
| 35 | Evaluates `driver_del_t` when `has_evaluate_driver_at_t` |
| 36 | Zeros `proposed_drivers` when no `evaluate_driver_at_t` |
| 37 | Evaluates `time_derivative_rhs` and scales by `dt_scalar` |
| 38 | Stage 0: evaluates `f(state)` and forms RHS with `gamma_stages[0] * time_derivative` |
| 39 | Stage 0: calls linear solver |
| 40 | Stage 0: accumulates output and error if needed |
| 41 | Stages 1..s: accumulates predecessor contributions via `a_coeffs` |
| 42 | Stages 1..s: evaluates drivers, observables, and f at stage state |
| 43 | Stages 1..s: captures direct output when `b_row` matches stage |
| 44 | Stages 1..s: captures direct error when `b_hat_row` matches stage |
| 45 | Last stage recalculates time derivative before forming RHS |
| 46 | Stages 1..s: forms RHS with C-correction + gamma-derivative terms |
| 47 | Uses previous stage solution as initial guess for linear solver |
| 48 | Forms error as `proposed_state - error` when `not accumulates_error` |
| 49 | Evaluates drivers at end time and observables on proposed state |

#### Properties

| # | Functionality |
|---|--------------|
| 50 | `is_multistage` returns `tableau.stage_count > 1` |
| 51 | `is_adaptive` returns `tableau.has_error_estimate` |
| 52 | `cached_auxiliary_count` lazily builds implicit helpers when `_cached_auxiliary_count` is None |
| 53 | `cached_auxiliary_count` returns cached value when already computed |
| 54 | `is_implicit` returns `True` |
| 55 | `order` returns `tableau.order` |
| 56 | `threads_per_step` returns 1 |

---

## 12. `generic_rosenbrockw_tableaus.py`

### `RosenbrockTableau` — Attrs Class

| # | Functionality |
|---|--------------|
| 1 | Adds `C` field (lower-triangular Jacobian update coefficients) |
| 2 | Adds `gamma` field (diagonal shift, default 0.25) |
| 3 | Adds `gamma_stages` field (per-stage diagonal shifts) |
| 4 | `typed_gamma_stages` returns precision-typed tuple via `typed_vector` |

### Module-Level Functions

| # | Functionality |
|---|--------------|
| 5 | `_ros3p_tableau()` constructs ROS3P tableau with computed C matrix from gamma |
| 6 | `_rodas3p_tableau()` constructs 5-stage RODAS3P tableau |
| 7 | `_rosenbrock_23_sciml_tableau()` constructs 3-stage SciML Rosenbrock23 tableau |

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 8 | `ROS3P_TABLEAU` — 3-stage, order 3, with `b_hat` |
| 9 | `RODAS3P_TABLEAU` — 5-stage, order 3, with `b_hat` |
| 10 | `ROSENBROCK_23_SCIML_TABLEAU` — 3-stage, order 3, with `b_hat` |
| 11 | `ROSENBROCK_TABLEAUS` maps string aliases to tableau instances |
| 12 | `DEFAULT_ROSENBROCK_TABLEAU` aliases `ROS3P_TABLEAU` |

---

## 13. `backwards_euler.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `ALGO_CONSTANTS` sets `beta=1.0`, `gamma=1.0`, `M=eye` |
| 2 | `BE_DEFAULTS` sets fixed controller with `dt=1e-3` |

### `BackwardsEulerStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 3 | `increment_cache_location` defaults `'local'`, validated `['local','shared']` |

### `BackwardsEulerStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 4 | Creates identity mass matrix from `ALGO_CONSTANTS['M'](n, dtype=precision)` |
| 5 | Builds `BackwardsEulerStepConfig` via `build_config` |
| 6 | Passes `BE_DEFAULTS.copy()` as controller defaults |
| 7 | Calls `register_buffers()` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 8 | Registers solver child allocators under name `'solver_scratch'` |
| 9 | Registers `increment_cache` (size n, persistent=True) |

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 10 | Initializes `proposed_state` from `increment_cache` (warm start) |
| 11 | Evaluates drivers at `next_time` when `has_evaluate_driver_at_t` |
| 12 | Calls Newton-Krylov solver at `next_time` |
| 13 | Stores increment in `increment_cache` for next step warm start |
| 14 | Computes `proposed_state = increment + state` |
| 15 | Evaluates observables on proposed state |
| 16 | Returns solver status code |

#### Properties

| # | Functionality |
|---|--------------|
| 17 | `is_multistage` returns `False` |
| 18 | `is_adaptive` returns `False` |
| 19 | `threads_per_step` returns 1 |
| 20 | `order` returns 1 |

---

## 14. `backwards_euler_predict_correct.py`

### `BackwardsEulerPCStep` — Class (subclass of `BackwardsEulerStep`)

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 1 | Evaluates `f(state)` to compute explicit predictor |
| 2 | Sets `proposed_state = dt * predictor` as initial guess |
| 3 | Evaluates drivers at `next_time` when `has_evaluate_driver_at_t` |
| 4 | Calls Newton-Krylov solver with predictor as initial guess |
| 5 | Computes `proposed_state = increment + state` |
| 6 | Evaluates observables on proposed state |
| 7 | Returns solver status code |
| 8 | No warm-start cache (unlike parent `BackwardsEulerStep`) |

---

## 15. `crank_nicolson.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `ALGO_CONSTANTS` sets `beta=1.0`, `gamma=1.0`, `M=eye` |
| 2 | `CN_DEFAULTS` sets PID controller with adaptive settings |

### `CrankNicolsonStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 3 | `dxdt_location` defaults `'local'`, validated `['local','shared']` |

### `CrankNicolsonStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 4 | Creates identity mass matrix |
| 5 | Builds `CrankNicolsonStepConfig` via `build_config` |
| 6 | Passes `CN_DEFAULTS.copy()` as controller defaults |
| 7 | Calls `register_buffers()` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 8 | Registers solver child allocators |
| 9 | Registers `cn_dxdt` (size n, aliases `solver_shared`) |

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 10 | Evaluates `f(state)` to compute `dxdt` at current time |
| 11 | Forms CN base state: `base_state[i] = state[i] + half_dt * dxdt[i]` |
| 12 | Aliases `error` as `base_state` (disjoint lifetimes) |
| 13 | Evaluates drivers at end time when `has_evaluate_driver_at_t` |
| 14 | Solves CN implicit system with `stage_coefficient=0.5` |
| 15 | Computes CN solution: `proposed_state = base_state + 0.5 * increment` |
| 16 | Stores CN increment into `base_state` |
| 17 | Solves BE implicit system with `be_coefficient=1.0` using CN increment as guess |
| 18 | Computes error as `proposed_state - (state + BE_increment)` |
| 19 | Evaluates observables on proposed state |
| 20 | Returns combined status from both solves (bitwise OR) |

#### Properties

| # | Functionality |
|---|--------------|
| 21 | `is_multistage` returns `False` |
| 22 | `is_adaptive` returns `True` |
| 23 | `threads_per_step` returns 1 |
| 24 | `order` returns 2 |


### Integrators — Step Controllers

| # | Source File | Inventory |
|---|------------|-----------|
| 49 | `step_control/base_step_controller.py` | [x] |
| 50 | `step_control/fixed_step_controller.py` | [x] |
| 51 | `step_control/adaptive_step_controller.py` | [x] |
| 52 | `step_control/adaptive_I_controller.py` | [x] |
| 53 | `step_control/adaptive_PI_controller.py` | [x] |
| 54 | `step_control/adaptive_PID_controller.py` | [x] |
| 55 | `step_control/gustafsson_controller.py` | [x] |
| 56 | `step_control/__init__.py` | [x] |


### Integrators — Solvers & Loops

| # | Source File | Inventory |
|---|------------|-----------|
| 57 | `matrix_free_solvers/base_solver.py` | [x] |
| 58 | `matrix_free_solvers/linear_solver.py` | [x] |
| 59 | `matrix_free_solvers/newton_krylov.py` | [x] |
| 60 | `matrix_free_solvers/__init__.py` | [x] |
| 61 | `norms.py` | [x] |
| 62 | `array_interpolator.py` | [x] |
| 63 | `loops/ode_loop_config.py` | [x] |
| 64 | `loops/ode_loop.py` | [x] |
| 65 | `loops/__init__.py` | [x] |
| 66 | `IntegratorRunSettings.py` | [x] |
| 67 | `integrators/__init__.py` | [x] |


## `step_control/base_step_controller.py`

### `ControllerCache` (attrs)

| # | Functionality |
|---|--------------|
| 1 | `device_function` defaults to `-1` |

### `BaseStepControllerConfig` (attrs, abstract)

| # | Functionality |
|---|--------------|
| 2 | `n` defaults to 1, validated as int >= 0 |
| 3 | `timestep_memory_location` defaults to `"local"`, validated in `["local", "shared"]` |
| 4 | `__attrs_post_init__` calls super |
| 5 | `dt_min` abstract property |
| 6 | `dt_max` abstract property |
| 7 | `dt0` abstract property |
| 8 | `is_adaptive` abstract property |
| 9 | `settings_dict` abstract property returns dict with `"n"` |

### `BaseStepController`

| # | Functionality |
|---|--------------|
| 10 | `__init__` calls `super().__init__()` |
| 11 | `register_buffers` registers `"timestep_buffer"` with size from `local_memory_elements`, location from `compile_settings.timestep_memory_location`, persistent=True |
| 12 | `build` abstract, returns `ControllerCache` |
| 13 | `n` forwards to `compile_settings.n` |
| 14 | `dt_min` forwards to `compile_settings.dt_min` |
| 15 | `dt_max` forwards to `compile_settings.dt_max` |
| 16 | `dt0` forwards to `compile_settings.dt0` |
| 17 | `is_adaptive` forwards to `compile_settings.is_adaptive` |
| 18 | `local_memory_elements` abstract property |
| 19 | `settings_dict` forwards to `compile_settings.settings_dict` |

#### `BaseStepController.update`

| # | Functionality |
|---|--------------|
| 20 | `updates_dict` defaults to `{}` when None |
| 21 | `kwargs` merged into `updates_dict` |
| 22 | Empty dict returns empty set immediately |
| 23 | `update_compile_settings` called with `silent=True` |
| 24 | Unrecognised keys split into valid-but-inapplicable (in `ALL_STEP_CONTROLLER_PARAMETERS`) vs truly invalid |
| 25 | Valid-but-inapplicable keys added to recognised set |
| 26 | Warning emitted for valid-but-inapplicable keys naming the controller class and listing params |
| 27 | No warning when valid-but-inapplicable is empty |
| 28 | `KeyError` raised for truly invalid keys when `silent=False` |
| 29 | No `KeyError` when `silent=True` with truly invalid keys |
| 30 | Returns full recognised set |

---

## `step_control/fixed_step_controller.py`

### `FixedStepControlConfig` (attrs)

| # | Functionality |
|---|--------------|
| 31 | `_dt` defaults to 1e-3, validated as float > 0 |
| 32 | `__attrs_post_init__` calls super and `_validate_config` |
| 33 | `dt` property returns `precision(self._dt)` |
| 34 | `dt_min` returns `self.dt` |
| 35 | `dt_max` returns `self.dt` |
| 36 | `dt0` returns `self.dt` |
| 37 | `is_adaptive` returns `False` |
| 38 | `settings_dict` extends super with `"dt"` key |

### `FixedStepController`

| # | Functionality |
|---|--------------|
| 39 | `__init__` creates `FixedStepControlConfig` via `build_config`, calls `setup_compile_settings` and `register_buffers` |
| 40 | `build` returns `ControllerCache` with device function that sets `accept_out[0] = 1` and returns `0` |
| 41 | `local_memory_elements` returns `0` |
| 42 | `dt` property forwards to `compile_settings.dt` |

---

## `step_control/adaptive_step_controller.py`

### `AdaptiveStepControlConfig` (attrs)

| # | Functionality |
|---|--------------|
| 43 | `_dt_min` defaults 1e-6, validated float > 0 |
| 44 | `_dt_max` defaults 1.0, validated float > 0 |
| 45 | `atol` defaults `[1e-6]`, converter `tol_converter` |
| 46 | `rtol` defaults `[1e-6]`, converter `tol_converter` |
| 47 | `algorithm_order` defaults 1, validated int >= 1 |
| 48 | `_min_gain` defaults 0.3, validated float in (0, 1) |
| 49 | `_max_gain` defaults 2.0, validated float > 1 |
| 50 | `_safety` defaults 0.9, validated float in (0, 1) |
| 51 | `_deadband_min` defaults 1.0, validated float in (0, 1.0) |
| 52 | `_deadband_max` defaults 1.2, validated float >= 1.0 |

#### `__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 53 | `dt_max` set to `dt_min * 100` when `_dt_max` is None |
| 54 | Warning + `dt_max = dt_min * 100` when `dt_max < dt_min` |
| 55 | No adjustment when `dt_max >= dt_min` |
| 56 | Deadband min/max swapped when `deadband_min > deadband_max` |
| 57 | No swap when `deadband_min <= deadband_max` |

#### Properties

| # | Functionality |
|---|--------------|
| 58 | `dt_min` returns `precision(self._dt_min)` |
| 59 | `dt_max` returns `precision(self._dt_max)`, fallback `dt_min * 100` if None |
| 60 | `dt0` returns `precision(sqrt(dt_min * dt_max))` |
| 61 | `is_adaptive` returns `True` |
| 62 | `min_gain` returns `precision(self._min_gain)` |
| 63 | `max_gain` returns `precision(self._max_gain)` |
| 64 | `safety` returns `precision(self._safety)` |
| 65 | `deadband_min` returns `precision(self._deadband_min)` |
| 66 | `deadband_max` returns `precision(self._deadband_max)` |
| 67 | `settings_dict` extends super with dt_min, dt_max, atol, rtol, algorithm_order, min_gain, max_gain, safety, deadband_min, deadband_max, dt |

### `BaseAdaptiveStepController`

| # | Functionality |
|---|--------------|
| 68 | `__init__` calls super, `setup_compile_settings`, `register_buffers` |
| 69 | `build` calls `build_controller` with config-derived args including `clamp_factory(precision)` |
| 70 | `build_controller` abstract method |
| 71 | `min_gain` forwards to `compile_settings.min_gain` |
| 72 | `max_gain` forwards to `compile_settings.max_gain` |
| 73 | `safety` forwards to `compile_settings.safety` |
| 74 | `deadband_min` forwards to `compile_settings.deadband_min` |
| 75 | `deadband_max` forwards to `compile_settings.deadband_max` |
| 76 | `algorithm_order` returns `int(compile_settings.algorithm_order)` |
| 77 | `atol` forwards to `compile_settings.atol` |
| 78 | `rtol` forwards to `compile_settings.rtol` |
| 79 | `local_memory_elements` abstract property |

---

## `step_control/adaptive_I_controller.py`

### `AdaptiveIController`

| # | Functionality |
|---|--------------|
| 80 | `__init__` builds `AdaptiveStepControlConfig` via `build_config`, passes to super |
| 81 | `local_memory_elements` returns `0` |

#### `build_controller` device function logic

| # | Functionality |
|---|--------------|
| 82 | `order_exponent` computed as `1 / (2 * (1 + algorithm_order))` |
| 83 | Norm computed as mean squared scaled error: `sum((error_i / tol)^2) / n` |
| 84 | Step accepted when `nrm2 <= 1.0` |
| 85 | Step rejected when `nrm2 > 1.0` |
| 86 | Gain computed as `safety * nrm2^(-order_exponent)` |
| 87 | Gain clamped to `[min_gain, max_gain]` |
| 88 | Deadband: gain set to 1.0 when within `[deadband_min, deadband_max]` and deadband enabled |
| 89 | Deadband disabled path: no modification when `deadband_min == 1.0 and deadband_max == 1.0` |
| 90 | `dt` updated as `dt * gain`, clamped to `[dt_min, dt_max]` |
| 91 | Returns `0` when `dt_new_raw > dt_min` (normal) |
| 92 | Returns `8` when `dt_new_raw <= dt_min` (minimum step reached) |

---

## `step_control/adaptive_PI_controller.py`

### `PIStepControlConfig` (attrs)

| # | Functionality |
|---|--------------|
| 93 | `_kp` defaults `1/18`, validated as float |
| 94 | `_ki` defaults `1/9`, validated as float |
| 95 | `kp` returns `precision(self._kp)` |
| 96 | `ki` returns `precision(self._ki)` |

### `AdaptivePIController`

| # | Functionality |
|---|--------------|
| 97 | `__init__` builds `PIStepControlConfig` via `build_config`, passes to super |
| 98 | `kp` forwards to `compile_settings.kp` |
| 99 | `ki` forwards to `compile_settings.ki` |
| 100 | `local_memory_elements` returns `1` |
| 101 | `settings_dict` extends super with `kp` and `ki` |

#### `build_controller` device function logic

| # | Functionality |
|---|--------------|
| 102 | `kp` and `ki` scaled by `1 / (2 * (algorithm_order + 1))` |
| 103 | Allocates timestep_buffer from buffer_registry |
| 104 | Reads `err_prev` from `timestep_buffer[0]` |
| 105 | Norm computed as mean squared scaled error |
| 106 | Step accepted when `nrm2 <= 1.0` |
| 107 | Proportional gain: `nrm2^(-kp)` |
| 108 | Integral gain uses `err_prev` when initialized (> 0), else uses `nrm2` as fallback |
| 109 | Combined gain: `safety * pgain * igain` |
| 110 | Gain clamped to `[min_gain, max_gain]` |
| 111 | Deadband applied when enabled |
| 112 | `dt` updated and clamped |
| 113 | `timestep_buffer[0]` updated with current `nrm2` |
| 114 | Returns `0` (normal) or `8` (min step reached) |

---

## `step_control/adaptive_PID_controller.py`

### `PIDStepControlConfig` (attrs, extends PIStepControlConfig)

| # | Functionality |
|---|--------------|
| 115 | `_kd` defaults 0.0, validated as float |
| 116 | `kd` returns `precision(self._kd)` |

### `AdaptivePIDController`

| # | Functionality |
|---|--------------|
| 117 | `__init__` builds `PIDStepControlConfig` via `build_config`, passes to super |
| 118 | `kp` forwards to `compile_settings.kp` |
| 119 | `ki` forwards to `compile_settings.ki` |
| 120 | `kd` forwards to `compile_settings.kd` |
| 121 | `local_memory_elements` returns `2` |
| 122 | `settings_dict` extends super with `kp`, `ki`, `kd` |

#### `build_controller` device function logic

| # | Functionality |
|---|--------------|
| 123 | Exponents: `kp / (2*(order+1))`, `ki / (2*(order+1))`, `kd / (2*(order+1))` |
| 124 | Allocates timestep_buffer (2 slots) |
| 125 | Reads `err_prev` from `timestep_buffer[0]`, `err_prev_prev` from `timestep_buffer[1]` |
| 126 | `err_prev_safe` falls back to `nrm2` when uninitialized |
| 127 | `err_prev_prev_safe` falls back to `err_prev_safe` when uninitialized |
| 128 | Gain: `safety * nrm2^(-expo1) * err_prev_safe^(-expo2) * err_prev_prev_safe^(-expo3)` |
| 129 | Gain clamped, deadband applied |
| 130 | `timestep_buffer[1]` = previous `err_prev`, `timestep_buffer[0]` = current `nrm2` |
| 131 | Returns `0` or `8` |

---

## `step_control/gustafsson_controller.py`

### `GustafssonStepControlConfig` (attrs)

| # | Functionality |
|---|--------------|
| 132 | `_gamma` defaults 0.9, validated float in (0, 1) |
| 133 | `_newton_max_iters` defaults 20, validated int >= 0 |
| 134 | `gamma` returns `precision(self._gamma)` |
| 135 | `newton_max_iters` returns `int(self._newton_max_iters)` |
| 136 | `settings_dict` extends super with `gamma` and `newton_max_iters` |

### `GustafssonController`

| # | Functionality |
|---|--------------|
| 137 | `__init__` builds `GustafssonStepControlConfig`, passes to super |
| 138 | `gamma` forwards to `compile_settings.gamma` |
| 139 | `newton_max_iters` forwards to `compile_settings.newton_max_iters` |
| 140 | `local_memory_elements` returns `2` |

#### `build_controller` device function logic

| # | Functionality |
|---|--------------|
| 141 | `expo` = `1 / (2 * (algorithm_order + 1))` |
| 142 | `gain_numerator` = `(1 + 2 * newton_max_iters) * gamma` |
| 143 | Allocates timestep_buffer (2 slots: dt_prev, err_prev) |
| 144 | Reads `dt_prev` and `err_prev` from buffer, floored at 1e-16 |
| 145 | Norm uses 1e-12 floor on error (different from I/PI/PID which use 1e-16) |
| 146 | Step accepted when `nrm2 <= 1.0` |
| 147 | Basic gain: `fac * nrm2^(-expo)` where `fac = min(gamma, gain_numerator / (niters + 2*newton_max_iters))` |
| 148 | Gustafsson gain: `safety * (dt/dt_prev) * (nrm2^2/err_prev)^(-expo) * gamma` |
| 149 | Final gain: min of basic and Gustafsson when accepted and `dt_prev > 1e-16` |
| 150 | Falls back to basic gain when not accepted or `dt_prev` uninitialized |
| 151 | Gain clamped, deadband applied |
| 152 | Buffer updated: `timestep_buffer[0] = current_dt`, `timestep_buffer[1] = nrm2` |
| 153 | Returns `0` or `8` |

---

## `matrix_free_solvers/base_solver.py`

### `MatrixFreeSolverConfig` (attrs)

| # | Functionality |
|---|--------------|
| 154 | `n` defaults 0, validated int >= 1 |
| 155 | `max_iters` defaults 100, validated int in [1, 32767], metadata `prefixed=True` |
| 156 | `norm_device_function` defaults None, `eq=False` |

### `MatrixFreeSolver`

| # | Functionality |
|---|--------------|
| 157 | `__init__` stores `solver_type`, calls super with `instance_label=solver_type`, creates `ScaledNorm` |
| 158 | `atol` forwards to `norm.atol` |
| 159 | `rtol` forwards to `norm.rtol` |
| 160 | `max_iters` forwards to `compile_settings.max_iters` |
| 161 | `n` forwards to `compile_settings.n` |

#### `MatrixFreeSolver.update`

| # | Functionality |
|---|--------------|
| 162 | Merges `updates_dict` and `kwargs` |
| 163 | Empty dict returns empty set |
| 164 | Forwards to `norm.update(silent=True)`, captures recognised |
| 165 | Injects `norm_device_function` into updates |
| 166 | Forwards to `update_compile_settings(silent=True)` |
| 167 | Returns union of recognised sets |

---

## `matrix_free_solvers/linear_solver.py`

### `LinearSolverConfig` (attrs)

| # | Functionality |
|---|--------------|
| 168 | `operator_apply` defaults None, validated optional device, `eq=False` |
| 169 | `preconditioner` defaults None, validated optional device, `eq=False` |
| 170 | `linear_correction_type` defaults `"minimal_residual"`, validated in `["steepest_descent", "minimal_residual"]` |
| 171 | `preconditioned_vec_location` defaults `"local"` |
| 172 | `temp_location` defaults `"local"` |
| 173 | `use_cached_auxiliaries` defaults False |
| 174 | `settings_dict` returns dict with `krylov_max_iters`, `linear_correction_type`, location fields |

### `LinearSolverCache` (attrs)

| # | Functionality |
|---|--------------|
| 175 | `linear_solver` field validated as device function |

### `LinearSolver`

| # | Functionality |
|---|--------------|
| 176 | `__init__` builds `LinearSolverConfig` with `instance_label="krylov"`, calls super with `solver_type="krylov"`, sets up settings and registers buffers |
| 177 | `register_buffers` registers `"preconditioned_vec"` and `"temp"` with configured locations |

#### `build`

| # | Functionality |
|---|--------------|
| 178 | Branch: `use_cached_auxiliaries=True` produces `linear_solver_cached` with `cached_aux` parameter |
| 179 | Branch: `use_cached_auxiliaries=False` produces `linear_solver` without `cached_aux` |
| 180 | Both variants compute initial residual: `rhs = rhs - operator_apply(x)` |
| 181 | Early convergence check: `scaled_norm(rhs, x) <= 1.0` |
| 182 | Warp-synchronous early exit via `all_sync(mask, converged)` |
| 183 | Branch: `preconditioned=True` applies preconditioner to rhs |
| 184 | Branch: `preconditioned=False` copies rhs to preconditioned_vec |
| 185 | Branch: `sd_flag` (steepest descent) computes `numerator = rhs . z`, `denominator = (A*z) . z` |
| 186 | Branch: `mr_flag` (minimal residual) computes `numerator = (A*z) . rhs`, `denominator = (A*z) . (A*z)` |
| 187 | `alpha = numerator / denominator` when denominator nonzero |
| 188 | `alpha = 0` when denominator is zero |
| 189 | Solution update only when thread not already converged |
| 190 | `converged = converged or (norm <= 1.0)` |
| 191 | Returns status `0` on convergence, `4` on max iterations |
| 192 | `krylov_iters_out[0]` set to iteration count |

#### `update`

| # | Functionality |
|---|--------------|
| 193 | Delegates to `super().update(silent=True)` |
| 194 | Updates buffer locations via `buffer_registry.update` |
| 195 | Calls `register_buffers()` |

#### Properties (forwarding)

| # | Property | Delegates to |
|---|----------|-------------|
| 196 | `device_function` | `get_cached_output("linear_solver")` |
| 197 | `linear_correction_type` | `compile_settings.linear_correction_type` |
| 198 | `krylov_atol` | `self.atol` (-> `norm.atol`) |
| 199 | `krylov_rtol` | `self.rtol` (-> `norm.rtol`) |
| 200 | `krylov_max_iters` | `self.max_iters` |
| 201 | `use_cached_auxiliaries` | `compile_settings.use_cached_auxiliaries` |
| 202 | `settings_dict` | merges `compile_settings.settings_dict` + `krylov_atol` + `krylov_rtol` |

---

## `matrix_free_solvers/newton_krylov.py`

### `NewtonKrylovConfig` (attrs)

| # | Functionality |
|---|--------------|
| 203 | `residual_function` defaults None, `eq=False` |
| 204 | `linear_solver_function` defaults None, `eq=False` |
| 205 | `_newton_damping` defaults 0.5, validated float in (0, 1) |
| 206 | `newton_max_backtracks` defaults 8, validated int in [1, 32767] |
| 207 | `delta_location` defaults `"local"` |
| 208 | `residual_location` defaults `"local"` |
| 209 | `residual_temp_location` defaults `"local"` |
| 210 | `stage_base_bt_location` defaults `"local"` |
| 211 | `krylov_iters_local_location` defaults `"local"` |
| 212 | `newton_damping` returns `precision(self._newton_damping)` |
| 213 | `settings_dict` returns dict with newton_max_iters, newton_damping, newton_max_backtracks, location fields |

### `NewtonKrylovCache` (attrs)

| # | Functionality |
|---|--------------|
| 214 | `newton_krylov_solver` field validated as device function |

### `NewtonKrylov`

| # | Functionality |
|---|--------------|
| 215 | `__init__` builds `NewtonKrylovConfig` with `instance_label="newton"`, stores `linear_solver`, sets up settings and registers buffers |
| 216 | `register_buffers` registers `delta`, `residual`, `residual_temp`, `stage_base_bt` (all config.n), `krylov_iters_local` (size 1, precision `np_int32`) |

#### `build` device function logic

| # | Functionality |
|---|--------------|
| 217 | Computes initial residual via `residual_function`, negates it |
| 218 | Initial convergence check: `scaled_norm(residual, stage_increment) <= 1.0` |
| 219 | Warp-synchronous exit via `all_sync(mask, converged)` |
| 220 | Active threads increment `iters_count` via `selp` |
| 221 | Calls `linear_solver_fn` to solve for delta |
| 222 | Accumulates `total_krylov_iters` for active threads |
| 223 | Backtracking loop with `alpha` starting at 1.0, multiplied by damping each iteration |
| 224 | `max_backtracks = config.newton_max_backtracks + 1` (off-by-one correction) |
| 225 | Backtrack inner loop uses `any_sync` for warp-synchronous check |
| 226 | Convergence: `norm2_new <= 1.0` sets both `converged` and `found_step` |
| 227 | Sufficient decrease: `norm2_new < norm2_prev` sets `found_step`, updates residual and norm |
| 228 | Backtrack failure reverts `stage_increment` to `stage_base_bt` |
| 229 | Status bit `2` set when not converged at exit |
| 230 | Status bit `1` set when last backtrack failed |
| 231 | Status ORed with `last_lin_status` when linear solver signaled non-zero |
| 232 | `counters[0] = iters_count`, `counters[1] = total_krylov_iters` |

#### `update`

| # | Functionality |
|---|--------------|
| 233 | Forwards krylov-prefixed params to `linear_solver.update(silent=True)` |
| 234 | Injects `linear_solver_function` from linear solver's device_function |
| 235 | Delegates to `super().update(silent=True)` for norm and compile settings |
| 236 | Updates buffer locations via `buffer_registry.update` |
| 237 | Calls `register_buffers()` |

#### Properties (forwarding)

| # | Property | Delegates to |
|---|----------|-------------|
| 238 | `device_function` | `get_cached_output("newton_krylov_solver")` |
| 239 | `newton_atol` | `norm.atol` |
| 240 | `newton_rtol` | `norm.rtol` |
| 241 | `newton_max_iters` | `self.max_iters` |
| 242 | `newton_damping` | `compile_settings.newton_damping` |
| 243 | `newton_max_backtracks` | `compile_settings.newton_max_backtracks` |
| 244 | `krylov_atol` | `linear_solver.atol` |
| 245 | `krylov_rtol` | `linear_solver.rtol` |
| 246 | `krylov_max_iters` | `linear_solver.max_iters` |
| 247 | `linear_correction_type` | `linear_solver.linear_correction_type` |
| 248 | `settings_dict` | merges `linear_solver.settings_dict` + `compile_settings.settings_dict` + `newton_atol` + `newton_rtol` |

---

## `norms.py`

### `resize_tolerances` (module-level function)

| # | Functionality |
|---|--------------|
| 249 | Sets `_n_changing = True` on instance during resize |
| 250 | For each of `atol`, `rtol`: skips if length already matches `n` |
| 251 | Expands uniform tolerance arrays (all equal values) to new size `n` |
| 252 | Leaves non-uniform arrays unchanged |
| 253 | Sets `_n_changing = False` after resize |

### `ScaledNormConfig` (attrs)

| # | Functionality |
|---|--------------|
| 254 | `n` defaults 1, validated int >= 1, `on_setattr=resize_tolerances` |
| 255 | `atol` defaults `[1e-6]`, prefixed, converter `tol_converter` |
| 256 | `rtol` defaults `[1e-6]`, prefixed, converter `tol_converter` |
| 257 | `_n_changing` internal field, not in init, not in eq |
| 258 | `inv_n` returns `precision(1.0 / n)` |
| 259 | `tol_floor` returns `precision(1e-16)` |

### `ScaledNormCache` (attrs)

| # | Functionality |
|---|--------------|
| 260 | `scaled_norm` field validated as device function |

### `ScaledNorm`

| # | Functionality |
|---|--------------|
| 261 | `__init__` calls super with `instance_label`, builds `ScaledNormConfig`, sets up settings |

#### `build` device function logic

| # | Functionality |
|---|--------------|
| 262 | For each element: `tol_i = atol[i] + rtol[i] * |reference[i]|` |
| 263 | `tol_i` floored at `1e-16` to avoid division by zero |
| 264 | Computes `sum(|values[i]| / tol_i)^2 / n` |
| 265 | Returns mean squared scaled norm |

#### `update`

| # | Functionality |
|---|--------------|
| 266 | Merges updates_dict and kwargs |
| 267 | Empty dict returns empty set |
| 268 | Delegates to `update_compile_settings` |

#### Properties (forwarding)

| # | Property | Delegates to |
|---|----------|-------------|
| 269 | `device_function` | `get_cached_output("scaled_norm")` |
| 270 | `precision` | `compile_settings.precision` |
| 271 | `n` | `compile_settings.n` |
| 272 | `atol` | `compile_settings.atol` |
| 273 | `rtol` | `compile_settings.rtol` |

---

## `loops/ode_loop_config.py`

### `ODELoopConfig` (attrs)

| # | Functionality |
|---|--------------|
| 274 | `n_states` defaults 0, validated int >= 0 |
| 275 | `n_parameters` defaults 0 |
| 276 | `n_drivers` defaults 0 |
| 277 | `n_observables` defaults 0 |
| 278 | `n_error` defaults 0 |
| 279 | `n_counters` defaults 0 |
| 280 | `state_summaries_buffer_height` defaults 0 |
| 281 | `observable_summaries_buffer_height` defaults 0 |
| 282 | 14 buffer location fields, each defaults `"local"`, validated in `["shared", "local"]` |
| 283 | `compile_flags` factory `OutputCompileFlags` |
| 284 | `_save_every` defaults None, optional float > 0 |
| 285 | `_summarise_every` defaults None, optional float > 0 |
| 286 | `_sample_summaries_every` defaults None, optional float > 0 |
| 287 | `save_last` defaults False |
| 288 | `save_regularly` defaults False |
| 289 | `summarise_regularly` defaults False |
| 290 | 7 device function fields (save_state_fn, update_summaries_fn, save_summaries_fn, step_controller_fn, step_function, evaluate_driver_at_t, evaluate_observables), each optional, `eq=False` |
| 291 | `_dt0` defaults 0.01, optional float > 0 |
| 292 | `is_adaptive` defaults False |

#### `samples_per_summary` property

| # | Functionality |
|---|--------------|
| 293 | Returns 0 when either `summarise_every` or `sample_summaries_every` is None |
| 294 | Computes integer ratio `round(summarise_every / sample_summaries_every)` |
| 295 | Warning emitted when adjusted value differs from raw `_summarise_every` (deviation <= 0.01) |
| 296 | `ValueError` raised when deviation > 0.01 (not integer multiple) |

#### Precision properties

| # | Functionality |
|---|--------------|
| 297 | `save_every` returns `precision(self._save_every)` or None |
| 298 | `summarise_every` returns `precision(self._summarise_every)` or None |
| 299 | `sample_summaries_every` returns `precision(self._sample_summaries_every)` or None |
| 300 | `dt0` returns `precision(self._dt0)` |

---

## `loops/ode_loop.py`

### `IVPLoopCache` (attrs)

| # | Functionality |
|---|--------------|
| 301 | `loop_function` field |

### `IVPLoop`

| # | Functionality |
|---|--------------|
| 302 | `__init__` builds `ODELoopConfig` via `build_config` with all named + kwargs, calls `setup_compile_settings` and `register_buffers` |
| 303 | `register_buffers` registers 15 buffers: state, proposed_state, parameters, drivers, proposed_drivers, observables, proposed_observables, error, counters, state_summary, observable_summary, dt, accept_step, proposed_counters (int32 precision) |

#### `build` device function logic

| # | Functionality |
|---|--------------|
| 304 | Initialises `t` from `t0` in float64 |
| 305 | Clears `persistent_local` and `shared_scratch` to zero on entry |
| 306 | Copies initial_states and parameters into local buffers |
| 307 | Evaluates drivers at `t0` when `evaluate_driver_at_t is not None and n_drivers > 0` |
| 308 | Evaluates observables at `t0` when `n_observables > 0` |
| 309 | When `settling_time == 0`: saves initial state, advances next_save and next_update_summary |
| 310 | When `settling_time == 0` and `summarise`: calls `save_summaries` to reset buffers |
| 311 | Main loop: finish condition depends on `save_regularly`/`summarise_regularly` flags |
| 312 | When neither save nor summarise regularly: finishes when `end_of_step > t_end` |
| 313 | `save_last`: `at_end` triggers final save when `t_prec < t_end` and otherwise finished |
| 314 | `irrecoverable` forces finish |
| 315 | Warp-synchronous exit via `all_sync(mask, finished)` |
| 316 | `do_save` computed from `end_of_step >= next_save` when save_regularly |
| 317 | `do_update_summary` computed from `end_of_step >= next_update_summary` when summarise_regularly |
| 318 | `dt_eff` adjusted to hit output boundary exactly when saving or summarising |
| 319 | Step function called with all buffers |
| 320 | `first_step_flag` cleared after first step |
| 321 | Step status ORed into cumulative status |
| 322 | Fixed mode: step failure is irrecoverable |
| 323 | Adaptive mode: step failure forces error to 1e16 (rejection) |
| 324 | Controller called in adaptive mode; acceptance from `accept_step[0]` AND no step failure |
| 325 | Fixed mode: accept = not step_failed |
| 326 | Controller status bit `0x8` triggers irrecoverable |
| 327 | Counter accumulation when `save_counters_bool`: newton iters (i<2), total steps (i==2), rejected steps (i==3 and not accept) |
| 328 | Stagnation detection: 2 consecutive `t_proposal == t` triggers status `0x40` and irrecoverable |
| 329 | State, drivers, observables committed via `selp(accept, new, old)` |
| 330 | Output gated on `accept` |
| 331 | `next_save` incremented by `save_every` after save |
| 332 | Counters reset after save |
| 333 | `next_update_summary` incremented by `sample_summaries_every` |
| 334 | `save_summaries` called when `update_idx % samples_per_summary == 0` |
| 335 | `summary_idx` incremented after saving summaries |

#### Properties (forwarding)

| # | Property | Delegates to |
|---|----------|-------------|
| 336 | `save_every` | `compile_settings.save_every` |
| 337 | `summarise_every` | `compile_settings.summarise_every` |
| 338 | `sample_summaries_every` | `compile_settings.sample_summaries_every` |
| 339 | `compile_flags` | `compile_settings.compile_flags` |
| 340 | `device_function` | `get_cached_output("loop_function")` |
| 341 | `save_state_fn` | `compile_settings.save_state_fn` |
| 342 | `update_summaries_fn` | `compile_settings.update_summaries_fn` |
| 343 | `save_summaries_fn` | `compile_settings.save_summaries_fn` |
| 344 | `step_controller_fn` | `compile_settings.step_controller_fn` |
| 345 | `step_function` | `compile_settings.step_function` |
| 346 | `evaluate_driver_at_t` | `compile_settings.evaluate_driver_at_t` |
| 347 | `evaluate_observables` | `compile_settings.evaluate_observables` |
| 348 | `dt0` | `compile_settings.dt0` |
| 349 | `is_adaptive` | `compile_settings.is_adaptive` |

#### `update`

| # | Functionality |
|---|--------------|
| 350 | Defaults `updates_dict` to `{}` when None |
| 351 | Merges kwargs |
| 352 | Empty dict returns empty set |
| 353 | Flattens nested dict values via `unpack_dict_values` |
| 354 | Delegates to `update_compile_settings(silent=True)` |
| 355 | Updates buffer locations via `buffer_registry.update(silent=True)` |
| 356 | Calls `register_buffers()` |
| 357 | `KeyError` raised for unrecognised keys when `silent=False` |
| 358 | No error when `silent=True` |
| 359 | Returns recognised union unpacked_keys |

---

## `array_interpolator.py`

### `InterpolatorCache` (attrs)

| # | Functionality |
|---|--------------|
| 360 | `evaluation_function` defaults None |
| 361 | `driver_del_t` defaults None |

### `ArrayInterpolatorConfig` (attrs)

| # | Functionality |
|---|--------------|
| 362 | `order` defaults 3, validated int > 0 |
| 363 | `wrap` defaults True, validated bool |
| 364 | `boundary_condition` defaults `"not-a-knot"`, validated in `{"natural", "periodic", "not-a-knot", "clamped"}` |
| 365 | `dt` init=False, defaults 1e-16, validated float > 0 |
| 366 | `t0` defaults 0.0, validated float >= 0 |
| 367 | `num_inputs` init=False, defaults 0 |
| 368 | `num_segments` init=False, defaults 0 |

### `ArrayInterpolator`

| # | Functionality |
|---|--------------|
| 369 | `__init__` creates config with precision only, stores `_coefficients=None`, `_input_array=None`, calls `update_from_dict` |

#### `update_from_dict`

| # | Functionality |
|---|--------------|
| 370 | Splits input_dict into config keys, input keys, and time keys |
| 371 | Updates compile settings with config keys |
| 372 | Normalises input array via `_normalise_input_array` |
| 373 | Returns False if input array unchanged |
| 374 | Validates time inputs via `_validate_time_inputs` |
| 375 | When `wrap=True` and no boundary_condition given: defaults to `"periodic"` |
| 376 | When `wrap=False` and no boundary_condition given: defaults to `"clamped"`, `num_segments = base + 2` |
| 377 | When `wrap=False` and boundary_condition `"clamped"`: `num_segments = base + 2` |
| 378 | When `wrap=False` and boundary_condition not clamped: `num_segments = base` |
| 379 | Computes coefficients via `_compute_coefficients` |
| 380 | Returns True when config or input changed |

#### `_normalise_input_array`

| # | Functionality |
|---|--------------|
| 381 | Converts each array to precision dtype |
| 382 | `ValueError` if array cannot be converted |
| 383 | `ValueError` if any array is not 1D |
| 384 | `ValueError` if arrays have different lengths |
| 385 | `ValueError` if fewer than `order + 1` samples |
| 386 | Returns column-stacked array |

#### `_validate_time_inputs`

| # | Functionality |
|---|--------------|
| 387 | `ValueError` if both `dt` and `time` provided |
| 388 | `dt` path: uses dt directly, t0 from dict or defaults to 0.0 |
| 389 | `time` path: `ValueError` if not 1D |
| 390 | `time` path: `ValueError` if length mismatch with num_samples |
| 391 | `time` path: `ValueError` if not strictly increasing |
| 392 | `time` path: `ValueError` if not uniformly spaced |
| 393 | `time` path: extracts dt from differences, t0 from first element |
| 394 | `ValueError` if neither dt nor time provided |

#### `build` device function logic

| # | Functionality |
|---|--------------|
| 395 | `evaluate_all`: wrapping path uses `idx % num_segments` |
| 396 | `evaluate_all`: non-wrapping path clamps segment index, returns zero outside range |
| 397 | `evaluate_all`: Horner's rule evaluation for each input polynomial |
| 398 | `evaluate_time_derivative`: same wrap/no-wrap logic |
| 399 | `evaluate_time_derivative`: derivative Horner's rule, scaled by `inv_resolution` |
| 400 | Clamped non-wrap: `evaluation_start` offset by `-resolution` |

#### `update`

| # | Functionality |
|---|--------------|
| 401 | Empty dict returns empty set |
| 402 | Delegates to `update_compile_settings(silent=True)` |
| 403 | `KeyError` for unrecognised keys when `silent=False` |

#### Properties (forwarding)

| # | Property | Delegates to |
|---|----------|-------------|
| 404 | `evaluation_function` | `get_cached_output("evaluation_function")` |
| 405 | `driver_del_t` | `get_cached_output("driver_del_t")` |
| 406 | `coefficients` | `self._coefficients` |
| 407 | `num_inputs` | `input_array.shape[1]` |
| 408 | `num_samples` | `input_array.shape[0]` |
| 409 | `input_array` | `self._input_array` |
| 410 | `order` | `compile_settings.order` |
| 411 | `wrap` | `compile_settings.wrap` |
| 412 | `boundary_condition` | `compile_settings.boundary_condition` |
| 413 | `num_segments` | `compile_settings.num_segments` |
| 414 | `t0` | `compile_settings.t0` |
| 415 | `dt` | `compile_settings.dt` |

#### `get_input_array`

| # | Functionality |
|---|--------------|
| 416 | Returns `self._input_array` |

#### `get_interpolated`

| # | Functionality |
|---|--------------|
| 417 | `ValueError` if `eval_times` not 1D |
| 418 | Returns empty array if `eval_times` is empty |
| 419 | `RuntimeError` if coefficients are None |
| 420 | Launches CUDA kernel to evaluate all times, returns host array |

#### `plot_interpolated`

| # | Functionality |
|---|--------------|
| 421 | `ImportError` if matplotlib not available |
| 422 | `ValueError` if `eval_times` not 1D |
| 423 | Wrapping mode: tiles sample markers across repeated periods |
| 424 | Non-wrapping mode: uses original sample times |
| 425 | Plots interpolated curves + sample markers for each input |
| 426 | Legend shown when `num_inputs > 1` |

#### `check_against_system_drivers` (static)

| # | Functionality |
|---|--------------|
| 427 | `ValueError` if number of inputs != number of system drivers |
| 428 | `ValueError` if input key set != system driver symbol set |

#### `_compute_coefficients`

| # | Functionality |
|---|--------------|
| 429 | `ValueError` for unsupported boundary condition |
| 430 | Clamped non-wrap: pads inputs with zero rows |
| 431 | Periodic: `ValueError` if `wrap=False` |
| 432 | Periodic: `ValueError` if first and last samples don't match |
| 433 | Builds tridiagonal-like system with function value constraints at both edges of each segment |
| 434 | Interior derivative continuity constraints for orders 1..order-1 |
| 435 | Natural BC: sets highest derivatives to zero at endpoints |
| 436 | Periodic BC: wraps derivative continuity from last to first segment |
| 437 | Clamped BC: sets first derivative to zero at endpoints |
| 438 | Not-a-knot BC: finite difference constraints at start and end |
| 439 | `ValueError` if assembled system is not square (row_index != num_coeffs) |
| 440 | Solves linear system, reshapes to (num_segments, num_inputs, order+1) |


### Output Handling

| # | Source File | Inventory |
|---|------------|-----------|
| 68 | `outputhandling/output_config.py` | [x] |
| 69 | `outputhandling/output_sizes.py` | [x] |
| 70 | `outputhandling/output_functions.py` | [x] |
| 71 | `outputhandling/save_state.py` | [x] |
| 72 | `outputhandling/save_summaries.py` | [x] |
| 73 | `outputhandling/update_summaries.py` | [x] |
| 74 | `outputhandling/__init__.py` | [x] |


### Summary Metrics

| # | Source File | Inventory |
|---|------------|-----------|
| 75 | `summarymetrics/metrics.py` | [x] |
| 76 | `summarymetrics/__init__.py` | [x] |
| 77-96 | `summarymetrics/{mean,max,min,std,rms,peaks,negative_peaks,extrema,max_magnitude,mean_std,mean_std_rms,std_rms,dxdt_max,dxdt_min,dxdt_extrema,d2xdt2_max,d2xdt2_min,d2xdt2_extrema,final_state,dxdt_final_state}.py` | [x] |
| 97 | `summarymetrics/d2xdt2_final_state.py` | [x] |


## `outputhandling/output_config.py`

### `_indices_validator` — Module-Level Function

| # | Functionality |
|---|--------------|
| 1 | Returns without error when `array` is `None` |
| 2 | Raises `TypeError` when array is not an ndarray |
| 3 | Raises `TypeError` when array dtype is not `np.int_` |
| 4 | Raises `ValueError` when any index is negative |
| 5 | Raises `ValueError` when any index >= `max_index` |
| 6 | Raises `ValueError` listing duplicated indices when duplicates present |
| 7 | Passes without error for valid unique in-range array |

### `OutputCompileFlags` — attrs class

| # | Functionality |
|---|--------------|
| 8 | Construction with all defaults yields all-False flags |
| 9 | Each boolean attribute validated as `instance_of(bool)` |
| 10 | `__attrs_post_init__` calls super (inherits `_CubieConfigBase` hashing) |

### `OutputConfig.__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 11 | Calls `validation_passes()` after construction |

### `OutputConfig.validation_passes`

| # | Functionality |
|---|--------------|
| 12 | Calls `update_from_outputs_list` with current `_output_types` |
| 13 | Calls `_check_saved_indices` |
| 14 | Calls `_check_summarised_indices` |
| 15 | Calls `_validate_index_arrays` |
| 16 | Calls `_check_for_no_outputs` |

### `OutputConfig._validate_index_arrays`

| # | Functionality |
|---|--------------|
| 17 | Validates all four index arrays against their respective maxima |
| 18 | State indices validated against `_max_states` |
| 19 | Observable indices validated against `_max_observables` |

### `OutputConfig._check_for_no_outputs`

| # | Functionality |
|---|--------------|
| 20 | Raises `ValueError` when no output types are enabled |
| 21 | Passes when at least one of save_state, save_observables, save_time, save_counters, save_summaries is True |

### `OutputConfig._check_saved_indices`

| # | Functionality |
|---|--------------|
| 22 | Converts `_saved_state_indices` to numpy int array |
| 23 | Converts `_saved_observable_indices` to numpy int array |

### `OutputConfig._check_summarised_indices`

| # | Functionality |
|---|--------------|
| 24 | Converts `_summarised_state_indices` to numpy int array |
| 25 | Converts `_summarised_observable_indices` to numpy int array |

### `OutputConfig.max_states` — Property + Setter

| # | Functionality |
|---|--------------|
| 26 | Getter returns `_max_states` |
| 27 | Setter: when saved indices span full range, expands to new size |
| 28 | Setter: when saved indices do NOT span full range, leaves them unchanged |
| 29 | Setter: updates `_max_states` and re-runs `__attrs_post_init__` |

### `OutputConfig.max_observables` — Property + Setter

| # | Functionality |
|---|--------------|
| 30 | Getter returns `_max_observables` |
| 31 | Setter: when saved observable indices span full range, expands to new size |
| 32 | Setter: when saved observable indices do NOT span full range, leaves them unchanged |
| 33 | Setter: updates `_max_observables` and re-runs `__attrs_post_init__` |

### `OutputConfig.save_state` — Calculated Property

| # | Functionality |
|---|--------------|
| 34 | Returns True when `_save_state` is True AND saved state indices non-empty |
| 35 | Returns False when `_save_state` is True AND saved state indices empty |
| 36 | Returns False when `_save_state` is False |

### `OutputConfig.save_observables` — Calculated Property

| # | Functionality |
|---|--------------|
| 37 | Returns True when `_save_observables` is True AND saved observable indices non-empty |
| 38 | Returns False when `_save_observables` is True AND observable indices empty |
| 39 | Returns False when `_save_observables` is False |

### `OutputConfig.save_time` — Forwarding Property

| # | Functionality |
|---|--------------|
| 40 | Returns `_save_time` |

### `OutputConfig.save_counters` — Forwarding Property

| # | Functionality |
|---|--------------|
| 41 | Returns `_save_counters` |

### `OutputConfig.save_summaries` — Calculated Property

| # | Functionality |
|---|--------------|
| 42 | Returns True when `_summary_types` is non-empty |
| 43 | Returns False when `_summary_types` is empty |

### `OutputConfig.summarise_state` — Calculated Property

| # | Functionality |
|---|--------------|
| 44 | Returns True when save_summaries True AND n_summarised_states > 0 |
| 45 | Returns False when save_summaries False |
| 46 | Returns False when n_summarised_states == 0 |

### `OutputConfig.summarise_observables` — Calculated Property

| # | Functionality |
|---|--------------|
| 47 | Returns True when save_summaries True AND n_summarised_observables > 0 |
| 48 | Returns False when save_summaries False |
| 49 | Returns False when n_summarised_observables == 0 |

### `OutputConfig.compile_flags` — Calculated Property

| # | Functionality |
|---|--------------|
| 50 | Returns `OutputCompileFlags` with fields derived from current config |
| 51 | `save_state` flag matches `self.save_state` |
| 52 | `save_observables` flag matches `self.save_observables` |
| 53 | `summarise` flag matches `self.save_summaries` |
| 54 | `summarise_observables` flag matches `self.summarise_observables` |
| 55 | `summarise_state` flag matches `self.summarise_state` |
| 56 | `save_counters` flag matches `self.save_counters` |

### `OutputConfig.saved_state_indices` — Property + Setter

| # | Functionality |
|---|--------------|
| 57 | Getter returns empty array when `_save_state` is False |
| 58 | Getter returns `_saved_state_indices` when `_save_state` is True |
| 59 | Setter converts to numpy int array, validates, checks no-outputs |

### `OutputConfig.saved_observable_indices` — Property + Setter

| # | Functionality |
|---|--------------|
| 60 | Getter returns empty array when `_save_observables` is False |
| 61 | Getter returns `_saved_observable_indices` when `_save_observables` is True |
| 62 | Setter converts to numpy int array, validates, checks no-outputs |

### `OutputConfig.summarised_state_indices` — Property + Setter

| # | Functionality |
|---|--------------|
| 63 | Getter returns empty array when `save_summaries` is False |
| 64 | Getter returns `_summarised_state_indices` when `save_summaries` is True |
| 65 | Setter converts to numpy int array, validates, checks no-outputs |

### `OutputConfig.summarised_observable_indices` — Property + Setter

| # | Functionality |
|---|--------------|
| 66 | Getter returns empty array when `save_summaries` is False |
| 67 | Getter returns `_summarised_observable_indices` when `save_summaries` is True |
| 68 | Setter converts to numpy int array, validates, checks no-outputs |

### `OutputConfig.n_saved_states` — Calculated Property

| # | Functionality |
|---|--------------|
| 69 | Returns length of `_saved_state_indices` when `_save_state` True |
| 70 | Returns 0 when `_save_state` False |

### `OutputConfig.n_saved_observables` — Calculated Property

| # | Functionality |
|---|--------------|
| 71 | Returns length of `_saved_observable_indices` when `_save_observables` True |
| 72 | Returns 0 when `_save_observables` False |

### `OutputConfig.n_summarised_states` — Calculated Property

| # | Functionality |
|---|--------------|
| 73 | Returns length of `_summarised_state_indices` when `save_summaries` True |
| 74 | Returns 0 when `save_summaries` False |

### `OutputConfig.n_summarised_observables` — Calculated Property

| # | Functionality |
|---|--------------|
| 75 | Returns length of `_summarised_observable_indices` when `save_summaries` True |
| 76 | Returns 0 when `save_summaries` False |

### `OutputConfig.summary_types` — Forwarding Property

| # | Functionality |
|---|--------------|
| 77 | Returns `_summary_types` tuple |

### `OutputConfig.summary_legend_per_variable` — Calculated Property

| # | Functionality |
|---|--------------|
| 78 | Returns empty dict when `_summary_types` is empty |
| 79 | Returns dict mapping indices to metric legend strings when types present |

### `OutputConfig.summary_unit_modifications` — Calculated Property

| # | Functionality |
|---|--------------|
| 80 | Returns empty dict when `_summary_types` is empty |
| 81 | Returns dict mapping indices to unit modification strings when types present |

### `OutputConfig.sample_summaries_every` — Forwarding Property

| # | Functionality |
|---|--------------|
| 82 | Returns `_sample_summaries_every` |

### `OutputConfig.summaries_buffer_height_per_var` — Calculated Property

| # | Functionality |
|---|--------------|
| 83 | Returns 0 when `summary_types` is empty |
| 84 | Delegates to `summary_metrics.summaries_buffer_height` when types present |

### `OutputConfig.summaries_output_height_per_var` — Calculated Property

| # | Functionality |
|---|--------------|
| 85 | Returns 0 when `_summary_types` is empty |
| 86 | Delegates to `summary_metrics.summaries_output_height` when types present |

### `OutputConfig.state_summaries_buffer_height` — Calculated Property

| # | Functionality |
|---|--------------|
| 87 | Returns `summaries_buffer_height_per_var * n_summarised_states` |

### `OutputConfig.observable_summaries_buffer_height` — Calculated Property

| # | Functionality |
|---|--------------|
| 88 | Returns `summaries_buffer_height_per_var * n_summarised_observables` |

### `OutputConfig.total_summary_buffer_size` — Calculated Property

| # | Functionality |
|---|--------------|
| 89 | Returns sum of state and observable summary buffer heights |

### `OutputConfig.state_summaries_output_height` — Calculated Property

| # | Functionality |
|---|--------------|
| 90 | Returns `summaries_output_height_per_var * n_summarised_states` |

### `OutputConfig.observable_summaries_output_height` — Calculated Property

| # | Functionality |
|---|--------------|
| 91 | Returns `summaries_output_height_per_var * n_summarised_observables` |

### `OutputConfig.buffer_sizes_dict` — Calculated Property

| # | Functionality |
|---|--------------|
| 92 | Returns dict with keys: n_saved_states, n_saved_observables, n_summarised_states, n_summarised_observables, state/observable buffer heights, total_summary_buffer_size, state/observable output heights, compile_flags |

### `OutputConfig.output_types` — Property + Setter

| # | Functionality |
|---|--------------|
| 93 | Getter returns `_output_types` list |
| 94 | Setter accepts tuple (converts to list) |
| 95 | Setter accepts string (wraps in list) |
| 96 | Setter raises `TypeError` for unsupported type |
| 97 | Setter calls `update_from_outputs_list` then `_check_for_no_outputs` |

### `OutputConfig.update_from_outputs_list`

| # | Functionality |
|---|--------------|
| 98 | Empty list: clears all flags and summary types |
| 99 | Non-empty: sets `_save_state` True when "state" in list |
| 100 | Non-empty: sets `_save_observables` True when "observables" in list |
| 101 | Non-empty: sets `_save_time` True when "time" in list |
| 102 | Non-empty: sets `_save_counters` True when "iteration_counters" in list |
| 103 | Non-empty: collects summary types matching `implemented_metrics` prefixes |
| 104 | Non-empty: warns for unrecognised output types |
| 105 | Non-empty: calls `_check_for_no_outputs` |

### `OutputConfig.from_loop_settings` — Classmethod

| # | Functionality |
|---|--------------|
| 106 | Converts None indices to empty numpy arrays |
| 107 | Copies output_types list before modifying |
| 108 | Passes all parameters through to constructor |

---

## `outputhandling/output_sizes.py`

### `ArraySizingClass.nonzero` — Property

| # | Functionality |
|---|--------------|
| 109 | Returns new object with all int/tuple fields coerced to at least 1 |
| 110 | Does not mutate the original object |

### `OutputArrayHeights` — attrs class

| # | Functionality |
|---|--------------|
| 111 | Construction with defaults yields all-1 heights |

### `OutputArrayHeights.from_output_fns` — Classmethod

| # | Functionality |
|---|--------------|
| 112 | state height = n_saved_states + 1 when save_time True |
| 113 | state height = n_saved_states when save_time False |
| 114 | observables height = n_saved_observables |
| 115 | state_summaries = state_summaries_output_height |
| 116 | observable_summaries = observable_summaries_output_height |
| 117 | per_variable = summaries_output_height_per_var |

### `SingleRunOutputSizes.from_solver` — Classmethod

| # | Functionality |
|---|--------------|
| 118 | state shape = (output_samples, heights.state) |
| 119 | observables shape = (output_samples, heights.observables) |
| 120 | state_summaries shape = (summarise_samples, heights.state_summaries) |
| 121 | observable_summaries shape = (summarise_samples, heights.observable_summaries) |

### `BatchInputSizes.from_solver` — Classmethod

| # | Functionality |
|---|--------------|
| 122 | initial_values = (states, num_runs) |
| 123 | parameters = (parameters, num_runs) |
| 124 | driver_coefficients = (None, drivers, None) |

### `BatchOutputSizes.from_solver` — Classmethod

| # | Functionality |
|---|--------------|
| 125 | Adds num_runs as third dimension to all single-run shapes |
| 126 | status_codes = (num_runs,) |
| 127 | iteration_counters = (n_saves, 4, num_runs) |

---

## `outputhandling/output_functions.py`

### `OutputFunctionCache` — attrs class

| # | Functionality |
|---|--------------|
| 128 | Stores save_state_function, update_summaries_function, save_summaries_function |

### `OutputFunctions.__init__`

| # | Functionality |
|---|--------------|
| 129 | Defaults `output_types` to `["state"]` when None |
| 130 | Creates `OutputConfig.from_loop_settings` with all parameters |
| 131 | Calls `setup_compile_settings` with the config |

### `OutputFunctions.update`

| # | Functionality |
|---|--------------|
| 132 | Merges `updates_dict` and `kwargs` |
| 133 | Returns empty set for empty dict |
| 134 | Delegates to `update_compile_settings(silent=True)` |
| 135 | Calls `validation_passes()` after update |
| 136 | Raises `KeyError` for unrecognised params when `silent=False` |
| 137 | Suppresses KeyError when `silent=True` |

### `OutputFunctions.build`

| # | Functionality |
|---|--------------|
| 138 | Updates `summary_metrics` with sample_summaries_every and precision |
| 139 | Calls `save_state_factory` with config indices and flags |
| 140 | Calls `update_summary_factory` with config summary parameters |
| 141 | Calls `save_summary_factory` with config summary parameters |
| 142 | Returns `OutputFunctionCache` with all three functions |

### `OutputFunctions` — Forwarding Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 143 | `save_state_func` | `get_cached_output("save_state_function")` |
| 144 | `update_summaries_func` | `get_cached_output("update_summaries_function")` |
| 145 | `save_summary_metrics_func` | `get_cached_output("save_summaries_function")` |
| 146 | `output_types` | `compile_settings.output_types` |
| 147 | `compile_flags` | `compile_settings.compile_flags` |
| 148 | `save_time` | `compile_settings.save_time` |
| 149 | `saved_state_indices` | `compile_settings.saved_state_indices` |
| 150 | `saved_observable_indices` | `compile_settings.saved_observable_indices` |
| 151 | `summarised_state_indices` | `compile_settings.summarised_state_indices` |
| 152 | `summarised_observable_indices` | `compile_settings.summarised_observable_indices` |
| 153 | `n_saved_states` | `compile_settings.n_saved_states` |
| 154 | `n_saved_observables` | `compile_settings.n_saved_observables` |
| 155 | `state_summaries_output_height` | `compile_settings.state_summaries_output_height` |
| 156 | `observable_summaries_output_height` | `compile_settings.observable_summaries_output_height` |
| 157 | `summaries_buffer_height_per_var` | `compile_settings.summaries_buffer_height_per_var` |
| 158 | `state_summaries_buffer_height` | `compile_settings.state_summaries_buffer_height` |
| 159 | `observable_summaries_buffer_height` | `compile_settings.observable_summaries_buffer_height` |
| 160 | `summaries_output_height_per_var` | `compile_settings.summaries_output_height_per_var` |
| 161 | `summary_legend_per_variable` | `compile_settings.summary_legend_per_variable` |
| 162 | `summary_unit_modifications` | `compile_settings.summary_unit_modifications` |
| 163 | `buffer_sizes_dict` | `compile_settings.buffer_sizes_dict` |

### `OutputFunctions.has_time_domain_outputs` — Calculated Property

| # | Functionality |
|---|--------------|
| 164 | Returns True when save_time True |
| 165 | Returns True when save_state True |
| 166 | Returns True when save_observables True |
| 167 | Returns False when all three False |

### `OutputFunctions.has_summary_outputs` — Calculated Property

| # | Functionality |
|---|--------------|
| 168 | Returns True when summarise_state True |
| 169 | Returns True when summarise_observables True |
| 170 | Returns False when neither is True |

### `OutputFunctions.output_array_heights` — Calculated Property

| # | Functionality |
|---|--------------|
| 171 | Returns `OutputArrayHeights.from_output_fns(self)` |

---

## `outputhandling/save_state.py`

### `save_state_factory`

| # | Functionality |
|---|--------------|
| 172 | Generated function copies selected states when `save_state` True |
| 173 | Generated function skips state copy when `save_state` False |
| 174 | Generated function appends time at position nstates when `save_time` True |
| 175 | Generated function skips time when `save_time` False |
| 176 | Generated function copies selected observables when `save_observables` True |
| 177 | Generated function skips observables when `save_observables` False |
| 178 | Generated function copies 4 counters when `save_counters` True |
| 179 | Generated function skips counters when `save_counters` False |
| 180 | State indices map correctly (uses `saved_state_indices[k]`) |
| 181 | Observable indices map correctly (uses `saved_observable_indices[m]`) |

---

## `outputhandling/save_summaries.py`

### `do_nothing` — Module-level CUDA function

| # | Functionality |
|---|--------------|
| 182 | No-op base case for empty metric chains |

### `chain_metrics`

| # | Functionality |
|---|--------------|
| 183 | Returns `do_nothing` when metric_functions is empty |
| 184 | Single metric: returns wrapper calling inner_chain then metric |
| 185 | Multiple metrics: recursively chains remaining metrics |
| 186 | Each wrapper slices buffer using offset and size |
| 187 | Each wrapper slices output using offset and size |
| 188 | Passes metric-specific params through to each function |

### `save_summary_factory`

| # | Functionality |
|---|--------------|
| 189 | Iterates over state variables when `summarise_states` True |
| 190 | Skips state variables when `summarise_states` False |
| 191 | Iterates over observable variables when `summarise_observables` True |
| 192 | Skips observable variables when `summarise_observables` False |
| 193 | Each variable gets buffer slice at `idx * total_buffer_size` |
| 194 | Each variable gets output slice at `idx * total_output_size` |
| 195 | Passes `summarise_every` through to chain |

---

## `outputhandling/update_summaries.py`

### `do_nothing` — Module-level CUDA function

| # | Functionality |
|---|--------------|
| 196 | No-op base case for empty metric chains |

### `chain_metrics`

| # | Functionality |
|---|--------------|
| 197 | Returns `do_nothing` when metric_functions is empty |
| 198 | Single metric: returns wrapper calling inner_chain then metric |
| 199 | Multiple metrics: recursively chains remaining metrics |
| 200 | Each wrapper slices buffer using offset and size |
| 201 | Passes metric-specific params through |

### `update_summary_factory`

| # | Functionality |
|---|--------------|
| 202 | Iterates over state variables when `summarise_states` True |
| 203 | Skips state variables when `summarise_states` False |
| 204 | Iterates over observable variables when `summarise_observables` True |
| 205 | Skips observable variables when `summarise_observables` False |
| 206 | Each variable gets buffer slice at `idx * total_buffer_size` |
| 207 | Passes current_state value for state vars via `summarised_state_indices[idx]` |
| 208 | Passes current_observables value for obs vars via `summarised_observable_indices[idx]` |

---

## `summarymetrics/metrics.py`

### `MetricFuncCache` — attrs class

| # | Functionality |
|---|--------------|
| 209 | Stores `update` and `save` callables with None defaults |

### `MetricConfig` — attrs class

| # | Functionality |
|---|--------------|
| 210 | `sample_summaries_every` defaults to 0.01, validated > 0 or None |
| 211 | Inherits precision from `CUDAFactoryConfig` |

### `register_metric` — Decorator factory

| # | Functionality |
|---|--------------|
| 212 | Instantiates the decorated class with `registry.precision` |
| 213 | Calls `registry.register_metric(instance)` |
| 214 | Returns the original class (not the instance) |

### `SummaryMetric.__init__`

| # | Functionality |
|---|--------------|
| 215 | Stores buffer_size, output_size, name, unit_modification |
| 216 | Creates MetricConfig with sample_summaries_every and precision |
| 217 | Sets up compile settings via `setup_compile_settings` |

### `SummaryMetric.update_device_func` — Forwarding Property

| # | Functionality |
|---|--------------|
| 218 | Returns `get_cached_output("update")` |

### `SummaryMetric.save_device_func` — Forwarding Property

| # | Functionality |
|---|--------------|
| 219 | Returns `get_cached_output("save")` |

### `SummaryMetric.update`

| # | Functionality |
|---|--------------|
| 220 | Delegates to `update_compile_settings(kwargs, silent=True)` |

### `SummaryMetric.build` — Abstract

| # | Functionality |
|---|--------------|
| 221 | Abstract method, must be overridden by subclasses |

### `SummaryMetrics.__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 222 | Resets `_params` to empty dict |
| 223 | Defines `_combined_metrics` mapping frozensets to combined names |

### `SummaryMetrics.update`

| # | Functionality |
|---|--------------|
| 224 | Updates `self.precision` if "precision" in kwargs |
| 225 | Propagates kwargs to all registered metric objects |

### `SummaryMetrics.register_metric`

| # | Functionality |
|---|--------------|
| 226 | Raises `ValueError` for duplicate metric name |
| 227 | Appends name to `_names` list |
| 228 | Stores buffer_size, output_size, metric object, and default param (0) |

### `SummaryMetrics._apply_combined_metrics`

| # | Functionality |
|---|--------------|
| 229 | Substitutes {mean, std, rms} with mean_std_rms when all three requested |
| 230 | Substitutes {mean, std} with mean_std when both requested |
| 231 | Substitutes {std, rms} with std_rms when both requested |
| 232 | Substitutes {max, min} with extrema when both requested |
| 233 | Substitutes {dxdt_max, dxdt_min} with dxdt_extrema when both requested |
| 234 | Substitutes {d2xdt2_max, d2xdt2_min} with d2xdt2_extrema when both requested |
| 235 | Prefers larger combinations (sorted by size descending) |
| 236 | Preserves original order of metrics |
| 237 | No substitution when only one of a pair is requested |
| 238 | Skips combined metric if not registered |

### `SummaryMetrics.preprocess_request`

| # | Functionality |
|---|--------------|
| 239 | Calls `parse_string_for_params` to extract parameters |
| 240 | Calls `_apply_combined_metrics` for substitution |
| 241 | Warns and removes unregistered metrics |
| 242 | Returns validated list of metric names |

### `SummaryMetrics.implemented_metrics` — Property

| # | Functionality |
|---|--------------|
| 243 | Returns `_names` list |

### `SummaryMetrics.summaries_buffer_height`

| # | Functionality |
|---|--------------|
| 244 | Sums buffer sizes for all preprocessed metrics |

### `SummaryMetrics.buffer_offsets`

| # | Functionality |
|---|--------------|
| 245 | Returns cumulative buffer offsets for each metric |

### `SummaryMetrics.buffer_sizes`

| # | Functionality |
|---|--------------|
| 246 | Returns tuple of buffer sizes for each metric |

### `SummaryMetrics.output_offsets`

| # | Functionality |
|---|--------------|
| 247 | Returns cumulative output offsets for each metric |

### `SummaryMetrics.summaries_output_height`

| # | Functionality |
|---|--------------|
| 248 | Sums output sizes for all preprocessed metrics |

### `SummaryMetrics._get_size`

| # | Functionality |
|---|--------------|
| 249 | Returns int directly when size is not callable |
| 250 | Calls size(param) when size is callable |
| 251 | Warns when callable size has param == 0 |

### `SummaryMetrics.legend`

| # | Functionality |
|---|--------------|
| 252 | Single-element metrics get name as heading |
| 253 | Multi-element metrics get `{name}_1`, `{name}_2`, etc. |

### `SummaryMetrics.unit_modifications`

| # | Functionality |
|---|--------------|
| 254 | Returns one unit modification per output element |
| 255 | Multi-element outputs repeat the same modification |

### `SummaryMetrics.output_sizes`

| # | Functionality |
|---|--------------|
| 256 | Returns tuple of output sizes for each metric |

### `SummaryMetrics.save_functions`

| # | Functionality |
|---|--------------|
| 257 | Returns tuple of save_device_func for each preprocessed metric |

### `SummaryMetrics.update_functions`

| # | Functionality |
|---|--------------|
| 258 | Returns tuple of update_device_func for each preprocessed metric |

### `SummaryMetrics.params`

| # | Functionality |
|---|--------------|
| 259 | Returns tuple of parameter values for each preprocessed metric |

### `SummaryMetrics.parse_string_for_params`

| # | Functionality |
|---|--------------|
| 260 | Extracts `[N]` parameter from metric string |
| 261 | Raises `ValueError` for non-integer parameter |
| 262 | Stores parsed param in `_params` dict |
| 263 | Metrics without `[N]` get default param 0 |
| 264 | Resets `_params` on each call |

---

## Individual Metric Files

All metrics follow the same pattern: `__init__` sets name/buffer_size/output_size/unit_modification, `build()` returns `MetricFuncCache(update, save)`.

### `mean.py` — Mean

| # | Functionality |
|---|--------------|
| 265 | buffer_size=1, output_size=1, unit_modification="[unit]" |
| 266 | update: accumulates sum in buffer[0] |
| 267 | save: divides by summarise_every, resets to 0.0 |

### `max.py` — Max

| # | Functionality |
|---|--------------|
| 268 | buffer_size=1, output_size=1, unit_modification="[unit]" |
| 269 | update: replaces buffer[0] when value > buffer[0] |
| 270 | update: no-op when value <= buffer[0] |
| 271 | save: copies buffer[0] to output, resets to -1e30 |

### `min.py` — Min

| # | Functionality |
|---|--------------|
| 272 | buffer_size=1, output_size=1, unit_modification="[unit]" |
| 273 | update: replaces buffer[0] when value < buffer[0] |
| 274 | update: no-op when value >= buffer[0] |
| 275 | save: copies buffer[0] to output, resets to 1e30 |

### `std.py` — Std

| # | Functionality |
|---|--------------|
| 276 | buffer_size=3, output_size=1, unit_modification="[unit]" |
| 277 | update: stores shift value on first sample (current_index==0), resets accumulators |
| 278 | update: accumulates shifted sum and shifted sum-of-squares |
| 279 | save: computes variance = mean_sq_shifted - mean_shifted^2, outputs sqrt(variance) |
| 280 | save: updates shift to mean for next period, resets accumulators |

### `rms.py` — RMS

| # | Functionality |
|---|--------------|
| 281 | buffer_size=1, output_size=1, unit_modification="[unit]" |
| 282 | update: resets sum_of_squares on first step (current_index==0) |
| 283 | update: adds value^2 to sum_of_squares |
| 284 | save: outputs sqrt(sum_of_squares / summarise_every), resets to 0.0 |

### `peaks.py` — Peaks

| # | Functionality |
|---|--------------|
| 285 | buffer_size=lambda n: 3+n, output_size=lambda n: n, unit_modification="s" |
| 286 | update: detects peaks when prev > value AND prev_prev < prev, with guards (index>=2, counter<npeaks, prev_prev!=0) |
| 287 | update: stores peak index at buffer[3+counter], increments counter |
| 288 | update: shifts prev/prev_prev history |
| 289 | save: copies n_peaks indices from buffer[3:] to output, resets |

### `negative_peaks.py` — NegativePeaks

| # | Functionality |
|---|--------------|
| 290 | buffer_size=lambda n: 3+n, output_size=lambda n: n, unit_modification="s" |
| 291 | update: detects negative peaks when prev < value AND prev_prev > prev, with guards |
| 292 | update: stores peak index, increments counter |
| 293 | save: copies indices and resets |

### `extrema.py` — Extrema

| # | Functionality |
|---|--------------|
| 294 | buffer_size=2, output_size=2, unit_modification="[unit]" |
| 295 | update: tracks max in buffer[0] and min in buffer[1] |
| 296 | save: outputs [max, min], resets to [-1e30, 1e30] |

### `max_magnitude.py` — MaxMagnitude

| # | Functionality |
|---|--------------|
| 297 | buffer_size=1, output_size=1, unit_modification="[unit]" |
| 298 | update: replaces buffer[0] when abs(value) > buffer[0] |
| 299 | save: copies buffer[0] to output, resets to 0.0 |

### `mean_std.py` — MeanStd

| # | Functionality |
|---|--------------|
| 300 | buffer_size=3, output_size=2, unit_modification="[unit]" |
| 301 | update: shifted data accumulation (same as std) |
| 302 | save: outputs [mean, std], resets with shift=mean |

### `mean_std_rms.py` — MeanStdRms

| # | Functionality |
|---|--------------|
| 303 | buffer_size=3, output_size=3, unit_modification="[unit]" |
| 304 | update: shifted data accumulation |
| 305 | save: outputs [mean, std, rms], resets with shift=mean |
| 306 | save: RMS uses E[X^2] = E[(X-shift)^2] + 2*shift*E[X-shift] + shift^2 |

### `std_rms.py` — StdRms

| # | Functionality |
|---|--------------|
| 307 | buffer_size=3, output_size=2, unit_modification="[unit]" |
| 308 | update: shifted data accumulation |
| 309 | save: outputs [std, rms], resets with shift=mean |

### `dxdt_max.py` — DxdtMax

| # | Functionality |
|---|--------------|
| 310 | buffer_size=2, output_size=1, unit_modification="[unit]*s^-1" |
| 311 | update: computes unscaled derivative (value - prev), updates max with predicated commit, guard on prev!=0 |
| 312 | save: scales by 1/sample_summaries_every, resets to -1e30 |

### `dxdt_min.py` — DxdtMin

| # | Functionality |
|---|--------------|
| 313 | buffer_size=2, output_size=1, unit_modification="[unit]*s^-1" |
| 314 | update: computes unscaled derivative, updates min with predicated commit, guard on prev!=0 |
| 315 | save: scales by 1/sample_summaries_every, resets to 1e30 |

### `dxdt_extrema.py` — DxdtExtrema

| # | Functionality |
|---|--------------|
| 316 | buffer_size=3, output_size=2, unit_modification="[unit]*s^-1" |
| 317 | update: computes unscaled derivative, updates max (buffer[1]) and min (buffer[2]) with predicated commit |
| 318 | save: scales both by 1/sample_summaries_every, outputs [max, min], resets sentinels |

### `d2xdt2_max.py` — D2xdt2Max

| # | Functionality |
|---|--------------|
| 319 | buffer_size=3, output_size=1, unit_modification="[unit]*s^-2" |
| 320 | update: computes central difference (value - 2*prev + prev_prev), updates max, guard on prev_prev!=0 |
| 321 | save: scales by 1/sample_summaries_every^2, resets to -1e30 |

### `d2xdt2_min.py` — D2xdt2Min

| # | Functionality |
|---|--------------|
| 322 | buffer_size=3, output_size=1, unit_modification="[unit]*s^-2" |
| 323 | update: computes central difference, updates min, guard on prev_prev!=0 |
| 324 | save: scales by 1/sample_summaries_every^2, resets to 1e30 |

### `d2xdt2_extrema.py` — D2xdt2Extrema

| # | Functionality |
|---|--------------|
| 325 | buffer_size=4, output_size=2, unit_modification="[unit]*s^-2" |
| 326 | update: computes central difference, updates max (buffer[2]) and min (buffer[3]), guard on prev_prev!=0 |
| 327 | save: scales both by 1/sample_summaries_every^2, outputs [max, min], resets sentinels |

---

**Total inventory items: 327**


### Memory Management

| # | Source File | Inventory |
|---|------------|-----------|
| 98 | `memory/mem_manager.py` | [x] |
| 99 | `memory/stream_groups.py` | [x] |
| 100 | `memory/cupy_emm.py` | [x] |
| 101 | `memory/chunk_buffer_pool.py` | [x] |
| 102 | `memory/array_requests.py` | [x] |
| 103 | `memory/__init__.py` | [x] |


### Batch Solving

| # | Source File | Inventory |
|---|------------|-----------|
| 104 | `batchsolving/_utils.py` | [x] |
| 105 | `batchsolving/SystemInterface.py` | [x] |
| 106 | `batchsolving/BatchInputHandler.py` | [x] |
| 107 | `batchsolving/BatchSolverConfig.py` | [x] |
| 108 | `batchsolving/BatchSolverKernel.py` | [x] |
| 109 | `batchsolving/solveresult.py` | [x] |
| 110 | `batchsolving/solver.py` | [x] |
| 111 | `batchsolving/writeback_watcher.py` | [x] |
| 112 | `batchsolving/arrays/BaseArrayManager.py` | [x] |
| 113 | `batchsolving/arrays/BatchInputArrays.py` | [x] |
| 114 | `batchsolving/arrays/BatchOutputArrays.py` | [x] |
| 115 | `batchsolving/arrays/__init__.py` | [x] |
| 116 | `batchsolving/__init__.py` | [x] |


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


### Init Files, IntegratorRunSettings & Vendored


## 1. `src/cubie/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Sets `NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS` env var to `"0"` at import time |
| 2 | Re-exports from `batchsolving`, `integrators`, `outputhandling`, `memory`, `odesystems`, `_utils` via star imports |
| 3 | Imports `TimeLogger` and `default_timelogger` from `cubie.time_logger` |
| 4 | `__all__` contains exactly: `summary_metrics`, `default_memmgr`, `ArrayTypes`, `Solver`, `solve_ivp`, `SymbolicODE`, `create_ODE_system`, `TimeLogger`, `default_timelogger`, `load_cellml_model` |
| 5 | `__version__` set from `importlib.metadata.version("cubie")` when installed |
| 6 | `__version__` falls back to `"unknown"` when `ImportError` raised (package not installed) |

---

## 2. `src/cubie/integrators/__init__.py`

### Module-Level Logic & Exports

| # | Functionality |
|---|--------------|
| 1 | Re-exports `SingleIntegratorRun` from `.SingleIntegratorRun` |
| 2 | Re-exports algorithm classes: `BackwardsEulerPCStep`, `BackwardsEulerStep`, `CrankNicolsonStep`, `ExplicitEulerStep`, `ExplicitStepConfig`, `ImplicitStepConfig`, `get_algorithm_step` |
| 3 | Re-exports `IVPLoop` from `.loops` |
| 4 | Re-exports solver classes: `LinearSolver`, `LinearSolverConfig`, `LinearSolverCache`, `NewtonKrylov`, `NewtonKrylovConfig`, `NewtonKrylovCache` |
| 5 | Re-exports controller classes: `AdaptiveIController`, `AdaptivePIController`, `AdaptivePIDController`, `FixedStepController`, `GustafssonController`, `get_controller` |

### `IntegratorReturnCodes` (IntEnum)

| # | Functionality |
|---|--------------|
| 6 | `SUCCESS = 0` |
| 7 | `NEWTON_BACKTRACKING_NO_SUITABLE_STEP = 1` |
| 8 | `MAX_NEWTON_ITERATIONS_EXCEEDED = 2` |
| 9 | `MAX_LINEAR_ITERATIONS_EXCEEDED = 4` |
| 10 | `STEP_TOO_SMALL = 8` |
| 11 | `DT_EFF_EFFECTIVELY_ZERO = 16` |
| 12 | `MAX_LOOP_ITERS_EXCEEDED = 32` |

---

## 3. `src/cubie/integrators/algorithms/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | `_ALGORITHM_REGISTRY` maps 8 string keys to step classes: `euler`, `backwards_euler`, `backwards_euler_pc`, `crank_nicolson`, `dirk`, `firk`, `erk`, `rosenbrock` |
| 2 | `_TABLEAU_REGISTRY_BY_ALGORITHM` initialised from `_ALGORITHM_REGISTRY` with `None` tableaus |
| 3 | ERK tableau aliases merged into `_TABLEAU_REGISTRY_BY_ALGORITHM` with `ERKStep` constructor |
| 4 | DIRK tableau aliases merged into `_TABLEAU_REGISTRY_BY_ALGORITHM` with `DIRKStep` constructor |
| 5 | FIRK tableau aliases merged into `_TABLEAU_REGISTRY_BY_ALGORITHM` with `FIRKStep` constructor |
| 6 | Rosenbrock tableau aliases merged into `_TABLEAU_REGISTRY_BY_ALGORITHM` with `GenericRosenbrockWStep` constructor |

### `resolve_alias(alias)`

| # | Functionality |
|---|--------------|
| 7 | Lowercases alias before lookup |
| 8 | Returns `(step_class, tableau)` tuple for known alias |
| 9 | Raises `KeyError` for unknown alias |

### `resolve_supplied_tableau(tableau)`

| # | Functionality |
|---|--------------|
| 10 | Returns `(ERKStep, tableau)` when `isinstance(tableau, ERKTableau)` |
| 11 | Returns `(DIRKStep, tableau)` when `isinstance(tableau, DIRKTableau)` |
| 12 | Returns `(FIRKStep, tableau)` when `isinstance(tableau, FIRKTableau)` |
| 13 | Returns `(GenericRosenbrockWStep, tableau)` when `isinstance(tableau, RosenbrockTableau)` |
| 14 | Raises `TypeError` for unrecognised tableau type |

### `get_algorithm_step(precision, settings, warn_on_unused, **kwargs)`

| # | Functionality |
|---|--------------|
| 15 | Merges `settings` dict and `kwargs` (kwargs override) |
| 16 | Pops `"algorithm"` key from merged settings |
| 17 | Raises `ValueError` when `algorithm` key is missing (`None`) |
| 18 | Branch: `algorithm` is `str` -> calls `resolve_alias` |
| 19 | `resolve_alias` `KeyError` wrapped as `ValueError("Unknown algorithm ...")` |
| 20 | Branch: `algorithm` is `ButcherTableau` -> calls `resolve_supplied_tableau` |
| 21 | Branch: `algorithm` is neither str nor tableau -> raises `TypeError` |
| 22 | Injects `precision` into settings |
| 23 | Injects `tableau` into settings when `resolved_tableau is not None` |
| 24 | Does NOT inject `tableau` when `resolved_tableau is None` |
| 25 | Instantiates and returns algorithm via `algorithm_type(**algorithm_settings)` |

---

## 4. `src/cubie/integrators/step_control/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | `_CONTROLLER_REGISTRY` maps 5 keys: `fixed`, `i`, `pi`, `pid`, `gustafsson` |
| 2 | Re-exports all 5 controller classes plus `get_controller` |

### `get_controller(precision, settings, warn_on_unused, **kwargs)`

| # | Functionality |
|---|--------------|
| 3 | Merges `settings` dict and `kwargs` (kwargs override) |
| 4 | Pops `"step_controller"` from merged settings |
| 5 | Raises `ValueError` when `step_controller` is `None` (missing) |
| 6 | Lowercases `step_controller` value |
| 7 | Looks up controller class in `_CONTROLLER_REGISTRY` |
| 8 | Raises `ValueError` wrapping `KeyError` for unknown controller type |
| 9 | Injects `precision` into settings |
| 10 | Instantiates and returns controller via `controller_type(**controller_settings)` |

---

## 5. `src/cubie/integrators/loops/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Re-exports `IVPLoop` from `.ode_loop` |
| 2 | `__all__` contains exactly `["IVPLoop"]` |

---

## 6. `src/cubie/integrators/matrix_free_solvers/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Re-exports `MatrixFreeSolverConfig` from `.base_solver` |
| 2 | Re-exports `LinearSolver`, `LinearSolverConfig`, `LinearSolverCache` from `.linear_solver` |
| 3 | Re-exports `NewtonKrylov`, `NewtonKrylovConfig`, `NewtonKrylovCache` from `.newton_krylov` |

### `SolverRetCodes` (IntEnum)

| # | Functionality |
|---|--------------|
| 4 | `SUCCESS = 0` |
| 5 | `NEWTON_BACKTRACKING_NO_SUITABLE_STEP = 1` |
| 6 | `MAX_NEWTON_ITERATIONS_EXCEEDED = 2` |
| 7 | `MAX_LINEAR_ITERATIONS_EXCEEDED = 4` |

---

## 7. `src/cubie/odesystems/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Re-exports `ODEData`, `SystemSizes` from `.ODEData` |
| 2 | Re-exports `SystemValues` from `.SystemValues` |
| 3 | Re-exports `BaseODE`, `ODECache` from `.baseODE` |
| 4 | Re-exports `SymbolicODE`, `create_ODE_system`, `load_cellml_model` from `.symbolic` |
| 5 | `__all__` contains 8 names |

---

## 8. `src/cubie/odesystems/symbolic/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Star-imports from `codegen`, `codegen.dxdt`, `indexedbasemaps`, `codegen.jacobian`, `odefile`, `parsing`, `symbolicODE`, `sym_utils`, `codegen.time_derivative` |
| 2 | `__all__` contains exactly `["SymbolicODE", "create_ODE_system", "load_cellml_model"]` |

---

## 9. `src/cubie/odesystems/symbolic/parsing/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Star-imports from `auxiliary_caching`, `cellml`, `jvp_equations`, `parser` |
| 2 | `__all__` contains exactly `["load_cellml_model"]` |

---

## 10. `src/cubie/odesystems/symbolic/codegen/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Star-imports from `linear_operators`, `nonlinear_residuals`, `preconditioners`, `numba_cuda_printer` |
| 2 | `__all__` is empty list |

---

## 11. `src/cubie/outputhandling/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Re-exports `OutputCompileFlags`, `OutputConfig` from `.output_config` |
| 2 | Re-exports `OutputFunctionCache`, `OutputFunctions` from `.output_functions` |
| 3 | Re-exports `BatchInputSizes`, `BatchOutputSizes`, `OutputArrayHeights`, `SingleRunOutputSizes` from `.output_sizes` |
| 4 | Re-exports `register_metric`, `summary_metrics` from `.summarymetrics` |
| 5 | `__all__` contains 10 names |

---

## 12. `src/cubie/outputhandling/summarymetrics/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Imports `SummaryMetrics` class and `register_metric` from `.metrics` |
| 2 | Creates module-level `summary_metrics` singleton as `SummaryMetrics(precision=float32)` |
| 3 | Default precision is `float32` (comment warns: only default dtype in project) |
| 4 | Imports 18 metric modules to trigger self-registration: `mean`, `max`, `rms`, `peaks`, `std`, `min`, `max_magnitude`, `extrema`, `negative_peaks`, `mean_std_rms`, `mean_std`, `std_rms`, `dxdt_max`, `dxdt_min`, `dxdt_extrema`, `d2xdt2_max`, `d2xdt2_min`, `d2xdt2_extrema` |
| 5 | `__all__` contains `["summary_metrics", "register_metric"]` |

---

## 13. `src/cubie/memory/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Re-exports `current_cupy_stream`, `CuPySyncNumbaManager`, `CuPyAsyncNumbaManager` from `.cupy_emm` |
| 2 | Re-exports `MemoryManager` from `.mem_manager` |
| 3 | Creates module-level singleton `default_memmgr = MemoryManager()` |
| 4 | `__all__` contains 5 names |

---

## 14. `src/cubie/batchsolving/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Defines `ArrayTypes` type alias: `Optional[Union[NDArray, DeviceNDArrayBase, MappedNDArray]]` |
| 2 | Re-exports `BatchInputHandler` from `.BatchInputHandler` |
| 3 | Re-exports `BatchSolverConfig`, `ActiveOutputs` from `.BatchSolverConfig` |
| 4 | Re-exports `BatchSolverKernel` from `.BatchSolverKernel` |
| 5 | Re-exports `SystemInterface` from `.SystemInterface` |
| 6 | Re-exports `ArrayContainer`, `BaseArrayManager`, `ManagedArray` from `.arrays.BaseArrayManager` |
| 7 | Re-exports `InputArrayContainer`, `InputArrays` from `.arrays.BatchInputArrays` |
| 8 | Re-exports `OutputArrayContainer`, `OutputArrays` from `.arrays.BatchOutputArrays` |
| 9 | Re-exports `Solver`, `solve_ivp` from `.solver` |
| 10 | Re-exports `SolveResult`, `SolveSpec` from `.solveresult` |
| 11 | Re-exports `summary_metrics` from `cubie.outputhandling` |
| 12 | `__all__` contains 17 names |

---

## 15. `src/cubie/batchsolving/arrays/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | File is empty (single blank line) — no exports, no logic |

---

## 16. `src/cubie/vendored/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Docstring only — no exports, no logic |

---

## 17. `src/cubie/integrators/IntegratorRunSettings.py`

### `IntegratorRunSettings` (attrs class, extends `CUDAFactoryConfig`)

| # | Functionality |
|---|--------------|
| 1 | `algorithm` attribute defaults to `"euler"`, validated as `str` |
| 2 | `step_controller` attribute defaults to `"fixed"`, validated as `str` |
| 3 | `__attrs_post_init__` calls `super().__attrs_post_init__()` (inherits precision handling from `CUDAFactoryConfig`) |
| 4 | Construction with explicit `algorithm` and `step_controller` |
| 5 | Construction with defaults only (no args beyond `precision`) |
| 6 | Validator rejects non-string `algorithm` |
| 7 | Validator rejects non-string `step_controller` |

---

## 18. `src/cubie/vendored/numba_cuda_cache.py`

### `_Cache` (abstract base class)

| # | Functionality |
|---|--------------|
| 1 | `cache_path` — abstract property |
| 2 | `load_overload(sig, target_context)` — abstract method |
| 3 | `save_overload(sig, data)` — abstract method |
| 4 | `enable()` — abstract method |
| 5 | `disable()` — abstract method |
| 6 | `flush()` — abstract method |

### `Cache(_Cache)` — concrete implementation

#### `__init__(self, py_func)`

| # | Functionality |
|---|--------------|
| 7 | Stores `repr(py_func)` as `_name` |
| 8 | Stores `py_func` as `_py_func` |
| 9 | Creates `_impl` from `_impl_class(py_func)` |
| 10 | Retrieves `_cache_path` from `_impl.locator.get_cache_path()` |
| 11 | Creates `IndexDataCacheFile` with `cache_path`, `filename_base`, `source_stamp` |
| 12 | Calls `self.enable()` at end of init |

#### `__repr__`

| # | Functionality |
|---|--------------|
| 13 | Returns `"<ClassName py_func=name>"` format string |

#### `cache_path` (property)

| # | Functionality |
|---|--------------|
| 14 | Returns `self._cache_path` — table-driven forwarding check against `_impl.locator.get_cache_path()` |

#### `enable` / `disable`

| # | Functionality |
|---|--------------|
| 15 | `enable()` sets `_enabled = True` |
| 16 | `disable()` sets `_enabled = False` |

#### `flush`

| # | Functionality |
|---|--------------|
| 17 | Delegates to `_cache_file.flush()` |

#### `load_overload(self, sig, target_context)`

| # | Functionality |
|---|--------------|
| 18 | Calls `target_context.refresh()` before loading |
| 19 | Wraps `_load_overload` call in `_guard_against_spurious_io_errors` context manager |

#### `_load_overload(self, sig, target_context)`

| # | Functionality |
|---|--------------|
| 20 | Returns `None` early when `_enabled` is `False` |
| 21 | Computes index key via `_index_key(sig, target_context.codegen())` |
| 22 | Loads data from `_cache_file.load(key)` |
| 23 | Branch: `data is not None` -> calls `_impl.rebuild(target_context, data)` |
| 24 | Branch: `data is None` -> returns `None` |

#### `save_overload(self, sig, data)`

| # | Functionality |
|---|--------------|
| 25 | Wraps `_save_overload` in `_guard_against_spurious_io_errors` context manager |

#### `_save_overload(self, sig, data)`

| # | Functionality |
|---|--------------|
| 26 | Returns early when `_enabled` is `False` |
| 27 | Returns early when `_impl.check_cachable(data)` is `False` |
| 28 | Calls `_impl.locator.ensure_cache_path()` |
| 29 | Computes index key via `_index_key(sig, data.codegen)` |
| 30 | Reduces data via `_impl.reduce(data)` |
| 31 | Saves via `_cache_file.save(key, data)` |

#### `_guard_against_spurious_io_errors`

| # | Functionality |
|---|--------------|
| 32 | On Windows (`os.name == "nt"`): catches `OSError` and re-raises only if `errno != EACCES` |
| 33 | On Windows: silently swallows `EACCES` errors |
| 34 | On non-Windows: yields without any error handling |

#### `_index_key(self, sig, codegen)`

| # | Functionality |
|---|--------------|
| 35 | Extracts `co_code` from `_py_func.__code__` |
| 36 | Branch: `__closure__ is not None` -> serialises closure cell contents via `dumps` |
| 37 | Branch: `__closure__ is None` -> uses empty bytes `b""` |
| 38 | Hashes code bytes and closure bytes with SHA-256 |
| 39 | Returns tuple of `(sig, codegen.magic_tuple(), (code_hash, closure_hash))` |

### `CUDACache(Cache)` — subclass

#### `load_overload(self, sig, target_context)`

| # | Functionality |
|---|--------------|
| 40 | Wraps parent `load_overload` in `utils.numba_target_override()` context manager |

---


### GUI (excluded -- Qt dependency)

| # | Source File | Inventory |
|---|------------|-----------|
| -- | `gui/__init__.py` | skip |
| -- | `gui/constants_editor.py` | skip |
| -- | `gui/states_editor.py` | skip |

---

## Removed Vestiges (during walkthrough)

- `SingleIntegratorRun.algorithm_key` -- identical to `algorithm`, removed.
  Consumer `BatchSolverKernel.algorithm` updated to use `.algorithm`.
- `SingleIntegratorRun.compiled_loop_function` -- alias for `device_function`, removed.
  Consumer `BatchSolverKernel` updated to use `.device_function`.
- `SingleIntegratorRun.threads_per_loop` -- alias for `threads_per_step`, removed.
  Consumer `BatchSolverKernel` updated to use `.threads_per_step`.

## Bugs Fixed (during walkthrough)

- `SingleIntegratorRunCore._switch_algos`: `return set("algorithm")` ->
  `return {"algorithm"}` (was returning set of characters).
- `SingleIntegratorRunCore._switch_controllers`: `return set("step_controller")` ->
  `return {"step_controller"}` (same bug).
