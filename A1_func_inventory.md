# A1 Functionality Inventory — Core + ODE + Symbolic

---

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
