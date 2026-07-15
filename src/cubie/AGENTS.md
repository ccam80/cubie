# cubie

## Purpose
CuBIE (CUDA Batch Integration Engine) is a Numba-CUDA JIT batch ODE/SDE solver: it
compiles CUDA device functions on the fly to integrate large numbers of systems in
parallel on NVIDIA GPUs without the user writing CUDA. This package-root directory
holds the cross-cutting infrastructure the entire codebase depends on: the
`CUDAFactory` cached-compilation base class, the singleton `buffer_registry`, shared
validators/converters (`_utils.py`), the CUDA-simulator compatibility layer
(`cuda_simsafe.py`), file-based kernel caching (`cubie_cache.py`), and timing
(`time_logger.py`). `__init__.py` assembles the public API by star-importing the
subpackages.

## Public API
`__init__.py` re-exports from the subpackages and declares `__all__`:

| Symbol | Origin | Role |
|--------|--------|------|
| `Solver`, `solve_ivp` | `batchsolving` | User-facing batch solver class and convenience function. |
| `SymbolicODE`, `create_ODE_system`, `load_cellml_model` | `odesystems` | Build ODE systems from symbolic expressions or CellML. |
| `summary_metrics` | `outputhandling` | Singleton summary-metric registry. |
| `default_memmgr` | `memory` | Global `MemoryManager` singleton. |
| `ArrayTypes` | `batchsolving` | Array-type helper exported at package level. |
| `TimeLogger`, `default_timelogger` | `time_logger` | Timing/verbosity logger and its global singleton. |
| `CUBIE_RESULT_CODES` | `result_codes` | Bit-flag status codes for the per-run status word (device→solver). |

`__init__.py` also sets `NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS="0"` at import time and
resolves `__version__` via `importlib.metadata.version("cubie")`.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package entry point: star-imports subpackages, sets the Numba occupancy-warning env var, defines `__all__` and `__version__`. |
| `CUDAFactory.py` | Core cached-compilation framework: `CUDAFactory` (ABC; exposes `jit_kwargs`, the property every `build()` splats into `@cuda.jit`), `CUDAFactoryConfig`/`_CubieConfigBase` (hashing attrs config; carries the `jit_flags: JITFlags` compile setting every factory honours, with a read-only `lineinfo` passthrough), `CUDADispatcherCache`, the `MultipleInstance*` variants, and `hash_tuple`. |
| `_env.py` | `CUBIE_*` environment-variable registry: `env_bool`, `lineinfo_default` (`CUBIE_LINEINFO`), `cache_dir_default` (`CUBIE_CACHE_DIR`). Env values are defaults; explicit solver arguments always win. |
| `cache_root.py` | Single source of truth for the on-disk cache root (`get_cache_root`/`set_cache_root`/`get_cache_root_override`; precedence: `set_cache_root` override → `CUBIE_CACHE_DIR` → `<cwd>/generated`). The codegen, CellML parse, and compiled-kernel caches all resolve through it. |
| `buffer_registry.py` | Singleton `buffer_registry` (`BufferRegistry`) managing CUDA buffer metadata, layout, aliasing, and allocator generation; defines `CUDABuffer` and `BufferGroup`. |
| `_utils.py` | Shared helpers: `PrecisionDType`, precision/buffer validators + converters, attrs validator factories, `build_config`, `merge_kwargs_into_settings`, `ensure_nonzero_size`, `slice_variable_dimension`, `clamp_factory`. |
| `cuda_simsafe.py` | CUDASIM compatibility layer: `CUDA_SIMULATION`, `compile_kwargs` (immutable base defaults), `JITFlags` (managed `cuda.jit` options — `lineinfo`, `lto`, and the `nsz`, `contract`, `arcp`, `afn` fast-math flags — stored on every factory's compile settings so option changes rehash and rebuild), `get_jit_kwargs(jit_flags)` (renders a `JITFlags` — or a bare lineinfo bool from generated modules — to decorator kwargs; factory builds reach it through the `CUDAFactory.jit_kwargs` property, the single sanctioned route for every runtime `@cuda.jit` site), `from_dtype`, `is_devfunc`, `is_cuda_array`, the warp intrinsics (`selp`, `activemask`, `all_sync`, `any_sync`, `syncwarp`), the store write-through hint `stwt`, and memory-manager/array stand-ins. |
| `cubie_cache.py` | File-based persistence of compiled kernels: `CUBIECache*`, `CacheConfig`, `CubieCacheHandler`, `ALL_CACHE_PARAMETERS`. Depends on numba-cuda internals. |
| `time_logger.py` | `TimeLogger` (verbosity-gated timing), `CUDAEvent` (GPU event pair with CUDASIM fallback), `TimingEvent`, `default_timelogger`. |
| `result_codes.py` | `CUBIE_RESULT_CODES(IntFlag)` — the package-central status vocabulary OR-combined into the per-run status word — plus `decode_status_codes` for host-side decoding. |
| `array_interpolator.py` | `ArrayInterpolator(CUDAFactory)`: builds piecewise-polynomial (spline) coefficients from sampled driver arrays and compiles `evaluate_all` (Horner evaluation of all drivers at `t`) and `evaluate_time_derivative`. Owned by `Solver` as `driver_interpolator`; defines `ArrayInterpolatorConfig`, `InterpolatorCache`. |
| `writing_cuda_functions.md` | Working notes on CUDA device-function *optimisation* conventions (predicated commit, warp-coherent loops, …). Under discussion — consult before hand-optimising device code. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `batchsolving/` | High-level batch integration API: `Solver`, `solve_ivp`, `BatchSolverKernel`, grid building, system interface, result containers, host/device array managers (see `batchsolving/AGENTS.md`). |
| `integrators/` | Numerical integration components: `SingleIntegratorRun`, algorithm step factories, step controllers, matrix-free solvers, and CUDA loop builders (see `integrators/AGENTS.md`). |
| `memory/` | GPU memory subsystem: `MemoryManager` singleton (`default_memmgr`), array request/response containers, stream groups, CuPy-backed device/pinned allocation (see `memory/AGENTS.md`). |
| `odesystems/` | ODE system definitions and the SymPy-driven CUDA codegen pipeline (see `odesystems/AGENTS.md`). |
| `outputhandling/` | Output and summary-metric system (see `outputhandling/AGENTS.md`). |
| `gui/` | Optional Qt-based editors for `SymbolicODE` constants/parameters/states (see `gui/AGENTS.md`). |
| `vendored/` | Third-party code vendored as compatibility shims (see `vendored/AGENTS.md`). |

## For AI Agents

This directory is the **compilation spine**. The invariants below are uniform across
the codebase; subpackage `AGENTS.md` files describe only what they *add* and point
back here. CUDA-authoring **optimisation** conventions (predicated commit,
warp-coherent loops, …) live in `writing_cuda_functions.md`.

### CUDAFactory (cached compilation)
- **Subclasses override `build()`** to return a `CUDADispatcherCache` subclass
  instance (a bare callable raises `TypeError`). They **expose compiled device
  functions as named properties** (e.g. `device_function`, `evaluate_f`); callers use
  those properties. `get_cached_output(name)` is the internal plumbing the properties
  use, not the external interface. Never call `build()` directly — storing a
  device-function reference and then updating settings yields a stale reference
  (rebuild is lazy, on next property access).
- **`build()` compiles by closure capture.** A `build()` reads the current
  `compile_settings` (plus registry allocators and child device-function references)
  and bakes those values into the compiled device function as **closure constants** —
  fixed at compile time, not read at call time. This is why any settings change needs a
  rebuild (cache Layer B): the old values are frozen into the old closure. Capturing
  Python scalars/booleans as constants also lets Numba constant-fold and drop dead
  branches — see `writing_cuda_functions.md`.
- **Three cache layers — know which one you are touching:**
  1. **Compiled-kernel cache** (`cubie_cache`), keyed by `config_hash` = each
     factory's `values_hash` re-hashed together with its child factories'. An
     unchanged `config_hash` reuses the on-disk compiled kernel, so the dispatcher
     does not recompile. `BaseODE` folds constant *values* into its `config_hash`.
  2. **Object build cache** (`CUDAFactory._cache` + `_cache_valid`).
     `update_compile_settings` invalidates it **only if a setting actually changed**,
     re-running `build()` on the next property access.
  3. **Codegen source cache** (`odesystems/symbolic`: `ODEFile`),
     keyed by `fn_hash` — the system *definition* (equations + constant/observable
     labels), NOT constant values and NOT the full config. Caches generated CUDA
     *source*, separate from compilation.
- **`update` / `update_compile_settings` contract (uniform):** keys are the
  non-underscored field names; raises `KeyError` on an unrecognised key unless
  `silent=True`; returns a **`set`** of recognised/updated labels (the config-level
  `update` returns a `(recognised, changed)` tuple). A subclass `update` documents
  **only its additions** over this contract — do not restate the base behaviour.
  Change configuration **only** through `update`/`update_compile_settings` (or a config
  method that re-validates); assigning fields directly bypasses cache invalidation and any
  derived-field revalidation.
- **`config_hash` recurses into child `CUDAFactory` attributes**, so a composite
  factory invalidates when any child's config changes.
- **`-1` sentinel:** `get_cached_output` raises `NotImplementedError` when a cache
  field is the integer `-1` ("not implemented by this subclass").
- **`MultipleInstanceCUDAFactory`** maps prefixed external keys (e.g. `krylov_atol`)
  to unprefixed internal fields via `instance_label`; build configs with
  `build_config(...)`.

### Config classes (attrs convention)
- Compile settings are attrs classes subclassing `CUDAFactoryConfig` /
  `MultipleInstanceCUDAFactoryConfig`. **Variable- or float-typed members are stored
  underscore-prefixed and exposed, type-coerced, through a same-named property**;
  attrs `__init__` and `update` take the **non-underscored** names, so the entire
  external interface is non-underscored. Never pass underscored names; never alias
  underscored fields.
- A system runs at **one precision** (`ALLOWED_PRECISIONS` = float16/32/64); float
  members are returned cast to it via `self.precision(...)`.
- **`eq=False`** marks fields excluded from config equality/hashing (device-fn
  handles, callables; array fields use a custom `eq`). Plain `dict`-typed fields are
  rejected at construction — wrap compile-critical data in its own attrs class.

### buffer_registry (CUDA memory layout)
- **Code requirement:** a factory with managed buffers must **register and allocate
  them through the registry** — `register(name, parent, size, location,
  persistent=...)` in `register_buffers()`, then `get_allocator(name, self)` /
  `get_child_allocators(parent, child, name)` for device-side allocation; sizes via
  the `*_buffer_size` properties. Locations: `'local'` (thread registers / persistent
  local) vs `'shared'` (block shared memory). The shared/persistent carve-out and
  buffer **aliasing** are registry-internal. `register_child(parent, child, name)`
  registers a child's buffer footprint with its parent and records the ownership edge
  (`get_child_allocators` calls it before returning allocators), and `clear_parent`
  cascades through recorded children — this is how hot-swap paths drop a replaced
  component's whole chain, so registering children through `register_child` /
  `get_child_allocators` is what keeps swap cleanup working.
- **Docs requirement:** a child `AGENTS.md` just **lists the buffers it registers**;
  it does not re-describe the registry mechanics or aliasing.

### Array sizing (`ArraySizingClass`)
Host-side array shapes are computed by small attrs helpers subclassing `ArraySizingClass`
(defined in `outputhandling/output_sizes.py`, also used by `batchsolving`). Each exposes a
**`.nonzero`** property returning a copy with every int/tuple dimension floored to a minimum
of 1 — call it before allocating host or device buffers to avoid zero-length allocations.

### Device-code conventions
- **Use `cuda_simsafe` for the CUDASIM-sensitive helpers it provides** — the warp
  intrinsics (`selp`, `activemask`, `all_sync`, `any_sync`, `syncwarp`), the store
  write-through hint `stwt`, `from_dtype`, `is_devfunc`/`is_cuda_array`, and the
  memory-manager/array stand-ins — so device code runs under `NUMBA_ENABLE_CUDASIM=1`. Other `numba.cuda` features
  (`cuda.jit`, `cuda.grid`, `shared.array`, …) are used directly. **Never set
  `NUMBA_ENABLE_CUDASIM` in source.**
- **`# no cover` on device functions:** coverage cannot see inside compiled
  `@cuda.jit` code, so device-function bodies/closures are wrapped with
  `# no cover: start` / `# no cover: end` (and `# pragma: no cover` where
  appropriate). Keep these brackets when editing device code.
- **Import aliasing:** import NumPy scalar types with an `np_` prefix
  (`from numpy import float32 as np_float32`) to disambiguate them from the
  same-named numba types. Prefer explicit symbol imports over `import numpy as np`.
- **Optimisation strategies** (predicated commit, warp-coherent loop exits, …) are
  under discussion in `writing_cuda_functions.md` — consult it before hand-optimising
  device code.

### Testing
See the repo-root `AGENTS.md` for the canonical simulator vs real-GPU commands, markers, and the
full-suite approval gate. Never `xfail`, `importorskip`, or otherwise conditionally skip
behaviour; use the shared `tests/conftest.py` fixtures rather than mocking cubie objects.

### Root-file gotchas
- **`cubie_cache` depends on numba-cuda internals** (`_Kernel`, `IndexDataCacheFile`,
  `CUDACache`) and may break across numba-cuda versions; under CUDASIM it uses the
  vendored `CUDACache`.
- **Timing is no-op by default:** `default_timelogger` starts at `verbosity=None`;
  enable via `solve_ivp(time_logging_level=...)` / `Solver(time_logging_level=...)`.

## Dependencies
### Internal
This root infrastructure is depended on by every subpackage. Within the root, the
dependency order is roughly `cuda_simsafe` ← `_utils` ← `buffer_registry`,
`CUDAFactory`; `cubie_cache` depends on `CUDAFactory`, `_utils`, `cuda_simsafe`,
`time_logger`, `vendored.numba_cuda_cache`, and `cache_root`. All three disk
cache layers (codegen source, CellML parse, compiled kernels) resolve their
base directory through `cache_root.get_cache_root()`; `set_cache_root()`
relocates them together.

### External
- **numba / numba-cuda** — CUDA JIT, device intrinsics, cache internals.
- **numpy** (`>=2.0`) — dtypes, array hashing/comparison, validators.
- **attrs** — all config/data containers.
- **sympy** — used downstream in `odesystems/symbolic`.
- Optional: **cupy** (memory pool, via `memory/`), **qtpy + a Qt backend** (`gui/`).
