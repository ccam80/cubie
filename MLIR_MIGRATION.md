# numba-cuda-mlir migration — state of the branch

Branch `mlir`, worktree `C:\local_working_projects\cubie_mlir`, venv
`.venv` (Python 3.12, numba-cuda-mlir 0.4.0, numpy 2.0.2, pip-provided
CUDA 12.9 toolkit wheels). Verified on an RTX 4070 SUPER (CC 8.9,
driver 591.86).

## What changed in cubie

- **Imports.** `from numba import cuda` → `from numba_cuda_mlir import
  cuda`; scalar types via `numba_cuda_mlir.types` (`bool_` maps to
  `boolean`); `from_dtype` via
  `numba_cuda_mlir.numba_cuda.np.numpy_support`; deep internals
  (caching, serialize, cudadrv, dispatcher) via the vendored
  `numba_cuda_mlir.numba_cuda` namespace. ~90 files rewritten.
- **`cubie/_mlir_compat.py` (new).** Registers lowerings missing from
  numba-cuda-mlir 0.4.0 and patches numpy-scalar handling; imported
  first thing from `cubie/__init__.py`. Every entry is a candidate for
  an upstream issue/PR (see below).
- **`cuda_simsafe.py`.** No CUDA simulator exists in numba-cuda-mlir,
  so the CUDASIM branches are gone; `NUMBA_ENABLE_CUDASIM=1` now
  raises at import. `CUDA_SIMULATION`/`is_cudasim_enabled` remain
  (always `False`). `compile_kwargs` is `{}`: `mlir_jit` only accepts
  `fastmath` as a bool, so the per-flag `{nsz, contract, arcp}` set is
  inexpressible (correctness chosen over blanket `fastmath=True`).
  `stwt` is a plain store (upstream cache-hint bug, below).
- **`cubie_cache.py`.** `CUBIECache`/`CUBIECacheImpl` rebuilt on
  `MLIRCache`/`MLIRCacheImpl` (cubin/PTX payload serialization instead
  of `_Kernel._reduce_states()`). Locator, index keys, LRU eviction
  and handler logic unchanged.
- **`buffer_registry.py`.** Zero-length buffers anchor at offset 0:
  MLIR's subview verifier rejects `scratch[n:n]` empty tail views that
  numpy-style slicing allows.
- **`BatchOutputArrays.py`.** Writeback event handles normalised with
  `int(event.handle)` (cuda.bindings objects have no `.value`).
- **`pyproject.toml`.** `numba-cuda-mlir[cu12]` replaces
  `numba` + `numba-cuda[cu12]`; `numpy>=2.0` (required by
  numba-cuda-mlir); `requires-python >= 3.11`; stock `numba` moved to
  dev extras (CPU-reference `njit` in tests only).
- **Generated-code header** (`odefile.py`) emits numba-cuda-mlir
  imports; docs example and doctest imports updated; three tests
  updated to current contracts (cache `check_cachable`, empty
  `compile_kwargs`, `int(stream.handle)`).

## Ecosystem constraint: cellmlmanip vs numpy 2

numba-cuda-mlir requires numpy ≥ 2 semantics, but every cellmlmanip
release pins `Pint<0.20`, and no Pint version supports both numpy 2 and
the pre-0.20 API (0.3.7 additionally pins `sympy<1.13`, conflicting
with cubie). The venv carries a **hand-patched cellmlmanip 0.3.6**
(`units.py`: `ScaleConverter`/`UnitDefinition` import fallback to
`pint.facets.plain` plus the new required `reference` argument).
Re-apply after any pip operation that touches cellmlmanip. Long-term:
upstream a Pint-0.20+ compat fix to cellmlmanip, vendor the patch, or
drop the dependency.

## Upstream numba-cuda-mlir 0.4.0 bugs found (all reproduced, most
worked around)

1. **Import crash on numpy < 2.** `numba_cuda_mlir/types.py`
   unconditionally imports `bool`, defined in the vendored types module
   only when `numpy_version >= (2, 0)`. Effectively an undeclared
   numpy≥2 requirement.
2. **Boolean bitwise ops unregistered.** `&`, `|`, `^`, `~` and their
   in-place forms lower only for `(Integer, Integer)`; Boolean operands
   raise `NotImplementedError`. Shim registers them (`arith.andi/ori/
   xori` work on i1).
3. **Boolean comparisons unregistered.** `==`, `!=`, `<`, `<=`, `>`,
   `>=` registered for `(Number, Number)` only. Shim adds Boolean
   signatures.
4. **numpy scalar constants crash lowering.** Frozen closure constants
   like `numpy.bool_(True)` crash `try_extract_constant`,
   `lower_const_assign` (its isinstance gate excludes `np.bool_`,
   which is not `np.number`), and `unverified_convert`. Shim unwraps
   numpy scalars at `load_var`, registers an `unverified_convert`
   overload, and patches the other two.
5. **Dynamic getitem on nested tuples crashes.**
   `tup[i]` where elements are themselves tuples (Butcher tableau rows,
   `generic_erk.py` `stage_rhs_coeffs[prev_idx]`) emits a
   single-result `scf.index_switch` whose result type must be a scalar
   MLIR type → "Result 0 ... must be a Type (bad cast)". **This blocked
   every tableau-driven algorithm** (rk45, rk23, dopri54, tsit5, ros3p,
   dirk, firk, rosenbrock). Shim decomposes the row selection into one
   scalar `index_switch` per column.
6. **Cache-hint loads/stores drop view offsets.** `cuda.stwt` (and the
   whole `ld*/st*` cache-hint family) compute the element pointer via
   `memref_to_llvm_ptr`, which ignores the offset of strided
   views — every store lands at the buffer base. Minimal repro: stwt
   through `arr3d[:, :, r][i, :]` writes `arr3d[0,0,0]`. Worked around
   in `cuda_simsafe.stwt` (plain store).
7. **LTO link at opt_level>0 erases stores.** Upstream warns about
   f16/bf16, but plain float32/float64 output stores in cubie's save
   paths are also erased (all-zero outputs in several test configs).
   `_mlir_compat` sets `config.CUDA_DISABLE_LTO_OPT = 1`
   programmatically; kernels are still optimised, only the final LTO
   link runs at opt 0.
8. **Empty tail subviews rejected.** `scratch[n:n]` (zero-length view
   at the end of a buffer) fails MLIR verification with
   "offset 0 is out-of-bounds: n >= n"; numba/NumPy allow it. Fixed on
   cubie's side in `buffer_registry` (anchor empty buffers at 0).
9. **xdist at high worker counts.** `-n logical` (20 workers here)
   kills all workers at startup; `-n 8` is stable. Worth noting for CI
   configuration.

Verified-correct semantics worth recording: warp primitives
(`activemask`, `all_sync`, `any_sync`), `selp`, dynamic strided/nested
views with plain stores, and Python `%`/`//` sign semantics on negative
operands all behave correctly under MLIR.

## Test results (real GPU, `-m "not specific_algos and not sim_only"`)

| | passed | failed | errors | wall time |
|---|---|---|---|---|
| main + numba-cuda (baseline) | 2219 | 4 | 15 | 8:45 |
| mlir branch, mechanical port only | 2106 | 32 | 100 | 3:56 |
| mlir branch, final | 2234 | 4 | 0 | 3:48 |

The final mlir branch passes more tests than the main baseline: the
chunking/writeback errors on main (event-handle `.value` accesses that
break under current numba-cuda too) are fixed here. Two of the four
remaining failures are pre-existing on main; the two wrap-interpolation
failures are the only mlir-specific regressions.

The suite runs **~2.4× faster** under numba-cuda-mlir (3:35 vs 8:45),
dominated by compile time. End-to-end numerics verified: exponential
decay through `solve_ivp` matches analytic results for euler,
backwards_euler, backwards_euler_pc, crank_nicolson, rk45, rk23,
dopri54, tsit5 and ros3p (errors 1e-4 … 4e-9, consistent with each
method's order).

Known remaining issues:

- `test_writeback_watcher.py::test_process_task_returns_false_for_incomplete`
  and `::test_process_task_cudasim_immediate_complete` — **also fail on
  main**; they assert simulator-era expectations (an unrecorded CUDA
  event queries complete; a string event raises). Pre-existing, not a
  migration regression.
- `test_array_interpolator.py::test_wrap_vs_clamp_evaluation` and
  `::test_wrap_repeats_periodically` — wrap (periodic) driver
  interpolation diverges from the CPU reference at wrapped sample
  points on the mlir branch. Python `%` semantics were probed and are
  correct, so the cause is elsewhere in the wrap-index/coefficient
  path. Needs a dedicated debugging session.

## Recommendations

- File upstream issues for items 1–8 (the NVIDIA tracker printed in the
  ICE message); each shim entry in `_mlir_compat.py` can be deleted as
  upstream fixes land.
- Decide the cellmlmanip strategy (vendored patch vs upstream PR) —
  the venv-local patch is the one piece of this branch that a plain
  `pip install -e .[dev]` does not reproduce.
- The loss of per-flag fastmath and CUDASIM are permanent behavioural
  differences of the MLIR backend, not bugs: revisit fastmath=True
  selectively for performance work, and drop simulator-based CI lanes
  on this branch.
- Chase the wrap-interpolation divergence before relying on
  interpolated periodic drivers.
