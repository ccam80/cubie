<!-- Parent: ../AGENTS.md -->

# vendored

## Purpose
A pinned snapshot of numba-cuda's `Cache`/`CUDACache` that gives CuBIE a stable base to extend
with its own file-based caching (`CUBIECache` in `cubie_cache.py`) without tracking upstream
churn, and that compiles under CUDASIM. The upstream source path and snapshot date are recorded
in the module docstring.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package docstring only; no exports. |
| `numba_cuda_cache.py` | Snapshot from NVIDIA/numba-cuda (`numba_cuda/numba/cuda/core/caching.py`), 2026-01-11: `_Cache` (ABC), `Cache` (per-function compile cache over `IndexDataCacheFile`), `CUDACache` (overrides `load_overload` to run under `utils.numba_target_override()`). |

## For AI Agents
- No inline modifications — it reads as a direct snapshot. CuBIE's customization lives in the
  subclass `CUBIECache(CUDACache)` (`cubie_cache.py`), which sets `_impl_class = CUBIECacheImpl`
  and adds the file-based caching; the vendored `CUDACache` leaves `_impl_class = None` and is
  never instantiated directly. `cuda_simsafe` selects this vendored base vs the real
  `numba.cuda.dispatcher.CUDACache` depending on CUDASIM.
- Only `_Cache`/`Cache`/`CUDACache` are vendored; the file imports live from
  `numba.cuda.core.caching`, `numba.cuda.serialize`, and `numba.cuda.utils`.
- To update, replace it with a newer upstream snapshot and bump the date; extend behaviour in
  `CUBIECache` rather than editing the snapshot.
- No license header; upstream numba-cuda (NVIDIA) is BSD — confirm before redistribution.

## Dependencies
- None internal (consumed by `cubie_cache`/`cuda_simsafe`). Live upstream imports:
  `numba.cuda.core.caching` (`IndexDataCacheFile`), `numba.cuda.serialize` (`dumps`),
  `numba.cuda.utils` (`numba_target_override`).
