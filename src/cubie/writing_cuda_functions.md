# Writing CUDA device functions in CuBIE

> **Status: STUB / under discussion.** This file collects the *optimisation*
> conventions for hand-writing CUDA device functions in CuBIE. The hard invariants
> (cuda_simsafe usage, `# no cover`, the CUDAFactory/buffer-registry contracts) are
> settled and live in `AGENTS.md`. The strategies below are still being agreed —
> treat them as guidance, not law, and raise questions before relying on them.

## Scope
This covers *how to write the body of a `@cuda.jit(device=True)` function* for
performance and warp-correctness. It does NOT cover the factory/cache/buffer
machinery — see `AGENTS.md` for those invariants.

## Topics to establish

### Predicated commit (branchless writes)
Prefer `selp(pred, a, b)` (from `cuda_simsafe`) over `if/else` for per-lane value
selection, so divergent lanes stay in lockstep. _Open: when is a real branch
preferable (e.g. expensive guarded work)?_

### Compile-time branching
`build()`/`build_step` compute Python scalars and booleans from config and capture them
in the device-function closure. Because they are Python constants at JIT time, Numba
constant-folds them and **eliminates the dead branches entirely** — no runtime cost or
divergence. Prefer this to a runtime predicate whenever the decision is fixed for a given
compilation. Examples: the integration loop bakes `save_regularly`, `summarise_regularly`,
`fixed_mode`, and the `OutputCompileFlags` booleans; the algorithm step factories bake
`accumulates_output`, `has_error`, `multistage`, `first_same_as_last`.
_Open: when to prefer runtime `selp` predicated commit instead._

### Warp-coherent loop exit
Iterative device loops gate their exit on warp votes (`all_sync`/`any_sync` over
`activemask()`) so every active lane agrees before breaking, avoiding divergence.
_Open: state the rule precisely and the cases that require `any_sync` vs `all_sync`._

### (further topics)
- Streamed / Kahan-summed accumulation patterns.
- When buffer aliasing is worth it vs separate buffers.
- Shared- vs local-memory tradeoffs for scratch.
- FSAL / last-step reuse guarded by warp votes.

## References
- `AGENTS.md` (repo root `src/cubie/`) — settled device-code invariants.
- `.github/copilot-instructions.md` — project-wide style/CUDA conventions.
