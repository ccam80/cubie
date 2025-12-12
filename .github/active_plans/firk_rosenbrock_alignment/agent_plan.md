# Technical Plan

## Components to update

- **FIRK factories in `tests/all_in_one.py`**
  - Recreate coupled-stage solver setup to match `generic_firk`:
    - Stage-flattened dimensions (`stage_count * n`) for residuals, operators,
      preconditioners, Newton/Krylov solvers, and buffer sizing.
    - Buffer flags aligned with FIRK buffer settings (solver scratch, stage
      increment, stage driver stack, stage state) honoring shared/local choices.
    - Error-estimate handling and controller defaults derived from tableau
      metadata; ensure adaptive vs fixed controller selection mirrors module.
  - Integrate driver handling across stages (driver stack per stage) consistent
    with module expectations.
  - Align solver return codes and counter propagation with module semantics.

- **Rosenbrock factories in `tests/all_in_one.py`**
  - Mirror `generic_rosenbrock_w` cached-Jacobian path:
    - Use cached Jacobian helpers (prepare + apply) and cached auxiliaries sized
      per tableau/helper outputs.
    - Implement time-derivative factory that matches module behaviour, using
      driver derivatives when present.
    - Use linear solver with cached helpers (cached operator, cached
      auxiliaries) instead of simplified inline solver.
  - Buffer layout parity (stage_rhs, stage_store, cached_auxiliaries) with
    shared/local toggles matching Rosenbrock buffer settings.
  - Ensure tableau parameters (gamma, c vectors, A/C matrices, error weights)
    flow into stage assembly exactly as in module step.

- **System helper codegen hooks**
  - Keep placeholders only where production relies on generated solver helpers
    (Jacobian, time-derivative, optional linear operator helpers); ensure
    placeholder shapes, return semantics, and invocation sites match the module
    interfaces so real codegen can be dropped in without structural changes.

- **Loop and controller integration**
  - Verify loop wiring uses algorithm-specific buffer shapes and persistent
    locals consistent with module `BufferSettings`/`LocalSizes`.
  - Confirm controller selection (fixed vs PID) matches algorithm defaults and
    accepts overrides; error estimates and status codes feed controllers as in
    production.
  - Ensure driver interpolation and derivatives feed both FIRK and Rosenbrock
    paths; no missing parameters when switching algorithm types.

- **Kernel launch and stride setup**
  - Preserve stride calculation and output sizing; confirm new buffers (if any)
    are included in shared/local sizing and dynamic shared memory calculation.

## Expected behaviour

- Selecting any FIRK tableau builds an n-stage coupled solver with matching
  buffer layout, status handling, and controller defaults identical to
  `generic_firk`.
- Selecting any Rosenbrock tableau builds stage storage, cached auxiliaries,
  and linear solver helpers identical to `generic_rosenbrock_w`, including
  time-derivative usage.
- Switching algorithm_type between ERK, DIRK, FIRK, and Rosenbrock requires no
  ad-hoc edits; buffer sizing, driver handling, and loop/controller wiring are
  consistent across paths.

## Architectural changes

- Update FIRK and Rosenbrock sections of `tests/all_in_one.py`; no module code
  changes.
- Introduce helper factories mirroring module interfaces (cached Jacobian,
  time-derivative) within the debug script.
- Adjust shared/local buffer sizing and slices to include FIRK and Rosenbrock
  buffers per module settings.

## Integration points

- Tableau registries: `FIRK_TABLEAU_REGISTRY`, `ROSENBROCK_TABLEAUS`.
- Buffer sizing: align with FIRK and Rosenbrock BufferSettings equivalents.
- Solver helpers: hook into driver interpolation/derivative factories and loop
  persistent locals for counters/status codes.
- Output handling: ensure counters and summaries receive iterations/status as
  in production for FIRK/Rosenbrock paths.

## Data structures and dependencies

- Uses existing precision/numba dtype conversions, driver interpolation
  factories, and loop stride utilities in `tests/all_in_one.py`.
- Relies on tableau objects for stage counts, coefficients, error weights, and
  gamma values.
- Cached auxiliary buffers for Rosenbrock sized from helper outputs; FIRK
  buffers sized from stage_count, n, and n_drivers.

## Edge cases to cover

- `stage_count == 1` (both FIRK and Rosenbrock) should still respect buffer
  sizing and solver setup without relying on multistage assumptions.
- `n_drivers == 0` vs `n_drivers > 0` with driver derivatives for Rosenbrock and
  stage driver stacks for FIRK.
- Fixed vs adaptive controllers; absence of error estimates in tableaus must
  force fixed-step defaults.
- Shared-memory disabled (all local) vs enabled for individual buffers; dynamic
  shared memory sizing must remain nonnegative and within launch limits.
