# Components and Behaviors

- **Driver evaluation and derivative device functions**
  - Source: production `ArrayInterpolator` inline device code (Horner evaluation, wrap/clamp handling, padded clamped start, segment indexing).
  - Expectations: identical signatures and constants (order, num_segments, t0, dt, wrap, boundary_condition); derivative multiplies by `inv_resolution`; zero padding outside range.
  - Integration: used in all step factories when `has_driver_function`; shared with proposal/accepted driver buffers.

- **Observables device function**
  - Source: generated Lorenz observables from production codegen output for the test system.
  - Expectations: same signature as production (`state, parameters, drivers, observables, t`) and writes into `observables` buffer with correct precision and ordering.
  - Integration: invoked wherever production step factories call observables (pre-stage and end-of-step paths).

- **Loop seeding and buffer initialization**
  - Source: production `IVPLoop`/`SingleIntegratorRun` initialization logic.
  - Expectations: seed `proposed_state`, `proposed_drivers`, `proposed_observables`, and caches at loop entry; reuse accepted buffers when allowed; call driver function to populate proposals when not reusable; ensure observables refreshed before dxdt where production does.

- **Driver coefficient handling**
  - Ensure coefficients array layout and stride usage mirror production (segment-major, driver-major, order+1).
  - Confirm index math uses `evaluation_start` and `inv_resolution` consistent with production ArrayInterpolator.

- **Observable refresh points inside step factories**
  - Maintain parity with production ERK/DIRK/FIRK/Rosenbrock steps for where observables are recomputed (start, per-stage, end).
  - Guarantee proposed buffers are populated before nonlinear solver calls that depend on drivers.

- **Type and precision alignment**
  - All inline functions must use `numba_from_dtype`-derived precision types and typed constants matching production generators.
  - Driver/observable factories must not introduce simulation-only shortcuts; obey compile-time flags from global configuration.

# Architectural Changes

- Replace placeholder observables factory with production-generated logic for the Lorenz fixture; remove `pass`.
- Replace any stubbed driver code with verbatim production inline functions (evaluation + derivative).
- Update loop seeding to copy production behavior for driver/observable buffers and cached values before the first step and after accepted steps.
- Keep existing algorithm factories intact; only adjust call sites to use the production-grade driver/observable functions and seeding.

# Integration Points

- `tests/all_in_one.py`: houses inline factories and loop; changes confined here.
- Production references: `src/cubie/integrators/array_interpolator.py`, generated system observables (Lorenz), and loop seeding patterns from `src/cubie/integrators/loops/ode_loop.py` / `SingleIntegratorRun`.

# Data Structures

- Driver coefficients: 3D device array `(segments, drivers, order+1)`; output buffers sized per `n_drivers`.
- Observables buffer: 1D device array sized `n_observables`; proposal buffer mirrors size.
- Seeding flags: reuse/accept flags, driver availability flags, and shared/local buffer selection already present in global config—must be respected during seeding.

# Edge Cases to Consider

- No drivers configured (`n_drivers == 0`): driver functions should be `None` and seeding should bypass driver calls without altering buffers.
- Wrapping vs clamped boundary conditions: ensure out-of-range handling matches production Horner evaluation.
- Single-stage algorithms (e.g., Explicit Euler) and multistage algorithms (DIRK/FIRK): observables must still be refreshed at stage boundaries consistent with production.
- CUDASIM mode: keep device function compatibility (no host-only constructs) and rely on existing simsafe imports.
