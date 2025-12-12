# Implementation Task List
# Feature: all_in_one_production_sync
# Plan Reference: .github/active_plans/all_in_one_production_sync/agent_plan.md

## Task Group 1: Driver device functions parity - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: tests/all_in_one.py (lines 422-589)
- File: src/cubie/integrators/array_interpolator.py (lines 343-446)

**Input Validation Required**:
- Ensure `interpolator.num_inputs > 0` before emitting driver/driver-derivative factories; otherwise return `None`.
- Assert `coefficients.shape[1] == n_drivers` inside host-side setup before passing to device code (no device-time checks).
- Validate `driver_coefficients.ndim == 3` and trailing dimension equals `order + 1` during host setup.

**Tasks**:
1. **Mirror ArrayInterpolator evaluate_all in driver_function_inline_factory**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details:
     ```python
     def driver_function(time: numba_prec, coefficients: numba_prec[:, :, ::1], out: numba_prec[::1]) -> None:
         scaled = (time - evaluation_start) * inv_resolution
         scaled_floor = floor(scaled)
         idx = int32(scaled_floor)
         if wrap:
             seg = int32(idx % num_segments)
             tau = prec(scaled - scaled_floor)
             in_range = True
         else:
             in_range = (scaled >= 0.0) and (scaled <= num_segments)
             seg = selp(idx < 0, int32(0), idx)
             seg = selp(seg >= num_segments, int32(num_segments - 1), seg)
             tau = scaled - float(seg)
         for driver_idx in range(num_drivers):
             acc = zero_value
             for k in range(order, -1, -1):
                 acc = acc * tau + coefficients[seg, driver_idx, k]
             out[driver_idx] = acc if in_range else zero_value
     ```
   - Edge cases: `wrap` vs clamped boundary, pad-clamped start segment, zero drivers (factory returns `None` so kernel skips call).
   - Integration: Used by all step factories when `has_driver_function` is True; must keep signature identical to production evaluate_all.

2. **Mirror ArrayInterpolator evaluate_time_derivative in driver_derivative_inline_factory**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details:
     ```python
     def driver_derivative(time: numba_prec, coefficients: numba_prec[:, :, ::1], out: numba_prec[::1]) -> None:
         scaled = (time - evaluation_start) * inv_resolution
         scaled_floor = floor(scaled)
         idx = int32(scaled_floor)
         if wrap:
             seg = int32(idx % num_segments)
             tau = prec(scaled - scaled_floor)
             in_range = True
         else:
             in_range = (scaled >= 0.0) and (scaled <= num_segments)
             seg = selp(idx < 0, int32(0), idx)
             seg = selp(seg >= num_segments, int32(num_segments - 1), seg)
             tau = scaled - float(seg)
         for driver_idx in range(num_drivers):
             acc = zero_value
             for k in range(order, 0, -1):
                 acc = acc * tau + prec(k) * coefficients[seg, driver_idx, k]
             out[driver_idx] = acc * inv_resolution if in_range else zero_value
     ```
   - Edge cases: Same as evaluate_all; derivative must scale by `inv_resolution`.
   - Integration: Supplies `driver_del_t` for Rosenbrock time-derivative and other driver-derivative call sites.

**Outcomes**:
- Files Modified:
  * tests/all_in_one.py (10 lines changed)
- Functions/Methods Added/Modified:
  * driver_function_inline_factory in tests/all_in_one.py
  * driver_derivative_inline_factory in tests/all_in_one.py
- Implementation Summary:
  Added host-side shape assertions for driver coefficients, guarded driver inline factories when no inputs, and kept inline implementations identical to production Horner evaluators and derivatives.
- Issues Flagged: None

---

## Task Group 2: Observables device logic - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [1]

**Required Context**:
- File: tests/all_in_one.py (lines 422-433)
- Generated Lorenz observables reference (Lorenz system observables from production codegen; same signature as `observables_factory`)

**Input Validation Required**:
- Assume `observables.size >= 3`; validate host-side that `n_observables` matches generated observable count (3) when enabling observables in debug config.

**Tasks**:
1. **Replace observables_factory stub with production Lorenz observables**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit((numba_prec[::1], numba_prec[::1], numba_prec[::1], numba_prec[::1], numba_prec), device=True, inline=True, **compile_kwargs)
     def get_observables(state, parameters, drivers, observables, t):
         observables[0] = state[0]
         observables[1] = state[1]
         observables[2] = state[2]
     ```
   - Edge cases: When `n_observables == 0`, factory returns `None` and loop/steps skip invocation; when `drivers` unused, keep signature but ignore buffer.
   - Integration: Called wherever step factories request observables (pre-stage and end-of-step).

**Outcomes**:
- Files Modified:
  * tests/all_in_one.py (8 lines changed)
- Functions/Methods Added/Modified:
  * observables_factory in tests/all_in_one.py
- Implementation Summary:
  Implemented Lorenz observables device helper and added host check enforcing three-output configuration when observables are enabled.
- Issues Flagged: None

---

## Task Group 3: Loop seeding parity with IVPLoop - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [1, 2]

**Required Context**:
- File: tests/all_in_one.py (lines 3148-3493)
- File: src/cubie/integrators/loops/ode_loop.py (lines 887-1189)

**Input Validation Required**:
- Host-side: ensure `driver_coefficients` is non-null when `n_drivers > 0`; otherwise set driver function to `None`.
- Validate `persistent_local` length covers slices `dt`, `accept_step`, `controller_temp`, and algorithm cache before launching kernel.
- Verify shared buffer spans computed offsets for driver/proposed driver/observable slices when shared flags are enabled.

**Tasks**:
1. **Seed state, drivers, observables, and proposals at loop entry**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details:
     - After buffer allocation, copy `initial_states` into both `state_buffer` and `state_proposal_buffer`.
     - When `has_driver_function` and `n_drivers > 0`, call `driver_function(t_prec, driver_coefficients, drivers_buffer)` and mirror results into `drivers_proposal_buffer`; else zero both driver buffers.
     - If `n_observables > 0`, compute observables with `observables_function(state_buffer, parameters_buffer, drivers_buffer, observables_buffer, t_prec)` and duplicate into `observables_proposal_buffer`.
   - Edge cases: Skip driver/observable calls when counts are zero; respect shared/local buffers by writing through slices already allocated.
   - Integration: Aligns initial buffers with production `IVPLoop` so first step sees populated proposals and base caches.

2. **Preserve counter seeding and save pipeline unchanged**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Ensure counter initialization still runs after new seeding, and that initial save paths reuse seeded proposals rather than uninitialized buffers.
   - Edge cases: `save_counters_bool` False path uses dummy `proposed_counters`; keep zeroing semantics.

**Outcomes**:
- Files Modified:
  * tests/all_in_one.py (30 lines changed)
- Functions/Methods Added/Modified:
  * loop_fn in tests/all_in_one.py
- Implementation Summary:
  Seeded proposal buffers for state, drivers, and observables at loop entry using driver and observable factories, zeroed driver paths when absent, mirrored values between accepted and proposal buffers, and added host-side checks for shared and persistent buffer capacities.
- Issues Flagged: None

---

## Task Group 4: Stage driver/observable refresh parity - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: tests/all_in_one.py (lines 940-2059 for ERK/DIRK/FIRK step factories; lines 2066-2590 for Rosenbrock step factory)
- File: src/cubie/integrators/loops/ode_loop.py (lines 979-1189 for refresh timing reference)

**Input Validation Required**:
- Guard driver refresh calls with `has_driver_function` to avoid invoking `None`.
- Ensure stage indices used for driver/observable slices stay within `stage_count`.

**Tasks**:
1. **Refresh drivers/observables at stage boundaries in DIRK/ERK**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details:
     - At stage 0, invoke `driver_function(stage_time, driver_coeffs, drivers_buffer or proposed_drivers)` before first `dxdt_fn` when drivers are enabled.
     - Recompute observables at stage 0 using the refreshed driver buffer prior to `dxdt_fn`.
     - Ensure per-stage loops continue to refresh proposed drivers via `driver_function` at each `stage_time` before calling `observables_function`/`dxdt_fn`, matching production Generic DIRK/ERK behaviour.
   - Edge cases: `multistage` False (explicit Euler) should still evaluate drivers/observables at start and end-of-step; skip when `n_drivers == 0`.

2. **Maintain FIRK stage driver stack parity**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Keep `stage_driver_stack` filled by `driver_function(stage_time, driver_coeffs, stage_slice)` for every stage before Newton solve, and reuse the cached stage drivers when copying into `proposed_drivers` and computing `observables_function`/`dxdt_fn`.
   - Edge cases: `stage_driver_stack_shared` flag controls buffer location; ensure writes honour shared/local slice.

3. **Rosenbrock driver refresh alignment**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Before each stage `dxdt_fn`/`stage_rhs` evaluation, refresh drivers with `driver_function(stage_time, driver_coeffs, proposed_drivers)` and use matching `driver_del_t` at `current_time` for `time_derivative` path; ensure end-of-step driver refresh uses `end_time` when drivers active.
   - Edge cases: When `driver_del_t` is `None`, zero out `proposed_drivers` to match production fallback.

**Outcomes**:
- Files Modified:
  * tests/all_in_one.py (60 lines changed)
- Functions/Methods Added/Modified:
  * erk_step_inline_factory in tests/all_in_one.py
  * firk_step_inline_factory in tests/all_in_one.py
  * rosenbrock_step_inline_factory in tests/all_in_one.py
- Implementation Summary:
  Refreshed drivers and observables at stage boundaries for ERK and Rosenbrock steps, guarded driver-derivative usage, zeroed driver proposals when derivatives unavailable, ensured FIRK stage driver stack initialization, and aligned end-of-step driver refresh with production timing.
- Issues Flagged: None
