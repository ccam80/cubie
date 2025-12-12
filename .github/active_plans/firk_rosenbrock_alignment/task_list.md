# Implementation Task List
# Feature: firk_rosenbrock_alignment
# Plan Reference: .github/active_plans/firk_rosenbrock_alignment/agent_plan.md

## Task Group 1: Source Context Review - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: tests/all_in_one.py (lines 1-1200, 1780-3400)
- File: src/cubie/integrators/algorithms/generic_firk.py (key buffer/local settings definitions and step wiring)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (buffer settings, cached Jacobian helpers, time-derivative handling)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- None (context review only).

**Tasks**:
1. **Map Module Parity Requirements**
   - File: tests/all_in_one.py
   - Action: Read
   - Details: Identify current FIRK and Rosenbrock sections (factories, buffer sizing, controller selection, loop wiring) to baseline gaps vs module counterparts.
   - Edge cases: Note handling of stage_count=1, n_drivers>0, and error_estimate absence.
   - Integration: Cross-compare with generic_firk.py and generic_rosenbrock_w.py expectations for buffers, helpers, and controller defaults.

2. **Extract Module Interface Contracts**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Read
   - Details: Capture BufferSettings/LocalSizes/shared_indices expectations, solver helper signatures (newton_krylov_solver_factory inputs), status code propagation, and controller defaults selection rules.
   - Edge cases: Shared vs local memory flags, flattened stage dimensions (stage_count * n), embedded error estimate presence.
   - Integration: Will inform inline factory replication in tests/all_in_one.py.

3. **Extract Rosenbrock Helper Contracts**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Read
   - Details: Capture cached Jacobian/time-derivative helper interfaces, BufferSettings (stage_rhs, stage_store, cached_auxiliaries), linear_solver_cached_factory expectations, and controller defaults for tableaus with/without error estimates.
   - Edge cases: cached_auxiliary_count sizing, gamma handling, driver derivatives, and stage store layout.
   - Integration: Will inform inline Rosenbrock factory parity in tests/all_in_one.py.

**Outcomes**:
- Consolidated notes on required parity between all_in_one inline factories and module implementations, including buffer sizing and helper signatures.

---

## Task Group 2: FIRK Parity Implementation Plan - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups [1]

**Required Context**:
- File: tests/all_in_one.py (FIRK factory and build wiring: lines ~1786-2100, 2890-2930, 3027-3400)
- File: src/cubie/integrators/algorithms/generic_firk.py (BufferSettings, LocalSizes, solver wiring)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (status code expectations, iteration counters)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (shared/local buffer sizing patterns)

**Input Validation Required**:
- tableau: ensure algorithm_type == 'firk' and selected tableau key exists in FIRK_TABLEAU_REGISTRY.
- memory flags: validate against {'local','shared'} prior to using in shared layout computations.
- n_drivers consistency: ensure driver stack sizing uses stage_count * n_drivers even when n_drivers==0.

**Tasks**:
1. **Flattened Stage Buffer Layout**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details:
     ```python
     all_stages_n = int32(stage_count) * int32(n_states)
     stage_increment_size = all_stages_n
     solver_scratch_size = int32(2) * all_stages_n
     ```
     Align solver_scratch, stage_increment, stage_state, and stage_driver_stack allocations with FIRKBufferSettings.shared_indices/local_sizes; stage_state size = n_states, driver_stack = stage_count * n_drivers.
   - Edge cases: stage_count==1, n_drivers==0; ensure sizes are non-negative and not multiplied by zero incorrectly.
   - Integration: Shared pointer offsets must mirror FIRKBufferSettings order (solver_scratch → stage_increment → stage_driver_stack → stage_state).

2. **Stage Driver Stack Handling**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details:
     ```python
     for stage_idx in range(stage_count):
         stage_time = current_time + dt_scalar * stage_time_fractions[stage_idx]
         driver_slice = stage_driver_stack[stage_idx * n_drivers:(stage_idx + 1) * n_drivers]
         driver_function(stage_time, driver_coeffs, driver_slice)
     ```
     Use pre-filled stage_driver_stack for driver-dependent RHS and nonlinear solver calls.
   - Edge cases: n_drivers==0 should skip loops without indexing errors.
   - Integration: Pass stage_driver_stack into nonlinear_solver and per-stage driver refresh for observables/dxdt.

3. **Coupled Nonlinear Solver Invocation**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Invoke newton_solver_fn with flattened stage_increment (stage_count * n) and pre-filled driver stack; capture status codes and propagate to counters/status outputs consistent with SolverRetCodes semantics.
   - Edge cases: ensure status_code accumulates bitwise OR across stages; handle zeroed accumulators for accumulates_output=False.
   - Integration: Align status propagation with loop counters (niters from upper bits if required).

4. **Stage Assembly and Accumulation**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Rebuild stage_state computation using tableau.a_flat coefficients; accumulate solution/error using solution_weights/error_weights with Kahan summation style; respect b_row/b_hat_row direct assignment when accumulates_output/error are False.
   - Edge cases: ends_at_one flag controls final observables/driver evaluation; b_row/b_hat_row None handling.
   - Integration: Match generic_firk accumulation ordering before controller error calculation.

5. **Controller Defaults & Flags**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: When algorithm_type=='firk', select controller defaults based on tableau.has_error_estimate (adaptive vs fixed) matching FIRK_ADAPTIVE_DEFAULTS/FIRK_FIXED_DEFAULTS; ensure dt0, fixed_mode, summarise flags honor FIRK path.
   - Edge cases: forced fixed controller when tableau lacks error estimate even if controller_type=='pid'.
   - Integration: Feed chosen controller into build section before loop wiring.

**Outcomes**:
- FIRK inline factory mirrors generic_firk buffer layout, solver wiring, and controller selection; stage driver stack and status codes aligned.

---

## Task Group 3: Rosenbrock Parity Implementation Plan - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups [1]

**Required Context**:
- File: tests/all_in_one.py (Rosenbrock factory/build wiring: lines ~2102-2515, 2920-2990, 3030-3095)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (BufferSettings, cached Jacobian helpers, time-derivative flow)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py and linear_solver_cached_factory (solver signatures, cached helpers)

**Input Validation Required**:
- tableau lookup: ensure algorithm_type == 'rosenbrock' and key in ROSENBROCK_TABLEAUS.
- cached_auxiliary_count sizing: ensure non-negative and matches helper output length.
- memory flags: enforce values in {'local','shared'} before slice calculations.

**Tasks**:
1. **Cached Jacobian Helper Integration**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Add prepare_jacobian helper mirroring generic_rosenbrock_w cached Jacobian prepare path, consuming (state, parameters, drivers_buffer, time, cached_auxiliaries). Preserve placeholder body but align signature; allocate cached_auxiliaries sized via helper return metadata (cached_auxiliary_count).
   - Edge cases: cached_auxiliary_count==0 should still allocate size 1 local buffer to avoid zero-sized arrays.
   - Integration: Pass cached_auxiliaries into linear_solver and time_derivative helpers.

2. **Time-Derivative RHS Factory**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Implement time_derivative factory matching module signature:
     ```python
     def time_derivative_rhs(state, parameters, drivers, observables, out, t):
         # placeholder sets zeros; structure matches solver_helpers.codegen
     ```
     Incorporate driver_del_t when n_drivers>0 using driver_derivative_inline_factory(interpolator).
   - Edge cases: n_drivers==0 bypasses driver_del_t call; ensure observables used to mirror module signature.
   - Integration: Feed time_derivative into rosenbrock_step_inline_factory and stage gamma corrections.

3. **Linear Solver with Cached Helpers**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Build linear_solver using linear_solver_inline_factory with cached_auxiliaries slice and cached operator; ensure signature matches generic_rosenbrock_w linear_solver_cached_factory (state, params, drivers, base_state, cached_auxiliaries, stage_time, dt, gamma, rhs, out, shared).
   - Edge cases: base_state placeholder must be correct slice (empty) yet type-safe; cached_auxiliaries length respected.
   - Integration: Update rosenbrock_step_inline_factory call sites to pass cached auxiliaries and cached operator helpers.

4. **Buffer Layout Parity**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Compute stage_rhs_size = n_states, stage_store_size = stage_count * n_states, cached_aux_size from helper; align shared_pointer offsets following RosenbrockBufferSettings order (stage_rhs → stage_store → cached_auxiliaries). Include sizes in shared_memory_bytes/local_memory_elements.
   - Edge cases: cached_aux_size zero; stage_count==1.
   - Integration: Update loop buffer sizing and dynamic shared memory calculation to include Rosenbrock buffers.

5. **Tableau Parameters in Stage Assembly**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Ensure gamma, gamma_stages, a_coeffs, C_coeffs, c vector flow into stage assembly exactly as generic_rosenbrock_w: stage corrections use C matrix and time-derivative scaling (inv_dt and gamma_i terms), final state/error accumulation respects accumulates_output/error flags and b_row/b_hat_row direct assignment.
   - Edge cases: no error estimate (fixed controller), stage_count==1.
   - Integration: Align status_code propagation from linear_solver with SolverRetCodes semantics.

**Outcomes**:
- Rosenbrock inline factory mirrors generic_rosenbrock_w structure with cached Jacobian/time-derivative placeholders only; buffer layouts and solver wiring match module expectations.

---

## Task Group 4: Loop, Controller, and Launch Parity - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups [2, 3]

**Required Context**:
- File: tests/all_in_one.py (loop buffer sizing and shared/local offsets: lines ~3023-3400; controller selection/dt0 setup; kernel launch shared memory calculation)
- File: src/cubie/integrators/loops/ode_loop_config.py (LoopSharedIndices/LocalSizes expectations)
- File: src/cubie/integrators/step_control/* (controller defaults for adaptive vs fixed)

**Input Validation Required**:
- shared_memory_bytes non-negative and below MAX_SHARED_MEMORY_PER_BLOCK.
- stride calculations: ensure shapes passed to get_strides are tuples of ints.

**Tasks**:
1. **Buffer Size Alignment**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Update accumulator_size and algorithm-specific scratch sizes to include FIRK/Rosenbrock shared/local buffers per BufferSettings.shared_memory_elements/local_memory_elements; ensure shared_pointer advances include new buffers for dynamic shared memory calculation.
   - Edge cases: shared flags off -> size 0; ensure no negative local_memory_elements.
   - Integration: Update local_memory_elements and shared_memory_bytes fed into kernel launch.

2. **Controller Selection Logic**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Mirror module default selection: if tableau.has_error_estimate is False, force controller_type='fixed' and skip adaptive PID config; otherwise permit PID defaults with tableau order. Ensure dt0 matches module (sqrt(dt_min*dt_max) for adaptive, dt for fixed).
   - Edge cases: user-specified controller_type incompatible with tableau; log/raise ValueError in parity with module behaviour.
   - Integration: Applies to FIRK and Rosenbrock paths; retain existing ERK/DIRK logic untouched.

3. **Driver/Derivative Wiring in Loop**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Ensure loop_persistent locals include FIRK driver stack slices and Rosenbrock cached auxiliaries where required; driver interpolation and derivatives passed into step_fn for all algorithm types.
   - Edge cases: n_drivers==0 should avoid stride/offset usage.
   - Integration: Align loop inputs with SingleIntegratorRun layout (state, parameters, drivers, observables, counters, summaries).

4. **Status/Counters Propagation**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details: Capture solver return codes from FIRK/Rosenbrock step_fn into loop counters and outputs consistent with IntegratorReturnCodes/SolverRetCodes; ensure upper bits (iteration counts) preserved for implicit solvers.
   - Edge cases: solver failure should propagate to controller accept flag and output counters.
   - Integration: Update saved outputs to include counter strides if save_counters_bool.

**Outcomes**:
- Loop/shared memory sizing and controller wiring remain consistent across algorithms; kernel launch uses correct shared bytes including FIRK/Rosenbrock buffers.

---

## Task Group 5: Validation & Lineinfo Integrity - PARALLEL
**Status**: [ ]
**Dependencies**: Groups [2, 3, 4]

**Required Context**:
- File: tests/all_in_one.py (entire file for sanity checks)
- Tests: None to run (lineinfo script), visual inspection only per instructions.

**Input Validation Required**:
- None (inspection only).

**Tasks**:
1. **Lineinfo/Signature Check**
   - File: tests/all_in_one.py
   - Action: Inspect
   - Details: Verify all inline factories keep device=True/inline=True, signatures match module counterparts, no unused placeholders beyond system helper codegen.
   - Edge cases: ensure cuda.local.array sizes use ints, no zero-sized arrays.
   - Integration: Confirm driver/interpolator placeholders compile in CUDASIM.

2. **Shared Memory Budget**
   - File: tests/all_in_one.py
   - Action: Inspect
   - Details: Confirm shared_memory_bytes <= MAX_SHARED_MEMORY_PER_BLOCK using updated sizes; adjust blocksize if needed (per instructions, only if FIRK/Rosenbrock additions push over budget).
   - Edge cases: handle use_shared_* flags toggling to all-shared.
   - Integration: Ensure dyn shared calculation includes new buffers exactly once.

**Outcomes**:
- Ready-to-run all_in_one.py with FIRK/Rosenbrock parity; no extraneous placeholders except system helper codegen.
