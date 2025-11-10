# Agent Implementation Plan: Dense Output Interpolants

## Overview

This plan provides detailed technical specifications for implementing dense output interpolants in CuBIE's Runge-Kutta integrators. The implementation eliminates error estimate inflation by computing accurate errors from full steps while using interpolation to reach save points precisely.

## Architecture Components

### 1. Base Infrastructure (COMPLETED)

**File:** `src/cubie/integrators/algorithms/base_algorithm_step.py`

**Status:** ✅ Already implemented

**Description:** The `ButcherTableau` class now supports:
- `b_interp` field: Optional tuple of tuples containing dense output polynomial coefficients
- `has_interpolant` property: Returns True when `b_interp` is not None
- `interpolant_coefficients(numba_precision)` method: Returns typed coefficient tuples
- Validation in `__attrs_post_init__` ensuring all `b_interp` rows match stage count

**Structure:**
```python
b_interp = (
    (b_0^0, b_1^0, ..., b_s^0),  # Constant term coefficients
    (b_0^1, b_1^1, ..., b_s^1),  # θ^1 term coefficients
    (b_0^2, b_1^2, ..., b_s^2),  # θ^2 term coefficients
    ...
)
```

Where θ ∈ [0, 1] is the interpolation parameter: θ = (t_save - t) / dt

### 2. Loop Modifications

**File:** `src/cubie/integrators/loops/ode_loop.py`

**Current Behavior (Line ~406):**
```python
do_save = (t + dt[0] + equality_breaker) >= next_save
dt_eff = selp(do_save, next_save - t, dt[0])

step_status = step_function(
    ...,
    dt_eff,  # Truncated step passed to step function
    ...
)
```

**Required Changes:**

#### 2.1 Add Shared Memory Buffer for next_save

**Location:** Loop setup, before kernel definition

Add shared memory allocation for `next_save`:
```python
# In shared memory allocation section
shared_next_save_bytes = precision_size  # 4 or 8 bytes
total_shared_bytes += shared_next_save_bytes
```

**Location:** Inside device kernel, shared memory declaration

Allocate shared memory slot:
```python
shared_next_save = cuda.shared.array(shape=(1,), dtype=precision)
```

#### 2.2 Populate next_save Buffer

**Location:** Inside integration loop, before step function call

Write `next_save` to shared memory:
```python
# Only one thread writes (thread 0 in block)
if cuda.threadIdx.x == 0:
    shared_next_save[0] = next_save

# Synchronize to ensure all threads see the value
cuda.syncthreads()
```

#### 2.3 Pass Full Step Size to Step Function

**Location:** Step function call site

**Change:** Remove `dt_eff` computation, pass `dt[0]` directly:
```python
# OLD:
dt_eff = selp(do_save, next_save - t, dt[0])
step_status = step_function(..., dt_eff, ...)

# NEW:
step_status = step_function(..., dt[0], ...)  # Always full step
```

**Rationale:** Step function will handle interpolation internally when needed. Error estimate comes from full step, eliminating inflation.

#### 2.4 Compile-Time Flag Passing

**Location:** Build method, closure over tableau properties

Capture `has_interpolant` flag:
```python
has_interpolant = algorithm.tableau.has_interpolant if hasattr(algorithm, 'tableau') else False
```

This flag is available in the closure when compiling the step function, allowing compile-time branching.

### 3. Step Function Modifications

**Files:**
- `src/cubie/integrators/algorithms/generic_dirk.py`
- `src/cubie/integrators/algorithms/generic_firk.py`
- `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

**Pattern:** All three step types require similar modifications. Using DIRK as the canonical example:

#### 3.1 Capture Compile-Time Flags

**Location:** `build_step()` method, before device function definition

```python
has_interpolant = self.tableau.has_interpolant
b_interp_coeffs = self.tableau.interpolant_coefficients(numba_precision) if has_interpolant else None
stage_count = self.tableau.stage_count
```

These are captured in the closure and become compile-time constants in the device function.

#### 3.2 Read next_save from Shared Memory

**Location:** Inside device step function, after parameter unpacking

```python
# Read next_save from shared memory
# Shared memory is passed as parameter to step function (added to signature)
next_save_value = shared_next_save_buffer[0]
```

**Note:** `shared_next_save_buffer` must be added to step function signature. This is the only signature change required.

#### 3.3 Compute Interpolation Condition

**Location:** After taking the full RK step, before committing to `proposed_state`

```python
# Always computed (no branching) - predicated execution
needs_interp = do_save and has_interpolant and (next_save_value >= t) and (next_save_value <= t + dt)
theta = (next_save_value - t) / dt
```

**Important:** Both `needs_interp` and `theta` are computed on every step, regardless of whether interpolation is actually used. This avoids warp divergence.

#### 3.4 Evaluate Interpolant

**Location:** After computing `needs_interp` and `theta`

**Pattern:** Nested loops over state variables, stages, and polynomial powers

```python
# Temporary local buffer for interpolated state (thread-local)
y_interp = cuda.local.array(n, dtype=numba_precision)

# Compute interpolant for all state variables
for i in range(n):
    y_interp[i] = state_buffer[i]  # Start with y(t)
    
    # Accumulate stage contributions with polynomial weights
    for stage_idx in range(stage_count):
        # Evaluate polynomial: b_interp[0][stage] + b_interp[1][stage]*θ + ...
        weight = numba_precision(0.0)
        theta_power = numba_precision(1.0)
        
        for poly_idx in range(len(b_interp_coeffs)):
            weight += b_interp_coeffs[poly_idx][stage_idx] * theta_power
            theta_power *= theta
        
        # Add weighted stage contribution
        y_interp[i] += dt * weight * stage_k[stage_idx][i]
```

**Note:** `stage_k` is the existing buffer of stage derivatives, already computed during the RK step.

#### 3.5 Conditional Commit to proposed_state

**Location:** After interpolant evaluation

**Pattern:** Use `selp` to conditionally commit interpolated values

```python
# Conditional commit (no branching)
for i in range(n):
    proposed_state[i] = selp(
        needs_interp,
        y_interp[i],           # Use interpolated value if saving
        proposed_state[i]      # Otherwise use full-step result
    )
```

**Rationale:** `selp(condition, true_value, false_value)` is a single instruction with no branching, maintaining warp lockstep.

#### 3.6 Update Time for Observables

**Location:** Before calling observables function

```python
# Time for observables evaluation
t_obs = selp(needs_interp, next_save_value, t + dt)
```

#### 3.7 Evaluate Drivers at Correct Time

**Location:** Before calling observables function

If driver function exists and is not None:
```python
# Evaluate drivers at t_obs (either save point or step end)
driver_function(drivers_buffer, t_obs)
```

**Note:** Drivers are cheap to evaluate (typically algebraic), so re-evaluation is negligible overhead.

#### 3.8 Compute Observables at Correct Time

**Location:** After driver evaluation

```python
observables_function(
    proposed_state,           # Either interpolated or full-step state
    parameters_buffer,
    drivers_buffer,           # Evaluated at t_obs
    proposed_observables,     # Output buffer
    t_obs                     # Either next_save or t+dt
)
```

**Note:** Observables function is called once per step, using `selp`-ed inputs. No conditional branching.

#### 3.9 Return Full-Step Error Estimate

**Location:** Step function return

**Critical:** Error estimate MUST come from the full step, not the interpolated step:
```python
# Error is from full RK step evaluation
# DO NOT modify error based on interpolation
return step_status  # Status from full step
```

**Rationale:** This is the key fix - error estimate is commensurate with `dt`, not `dt_eff`.

### 4. Tableau Coefficient Updates

**Files:**
- `src/cubie/integrators/algorithms/generic_dirk_tableaus.py`
- `src/cubie/integrators/algorithms/generic_firk_tableaus.py`
- `src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py`

**Task:** Add `b_interp` field to tableaus with published dense output coefficients.

#### 4.1 DIRK Tableaus

**File:** `generic_dirk_tableaus.py`

**Tableau 1: TRAPEZOIDAL_DIRK_TABLEAU**
- **Status:** Placeholder linear interpolant already present
- **Action Required:** Replace with correct Hermite cubic interpolant
- **Source:** Hairer & Wanner (1996), Section IV.6 or derive from Hermite basis
- **Order:** 3rd-order interpolant (method is 2nd-order)

**Tableau 2: LOBATTO_IIIC_3_TABLEAU** (if exists)
- **Source:** Hairer & Wanner (1996), Table IV.6.X
- **Action:** Look up coefficients from book, add `b_interp` field
- **Order:** Match method order (4th order)

#### 4.2 FIRK Tableaus

**File:** `generic_firk_tableaus.py`

**Tableau 1: GAUSS_LEGENDRE_2_TABLEAU**
- **Source:** Hairer & Wanner (1996), Section IV.6, Gauss methods
- **Action:** Look up coefficients, add `b_interp` field
- **Order:** Match method order (4th order)

**Tableau 2: RADAU_IIA_5_TABLEAU** (if exists)
- **Source:** Hairer & Wanner (1996), Table IV.6.X
- **Action:** Look up coefficients, add `b_interp` field
- **Order:** Match method order (5th order)

#### 4.3 Rosenbrock Tableaus

**File:** `generic_rosenbrockw_tableaus.py`

**Tableau: ROS3P_TABLEAU**
- **Source:** Lang & Verwer (2001), "ROS3P Paper"
- **Action:** Check paper for dense output formula, convert to `b_interp` format
- **Fallback:** If not available in paper, skip (not required for MVP)
- **Order:** 3rd order

**Coefficient Lookup Strategy:**
1. Consult Hairer & Wanner (1996) primary reference
2. Cross-reference with SciML/DifferentialEquations.jl implementation
3. Validate against scipy.integrate if available
4. Only add if coefficients are published and verifiable

**Format:**
```python
TABLEAU_NAME = TableauType(
    a=...,
    b=...,
    c=...,
    order=...,
    b_interp=(
        (c0_stage0, c0_stage1, ...),  # Constant term
        (c1_stage0, c1_stage1, ...),  # θ^1 coefficient
        (c2_stage0, c2_stage1, ...),  # θ^2 coefficient
        ...
    ),
)
```

### 5. Edge Cases and Special Considerations

#### 5.1 Save Point at Step Boundary

**Condition:** `next_save == t + dt` (within floating-point tolerance)

**Behavior:** Interpolation logic computes θ=1.0, which should give the full-step result. Verify that:
```python
b_interp evaluated at θ=1 equals b
```

This is a requirement for valid interpolants and should be validated in tableau tests.

#### 5.2 Save Point Before Current Time

**Condition:** `next_save < t` (shouldn't happen with correct loop logic)

**Behavior:** `needs_interp` will be False, interpolation skipped. Full-step result used.

#### 5.3 No Interpolant Available

**Condition:** `has_interpolant = False` for the tableau

**Behavior:** Interpolation logic is skipped at compile time (if has_interpolant guards are compile-time constants). Fall back to current truncation behavior.

**Implementation Note:** This maintains backward compatibility - tableaus without `b_interp` behave exactly as before.

#### 5.4 Multiple Threads Accessing Shared Memory

**Condition:** All threads in block read `next_save` simultaneously

**Behavior:** Safe - shared memory reads are broadcast, all threads receive same value. Only one thread (thread 0) writes, synchronized with `cuda.syncthreads()`.

#### 5.5 Floating-Point Precision Considerations

**Issue:** `theta = (next_save - t) / dt` may have rounding errors

**Mitigation:**
- Use same precision as computation (already dtype=precision)
- Clamp θ to [0, 1] if necessary:
  ```python
  theta = max(0.0, min(1.0, (next_save_value - t) / dt))
  ```

#### 5.6 Warp Divergence in needs_interp

**Issue:** Different threads may have different `do_save` values

**Mitigation:** All threads compute `needs_interp`, `theta`, and `y_interp`. The `selp()` commit is predicated, so no branching occurs. All threads execute the same code path.

**Verification:** Use NVIDIA profiler to check warp efficiency remains >95%.

### 6. Integration with Existing Components

#### 6.1 Step Controllers

**Interface:** No changes required

**Rationale:** Step controllers receive error estimates from full steps. The `dt` they propose is used directly (not truncated), so their logic is unchanged.

**Verification:** Existing step controller tests should pass without modification.

#### 6.2 Output Functions

**Interface:** No changes required

**Rationale:** Output functions receive `proposed_state` and `proposed_observables` from step function, which now contain either full-step results or interpolated results (transparent to output logic).

**Verification:** Existing output tests should pass without modification.

#### 6.3 Memory Management

**Interface:** No changes required

**Rationale:** No new device buffers allocated. Shared memory increased by 1 scalar, which is negligible.

**Verification:** Memory tests should pass; check shared memory usage is within limits.

#### 6.4 Algorithm Selection

**Interface:** No changes required

**Rationale:** Algorithm selection is by name string. Interpolation is an internal implementation detail, activated by tableau `b_interp` field.

**Verification:** Algorithm selection tests should pass unchanged.

### 7. Data Structures and Dependencies

#### 7.1 Required Imports

**In step function files:**
```python
from cubie.cuda_simsafe import selp  # Already imported
from numba import cuda  # Already imported
```

No new imports required.

#### 7.2 Data Structure: b_interp Coefficients

**Type:** `Tuple[Tuple[float, ...], ...]`

**Shape:** `(polynomial_order + 1, stage_count)`
- First index: power of θ (0, 1, 2, ...)
- Second index: stage index (0, 1, ..., stage_count-1)

**Access Pattern:**
```python
coeff = b_interp_coeffs[poly_power][stage_idx]
```

**Typing:** Coefficients are typed to `numba_precision` by `interpolant_coefficients()` method.

#### 7.3 Local Memory Requirements

**Additional local buffers:**
- `y_interp[n]`: Temporary interpolated state (dtype=numba_precision)

**Size:** `n * sizeof(precision)` bytes per thread

**Note:** This is thread-local (registers or L1 cache), not shared memory.

**Impact:** For n=10, precision=float64: 80 bytes per thread. Negligible for modern GPUs.

#### 7.4 Shared Memory Requirements

**Additional shared memory:**
- `shared_next_save[1]`: Scalar save point time

**Size:** `sizeof(precision)` bytes per block (4 or 8 bytes)

**Impact:** Negligible - typical shared memory is 48KB per block.

### 8. Expected Behavior

#### 8.1 When Interpolation is Used

**Conditions:**
1. Tableau has `b_interp` coefficients (`has_interpolant=True`)
2. Current step spans a save point (`do_save=True` and `next_save ∈ [t, t+dt]`)

**Behavior:**
1. Full RK step taken with `dt` (not truncated)
2. Error estimate computed from full step
3. Interpolant evaluated at `θ = (next_save - t) / dt`
4. `proposed_state` receives interpolated values at `next_save`
5. Observables evaluated at `next_save` using interpolated state
6. Error estimate returned is from full step (accurate, not inflated)

**Result:** Step controller receives accurate error, can make informed decisions about step size.

#### 8.2 When Interpolation is Not Used

**Conditions:**
1. Tableau lacks `b_interp` (`has_interpolant=False`), OR
2. No save point in current step (`do_save=False` or `next_save` outside `[t, t+dt]`)

**Behavior:**
1. Full RK step taken
2. `proposed_state` receives full-step result (via `selp` with `needs_interp=False`)
3. Observables evaluated at `t + dt`
4. Error estimate from full step

**Result:** Identical to current behavior (backward compatible).

#### 8.3 Interaction with Adaptive Stepping

**Scenario:** Step is proposed, spans save point, step is accepted

**Sequence:**
1. Controller proposes `dt`
2. Step function takes full step with `dt`
3. Interpolation to `next_save` computed
4. Error estimate from full step returned
5. Controller evaluates error against tolerance
6. If accepted: `proposed_state` (interpolated) becomes new `state`
7. Next iteration starts from `next_save` with state at that point
8. Controller proposes new `dt` based on accurate error

**Key:** Error estimate drives step size adaptation, and it's now accurate (not inflated).

### 9. Testing Strategy (For Implementation Phase)

#### 9.1 Unit Tests for Interpolants

**Objectives:**
- Verify `b_interp` coefficients match literature
- Check boundary conditions (θ=0 gives y(t), θ=1 gives y(t+dt))
- Validate interpolation accuracy against exact solutions

**Test Cases:**
1. Linear ODE with known solution (e^(λt))
2. Polynomial ODE where interpolant should be exact
3. Boundary verification: θ=0, θ=1, θ=0.5
4. Compare against reference implementations (scipy, Julia)

#### 9.2 Integration Tests

**Objectives:**
- Verify error estimates are accurate (not inflated)
- Check step acceptance rate improves
- Validate observables at save points

**Test Cases:**
1. Stiff ODE with dense save points (van der Pol, Robertson)
2. Compare step rejection rate with/without interpolation
3. Verify state trajectory matches expected solution
4. Check observables match analytical values at save points

#### 9.3 Performance Tests

**Objectives:**
- Measure interpolation overhead
- Verify warp efficiency maintained
- Benchmark step acceptance rate improvement

**Metrics:**
1. Execution time: with interpolation vs without
2. GPU profiler: warp occupancy, branch divergence
3. Step statistics: acceptance rate, average step size

#### 9.4 Edge Case Tests

**Test Cases:**
1. Save point exactly at step boundary
2. Multiple save points in single step
3. Save point before current time (should not happen)
4. Very small θ (near 0) and very large θ (near 1)
5. Mixed tableaus (some with interpolants, some without)

### 10. Implementation Order and Dependencies

**Phase 1: Loop Modifications** (Enables passing `next_save` to steps)
- Add shared memory buffer for `next_save`
- Populate buffer in integration loop
- Pass full `dt[0]` to step function (remove `dt_eff` truncation)

**Phase 2: Step Function Updates** (Implements interpolation logic)
- Add interpolation logic to `generic_dirk.py`
- Add interpolation logic to `generic_firk.py`
- Add interpolation logic to `generic_rosenbrock_w.py`
- Each step type can be implemented independently

**Phase 3: Tableau Coefficients** (Provides interpolants for methods)
- Look up coefficients from Hairer & Wanner (1996)
- Add `b_interp` to DIRK tableaus
- Add `b_interp` to FIRK tableaus
- Add `b_interp` to Rosenbrock tableaus (if available)

**Phase 4: Testing** (Validates correctness and performance)
- Unit tests for interpolant accuracy
- Integration tests for error estimate accuracy
- Performance benchmarks
- Edge case validation

**Dependencies:**
- Phase 2 depends on Phase 1 (needs `shared_next_save` buffer)
- Phase 3 can proceed in parallel with Phases 1-2 (but interpolation won't activate until all phases complete)
- Phase 4 requires Phases 1-3 complete

### 11. Success Criteria

**Functional:**
- [ ] Interpolation logic executes without errors
- [ ] Error estimates are accurate (within 1% of true error)
- [ ] State and observables at save points match expected values
- [ ] Backward compatibility maintained (tableaus without `b_interp` work as before)

**Performance:**
- [ ] Interpolation overhead <10% when activated
- [ ] Warp efficiency >95% (verified via profiling)
- [ ] Step acceptance rate improves 15-25% on test problems
- [ ] Memory footprint unchanged from baseline

**Correctness:**
- [ ] All existing tests pass
- [ ] New unit tests validate interpolant accuracy
- [ ] Integration tests verify error estimate accuracy
- [ ] Edge cases handled correctly

**Documentation:**
- [ ] Literature references for all `b_interp` coefficients
- [ ] Docstrings updated in modified functions
- [ ] User-facing documentation mentions interpolation capability
