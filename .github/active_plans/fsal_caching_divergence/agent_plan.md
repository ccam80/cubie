# Agent Plan: FSAL Caching Warp Divergence Fix

## Problem Context

CuBIE's generic_erk and generic_dirk algorithms use FSAL (First-Same-As-Last) caching to avoid redundant RHS evaluations. The caching decision depends on `accepted_flag`, which varies per thread in adaptive stepping. This creates warp divergence when some threads in a warp accepted their previous step while others rejected it.

**Current Divergent Code Pattern (ERK example, line 205-228):**
```python
use_cached_rhs = ((not first_step_flag) and accepted_flag and first_same_as_last)
if multistage:
    if use_cached_rhs:
        for idx in range(n):
            stage_rhs[idx] = stage_cache[idx]
    else:
        dxdt_fn(...)
else:
    dxdt_fn(...)
```

**Issue:** `accepted_flag` differs between threads → warp serializes both branches → performance penalty

## Solution Architecture

### Core Change: Warp-Synchronized Cache Decision

Replace per-thread cache decision with warp-level vote. Cache only when ALL threads in the warp have accepted their previous step.

**New Pattern:**
```python
mask = activemask()
all_threads_accepted = all_sync(mask, accepted_flag != int16(0))
use_cached_rhs = all_threads_accepted and (not first_step_flag) and first_same_as_last

if multistage:
    if use_cached_rhs:
        # All threads execute this OR
    else:
        # All threads execute this
```

### Component Behavior

#### 1. Generic ERK Step (`src/cubie/integrators/algorithms/generic_erk.py`)

**Location:** `build_step()` method, inside the `step()` device function

**Current State (lines 205-228):**
- Per-thread cache decision causes divergence
- `use_cached_rhs` evaluated independently per thread
- Conditional branch on `use_cached_rhs` and `multistage`

**Required Changes:**
- Import `activemask` and `all_sync` from `cuda_simsafe` at module level
- Inside `step()` device function, before Stage 0 processing:
  - Call `mask = activemask()` to get active thread mask
  - Call `all_accepted = all_sync(mask, accepted_flag != int16(0))`
  - Modify `use_cached_rhs` condition to use `all_accepted` instead of `accepted_flag`
- No changes to caching logic, just the decision criterion

**Expected Behavior:**
- When all threads in warp accepted: cache is used (FSAL benefit retained)
- When any thread in warp rejected: all threads compute fresh RHS (no divergence)
- Single execution path taken by entire warp in both cases

#### 2. Generic DIRK Step (`src/cubie/integrators/algorithms/generic_dirk.py`)

**Location:** `build_step()` method, inside the `step()` device function

**Current State (lines 318-336):**
- Per-thread cache decision via `prev_state_accepted = accepted_flag != int16(0)`
- `use_cached_rhs` depends on `prev_state_accepted` (divergent)
- Cache implementation via comment "# RHS is aliased onto solver scratch cache at step-end"

**Required Changes:**
- Import `activemask` and `all_sync` from `cuda_simsafe` at module level
- Inside `step()` device function, before Stage 0 processing:
  - Call `mask = activemask()` to get active thread mask
  - Call `all_threads_accepted = all_sync(mask, accepted_flag != int16(0))`
  - Replace `prev_state_accepted` with `all_threads_accepted` in `use_cached_rhs` condition
- No changes to the cache aliasing mechanism

**Expected Behavior:**
- Identical to ERK: warp-synchronized caching decision
- Correctness maintained through aliasing design
- Zero divergence in Stage 0 processing

#### 3. Import Changes

**File:** `src/cubie/integrators/algorithms/generic_erk.py`
**Location:** Top of file (after existing imports)

**Add:**
```python
from cubie.cuda_simsafe import activemask, all_sync
```

**File:** `src/cubie/integrators/algorithms/generic_dirk.py`
**Location:** Top of file (after existing imports)

**Add:**
```python
from cubie.cuda_simsafe import activemask, all_sync
```

**Note:** These imports already exist in:
- `integrators/loops/ode_loop.py`
- `integrators/matrix_free_solvers/newton_krylov.py`
- `integrators/matrix_free_solvers/linear_solver.py`

Pattern is well-established in codebase.

### Integration Points

#### 1. Connection to IVP Loop

**File:** `src/cubie/integrators/loops/ode_loop.py`

**Current Flow (lines 399, 442-446):**
- Loop passes `accepted_flag` to step function
- `prev_step_accepted_flag` set via `selp(accept, int16(1), int16(0))`
- Value varies per thread based on controller decision

**No Changes Required:**
- Loop continues to pass per-thread `accepted_flag`
- Step functions now aggregate this at warp level
- API and data flow unchanged

#### 2. Interaction with Step Controllers

**Files:** `src/cubie/integrators/step_control/*.py`

**Current Behavior:**
- Controllers set `accept_step[0]` independently per thread
- Based on local error estimates and tolerances
- No synchronization between threads

**No Changes Required:**
- Controllers remain independent per thread
- Warp synchronization happens inside step functions
- Separation of concerns maintained

#### 3. Tableau Property: `first_same_as_last`

**File:** `src/cubie/integrators/algorithms/base_algorithm_step.py`

**Current Implementation (lines 151-156):**
```python
@property
def first_same_as_last(self) -> bool:
    return bool(self.c and self.c[0] == 0.0 and self.c[-1] == 1.0 
                and self.a[0][0] == 0.0)
```

**No Changes Required:**
- Property correctly identifies FSAL tableaus
- Compile-time constant (tableaus are frozen attrs classes)
- Used in cache decision without modification

### Data Structures

#### Shared Memory Layout (ERK)

**Current (lines 186-195):**
```python
stage_accumulator = shared[:accumulator_length]
if multistage:
    stage_cache = stage_accumulator[:n]  # Aliased onto first slice
```

**No Changes Required:**
- Cache storage unchanged
- Aliasing strategy remains valid
- Warp-sync decision doesn't affect memory layout

#### Shared Memory Layout (DIRK)

**Current (lines 298-301):**
```python
stage_accumulator = shared[acc_start:acc_end]
solver_scratch = shared[solver_start:solver_end]
stage_rhs = solver_scratch[:n]
increment_cache = solver_scratch[n:2*n]
```

**No Changes Required:**
- RHS aliased onto solver_scratch for caching
- Memory layout independent of sync strategy
- Cache commit at line 495: `increment_cache[idx] = stage_increment[idx]`

### Expected Interactions

#### 1. Warp Composition in Batch Runs

**Scenario:** Batch of 1024 systems on GPU with 32 threads per warp

- Systems grouped into warps by thread scheduler
- Each warp contains 32 independent systems
- Acceptance varies across systems (heterogeneous error growth)

**Interaction:**
- `all_sync()` evaluates within each warp independently
- Warp A might cache (all accepted) while Warp B doesn't (mixed)
- No cross-warp communication
- Correct behavior: cache hit rate varies by warp composition

#### 2. First Step Handling

**Current Logic (ERK line 206, DIRK line 322):**
```python
use_cached_rhs = ... and not first_step_flag ...
```

**Interaction:**
- `first_step_flag` uniform across all threads (same initial condition)
- First step always bypasses cache (no previous data)
- Warp-sync only affects subsequent steps
- Correctness: first step unaffected by change

#### 3. Fixed-Step Mode

**Scenario:** User specifies `is_adaptive=False`

- All threads always accept (fixed step size)
- `accepted_flag` uniformly True after first step
- `all_sync()` always returns True

**Interaction:**
- FSAL cache always used (100% hit rate)
- Warp-sync overhead amortized (single call per step)
- Performance: identical or better than current implementation

### Edge Cases

#### 1. Single-Stage Methods

**Example:** Explicit Euler (no multistage)

**Current Code (ERK lines 220-228):**
```python
else:  # not multistage
    dxdt_fn(...)
```

**Interaction:**
- No cache path (single stage can't reuse)
- Warp-sync logic short-circuits (multistage=False)
- No performance impact

#### 2. Non-FSAL Tableaus

**Example:** Classical RK4 (c = [0, 0.5, 0.5, 1] but a[0][0] ≠ 0 check fails)

**Interaction:**
- `first_same_as_last` property returns False
- `use_cached_rhs` always False regardless of sync
- No warp-sync call executed (short-circuit evaluation)
- Zero overhead for non-FSAL methods

#### 3. CUDASIM Mode

**Scenario:** `NUMBA_ENABLE_CUDASIM=1` for CPU testing

**Interaction:**
- `activemask()` and `all_sync()` defined in `cuda_simsafe` module
- CUDASIM fallbacks simulate single-threaded execution
- `all_sync()` trivially returns True (single thread)
- Tests continue to pass in CUDASIM

#### 4. Warp Size Variations

**Consideration:** Future GPUs with different warp sizes

**Interaction:**
- `activemask()` returns hardware-appropriate mask
- `all_sync()` operates on actual active threads
- Portable across architectures
- No hardcoded assumptions about warp size=32

### Dependencies

#### Required Imports

Both `generic_erk.py` and `generic_dirk.py` need:
```python
from cubie.cuda_simsafe import activemask, all_sync
```

**Module:** `src/cubie/cuda_simsafe.py`

**Available Functions:**
- `activemask()` - Returns mask of active threads in warp
- `all_sync(mask, predicate)` - Returns True if predicate is True for all threads in mask
- Already used in: `ode_loop.py`, `newton_krylov.py`, `linear_solver.py`

#### No New Dependencies

- All required primitives exist
- No external library additions
- No version constraints
- No platform-specific code

### Testing Implications

#### Functional Correctness

**Existing Tests to Validate:**
- `tests/integrators/algorithms/test_generic_erk_tableaus.py`
- `tests/integrators/algorithms/test_dirk_tableaus.py`
- Any tests using adaptive stepping with ERK/DIRK

**Expected:** All tests pass with identical numerical results

**Reason:** Cache decision more conservative (requires warp consensus) but correct when used

#### Performance Benchmarks (New)

**Test Scenarios:**
1. Uniform acceptance (all threads accept every step)
   - Measure: FSAL cache hit rate should be ~100%
   - Compare: Current vs. warp-sync performance

2. Uniform rejection (all threads reject every step)
   - Measure: FSAL cache hit rate should be ~0%
   - Compare: Overhead of warp-sync call

3. Mixed acceptance (50% accept, 50% reject)
   - Measure: Warp-level cache hit rate
   - Compare: Divergence elimination benefit

**Metrics:**
- Total kernel execution time
- RHS evaluation count
- Warp execution efficiency (via profiler)

#### Divergence Validation

**Tool:** `nvprof` or Nsight Compute

**Metric:** Warp execution efficiency / branch divergence

**Expected:**
- Current implementation: Low warp efficiency on mixed acceptance
- Warp-sync implementation: High warp efficiency in all scenarios

### Implementation Notes

#### Minimal Code Changes

**ERK (generic_erk.py):**
- Line ~7: Add import
- Lines ~202-206: Modify cache decision (3-4 lines)

**DIRK (generic_dirk.py):**
- Line ~7: Add import  
- Lines ~318-323: Modify cache decision (3-4 lines)

**Total:** ~10 lines changed across 2 files

#### No API Changes

- Function signatures unchanged
- Tableau definitions unchanged
- User-facing API unchanged
- Backward compatible

#### Compile-Time vs. Runtime

**Compile-Time Constants:**
- `first_same_as_last` (tableau property)
- `multistage` (derived from tableau)

**Runtime Variables:**
- `accepted_flag` (per-thread state)
- `first_step_flag` (loop state)
- `mask` (hardware state)

**Optimization:** JIT compiler can eliminate dead branches for non-FSAL tableaus

### Validation Strategy

1. **Unit Tests:** Verify warp-sync logic in isolation
2. **Integration Tests:** Run existing test suite
3. **Benchmarks:** Compare performance across scenarios
4. **Profiling:** Confirm divergence elimination
5. **Review:** Compare against issue #149 requirements

### Future Considerations

#### Optional: Configuration Parameter

If benchmarks show scenarios where FSAL caching hurts performance:

**Add to tableau:**
```python
@attrs.define(frozen=True)
class ERKTableau(ButcherTableau):
    enable_fsal_caching: bool = attrs.field(default=True)
```

**Modify decision:**
```python
use_cached_rhs = (all_threads_accepted and first_same_as_last 
                  and tableau.enable_fsal_caching and not first_step_flag)
```

**Deferred:** Only implement if data shows necessity

#### Rosenbrock Methods

**Current State:** Also use FSAL caching (CHANGELOG line 19)

**File:** `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

**Action:** Apply identical fix if benchmarks confirm ERK/DIRK benefit

**Deferred:** Address in follow-up PR after validating approach

### Success Criteria

1. **Correctness:** All existing tests pass
2. **Performance:** No regression in uniform acceptance scenarios
3. **Divergence:** Eliminated in mixed acceptance scenarios (profiler confirms)
4. **Code Quality:** Changes follow repository style (PEP8, numpydoc)
5. **Documentation:** Code comments explain warp-sync rationale
