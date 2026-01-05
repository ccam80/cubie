# Loop Timing Refactor: Agent Plan

## Overview

This plan addresses the completion of the loop-timing refactor started in PR #279, resolving all review comments and completing the migration from `dt_save`/`dt_summarise` to the new naming scheme with independent `sample_summaries_every` functionality.

---

## Component 1: Replace selp with min for dt_eff Calculation

**File:** `src/cubie/integrators/loops/ode_loop.py`  
**Location:** Lines 614-622 (approximately)

**Current Behavior:**
```python
next_event = selp(
    next_save < next_update_summary,
    next_save,
    next_update_summary
)
```

**Expected Behavior:**
Use the built-in `min()` function which has a fast path in CUDA hardware. The result should compute the minimum of `next_save` and `next_update_summary` directly.

**Integration Points:**
- The `min()` function is available as a Python built-in and works in Numba CUDA
- No import changes required
- No change to the overall logic, just the implementation

---

## Component 2: Remove Timing Aliases in ode_loop.py

**File:** `src/cubie/integrators/loops/ode_loop.py`  
**Location:** Lines 367-369 (approximately)

**Current Pattern:**
```python
updates_per_summary = config.samples_per_summary
dt_save = precision(config.save_every)
dt_update_summaries = precision(config.sample_summaries_every)
```

**Expected Behavior:**
- Remove `dt_save` and `dt_update_summaries` aliases
- Access `config.save_every` and `config.sample_summaries_every` directly in device code
- Keep `updates_per_summary` as it's computed from config, not an alias

**Affected Locations:**
All references to `dt_save` and `dt_update_summaries` in the device function must be replaced:
- Next save calculations
- Next update summary calculations
- Anywhere these timing values are used

**Edge Cases:**
- Ensure precision casting is applied when needed (config properties already return precision-cast values)

---

## Component 3: Remove Sentinel Values in ODELoopConfig

**File:** `src/cubie/integrators/loops/ode_loop_config.py`  
**Location:** `__attrs_post_init__` method, lines 261-273

**Current Behavior:**
When all timing parameters are None, sets sentinel values:
```python
self._save_every = 0.1
self._summarise_every = 1.0
self._sample_summaries_every = 0.1
```

**Expected Behavior:**
- Remove sentinel value assignment
- Set `save_last=True` and `summarise_last=True` flags only
- Defer actual timing value requirements to compile-time validation
- Handle the None-to-compile flow by checking `compile_flags` at build time

**Architectural Change:**
The validation now splits into two phases:
1. **Configuration phase:** Accept None values, set behavioral flags
2. **Compile phase:** Validate that required timing values are present based on output types

**Dependencies:**
- OutputCompileFlags must be accessible during build
- IVPLoop.build() must validate timing parameters if relevant outputs enabled

---

## Component 4: Tolerant Integer Multiple Validation

**File:** `src/cubie/integrators/loops/ode_loop_config.py`  
**Location:** Lines 318-327

**Current Behavior:**
Strict tolerance check, raises error if not within tolerance.

**Expected Behavior:**
1. Calculate ratio: `ratio = summarise_every / sample_summaries_every`
2. Calculate deviation: `deviation = abs(ratio - round(ratio))`
3. If deviation <= 0.01 (1%):
   - Compute adjusted `summarise_every = round(ratio) * sample_summaries_every`
   - If adjusted != original: warn with new value
   - If adjusted == original (floating point rounding): warn with actual behavior
4. If deviation > 0.01:
   - Raise ValueError with clear message explaining incompatibility
   - Example: "0.2 and 0.5 are incompatible: 0.5/0.2 = 2.5, not an integer"

**Warning Message Templates:**
```python
# Auto-adjusted
f"summarise_every adjusted from {original} to {adjusted}, "
f"the nearest integer multiple of sample_summaries_every ({sample})"

# Floating point issue
f"summarise_every ({value}) is not exactly an integer multiple of "
f"sample_summaries_every ({sample}) due to floating point precision. "
f"Actual summary interval will be {actual_interval}"

# Incompatible
f"summarise_every ({se}) must be an integer multiple of "
f"sample_summaries_every ({sse}). The ratio {ratio:.4f} is not close "
f"to any integer. Example incompatible values: 0.2 and 0.5"
```

---

## Component 5: Default Timing Parameter Handling

**File:** `src/cubie/outputhandling/output_config.py`  
**Location:** Lines 182-185

**Current Behavior:**
```python
_save_every: float = attrs.field(
    default=0.01,
    validator=opt_gttype_validator(float, 0.0)
)
```

**Expected Behavior:**
- Change default to `None` or `0.0`
- Add validation at compile-time based on output types
- If `save_state` or `save_observables` or `save_time` is True, require `save_every > 0`
- If summaries requested, require `sample_summaries_every > 0` and `summarise_every > 0`

**Compile-Time Validation Points:**
1. In OutputFunctions.build() or during config construction
2. In IVPLoop when building with compile_flags

**Integration:**
- `compile_flags` must be accessible where timing validation occurs
- Error messages should guide users to set required parameters

---

## Component 6: Add sample_summaries_every to Config Plumbing Tests

**File:** `tests/batchsolving/test_config_plumbing.py`  
**Location:** Around line 278

**Current Behavior:**
Tests check `save_every` and `summarise_every` propagation.

**Expected Behavior:**
Add `sample_summaries_every` to the same checks:
- In `assert_ivploop_config()`: add check for `sample_summaries_every`
- In test updates dict: include `sample_summaries_every`
- Verify propagation through solver → kernel → loop hierarchy

**Additions Needed:**
```python
assert loop.sample_summaries_every == pytest.approx(
    settings["sample_summaries_every"], 
    rel=tolerance.rel_tight, 
    abs=tolerance.abs_tight
)
```

---

## Component 7: Consolidate SolveResult Field Tests

**File:** `tests/batchsolving/test_solveresult.py`  
**Location:** Lines 600-630 (individual field tests)

**Current Behavior:**
Individual tests for each SolveSpec field (e.g., `test_solvespec_save_every_field`, `test_solvespec_summarise_every_field`).

**Expected Behavior:**
Single test that verifies all expected attributes exist:
```python
def test_solvespec_has_all_expected_attributes():
    expected_attrs = [
        'dt', 'dt_min', 'dt_max', 'save_every', 'summarise_every',
        'sample_summaries_every', 'atol', 'rtol', 'duration', 'warmup',
        't0', 'algorithm', 'saved_states', 'saved_observables',
        'summarised_states', 'summarised_observables', 'output_types',
        'precision'
    ]
    spec = SolveSpec(...)
    for attr in expected_attrs:
        assert hasattr(spec, attr), f"SolveSpec missing attribute: {attr}"
```

---

## Component 8: Update Timing Validation Tests

**File:** `tests/integrators/loops/test_dt_update_summaries_validation.py`

**Current Behavior:**
Tests rely on sentinel values being set when timing parameters are None.

**Expected Behavior:**
Update tests for new behavior:
1. When all None: only flags set, no sentinel values
2. Compile-time validation when building with outputs requiring timing
3. Valid configurations should pass validation

**Test Cases to Update:**
- `test_all_none_uses_defaults`: Remove sentinel value assertions
- Add test: None timing params + no state output = valid
- Add test: None timing params + state output = compile error

---

## Component 9: Remove Mutual Exclusivity Assumption

**Files:** 
- `src/cubie/integrators/loops/ode_loop_config.py`
- Various test files

**Current Behavior:**
Some validation or test logic assumes `save_last` and `save_every` are mutually exclusive.

**Expected Behavior:**
- Both can be True simultaneously
- User may want periodic saves AND final state capture
- Remove any validation that prevents this combination
- Update tests that assert mutual exclusivity

**Search Pattern:**
Look for logic like:
```python
if save_last and save_every is not None:
    raise ValueError(...)
```

---

## Component 10: Implement summarise_last Logic in ode_loop

**File:** `src/cubie/integrators/loops/ode_loop.py`  
**Location:** Main loop, near save_last handling (lines 594-601)

**Current Behavior:**
`save_last` is implemented but `summarise_last` is not.

**Expected Behavior:**
Mirror `save_last` pattern for `summarise_last`:
1. At end of integration, if `summarise_last=True`:
   - Call `update_summaries()` with current state
   - Call `save_summaries()` to write final summary

**Implementation Pattern:**
```python
# Similar to save_last handling:
if summarise_last:
    at_last_summarise = finished and t_prec < t_end
    # Force summary collection on final step
```

**Edge Cases:**
- Handle case where regular summaries and last summary might double-write
- Ensure update_idx is correctly managed

---

## Component 11: Update solver_settings Default with sample_summaries_every

**File:** `tests/conftest.py`  
**Location:** `solver_settings` fixture, around line 388

**Current Behavior:**
Fixture has `save_every` and `summarise_every` but may lack `sample_summaries_every`.

**Expected Behavior:**
Add default:
```python
"sample_summaries_every": precision(0.05),  # Match save_every
```

Also update SHORT, MID, LONG run param sets if they exist.

---

## Component 12: Update CPU Reference Implementation

**File:** `tests/integrators/cpu_reference/loops.py`

**Changes Required:**
1. Replace `dt_save` references with `save_every`
2. Replace `dt_summarise` references with `summarise_every`
3. Add `sample_summaries_every` handling
4. Update summary generation logic:
   - Current: `summarise_every = int(dt_summarise / dt_save)` (confusing integer)
   - New: `samples_per_summary = summarise_every / sample_summaries_every`

**Summary Generation Logic:**
The `calculate_expected_summaries` function needs updating:
- Accept `sample_summaries_every` as parameter
- Handle cases where sampling frequency differs from save frequency
- Generate test data that reflects non-aligned sampling

---

## Component 13: Update Event Overlap Warning

**File:** `src/cubie/integrators/loops/ode_loop_config.py`

**Expected Behavior:**
When `sample_summaries_every` and `save_every` don't share a common period:
1. Calculate their LCM behavior
2. If events rarely overlap, warn about performance impact
3. Suggest alignment for better performance

**Warning Template:**
```python
f"save_every ({save}) and sample_summaries_every ({sample}) are not aligned. "
f"This may increase execution time. Consider aligning these values for "
f"better performance (e.g., one as an integer multiple of the other)."
```

---

## Validation Checklist for Detailed Implementer

### For Each Component:
- [ ] Identify exact line numbers in current code
- [ ] Determine imports needed (if any)
- [ ] List all affected tests
- [ ] Identify any precision/dtype considerations
- [ ] Note any CUDA-specific constraints

### Integration Testing:
- [ ] End-to-end test with new timing parameters
- [ ] Test with CUDASIM enabled
- [ ] Test with various precision levels (float32, float64)
- [ ] Test edge cases (very small timing values, large ratios)

### Documentation:
- [ ] Update docstrings for changed functions
- [ ] Update any inline comments affected

---

## Execution Order

Suggested implementation order to minimize conflicts:

1. **Component 6** (Test additions) - Can be done first as tests should fail
2. **Component 2** (Remove aliases) - Simple refactor
3. **Component 1** (Replace selp with min) - Simple optimization
4. **Component 4** (Tolerant validation) - Needed before sentinel removal
5. **Component 5** (Default timing) - Changes default behavior
6. **Component 3** (Remove sentinels) - Depends on 4, 5
7. **Component 8** (Update validation tests) - After 3
8. **Component 10** (summarise_last) - Independent
9. **Component 9** (Mutual exclusivity) - After 10
10. **Component 11** (Fixture updates) - Test infrastructure
11. **Component 12** (CPU reference) - After main changes
12. **Component 7** (Consolidate tests) - Cleanup
13. **Component 13** (Overlap warning) - Enhancement
