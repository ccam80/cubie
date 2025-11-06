# Derivative-Based Output Metrics - Human Overview

## User Stories

### Story 1: First Derivative Max Detection (Issue #66 - part 1)
**As a** researcher analyzing ECG waveforms  
**I want to** detect maximum slopes (dxdt_max) in real-time during GPU integration  
**So that I** can identify critical upslope features like QRS complexes without post-processing

**Acceptance Criteria:**
- Metric tracks maximum positive slope (max dxdt) over each summary interval
- Numerical differentiation computed using finite differences between consecutive saved states
- Works with variable dt_save intervals
- Compatible with all existing integrators and precision modes
- Scaling by dt_save done at save time to minimize roundoff error

### Story 2: First Derivative Min Detection (Issue #66 - part 2)
**As a** researcher analyzing ECG waveforms  
**I want to** detect minimum slopes (dxdt_min) in real-time during GPU integration  
**So that I** can identify critical downslope features without post-processing

**Acceptance Criteria:**
- Metric tracks maximum negative slope (min dxdt) over each summary interval
- Same implementation as dxdt_max but for minimum
- Can be combined with dxdt_max via dxdt_extrema combined metric

### Story 3: First Derivative Extrema (Issue #66 - combined)
**As a** user requesting both dxdt_max and dxdt_min  
**I want to** automatically get the more efficient combined metric  
**So that I** reduce buffer usage following the existing combined_metrics pattern

**Acceptance Criteria:**
- Auto-substitution when both dxdt_max and dxdt_min requested
- Follows pattern of mean_std_rms, extrema, etc.
- Single buffer tracks both max and min

### Story 4: Second Derivative Max Detection (Issue #67 - part 1)
**As a** cardiovascular researcher analyzing pulse wave dynamics  
**I want to** detect maximum acceleration (d2xdt2_max) during GPU integration  
**So that I** can identify compliance changes in real-time

**Acceptance Criteria:**
- Metric tracks maximum positive acceleration over each summary interval
- Second derivative computed using three-point finite differences
- Requires minimum 2 previous save points for valid computation
- Scaling by dt_save² done at save time

### Story 5: Second Derivative Min Detection (Issue #67 - part 2)
**As a** cardiovascular researcher analyzing pulse wave dynamics  
**I want to** detect minimum acceleration (d2xdt2_min) during GPU integration  
**So that I** can identify deceleration events in real-time

**Acceptance Criteria:**
- Same as d2xdt2_max but for minimum
- Can be combined with d2xdt2_max via d2xdt2_extrema

### Story 6: Second Derivative Extrema (Issue #67 - combined)
**As a** user requesting both d2xdt2_max and d2xdt2_min  
**I want to** automatically get the combined metric  
**So that I** reduce buffer usage

**Acceptance Criteria:**
- Auto-substitution following combined_metrics pattern

### Story 7: NOT IMPLEMENTED - dxdt Time Series (Issue #68)
**Status:** Deferred per user request  
**Rationale:** Issue states "easily done offline, only add if required in real-time"

## Overview

This feature implements **six new summary metrics** for derivative-based analysis during GPU batch integration, following the existing combined_metrics pattern:

**Individual Metrics:**
- dxdt_max - Maximum first derivative
- dxdt_min - Minimum first derivative
- d2xdt2_max - Maximum second derivative
- d2xdt2_min - Minimum second derivative

**Combined Metrics (auto-substituted):**
- dxdt_extrema - Combines dxdt_max + dxdt_min
- d2xdt2_extrema - Combines d2xdt2_max + d2xdt2_min

All metrics use **numerical differentiation** of saved state values.

**Issue #68 (dxdt time series) is NOT implemented** per user request.

### Key Architectural Decision: Numerical vs Analytical Derivatives

**Decision:** Use finite difference approximations of saved state values  
**Rationale:**
1. Summary metrics operate at save intervals (dt_save), not internal algorithm steps
2. ODE solver derivatives (dxdt) are computed at algorithm sub-steps with varying dt
3. Accessing solver derivatives would require fundamental changes to the output system architecture
4. Finite differences of saved states provide consistent derivative estimates aligned with output sampling
5. This approach is standard practice (NumPy's `np.gradient`, SciPy's `derivative`)

**Trade-offs:**
- ✅ Architecturally clean - no changes to integrator/algorithm interface
- ✅ Consistent derivative estimates at save intervals
- ✅ Works uniformly across all integration algorithms
- ✅ Scaling at save time reduces roundoff error
- ✅ No buffer flags - use zero guards
- ❌ Requires buffering previous state values
- ❌ Lower accuracy than analytical derivatives (acceptable for feature detection)

### ASCII Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Integration Loop                          │
│  (integrators/loops/ode_loop.py)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ At each dt_save interval
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              update_summaries()                              │
│  Passes: state_buffer, observables_buffer, save_idx         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ For each metric
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Metric.update(value, buffer, idx, param)           │
│                                                              │
│  EXISTING METRICS:          NEW DERIVATIVE METRICS:          │
│  ┌──────────────┐           ┌──────────────────────┐        │
│  │ mean         │           │ dxdt_extrema         │        │
│  │ max          │           │ - Store value[n-1]   │        │
│  │ std          │           │ - Compute dxdt       │        │
│  │ peaks        │           │ - Track max/min      │        │
│  │ extrema      │           └──────────────────────┘        │
│  └──────────────┘           ┌──────────────────────┐        │
│                             │ d2xdt2_extrema       │        │
│                             │ - Store value[n-1]   │        │
│                             │ - Store value[n-2]   │        │
│                             │ - Compute d2xdt2     │        │
│                             │ - Track max/min      │        │
│                             └──────────────────────┘        │
│                             ┌──────────────────────┐        │
│                             │ dxdt (optional)      │        │
│                             │ - Store value[n-1]   │        │
│                             │ - Compute dxdt       │        │
│                             │ - Accumulate series  │        │
│                             └──────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                       │
                       │ At summary interval
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         save_summaries()                                     │
│  Writes accumulated results to output arrays                │
└─────────────────────────────────────────────────────────────┘
```

### Finite Difference Schemes

**First Derivative (dxdt):**
- Backward difference: `dxdt ≈ (x[n] - x[n-1])`
- Division by dt_save done at save time: `output = unscaled_dxdt / dt_save`
- Error: O(dt_save)
- Buffer: previous value + accumulated max/min (unscaled)

**Second Derivative (d2xdt2):**
- Central difference: `d2xdt2 ≈ (x[n] - 2*x[n-1] + x[n-2])`
- Division by dt_save² done at save time: `output = unscaled_d2xdt2 / dt_save²`
- Error: O(dt_save²)
- Buffer: 2 previous values + accumulated max/min (unscaled)

**Key Change:** Accumulate unscaled differences, apply dt_save scaling at save to reduce roundoff error.

### Buffer Requirements

| Metric | Buffer Size | Storage |
|--------|-------------|---------|
| dxdt_max | 3 | prev_value, max_dxdt_unscaled, dt_save |
| dxdt_min | 3 | prev_value, min_dxdt_unscaled, dt_save |
| dxdt_extrema | 4 | prev_value, max_unscaled, min_unscaled, dt_save |
| d2xdt2_max | 4 | prev_value, prev_prev_value, max_unscaled, dt_save |
| d2xdt2_min | 4 | prev_value, prev_prev_value, min_unscaled, dt_save |
| d2xdt2_extrema | 5 | prev_value, prev_prev_value, max_unscaled, min_unscaled, dt_save |

**No initialization flags** - Use buffer[0] == 0.0 guards or assume zeroed memory + save called before updates.

### Integration Points

**Files to Create:**
- `src/cubie/outputhandling/summarymetrics/dxdt_max.py`
- `src/cubie/outputhandling/summarymetrics/dxdt_min.py`
- `src/cubie/outputhandling/summarymetrics/dxdt_extrema.py`
- `src/cubie/outputhandling/summarymetrics/d2xdt2_max.py`
- `src/cubie/outputhandling/summarymetrics/d2xdt2_min.py`
- `src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py`

**Files to Modify:**
- `src/cubie/outputhandling/summarymetrics/metrics.py` - Add dt_save to build(), update registry
- `src/cubie/outputhandling/summarymetrics/__init__.py` - Import new metrics
- `src/cubie/outputhandling/output_functions.py` - Pass dt_save to metric build()
- ALL existing metric files - Update build(self, dt_save) signature
- `tests/outputhandling/summarymetrics/test_summary_metrics.py` - Add validation tests

**Testing:**
- Add to existing `test_all_summaries_long_run` 
- Add to `test_all_summary_metrics_numerical_check`
- Validate against numpy.diff and finite difference formulas
- No new test files or performance tests

### Key Technical Decisions

1. **dt_save Access:** Modify ALL summary metric factory signatures
   - Add dt_save parameter to build() method: `def build(self, dt_save: float)`
   - Pass dt_save from output_functions.py during metric compilation
   - Change customisable_variable type from int32 to float32/64
   - All existing metrics must be updated for consistency

2. **Initialization:** Use zero guards, no flags
   - buffer[0] == 0.0 means first call or post-save
   - Presume: (a) memories zeroed before start, (b) save called once before updates
   - No initialization counters or flags (buffer memory is precious)

3. **Scaling:** Done at save time, not during accumulation
   - Store unscaled differences: `unscaled_dxdt = x[n] - x[n-1]`
   - Scale at save: `output = unscaled_dxdt / dt_save`
   - Reduces roundoff error accumulation

4. **No Conditional Returns:** All CUDA device code must reach end
   - Use guards with if statements but always continue
   - No early returns in device functions

5. **Combined Metrics:** Follow existing pattern
   - dxdt_max + dxdt_min → dxdt_extrema (auto-substitute)
   - d2xdt2_max + d2xdt2_min → d2xdt2_extrema (auto-substitute)
   - Registered in _combined_metrics dict

6. **Issue #68 Deferred:** Do NOT implement dxdt time series
   - User explicitly requested deferral
   - Issue notes: "easily done offline, only add if required in real-time"

### Expected Impact on Architecture

**Signature Changes (Breaking):**
- ALL summary metric build() methods must add dt_save parameter
- customisable_variable changes from int32 to float32/64 in device function signatures
- Affects ALL existing metrics: mean, max, min, rms, std, peaks, negative_peaks, max_magnitude, extrema, mean_std, std_rms, mean_std_rms

**Minimal Other Changes:**
- New metric files follow existing pattern
- Auto-registration via @register_metric decorator
- No changes to core integration loop
- output_functions.py passes dt_save to build()

**Combined Metrics Registry:**
- Add two new entries to _combined_metrics dict
- Auto-substitution when both parts requested

### Performance Considerations

**Memory:**
- dxdt_max/min: +3 floats per variable per batch run
- d2xdt2_max/min: +4 floats per variable per batch run
- Combined versions save 1 buffer slot each
- Negligible compared to state output arrays

**Computation:**
- Each update adds 2-3 FLOPs (subtract, compare, store)
- Fully inlined CUDA device functions
- Division by dt_save only at save time (once per period)
- Expected overhead: minimal (no performance testing required)

**Note:** Performance testing explicitly NOT required per user request.

### Alternative Approaches Considered

1. **Access ODE Solver Derivatives Directly**
   - ❌ Requires changing integrator loop to store/pass dxdt
   - ❌ Algorithm-dependent (implicit methods compute differently)
   - ❌ Sub-step derivatives don't align with save intervals

2. **Higher-Order Finite Differences**
   - ❌ Requires more buffer storage
   - ❌ Marginal accuracy improvement for feature detection
   - ✅ Could add later if needed (backward compatible)

3. **Spline Interpolation for Derivatives**
   - ❌ Computationally expensive on GPU
   - ❌ Overkill for extrema detection
   - ✅ Offline post-processing more appropriate

## Research Findings

### Related GitHub Issues
- Issue #141: Parent tracking issue for output functions
- Issues #61-65: Already implemented (max_magnitude, std, min, negative_peaks, extrema)
- Issue #66: dxdt extrema (this plan)
- Issue #67: d2xdt2 extrema (this plan)
- Issue #68: dxdt output (optional - awaiting user confirmation)

### Existing Metric Patterns
- All metrics inherit from SummaryMetric (CUDAFactory)
- Update function signature: `(value, buffer, current_index, customisable_variable)`
- Save function signature: `(buffer, output_array, summarise_every, customisable_variable)`
- Parameterized metrics use callable buffer_size: `lambda n: ...`

### Numerical Methods References
- Standard finite difference formulas (Burden & Faires, Numerical Analysis)
- NumPy gradient implementation uses central differences when possible
- SciPy derivative uses Richardson extrapolation for higher accuracy
- Our use case (feature detection) doesn't require highest accuracy

## Open Questions for User

**ALL QUESTIONS ANSWERED - proceeding with implementation based on user feedback:**

1. ✅ **Issue #68 Priority:** NOT implementing dxdt time series (per user request)
2. ✅ **dt_save Passing:** Modify ALL metric factory signatures to accept dt_save parameter
3. ✅ **Scaling:** Done at save time, not during accumulation (per user request)
4. ✅ **Initialization:** Use zero guards, no flags (per user request)
5. ✅ **Combined Metrics:** Implement individual + combined versions following existing pattern
6. ✅ **Testing:** Add to existing tests only (per user request)
7. ✅ **Performance:** No performance tests (per user request)
8. ✅ **Documentation:** No examples/guides updates (per user request)
