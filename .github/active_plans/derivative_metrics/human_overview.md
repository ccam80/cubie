# Derivative-Based Output Metrics - Human Overview

## User Stories

### Story 1: First Derivative Extrema Detection (Issue #66)
**As a** researcher analyzing ECG waveforms  
**I want to** detect maximum slopes (dxdt extrema) in real-time during GPU integration  
**So that I** can identify critical features like QRS complexes without post-processing

**Acceptance Criteria:**
- Metric tracks maximum positive slope (max dxdt) over each summary interval
- Metric tracks maximum negative slope (min dxdt) over each summary interval
- Numerical differentiation computed using finite differences between consecutive saved states
- Output includes both max and min slope values per summary period
- Works with variable dt_save intervals
- Compatible with all existing integrators and precision modes

**Success Metrics:**
- Correctly identifies peak slopes in test ECG-like waveforms
- Performance overhead < 5% compared to existing extrema metric
- Passes validation against scipy gradient-based post-processing

### Story 2: Second Derivative Extrema Detection (Issue #67)
**As a** cardiovascular researcher (Liam Murphy) analyzing pulse wave dynamics  
**I want to** detect acceleration extrema (d2xdt2 extrema) during GPU integration  
**So that I** can identify inflection points and compliance changes in real-time

**Acceptance Criteria:**
- Metric tracks maximum positive acceleration (max d2xdt2) over each summary interval
- Metric tracks maximum negative acceleration (min d2xdt2) over each summary interval
- Second derivative computed using three-point finite differences
- Output includes both max and min acceleration values per summary period
- Requires minimum 3 save points for valid computation
- Compatible with all existing integrators and precision modes

**Success Metrics:**
- Correctly identifies acceleration extrema in pulse wave test signals
- Performance overhead < 7% compared to existing extrema metric
- Passes validation against scipy second derivative post-processing

### Story 3: Real-Time Derivative Output (Issue #68)
**As a** researcher requiring online derivative information  
**I want to** optionally output full derivative time series (dxdt) during integration  
**So that I** can perform real-time derivative-dependent analysis without offline post-processing

**Acceptance Criteria:**
- Metric outputs full time series of first derivatives at each save point
- Only implemented if explicitly required (not default)
- Uses same finite difference method as derivative extrema metrics
- Compatible with saved state output format
- Minimal memory overhead (shares computation with extrema if both requested)

**Success Metrics:**
- Derivative output matches post-processing with np.gradient to within numerical precision
- Can be disabled when not needed (zero overhead when not requested)
- Seamlessly integrates with existing output system

## Overview

This feature implements three new summary metrics for derivative-based analysis during GPU batch integration. All three metrics use **numerical differentiation** of saved state values rather than accessing ODE solver derivatives directly.

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
- ❌ Requires buffering previous state values
- ❌ Lower accuracy than analytical derivatives (acceptable for feature detection)
- ❌ Requires minimum history for second derivatives

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
- Forward difference: `dxdt ≈ (x[n] - x[n-1]) / dt_save`
- Backward difference used (simpler, requires only 1 previous value)
- Error: O(dt_save)

**Second Derivative (d2xdt2):**
- Central difference: `d2xdt2 ≈ (x[n] - 2*x[n-1] + x[n-2]) / dt_save²`
- Requires 2 previous values in buffer
- Error: O(dt_save²)

### Buffer Requirements

| Metric | Buffer Size | Storage |
|--------|-------------|---------|
| dxdt_extrema | 4 | prev_value, max_dxdt, min_dxdt, init_flag |
| d2xdt2_extrema | 5 | prev_value, prev_prev_value, max_d2xdt2, min_d2xdt2, init_flag |
| dxdt (optional) | lambda n: n + 1 | prev_value + full series accumulation |

### Integration Points

**Files to Create:**
- `src/cubie/outputhandling/summarymetrics/dxdt_extrema.py`
- `src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py`
- `src/cubie/outputhandling/summarymetrics/dxdt.py` (conditional - only if needed)

**Files to Modify:**
- None (metrics auto-register via decorator)

**Testing:**
- `tests/outputhandling/summarymetrics/test_summary_metrics.py` - extend existing tests
- New validation tests against scipy.signal and numpy.gradient

### Key Technical Decisions

1. **dt_save Access:** Metrics need dt_save for finite difference denominators
   - Solution: Pass dt_save as the customisable_variable parameter
   - Requires coordination with output_functions.py compilation

2. **Initialization:** First 1-2 points have insufficient history
   - Solution: Use sentinel flag in buffer, only update after warmup
   - d2xdt2_extrema requires 2 warmup points

3. **Combined Metrics:** Should dxdt + d2xdt2 be combined?
   - Decision: Keep separate initially (different use cases)
   - Can add combined metric later if performance critical

4. **Optional dxdt Output:** Full derivative time series
   - Decision: Implement only if user confirms requirement
   - Large memory footprint (same as state output)
   - Note in issue #68: "easily done offline, only add if required in real-time"

### Expected Impact on Architecture

**Minimal Changes:**
- New metric files follow existing pattern (Mean, Extrema, Peaks)
- Auto-registration via @register_metric decorator
- No changes to core integration loop
- No changes to output system interfaces

**Potential Enhancement:**
- Output system may need dt_save accessible to metrics
- Current customisable_variable can pass dt_save as float
- Alternative: Add dt_save to metric update signature (breaking change - avoid)

### Performance Considerations

**Memory:**
- dxdt_extrema: +4 floats per variable per batch run
- d2xdt2_extrema: +5 floats per variable per batch run
- Negligible compared to state output arrays

**Computation:**
- Each update adds 2-3 FLOPs (subtract, divide, compare)
- Fully inlined CUDA device functions
- Expected overhead: 3-7% vs existing extrema metric

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

1. **Issue #68 Priority:** Should we implement full dxdt time series output, or defer until explicitly needed?
   - Recommendation: Defer - issue states "only add if required in real-time"
   - Can easily add later if needed

2. **dt_save Passing:** Current approach passes dt_save as customisable_variable parameter
   - Is this acceptable, or prefer alternative method?
   - Alternative would require signature change (breaking)

3. **Combined Metrics:** Performance testing will show if dxdt+d2xdt2 combined metric is beneficial
   - Implement separately first?
   - Add combined version if bottleneck identified?
