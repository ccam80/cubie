# Interpolants Investigation - Summary for Issue #XXX

## Executive Summary

I've completed the investigation into what additions would be required to implement dense output (continuous extension) via interpolants in `generic_firk`, `generic_dirk`, and `generic_rosenbrock`. The infrastructure is now in place, and I've provided a comprehensive analysis with a working proof-of-concept.

## What Was Delivered

### 1. Core Infrastructure (✅ Complete)

**File**: `src/cubie/integrators/algorithms/base_algorithm_step.py`

- Added `b_interp` field to `ButcherTableau` (optional tuple of tuples for dense output coefficients)
- Added `has_interpolant` property (returns `True` when `b_interp` is not `None`)
- Added `interpolant_coefficients()` method for type-safe access
- Added validation ensuring all `b_interp` rows match stage count

**Testing**: All existing tests pass. New infrastructure is backward-compatible.

### 2. Comprehensive Investigation Document

**File**: `INTERPOLANTS_INVESTIGATION.md` (373 lines)

Key sections:
- **Current Problem**: Step truncation inflates error estimates (line 406 in `ode_loop.py`)
- **Proposed Solution**: Use interpolants to take full steps while computing intermediate values
- **Requirements by Method**: Detailed analysis for FIRK, DIRK, and Rosenbrock-W
- **Implementation Challenges**: Warp divergence, buffer management, signature changes
- **Testing Strategy**: Unit, integration, and performance tests
- **Literature References**: Primary sources for interpolant coefficients

### 3. Working Proof-of-Concept

**File**: `src/cubie/integrators/algorithms/generic_dirk_tableaus.py`

- Added placeholder linear interpolant to `TRAPEZOIDAL_DIRK_TABLEAU`
- Demonstrates the pattern: `b_interp = ((c0_0, c0_1), (c1_0, c1_1), ...)`
- Note: Currently uses simple linear interpolation; proper 2nd-order dense output requires cubic Hermite interpolation

**File**: `tests/integrators/algorithms/test_interpolant_concept.py`

Three tests demonstrating interpolant usage:
1. `test_trapezoidal_has_interpolant`: Verifies tableau has interpolant support
2. `test_interpolant_evaluation_concept`: Shows evaluation pattern for device code
3. `test_interpolant_at_boundaries`: Validates boundary conditions (theta=0 and theta=1)

**All tests passing** ✅

## Findings Summary

### Complexity Assessment

| Method Type | Tableau Example | Complexity | Reason |
|------------|----------------|-----------|---------|
| DIRK | Trapezoidal | **Moderate** | Simple linear/Hermite interpolation |
| DIRK | Lobatto IIIC | **High** | 4th order dense output required |
| FIRK | Gauss-Legendre | **High** | Symmetric stages complicate derivation |
| FIRK | Radau IIA | **Very High** | 5th order with complex structure |
| Rosenbrock | ROS3P | **Moderate** | Linear implicit methods simpler |

### Implementation Phases

**Phase 1** (✅ Complete): Infrastructure
- Base tableau support for interpolants
- Documentation of requirements

**Phase 2** (Recommended Next): Loop Modifications
- Pass `next_save` to step functions
- Add compile-time branch: if `has_interpolant`, don't truncate `dt`
- Add interpolated state/observables buffers to shared memory layout

**Phase 3**: Step Function Updates
- Check if `next_save ∈ [t, t+dt]` in each step type
- Compute interpolant when needed
- Store in appropriate buffers

**Phase 4**: Tableau Coefficients
- Research/derive interpolant coefficients for each tableau
- Start with simple cases (trapezoidal Hermite)
- Validate against literature

**Phase 5**: Testing & Validation
- Unit tests for interpolant accuracy
- Integration tests comparing truncation vs interpolation
- Performance benchmarks

## Specific Method Requirements

### generic_dirk.py

**Changes Needed**:
1. In `build_step()`, capture `tableau.has_interpolant` at compile time
2. Add device code to:
   - Check if `next_save` falls within current step
   - Compute `theta = (next_save - t) / dt`
   - Evaluate `y_interp = y(t) + dt * Σ b_interp_i(theta) * k_i`
   - Store interpolated state/observables

**Tableaus to Update**:
- `TRAPEZOIDAL_DIRK_TABLEAU`: Start here (simplest, 2nd order)
- `LOBATTO_IIIC_3_TABLEAU`: Next priority (4th order)
- Others: Lower priority

**Coefficient Sources**: Hairer & Wanner (1996), Chapter IV

### generic_firk.py

**Changes Needed**:
Similar to DIRK, but:
- All stages solved simultaneously (coupled system)
- Interpolant computed after full solve completes
- May reuse stage_rhs_flat buffer containing all k_i values

**Tableaus to Update**:
- `GAUSS_LEGENDRE_2_TABLEAU`: 4th order method, 3rd order dense output
- `RADAU_IIA_5_TABLEAU`: 5th order method, complex dense output

**Coefficient Sources**: Hairer & Wanner (1996), Chapter IV; Butcher (2016)

**Challenge**: Symmetric Gauss methods have special structure requiring careful derivation

### generic_rosenbrock_w.py

**Changes Needed**:
Similar pattern, but:
- Rosenbrock methods compute stage increments, not derivatives
- Interpolant formula may differ slightly
- Linear implicit structure may simplify coefficients

**Tableaus to Update**:
- `ROS3P_TABLEAU`: 3rd order method, likely has documented dense output

**Coefficient Sources**: Lang & Verwer (2001) - original ROS3P paper

**Advantage**: Linear implicit methods generally have simpler dense output formulas

## Loop Integration

**File**: `src/cubie/integrators/loops/ode_loop.py`

**Required Changes** (lines ~405-430):

```python
# Current (line 405-406):
do_save = (t + dt[0] + equality_breaker) >= next_save
dt_eff = selp(do_save, next_save - t, dt[0])

# Proposed (when has_interpolant=True):
do_save = (t + dt[0] + equality_breaker) >= next_save
if has_interpolant:
    dt_eff = dt[0]  # Always take full step
    # Pass next_save to step function
else:
    dt_eff = selp(do_save, next_save - t, dt[0])  # Current behavior
```

**Parameter Passing**: Two options
1. Add `next_save` parameter to step function signature
2. Pass via shared memory buffer (avoids signature changes)

**Recommendation**: Option 1 (explicit parameter) for clarity

## Open Questions (from Investigation)

1. Should drivers be interpolated or evaluated at `next_save`?
   - **Recommendation**: Evaluate at `next_save` (simple, cheap)

2. How to handle counter accumulation when steps aren't truncated?
   - **Answer**: No change needed - counters accumulate for full steps

3. Should error estimates include interpolation error?
   - **Answer**: No - error estimate is for the full step

4. What precision for hitting save points?
   - **Current**: Uses `equality_breaker` (1e-7 or 1e-14)
   - **Recommendation**: Keep current approach

## Recommended Implementation Path

### Minimal Working Example (1-2 days)

1. Derive proper cubic Hermite coefficients for `TRAPEZOIDAL_DIRK_TABLEAU`
2. Modify loop to pass `next_save` (conditional on `has_interpolant`)
3. Add interpolation logic to `DIRKStep.build_step()`
4. Test on simple exponential decay problem
5. Verify error estimates are accurate (not inflated)

### Full DIRK Support (1 week)

1. Add interpolants to all DIRK tableaus with literature validation
2. Comprehensive testing against reference solutions
3. Performance profiling and optimization

### FIRK and Rosenbrock Support (2-3 weeks)

1. Research and validate coefficients for each method
2. Implement in respective step builders
3. Extensive validation and performance testing

## Testing Recommendations

### Unit Tests

```python
def test_interpolant_matches_exact_solution():
    """For y' = -y, y(0) = 1, verify interpolant matches e^(-t)."""
    # Take step from t=0 to t=1 with dt=1
    # Compute interpolant at theta=0.5
    # Compare to e^(-0.5) ≈ 0.6065
    
def test_interpolant_order():
    """Verify dense output maintains expected order."""
    # For 2nd order method with 2nd order dense output:
    # Error should scale as O(theta^3) not O(theta^2)
```

### Integration Tests

```python
def test_error_estimate_accuracy():
    """Verify full steps give accurate error estimates."""
    # Compare error estimates with and without interpolation
    # Without: error is inflated due to truncation
    # With: error should be accurate for full step

def test_save_point_precision():
    """Verify interpolated saves hit correct times."""
    # Run with irregular save interval
    # Check that saved times match requested times
    # Verify interpolated values are accurate
```

## Conclusion

The investigation is complete, and the foundation is in place. Key takeaways:

✅ **Infrastructure Ready**: `has_interpolant` and `b_interp` fields are available
✅ **Pattern Established**: Proof-of-concept shows how to evaluate interpolants
✅ **Requirements Documented**: Clear roadmap for each method type
✅ **Backward Compatible**: All existing tests pass

**Next Action**: Decide implementation priority:
- **Quick Win**: Complete trapezoidal DIRK with proper Hermite coefficients (2-3 days)
- **Full Support**: All three method types with comprehensive testing (4-6 weeks)
- **Staged Rollout**: Start with DIRK (moderate), then Rosenbrock (moderate), finally FIRK (high complexity)

**Benefit**: Eliminating step truncation will improve step controller efficiency and allow larger timesteps, particularly for stiff problems where adaptive stepping is crucial.

## Files Modified

1. `src/cubie/integrators/algorithms/base_algorithm_step.py` - Core infrastructure
2. `src/cubie/integrators/algorithms/generic_dirk_tableaus.py` - Proof-of-concept
3. `tests/integrators/algorithms/test_interpolant_concept.py` - Demonstration tests
4. `INTERPOLANTS_INVESTIGATION.md` - Comprehensive documentation

## References

- Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential Equations II*
- Lang, J., & Verwer, J. (2001). "ROS3P—An Accurate Third-Order Rosenbrock Solver"
- Shampine, L. F. (1985). "Interpolation for Runge-Kutta Methods"
- SciML/DifferentialEquations.jl - Julia reference implementation
