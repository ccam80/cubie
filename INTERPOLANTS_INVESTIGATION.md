# Interpolants Investigation: Requirements for FIRK, DIRK, and Rosenbrock

## Executive Summary

This document investigates what is required to implement dense output (continuous extension) via interpolants for generic_firk, generic_dirk, and generic_rosenbrock_w, as requested in the interpolants feature issue.

**Status**: Infrastructure added (`has_interpolant` property, `b_interp` field). Full implementation requires additional research and development.

## Background

### Current Problem

The integration loop in `ode_loop.py` (line 406) truncates steps to hit save points exactly:
```python
dt_eff = selp(do_save, next_save - t, dt[0])
```

The truncated `dt_eff` is passed to step functions, but the resulting error estimate is used by the step controller as if it came from a full step. This **inflates the error estimate**, potentially causing step size increases that will subsequently be rejected.

### Proposed Solution

Use interpolants to:
1. Always take **full steps** (use `dt[0]`, never truncate)
2. Compute **accurate error estimates** for full steps
3. When save points fall within steps, compute **interpolated values** at those points
4. Save interpolated values while the integration continues with full steps

This preserves error estimate accuracy while hitting save points precisely.

## Changes Required

### 1. Base Infrastructure (✅ COMPLETED)

**File**: `src/cubie/integrators/algorithms/base_algorithm_step.py`

**Changes Made**:
- Added `b_interp` field to `ButcherTableau` (optional tuple of tuples)
- Added `has_interpolant` property returning `bool`
- Added `interpolant_coefficients()` method returning typed coefficients
- Added validation for `b_interp` dimensions in `__attrs_post_init__`

**Testing**: Basic import and property access verified.

### 2. Loop Modifications (TODO)

**File**: `src/cubie/integrators/loops/ode_loop.py`

**Required Changes**:
1. Add `next_save` to step function call parameters (line ~410)
2. Add compile-time toggle based on `tableau.has_interpolant`:
   ```python
   has_interpolant = <from step function metadata>
   if not has_interpolant:
       dt_eff = selp(do_save, next_save - t, dt[0])  # Current behavior
   else:
       dt_eff = dt[0]  # Always full step
   ```
3. Add interpolated state/observables buffers to shared memory layout
4. When `do_save=True` and `has_interpolant=True`, save interpolated values

**Shared Memory Layout Changes**:
Current layout in `LoopSharedIndices` needs additional slices for:
- `interpolated_state`: size `n_states` (when interpolants enabled)
- `interpolated_observables`: size `n_observables` (when interpolants enabled)

**Alternative Approach**: Pass `next_save` via shared memory to avoid changing step function signatures.

### 3. Step Function Modifications

Each of the three step types needs similar modifications. Using DIRK as the example:

**File**: `src/cubie/integrators/algorithms/generic_dirk.py`

**In `build_step()` method**:

```python
# Capture compile-time flag
has_interpolant = self.tableau.has_interpolant

# Inside the device function @cuda.jit step(...):
if has_interpolant:
    # Check if next_save falls in [current_time, current_time + dt_value]
    save_in_step = (next_save >= current_time) and (
        next_save <= current_time + dt_value
    )
    
    if save_in_step:
        # Compute theta = (next_save - current_time) / dt_value
        theta = (next_save - current_time) / dt_value
        
        # Evaluate interpolant using b_interp coefficients and stage values
        # Details depend on interpolant formula (see below)
        for idx in range(n):
            interpolated_state[idx] = evaluate_interpolant(
                theta, state[idx], stage_rhs, b_interp_coeffs
            )
        
        # Compute interpolated observables
        observables_function(
            interpolated_state,
            parameters,
            drivers_at_save,  # May need driver evaluation at next_save
            interpolated_observables,
            next_save,
        )
```

**Similar changes needed for**:
- `src/cubie/integrators/algorithms/generic_firk.py`
- `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

### 4. Tableau Coefficient Updates (TODO)

Each tableau that supports dense output needs `b_interp` coefficients added.

#### 4.1 DIRK Tableaus

**File**: `src/cubie/integrators/algorithms/generic_dirk_tableaus.py`

**Tableaus to Update**:

1. **TRAPEZOIDAL_DIRK_TABLEAU** (Crank-Nicolson)
   - Method: 2nd order, 2 stages at c=[0, 1]
   - Interpolant: Cubic Hermite (3rd order dense output)
   - Coefficients: Need to derive from Hermite basis
   - Complexity: **Moderate** (well-documented)

2. **LOBATTO_IIIC_3_TABLEAU**
   - Method: 4th order, 3 stages
   - Interpolant: Requires 4th+ order dense output
   - Coefficients: Available in Hairer & Wanner (1996)
   - Complexity: **High** (complex derivation)

3. **Other DIRK tableaus**: Lower priority, would follow same pattern

#### 4.2 FIRK Tableaus

**File**: `src/cubie/integrators/algorithms/generic_firk_tableaus.py`

**Tableaus to Update**:

1. **GAUSS_LEGENDRE_2_TABLEAU**
   - Method: 4th order, 2 stages
   - Interpolant: 3rd order Hermite typically used
   - Coefficients: Available in literature
   - Complexity: **High** (symmetric stages complicate derivation)

2. **RADAU_IIA_5_TABLEAU**
   - Method: 5th order, 3 stages
   - Interpolant: 4th+ order dense output
   - Coefficients: Available in Hairer & Wanner
   - Complexity: **Very High**

#### 4.3 Rosenbrock Tableaus

**File**: `src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py`

**Tableaus to Update**:

1. **ROS3P_TABLEAU**
   - Method: 3rd order, 3 stages
   - Interpolant: Likely documented in Lang & Verwer (2001)
   - Coefficients: Need to check original paper
   - Complexity: **Moderate** (linear implicit methods have simpler dense output)

## Dense Output Formulas

### General Form

For a Runge-Kutta method, dense output is typically expressed as:
```
y(t + theta*dt) = y(t) + dt * Σ b_i(theta) * k_i
```

where:
- `theta ∈ [0, 1]` is the interpolation parameter
- `k_i` are the stage derivatives
- `b_i(theta)` are interpolant coefficient polynomials

### Representation in Code

The `b_interp` field stores coefficients for the polynomial representation:
```python
b_interp = (
    (b_0_c0, b_1_c0, ...),  # Constant term coefficients
    (b_0_c1, b_1_c1, ...),  # theta^1 term coefficients  
    (b_0_c2, b_1_c2, ...),  # theta^2 term coefficients
    ...
)
```

Then the interpolant is evaluated as:
```python
for stage_idx in range(stage_count):
    weight = 0
    theta_power = 1
    for coeff_row in b_interp:
        weight += coeff_row[stage_idx] * theta_power
        theta_power *= theta
    y_interp[idx] += dt * weight * k[stage_idx][idx]
```

### Hermite Interpolation (Common Approach)

For methods with stages at c=[0, 1], cubic Hermite interpolation is common:

**Basis Functions**:
- `H_0(theta) = (1-theta)^2 * (1+2*theta)`  # Value at start
- `H_1(theta) = theta^2 * (3-2*theta)`      # Value at end
- `H_0'(theta) = theta * (1-theta)^2`        # Derivative at start
- `H_1'(theta) = theta^2 * (theta-1)`        # Derivative at end

**Interpolant**:
```
y(theta) = H_0(theta)*y(t) + H_1(theta)*y(t+dt) 
         + dt*H_0'(theta)*k_0 + dt*H_1'(theta)*k_1
```

Since `y(t+dt) = y(t) + dt*(b_0*k_0 + b_1*k_1)`, this can be rewritten in the standard form.

## Implementation Challenges

### 1. Warp Divergence

**Issue**: Different threads may reach save points at different times, causing warp divergence.

**Mitigation**: Already present in current save logic. Use predicated execution:
```python
do_interpolation = save_in_step and has_interpolant
for idx in range(n):
    value = selp(do_interpolation, interpolated_value, current_value)
    output[idx] = value
```

### 2. Driver Evaluation at Save Points

**Issue**: Observables may depend on drivers, which need evaluation at `next_save` (not just at step start/end).

**Options**:
- Evaluate drivers at `next_save` when interpolating
- Interpolate drivers using same approach as state
- Require driver interpolation to be linear (simple)

**Recommendation**: Evaluate drivers at `next_save` directly (simplest, drivers are typically cheap to evaluate).

### 3. Buffer Management

**Issue**: Need space for interpolated state/observables.

**Options**:
A. **Dedicated buffers** in shared memory (increases memory usage)
B. **Reuse proposed_state** buffer with careful ordering
C. **Thread-local storage** in per-thread local memory

**Recommendation**: Option C (local memory) for small state dimensions, Option A for larger systems.

### 4. Step Function Signature

**Issue**: Need to pass `next_save` to step functions.

**Options**:
A. Add `next_save` parameter to signature
B. Pass via shared memory buffer
C. Pass via persistent local memory

**Recommendation**: Option A (cleanest, most explicit) or B (avoids signature churn).

### 5. Counter Accumulation

**Issue**: When steps aren't truncated, iteration counters may accumulate differently.

**Solution**: Counters should accumulate for full steps regardless of saves. No change needed.

## Testing Strategy

### Unit Tests

1. **Interpolant Accuracy**:
   - Test against exact solutions (exponential, polynomial)
   - Verify interpolant matches analytical solution at intermediate points
   - Check that interpolant equals step-end value at theta=1

2. **Tableau Validation**:
   - Verify `b_interp` coefficients match literature
   - Check order conditions for dense output
   - Validate against reference implementations (scipy, Julia)

### Integration Tests

1. **Error Estimate Accuracy**:
   - Compare error estimates with vs without interpolation
   - Verify full-step errors are not inflated
   - Check step controller behavior

2. **Save Point Precision**:
   - Verify interpolated values match truncated values (same save points)
   - Test with non-uniform save intervals
   - Check edge cases (save at step boundaries)

3. **Stiff Problem Performance**:
   - Compare truncation vs interpolation on van der Pol, Robertson, etc.
   - Measure step acceptance rate improvement
   - Benchmark interpolation overhead

### Performance Tests

1. **Memory Usage**: Check shared memory requirements with interpolation buffers
2. **Warp Efficiency**: Profile divergence in interpolation branches
3. **Throughput**: Benchmark steps/second with/without interpolation

## Literature References

### Primary Sources

1. **Hairer, E., & Wanner, G. (1996)**. *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.). Springer.
   - Sections on dense output for DIRK and FIRK methods
   - Tableau coefficients for Gauss-Legendre, Radau IIA, Lobatto

2. **Lang, J., & Verwer, J. (2001)**. "ROS3P—An Accurate Third-Order Rosenbrock Solver Designed for Parabolic Problems." *BIT Numerical Mathematics* 41, 731–738.
   - Original ROS3P paper (may contain dense output formula)

3. **Shampine, L. F. (1985)**. "Interpolation for Runge-Kutta Methods." *SIAM Journal on Numerical Analysis*, 22(5), 1014-1027.
   - General theory of continuous extension for RK methods

### Implementation References

4. **SciML/DifferentialEquations.jl**: Julia ODE solver library
   - Contains dense output implementations for many tableaus
   - GitHub: https://github.com/SciML/OrdinaryDiffEq.jl

5. **scipy.integrate.solve_ivp**: Python reference implementation
   - Dense output for Dormand-Prince and Radau

## Next Steps

### Immediate (Investigation Complete)

- [x] Add base infrastructure to `ButcherTableau`
- [x] Document requirements for each method type
- [x] Identify implementation challenges
- [x] Create testing strategy

### Phase 2 (Optional Follow-up)

- [ ] Derive/research `b_interp` coefficients for one simple tableau (e.g., trapezoidal)
- [ ] Implement interpolation in one step type as proof-of-concept
- [ ] Modify loop for single tableau case
- [ ] Write tests for proof-of-concept

### Phase 3 (Full Implementation)

- [ ] Derive coefficients for all supported tableaus
- [ ] Implement interpolation in all three step types
- [ ] Comprehensive testing suite
- [ ] Performance optimization
- [ ] Documentation updates

## Conclusion

Implementing dense output via interpolants is **feasible** but requires:

1. **Moderate effort** for infrastructure and simple tableaus (trapezoidal)
2. **High effort** for complex tableaus (FIRK methods, high-order DIRK)
3. **Careful testing** to ensure interpolant accuracy and performance

The main technical challenges are:
- Deriving/verifying interpolant coefficients from literature
- Managing additional buffer requirements
- Maintaining warp efficiency with predicated interpolation

The main **benefit** is eliminating error estimate inflation, which should improve step controller efficiency and allow larger time steps on average.

**Recommendation**: Start with a proof-of-concept for TRAPEZOIDAL_DIRK_TABLEAU (simplest case) before tackling more complex methods.
