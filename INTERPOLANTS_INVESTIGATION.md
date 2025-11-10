# Interpolants Implementation Plan

## Overview

This document provides an implementation plan for adding dense output (continuous extension) via interpolants to generic_firk, generic_dirk, and generic_rosenbrock_w, as requested in issue #180.

**Status**: Infrastructure added (`has_interpolant` property, `b_interp` field). This document outlines the implementation approach based on investigation and feedback.

## Background

### Current Problem

The integration loop in `ode_loop.py` (line 406) truncates steps to hit save points exactly:
```python
dt_eff = selp(do_save, next_save - t, dt[0])
```

The truncated `dt_eff` is passed to step functions, but the resulting error estimate is used by the step controller as if it came from a full step. This **inflates the error estimate**, potentially causing step size increases that will subsequently be rejected.

### Proposed Solution

Instead of truncating steps, the solver will:
1. Take **truncated steps when approaching save points** (preserving the truncation behavior)
2. But the step will be **repeated** after the save, eliminating the wasted work
3. The error estimate will be **commensurate with the truncated step**, addressing the inflation issue
4. This approach duplicates a portion of the next step but avoids the error estimate problem

This solution trades some duplicate computation for accurate error estimates without requiring additional buffer space.

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
1. Pass `next_save` via shared memory buffer to avoid changing step function signatures
2. Step functions will check on every iteration whether to compute interpolant
3. Use conditional commit with `selp` to avoid warp divergence:
   ```python
   # Execute interpolation logic on every step
   needs_interp = do_save and has_interpolant and (next_save >= t) and (next_save <= t + dt)
   
   # Compute theta for interpolation (always computed, conditionally used)
   theta = (next_save - t) / dt
   
   # Interpolate to next_save, writing to proposed_state/observables buffers
   # Use selp to conditionally commit the interpolated values
   t_save = selp(needs_interp, next_save, t + dt)
   ```
4. Write interpolated values directly to `proposed_state` and `proposed_observables` buffers
5. Update time with `selp(needs_interp, next_save, t + dt)` to handle save point

**No Additional Buffers Required**: Reuse existing `proposed_state`, `proposed_observables` buffers with careful timing.

### 3. Step Function Modifications

Each of the three step types needs similar modifications. Using DIRK as the example:

**File**: `src/cubie/integrators/algorithms/generic_dirk.py`

**In `build_step()` method**:

```python
# Capture compile-time flag
has_interpolant = self.tableau.has_interpolant

# Inside the device function @cuda.jit step(...):
# Retrieve next_save from shared memory buffer
next_save = shared_next_save_buffer[0]

# Always compute these (executed on every step to avoid warp divergence)
needs_interp = do_save and has_interpolant and (next_save >= current_time) and (next_save <= current_time + dt_value)
theta = (next_save - current_time) / dt_value

# Evaluate interpolant using b_interp coefficients and stage values
# This is always computed but only conditionally committed
for idx in range(n):
    y_interp = evaluate_interpolant(theta, state[idx], stage_rhs, b_interp_coeffs)
    # Conditional commit: use interpolated value if needed, else use step-end value
    proposed_state[idx] = selp(needs_interp, y_interp, proposed_state[idx])

# Update time with selp - use next_save if interpolating, else step-end time
t_proposed = selp(needs_interp, next_save, current_time + dt_value)

# Compute observables at the appropriate time
# Use selped time and drivers buffer
t_obs = selp(needs_interp, next_save, current_time + dt_value)
observables_function(
    proposed_state,
    parameters,
    drivers_buffer,  # Evaluated at appropriate time
    proposed_observables,
    t_obs,
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
   - Interpolant: Coefficients available in literature
   - Source: Hairer & Wanner (1996) or similar
   - Complexity: **Low** (coefficients can be read from literature)

2. **LOBATTO_IIIC_3_TABLEAU**
   - Method: 4th order, 3 stages
   - Interpolant: Coefficients available in Hairer & Wanner (1996)
   - Complexity: **Low** (coefficients available in literature)

3. **Other DIRK tableaus**: Only implement if coefficients exist in literature or are part of the `a` matrix

#### 4.2 FIRK Tableaus

**File**: `src/cubie/integrators/algorithms/generic_firk_tableaus.py`

**Tableaus to Update**:

1. **GAUSS_LEGENDRE_2_TABLEAU**
   - Method: 4th order, 2 stages
   - Interpolant: Coefficients available in literature
   - Source: Hairer & Wanner (1996)
   - Complexity: **Low** (coefficients available in literature)

2. **RADAU_IIA_5_TABLEAU**
   - Method: 5th order, 3 stages
   - Interpolant: Coefficients available in Hairer & Wanner
   - Complexity: **Low** (coefficients available in literature)

#### 4.3 Rosenbrock Tableaus

**File**: `src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py`

**Tableaus to Update**:

1. **ROS3P_TABLEAU**
   - Method: 3rd order, 3 stages
   - Interpolant: Coefficients may be documented in Lang & Verwer (2001) or related literature
   - Complexity: **Low** if coefficients are available in literature

**Note**: For all tableaus, interpolants will only be added if coefficients are available in published literature or can be derived from the existing `a` matrix. No original derivation of interpolant coefficients will be performed.

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

### Phase 2 (Implementation)

- [ ] Look up `b_interp` coefficients from literature for available tableaus
- [ ] Implement interpolation in step functions (DIRK, FIRK, Rosenbrock)
- [ ] Modify loop to pass `next_save` via shared memory
- [ ] Add compile-time toggle based on `has_interpolant`
- [ ] Write unit and integration tests

### Phase 3 (Validation)

- [ ] Verify coefficients match literature
- [ ] Test against reference implementations (scipy, Julia)
- [ ] Validate on stiff test problems
- [ ] Document which tableaus have interpolants available

## Conclusion

Implementing dense output via interpolants is **feasible** and has **low implementation complexity** when:

1. Interpolant coefficients are available in published literature (Hairer & Wanner, Lang & Verwer, etc.)
2. No additional buffers are required - reuse existing `proposed_state` and `proposed_observables`
3. Warp divergence is avoided by executing interpolation on every step with conditional commits using `selp`

**Implementation approach**:
- Pass `next_save` via shared memory buffer
- Execute interpolation logic on every step (no branching)
- Use `selp` for conditional commits to avoid warp divergence
- Reuse existing observables function with `selp`ed time and drivers

**Key constraint**: Only implement interpolants where coefficients exist in literature or can be derived from the existing `a` matrix. No original derivation will be performed.

The main **benefit** is eliminating error estimate inflation, which should improve step controller efficiency and allow larger time steps on average.

**Recommendation**: Start with a proof-of-concept for TRAPEZOIDAL_DIRK_TABLEAU (simplest case) before tackling more complex methods.
