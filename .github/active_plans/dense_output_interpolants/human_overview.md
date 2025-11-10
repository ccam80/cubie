# Dense Output Interpolants Implementation

## User Stories

### Story 1: Accurate Error Estimates at Save Points
**As a** scientist using CuBIE for adaptive ODE integration  
**I want** error estimates that accurately reflect the actual step taken  
**So that** the step controller makes informed decisions about step size adjustments

**Acceptance Criteria:**
- Error estimates are computed for the actual step size used (not inflated by truncation)
- Step controller receives commensurate error estimates when steps are truncated to save points
- Error inflation of ~10-100% is eliminated
- Step acceptance rate improves for problems with frequent save points

**Success Metrics:**
- Error estimate inflation reduced from ~10-100% to <1%
- Step rejection rate decreased on test problems with dense save points
- Average step size increases on stiff problems with adaptive stepping

### Story 2: Efficient Integration with Dense Output
**As a** user integrating stiff ODEs with specified save points  
**I want** dense output via interpolation instead of step truncation  
**So that** integration is more efficient and accurate

**Acceptance Criteria:**
- Interpolation is used when tableaus have dense output coefficients available
- Interpolated values at save points match the solution at those points
- Implementation avoids warp divergence on GPU
- No additional device memory buffers required
- Works seamlessly with existing save point logic

**Success Metrics:**
- Interpolation overhead is <10% of step computation time
- GPU warp efficiency remains high (no branching divergence)
- Memory footprint unchanged from current implementation

### Story 3: Literature-Based Interpolant Coefficients
**As a** maintainer of CuBIE's numerical methods  
**I want** dense output coefficients sourced from published literature  
**So that** the implementation is trustworthy and verifiable

**Acceptance Criteria:**
- All interpolant coefficients come from published sources (Hairer & Wanner, Lang & Verwer)
- Each tableau's `b_interp` coefficients are documented with literature references
- No original coefficient derivation is performed
- Interpolants match expected order of accuracy from literature

**Success Metrics:**
- All added `b_interp` coefficients have literature citations
- Interpolant accuracy validated against reference implementations (SciPy, Julia DifferentialEquations.jl)

## Executive Summary

This feature implements dense output (continuous extension) for CuBIE's Runge-Kutta integrators to address error estimate inflation when steps are truncated to hit save points. The current implementation truncates steps using `dt_eff = selp(do_save, next_save - t, dt[0])`, but the step controller receives error estimates as if full steps were taken, inflating errors by ~10-100%.

The solution uses interpolation within completed steps to compute state and observables at save points, allowing the step controller to receive accurate error estimates. This trades minimal duplicate computation (re-evaluating observables) for improved step controller efficiency.

**Key Design Decision:** Execute interpolation logic on every step with conditional commit using `selp()` to avoid warp divergence, rather than branching only when save points occur.

## Architecture Overview

### Current Save Point Behavior (Truncation)

```
Timeline:  |-------- full step --------|
           t                            t + dt[0]
                     ↓ next_save
           |-- dt_eff --|
           
Step taken: dt_eff (truncated)
Error used: error estimate from dt_eff step (INFLATED relative to dt[0])
```

### New Behavior (Interpolation)

```
Timeline:  |-------- full step --------|
           t                            t + dt[0]
                     ↓ next_save
                     θ = (next_save - t) / dt[0]
           
Step taken: dt[0] (full step)
At save:    interpolate to next_save using θ
Error used: error estimate from dt[0] step (ACCURATE)
```

### Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ ODE Loop (ode_loop.py)                                      │
│                                                               │
│  ┌─────────────────┐                                        │
│  │ Shared Memory   │                                        │
│  │ Buffer:         │                                        │
│  │ next_save       │ ← Written by loop coordinator         │
│  └────────┬────────┘                                        │
│           │ Read by step functions                          │
│           ↓                                                  │
│  ┌──────────────────────────────────────────┐               │
│  │ Adaptive Save Logic                      │               │
│  │ - Compute do_save flag                   │               │
│  │ - Write next_save to shared memory       │               │
│  └──────────────────────────────────────────┘               │
│           │                                                  │
│           ↓                                                  │
│  ┌──────────────────────────────────────────┐               │
│  │ Call Step Function                       │               │
│  │ - Pass dt[0] (NOT truncated)             │               │
│  │ - Step reads next_save from shared mem   │               │
│  └──────────────────────────────────────────┘               │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────┐
│ Step Function (generic_dirk.py, etc.)                       │
│                                                               │
│  ┌──────────────────────────────────────────┐               │
│  │ Read Compile-Time Flags                  │               │
│  │ has_interpolant = tableau.has_interpolant│               │
│  └──────────────────────────────────────────┘               │
│           │                                                  │
│           ↓                                                  │
│  ┌──────────────────────────────────────────┐               │
│  │ Compute Full Step                        │               │
│  │ - Take step with dt (not truncated)      │               │
│  │ - Store stage derivatives k_i            │               │
│  │ - Compute proposed_state, error          │               │
│  └──────────────────────────────────────────┘               │
│           │                                                  │
│           ↓                                                  │
│  ┌──────────────────────────────────────────┐               │
│  │ Interpolation (Always Computed)          │               │
│  │ needs_interp = do_save AND               │               │
│  │                has_interpolant AND        │               │
│  │                (next_save in [t, t+dt])  │               │
│  │                                           │               │
│  │ theta = (next_save - t) / dt             │               │
│  │                                           │               │
│  │ FOR each state variable i:               │               │
│  │   y_interp[i] = evaluate_interpolant(    │               │
│  │     theta, state[i], stage_k[], b_interp)│               │
│  │                                           │               │
│  │ Conditional Commit (NO BRANCHING):       │               │
│  │   proposed_state[i] = selp(              │               │
│  │     needs_interp,                         │               │
│  │     y_interp[i],                          │               │
│  │     proposed_state[i]                     │               │
│  │   )                                       │               │
│  └──────────────────────────────────────────┘               │
│           │                                                  │
│           ↓                                                  │
│  ┌──────────────────────────────────────────┐               │
│  │ Observables Computation                  │               │
│  │ t_obs = selp(needs_interp, next_save,    │               │
│  │              t + dt)                      │               │
│  │                                           │               │
│  │ Call observables_function(               │               │
│  │   proposed_state,                         │               │
│  │   parameters,                             │               │
│  │   drivers_at_t_obs,  ← evaluated at t_obs│               │
│  │   proposed_observables,                   │               │
│  │   t_obs                                   │               │
│  │ )                                         │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow: Interpolant Evaluation

```
Input:
  - state[n]: State at start of step (t)
  - stage_k[s][n]: Stage derivatives from RK step
  - theta: Interpolation parameter ∈ [0, 1]
  - b_interp[p][s]: Interpolant coefficient matrix

Computation:
  FOR i in range(n):  # Each state variable
    y_interp[i] = state[i]
    
    FOR stage_idx in range(s):  # Each stage
      weight = 0.0
      theta_power = 1.0
      
      FOR poly_power in range(p):  # Polynomial terms
        weight += b_interp[poly_power][stage_idx] * theta_power
        theta_power *= theta
      
      y_interp[i] += dt * weight * stage_k[stage_idx][i]

Output:
  - y_interp[n]: Interpolated state at t + theta*dt
```

### Memory Layout (Reusing Existing Buffers)

```
Existing Buffers (UNCHANGED):
┌────────────────────┐
│ state_buffer       │ ← Current state y(t)
├────────────────────┤
│ proposed_state     │ ← Step result OR interpolated result (selp commit)
├────────────────────┤
│ proposed_observ... │ ← Observables OR interpolated observables (selp commit)
├────────────────────┤
│ stage_k[s][n]      │ ← Stage derivatives (already used by RK methods)
├────────────────────┤
│ error[n]           │ ← Error estimate (from full step)
└────────────────────┘

Shared Memory Addition:
┌────────────────────┐
│ next_save (scalar) │ ← Single precision value, written by loop
└────────────────────┘

No new device buffers required!
```

## Key Technical Decisions

### Decision 1: Predicated Execution Over Branching
**Rationale:** GPU warps execute in lockstep. Branching (`if/else`) causes divergence when threads take different paths, serializing execution. By computing interpolation on every step and using `selp()` for conditional commits, all threads execute the same code path, maintaining warp efficiency.

**Trade-off:** Slight computational overhead (~5-10%) when interpolation is not needed, but eliminates divergence penalty which can be 2x-10x worse.

### Decision 2: Reuse Existing Buffers
**Rationale:** GPU memory is constrained. The current implementation already has `proposed_state` and `proposed_observables` buffers. Using `selp()` for conditional commit allows reusing these buffers without allocation.

**Trade-off:** None - this is strictly better than allocating new buffers.

### Decision 3: Pass next_save via Shared Memory
**Rationale:** Avoids modifying step function signatures, which would require changes across all algorithm implementations and testing infrastructure.

**Trade-off:** Requires one shared memory slot (negligible - shared memory is 48KB, this uses 4-8 bytes).

### Decision 4: Literature-Only Coefficients
**Rationale:** Dense output coefficient derivation is complex and error-prone. Using published coefficients ensures correctness and makes the implementation verifiable.

**Trade-off:** Limited to tableaus with published interpolants (but covers most common methods: Gauss-Legendre, Radau IIA, Lobatto IIIC, ROS3P).

## Research Findings

### Available Interpolants from Literature

**Source: Hairer & Wanner (1996), "Solving ODEs II"**
- Gauss-Legendre methods (FIRK): 2-stage (4th order), higher stages available
- Radau IIA methods (FIRK): 3-stage (5th order), higher stages available
- Lobatto IIIC methods (DIRK): 3-stage (4th order), higher stages available
- Coefficients in Chapter IV, Section 6 (Dense Output)

**Source: Lang & Verwer (2001), "ROS3P Paper"**
- ROS3P (Rosenbrock): 3rd order, 3 stages
- May contain dense output formula (requires verification)

**Source: Shampine (1985), SIAM J. Numerical Analysis**
- General theory of continuous extension
- Hermite interpolation approach for methods with c=[0, 1]

### Hermite Interpolation Pattern (For Trapezoidal/Crank-Nicolson)

For methods with stages at `c=[0, 1]`, cubic Hermite interpolation is standard:

```
Basis functions:
  H₀(θ) = (1-θ)²(1+2θ)    # Value at start
  H₁(θ) = θ²(3-2θ)         # Value at end
  H₀'(θ) = θ(1-θ)²         # Derivative at start
  H₁'(θ) = θ²(θ-1)         # Derivative at end

Interpolant:
  y(θ) = H₀(θ)·y(t) + H₁(θ)·y(t+dt) + dt·H₀'(θ)·k₀ + dt·H₁'(θ)·k₁

Since y(t+dt) = y(t) + dt·(b₀·k₀ + b₁·k₁), this can be rewritten as:
  y(θ) = y(t) + dt·[b_interp₀(θ)·k₀ + b_interp₁(θ)·k₁]

Where b_interp coefficients are polynomials in θ.
```

## Expected Impact on Architecture

### Modified Components
1. **ode_loop.py**: Add shared memory buffer for `next_save`, populate on each iteration
2. **generic_dirk.py**: Add interpolation logic with conditional commit
3. **generic_firk.py**: Add interpolation logic with conditional commit
4. **generic_rosenbrock_w.py**: Add interpolation logic with conditional commit
5. **Tableau files**: Add `b_interp` coefficients from literature

### Unchanged Components
- Step controller logic (receives accurate errors, no changes needed)
- Memory management (no new buffers)
- Output functions (save logic unchanged)
- Algorithm signatures (next_save via shared memory, not parameters)

### Integration Points
- Loop writes `next_save` to shared memory before calling step function
- Step function reads from shared memory when `has_interpolant=True`
- Existing `selp()` pattern extended to state/observable commits
- Error estimates from full steps, not truncated steps

## Alternatives Considered

### Alternative 1: Dedicated Interpolation Buffers
**Rejected:** Would require additional device memory allocation, increasing memory pressure. The `selp()` conditional commit pattern allows reusing existing buffers with zero overhead.

### Alternative 2: Branching on do_save
**Rejected:** Would cause warp divergence when threads have different save schedules. Predicated execution with `selp()` maintains warp lockstep at cost of ~5-10% overhead when interpolation not needed.

### Alternative 3: Derive Interpolant Coefficients
**Rejected:** Complex, error-prone, and unnecessary. Published literature provides coefficients for all commonly-used tableaus. Focus on implementation, not research.

### Alternative 4: Add next_save to Step Function Signature
**Rejected:** Would require modifying all step function signatures, extensive testing changes, and complicate the interface. Shared memory approach is cleaner and isolated.

## Performance Expectations

### Computational Overhead
- Interpolation computation: ~5-10% of step cost (polynomial evaluation is cheap)
- Executed on every step (predicated execution), but only committed when needed
- Observables re-evaluation at save points: negligible (typically cheap functions)

### Memory Overhead
- Shared memory: +1 scalar (4-8 bytes) per block
- Device memory: 0 bytes (reuses existing buffers)

### Performance Gains
- Reduced step rejections from accurate error estimates
- Larger average step sizes on stiff problems
- Expected 10-30% speedup on problems with dense save points

### Validation Metrics
- Error estimate inflation: <1% (down from 10-100%)
- Step acceptance rate: +15-25% on test problems
- Warp efficiency: maintained at >95% (verified via profiling)

## References

1. Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.). Springer. Chapter IV, Section 6.
2. Lang, J., & Verwer, J. (2001). "ROS3P—An Accurate Third-Order Rosenbrock Solver Designed for Parabolic Problems." *BIT Numerical Mathematics* 41, 731–738.
3. Shampine, L. F. (1985). "Interpolation for Runge-Kutta Methods." *SIAM Journal on Numerical Analysis*, 22(5), 1014-1027.
4. Current implementation: `INTERPOLANTS_INVESTIGATION.md`
5. Proof-of-concept: `tests/integrators/algorithms/test_interpolant_concept.py`
