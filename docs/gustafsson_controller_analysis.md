# Mathematical Analysis of Gustafsson Controller Implementations

## Executive Summary

This document provides an in-depth mathematical analysis of three step-size control implementations:

1. **CuBIE's Gustafsson Controller** (`gustafsson_controller.py`)
2. **OrdinaryDiffEq.jl's PredictiveController** 
3. **Textbook Gustafsson Controller** (Hairer & Wanner literature)

The analysis focuses on the mathematical formulations, key differences, and effects on step size adaptation.

---

## 1. CuBIE's Gustafsson Controller

### Mathematical Formulation

The CuBIE implementation (lines 224-296 in `gustafsson_controller.py`) computes the next step size using:

```
Step size gain calculation:
1. Compute normalized error norm:
   nrm2 = (1/n) * Σ[(error_i / (atol_i + rtol_i * max(|state_i|, |state_prev_i|)))²]

2. Accept step if: nrm2 ≤ 1.0

3. Compute damping factor:
   denom = niters + 2 * max_newton_iters
   tmp = gain_numerator / denom
   fac = min(gamma, tmp)
   where gain_numerator = (1 + 2 * max_newton_iters) * gamma

4. Compute basic gain (I-controller style):
   expo = 1 / (2 * (algorithm_order + 1))
   gain_basic = fac * (nrm2^(-expo))

5. Compute Gustafsson predictive gain:
   ratio = (nrm2 * nrm2) / err_prev
   gain_gus = safety * (dt_current / dt_prev) * (ratio^(-expo)) * gamma

6. Select minimum and apply fallback:
   gain = min(gain_gus, gain_basic)
   if (!accept OR dt_prev < 1e-16):
       gain = gain_basic

7. Apply limits and deadband:
   gain = clamp(gain, min_gain, max_gain)
   if deadband_enabled and gain ∈ [deadband_min, deadband_max]:
       gain = 1.0

8. Compute new step size:
   dt_new = dt_current * gain
   dt_new = clamp(dt_new, dt_min, dt_max)
```

### Key Mathematical Expressions

**Error norm (Hairer norm):**
```
nrm2 = (1/n) * Σᵢ[(εᵢ / (Atolᵢ + Rtolᵢ * max(|yᵢ|, |yᵢ₋₁|)))²]
```

**Exponent:**
```
expo = 1 / (2 * (p + 1))
```
where `p` is the algorithm order.

**Adaptive damping factor:**
```
fac = min(γ, ((1 + 2M)γ) / (k + 2M))
```
where:
- `γ` = gamma (default 0.9)
- `M` = max_newton_iters (default 20 for implicit methods, 0 for explicit)
- `k` = actual Newton iterations in current step

**Basic gain (I-controller):**
```
gain_basic = fac * (nrm2)^(-expo)
```

**Gustafsson predictive gain:**
```
gain_gus = safety * (hₙ / hₙ₋₁) * ((nrm2² / nrm2_prev)^(-expo)) * γ
```

**Fallback logic:**
```
gain = {
  gain_gus           if gain_gus < gain_basic AND step_accepted AND dt_prev > 1e-16
  gain_basic         otherwise
}
```

### Parameters

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `gamma` | 0.9 | (0, 1) | Safety factor damping |
| `max_newton_iters` | 20 (implicit), 0 (explicit) | ≥ 0 | Expected Newton iterations for implicit solvers |
| `safety` | 0.9 | (0, 1) | Inherited from base class |
| `min_gain` | 0.2 | > 0 | Lower bound on step size change |
| `max_gain` | 5.0 | > min_gain | Upper bound on step size change |
| `deadband_min` | 1.0 | - | Lower deadband threshold |
| `deadband_max` | 1.2 | > deadband_min | Upper deadband threshold |

### Unique Features

1. **Adaptive damping based on Newton iterations**: The factor `fac` depends on the actual number of Newton iterations `niters` from the implicit solver, becoming more conservative when convergence is slower.

2. **Minimum selection strategy**: Always chooses `min(gain_gus, gain_basic)`, providing a conservative bound on step size increases.

3. **Predictive term uses squared error ratio**: The ratio is `(nrm2² / nrm2_prev)` rather than `(nrm2 / nrm2_prev)`, which is more sensitive to error changes.

4. **Fallback to basic gain on rejection or startup**: Uses simpler I-controller gain when step rejected or no previous history exists.

---

## 2. OrdinaryDiffEq.jl PredictiveController

### Mathematical Formulation

Based on the actual OrdinaryDiffEq.jl source code:

```
Step size computation (stepsize_controller!):
1. Compute normalized error estimate EEst

2. Compute adaptive damping factor:
   if fac_default_gamma(alg):
       fac = gamma
   else:
       fac = min(gamma, (1 + 2*maxiters)*gamma / (iter + 2*maxiters))
   
3. Compute basic step size ratio:
   expo = 1 / (adaptive_order + 1)
   qtmp = EEst^expo / fac
   q = clamp(qtmp, 1/qmax, 1/qmin)
   
4. Store q in integrator.qold

Step acceptance (step_accept_controller!):
5. If success_iter > 0 (not first successful step):
   expo = 1 / (adaptive_order + 1)
   qgus = (dtacc / dt) * ((EEst² / erracc)^expo)
   qgus = clamp(qgus / gamma, 1/qmax, 1/qmin)
   qacc = max(q, qgus)
   else:
   qacc = q

6. Apply deadband (qsteady_min to qsteady_max):
   if qsteady_min ≤ qacc ≤ qsteady_max:
       qacc = 1.0

7. Update history and compute new dt:
   dtacc = dt
   erracc = max(1e-2, EEst)
   dt_new = dt / qacc

Step rejection (step_reject_controller!):
8. If success_iter == 0 (first rejection):
   dt_new = 0.1 * dt
   else:
   dt_new = dt / qold
```

### Key Mathematical Expressions

**Adaptive damping factor (identical to CuBIE):**
```
fac = min(γ, ((1 + 2M)γ) / (k + 2M))
```
where:
- `γ` = gamma (safety factor)
- `M` = maxiters (max Newton/FIRK iterations)
- `k` = iter (actual iterations in current step)

**Basic step size ratio:**
```
q = clamp(EEst^(1/(p+1)) / fac, 1/qmax, 1/qmin)
```

**Gustafsson predictive ratio (on acceptance with history):**
```
qgus = (dtacc / dt) * ((EEst² / erracc)^(1/(p+1)))
qgus = clamp(qgus / gamma, 1/qmax, 1/qmin)
qacc = max(q, qgus)
```

**Note on error representation:** OrdinaryDiffEq.jl's `EEst` is a normalized error that is already dimensionless (like a squared norm without square root), similar to CuBIE's `nrm2`.

**Exponent:**
```
expo = 1 / (p + 1)
```
where `p` is the adaptive order of the method.

**Rejection logic:**
- First rejection (`success_iter == 0`): aggressive 10× reduction (`dt * 0.1`)
- Subsequent rejections: divide by previous ratio (`dt / qold`)

### Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `gamma` | 0.9 | Safety factor |
| `qmin` | 0.2 (1/5) | Minimum inverse step size ratio |
| `qmax` | 5.0 | Maximum inverse step size ratio |
| `qsteady_min` | 1.0 | Lower deadband threshold |
| `qsteady_max` | 1.2 | Upper deadband threshold |
| `maxiters` | Varies | Maximum iterations for implicit solver |

### Unique Features

1. **Ratio-based formulation**: Uses `q = 1/(step_size_multiplier)`, so smaller `q` means larger next step. This is opposite to the typical gain convention.

2. **Maximum selection for Gustafsson**: Uses `qacc = max(q, qgus)`, choosing the **larger** of the two ratios, which corresponds to the **smaller** step size increase (more conservative).

3. **Squared error ratio with NO additional squaring**: `(EEst² / erracc)^expo` where `EEst` is already like error². This is mathematically: `(ε² / ε²_prev)^(1/(p+1))`.

4. **Explicit history tracking**: Stores `dtacc` (previous accepted dt) and `erracc` (previous accepted error) separately from current step values.

5. **Only applies Gustafsson on successful steps**: The predictive component only activates after at least one successful step (`success_iter > 0`).

---

## 3. Textbook Gustafsson Controller (Hairer & Wanner)

### Mathematical Formulation

The classical H211b controller from Gustafsson, analyzed by Hairer and Wanner:

```
Step size update:
hₙ₊₁ = (tol/rₙ)^(1/(bk)) * (tol/rₙ₋₁)^(1/(bk)) * (hₙ/hₙ₋₁)^(-1/b) * hₙ

Equivalently, using error ratios:
hₙ₊₁ = hₙ * εₙ^(1/(bk)) * εₙ₋₁^(1/(bk)) * ρₙ^(-1/b)

where:
- εₙ = tol/rₙ (inverse normalized error ratio)
- ρₙ = hₙ/hₙ₋₁ (step size ratio)
- b = bandwidth parameter (typically 2, 4, or 6)
- k = p + 1 (one plus method order)
```

### Key Mathematical Expressions

**Full H211b formula:**
```
ρₙ₊₁ = εₙ^(1/(bk)) * εₙ₋₁^(1/(bk)) * ρₙ^(-1/b)
hₙ₊₁ = ρₙ₊₁ * hₙ
```

**Alternative formulation (PI-style):**
```
hₙ₊₁ = hₙ * γ * εₙ^(β₁) * εₙ₋₁^(β₂)

where:
β₁ = 0.4 or 2/p (proportional gain)
β₂ = -0.2 or -1/p (integral gain)
```

### Parameters

| Parameter | Typical Value | Purpose |
|-----------|--------------|---------|
| `b` | 2, 4, or 6 | Digital filter bandwidth |
| `gamma` | 0.9 | Safety factor |
| `qmin` | 0.1 | Minimum step size ratio |
| `qmax` | 5.0 | Maximum step size ratio |
| `β₁` (PI variant) | 0.4 or 2/p | Proportional gain |
| `β₂` (PI variant) | -0.2 or -1/p | Integral gain |

### Unique Features

1. **Digital filter design**: Explicitly designed using frequency-domain analysis to attenuate high-frequency oscillations.

2. **Symmetry in error terms**: Both current and previous errors appear with the same exponent `1/(bk)`, creating symmetric filtering.

3. **Explicit step ratio damping**: The term `ρₙ^(-1/b)` provides negative feedback on step size changes, smoothing the sequence.

4. **Tunable bandwidth**: Parameter `b` allows trading responsiveness (small `b`) for smoothness (large `b`).

---

## Comparison of the Three Implementations

### Side-by-Side Formula Comparison

| Controller | Step Size Formula | Error Terms | History Terms |
|------------|------------------|-------------|---------------|
| **CuBIE Gustafsson** | `h_{n+1} = h_n * min(gain_gus, gain_basic)` | `nrm2_n`, `nrm2_{n-1}` | `h_{n-1}`, `niters` |
| **OrdinaryDiffEq.jl** | `h_{n+1} = h_n / max(q, qgus)` | `EEst_n`, `EEst_{n-1}` | `dt_{n-1}`, `iter` |
| **Textbook H211b** | `h_{n+1} = h_n * ε_n^(1/(bk)) * ε_{n-1}^(1/(bk)) * ρ_n^(-1/b)` | `ε_n`, `ε_{n-1}` | `ρ_n = h_n/h_{n-1}` |

### Critical Mathematical Differences Between CuBIE and OrdinaryDiffEq.jl

Both implementations are based on the same Gustafsson algorithm but have **subtle yet significant** mathematical differences:

#### 1. Error Representation and Exponent

**CuBIE:**
- Error norm: `nrm2 = (1/n) * Σ[(error_i / tol_i)²]` - **squared** without square root
- Exponent: `expo = 1/(2(p+1))` - the factor of **1/2** compensates for squared norm
- For p=5: `expo = 1/12 ≈ 0.083`

**OrdinaryDiffEq.jl:**
- Error estimate: `EEst` - normalized error (implementation-dependent, typically also squared)
- Exponent: `expo = 1/(p+1)` - **no factor of 1/2**
- For p=5: `expo = 1/6 ≈ 0.167`

**Implication:** If both use squared error norms, OrdinaryDiffEq.jl's larger exponent makes it respond **more aggressively** to error changes.

#### 2. Gustafsson Predictive Term - THE KEY DIFFERENCE

**CuBIE:**
```
ratio = nrm2 * nrm2 / err_prev    # This is ε⁴ / ε²_prev
gain_gus = safety * (h_n/h_{n-1}) * (ratio)^(-expo) * gamma
         = safety * (h_n/h_{n-1}) * (ε⁴/ε²_prev)^(-1/(2(p+1))) * gamma
         = safety * (h_n/h_{n-1}) * ε^(-2/(p+1)) * ε_prev^(1/(p+1)) * gamma
```

**OrdinaryDiffEq.jl:**
```
qgus = (dtacc/dt) * ((EEst²/erracc)^expo)
     = (h_{n-1}/h_n) * ((ε²/ε²_prev)^(1/(p+1)))
     = (h_{n-1}/h_n) * ε^(-2/(p+1)) * ε_prev^(2/(p+1))
```

**Critical Difference:**
- **CuBIE**: Previous error has exponent `+1/(p+1)`
- **OrdinaryDiffEq.jl**: Previous error has exponent `+2/(p+1)` (twice as much weight!)

This means OrdinaryDiffEq.jl gives **double the weight** to the previous error in the predictive term.

#### 3. Adaptive Damping - IDENTICAL!

**Both implementations use the same formula:**
```
fac = min(γ, ((1 + 2M)γ) / (k + 2M))
```
where:
- `γ` = gamma (typically 0.9)
- `M` = max iterations (maxiters)
- `k` = actual iterations (niters or iter)

This makes both controllers more conservative when implicit solves converge slowly.

#### 4. Gain Selection Strategy - OPPOSITE CONVENTIONS!

**CuBIE:**
```
gain = min(gain_gus, gain_basic)   # Choose smaller gain = smaller step increase
```
Uses "gain" convention where `h_new = h_old * gain` and smaller gain = more conservative.

**OrdinaryDiffEq.jl:**
```
qacc = max(q, qgus)                # Choose larger q = smaller step increase
```
Uses "ratio" convention where `h_new = h_old / q` and larger q = more conservative.

**Mathematically equivalent strategies** - both choose the more conservative option!

#### 5. When Predictive Term is Applied

**CuBIE:**
- Applied on every step (if step accepted and history exists)
- Falls back to `gain_basic` on rejection or no history

**OrdinaryDiffEq.jl:**
- Applied only after at least one successful step (`success_iter > 0`)
- Falls back to `q` on first successful step

#### 6. Error Ratio in Predictive Term - THE SQUARED DIFFERENCE

**CuBIE explicitly cubes the current error:**
```
ratio = nrm2 * nrm2 / err_prev     # ε⁴ / ε²_prev
```

**OrdinaryDiffEq.jl squares the error ratio:**
```
(EEst² / erracc)                    # (ε²)² / ε²_prev = ε⁴ / ε²_prev
```

**These are mathematically IDENTICAL** if both use squared norms! The difference in exponents (1/(2(p+1)) vs 1/(p+1)) is what matters.

### Summary: CuBIE vs OrdinaryDiffEq.jl Key Differences

The implementations are **very similar** in structure but differ in critical details:

| Aspect | CuBIE | OrdinaryDiffEq.jl | Impact |
|--------|-------|------------------|---------|
| **Error exponent** | `1/(2(p+1))` | `1/(p+1)` | ODE.jl responds 2× faster to error changes |
| **Previous error weight** | `ε_prev^(1/(p+1))` | `ε_prev^(2/(p+1))` | ODE.jl gives 2× weight to history |
| **Basic gain computation** | `fac * (ε²)^(-1/(2(p+1)))` | `(ε²)^(-1/(p+1)) / fac` | Mathematically different but same purpose |
| **Selection strategy** | `min(gain_gus, gain_basic)` | `max(q, qgus)` | Both choose conservative option |
| **When predictive applied** | Every accepted step | After first success | CuBIE more consistent |
| **Adaptive damping** | **IDENTICAL** | **IDENTICAL** | Both use same `fac` formula |
| **Squared error handling** | **IDENTICAL** | **IDENTICAL** | Both use `ε⁴/ε²_prev` |

### Mathematical Implications

1. **Exponent difference:** OrdinaryDiffEq.jl's `1/(p+1)` exponent (vs CuBIE's `1/(2(p+1))`) makes it respond **twice as aggressively** to error changes. For p=5:
   - CuBIE: `ε^(±1/12) ≈ ε^(±0.083)`
   - ODE.jl: `ε^(±1/6) ≈ ε^(±0.167)`
   - If error doubles (ε=2), CuBIE changes step by factor `2^(1/12) ≈ 1.06` (6%), ODE.jl by `2^(1/6) ≈ 1.12` (12%)

2. **Previous error weight:** OrdinaryDiffEq.jl's `ε_prev^(2/(p+1))` gives twice the exponent to historical error compared to CuBIE's `ε_prev^(1/(p+1))`. This makes ODE.jl's predictor **more heavily influenced by past behavior**.

3. **Combined effect on Gustafsson gain:** For error decreasing from ε_prev=2.0 to ε=1.5, with p=5:
   - CuBIE: `gain_gus ∝ (1.5)^(-2/6) * (2.0)^(1/6) ≈ 0.817 * 1.122 ≈ 0.917`
   - ODE.jl: `qgus ∝ (1.5)^(-2/6) * (2.0)^(2/6) ≈ 0.817 * 1.260 ≈ 1.030`
   - ODE.jl gives **13% more weight** to the improving trend

### Historical Information Comparison

| Implementation | Tracks | Uses For |
|----------------|--------|----------|
| **CuBIE** | `err_prev`, `dt_prev`, `niters` | Predictive gain, adaptive damping |
| **OrdinaryDiffEq.jl** | `erracc`, `dtacc`, `qold`, `iter`, `success_iter` | Predictive gain, adaptive damping, rejection logic |
| **Textbook H211b** | `ε_{n-1}`, `ρ_n` | Symmetric filtering |

Both CuBIE and OrdinaryDiffEq.jl track similar information but use it differently in their predictive formulas.

---

## Effects on Output Step Size

### Step Size Adaptation Behavior

#### Smooth Problems (slowly varying error)

**CuBIE Gustafsson:**
- Smaller exponent `1/(2(p+1))` produces gentler step size changes
- Previous error weight `ε_prev^(1/(p+1))` provides moderate historical influence
- `min(gain_gus, gain_basic)` always chooses more conservative option
- Adaptive damping prevents aggressive stepping when Newton convergence is slow
- **Result**: Smooth, very conservative step size sequence, emphasis on reliability

**OrdinaryDiffEq.jl Predictive:**
- Larger exponent `1/(p+1)` produces more aggressive step size changes (2× faster response)
- Previous error weight `ε_prev^(2/(p+1))` provides stronger historical influence (2× weight)
- `max(q, qgus)` chooses more conservative of two ratios (mathematically equivalent to CuBIE's min)
- After first success, predictive term helps adapt to trends
- **Result**: More aggressive adaptation to changing error behavior, faster response to improving conditions

**Textbook H211b:**
- Symmetric error terms `ε_n^(1/(bk)) * ε_{n-1}^(1/(bk))` provide balanced filtering
- Step ratio damping `ρ_n^(-1/b)` explicitly smooths oscillations
- Tunable bandwidth parameter `b` allows customization
- **Result**: Smoothest step size sequence, minimal oscillations, particularly with `b=4` or `b=6`

#### Stiff Problems (rapidly changing error)

**CuBIE Gustafsson:**
- Adaptive damping responds to convergence difficulty (increasing `niters` → smaller `fac`)
- Fallback to basic gain on rejection provides stable recovery
- Minimum selection prevents overshoot
- **Result**: Stable behavior on stiff problems, conservative in difficult regions

**OrdinaryDiffEq.jl Predictive:**
- 10× reduction on first rejection provides aggressive response to stiffness
- Division by `q_old` on subsequent rejections provides gentler adaptation
- Less conservative than CuBIE in accepted steps
- **Result**: Faster adaptation to changing stiffness, may require more rejected steps initially

**Textbook H211b:**
- Explicit frequency-domain design attenuates high-frequency error variations
- No direct coupling to solver convergence (no `niters` dependence)
- Strong damping from `ρ_n^(-1/b)` term stabilizes step size
- **Result**: Excellent for stiff problems with appropriate `b` choice, minimal step size oscillation

#### Problems with Localized Difficulty

**CuBIE Gustafsson:**
- Quickly reduces step size when entering difficult region (both gains respond to error increase)
- Conservative exit from difficult region (minimum selection)
- Gradual step size recovery as errors decrease
- **Result**: Safe passage through difficult regions, potentially slow recovery

**OrdinaryDiffEq.jl Predictive:**
- 10× reduction on first rejected step entering difficult region
- Rapid step size increase when exiting (division by `q_old` removes previous conservatism)
- More aggressive in smooth regions following difficulty
- **Result**: Fast adaptation entering and exiting difficult regions

**Textbook H211b:**
- Filtered response reduces overreaction to temporary spikes
- Symmetric error treatment provides balanced entry/exit behavior
- Recovery rate depends on bandwidth parameter `b`
- **Result**: Smooth transition through difficulty, rate controlled by `b`

### Quantitative Comparison

For a typical 5th-order method (p=5) with normalized **squared** error `ε² = 0.5` (error below tolerance):

**CuBIE (assuming accepted step, niters=3, M=20, γ=0.9, safety=0.9):**
```
expo = 1/(2*6) = 1/12 ≈ 0.083
fac = min(0.9, (1 + 40)*0.9 / (3 + 40)) ≈ min(0.9, 0.858) = 0.858

Basic gain:
nrm2 = 0.5
gain_basic = 0.858 * (0.5)^(-0.083) ≈ 0.858 * 1.058 ≈ 0.908

Gustafsson gain (assuming err_prev = 0.8, h_n/h_{n-1} = 1.0):
ratio = 0.5 * 0.5 / 0.8 = 0.3125
gain_gus = 0.9 * 1.0 * (0.3125)^(-0.083) * 0.9
         ≈ 0.81 * 1.098 ≈ 0.889

gain = min(0.889, 0.908) = 0.889
Step size DECREASES by ~11%
```

**OrdinaryDiffEq.jl (assuming iter=3, M=20, γ=0.9, erracc=0.8, dtacc/dt=1.0):**
```
expo = 1/6 ≈ 0.167
fac = min(0.9, (1 + 40)*0.9 / (3 + 40)) ≈ 0.858

Basic ratio:
EEst = 0.5
qtmp = (0.5)^(0.167) / 0.858 ≈ 0.890 / 0.858 ≈ 1.037
q = 1.037 (clamped to range)

Gustafsson ratio (success_iter > 0):
qgus = 1.0 * ((0.5)² / 0.8)^(0.167)
     = ((0.25 / 0.8)^(0.167))
     = (0.3125)^(0.167)
     ≈ 0.816
qgus_final = 0.816 / 0.9 ≈ 0.907

qacc = max(1.037, 0.907) = 1.037
dt_new = dt / 1.037
Step size DECREASES by ~4%
```

**Key observation:** With error below tolerance (ε² < 1.0), OrdinaryDiffEq.jl actually **increases** the step size more aggressively due to the larger exponent. The example shows ODE.jl reduces less (4%) compared to CuBIE (11%).

**For error above tolerance (ε² = 1.5):**

**CuBIE:**
```
gain_basic = 0.858 * (1.5)^(-0.083) ≈ 0.858 * 0.967 ≈ 0.830
Step size DECREASES by ~17%
```

**OrdinaryDiffEq.jl:**
```
qtmp = (1.5)^(0.167) / 0.858 ≈ 1.068 / 0.858 ≈ 1.245
Step size DECREASES by ~20% (dt/1.245)
```

**When error exceeds tolerance, OrdinaryDiffEq.jl reduces MORE aggressively** (20% vs 17%).

### Expected Step Count Differences

Based on the mathematical formulations and actual behavior:

**CuBIE Gustafsson:**
- Smaller exponent `1/(2(p+1))` produces gentler responses
- Conservative minimum-selection strategy
- Expected step count: **similar to OrdinaryDiffEq.jl** for well-behaved problems
- Advantage: More stable for problems where Newton convergence varies

**OrdinaryDiffEq.jl Predictive:**
- Larger exponent `1/(p+1)` produces 2× faster response
- More aggressive adaptation when error is small
- More conservative reduction when error is large
- Expected step count: **similar to CuBIE** for well-behaved problems
- Advantage: Faster adaptation to changing problem difficulty

**Textbook H211b (b=4):**
- Smoothest step size sequence
- Expected step count: **baseline ± 3%** compared to standard PI controller
- Advantage: Minimal step size oscillation, predictable behavior

**Key insight:** The mathematical differences between CuBIE and OrdinaryDiffEq.jl are **compensating** - ODE.jl's larger exponent is balanced by its different gain selection strategy, resulting in similar overall efficiency but different transient behavior.

---

## Summary Table: CuBIE vs OrdinaryDiffEq.jl

| Aspect | CuBIE Gustafsson | OrdinaryDiffEq.jl Predictive |
|--------|-----------------|----------------------------|
| **Error exponent** | `1/(2(p+1))` | `1/(p+1)` **(2× larger)** |
| **Previous error weight** | `ε_prev^(1/(p+1))` | `ε_prev^(2/(p+1))` **(2× larger)** |
| **Adaptive damping** | **IDENTICAL** `fac = min(γ, ...)` | **IDENTICAL** `fac = min(γ, ...)` |
| **Error term in predictor** | `(ε⁴/ε²_prev)^(-1/(2(p+1)))` | `(ε⁴/ε²_prev)^(-1/(p+1))` |
| **Selection strategy** | `min(gain_gus, gain_basic)` | `max(q, qgus)` **(equivalent!)** |
| **When predictor applies** | Every accepted step | After first success |
| **Response to error < tol** | Gentler increase | Faster increase **(2×)** |
| **Response to error > tol** | Gentler decrease | Faster decrease **(2×)** |
| **Overall efficiency** | **Similar** | **Similar** |
| **Transient behavior** | Smoother, more damped | Snappier, faster adapting |
| **Best for** | Implicit methods with varying convergence | General purpose, rapid adaptation |

---

## Conclusion

### CuBIE vs OrdinaryDiffEq.jl: Core Differences

The implementations are **nearly identical in structure** but differ in **two critical parameters**:

1. **Exponent:** OrdinaryDiffEq.jl uses `1/(p+1)` while CuBIE uses `1/(2(p+1))` - exactly **half**
2. **Previous error weight:** OrdinaryDiffEq.jl uses `ε_prev^(2/(p+1))` while CuBIE uses `ε_prev^(1/(p+1))` - exactly **double**

These differences mean:
- **OrdinaryDiffEq.jl responds 2× faster** to error changes (both increasing and decreasing step size)
- **OrdinaryDiffEq.jl gives 2× more weight** to historical error trends
- Both use **identical adaptive damping** based on iteration counts
- Both use **identical squared error ratios** `ε⁴/ε²_prev` in the predictive term
- Both **choose the more conservative** option (mathematically equivalent selection)

### Practical Implications

**CuBIE's approach:**
- More damped, smoother step size changes
- Less sensitive to error fluctuations
- Better for problems where error estimates are noisy
- Matches intention to work like OrdinaryDiffEq with slightly different tuning

**OrdinaryDiffEq.jl's approach:**
- Snappier response to changing conditions
- Faster adaptation to improving/degrading situations
- Better for problems with rapid transients
- Default choice for general-purpose solving

**Both implementations are sound** - they represent different points on the conservatism-responsiveness trade-off. The mathematical structure is nearly identical, with only the tuning parameters differing by factors of 2.
- Use **OrdinaryDiffEq.jl Predictive** for general-purpose work with good efficiency
- Use **Textbook H211b** when step size smoothness is critical or when theoretical guarantees are needed

The mathematical differences in exponents, gain selection strategies, and use of historical information lead to measurably different step size sequences, with CuBIE being most conservative, OrdinaryDiffEq.jl most aggressive, and H211b most smooth.
