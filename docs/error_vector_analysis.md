# Error Vector Calculation Analysis

This document provides a detailed mathematical trace of how the error vector is calculated in each of CuBIE's integrator algorithms: ERK (Explicit Runge-Kutta), DIRK (Diagonally Implicit RK), FIRK (Fully Implicit RK), and Rosenbrock.

## Table of Contents

1. [Background: Butcher Tableau and Error Estimation](#background)
2. [ERK (Explicit Runge-Kutta)](#erk)
3. [DIRK (Diagonally Implicit RK)](#dirk)
4. [FIRK (Fully Implicit RK)](#firk)
5. [Rosenbrock](#rosenbrock)
6. [Comparison Summary](#comparison)
7. [Identified Issues](#issues)

---

## Background: Butcher Tableau and Error Estimation {#background}

### Standard Runge-Kutta Form

For an ODE system `dy/dt = f(t, y)`, a Runge-Kutta method computes:

```
Y_i = y_n + h * Σ_{j=1}^{s} a_{ij} * k_j        (stage values)
k_i = f(t_n + c_i*h, Y_i)                        (stage derivatives)
y_{n+1} = y_n + h * Σ_{i=1}^{s} b_i * k_i       (solution)
ŷ_{n+1} = y_n + h * Σ_{i=1}^{s} b̂_i * k_i       (embedded estimate)
```

### Error Estimation

The local truncation error estimate is:

```
error = y_{n+1} - ŷ_{n+1} 
      = h * Σ_{i=1}^{s} (b_i - b̂_i) * k_i
      = h * Σ_{i=1}^{s} d_i * k_i
```

where `d_i = b_i - b̂_i` are the error weights.

### Key Point for Classical RK Methods (ERK, DIRK, FIRK)

**The error weights `d` multiply the stage derivatives `k_i = f(Y_i)`, not the stage increments or stage states.** The result is then scaled by `h` (dt).

### Key Point for Rosenbrock Methods

Rosenbrock methods are structurally different. The stage increments `k_i` in Rosenbrock methods are **not** equal to `f(Y_i)`. Instead, they solve:

```
(I - γ*h*J) * k_i = h * [f(Y_i) + Σ_{j<i} (C_{ij}/h) * k_j + γ_i * (∂f/∂t)]
```

The error formula is:
```
error = Σ_i d_i * k_i       (NO explicit h scaling)
```

The step size is already embedded in the definition of `k_i`.

### CuBIE Tableau Properties

In `base_algorithm_step.py`, the `ButcherTableau` class defines:
- `d` property: Returns `b - b_hat` (error weights)
- `error_weights()` method: Returns typed `d` values
- `accumulates_error` property: `True` if error must be accumulated stage-by-stage
- `b_hat_matches_a_row` property: Returns row index if `b_hat` matches an `a` row (optimization)

---

## ERK (Explicit Runge-Kutta) {#erk}

**File**: `src/cubie/integrators/algorithms/generic_erk.py`

### Key Buffer Contents

| Buffer | Contents | Units |
|--------|----------|-------|
| `stage_rhs` | `f(Y_i)` | derivative (1/time) |
| `stage_accumulator` | Accumulated `Σ a_{ij} * f(Y_j)` | derivative (1/time) |
| `proposed_state` (during loop) | Accumulated `Σ b_i * f(Y_i)` | derivative (1/time) |
| `error` (during loop) | Accumulated `Σ d_i * f(Y_i)` | derivative (1/time) |

### Stage Computation

For each stage `i` (lines 687-736):
```python
# stage_accumulator stores unscaled derivative sums
for successor_idx in range(stages_except_first):
    coeff = matrix_col[successor_idx + int32(1)]
    row_offset = successor_idx * n
    for idx in range(n):
        increment = stage_rhs[idx]  # f(Y_prev) - unscaled derivative
        stage_accumulator[row_offset + idx] += coeff * increment
        # At this point: Σ_{j<i} a_{ij} * f(Y_j)

# Convert to stage state (line 706-710):
dt_stage = dt_scalar * stage_nodes[stage_idx]
for idx in range(n):
    stage_accumulator[base] = (
        stage_accumulator[base] * dt_scalar + state[idx]  # Y_i = y_n + h * Σ
    )

# Evaluate derivative at stage state:
dxdt_fn(..., stage_rhs, stage_time)  # stage_rhs = f(Y_i)
```

### Error Accumulation Path

**Standard accumulation path** (when `accumulates_error == True`, lines 673-679, 760-772):

```python
# During stage loop (line 678-679):
if accumulates_error:
    increment = stage_rhs[idx]           # f(Y_i) - unscaled derivative
    error[idx] = error[idx] + error_weights[0] * increment
                                         # Σ d_i * f(Y_i)

# After stage loop (line 767-769):
if accumulates_error:
    error[idx] *= dt_scalar              # h * Σ d_i * f(Y_i)
```

**Mathematical form**:
```
error = h * Σ_i d_i * f(Y_i) = h * Σ_i d_i * k_i    ✓ CORRECT
```

**Non-accumulating path** (when `b_hat_matches_a_row is not None`, lines 756-772):

```python
# Capture stage state directly (lines 756-758):
if b_hat_row is not None:
    for idx in range(n):
        error[idx] = stage_accumulator[(b_hat_row - 1) * n + idx]
        # This is Y_{b_hat_row} = y_n + h * Σ_j a_{r,j} * f(Y_j)
        
# After loop (lines 770-772):
else:
    error[idx] = proposed_state[idx] - error[idx]
    # = (y_n + h * Σ b_i * k_i) - (y_n + h * Σ a_{r,j} * k_j)
    # = h * Σ (b_i - b̂_i) * k_i  (since b̂ = a_row_r)
```

**Mathematical form**:
```
error = y_{n+1} - Y_{b_hat_row} 
      = y_{n+1} - ŷ_{n+1}           ✓ CORRECT
```

---

## DIRK (Diagonally Implicit RK) {#dirk}

**File**: `src/cubie/integrators/algorithms/generic_dirk.py`

### Key Buffer Contents

| Buffer | Contents | Units |
|--------|----------|-------|
| `stage_rhs` | `f(Y_i)` after implicit solve converges | derivative (1/time) |
| `stage_increment` | Newton solver solution (note: NOT `h * f(Y)`) | varies |
| `stage_accumulator` | Accumulated `h * Σ a_{ij} * f(Y_j)` | state increment |
| `stage_base` | Current stage state `Y_i` | state |
| `proposed_state` (during loop) | Accumulated `Σ b_i * f(Y_i)` | derivative (1/time) |
| `error` (during loop) | Accumulated `Σ d_i * f(Y_i)` | derivative (1/time) |

### Important: Understanding the Implicit Solve

The DIRK nonlinear solver finds `stage_increment` such that:

```
residual = β * M * u - γ * h * f(base_state + a_{ii} * u) = 0
```

With `β = 1`, `γ = 1`, `M = I`:
```
u = h * f(base_state + a_{ii} * u)
```

So `stage_increment = h * f(Y_i)` where `Y_i` is the converged stage state.

**However**, after the implicit solve, the code **re-evaluates** `f(Y_i)` into `stage_rhs` (lines 977-984):

```python
observables_function(stage_base, ...)
dxdt_fn(stage_base, ..., stage_rhs, stage_time)
```

This gives `stage_rhs = f(Y_i)` which is the **unscaled derivative**.

### Stage Computation

For stages 1 through s-1 (lines 928-933):
```python
# Stream previous stage's RHS into accumulators for successors:
for successor_idx in range(stages_except_first):
    coeff = matrix_col[successor_idx + int32(1)]
    row_offset = successor_idx * n
    for idx in range(n):
        contribution = coeff * stage_rhs[idx] * dt_scalar  # h * a_{ij} * f(Y_j)
        stage_accumulator[row_offset + idx] += contribution

# Convert accumulator to stage state (line 948):
stage_base[idx] = stage_accumulator[stage_offset + idx] + state[idx]
# Y_i = y_n + h * Σ_{j<i} a_{ij} * f(Y_j)
```

### Error Accumulation Path

**Standard accumulation path** (when `accumulates_error == True`, lines 995-997, 1008-1010):

```python
# During stage loop (lines 995-997):
if accumulates_error:
    increment = stage_rhs[idx]  # f(Y_i) - unscaled derivative
    error[idx] += error_weight * increment

# After loop (lines 1008-1010):
if accumulates_error:
    error[idx] *= dt_scalar     # h * Σ d_i * f(Y_i)
```

**Mathematical form**:
```
error = h * Σ_i d_i * f(Y_i)    ✓ CORRECT
```

**Non-accumulating path** (lines 998-1000, 1011-1012):

```python
# Capture stage state (lines 998-1000):
elif b_hat_row == stage_idx:
    error[idx] = stage_base[idx]  # Y_{b_hat_row}

# After loop (lines 1011-1012):
else:
    error[idx] = proposed_state[idx] - error[idx]
    # = y_{n+1} - Y_{b_hat_row}
```

**Mathematical form**:
```
error = y_{n+1} - Y_{b_hat_row} = y_{n+1} - ŷ_{n+1}    ✓ CORRECT
```

---

## FIRK (Fully Implicit RK) {#firk}

**File**: `src/cubie/integrators/algorithms/generic_firk.py`

### Key Buffer Contents

| Buffer | Contents | Units |
|--------|----------|-------|
| `stage_increment` | Newton solver solution for all stages | `h * f(Y)` for each stage |
| `stage_rhs_flat` | `f(Y_i)` after explicit evaluation | derivative (1/time) |
| `stage_state` | Current stage state `Y_i` | state |
| `proposed_state` | Final solution | state |
| `error` | Error estimate | state |

### Important: Understanding the FIRK Implicit Solve

The FIRK n-stage residual function solves for all stage increments simultaneously:

```
residual_i = β * M * u_i - γ * h * f(t + c_i*h, y_n + Σ_j a_{ij} * u_j) = 0
```

With `β = 1`, `γ = 1`, `M = I`:
```
u_i = h * f(Y_i)
```

where `Y_i = y_n + Σ_j a_{ij} * u_j`.

So `stage_increment` contains `h * k_j` where `k_j = f(Y_j)`.

### Stage State Computation (lines 777-786)

```python
for idx in range(n):
    value = state[idx]  # y_n
    for contrib_idx in range(stage_count):
        flat_idx = stage_idx * stage_count + contrib_idx
        coeff = stage_rhs_coeffs[flat_idx]  # a_{ij}
        if coeff != typed_zero:
            value += coeff * stage_increment[contrib_idx * n + idx]
            # += a_{ij} * (h * k_j) = h * a_{ij} * k_j
    stage_state[idx] = value
    # Y_i = y_n + h * Σ_j a_{ij} * k_j    ✓ CORRECT
```

**Note**: The a_{ij} coefficients multiply the stage_increment which already contains `h * k_j`. This computes `Y_i = y_n + Σ_j a_{ij} * (h * k_j) = y_n + h * (Σ_j a_{ij} * k_j)`. ✓

### Derivative Re-evaluation (lines 812-821)

After computing the stage state, the code **re-evaluates** the derivative:

```python
if do_more_work:
    observables_function(stage_state, ...)
    stage_rhs = stage_rhs_flat[stage_idx * n:(stage_idx + 1) * n]
    dxdt_fn(stage_state, ..., stage_rhs, stage_time)
    # stage_rhs = f(Y_i) - unscaled derivative
```

This is crucial: `stage_rhs_flat` now contains `f(Y_i)`, not `h * f(Y_i)`.

### Error Accumulation Path (lines 839-846)

**Standard accumulation path** (when `accumulates_error == True`):

```python
if has_error and accumulates_error:
    for idx in range(n):
        error_acc = typed_zero
        for stage_idx in range(stage_count):
            rhs_value = stage_rhs_flat[stage_idx * n + idx]  # f(Y_i)
            error_acc += error_weights[stage_idx] * rhs_value
        error[idx] = dt_scalar * error_acc  # h * Σ d_i * f(Y_i)
```

**Mathematical form**:
```
error = h * Σ_i d_i * f(Y_i)    ✓ CORRECT
```

**Non-accumulating path** (lines 793-796, 864-866):

```python
# Capture stage state (lines 793-796):
if not accumulates_error:
    if b_hat_row == stage_idx:
        for idx in range(n):
            error[idx] = stage_state[idx]  # Y_{b_hat_row}

# After all stages (lines 864-866):
if not accumulates_error:
    for idx in range(n):
        error[idx] = proposed_state[idx] - error[idx]
        # = y_{n+1} - Y_{b_hat_row}
```

**Mathematical form**:
```
error = y_{n+1} - Y_{b_hat_row} = y_{n+1} - ŷ_{n+1}    ✓ CORRECT
```

---

## Rosenbrock {#rosenbrock}

**File**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

### Key Buffer Contents

| Buffer | Contents | Units |
|--------|----------|-------|
| `stage_rhs` | RHS of linear system before solve | scaled |
| `stage_increment` (after solve) | `k_i` - Rosenbrock stage increment | state increment (already h-scaled) |
| `stage_store` | Previous stage increments `k_j` | state increment |
| `stage_slice` | Stage state `Y_i = y_n + Σ_{j<i} a_{ij} * k_j` | state |
| `proposed_state` | Final solution | state |
| `error` | Error estimate | state |

### Rosenbrock Method Form

Rosenbrock methods solve for stage increments `k_i` via:

```
(M - γ*h*J) * k_i = h * f(Y_i) + Σ_{j<i} C_{ij} * k_j + γ_i * h * (∂f/∂t)
```

where:
- `Y_i = y_n + Σ_{j<i} a_{ij} * k_j` (stage state)
- `J = ∂f/∂y` (Jacobian at y_n)
- `γ` is the diagonal shift parameter
- `C_{ij}` are Jacobian-update coefficients
- `γ_i` are per-stage time-derivative weights

**Key difference from RK**: The `k_i` are NOT equal to `h * f(Y_i)`. They include corrections from the Jacobian and time derivative terms.

### Linear System RHS Construction (lines 743-750, 868-884)

**Stage 0:**
```python
f_value = stage_rhs[idx]  # f(y_n)
rhs_value = (f_value + gamma_stages[0] * time_derivative[idx]) * dt_scalar
stage_rhs[idx] = rhs_value * gamma
# = γ * h * (f(y_n) + γ_0 * ∂f/∂t)
```

**Stages 1+:**
```python
correction = Σ_{j<i} C_{ij} * k_j  # C-term corrections
f_stage_val = stage_rhs[idx]       # f(Y_i)
deriv_val = stage_gamma * time_derivative[idx]  # γ_i * h * ∂f/∂t
rhs_value = f_stage_val + correction * inv_dt + deriv_val
stage_rhs[idx] = rhs_value * dt_scalar * gamma
# = γ * h * (f(Y_i) + Σ C_{ij}/h * k_j + γ_i * ∂f/∂t)
```

### Stage State and Solution (lines 797-813)

```python
# Compute stage state:
stage_slice[idx] = state[idx]  # y_n
for predecessor_idx in range(stages_except_first):
    a_coeff = a_col[stage_idx]
    if predecessor_idx < stage_idx:
        prior_val = stage_store[base_idx + idx]  # k_j
        stage_slice[idx] += a_coeff * prior_val  # Y_i = y_n + Σ a_{ij} * k_j

# Note: NO h-scaling here because k_j already incorporates the step size
```

### Error Accumulation Path

**Standard accumulation path** (when `accumulates_error == True`, lines 779-781, 916-922):

```python
# During stage loop (lines 779-781):
if accumulates_error:
    error[idx] += stage_increment[idx] * error_weights[stage_idx]
    # Σ d_i * k_i

# NO dt scaling afterwards!
```

**Mathematical form**:
```
error = Σ_i d_i * k_i    ✓ CORRECT for Rosenbrock
```

This is correct because in Rosenbrock methods, the solution formula is:
```
y_{n+1} = y_n + Σ_i b_i * k_i    (NO explicit h)
ŷ_{n+1} = y_n + Σ_i b̂_i * k_i
error = y_{n+1} - ŷ_{n+1} = Σ_i (b_i - b̂_i) * k_i = Σ_i d_i * k_i
```

**Non-accumulating path** (lines 841-846, 925-927):

```python
# Capture stage state (lines 844-846):
if b_hat_row == stage_idx:
    for idx in range(n):
        error[idx] = stage_slice[idx]  # Y_{b_hat_row}

# After loop (lines 925-927):
if not accumulates_error:
    for idx in range(n):
        error[idx] = proposed_state[idx] - error[idx]
        # = y_{n+1} - Y_{b_hat_row}
```

**Mathematical form**:
```
error = y_{n+1} - Y_{b_hat_row} = y_{n+1} - ŷ_{n+1}    ✓ CORRECT
```

---

## Comparison Summary {#comparison}

### Quantity Multiplied by Error Weights

| Algorithm  | Quantity Multiplied | What It Is | Scaling |
|------------|---------------------|------------|---------|
| ERK        | `stage_rhs[idx]` | `f(Y_i)` | `error *= h` after loop |
| DIRK       | `stage_rhs[idx]` | `f(Y_i)` | `error *= h` after loop |
| FIRK       | `stage_rhs_flat[stage_idx * n + idx]` | `f(Y_i)` | `error = h * Σ` inline |
| Rosenbrock | `stage_increment[idx]` | `k_i` (stage increment) | No scaling (implicit) |

### Mathematical Formulas

| Algorithm  | Accumulation Formula | Non-Accumulating Formula |
|------------|---------------------|--------------------------|
| ERK        | `error = h * Σ d_i * f(Y_i)` | `error = y_{n+1} - Y_{b_hat_row}` |
| DIRK       | `error = h * Σ d_i * f(Y_i)` | `error = y_{n+1} - Y_{b_hat_row}` |
| FIRK       | `error = h * Σ d_i * f(Y_i)` | `error = y_{n+1} - Y_{b_hat_row}` |
| Rosenbrock | `error = Σ d_i * k_i` | `error = y_{n+1} - Y_{b_hat_row}` |

### Key Differences

1. **ERK, DIRK, FIRK** all use `f(Y_i)` (the stage derivative) and apply `h` scaling
2. **Rosenbrock** uses `k_i` (the stage increment) with NO explicit `h` scaling because:
   - Rosenbrock `k_i` are solutions to linear systems that already incorporate `h`
   - The solution formula is `y_{n+1} = y_n + Σ b_i * k_i` (no `h` factor)

---

## Identified Issues {#issues}

### All Paths Verified Correct ✓

After detailed analysis, all error calculation paths appear to be mathematically correct:

1. **ERK**: Uses `f(Y_i)`, scales by `h` at the end. ✓
2. **DIRK**: Uses `f(Y_i)` (re-evaluated after implicit solve), scales by `h` at the end. ✓
3. **FIRK**: Uses `f(Y_i)` (explicitly evaluated for each stage), scales by `h` inline. ✓
4. **Rosenbrock**: Uses `k_i` (stage increments), no scaling needed. ✓

### Potential Confusion Points

#### 1. DIRK Variable Naming (lines 989, 997)

```python
increment = stage_rhs[idx]  # Named "increment" but is actually f(Y_i)
```

The variable is named `increment` but contains the derivative `f(Y_i)`. This is confusing but mathematically correct.

**Recommendation**: Rename to `derivative` or `rhs_value` for clarity.

#### 2. FIRK stage_increment vs stage_rhs_flat

In FIRK:
- `stage_increment` contains `h * f(Y)` from the Newton solver
- `stage_rhs_flat` contains `f(Y)` after explicit re-evaluation

The error calculation correctly uses `stage_rhs_flat`, not `stage_increment`. ✓

#### 3. ERK Stage Accumulator Indexing (line 757)

```python
error[idx] = stage_accumulator[(b_hat_row - 1) * n + idx]
```

The `-1` offset accounts for stage 0 not being stored in the accumulator. This is correct but could be confusing.

### Buffer Content Summary

To clarify what each integrator stores in key buffers:

| Buffer Name | ERK | DIRK | FIRK | Rosenbrock |
|-------------|-----|------|------|------------|
| `stage_rhs` | `f(Y_i)` | `f(Y_i)` | N/A | Linear RHS |
| `stage_rhs_flat` | N/A | N/A | `f(Y_i)` | N/A |
| `stage_increment` | N/A | `h*f(Y_i)` | `h*f(Y_i)` | `k_i` |
| Accumulator | `Σ a_{ij}*f(Y_j)` | `h*Σ a_{ij}*f(Y_j)` | N/A | N/A |
| `stage_store` | N/A | N/A | N/A | `k_j` |

### Verification Steps for Debugging

If error estimates seem incorrect, verify:

1. **Check tableau definition**: Ensure `b_hat` sums to 1.0 and `d = b - b_hat` is computed correctly
2. **Check accumulates_error**: Verify `b_hat_matches_a_row` returns correct value for the tableau
3. **Check dt scaling**: ERK/DIRK/FIRK should scale by `h`, Rosenbrock should not
4. **Check derivative evaluation**: Ensure `dxdt_fn` is called with correct stage state after implicit solves
