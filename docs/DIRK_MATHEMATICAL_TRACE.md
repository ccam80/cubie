# DIRK Implementation: Detailed Mathematical Trace

## Overview

This document provides a complete line-by-line mathematical trace of the DIRK implementation for a 3-stage example, verifying correctness at each step.

---

## Example Tableau: 3-Stage DIRK

```
A matrix:
     col0    col1    col2
row0 [a00,   0,      0   ]
row1 [a10,   a11,    0   ]
row2 [a20,   a21,    a22 ]

b = [b0, b1, b2]
c = [c0, c1, c2]
```

This is a lower triangular tableau (DIRK property).

---

## Stage 0: Initial Stage

### Code Execution (Lines 1173-1270)

**Setup:**
```python
# Line 1185
stage_time = current_time + dt_scalar * stage_time_fractions[0]  # t_0 = t_n + c0 * dt
diagonal_coeff = diagonal_coeffs[0]  # a00

# Lines 1188-1191
for idx in range(n):
    stage_base[idx] = state[idx]  # stage_base = y_n
    if accumulates_output:
        proposed_state[idx] = typed_zero  # proposed_state = 0
```

**Mathematical State:**
- `t_0 = t_n + c0 * dt`
- `stage_base = y_n`
- `proposed_state = 0` (assuming accumulation)

### Implicit Solve

**Code:**
```python
# Lines 1215-1230
if stage_implicit[0]:  # if a00 != 0
    status_code |= nonlinear_solver(
        stage_increment,  # Solve for z_0
        parameters,
        proposed_drivers,
        stage_time,       # t_0
        dt_scalar,
        diagonal_coeffs[0],  # a00
        stage_base,       # y_n
        solver_scratch,
        counters,
    )
    for idx in range(n):
        stage_base[idx] += diagonal_coeff * stage_increment[idx]
```

**Mathematical Operation:**
- Solve nonlinear equation: `z_0 = dt * a00 * f(t_0, y_n + z_0)`
- Newton iteration finds `z_0` such that equation holds
- Update: `Y_0 = y_n + z_0 = y_n + dt * a00 * k_0`

**Mathematical State:**
- `Y_0 = y_n + dt * a00 * k_0` (where `k_0 = f(t_0, Y_0)`)

### Evaluate Derivative

**Code:**
```python
# Lines 1233-1248
observables_function(
    stage_base,  # Y_0
    parameters,
    proposed_drivers,
    proposed_observables,
    stage_time,  # t_0
)

dxdt_fn(
    stage_base,  # Y_0
    parameters,
    proposed_drivers,
    proposed_observables,
    stage_rhs,   # OUTPUT: k_0
    stage_time,  # t_0
)
```

**Mathematical Operation:**
- Compute observables at `Y_0`
- Compute `k_0 = f(t_0, Y_0)`

**Mathematical State:**
- `stage_rhs = k_0 = f(t_0, Y_0)`

### Accumulate Solution

**Code:**
```python
# Lines 1250-1267
solution_weight = solution_weights[0]  # b0
error_weight = error_weights[0]        # d0

for idx in range(n):
    rhs_value = stage_rhs[idx]  # k_0
    if accumulates_output:
        proposed_state[idx] += solution_weight * rhs_value  # += b0 * k_0
    elif b_row == int32(0):
        proposed_state[idx] = stage_base[idx]  # = Y_0
    
    if has_error:
        if accumulates_error:
            error[idx] += error_weight * rhs_value  # += d0 * k_0
        elif b_hat_row == int32(0):
            error[idx] = stage_base[idx]
```

**Mathematical Operation:**
- `proposed_state += b0 * k_0`
- `error += d0 * k_0` (if accumulating error)

**Mathematical State:**
- `proposed_state = b0 * k_0`
- `error = d0 * k_0`

### Initialize Accumulator

**Code:**
```python
# Lines 1269-1270
for idx in range(accumulator_length):
    stage_accumulator[idx] = typed_zero
```

**Mathematical State:**
- `accumulator[0:3n] = [0, 0, 0, ..., 0]` (all zeros)
- Ready for streaming from future stages

---

## Stage 1: Second Stage (prev_idx=0, stage_idx=1)

### Loop Entry (Lines 1276-1279)

**Code:**
```python
# prev_idx = 0 (first iteration)
stage_offset = int32(prev_idx * n)  # = 0
stage_idx = prev_idx + int32(1)      # = 1
matrix_col = explicit_a_coeffs[prev_idx]  # = explicit_a_coeffs[0] = column 0
```

**Mathematical State:**
- `stage_offset = 0`
- `stage_idx = 1`
- `matrix_col = [a[0][0], a[1][0], a[2][0]]` after fix
- But `a[0][0]` is diagonal (excluded), so `matrix_col = [0, a10, a20]`

### Streaming k_0 (Lines 1281-1287)

**Code:**
```python
for successor_idx in range(stages_except_first):  # successor_idx = 0, 1
    coeff = matrix_col[successor_idx + int32(1)]  # matrix_col[1], matrix_col[2]
    row_offset = successor_idx * n
    for idx in range(n):
        contribution = coeff * stage_rhs[idx] * dt_scalar  # coeff * k_0 * dt
        stage_accumulator[row_offset + idx] += contribution
```

**Mathematical Operation:**

**Iteration successor_idx=0:**
- `coeff = matrix_col[1] = a10`
- `row_offset = 0 * n = 0`
- `accumulator[0:n] += a10 * k_0 * dt`

**Iteration successor_idx=1:**
- `coeff = matrix_col[2] = a20`
- `row_offset = 1 * n = n`
- `accumulator[n:2n] += a20 * k_0 * dt`

**Mathematical State:**
- `accumulator[0:n] = dt * a10 * k_0` (for stage 1)
- `accumulator[n:2n] = dt * a20 * k_0` (for stage 2)

### Build Stage Base (Lines 1289-1303)

**Code:**
```python
# Line 1289-1291
stage_time = current_time + dt_scalar * stage_time_fractions[stage_idx]  # t_1

# Lines 1293-1298
if has_driver_function:
    driver_function(stage_time, driver_coeffs, proposed_drivers)

# Line 1301
stage_base = stage_accumulator[stage_offset:stage_offset + n]  # accumulator[0:n]

# Lines 1302-1303
for idx in range(n):
    stage_base[idx] += state[idx]  # stage_base = accumulator[0:n] + y_n
```

**Mathematical Operation:**
- `t_1 = t_n + c1 * dt`
- Update drivers to time `t_1`
- `stage_base = accumulator[0:n] + y_n`
- `stage_base = dt * a10 * k_0 + y_n`

**Mathematical State:**
- `stage_base = y_n + dt * a10 * k_0`

### Implicit Solve (Lines 1307-1321)

**Code:**
```python
diagonal_coeff = diagonal_coeffs[stage_idx]  # a11

if stage_implicit[stage_idx]:  # if a11 != 0
    status_code |= nonlinear_solver(
        stage_increment,  # Solve for z_1
        parameters,
        proposed_drivers,
        stage_time,       # t_1
        dt_scalar,
        diagonal_coeffs[stage_idx],  # a11
        stage_base,       # y_n + dt * a10 * k_0
        solver_scratch,
        counters,
    )
    
    for idx in range(n):
        stage_base[idx] += diagonal_coeff * stage_increment[idx]
```

**Mathematical Operation:**
- Solve: `z_1 = dt * a11 * f(t_1, stage_base + z_1)`
- Update: `Y_1 = stage_base + z_1`
- `Y_1 = y_n + dt * a10 * k_0 + dt * a11 * k_1`
- `Y_1 = y_n + dt * (a10 * k_0 + a11 * k_1)`

**Mathematical State:**
- `Y_1 = y_n + dt * (a10 * k_0 + a11 * k_1)`
- `stage_base = Y_1`

### Evaluate Derivative (Lines 1323-1338)

**Code:**
```python
observables_function(
    stage_base,  # Y_1
    parameters,
    proposed_drivers,
    proposed_observables,
    stage_time,  # t_1
)

dxdt_fn(
    stage_base,  # Y_1
    parameters,
    proposed_drivers,
    proposed_observables,
    stage_rhs,   # OUTPUT: k_1 (overwrites k_0!)
    stage_time,  # t_1
)
```

**Mathematical Operation:**
- `k_1 = f(t_1, Y_1)`
- `stage_rhs = k_1` (overwrites previous `k_0`)

**Mathematical State:**
- `stage_rhs = k_1 = f(t_1, Y_1)`

### Accumulate Solution (Lines 1340-1354)

**Code:**
```python
solution_weight = solution_weights[stage_idx]  # b1
error_weight = error_weights[stage_idx]        # d1

for idx in range(n):
    increment = stage_rhs[idx]  # k_1
    if accumulates_output:
        proposed_state[idx] += solution_weight * increment  # += b1 * k_1
    elif b_row == stage_idx:
        proposed_state[idx] = stage_base[idx]  # = Y_1
    
    if has_error:
        if accumulates_error:
            error[idx] += error_weight * increment  # += d1 * k_1
        elif b_hat_row == stage_idx:
            error[idx] = stage_base[idx]
```

**Mathematical Operation:**
- `proposed_state += b1 * k_1`
- `error += d1 * k_1`

**Mathematical State:**
- `proposed_state = b0 * k_0 + b1 * k_1`
- `error = d0 * k_0 + d1 * k_1`

---

## Stage 2: Third Stage (prev_idx=1, stage_idx=2)

### Loop Entry

**Code:**
```python
# prev_idx = 1 (second iteration)
stage_offset = int32(prev_idx * n)  # = n
stage_idx = prev_idx + int32(1)      # = 2
matrix_col = explicit_a_coeffs[prev_idx]  # = explicit_a_coeffs[1] = column 1
```

**Mathematical State:**
- `stage_offset = n`
- `stage_idx = 2`
- `matrix_col = [0, 0, a21]` (strict lower triangular)

### Streaming k_1 (Lines 1281-1287)

**Code:**
```python
for successor_idx in range(stages_except_first):  # successor_idx = 0, 1
    coeff = matrix_col[successor_idx + int32(1)]
    row_offset = successor_idx * n
    for idx in range(n):
        contribution = coeff * stage_rhs[idx] * dt_scalar  # coeff * k_1 * dt
        stage_accumulator[row_offset + idx] += contribution
```

**Mathematical Operation:**

**Iteration successor_idx=0:**
- `coeff = matrix_col[1] = 0`
- `accumulator[0:n] += 0 * k_1 * dt` (no effect - stage 1 already done)

**Iteration successor_idx=1:**
- `coeff = matrix_col[2] = a21`
- `row_offset = n`
- `accumulator[n:2n] += a21 * k_1 * dt`

**Mathematical State:**
- `accumulator[n:2n] = dt * a20 * k_0 + dt * a21 * k_1`

### Build Stage Base

**Code:**
```python
stage_time = current_time + dt_scalar * stage_time_fractions[stage_idx]  # t_2

if has_driver_function:
    driver_function(stage_time, driver_coeffs, proposed_drivers)

stage_base = stage_accumulator[stage_offset:stage_offset + n]  # accumulator[n:2n]

for idx in range(n):
    stage_base[idx] += state[idx]
```

**Mathematical Operation:**
- `t_2 = t_n + c2 * dt`
- `stage_base = accumulator[n:2n] + y_n`
- `stage_base = dt * (a20 * k_0 + a21 * k_1) + y_n`

**Mathematical State:**
- `stage_base = y_n + dt * (a20 * k_0 + a21 * k_1)`

### Implicit Solve

**Code:**
```python
diagonal_coeff = diagonal_coeffs[stage_idx]  # a22

if stage_implicit[stage_idx]:
    status_code |= nonlinear_solver(
        stage_increment,
        parameters,
        proposed_drivers,
        stage_time,
        dt_scalar,
        diagonal_coeffs[stage_idx],
        stage_base,
        solver_scratch,
        counters,
    )
    
    for idx in range(n):
        stage_base[idx] += diagonal_coeff * stage_increment[idx]
```

**Mathematical Operation:**
- Solve: `z_2 = dt * a22 * f(t_2, stage_base + z_2)`
- Update: `Y_2 = stage_base + z_2`
- `Y_2 = y_n + dt * (a20 * k_0 + a21 * k_1 + a22 * k_2)`

**Mathematical State:**
- `Y_2 = y_n + dt * (a20 * k_0 + a21 * k_1 + a22 * k_2)`

### Evaluate Derivative

**Code:**
```python
observables_function(stage_base, ..., stage_time)
dxdt_fn(stage_base, ..., stage_rhs, stage_time)
```

**Mathematical Operation:**
- `k_2 = f(t_2, Y_2)`

**Mathematical State:**
- `stage_rhs = k_2`

### Accumulate Solution

**Code:**
```python
solution_weight = solution_weights[stage_idx]  # b2
error_weight = error_weights[stage_idx]        # d2

for idx in range(n):
    increment = stage_rhs[idx]  # k_2
    if accumulates_output:
        proposed_state[idx] += solution_weight * increment  # += b2 * k_2
    
    if has_error and accumulates_error:
        error[idx] += error_weight * increment  # += d2 * k_2
```

**Mathematical Operation:**
- `proposed_state += b2 * k_2`
- `error += d2 * k_2`

**Mathematical State:**
- `proposed_state = b0 * k_0 + b1 * k_1 + b2 * k_2`
- `error = d0 * k_0 + d1 * k_1 + d2 * k_2`

---

## Finalization (Lines 1358-1391)

### Scale and Add Base State

**Code:**
```python
# Lines 1358-1366
for idx in range(n):
    if accumulates_output:
        proposed_state[idx] *= dt_scalar
        proposed_state[idx] += state[idx]
    
    if has_error:
        if accumulates_error:
            error[idx] *= dt_scalar
        else:
            error[idx] = proposed_state[idx] - error[idx]
```

**Mathematical Operation:**
- `proposed_state = proposed_state * dt + y_n`
- `proposed_state = dt * (b0 * k_0 + b1 * k_1 + b2 * k_2) + y_n`
- `y_{n+1} = y_n + dt * Σ(i=0 to 2) bi * ki` ✓

- `error = error * dt`
- `error = dt * (d0 * k_0 + d1 * k_1 + d2 * k_2)` ✓

**Final Mathematical State:**
- `y_{n+1} = y_n + dt * (b0 * k_0 + b1 * k_1 + b2 * k_2)` ✓
- `error = dt * (d0 * k_0 + d1 * k_1 + d2 * k_2)` ✓

### Update Drivers and Observables

**Code:**
```python
# Lines 1368-1381
if has_driver_function:
    driver_function(end_time, driver_coeffs, proposed_drivers)

observables_function(
    proposed_state,
    parameters,
    proposed_drivers,
    proposed_observables,
    end_time,
)
```

**Mathematical Operation:**
- Update drivers to end time `t_{n+1}`
- Compute final observables at `y_{n+1}`

---

## Verification Summary

### Stage Computations

✅ **Stage 0:** `Y_0 = y_n + dt * a00 * k_0`

✅ **Stage 1:** `Y_1 = y_n + dt * (a10 * k_0 + a11 * k_1)`

✅ **Stage 2:** `Y_2 = y_n + dt * (a20 * k_0 + a21 * k_1 + a22 * k_2)`

### General Formula

✅ **Stage i:** `Y_i = y_n + dt * Σ(j=0 to i) a_{i,j} * k_j`

### Solution

✅ **Solution:** `y_{n+1} = y_n + dt * Σ(i=0 to 2) bi * ki`

### Error

✅ **Error:** `error = dt * Σ(i=0 to 2) di * ki`

---

## Conclusion

The implementation correctly evaluates the DIRK method according to the mathematical specification. Each stage properly:
1. Accumulates explicit contributions from previous stages
2. Solves the implicit equation for the diagonal term
3. Evaluates the derivative at the stage value
4. Accumulates to the solution and error estimates

The final solution and error estimates match the expected DIRK formulas exactly.

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-10  
**Verification Status:** ✅ CORRECT
