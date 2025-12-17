# FIRK Solver Performance Difference Report

This report documents differences between the optimized CUDA functions in
`tests/all_in_one.py` and the module code in `src/cubie/integrators/`. Only
significant differences affecting datatypes, casts, index explicitness, and
line order are documented. Naming differences are excluded.

## Executive Summary

Several key differences were identified that may contribute to the ~100x
performance gap between `all_in_one.py` and the module code.

---

## 1. Linear Solver Factory

### 1.1 Correction Type Flag Datatypes

**all_in_one.py (lines 1189-1190):**
```python
sd_flag = True if correction_type == "steepest_descent" else False
mr_flag = True if correction_type == "minimal_residual" else False
```

**linear_solver.py (lines 207-208):**
```python
sd_flag = 1 if correction_type == "steepest_descent" else 0
mr_flag = 1 if correction_type == "minimal_residual" else 0
```

**Difference:** `all_in_one.py` uses Python `bool` (`True`/`False`) while
module uses `int` (`1`/`0`). Boolean values may have different JIT
compilation behavior and branch optimization characteristics.

### 1.2 Preconditioner Flag

**all_in_one.py:** Does not define a `preconditioned` flag - always calls
preconditioner unconditionally.

**linear_solver.py (line 213):**
```python
preconditioned = 1 if preconditioner is not None else 0
```

Then uses:
```python
if preconditioned:
    preconditioner(...)
else:
    for i in range(n_val):
        preconditioned_vec[i] = rhs[i]
```

**Difference:** `all_in_one.py` has no conditional branching for
preconditioner existence; the module adds an extra branch.

### 1.3 Return Statement Type Cast

**all_in_one.py (line 1270):**
```python
return int32(final_status)
```

**linear_solver.py (line 401):**
```python
return final_status
```

**Difference:** `all_in_one.py` explicitly casts the return value to `int32`,
while module returns without explicit cast.

---

## 2. Newton-Krylov Solver Factory

### 2.1 max_backtracks Calculation

**all_in_one.py (line 1403):**
```python
max_backtracks = int32(max_backtracks + 1)
```

**newton_krylov.py (line 280):**
```python
max_backtracks = int32(max_backtracks)
```

**Difference:** `all_in_one.py` adds 1 to `max_backtracks` before converting
to `int32`, while module does not. This affects the backtracking loop count.

### 2.2 residual_temp Allocation Location

**all_in_one.py (lines 1517):**
```python
residual_temp = cuda.local.array(n_arraysize, numba_prec)
```
Allocated **inside the backtracking loop**, within `if active_bt:` block.

**newton_krylov.py (lines 368-373):**
```python
if residual_temp_shared:
    residual_temp = shared_scratch[residual_temp_slice]
else:
    residual_temp = cuda.local.array(
        residual_temp_local_size, precision
    )
```
Allocated **before the main Newton iteration loop**.

**Difference:** `all_in_one.py` allocates `residual_temp` inside the
backtracking loop each time, potentially enabling better register allocation.
Module allocates once before the main loop with selective shared/local memory
support, adding overhead for buffer selection logic.

### 2.3 Linear Solver Shared Memory Slicing

**all_in_one.py (lines 1477-1489):**
Linear solver is called without passing a separate shared memory slice:
```python
lin_status = linear_solver(
    stage_increment,
    parameters,
    drivers,
    base_state,
    t,
    h,
    a_ij,
    residual,
    delta,
    krylov_iters_local,
)
```

**newton_krylov.py (lines 414-429):**
```python
lin_shared = shared_scratch[lin_solver_start:]
krylov_iters_local[0] = int32(0)
lin_status = linear_solver(
    stage_increment,
    parameters,
    drivers,
    base_state,
    t,
    h,
    a_ij,
    residual,
    delta,
    lin_shared,
    krylov_iters_local,
)
```

**Difference:** Module passes additional `lin_shared` slice to linear solver,
while `all_in_one.py` does not. This suggests `all_in_one.py` uses a simpler
linear solver interface without shared memory buffer passing.

### 2.4 Delta/Residual Buffer Allocation

**all_in_one.py (lines 1435-1436):**
```python
delta = shared_scratch[:n]
residual = shared_scratch[n : int32(2 * n)]
```
Direct slice notation with explicit `int32` cast on end index.

**newton_krylov.py (lines 354-366):**
```python
if delta_shared:
    delta = shared_scratch[delta_slice]
else:
    delta = cuda.local.array(delta_local_size, precision)
    for _i in range(delta_local_size):
        delta[_i] = typed_zero

if residual_shared:
    residual = shared_scratch[residual_slice]
else:
    residual = cuda.local.array(residual_local_size, precision)
    for _i in range(residual_local_size):
        residual[_i] = typed_zero
```

**Difference:** `all_in_one.py` unconditionally slices from shared_scratch;
module has conditional allocation with explicit zeroing of local arrays. The
zeroing loops add computational overhead.

---

## 3. FIRK Step Factory

### 3.1 Buffer Initialization in Step Function

**all_in_one.py (lines 2584-2613):**
Allocates buffers without initialization loops:
```python
if stage_state_shared:
    stage_state = shared[stage_state_start:stage_state_end]
else:
    stage_state = cuda.local.array(
        stage_state_size_ary, numba_precision
    )

if solver_scratch_shared:
    solver_scratch = shared[solver_scratch_start:solver_scratch_end]
else:
    solver_scratch = cuda.local.array(
        solver_scratch_ary, numba_precision
    )
```

**generic_firk.py (lines 717-744):**
Adds initialization loops when using local memory:
```python
if stage_state_shared:
    stage_state = shared[stage_state_slice]
else:
    stage_state = cuda.local.array(stage_state_local_size,
                                   precision)
    for _i in range(stage_state_local_size):
        stage_state[_i] = numba_precision(0.0)
```

**Difference:** Module adds explicit zero-initialization loops for local
arrays that `all_in_one.py` omits. These loops add overhead but may be
unnecessary if arrays are fully written before read.

### 3.2 Stage Driver Stack Size Calculation

**all_in_one.py (line 2470):**
```python
stage_driver_stack_local_size = max(int(stage_driver_stack_size), 1)
```

**generic_firk.py (line 670):**
```python
stage_driver_stack_local_size = local_sizes.nonzero('stage_driver_stack')
```

**Difference:** Direct `max(..., 1)` calculation vs method call through
buffer settings object. The property accessor adds indirection.

---

## 4. Compile-Time Constants

### 4.1 compile_kwargs Usage

**all_in_one.py:**
Uses `**compile_kwargs` consistently on all device function decorators:
```python
@cuda.jit(device=True, inline=True, **compile_kwargs)
def step(...):
```

**Module code:**
Missing `**compile_kwargs` on critical FIRK solver device functions:

- `newton_krylov.py` line 283-294:
  ```python
  @cuda.jit(device=True, inline=True)
  def newton_krylov_solver(...):
  ```

- `generic_firk.py` line 673-694:
  ```python
  @cuda.jit(device=True, inline=True,)
  def step(...):
  ```

**compile_kwargs definition** (`cuda_simsafe.py` lines 23-34):
```python
compile_kwargs = (
    {} if CUDA_SIMULATION
    else {
        'lineinfo': True,
        'fastmath': {
            'nsz': True,
            'contract': True,
            'arcp': True,
        },
    }
)
```

**Difference:** Missing `**compile_kwargs` means the Newton-Krylov solver and
FIRK step functions do NOT receive `fastmath` optimizations. This is a
HIGH IMPACT difference as `fastmath` enables:
- `nsz`: No signed zeros
- `contract`: Floating-point contraction (FMA)
- `arcp`: Approximate reciprocals

These optimizations can significantly improve GPU kernel performance.

---

## 5. Type Precision Handling

### 5.1 Precision Type Usage in Array Allocation

**all_in_one.py:**
Uses `numba_precision` (from `numba_from_dtype`) consistently:
```python
stage_state = cuda.local.array(stage_state_size_ary, numba_precision)
```

**generic_firk.py (lines 720-721):**
Uses raw `precision` (the dtype) in some places:
```python
stage_state = cuda.local.array(stage_state_local_size, precision)
```

**Difference:** `precision` is the raw numpy dtype, `numba_precision` is the
numba type derived from it. Using raw dtype may cause type inference overhead.

---

## Summary of High-Impact Differences

| Location | all_in_one.py | Module Code | Impact |
|----------|---------------|-------------|--------|
| **compile_kwargs** | **Consistent** | **MISSING** | **HIGH: No fastmath** |
| Linear solver flags | `bool` type | `int` type | Branch behavior |
| max_backtracks | `+1` offset | No offset | Loop count |
| residual_temp alloc | Inside loop | Before loop | Register pressure |
| Local array init | No zeroing | Zero loops | Overhead |
| delta/residual | Direct slice | Conditional | Branch overhead |
| Precision type | `numba_precision` | Mixed | Type inference |
| Return cast | Explicit `int32` | Implicit | Type consistency |

---

## Recommendations

1. **[HIGHEST PRIORITY] Add `**compile_kwargs`** to Newton-Krylov solver and
   FIRK step decorators. This enables fastmath optimizations and is likely
   the primary cause of the performance gap.

2. **Use boolean flags** in module code for `sd_flag`/`mr_flag` to match
   `all_in_one.py`.

3. **Add `+1` to max_backtracks** in `newton_krylov_solver_factory` to match
   the debug script behavior.

4. **Remove zero-initialization loops** for local arrays that are fully
   written before read, or verify they are necessary.

5. **Consider allocating residual_temp inside the backtracking loop** if
   register pressure permits.

6. **Use `numba_precision` consistently** for `cuda.local.array` allocations.

7. **Add explicit `int32()` cast** on return statements for type consistency.
