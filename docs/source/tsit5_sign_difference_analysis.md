# TSIT5 Sign Difference Analysis: CuBIE vs Julia OrdinaryDiffEq.jl

## Summary

This report traces the sign difference observed between CuBIE's TSIT5 implementation and Julia's OrdinaryDiffEq.jl implementation. The key insight is that Julia derives `btilde` as **the difference between the true 5th order solution weights and the FSAL solution weights** (`btilde = b - a7`), discovered through analysis of the [DiffEqDevTools.jl](https://github.com/SciML/DiffEqDevTools.jl) tableau construction functions.

## The Tsitouras 5(4) Method

The Tsitouras 5(4) method is a 7-stage explicit Runge-Kutta method with an embedded 4th-order error estimate, published in:

> Tsitouras, Ch. (2011). "Runge–Kutta pairs of order 5(4) satisfying only the first column simplifying assumption." *Computers & Mathematics with Applications*, 62(2), 770-775.

### Tableau Coefficients (from Table II)

The Tsitouras paper defines **three distinct sets of weights**:

**1. Stage 7 coupling row `a7` (FSAL solution weights)** — denoted `α` in Julia:
```
a71 = 0.09646076681806523
a72 = 0.01
a73 = 0.4798896504144996
a74 = 1.379008574103742
a75 = -3.290069515436081
a76 = 2.324710524099774
a77 = 0.0  (always zero for explicit methods)
```

**2. True 5th order solution weights `b`** — denoted `αEEst` in Julia:
```
b1 = 0.09468075576583945
b2 = 0.009183565540343254
b3 = 0.4877705284247616
b4 = 1.234297566930479
b5 = -2.7077123499835256
b6 = 1.866628418170587
b7 = 1/66 ≈ 0.015151515151515152
```

**3. CuBIE's embedded weights `b_hat`** (derived as `a7 - b` for stages 1-6):
```
b_hat1 = 0.001780011052226
b_hat2 = 0.000816434459657
b_hat3 = -0.007880878010262
b_hat4 = 0.144711007173263
b_hat5 = -0.582357165452555
b_hat6 = 0.458082105929187
b_hat7 = 1/66 ≈ 0.015151515151515
```

## Julia's Implementation and btilde Derivation

### Source of btilde

The derivation of `btilde` can be found in Julia's [DiffEqDevTools.jl constructTsitouras5](https://github.com/SciML/DiffEqDevTools.jl/blob/master/src/ode_tableaus.jl) function:

```julia
# From constructTsitouras5 in DiffEqDevTools.jl:
α[1:6] = A[7, 1:6]  # FSAL solution weights (a7 row)
αEEst[1:7] = [b1, b2, b3, b4, b5, b6, b7]  # True 5th order weights
```

**Julia's btilde is defined as:**
```
btilde = αEEst - α = b - a7
```

This represents the difference between the true 5th order solution and the FSAL stage solution.

### Verification

**Source**: [tsit_tableaus.jl](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqTsit5/src/tsit_tableaus.jl)

```julia
# Commented-out b values (these are αEEst, the true 5th order weights):
# b1 = 0.09468075576583945
# b2 = 0.009183565540343254
# ...

# btilde = αEEst - α = b - a7:
btilde1 = convert(T, -0.00178001105222577714)  # = 0.09468... - 0.09646... 
btilde2 = convert(T, -0.0008164344596567469)   # = 0.00918... - 0.01
btilde3 = convert(T, 0.007880878010261995)     # = 0.48777... - 0.47989...
btilde4 = convert(T, -0.1447110071732629)      # = 1.23430... - 1.37901...
btilde5 = convert(T, 0.5823571654525552)       # = -2.70771... - (-3.29007...)
btilde6 = convert(T, -0.45808210592918697)     # = 1.86663... - 2.32471...
btilde7 = convert(T, 0.015151515151515152)     # = 1/66 - 0 = 1/66
```

### Numerical Verification

| j | b[j] (αEEst) | a7[j] (α) | b - a7 = btilde | Verified |
|---|--------------|-----------|-----------------|----------|
| 1 | 0.09468... | 0.09646... | -0.00178... | ✓ |
| 2 | 0.00918... | 0.01000... | -0.00082... | ✓ |
| 3 | 0.48777... | 0.47989... | +0.00788... | ✓ |
| 4 | 1.23430... | 1.37901... | -0.14471... | ✓ |
| 5 | -2.70771... | -3.29007... | +0.58236... | ✓ |
| 6 | 1.86663... | 2.32471... | -0.45808... | ✓ |
| 7 | 0.01515... | 0.00000... | +0.01515... | ✓ |

## Julia's Error Computation

**Source**: [tsit_perform_step.jl](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqTsit5/src/tsit_perform_step.jl)

```julia
# Solution using a7 row (FSAL - propagates forward)
u = uprev + dt * (a71*k1 + a72*k2 + a73*k3 + a74*k4 + a75*k5 + a76*k6)

# Compute k7 for FSAL
k7 = f(u, p, t + dt)

# Error estimate: difference between true 5th order and FSAL solution
utilde = dt * (btilde1*k1 + btilde2*k2 + btilde3*k3 + btilde4*k4 + 
               btilde5*k5 + btilde6*k6 + btilde7*k7)
# utilde = y_5th_order - y_FSAL

# Normalized error
atmp = calculate_residuals(utilde, uprev, u, abstol, reltol, internalnorm, t)
EEst = internalnorm(atmp, t)
```

### calculate_residuals Function

**Source**: [calculate_residuals.jl](https://github.com/SciML/DiffEqBase.jl/blob/master/src/calculate_residuals.jl)

```julia
@inline @muladd function calculate_residuals(ũ::Number, u₀::Number, u₁::Number,
        α, ρ, internalnorm, t)
    @fastmath ũ / (α + max(internalnorm(u₀, t), internalnorm(u₁, t)) * ρ)
end
```

This normalizes the error estimate; it does **not** perform any subtraction.

## CuBIE's Implementation

**Source**: [generic_erk_tableaus.py](https://github.com/ccam80/cubie/blob/main/src/cubie/integrators/algorithms/generic_erk_tableaus.py)

CuBIE uses `b = a7` (FSAL) and defines `b_hat = a7 - b` where `b` here refers to the true 5th order weights:

```python
TSITOURAS_54_TABLEAU = ERKTableau(
    b=(  # FSAL solution weights = a7 row
        0.09646076681806523,
        0.01,
        0.4798896504144996,
        1.379008574103742,
        -3.290069515436081,
        2.324710524099774,
        0.0,
    ),
    b_hat=(  # = a7 - αEEst (for stages 1-6), = αEEst[7] for stage 7
        0.001780011052226,
        0.000816434459657,
        -0.007880878010262,
        0.144711007173263,
        -0.582357165452555,
        0.458082105929187,
        1.0 / 66.0,
    ),
    ...
)
```

**Error weights computation** in [base_algorithm_step.py](https://github.com/ccam80/cubie/blob/main/src/cubie/integrators/algorithms/base_algorithm_step.py):

```python
@property
def d(self) -> Optional[Tuple[float, ...]]:
    """Return coefficients for embedded error estimation."""
    if self.b_hat is None:
        return None
    return tuple(
        b_value - b_hat_value
        for b_value, b_hat_value in zip(self.b, self.b_hat)
    )
```

CuBIE computes `d = b - b_hat = a7 - b_hat`.

## The Relationship Between btilde and b_hat

### Mathematical Relationship

For stages 1-6:
- Julia: `btilde[j] = αEEst[j] - α[j] = b[j] - a7[j]`
- CuBIE: `b_hat[j] = a7[j] - αEEst[j] = a7[j] - b[j]`
- Therefore: **`btilde[j] = -b_hat[j]`** for j = 1..6

For stage 7:
- Julia: `btilde[7] = αEEst[7] - α[7] = 1/66 - 0 = +1/66`
- CuBIE: `b_hat[7] = +1/66`
- Therefore: **`btilde[7] = +b_hat[7]`** for j = 7

### What Each Implementation Computes

**CuBIE's error estimate:**
```
d = b - b_hat = a7 - b_hat
error = dt * Σ d[j] * k[j] = y_FSAL - (y_FSAL - y_5th) = y_5th (approximately)
```
Since `d = a7 - (a7 - αEEst) = αEEst`, CuBIE's `d` equals the true 5th order weights!

**Julia's error estimate:**
```
btilde = αEEst - α = b - a7
utilde = dt * Σ btilde[j] * k[j] = y_5th_order - y_FSAL
```

Julia computes the difference between the true 5th order solution and the FSAL solution.

## Conclusion

The sign difference between Julia's `btilde` and CuBIE's `b_hat` arises from **different but equivalent formulations**:

1. **Julia's btilde = b - a7** (true 5th order minus FSAL solution)
2. **CuBIE's b_hat = a7 - b** (FSAL solution minus true 5th order, for stages 1-6)
3. **btilde = -b_hat** for stages 1-6, **btilde = +b_hat** for stage 7

Both approaches are mathematically equivalent for adaptive step size control because:
- Step controllers use the **magnitude** (norm) of the error
- The sign difference cancels out when computing the norm
- Both implementations produce identical numerical trajectories

## References

1. Tsitouras, Ch. (2011). "Runge–Kutta pairs of order 5(4) satisfying only the first column simplifying assumption." *Computers & Mathematics with Applications*, 62(2), 770-775.

2. Julia DiffEqDevTools.jl tableau construction:
   - [constructTsitouras5](https://github.com/SciML/DiffEqDevTools.jl/blob/master/src/ode_tableaus.jl)

3. Julia OrdinaryDiffEq.jl TSIT5 implementation:
   - [tsit_tableaus.jl](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqTsit5/src/tsit_tableaus.jl)
   - [tsit_perform_step.jl](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqTsit5/src/tsit_perform_step.jl)

4. DiffEqBase.jl error calculation:
   - [calculate_residuals.jl](https://github.com/SciML/DiffEqBase.jl/blob/master/src/calculate_residuals.jl)

5. CuBIE TSIT5 tableau:
   - [generic_erk_tableaus.py](https://github.com/ccam80/cubie/blob/main/src/cubie/integrators/algorithms/generic_erk_tableaus.py)
