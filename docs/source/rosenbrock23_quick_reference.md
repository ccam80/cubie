# Rosenbrock23 Quick Reference

This is a condensed reference for the Rosenbrock23 implementation analysis. For complete details, see `rosenbrock23_mathematical_analysis.md`.

## Key Findings

- ✅ **CuBIE and OrdinaryDiffEq.jl are mathematically equivalent**
- ✅ **Both conform exactly to the textbook Rosenbrock-W formulation**
- ✅ **Differences are purely implementational** (GPU vs CPU optimizations)

## Tableau Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| d (gamma) | 1/(2+√2) ≈ 0.293 | Diagonal shift in linear system |
| C₁₀ | (1-d)/d² ≈ 8.243 | Jacobian coupling coefficient |
| Order | 3 | Classical convergence order |
| Stages | 3 | Number of implicit stages |
| Error | Embedded | Uses 3rd-order embedded formula |

## Algorithm at a Glance

```
Stage 0: (I - d·h·J)·K₀ = d·h·(f(tₙ, yₙ) + d·h·∂f/∂t)

Stage 1: (I - d·h·J)·K₁ = d·h·(f(tₙ + 0.5h, yₙ + 0.5·K₀) + C₁₀·K₀/h + d·h·∂f/∂t)

Stage 2: (I - d·h·J)·K₂ = d·h·(f(tₙ + h, yₙ + K₁) + d·h·∂f/∂t)

Solution:  yₙ₊₁ = yₙ + K₁
Error:     error = (1/6)·(K₀ - 2·K₁ + K₂)
```

## Symbol Mapping

| Math | CuBIE | Julia | Meaning |
|------|-------|-------|---------|
| h | `dt_scalar` | `dt` | Step size |
| yₙ | `state` | `u` | Current state |
| Kᵢ | `stage_increment` | `ki` | Stage increment |
| f | `dxdt_fn(...)` | `f(u,p,t)` | ODE RHS |
| J | via `prepare_jacobian` | `J` | Jacobian |

## Implementation Differences

1. **Linear Solver**: CuBIE uses Krylov (GMRES), Julia uses LU factorization
2. **Jacobian**: CuBIE computes once and caches, Julia may recompute
3. **Memory**: CuBIE uses contiguous stage storage for GPU, Julia uses named variables
4. **Optimization**: CuBIE has GPU-specific compile-time flags and predicated operations

**None of these affect mathematical correctness.**

## Files

- `src/cubie/integrators/algorithms/generic_rosenbrock_w.py` - CuBIE implementation
- `src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py` - Tableau definitions
- `src/cubie/integrators/loops/ode_loop.py` - Integration loop

## References

See `rosenbrock23_mathematical_analysis.md` for complete references and detailed derivations.
