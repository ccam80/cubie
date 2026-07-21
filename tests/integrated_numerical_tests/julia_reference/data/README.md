# Vendored Julia reference data

Machine-independent golden data for the Julia reference gate
(`tests/integrated_numerical_tests/julia_reference/`), vendored from a
GPUODEBenchmarks checkout by `benchmarks/vendor_julia_reference.py`.

- `julia_reference_ne.npz` — compressed arrays:
  - `golden_rho` (1024,) float64: the float32-rounded rho grid every
    consumer integrates (bit-identical inputs).
  - `golden_states` (1024, 3) float64: Lorenz ensemble final states at
    t=1 from DifferentialEquations.jl Vern9 at tol 1e-13 in Float64
    (`generate_golden_ne.jl`).
  - `fixed_<alias>_dts` / `fixed_<alias>_finals`: per-algorithm Float32
    final states from raw DifferentialEquations.jl (CPU) at each dt of
    the dyadic grid 2^-1 .. 2^-13.
  - `adaptive_<alias>_tols` / `adaptive_<alias>_finals`: per-algorithm
    Float32 final states at each atol=rtol tolerance of the grid
    1e-2 .. 1e-6, run with OrdinaryDiffEq's resolved default controller.
- `algorithms.csv` — the mutual algorithm table: cubie alias, the
  matching DifferentialEquations.jl constructor, theoretical order,
  family, and notes.
- `controller_constants.csv` — OrdinaryDiffEq's resolved
  default-controller constants per algorithm, exported by the Julia
  runner; the gate mirrors these onto cubie's controllers for the
  matched adaptive tier.

To regenerate after the Julia sweeps change, run from the repo root:

    python benchmarks/vendor_julia_reference.py [path-to-GPUODEBenchmarks]
