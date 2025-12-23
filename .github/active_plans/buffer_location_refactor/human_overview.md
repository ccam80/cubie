### User Stories

1. **Configure solver buffer locations via algorithms**  
   - Acceptance: When users supply buffer location keywords (e.g.,
     `preconditioned_vec_location`, `delta_location`) to any implicit
     algorithm constructor or via `get_algorithm_step`, those values appear in
     the solver compile settings and resulting buffer registrations.  
   - Success: Buffer registry entries for solver-owned buffers use the
     provided locations; absent keywords leave solver defaults unchanged.

2. **Factory accepts new buffer location parameters without spurious
   warnings**  
   - Acceptance: `ALL_ALGORITHM_STEP_PARAMETERS` allows the new solver buffer
     location keywords so `get_algorithm_step` and per-algorithm filtering do
     not raise unused/invalid warnings when they are provided.  
   - Success: Algorithm selection and settings filtering succeed with or
     without the new keywords; defaults remain intact when parameters are not
     supplied.

3. **Preserve optional semantics for solver defaults**  
   - Acceptance: Passing `None` for any buffer location does not override the
     solver config defaults; only explicitly provided values propagate to
     solver constructors.  
   - Success: Solver compile settings reflect defaults unless users supply
     overrides; no regression in existing implicit algorithms’ default
     behavior.

### Overview

- **Goal**: Route buffer location configuration through implicit algorithm
  constructors into `ODEImplicitStep`, which then forwards the settings to
  `LinearSolver` and `NewtonKrylov` during initialization, while extending
  the algorithm parameter whitelist to cover these keywords.
- **Scope**: `src/cubie/integrators/algorithms` (implicit steps and
  parameter filtering). No solver core logic changes; only constructor
  wiring and constants.
- **Key touchpoints**:
  - `ODEImplicitStep.__init__` currently instantiates solvers without the
    buffer location kwargs from `LinearSolver`/`NewtonKrylov`.
  - Implicit algorithm `__init__` methods build `solver_kwargs` but do not
    expose solver buffer locations.
  - `ALL_ALGORITHM_STEP_PARAMETERS` lacks solver buffer location keywords,
    causing filtering to drop them.
- **Approach**:
  - Add optional buffer location parameters to implicit algorithm
    constructors and forward them to `ODEImplicitStep`.
  - Update `ODEImplicitStep.__init__` to pass non-`None` locations into
    `LinearSolver` and `NewtonKrylov` constructors, preserving optional
    semantics.
  - Extend `ALL_ALGORITHM_STEP_PARAMETERS` with the solver buffer location
    keys so factory filtering accepts them.
- **Trade-offs**:
  - Adding constructor keywords slightly increases surface area but avoids
    touching solver defaults or registration logic.
  - Keeps changes localized to algorithms, minimizing risk to solver device
    code.
- **References**:
  - `src/cubie/integrators/algorithms/ode_implicitstep.py`
  - `src/cubie/integrators/matrix_free_solvers/linear_solver.py`
  - `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`
  - `src/cubie/integrators/algorithms/base_algorithm_step.py`
- **Expected impact**: Users can steer solver buffer residency from the
  algorithm API without bypassing factory filtering; defaults remain
  unchanged when parameters are omitted.

```mermaid
flowchart LR
    A[Algorithm __init__\n(DIRK/FIRK/BE...)] -->|optional buffer locations| B[solver_kwargs]
    B --> C[ODEImplicitStep.__init__]
    C -->|non-None only| D[LinearSolver ctor]
    C -->|non-None only| E[NewtonKrylov ctor]
    D & E --> F[buffer_registry registrations]
    A -->|ALL_ALGORITHM_STEP_PARAMETERS| G[algorithm factory filtering]
```
