### Components to Adjust
- **ODEImplicitStep (`src/cubie/integrators/algorithms/ode_implicitstep.py`)**
  - Constructor wiring to instantiate `LinearSolver` and `NewtonKrylov` with
    buffer location kwargs when provided.
- **Implicit algorithm constructors**
  - `BackwardsEulerStep`, `BackwardsEulerPCStep` (inherits), `DIRKStep`,
    `FIRKStep`, `CrankNicolsonStep`, `GenericRosenbrockWStep`:
    expose solver buffer location kwargs and forward them to
    `ODEImplicitStep`.
- **Parameter whitelist**
  - `ALL_ALGORITHM_STEP_PARAMETERS` in
    `base_algorithm_step.py`: extend with solver buffer location keywords so
    factory filtering accepts them.

### Expected Behavior
- Optional solver buffer location keywords can be passed through algorithm
  constructors (and `get_algorithm_step`) into solver compile settings.
- If a buffer location argument is omitted or `None`, solver defaults in
  `LinearSolverConfig`/`NewtonKrylovConfig` remain unchanged.
- Buffer registry entries for solver-owned buffers reflect user-provided
  locations when supplied.

### Architectural Changes
- Add optional kwargs to `ODEImplicitStep.__init__` signature for:
  `preconditioned_vec_location`, `temp_location`, `delta_location`,
  `residual_location`, `residual_temp_location`, `stage_base_bt_location`.
- Pass these kwargs into `LinearSolver` and `NewtonKrylov` constructors only
  when not `None` (preserving current defaults).
- Extend each implicit algorithm `__init__` to accept the same optional
  buffer location kwargs and include them in the `solver_kwargs` forwarded
  to `super().__init__`.
- Update `ALL_ALGORITHM_STEP_PARAMETERS` to include the six new keywords so
  algorithm factory filtering recognizes them.

### Integration Points
- **Algorithm factory**: `get_algorithm_step` relies on
  `ALL_ALGORITHM_STEP_PARAMETERS`; ensure new keys pass validation.
- **Solver configs**: New kwargs map to existing fields in
  `LinearSolverConfig` (`preconditioned_vec_location`, `temp_location`) and
  `NewtonKrylovConfig` (`delta_location`, `residual_location`,
  `residual_temp_location`, `stage_base_bt_location`).
- **Buffer registry**: Solver constructors already register buffers using
  compile settings; propagation of locations suffices to change allocation
  behavior.

### Data Structures / Parameters
- Linear solver buffer locations: `preconditioned_vec_location`,
  `temp_location`.
- Newton–Krylov buffer locations: `delta_location`, `residual_location`,
  `residual_temp_location`, `stage_base_bt_location`.
- Algorithm-level kwargs collection (`solver_kwargs`) should conditionally
  include these keys.

### Edge Cases and Safeguards
- Preserve optional semantics: do not set solver kwargs when argument is
  `None`.
- Ensure backward compatibility: existing callers without new kwargs should
  behave identically (default locations remain 'local').
- Verify subclasses without explicit `__init__` (e.g., predictor-corrector)
  inherit updated base behavior without extra changes.

### Dependencies / Notes
- No changes expected in device code or buffer registration logic; only
  constructor parameter plumbing.
- Instrumented algorithm copies are unaffected because device functions are
  unchanged, but re-run syncing if constructors are mirrored elsewhere.
