# Implementation Task List
# Feature: buffer_location_refactor
# Plan Reference: .github/active_plans/buffer_location_refactor/agent_plan.md

## Task Group 1: Constructor plumbing in ODEImplicitStep - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 86-153)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 154-207)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 182-255)

**Input Validation Required**:
- preconditioned_vec_location, temp_location, delta_location, residual_location, residual_temp_location, stage_base_bt_location: validate values are either "local" or "shared" before forwarding; do not coerce None.

**Tasks**:
1. **Accept solver buffer location kwargs in ODEImplicitStep**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     def __init__(..., preconditioned_vec_location: Optional[str] = None, temp_location: Optional[str] = None, delta_location: Optional[str] = None, residual_location: Optional[str] = None, residual_temp_location: Optional[str] = None, stage_base_bt_location: Optional[str] = None) -> None:
         # Implementation logic:
         # 1. Keep existing parameter order; append new optional keyword-only args with default None.
         # 2. Build a linear_solver_kwargs dict including preconditioned_vec_location and temp_location only when not None.
         # 3. Instantiate LinearSolver with **linear_solver_kwargs.
         # 4. For NewtonKrylov creation, build newton_kwargs that include delta_location, residual_location, residual_temp_location, stage_base_bt_location only when not None, alongside existing Newton kwargs.
         # 5. Ensure solver_type branch still supports 'linear' path using the constructed linear_solver instance.
     ```
   - Edge cases: Preserve behavior when all new kwargs are omitted (defaults from configs apply). Reject invalid solver_type as current code does. Ensure None values do not override defaults.
   - Integration: Pass-through kwargs land in LinearSolverConfig/NewtonKrylovConfig to update buffer_registry allocations.

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/ode_implicitstep.py (~60 lines changed)
- Functions/Methods Added/Modified:
  * ODEImplicitStep.__init__ in ode_implicitstep.py
- Implementation Summary:
  Added optional solver buffer location parameters with validation and passed
  non-None values through LinearSolver and NewtonKrylov kwargs while retaining
  defaults for omitted arguments; documented location kwargs in the constructor
  docstring.
- Issues Flagged: None

---

## Task Group 2: Expose buffer location kwargs on implicit algorithms - PARALLEL
**Status**: [x]
**Dependencies**: Groups [1]

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler.py (lines 42-142)
- File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py (inherits BackwardsEulerStep; no __init__)
- File: src/cubie/integrators/algorithms/crank_nicolson.py (lines 48-158)
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 132-286)
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 141-295)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 146-279)

**Input Validation Required**:
- For each constructor: validate provided buffer location kwargs are "local" or "shared" before adding to solver_kwargs; skip None to preserve defaults.

**Tasks**:
1. **Plumb solver buffer locations through BackwardsEulerStep**
   - File: src/cubie/integrators/algorithms/backwards_euler.py
   - Action: Modify
   - Details:
     ```python
     def __init__(..., preconditioned_vec_location: Optional[str] = None, temp_location: Optional[str] = None, delta_location: Optional[str] = None, residual_location: Optional[str] = None, residual_temp_location: Optional[str] = None, stage_base_bt_location: Optional[str] = None) -> None:
         # Implementation logic:
         # 1. Add new optional parameters to signature and docstring.
         # 2. When building solver_kwargs, include each new location key only if not None and value in {"local", "shared"}.
         # 3. Forward solver_kwargs to super().__init__ unchanged.
     ```
   - Edge cases: None leaves defaults; invalid strings should raise ValueError before passing down.
   - Integration: BackwardsEulerPCStep inherits constructor; no extra changes.

2. **Plumb solver buffer locations through CrankNicolsonStep**
   - File: src/cubie/integrators/algorithms/crank_nicolson.py
   - Action: Modify
   - Details:
     ```python
     def __init__(..., preconditioned_vec_location: Optional[str] = None, temp_location: Optional[str] = None, delta_location: Optional[str] = None, residual_location: Optional[str] = None, residual_temp_location: Optional[str] = None, stage_base_bt_location: Optional[str] = None, dxdt_location: Optional[str] = None) -> None:
         # Add new optional params.
         # Validate locations against {"local", "shared"} before use.
         # Insert non-None values into solver_kwargs for pass-through to ODEImplicitStep.
     ```
   - Edge cases: Existing dxdt_location handling unchanged; ensure new kwargs do not overwrite defaults when None.
   - Integration: Controller defaults unaffected.

3. **Plumb solver buffer locations through DIRKStep**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     ```python
     def __init__(..., preconditioned_vec_location: Optional[str] = None, temp_location: Optional[str] = None, delta_location: Optional[str] = None, residual_location: Optional[str] = None, residual_temp_location: Optional[str] = None, stage_base_bt_location: Optional[str] = None, ...):
         # Append new optional params.
         # Validate against {"local", "shared"}.
         # Add non-None entries to solver_kwargs before calling super().__init__.
     ```
   - Edge cases: Maintain tableau-based defaults; avoid mutating config_kwargs.
   - Integration: Ensures stage solvers get buffer overrides.

4. **Plumb solver buffer locations through FIRKStep**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     def __init__(..., preconditioned_vec_location: Optional[str] = None, temp_location: Optional[str] = None, delta_location: Optional[str] = None, residual_location: Optional[str] = None, residual_temp_location: Optional[str] = None, stage_base_bt_location: Optional[str] = None, ...):
         # Add parameters, validate, and include in solver_kwargs when not None.
     ```
   - Edge cases: Preserve stage_count sizing update and registration order; None skips overrides.
   - Integration: Applies to coupled stage solver allocations.

5. **Plumb solver buffer locations through GenericRosenbrockWStep**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     def __init__(..., preconditioned_vec_location: Optional[str] = None, temp_location: Optional[str] = None, delta_location: Optional[str] = None, residual_location: Optional[str] = None, residual_temp_location: Optional[str] = None, stage_base_bt_location: Optional[str] = None, ...):
         # Append optional params; validate values.
         # Insert non-None linear solver locations (preconditioned_vec_location, temp_location) into solver_kwargs for Linear solver path.
         # Insert non-None Newton buffer locations (delta_location, residual_location, residual_temp_location, stage_base_bt_location) into solver_kwargs even though solver_type='linear' keeps Newton args unused.
     ```
   - Edge cases: Preserve solver_type='linear' usage; ensure None does not alter defaults; maintain cached_aux handling.

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/backwards_euler.py (~80 lines changed)
  * src/cubie/integrators/algorithms/crank_nicolson.py (~80 lines changed)
  * src/cubie/integrators/algorithms/generic_dirk.py (~80 lines changed)
  * src/cubie/integrators/algorithms/generic_firk.py (~80 lines changed)
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (~90 lines changed)
- Functions/Methods Added/Modified:
  * BackwardsEulerStep.__init__ in backwards_euler.py
  * CrankNicolsonStep.__init__ in crank_nicolson.py
  * DIRKStep.__init__ in generic_dirk.py
  * FIRKStep.__init__ in generic_firk.py
  * GenericRosenbrockWStep.__init__ in generic_rosenbrock_w.py
- Implementation Summary:
  Added solver buffer location keyword parameters, validated values against
  allowed locations, and forwarded non-None overrides into solver kwargs while
  preserving default behaviors and existing configuration handling; updated
  parameter documentation accordingly and relied on shared validation in
  ODEImplicitStep to avoid duplicate per-algorithm checks.
- Issues Flagged: None

---

## Task Group 3: Parameter whitelist update - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [1, 2]

**Required Context**:
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (lines 23-44)

**Input Validation Required**:
- None beyond set membership update.

**Tasks**:
1. **Add solver buffer location keys to ALL_ALGORITHM_STEP_PARAMETERS**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     ```python
     ALL_ALGORITHM_STEP_PARAMETERS = {
         ...
         'preconditioned_vec_location', 'temp_location',
         'delta_location', 'residual_location',
         'residual_temp_location', 'stage_base_bt_location',
     }
     ```
   - Edge cases: Maintain formatting and grouping with other buffer location parameters; ensure set contains new keys for factory filtering.
   - Integration: Allows get_algorithm_step and per-algorithm filtering to accept new kwargs without warnings.

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/base_algorithm_step.py (~6 lines changed)
- Functions/Methods Added/Modified:
  * ALL_ALGORITHM_STEP_PARAMETERS in base_algorithm_step.py
- Implementation Summary:
  Included solver buffer location keys in the algorithm parameter whitelist to
  allow factory filtering to accept the new kwargs.
- Issues Flagged: None
