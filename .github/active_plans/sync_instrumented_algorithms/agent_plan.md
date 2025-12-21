# Agent Plan: Synchronize Instrumented Algorithm Tests

## Purpose

Update the 6 instrumented algorithm files to match the production buffer allocation refactoring while preserving logging capabilities.

---

## Task Groups

Each task group corresponds to ONE file. The implementation should process files sequentially, ensuring each file is complete before moving to the next.

---

## Task Group 1: backwards_euler.py

**Production File:** `src/cubie/integrators/algorithms/backwards_euler.py`
**Instrumented File:** `tests/integrators/algorithms/instrumented/backwards_euler.py`

### Component Description

The BackwardsEulerStep is a single-stage implicit method using Newton-Krylov iteration. The production version uses:
- `buffer_registry.get_child_allocators()` for solver scratch buffers
- Conditional kwargs passing to parent `__init__`
- No `register_buffers()` method (simple algorithm)

### Changes Required

1. **Remove `build_implicit_helpers()` override** - Production does not override this method; the parent handles solver creation via `solver.update()`. However, instrumented needs to create InstrumentedSolvers, so retain but match production solver setup pattern.

2. **Update `__init__` method**:
   - Match production's parameter handling exactly
   - Use conditional solver_kwargs dict pattern
   - Pass kwargs to parent `__init__`

3. **Update `build_step()` method**:
   - Replace `solver_shared_elements` with `buffer_registry.get_child_allocators()`
   - Use `alloc_solver_shared()` and `alloc_solver_persistent()` allocator functions
   - Keep logging parameters and logging code blocks
   - Use `cuda.local.array()` for logging buffers (already present)

4. **Import Updates**:
   - Add: `from cubie.buffer_registry import buffer_registry`

### Expected Behavior

- `step()` device function receives shared/persistent_local buffers
- Allocators slice buffers appropriately
- Instrumented solver is called with logging arrays
- Status codes flow correctly

---

## Task Group 2: crank_nicolson.py

**Production File:** `src/cubie/integrators/algorithms/crank_nicolson.py`
**Instrumented File:** `tests/integrators/algorithms/instrumented/crank_nicolson.py`

### Component Description

CrankNicolsonStep is a 2nd-order implicit method with embedded backward Euler error estimation. Production version:
- Has `CrankNicolsonStepConfig` attrs class with `dxdt_location` field
- Has `register_buffers()` method
- Uses `buffer_registry.get_allocator()` for cn_dxdt buffer
- Uses `buffer_registry.get_child_allocators()` for solver scratch

### Changes Required

1. **Add `CrankNicolsonStepConfig` class** matching production:
   - Inherit from `ImplicitStepConfig`
   - Add `dxdt_location` field with validator

2. **Update `__init__` method**:
   - Build config kwargs dict conditionally
   - Create `CrankNicolsonStepConfig` (not just `ImplicitStepConfig`)
   - Call `self.register_buffers()` after parent init

3. **Add `register_buffers()` method**:
   - Register `cn_dxdt` buffer with location and alias

4. **Update `build_step()` method**:
   - Use `buffer_registry.get_child_allocators()` for solver buffers
   - Use `buffer_registry.get_allocator('cn_dxdt', self)` for dxdt buffer
   - Keep instrumented solver calls with logging parameters

5. **Retain `build_implicit_helpers()`**:
   - Create InstrumentedLinearSolver and InstrumentedNewtonKrylov
   - Assign to `self.solver`

### Expected Behavior

- CN and BE solves use same solver instance
- dxdt buffer allocated according to `dxdt_location` config
- Logging captures both solver iterations

---

## Task Group 3: generic_dirk.py

**Production File:** `src/cubie/integrators/algorithms/generic_dirk.py`
**Instrumented File:** `tests/integrators/algorithms/instrumented/generic_dirk.py`

### Component Description

DIRKStep implements diagonally implicit Runge-Kutta methods. Production version:
- Has `DIRKStepConfig` with location fields: `stage_increment_location`, `stage_base_location`, `accumulator_location`, `stage_rhs_location`
- Has complex `register_buffers()` method with aliasing
- Uses multiple `buffer_registry.get_allocator()` calls

### Changes Required

1. **Update `DIRKStepConfig` class**:
   - Add `stage_rhs_location` field (missing in instrumented)
   - Add validators for location fields

2. **Update `__init__` method**:
   - Match production's conditional config_kwargs building
   - Match production's conditional solver_kwargs building
   - Remove direct solver config assignment to config
   - Call parent with solver_kwargs

3. **Add `register_buffers()` method**:
   - Match production buffer registrations exactly
   - Include `get_child_allocators()` for solver buffers
   - Register all algorithm buffers with correct locations and aliases

4. **Remove local buffer registration from `__init__`**:
   - Buffer registration should happen in `register_buffers()`

5. **Update `build_step()` method**:
   - Use `buffer_registry.get_child_allocators()` from parent's solver
   - Use `buffer_registry.get_allocator()` for all algorithm buffers
   - Keep instrumented solver calls with logging parameters

6. **Update `build_implicit_helpers()`**:
   - Match production's `solver.update()` pattern
   - Create InstrumentedSolvers and assign to `self.solver`

### Expected Behavior

- All DIRK stages execute with correct buffer allocation
- Stage loop uses allocated buffers correctly
- FSAL caching works with persistent buffers

---

## Task Group 4: generic_erk.py

**Production File:** `src/cubie/integrators/algorithms/generic_erk.py`
**Instrumented File:** `tests/integrators/algorithms/instrumented/generic_erk.py`

### Component Description

ERKStep implements explicit Runge-Kutta methods. Production version:
- Has `ERKStepConfig` with `stage_rhs_location`, `stage_accumulator_location` fields
- Has `register_buffers()` method
- No implicit solvers (explicit method)

### Changes Required

1. **Update `ERKStepConfig` class**:
   - Add validators for location fields

2. **Update `__init__` method**:
   - Match production's buffer registration pattern
   - Call `self.register_buffers()` after parent init (production pattern)
   - Remove inline buffer registration from `__init__`

3. **Add/Update `register_buffers()` method**:
   - Match production exactly
   - Clear existing registrations first
   - Register `stage_rhs`, `stage_accumulator` with persistent flag where appropriate

4. **Update `build_step()` method**:
   - Use `buffer_registry.get_allocator()` for all buffers
   - Keep logging parameters and logging code

5. **Add missing properties**:
   - `shared_memory_required`, `local_scratch_required`, `persistent_local_required` using buffer_registry

### Expected Behavior

- ERK stages execute with correct buffer allocation
- Stage accumulator properly sized for tableau
- FSAL optimization works with stage_cache

---

## Task Group 5: generic_firk.py

**Production File:** `src/cubie/integrators/algorithms/generic_firk.py`
**Instrumented File:** `tests/integrators/algorithms/instrumented/generic_firk.py`

### Component Description

FIRKStep implements fully implicit Runge-Kutta methods. Production version:
- Has `FIRKStepConfig` with location fields
- Has `register_buffers()` method
- Solves coupled n-stage system

### Changes Required

1. **Update `FIRKStepConfig` class**:
   - Add validators for location fields

2. **Update `__init__` method**:
   - Match production's conditional kwargs building
   - Remove direct solver config from config kwargs
   - Call parent with solver_kwargs
   - Call `self.register_buffers()` after parent init

3. **Add `register_buffers()` method**:
   - Match production buffer registrations
   - Register `stage_increment`, `stage_driver_stack`, `stage_state`

4. **Update `build_step()` method**:
   - Use `buffer_registry.get_allocator()` for all buffers
   - Use `buffer_registry.get_child_allocators()` for solver buffers
   - Keep instrumented solver calls with logging parameters

5. **Update `build_implicit_helpers()`**:
   - Match production's helper retrieval
   - Create InstrumentedSolvers with all_stages_n dimension

### Expected Behavior

- All FIRK stages solved simultaneously
- Stage driver stack properly populated
- Coupled solver receives correct dimension

---

## Task Group 6: generic_rosenbrock_w.py

**Production File:** `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
**Instrumented File:** `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`

### Component Description

GenericRosenbrockWStep implements Rosenbrock-W methods. Production version:
- Has `RosenbrockWStepConfig` with location fields and device function fields
- Has `register_buffers()` method with dynamic cached_auxiliaries update
- Uses linear solver only (no Newton)

### Changes Required

1. **Update `RosenbrockWStepConfig` class**:
   - Add validators for location fields
   - Add device function fields matching production

2. **Update `__init__` method**:
   - Match production's conditional config_kwargs building
   - Remove solver configs from config kwargs
   - Call parent with solver_kwargs and solver_type='linear'
   - Call `self.register_buffers()` after parent init

3. **Add `register_buffers()` method**:
   - Match production buffer registrations
   - Include `cached_auxiliaries` with initial 0 size
   - Include `stage_increment` with persistence and alias

4. **Update `build_implicit_helpers()`**:
   - Match production's helper retrieval
   - Use `buffer_registry.update_buffer()` for cached_auxiliaries size
   - Create InstrumentedLinearSolver only
   - Assign helpers and solver to instance attributes

5. **Update `build_step()` method**:
   - Use `buffer_registry.get_allocator()` for all buffers
   - Access solver and helpers from instance attributes
   - Keep instrumented solver calls with logging parameters

6. **Update/Add `build()` method**:
   - Match production's build pattern if present
   - Or remove if production doesn't have it

### Expected Behavior

- Rosenbrock stages execute with linear solver
- Cached auxiliaries properly sized after helper build
- Jacobian preparation uses cached data

---

## Integration Points

### Buffer Registry

All files must:
- Import `from cubie.buffer_registry import buffer_registry`
- Use `buffer_registry.register()` for buffer allocation
- Use `buffer_registry.get_allocator()` for obtaining allocator functions
- Use `buffer_registry.get_child_allocators()` for parent-child solver relationships
- Use `buffer_registry.update_buffer()` for dynamic size updates

### Instrumented Solvers

All implicit method files must:
- Import from `.matrix_free_solvers` (relative import)
- Create `InstrumentedLinearSolver` and/or `InstrumentedNewtonKrylov`
- Assign to `self.solver` to replace parent's solver

### Logging Pattern

All step functions must:
- Accept additional logging array parameters
- Use `cuda.local.array()` for logging buffers
- Record state at specified points in algorithm

---

## Edge Cases

1. **Buffer aliasing**: When production uses aliases (e.g., `stage_base` aliases `accumulator`), instrumented must use same aliasing
2. **Conditional buffer sizes**: When buffer sizes depend on tableau properties (e.g., `stage_count - 1`), use same calculations
3. **Solver dimension**: FIRK uses `all_stages_n` dimension for solver; ensure instrumented uses same
4. **Linear-only solvers**: Rosenbrock uses only linear solver, not Newton

---

## Dependencies

- `cubie.buffer_registry` - Buffer allocation system
- `cubie.integrators.algorithms.ode_implicitstep.ODEImplicitStep` - Parent class for implicit methods
- `cubie.integrators.algorithms.ode_explicitstep.ODEExplicitStep` - Parent class for explicit methods
- `tests/integrators/algorithms/instrumented/matrix_free_solvers.py` - Instrumented solver implementations
