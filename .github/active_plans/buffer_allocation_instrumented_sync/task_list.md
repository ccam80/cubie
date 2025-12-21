# Implementation Task List
# Feature: Buffer Allocation Instrumented Sync
# Plan Reference: .github/active_plans/buffer_allocation_instrumented_sync/agent_plan.md

## Task Group 1: backwards_euler.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler.py (lines 1-327) - Complete source reference
- File: tests/integrators/algorithms/instrumented/backwards_euler.py (entire file)

**Input Validation Required**:
- None (pattern replication task)

**Tasks**:
1. **Add buffer_registry import**
   - File: tests/integrators/algorithms/instrumented/backwards_euler.py
   - Action: Modify
   - Details:
     ```python
     # Add import after existing cubie imports (around line 9):
     from cubie.buffer_registry import buffer_registry
     ```
   - Edge cases: Import ordering must follow existing pattern
   - Integration: Required for subsequent allocator usage

2. **Add child allocator acquisition in build_step()**
   - File: tests/integrators/algorithms/instrumented/backwards_euler.py
   - Action: Modify
   - Details:
     ```python
     # In build_step(), after solver_fn assignment (around line 240), add:
     alloc_solver_shared, alloc_solver_persistent = (
         buffer_registry.get_child_allocators(self, self.solver,
                                              name='solver_scratch')
     )
     ```
   - Edge cases: Must reference `self.solver` which is set in `build_implicit_helpers()`
   - Integration: Provides allocators for step function closure

3. **Remove solver_shared_elements capture**
   - File: tests/integrators/algorithms/instrumented/backwards_euler.py
   - Action: Modify
   - Details:
     ```python
     # Remove line 237:
     # solver_shared_elements = self.solver_shared_elements
     ```
   - Edge cases: None
   - Integration: This variable is replaced by allocator pattern

4. **Replace manual slicing with allocator calls in step()**
   - File: tests/integrators/algorithms/instrumented/backwards_euler.py
   - Action: Modify
   - Details:
     ```python
     # Replace line 357 (solver_scratch = shared[: solver_shared_elements]):
     solver_scratch = alloc_solver_shared(shared, persistent_local)
     solver_persistent = alloc_solver_persistent(shared, persistent_local)
     ```
   - Edge cases: solver_persistent is new parameter for solver call
   - Integration: Must also update solver_fn call to include solver_persistent

5. **Update solver_fn call to include solver_persistent**
   - File: tests/integrators/algorithms/instrumented/backwards_euler.py
   - Action: Modify
   - Details:
     ```python
     # Update the solver_fn call (around lines 386-407) to include solver_persistent
     # after solver_scratch and before counters:
     status = solver_fn(
         proposed_state,
         parameters,
         proposed_drivers,
         next_time,
         dt_scalar,
         a_ij,
         state,
         solver_scratch,
         solver_persistent,  # ADD THIS LINE
         counters,
         int32(0),
         newton_initial_guesses,
         # ... rest of logging arrays
     )
     ```
   - Edge cases: Parameter order must match source exactly
   - Integration: solver_persistent comes after solver_scratch, before counters

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: crank_nicolson.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/crank_nicolson.py (lines 1-401) - Complete source reference
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py (entire file)

**Input Validation Required**:
- None (pattern replication task)

**Tasks**:
1. **Add buffer_registry and attrs imports**
   - File: tests/integrators/algorithms/instrumented/crank_nicolson.py
   - Action: Modify
   - Details:
     ```python
     # Add after line 4 (existing imports):
     import attrs
     from cubie.buffer_registry import buffer_registry
     ```
   - Edge cases: Import ordering must follow existing pattern
   - Integration: Required for config class and allocator usage

2. **Add CrankNicolsonStepConfig class**
   - File: tests/integrators/algorithms/instrumented/crank_nicolson.py
   - Action: Modify
   - Details:
     ```python
     # Add after CN_DEFAULTS (around line 43), before CrankNicolsonStep class:
     @attrs.define
     class CrankNicolsonStepConfig(ImplicitStepConfig):
         """Configuration for Crank-Nicolson step."""
         
         dxdt_location: str = attrs.field(
             default='local',
             validator=attrs.validators.in_(['local', 'shared'])
         )
     ```
   - Edge cases: Must extend ImplicitStepConfig
   - Integration: Used in __init__ to build config

3. **Update __init__ to use new config and call register_buffers**
   - File: tests/integrators/algorithms/instrumented/crank_nicolson.py
   - Action: Modify
   - Details:
     ```python
     # In __init__, around line 107, change ImplicitStepConfig to:
     # Build config kwargs conditionally
     config_kwargs = {
         'precision': precision,
         'n': n,
         'get_solver_helper_fn': get_solver_helper_fn,
         'beta': beta,
         'gamma': gamma,
         'M': M,
         'dxdt_function': dxdt_function,
         'observables_function': observables_function,
         'driver_function': driver_function,
     }
     if preconditioner_order is not None:
         config_kwargs['preconditioner_order'] = preconditioner_order
     
     config = CrankNicolsonStepConfig(**config_kwargs)
     
     # After super().__init__ call, add:
     self.register_buffers()
     ```
   - Edge cases: Must add dxdt_location parameter to __init__ signature
   - Integration: register_buffers depends on compile_settings being set

4. **Add register_buffers method**
   - File: tests/integrators/algorithms/instrumented/crank_nicolson.py
   - Action: Modify
   - Details:
     ```python
     # Add after __init__ (before build_implicit_helpers):
     def register_buffers(self) -> None:
         """Register buffers with buffer_registry."""
         config = self.compile_settings
         buffer_registry.register(
             'cn_dxdt',
             self,
             config.n,
             config.dxdt_location,
             aliases='solver_scratch_shared',
             precision=config.precision
         )
     ```
   - Edge cases: Uses config.dxdt_location which defaults to 'local'
   - Integration: Called from __init__ after super()

5. **Add child allocator acquisition and dxdt allocator in build_step**
   - File: tests/integrators/algorithms/instrumented/crank_nicolson.py
   - Action: Modify
   - Details:
     ```python
     # In build_step(), before the @cuda.jit decorator, add:
     # Get child allocators for Newton solver
     alloc_solver_shared, alloc_solver_persistent = (
         buffer_registry.get_child_allocators(self, self.solver,
                                              name='solver_scratch')
     )
     alloc_dxdt = buffer_registry.get_allocator('cn_dxdt', self)
     ```
   - Edge cases: self.solver must exist (set in build_implicit_helpers)
   - Integration: Provides allocators for step function closure

6. **Replace manual slicing with allocator calls in step**
   - File: tests/integrators/algorithms/instrumented/crank_nicolson.py
   - Action: Modify
   - Details:
     ```python
     # Replace lines 377-379:
     # solver_scratch = shared[:solver_shared_elements]
     # dxdt_buffer = solver_scratch[:n]
     # With:
     solver_scratch = alloc_solver_shared(shared, persistent_local)
     solver_persistent = alloc_solver_persistent(shared, persistent_local)
     dxdt = alloc_dxdt(shared, persistent_local)
     
     # And update dxdt_buffer references to dxdt throughout step()
     ```
   - Edge cases: Must update all references to dxdt_buffer
   - Integration: solver_persistent added to solver calls

7. **Update both solver_fn calls to include solver_persistent**
   - File: tests/integrators/algorithms/instrumented/crank_nicolson.py
   - Action: Modify
   - Details:
     ```python
     # First solver call (around line 405) - add solver_persistent after solver_scratch:
     status = solver_fn(
         proposed_state,
         parameters,
         proposed_drivers,
         end_time,
         dt_scalar,
         stage_coefficient,
         base_state,
         solver_scratch,
         solver_persistent,  # ADD
         counters,
         ...
     )
     
     # Second solver call (around line 442) - same pattern:
     be_status = solver_fn(
         base_state,
         parameters,
         proposed_drivers,
         end_time,
         dt_scalar,
         be_coefficient,
         state,
         solver_scratch,
         solver_persistent,  # ADD
         counters,
         ...
     )
     ```
   - Edge cases: Both calls need update
   - Integration: Parameter order matches source

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: explicit_euler.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/explicit_euler.py (lines 1-229) - Complete source reference
- File: tests/integrators/algorithms/instrumented/explicit_euler.py (entire file)

**Input Validation Required**:
- None (verification only)

**Tasks**:
1. **Verify no changes needed**
   - File: tests/integrators/algorithms/instrumented/explicit_euler.py
   - Action: Verify (no modification)
   - Details:
     ```python
     # Explicit Euler has no solver buffers - no buffer allocation changes in source
     # Compare source and instrumented - only logging additions should differ
     # The instrumented version should already match source buffer patterns
     ```
   - Edge cases: Source has no buffer_registry usage
   - Integration: None needed - explicit methods don't use buffer_registry

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: generic_dirk.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 1-842) - Complete source reference
- File: tests/integrators/algorithms/instrumented/generic_dirk.py (entire file)

**Input Validation Required**:
- None (pattern replication task)

**Tasks**:
1. **Add stage_rhs_location field to DIRKStepConfig**
   - File: tests/integrators/algorithms/instrumented/generic_dirk.py
   - Action: Modify
   - Details:
     ```python
     # In DIRKStepConfig class (around line 57-66), add after accumulator_location:
     stage_rhs_location: str = attrs.field(
         default='local',
         validator=attrs.validators.in_(['local', 'shared'])
     )
     ```
   - Edge cases: Must add validator
   - Integration: Used in register_buffers

2. **Move buffer registration to register_buffers method**
   - File: tests/integrators/algorithms/instrumented/generic_dirk.py
   - Action: Modify
   - Details:
     ```python
     # Remove buffer registration code from __init__ (lines 130-180)
     # Add new method after __init__:
     def register_buffers(self) -> None:
         """Register buffers according to locations in compile settings."""
         config = self.compile_settings
         precision = config.precision
         n = config.n
         tableau = config.tableau

         # Clear any existing buffer registrations
         buffer_registry.clear_parent(self)

         # Calculate buffer sizes
         accumulator_length = max(tableau.stage_count - 1, 0) * n

         # Register solver scratch and solver persistent buffers so they can
         # be aliased
         _ = buffer_registry.get_child_allocators(
                 self,
                 self.solver,
                 name='solver'
         )

         # Register buffers
         buffer_registry.register(
             'stage_increment',
             self,
             n,
             config.stage_increment_location,
             persistent=True,
             precision=precision
         )
         buffer_registry.register(
             'accumulator',
             self,
             accumulator_length,
             config.accumulator_location,
             precision=precision
         )

         buffer_registry.register(
             'stage_base',
             self,
             n,
             config.stage_base_location,
             aliases='accumulator',
             precision=precision
         )

         buffer_registry.register(
             'stage_rhs',
             self,
             n,
             config.stage_rhs_location,
             persistent=True,
             precision=precision
         )
     ```
   - Edge cases: Uses shorter buffer names (no dirk_ prefix)
   - Integration: Called from __init__ after super()

3. **Update __init__ to call register_buffers and add location parameters**
   - File: tests/integrators/algorithms/instrumented/generic_dirk.py
   - Action: Modify
   - Details:
     ```python
     # Add stage_rhs_location parameter to __init__ signature
     # Add to config_kwargs:
     if stage_rhs_location is not None:
         config_kwargs["stage_rhs_location"] = stage_rhs_location
     
     # After super().__init__ call, add:
     self.register_buffers()
     ```
   - Edge cases: Must remove old buffer registration code
   - Integration: register_buffers requires self.solver to exist

4. **Update build_step to use new allocator pattern**
   - File: tests/integrators/algorithms/instrumented/generic_dirk.py
   - Action: Modify
   - Details:
     ```python
     # Replace allocator acquisitions (around lines 336-354) with:
     # Get child allocators for Newton solver
     alloc_solver_shared, alloc_solver_persistent = (
         buffer_registry.get_child_allocators(self, self._newton_solver,
                                              name='solver')
     )

     # Get allocators from buffer registry
     getalloc = buffer_registry.get_allocator
     alloc_stage_increment = getalloc('stage_increment', self)
     alloc_accumulator = getalloc('accumulator', self)
     alloc_stage_base = getalloc('stage_base', self)
     alloc_stage_rhs = getalloc('stage_rhs', self)
     ```
   - Edge cases: Uses `self._newton_solver` not `self.solver` - check which is correct
   - Integration: Allocator names match registered buffer names

5. **Update step() allocator calls and solver parameter**
   - File: tests/integrators/algorithms/instrumented/generic_dirk.py
   - Action: Modify
   - Details:
     ```python
     # In step() function, update allocator calls (around lines 476-479):
     stage_increment = alloc_stage_increment(shared, persistent_local)
     stage_accumulator = alloc_accumulator(shared, persistent_local)
     stage_base = alloc_stage_base(shared, persistent_local)
     solver_shared = alloc_solver_shared(shared, persistent_local)
     solver_persistent = alloc_solver_persistent(shared, persistent_local)
     stage_rhs = alloc_stage_rhs(shared, persistent_local)
     
     # Remove manual slicing like:
     # stage_rhs = solver_scratch[:n]
     # increment_cache = solver_scratch[n:int32(2)*n]
     # rhs_cache = solver_scratch[:n]
     
     # Update nonlinear_solver calls to include solver_persistent parameter
     ```
   - Edge cases: FSAL cache handling may need adjustment
   - Integration: solver_persistent added to solver calls

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: generic_erk.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 1-595) - Complete source reference
- File: tests/integrators/algorithms/instrumented/generic_erk.py (entire file)

**Input Validation Required**:
- None (pattern replication task)

**Tasks**:
1. **Move buffer registration to register_buffers method**
   - File: tests/integrators/algorithms/instrumented/generic_erk.py
   - Action: Modify
   - Details:
     ```python
     # Remove buffer registration from __init__ (lines 103-137)
     # Add new method after __init__:
     def register_buffers(self) -> None:
         """Register buffers with buffer_registry."""
         config = self.compile_settings
         precision = config.precision
         n = config.n
         tableau = config.tableau

         # Clear any existing buffer registrations
         buffer_registry.clear_parent(self)

         # Calculate buffer sizes
         accumulator_length = max(tableau.stage_count - 1, 0) * n

         # Register algorithm buffers using config values
         buffer_registry.register(
             'stage_rhs',
             self,
             n,
             config.stage_rhs_location,
             persistent=True,
             precision=precision
         )
         buffer_registry.register(
             'stage_accumulator',
             self,
             accumulator_length,
             config.stage_accumulator_location,
             precision=precision
         )
     ```
   - Edge cases: Source uses shorter buffer names (no erk_ prefix)
   - Integration: Called after super().__init__

2. **Update __init__ to call register_buffers after super**
   - File: tests/integrators/algorithms/instrumented/generic_erk.py
   - Action: Modify
   - Details:
     ```python
     # After super().__init__(config, defaults) call (around line 144), add:
     self.register_buffers()
     ```
   - Edge cases: Remove old buffer registration code
   - Integration: register_buffers uses self.compile_settings

3. **Update build_step allocator acquisitions**
   - File: tests/integrators/algorithms/instrumented/generic_erk.py
   - Action: Modify
   - Details:
     ```python
     # Replace allocator acquisitions (around lines 197-203) with:
     # Get allocators from buffer registry
     getalloc = buffer_registry.get_allocator
     alloc_stage_rhs = getalloc('stage_rhs', self)
     alloc_stage_accumulator = getalloc('stage_accumulator', self)
     
     # Remove alloc_stage_cache if not in source pattern
     ```
   - Edge cases: FSAL cache handling differs from source
   - Integration: Allocator names match registered buffers

4. **Update step() allocator calls**
   - File: tests/integrators/algorithms/instrumented/generic_erk.py
   - Action: Modify
   - Details:
     ```python
     # In step() function, update allocator calls (around lines 313-317):
     stage_rhs = alloc_stage_rhs(shared, persistent_local)
     stage_accumulator = alloc_stage_accumulator(shared, persistent_local)
     
     # Adjust FSAL cache handling to match source pattern
     ```
   - Edge cases: Source has different FSAL logic
   - Integration: Must preserve all LOGGING blocks

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 6: generic_firk.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 1-654) - Complete source reference
- File: tests/integrators/algorithms/instrumented/generic_firk.py (entire file)

**Input Validation Required**:
- None (pattern replication task)

**Tasks**:
1. **Move buffer registration to register_buffers method**
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # Remove buffer registration from __init__ (lines 142-168)
     # Add new method after __init__:
     def register_buffers(self) -> None:
         """Register buffers according to locations in compile settings."""
         config = self.compile_settings
         precision = config.precision
         n = config.n
         tableau = config.tableau
         
         # Clear any existing buffer registrations
         buffer_registry.clear_parent(self)

         # Calculate buffer sizes
         all_stages_n = tableau.stage_count * n
         stage_driver_stack_elements = tableau.stage_count * config.n_drivers

         # Register algorithm buffers using config values
         buffer_registry.register(
             'stage_increment',
             self,
             all_stages_n,
             config.stage_increment_location,
             persistent=True,
             precision=precision
         )
         buffer_registry.register(
             'stage_driver_stack', self, stage_driver_stack_elements,
             config.stage_driver_stack_location, precision=precision
         )
         buffer_registry.register(
             'stage_state', self, n, config.stage_state_location,
             precision=precision
         )
     ```
   - Edge cases: Source uses shorter buffer names (no firk_ prefix)
   - Integration: Remove firk_solver_scratch registration

2. **Update __init__ to call register_buffers after super**
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # After super().__init__ call (around line 184), add:
     self.register_buffers()
     
     # Remove old buffer registration code
     ```
   - Edge cases: Must remove firk_solver_scratch registration
   - Integration: register_buffers uses self.compile_settings

3. **Update build_step allocator acquisitions**
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # Replace allocator acquisitions (around lines 322-334) with:
     # Get allocators from buffer registry
     getalloc = buffer_registry.get_allocator
     alloc_stage_increment = getalloc('stage_increment', self)
     alloc_stage_driver_stack = getalloc('stage_driver_stack', self)
     alloc_stage_state = getalloc('stage_state', self)
     
     # Get child allocators for Newton solver
     alloc_solver_shared, alloc_solver_persistent = (
         buffer_registry.get_child_allocators(self, nonlinear_solver)
     )
     ```
   - Edge cases: Child allocators from nonlinear_solver parameter
   - Integration: Must update step() accordingly

4. **Update step() allocator calls**
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # In step() function, update allocator calls (around lines 413-417):
     stage_state = alloc_stage_state(shared, persistent_local)
     solver_shared = alloc_solver_shared(shared, persistent_local)
     solver_persistent = alloc_solver_persistent(shared, persistent_local)
     stage_increment = alloc_stage_increment(shared, persistent_local)
     stage_driver_stack = alloc_stage_driver_stack(shared, persistent_local)
     
     # Remove alloc_solver_scratch usage
     ```
   - Edge cases: solver_persistent is new
   - Integration: Update solver call to include solver_persistent

5. **Update solver call to include solver_persistent**
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # Update nonlinear_solver call (around line 457) to include solver_persistent:
     solver_ret = nonlinear_solver(
         stage_increment,
         parameters,
         stage_driver_stack,
         current_time,
         dt_scalar,
         typed_zero,
         state,
         solver_shared,
         solver_persistent,  # ADD
         counters,
         int32(0),
         newton_initial_guesses,
         ...
     )
     ```
   - Edge cases: Parameter order must match source
   - Integration: solver_persistent after solver_shared

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 7: generic_rosenbrock_w.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 1-829) - Complete source reference
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py (entire file)

**Input Validation Required**:
- None (pattern replication task)

**Tasks**:
1. **Move buffer registration to register_buffers method**
   - File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # Remove buffer registration from __init__ (lines 123-156)
     # Add new method after __init__:
     def register_buffers(self) -> None:
         """Register buffers according to locations in compile settings."""
         config = self.compile_settings
         precision = config.precision
         n = config.n
         tableau = config.tableau

         # Clear any existing buffer registrations
         buffer_registry.clear_parent(self)

         # Calculate buffer sizes
         stage_store_elements = tableau.stage_count * n

         # Register algorithm buffers using config values
         buffer_registry.register(
             'stage_rhs', self, n, config.stage_rhs_location,
             precision=precision
         )
         buffer_registry.register(
             'stage_store', self, stage_store_elements,
             config.stage_store_location, precision=precision
         )
         # cached_auxiliaries registered with 0 size; updated in build_implicit_helpers
         buffer_registry.register(
             'cached_auxiliaries', self, 0,
             config.cached_auxiliaries_location, precision=precision
         )

         # Stage increment should persist between steps for initial guess
         buffer_registry.register(
             'stage_increment', self, n,
             config.stage_store_location,
             aliases='stage_store',
             persistent=True,
             precision=precision
         )
     ```
   - Edge cases: Source uses shorter buffer names (no rosenbrock_ prefix)
   - Integration: Called after super().__init__

2. **Update __init__ to call register_buffers after super**
   - File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # After super().__init__ call (around line 167), add:
     self.register_buffers()
     
     # Remove old buffer registration code
     ```
   - Edge cases: Must remove rosenbrock_ prefixed registrations
   - Integration: register_buffers uses self.compile_settings

3. **Update build_implicit_helpers to use short buffer names**
   - File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # Update buffer_registry.update_buffer call (around line 215-218):
     buffer_registry.update_buffer(
         'cached_auxiliaries', self,  # Changed from 'rosenbrock_cached_auxiliaries'
         size=self._cached_auxiliary_count
     )
     ```
   - Edge cases: Buffer name must match registration
   - Integration: Updates size after getting cached_aux_count

4. **Update build_step allocator acquisitions**
   - File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # Replace allocator acquisitions (around lines 329-340) with:
     # Get allocators from buffer registry
     getalloc = buffer_registry.get_allocator
     alloc_stage_rhs = getalloc('stage_rhs', self)
     alloc_stage_store = getalloc('stage_store', self)
     alloc_cached_auxiliaries = getalloc('cached_auxiliaries', self)
     alloc_stage_increment = getalloc('stage_increment', self)
     ```
   - Edge cases: Uses short buffer names
   - Integration: Allocator names match registered buffers

5. **Update step() allocator calls**
   - File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # In step() function, update allocator calls (around lines 441-443):
     stage_rhs = alloc_stage_rhs(shared, persistent_local)
     stage_store = alloc_stage_store(shared, persistent_local)
     cached_auxiliaries = alloc_cached_auxiliaries(shared, persistent_local)
     stage_increment = alloc_stage_increment(shared, persistent_local)
     ```
   - Edge cases: Must preserve stage_increment usage pattern
   - Integration: Must preserve all LOGGING blocks

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

| Group | File | Status | Complexity |
|-------|------|--------|------------|
| 1 | backwards_euler.py | [ ] | Medium |
| 2 | crank_nicolson.py | [ ] | High |
| 3 | explicit_euler.py | [ ] | None (verify only) |
| 4 | generic_dirk.py | [ ] | High |
| 5 | generic_erk.py | [ ] | Medium |
| 6 | generic_firk.py | [ ] | Medium |
| 7 | generic_rosenbrock_w.py | [ ] | Medium |

### Dependency Chain
All task groups are independent and can be executed in parallel.

### Parallel Execution Opportunities
All 7 task groups can be executed in parallel since they modify different files.

### Estimated Complexity
- **Low**: explicit_euler.py (verification only)
- **Medium**: backwards_euler.py, generic_erk.py, generic_firk.py, generic_rosenbrock_w.py
- **High**: crank_nicolson.py (needs new config class), generic_dirk.py (complex buffer aliasing)
