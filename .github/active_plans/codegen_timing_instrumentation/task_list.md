# Implementation Task List
# Feature: Codegen and Parsing Timing Instrumentation
# Plan Reference: .github/active_plans/codegen_timing_instrumentation/agent_plan.md

## Task Group 1: Core Timing Infrastructure in symbolicODE.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/odesystems/symbolic/symbolicODE.py (lines 1-596, entire file)
- File: src/cubie/time_logger.py (entire file for TimeLogger API)
- File: .github/active_plans/codegen_timing_instrumentation/agent_plan.md (sections on SymbolicODE instrumentation)

**Input Validation Required**:
None - this task adds instrumentation only, no validation changes required

**Tasks**:
1. **Add TimeLogger import to symbolicODE.py**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     - Add import after existing imports (after line 46):
       ```python
       from cubie.time_logger import _default_logger
       ```
   - Edge cases: None
   - Integration: Required for all subsequent timing calls in this module

2. **Add module-level tracking variables**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     - Add after imports, before the `create_ODE_system` function definition (around line 48):
       ```python
       # Event registration tracking for solver helpers
       _registered_helper_events = set()
       # One-time registration flag for parsing event
       _parsing_event_registered = False
       ```
   - Edge cases: Thread safety not required per architecture
   - Integration: Used by create() and get_solver_helper() methods

3. **Instrument SymbolicODE.create() classmethod for parsing timing**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     - In `create()` classmethod (lines 212-285):
     - Add registration check and event registration after line 261 (after driver processing):
       ```python
       global _parsing_event_registered
       if not _parsing_event_registered:
           _default_logger._register_event(
               "symbolic_ode_parsing",
               "codegen",
               "Codegen time for symbolic ODE parsing: "
           )
           _parsing_event_registered = True
       ```
     - Add timing start after registration, before line 268 (`sys_components = parse_input(...)`):
       ```python
       _default_logger.start_event("symbolic_ode_parsing")
       ```
     - Add timing stop after line 279 (after `return cls(...)`), just before the return:
       ```python
       symbolic_ode = cls(equations=equations,
                          all_indexed_bases=index_map,
                          all_symbols=all_symbols,
                          name=name,
                          fn_hash=int(fn_hash),
                          user_functions = functions,
                          precision=precision)
       _default_logger.stop_event("symbolic_ode_parsing")
       return symbolic_ode
       ```
   - Edge cases:
     - Multiple calls to create() should reuse registered event
     - Global flag ensures registration happens only once
   - Integration: Times complete parsing from parse_input() through SymbolicODE construction

4. **Instrument get_solver_helper() method**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     - In `get_solver_helper()` method (lines 377-595):
     - Add event registration after line 434 (after solver_updates), before try block:
       ```python
       # Register timing event for this helper type if not already registered
       event_name = f"solver_helper_{func_type}"
       global _registered_helper_events
       if event_name not in _registered_helper_events:
           _default_logger._register_event(
               event_name,
               "codegen",
               f"Codegen time for solver helper {func_type}: "
           )
           _registered_helper_events.add(event_name)
       ```
     - Add timing start after registration and the try-except block (line 439):
       ```python
       _default_logger.start_event(event_name)
       ```
     - Add timing stop before each return statement:
       - Before line 437 (cached return): Add stop before return
       - Before line 486 (cached_aux_count return): Add stop before return
       - Before line 595 (final return): Add stop before return
       ```python
       _default_logger.stop_event(event_name)
       return func  # or appropriate return value
       ```
   - Edge cases:
     - Different func_types get different event names
     - Events registered lazily on first use of each func_type
     - "cached_aux_count" returns int, not function - still timed
   - Integration: Wraps entire helper generation including codegen function calls

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 2: dxdt.py Codegen Timing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/dxdt.py (entire file)
- File: src/cubie/time_logger.py (entire file for TimeLogger API)
- File: .github/active_plans/codegen_timing_instrumentation/agent_plan.md (section on dxdt.py instrumentation)

**Input Validation Required**:
None - this task adds instrumentation only, no validation changes required

**Tasks**:
1. **Add TimeLogger import to dxdt.py**
   - File: src/cubie/odesystems/symbolic/codegen/dxdt.py
   - Action: Modify
   - Details:
     - Add import after existing imports (around line 14):
       ```python
       from cubie.time_logger import _default_logger
       ```
   - Edge cases: None
   - Integration: Required for timing calls in this module

2. **Register events at module level**
   - File: src/cubie/odesystems/symbolic/codegen/dxdt.py
   - Action: Modify
   - Details:
     - Add after imports, before DXDT_TEMPLATE (around line 16):
       ```python
       # Register timing events for codegen functions
       _default_logger._register_event(
           "codegen_generate_dxdt_fac_code",
           "codegen",
           "Codegen time for generate_dxdt_fac_code: "
       )
       _default_logger._register_event(
           "codegen_generate_observables_fac_code",
           "codegen",
           "Codegen time for generate_observables_fac_code: "
       )
       ```
   - Edge cases: Module imported once, registration happens once
   - Integration: Events available for all calls to these functions

3. **Instrument generate_dxdt_fac_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/dxdt.py
   - Action: Modify
   - Details:
     - Find `generate_dxdt_fac_code()` function (around line 131)
     - Add timing start as first line of function body:
       ```python
       def generate_dxdt_fac_code(
           equations: ParsedEquations,
           index_map: IndexedBases,
           func_name: str = "dxdt_factory",
           cse: bool = True,
       ) -> str:
           _default_logger.start_event("codegen_generate_dxdt_fac_code")
           # ... existing implementation ...
       ```
     - Add timing stop before the return statement:
       ```python
           _default_logger.stop_event("codegen_generate_dxdt_fac_code")
           return DXDT_TEMPLATE.format(...)
       ```
   - Edge cases: May be called multiple times for different systems
   - Integration: Called from SymbolicODE.build()

4. **Instrument generate_observables_fac_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/dxdt.py
   - Action: Modify
   - Details:
     - Find `generate_observables_fac_code()` function (around line 149)
     - Add timing start as first line of function body:
       ```python
       def generate_observables_fac_code(
           equations: ParsedEquations,
           index_map: IndexedBases,
           func_name: str = "observables_factory",
           cse: bool = True,
       ) -> str:
           _default_logger.start_event("codegen_generate_observables_fac_code")
           # ... existing implementation ...
       ```
     - Add timing stop before the return statement:
       ```python
           _default_logger.stop_event("codegen_generate_observables_fac_code")
           return OBSERVABLES_TEMPLATE.format(...)
       ```
   - Edge cases: May be called multiple times for different systems
   - Integration: Called from SymbolicODE.build()

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 3: linear_operators.py Codegen Timing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/linear_operators.py (entire file)
- File: src/cubie/time_logger.py (entire file for TimeLogger API)
- File: .github/active_plans/codegen_timing_instrumentation/agent_plan.md (section on linear_operators.py instrumentation)

**Input Validation Required**:
None - this task adds instrumentation only, no validation changes required

**Tasks**:
1. **Add TimeLogger import to linear_operators.py**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Add import after existing imports (around line 23):
       ```python
       from cubie.time_logger import _default_logger
       ```
   - Edge cases: None
   - Integration: Required for timing calls in this module

2. **Register events at module level**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Add after imports, before templates (around line 25):
       ```python
       # Register timing events for codegen functions
       _default_logger._register_event(
           "codegen_generate_operator_apply_code",
           "codegen",
           "Codegen time for generate_operator_apply_code: "
       )
       _default_logger._register_event(
           "codegen_generate_cached_operator_apply_code",
           "codegen",
           "Codegen time for generate_cached_operator_apply_code: "
       )
       _default_logger._register_event(
           "codegen_generate_prepare_jac_code",
           "codegen",
           "Codegen time for generate_prepare_jac_code: "
       )
       _default_logger._register_event(
           "codegen_generate_cached_jvp_code",
           "codegen",
           "Codegen time for generate_cached_jvp_code: "
       )
       _default_logger._register_event(
           "codegen_generate_n_stage_linear_operator_code",
           "codegen",
           "Codegen time for generate_n_stage_linear_operator_code: "
       )
       ```
   - Edge cases: Module imported once, registration happens once
   - Integration: Events available for all calls to these functions

3. **Instrument generate_operator_apply_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Find `generate_operator_apply_code()` function definition
     - Add timing start as first line of function body
     - Add timing stop before the return statement
     - Pattern:
       ```python
       def generate_operator_apply_code(...) -> str:
           _default_logger.start_event("codegen_generate_operator_apply_code")
           # ... existing implementation ...
           _default_logger.stop_event("codegen_generate_operator_apply_code")
           return result
       ```
   - Edge cases: Multiple calls for different systems
   - Integration: Called from SymbolicODE.get_solver_helper()

4. **Instrument generate_cached_operator_apply_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Add timing start and stop following same pattern as task 3
     - Event name: "codegen_generate_cached_operator_apply_code"
   - Edge cases: Multiple calls for different systems
   - Integration: Called from SymbolicODE.get_solver_helper()

5. **Instrument generate_prepare_jac_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Add timing start and stop following same pattern
     - Event name: "codegen_generate_prepare_jac_code"
     - Note: This function returns a tuple (code, aux_count)
   - Edge cases: Returns tuple, not just string
   - Integration: Called from SymbolicODE.get_solver_helper()

6. **Instrument generate_cached_jvp_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Add timing start and stop following same pattern
     - Event name: "codegen_generate_cached_jvp_code"
   - Edge cases: Multiple calls for different systems
   - Integration: Called from SymbolicODE.get_solver_helper()

7. **Instrument generate_n_stage_linear_operator_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Add timing start and stop following same pattern
     - Event name: "codegen_generate_n_stage_linear_operator_code"
   - Edge cases: Used for FIRK methods with multiple stages
   - Integration: Called from SymbolicODE.get_solver_helper()

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 4: preconditioners.py Codegen Timing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/preconditioners.py (entire file)
- File: src/cubie/time_logger.py (entire file for TimeLogger API)
- File: .github/active_plans/codegen_timing_instrumentation/agent_plan.md (section on preconditioners.py instrumentation)

**Input Validation Required**:
None - this task adds instrumentation only, no validation changes required

**Tasks**:
1. **Add TimeLogger import to preconditioners.py**
   - File: src/cubie/odesystems/symbolic/codegen/preconditioners.py
   - Action: Modify
   - Details:
     - Add import after existing imports (around line 23):
       ```python
       from cubie.time_logger import _default_logger
       ```
   - Edge cases: None
   - Integration: Required for timing calls in this module

2. **Register events at module level**
   - File: src/cubie/odesystems/symbolic/codegen/preconditioners.py
   - Action: Modify
   - Details:
     - Add after imports, before templates (around line 25):
       ```python
       # Register timing events for codegen functions
       _default_logger._register_event(
           "codegen_generate_neumann_preconditioner_code",
           "codegen",
           "Codegen time for generate_neumann_preconditioner_code: "
       )
       _default_logger._register_event(
           "codegen_generate_neumann_preconditioner_cached_code",
           "codegen",
           "Codegen time for generate_neumann_preconditioner_cached_code: "
       )
       _default_logger._register_event(
           "codegen_generate_n_stage_neumann_preconditioner_code",
           "codegen",
           "Codegen time for generate_n_stage_neumann_preconditioner_code: "
       )
       ```
   - Edge cases: Module imported once, registration happens once
   - Integration: Events available for all calls to these functions

3. **Instrument generate_neumann_preconditioner_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/preconditioners.py
   - Action: Modify
   - Details:
     - Find `generate_neumann_preconditioner_code()` function definition
     - Add timing start as first line of function body
     - Add timing stop before the return statement
     - Pattern:
       ```python
       def generate_neumann_preconditioner_code(...) -> str:
           _default_logger.start_event("codegen_generate_neumann_preconditioner_code")
           # ... existing implementation ...
           _default_logger.stop_event("codegen_generate_neumann_preconditioner_code")
           return result
       ```
   - Edge cases: Multiple calls for different systems
   - Integration: Called from SymbolicODE.get_solver_helper()

4. **Instrument generate_neumann_preconditioner_cached_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/preconditioners.py
   - Action: Modify
   - Details:
     - Add timing start and stop following same pattern as task 3
     - Event name: "codegen_generate_neumann_preconditioner_cached_code"
   - Edge cases: Multiple calls for different systems
   - Integration: Called from SymbolicODE.get_solver_helper()

5. **Instrument generate_n_stage_neumann_preconditioner_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/preconditioners.py
   - Action: Modify
   - Details:
     - Add timing start and stop following same pattern
     - Event name: "codegen_generate_n_stage_neumann_preconditioner_code"
   - Edge cases: Used for FIRK methods with multiple stages
   - Integration: Called from SymbolicODE.get_solver_helper()

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 5: nonlinear_residuals.py Codegen Timing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py (entire file)
- File: src/cubie/time_logger.py (entire file for TimeLogger API)
- File: .github/active_plans/codegen_timing_instrumentation/agent_plan.md (section on nonlinear_residuals.py instrumentation)

**Input Validation Required**:
None - this task adds instrumentation only, no validation changes required

**Tasks**:
1. **Add TimeLogger import to nonlinear_residuals.py**
   - File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py
   - Action: Modify
   - Details:
     - Add import after existing imports (around line 21):
       ```python
       from cubie.time_logger import _default_logger
       ```
   - Edge cases: None
   - Integration: Required for timing calls in this module

2. **Register events at module level**
   - File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py
   - Action: Modify
   - Details:
     - Add after imports, before templates (around line 23):
       ```python
       # Register timing events for codegen functions
       _default_logger._register_event(
           "codegen_generate_stage_residual_code",
           "codegen",
           "Codegen time for generate_stage_residual_code: "
       )
       _default_logger._register_event(
           "codegen_generate_n_stage_residual_code",
           "codegen",
           "Codegen time for generate_n_stage_residual_code: "
       )
       ```
   - Edge cases: Module imported once, registration happens once
   - Integration: Events available for all calls to these functions

3. **Instrument generate_stage_residual_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py
   - Action: Modify
   - Details:
     - Find `generate_stage_residual_code()` function definition
     - Add timing start as first line of function body
     - Add timing stop before the return statement
     - Pattern:
       ```python
       def generate_stage_residual_code(...) -> str:
           _default_logger.start_event("codegen_generate_stage_residual_code")
           # ... existing implementation ...
           _default_logger.stop_event("codegen_generate_stage_residual_code")
           return result
       ```
   - Edge cases: Multiple calls for different systems
   - Integration: Called from SymbolicODE.get_solver_helper()

4. **Instrument generate_n_stage_residual_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py
   - Action: Modify
   - Details:
     - Add timing start and stop following same pattern as task 3
     - Event name: "codegen_generate_n_stage_residual_code"
   - Edge cases: Used for FIRK methods with multiple stages
   - Integration: Called from SymbolicODE.get_solver_helper()

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 6: time_derivative.py Codegen Timing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/time_derivative.py (entire file)
- File: src/cubie/time_logger.py (entire file for TimeLogger API)
- File: .github/active_plans/codegen_timing_instrumentation/agent_plan.md (section on time_derivative.py instrumentation)

**Input Validation Required**:
None - this task adds instrumentation only, no validation changes required

**Tasks**:
1. **Add TimeLogger import to time_derivative.py**
   - File: src/cubie/odesystems/symbolic/codegen/time_derivative.py
   - Action: Modify
   - Details:
     - Add import after existing imports (around line 18):
       ```python
       from cubie.time_logger import _default_logger
       ```
   - Edge cases: None
   - Integration: Required for timing calls in this module

2. **Register event at module level**
   - File: src/cubie/odesystems/symbolic/codegen/time_derivative.py
   - Action: Modify
   - Details:
     - Add after imports, before TIME_DERIVATIVE_TEMPLATE (around line 21):
       ```python
       # Register timing event for codegen function
       _default_logger._register_event(
           "codegen_generate_time_derivative_fac_code",
           "codegen",
           "Codegen time for generate_time_derivative_fac_code: "
       )
       ```
   - Edge cases: Module imported once, registration happens once
   - Integration: Event available for all calls to this function

3. **Instrument generate_time_derivative_fac_code() function**
   - File: src/cubie/odesystems/symbolic/codegen/time_derivative.py
   - Action: Modify
   - Details:
     - Find `generate_time_derivative_fac_code()` function (line 140)
     - Add timing start as first line of function body:
       ```python
       def generate_time_derivative_fac_code(
           equations: ParsedEquations,
           index_map: IndexedBases,
           func_name: str = "time_derivative_rhs_factory",
           cse: bool = True,
       ) -> str:
           _default_logger.start_event("codegen_generate_time_derivative_fac_code")
           # ... existing implementation ...
       ```
     - Add timing stop before the return statement:
       ```python
           _default_logger.stop_event("codegen_generate_time_derivative_fac_code")
           return TIME_DERIVATIVE_TEMPLATE.format(...)
       ```
   - Edge cases: Multiple calls for different systems
   - Integration: Called from SymbolicODE.get_solver_helper()

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 6

**Dependency Chain**:
```
Task Group 1 (symbolicODE.py core infrastructure)
  ├─> Task Group 2 (dxdt.py)
  ├─> Task Group 3 (linear_operators.py)
  ├─> Task Group 4 (preconditioners.py)
  ├─> Task Group 5 (nonlinear_residuals.py)
  └─> Task Group 6 (time_derivative.py)
```

**Parallel Execution Opportunities**:
- Task Groups 2-6 can execute in parallel after Task Group 1 completes
- All are independent codegen modules with no cross-dependencies

**Implementation Notes**:
1. Task Group 1 must complete first as it establishes the timing infrastructure in symbolicODE.py
2. All subsequent groups are independent and can be executed concurrently
3. No breaking changes to function signatures or return values
4. Zero overhead when verbosity=None
5. All timing events use category "codegen"
6. Module-level registration ensures events are registered once per import
7. Helper timing uses lazy registration per func_type

**Expected Event Count**: 14-15+ timing events
- 1 parsing event (symbolic_ode_parsing)
- 2 events in dxdt.py
- 5 events in linear_operators.py
- 3 events in preconditioners.py
- 2 events in nonlinear_residuals.py
- 1 event in time_derivative.py
- N solver helper events (dynamic based on helper types used)

**Estimated Complexity**: Medium
- Straightforward instrumentation pattern applied consistently
- No algorithmic changes required
- Main complexity is ensuring all function entry/exit points are covered
- Testing will verify timing accuracy and no-op behavior
