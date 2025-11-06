# Iteration Counters - Remaining Implementation Tasks

## Overview
Complete the iteration counters feature implementation by adding counters parameter to step functions and implementing counter tracking in the integration loop.

## Task Group 1: Step Function Modifications - Main Source Files (PARALLEL)

**Dependencies**: None

### Task 1.1: backwards_euler_predict_correct.py (Main)
- [ ] **Complete**

**File**: `src/cubie/integrators/algorithms/backwards_euler_predict_correct.py`

**Changes**:
1. Add `int32` to imports if not already present
2. Add `counters` parameter to step device function signature (after persistent_local parameter)
3. Add `int32[:]` to the type signature tuple for counters
4. Pass `counters` to any solver_fn/nonlinear_solver calls

**Outcomes**:
_To be filled by do_task agent_

---

### Task 1.2: crank_nicolson.py (Main)
- [ ] **Complete**

**File**: `src/cubie/integrators/algorithms/crank_nicolson.py`

**Changes**:
1. Add `int32` to imports if not already present
2. Add `counters` parameter to step device function signature (after persistent_local parameter)
3. Add `int32[:]` to the type signature tuple for counters
4. Pass `counters` to any solver_fn/nonlinear_solver calls

**Outcomes**:
_To be filled by do_task agent_

---

### Task 1.3: generic_dirk.py (Main)
- [ ] **Complete**

**File**: `src/cubie/integrators/algorithms/generic_dirk.py`

**Changes**:
1. Add `int32` to imports if not already present
2. Add `counters` parameter to step device function signature (after persistent_local parameter)
3. Add `int32[:]` to the type signature tuple for counters
4. Pass `counters` to any solver_fn/nonlinear_solver calls

**Outcomes**:
_To be filled by do_task agent_

---

### Task 1.4: generic_firk.py (Main)
- [ ] **Complete**

**File**: `src/cubie/integrators/algorithms/generic_firk.py`

**Changes**:
1. Add `int32` to imports if not already present
2. Add `counters` parameter to step device function signature (after persistent_local parameter)
3. Add `int32[:]` to the type signature tuple for counters
4. Pass `counters` to any solver_fn/nonlinear_solver calls

**Outcomes**:
_To be filled by do_task agent_

---

### Task 1.5: generic_rosenbrock_w.py (Main)
- [ ] **Complete**

**File**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

**Changes**:
1. Add `int32` to imports if not already present
2. Add `counters` parameter to step device function signature (after persistent_local parameter)
3. Add `int32[:]` to the type signature tuple for counters
4. Pass `counters` to any solver_fn/nonlinear_solver calls

**Outcomes**:
_To be filled by do_task agent_

---

## Task Group 2: Step Function Modifications - Instrumented Files (PARALLEL)

**Dependencies**: None (independent of Group 1)

### Task 2.1: backwards_euler_predict_correct.py (Instrumented)
- [ ] **Complete**

**File**: `tests/integrators/algorithms/instrumented/backwards_euler_predict_correct.py`

**Changes**:
1. Add `int32` to imports if not already present
2. Add `counters` parameter to step device function signature (after persistent_local parameter)
3. Add `int32[:]` to the type signature tuple for counters
4. Pass `counters` to any solver_fn/nonlinear_solver calls

**Outcomes**:
_To be filled by do_task agent_

---

### Task 2.2: crank_nicolson.py (Instrumented)
- [ ] **Complete**

**File**: `tests/integrators/algorithms/instrumented/crank_nicolson.py`

**Changes**:
1. Add `int32` to imports if not already present
2. Add `counters` parameter to step device function signature (after persistent_local parameter)
3. Add `int32[:]` to the type signature tuple for counters
4. Pass `counters` to any solver_fn/nonlinear_solver calls

**Outcomes**:
_To be filled by do_task agent_

---

### Task 2.3: generic_dirk.py (Instrumented)
- [ ] **Complete**

**File**: `tests/integrators/algorithms/instrumented/generic_dirk.py`

**Changes**:
1. Add `int32` to imports if not already present
2. Add `counters` parameter to step device function signature (after persistent_local parameter)
3. Add `int32[:]` to the type signature tuple for counters
4. Pass `counters` to any solver_fn/nonlinear_solver calls

**Outcomes**:
_To be filled by do_task agent_

---

### Task 2.4: generic_firk.py (Instrumented)
- [ ] **Complete**

**File**: `tests/integrators/algorithms/instrumented/generic_firk.py`

**Changes**:
1. Add `int32` to imports if not already present
2. Add `counters` parameter to step device function signature (after persistent_local parameter)
3. Add `int32[:]` to the type signature tuple for counters
4. Pass `counters` to any solver_fn/nonlinear_solver calls

**Outcomes**:
_To be filled by do_task agent_

---

### Task 2.5: generic_rosenbrock_w.py (Instrumented)
- [ ] **Complete**

**File**: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`

**Changes**:
1. Add `int32` to imports if not already present
2. Add `counters` parameter to step device function signature (after persistent_local parameter)
3. Add `int32[:]` to the type signature tuple for counters
4. Pass `counters` to any solver_fn/nonlinear_solver calls

**Outcomes**:
_To be filled by do_task agent_

---

## Task Group 3: Integration Loop Counter Tracking (SEQUENTIAL)

**Dependencies**: Groups 1 and 2 must complete first

### Task 3.1: Integration Loop Counter Implementation
- [ ] **Complete**

**File**: `src/cubie/integrators/loops/ode_loop.py`

**Changes** in `IVPLoop.build()` method's loop_fn device function:

1. Add iteration_counters_output parameter to loop_fn signature

2. Implement conditional counter tracking based on flags.output_iteration_counters:
   
   **If NOT active**:
   - Before step call: `step_counters = cuda.local.array(0, int32)`
   
   **If active**:
   - Allocate: `counters_since_save = cuda.local.array(4, int32)` and initialize to 0
   - Before step call: `step_counters = cuda.local.array(2, int32)`
   - After step call:
     - Extract from step_counters: `counters_since_save[0] += step_counters[0]` (Newton)
     - Extract from step_counters: `counters_since_save[1] += step_counters[1]` (Krylov)
     - Track steps: `counters_since_save[2] += int32(1)`
     - Track rejections: `if not accept: counters_since_save[3] += int32(1)`
   - On save:
     - Pass counters_since_save to save_state (if needed for later)
     - Reset all 4 counters to 0

3. Pass step_counters to step_function call

**Outcomes**:
_To be filled by do_task agent_

---

## Execution Plan

1. Execute Task Group 1 in PARALLEL (5 tasks for main source files)
2. Execute Task Group 2 in PARALLEL (5 tasks for instrumented files)
3. Execute Task Group 3 SEQUENTIALLY after Groups 1 and 2 complete (1 task for integration loop)

## Notes

- All step function modifications follow the same pattern
- Integration loop changes are more complex and must be done after step functions are updated
- Counter tracking uses compile-time flags to avoid overhead when not needed
