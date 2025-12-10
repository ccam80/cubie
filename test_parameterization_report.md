# CuBIE Test Parameterization Analysis Report

**Date**: 2025-12-10

**Objective**: Analyze and consolidate `solver_settings_override` and `solver_settings_override2` parameterization to reduce test compilation time.

---

## Executive Summary

### Problem Statement

Test duration has exploded due to:
1. Long compile times for each session-scoped object
2. New compilation triggered for each unique parameter set
3. ~37 unique parameter combinations across the test suite
4. Estimated 20-40 minutes of pure compilation overhead per test run

### Key Findings

- **14 parameter sets** are in the "short run" category (quick API/structural tests)
- **3 parameter sets** are in the "mid run" category (numerical error accumulation)
- **1 parameter set** is in the "long run" category (full accuracy validation)
- **Consolidation opportunity**: Reduce to 3-4 standard sets + edge cases
- **Estimated savings**: 60-75% reduction in compilation time

**Note**: Parameters marked as "default" use the values from `tests/conftest.py`:
- `dt_min=1e-7`, `dt_max=1.0`, `atol=1e-6`, `rtol=1e-6`
- `newton_tolerance=1e-6`, `krylov_tolerance=1e-6`
- `step_controller='fixed'`, `output_types=['state']`

---

## Detailed Analysis

### Parameter Set Distribution

```
Test Categories by Duration:

SHORT RUN (0-0.06s)          |████████████████| 14 sets (78%)
MID RUN (0.1-0.2s)           |███             |  3 sets (17%)
LONG RUN (0.3s+)             |█               |  1 set  ( 5%)
                              0         10        20
```

### Compilation Overhead Breakdown

```
Current State:
┌─────────────────────────────────────────────────────────────┐
│ Test File                      │ Param Sets │ Est. Overhead │
├────────────────────────────────┼────────────┼───────────────┤
│ test_step_algorithms.py        │     30+    │   15-30 min   │
│ test_ode_loop.py               │     30+    │   15-30 min   │
│ test_solveresult.py            │      3     │    2-3 min    │
│ test_output_sizes.py           │      2     │    1-2 min    │
│ test_solver.py                 │      4     │    2-4 min    │
│ test_controllers.py            │      6     │    3-6 min    │
│ Other test files               │      5     │    3-5 min    │
├────────────────────────────────┼────────────┼───────────────┤
│ TOTAL                          │    ~80     │   40-80 min   │
└─────────────────────────────────────────────────────────────┘

After Consolidation:
┌─────────────────────────────────────────────────────────────┐
│ Standard Sets                  │      3     │    1.5-3 min  │
│ Edge Case Sets                 │     5-8    │    2.5-4 min  │
├────────────────────────────────┼────────────┼───────────────┤
│ TOTAL                          │    8-11    │    4-7 min    │
│ SAVINGS                        │    ~70     │   30-70 min   │
└─────────────────────────────────────────────────────────────┘
```

---

## Category 1: SHORT RUN Tests

### Purpose
Quick structural and API tests that don't require numerical accuracy validation. These tests verify:
- Object instantiation
- Data structure shapes
- API contracts
- Result formatting

### Current Parameter Sets (8 catalogued)

| Name | algorithm | duration | dt | dt_min | dt_max | dt_save | dt_summarise | atol | rtol | newton_tol | krylov_tol | step_ctrl | output_types |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solveresult_short | default | 0.05 | default | default | default | 0.02 | 0.04 | default | default | default | default | default | state, observables, time... |
| solveresult_full | default | 0.06 | default | default | default | 0.02 | 0.04 | default | default | default | default | default | state, observables, time... |
| solveresult_status | default | 0.05 | default | default | default | 0.02 | default | default | default | default | default | default | state, observables, time |
| solver_basic | default | 0.05 | default | default | default | 0.02 | 0.04 | default | default | default | default | default | state, time, observables... |
| solver_grid_types | default | 0.05 | default | default | default | 0.02 | 0.04 | default | default | default | default | default | state, time, observables... |
| solver_prebuilt | default | 0.05 | default | default | default | 0.02 | 0.04 | default | default | default | default | default | state, time |
| output_sizes_default | default | 0.0e+00 | default | default | default | default | default | default | default | default | default | default | time, state, observables... |
| loop_crank_nicolson | crank_nicolson | 0.05 | default | default | default | 0.05 | default | default | default | default | default | default | state, observables |

**Note**: "default" indicates the parameter uses the conftest.py default value.

### Usage Details

- **solveresult_short**: 13× SolveResult API tests
- **solveresult_full**: 1× full instantiation test
- **solveresult_status**: 1× status codes test
- **solver_basic**: 1× basic solve test
- **solver_grid_types**: 1× grid types test
- **solver_prebuilt**: 2× prebuilt arrays tests
- **output_sizes_default**: 4× output size calc tests
- **loop_crank_nicolson**: 1× initial observable seed test

### Consolidation Strategy

**Replace all SHORT RUN sets with single parameter set:**

```python
SHORT_RUN_PARAMS = {
    'duration': 0.05,
    'dt': 0.01,              # default
    'dt_min': 1e-7,          # default
    'dt_max': 1.0,           # default
    'dt_save': 0.05,         # Save only at end
    'dt_summarise': 0.05,    # Summarize only at end
    'atol': 1e-6,            # default
    'rtol': 1e-6,            # default
    'newton_tolerance': 1e-6,  # default
    'krylov_tolerance': 1e-6,  # default
    'step_controller': 'fixed',  # default
    'output_types': ['state', 'time'],
}
```

---

## Category 2: MID RUN Tests

### Purpose
Medium-length runs with frequent saves to allow numerical errors to accumulate across steps. These tests verify:
- Numerical stability over multiple steps
- Error accumulation patterns
- Algorithm correctness
- CPU/GPU result agreement

### Current Parameter Sets (3 total)

| Name | algorithm | duration | dt | dt_min | dt_max | dt_save | dt_summarise | atol | rtol | newton_tol | krylov_tol | step_ctrl | output_types |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step_overrides | default | default | 0.001953 | 1.0e-06 | default | default | default | 1.0e-06 | 1.0e-06 | 1.0e-06 | 1.0e-06 | default | state |
| loop_default | default | default | 0.001 | 1.0e-08 | 0.5 | 0.02 | default | 1.0e-05 | 1.0e-06 | 1.0e-07 | 1.0e-07 | default | state, time |
| loop_metrics | euler | default | 0.0025 | default | default | 0.01 | 0.1 | default | default | default | default | default | default |

**Note**: "default" indicates the parameter uses the conftest.py default value.

### Usage Details

- **step_overrides**: 3× step algo tests × ~30 algorithms
- **loop_default**: 1× loop test × ~30 algo/controller combos
- **loop_metrics**: 1× metrics test × ~15 output types

### Consolidation Strategy

**Standardize to single MID_RUN parameter set:**

```python
MID_RUN_PARAMS = {
    'duration': 0.2,         # default
    'dt': 0.001,
    'dt_min': 1e-7,          # default
    'dt_max': 0.5,
    'dt_save': 0.02,         # 10 save points
    'dt_summarise': 0.1,     # 2 summary points
    'atol': 1e-6,            # default
    'rtol': 1e-6,            # default
    'newton_tolerance': 1e-6,  # default
    'krylov_tolerance': 1e-6,  # default
    'step_controller': 'fixed',  # default (varies by test)
    'output_types': ['state', 'time', 'mean'],
}
```

---

## Category 3: LONG RUN Tests

### Purpose
Full numerical accuracy and drift validation over extended integration periods. These tests verify:
- Long-term numerical accuracy
- Drift behavior
- Full system integration

### Current Parameter Sets (1 total)

| Name | algorithm | duration | dt | dt_min | dt_max | dt_save | dt_summarise | atol | rtol | newton_tol | krylov_tol | step_ctrl | output_types |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solverkernel_run | default | 0.3 | 0.001 | default | default | 0.1 | 0.3 | default | default | default | default | default | state, observables, time... |

**Note**: "default" indicates the parameter uses the conftest.py default value.

### Consolidation Strategy

**Keep as LONG_RUN parameter set:**

```python
LONG_RUN_PARAMS = {
    'duration': 0.3,
    'dt': 0.0005,
    'dt_min': 1e-7,          # default
    'dt_max': 1.0,           # default
    'dt_save': 0.05,
    'dt_summarise': 0.15,
    'atol': 1e-6,            # default
    'rtol': 1e-6,            # default
    'newton_tolerance': 1e-6,  # default
    'krylov_tolerance': 1e-6,  # default
    'step_controller': 'fixed',  # default
    'output_types': ['state', 'observables', 'time', 'mean', 'rms'],
}
```

---

## Edge Case Tests (Keep Separate)

### Tests with Unique Requirements

| Name | algorithm | duration | dt | dt_min | dt_max | dt_save | dt_summarise | atol | rtol | newton_tol | krylov_tol | step_ctrl | output_types |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loop_float32_small | euler | 1.0e-04 | 1.0e-07 | default | default | 2.0e-05 | default | default | default | default | default | default | state, time |
| loop_large_t0 | euler | 0.001 | 1.0e-06 | default | default | 2.0e-04 | default | default | default | default | default | default | state, time |
| loop_adaptive | crank_nicolson | 1.0e-04 | default | 1.0e-07 | 1.0e-06 | 2.0e-05 | default | default | default | default | default | PI | state, time |
| controller_dt_clamps | default | default | default | 0.1 | 0.2 | default | default | default | default | default | default | default | default |
| controller_gain_clamps | default | default | default | 1.0e-04 | 1 | default | default | default | default | default | default | default | default |

**Note**: "default" indicates the parameter uses the conftest.py default value.

### Rationale for Keeping Separate

- **loop_float32_small**: 1× float32 accumulation test
- **loop_large_t0**: 1× large t0 test
- **loop_adaptive**: 1× adaptive controller test
- **controller_dt_clamps**: 2× dt clamping tests
- **controller_gain_clamps**: 2× gain clamping tests

---

## Implementation Plan

### Phase 1: Define Standard Parameter Sets

Add to `tests/conftest.py`:

```python
# Standard parameter sets for common test categories
SHORT_RUN_PARAMS = {
    'duration': 0.05,
    'dt_save': 0.05,
    'dt_summarise': 0.05,
    'output_types': ['state', 'time'],
}

MID_RUN_PARAMS = {
    'dt': 0.001,
    'dt_save': 0.02,
    'dt_summarise': 0.1,
    'dt_max': 0.5,
    'output_types': ['state', 'time', 'mean'],
}

LONG_RUN_PARAMS = {
    'duration': 0.3,
    'dt': 0.0005,
    'dt_save': 0.05,
    'dt_summarise': 0.15,
    'output_types': ['state', 'observables', 'time', 'mean', 'rms'],
}
```

### Phase 2: Update Test Files

#### 2.1 SolveResult Tests (`test_solveresult.py`)

**Current**: 13 tests use variations of short parameters

**Change**:
```python
# Before (multiple parameter sets)
@pytest.mark.parametrize('solver_settings_override', [
    {'duration': 0.05, 'dt_save': 0.02, ...},
    {'duration': 0.06, 'dt_save': 0.02, ...},
], indirect=True)

# After (single standard set)
@pytest.mark.parametrize('solver_settings_override', [
    SHORT_RUN_PARAMS,
], indirect=True)
```

#### 2.2 Step Algorithm Tests (`test_step_algorithms.py`)

**Current**: STEP_OVERRIDES dict used with ~30 algorithm cases

**Change**:
```python
# Before
STEP_OVERRIDES = {'dt': 0.001953125, 'dt_min': 1e-6, ...}

# After
STEP_OVERRIDES = MID_RUN_PARAMS
```

#### 2.3 Loop Tests (`test_ode_loop.py`)

**Current**: DEFAULT_OVERRIDES dict used with ~30 cases

**Change**:
```python
# Before
DEFAULT_OVERRIDES = {'dt': 0.001, 'dt_save': 0.02, ...}

# After
DEFAULT_OVERRIDES = MID_RUN_PARAMS
```

### Phase 3: Validation

1. Run tests with new parameter sets
2. Verify numerical results remain valid
3. Measure compilation time reduction
4. Document changes in test suite

---

## Appendix A: Complete Parameter Set Inventory

### All Parameter Sets with Full Details

#### SHORT RUN

| Name | algorithm | duration | dt | dt_min | dt_max | dt_save | dt_summarise | atol | rtol | newton_tol | krylov_tol | step_ctrl | output_types |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solveresult_short | default | 0.05 | default | default | default | 0.02 | 0.04 | default | default | default | default | default | state, observables, time... |
| solveresult_full | default | 0.06 | default | default | default | 0.02 | 0.04 | default | default | default | default | default | state, observables, time... |
| solveresult_status | default | 0.05 | default | default | default | 0.02 | default | default | default | default | default | default | state, observables, time |
| solver_basic | default | 0.05 | default | default | default | 0.02 | 0.04 | default | default | default | default | default | state, time, observables... |
| solver_grid_types | default | 0.05 | default | default | default | 0.02 | 0.04 | default | default | default | default | default | state, time, observables... |
| solver_prebuilt | default | 0.05 | default | default | default | 0.02 | 0.04 | default | default | default | default | default | state, time |
| output_sizes_default | default | 0.0e+00 | default | default | default | default | default | default | default | default | default | default | time, state, observables... |
| loop_crank_nicolson | crank_nicolson | 0.05 | default | default | default | 0.05 | default | default | default | default | default | default | state, observables |

#### MID RUN

| Name | algorithm | duration | dt | dt_min | dt_max | dt_save | dt_summarise | atol | rtol | newton_tol | krylov_tol | step_ctrl | output_types |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step_overrides | default | default | 0.001953 | 1.0e-06 | default | default | default | 1.0e-06 | 1.0e-06 | 1.0e-06 | 1.0e-06 | default | state |
| loop_default | default | default | 0.001 | 1.0e-08 | 0.5 | 0.02 | default | 1.0e-05 | 1.0e-06 | 1.0e-07 | 1.0e-07 | default | state, time |
| loop_metrics | euler | default | 0.0025 | default | default | 0.01 | 0.1 | default | default | default | default | default | default |

#### LONG RUN

| Name | algorithm | duration | dt | dt_min | dt_max | dt_save | dt_summarise | atol | rtol | newton_tol | krylov_tol | step_ctrl | output_types |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solverkernel_run | default | 0.3 | 0.001 | default | default | 0.1 | 0.3 | default | default | default | default | default | state, observables, time... |

#### EDGE CASES

| Name | algorithm | duration | dt | dt_min | dt_max | dt_save | dt_summarise | atol | rtol | newton_tol | krylov_tol | step_ctrl | output_types |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loop_float32_small | euler | 1.0e-04 | 1.0e-07 | default | default | 2.0e-05 | default | default | default | default | default | default | state, time |
| loop_large_t0 | euler | 0.001 | 1.0e-06 | default | default | 2.0e-04 | default | default | default | default | default | default | state, time |
| loop_adaptive | crank_nicolson | 1.0e-04 | default | 1.0e-07 | 1.0e-06 | 2.0e-05 | default | default | default | default | default | PI | state, time |
| controller_dt_clamps | default | default | default | 0.1 | 0.2 | default | default | default | default | default | default | default | default |
| controller_gain_clamps | default | default | default | 1.0e-04 | 1 | default | default | default | default | default | default | default | default |

---

## Appendix B: Test File Summary

| Test File | Current Param Sets | After Consolidation | Reduction |
|-----------|-------------------|---------------------|-----------|
| test_step_algorithms.py | 30+ (STEP_CASES × sets) | 1 (MID_RUN) | 97% |
| test_ode_loop.py | 30+ (LOOP_CASES × sets) + 6 edge | 1 (MID_RUN) + 3 edge | 90% |
| test_solveresult.py | 3 variations | 1 (SHORT_RUN) | 67% |
| test_solver.py | 4 variations | 1 (SHORT_RUN) | 75% |
| test_output_sizes.py | 2 | 1 (SHORT_RUN) + 1 edge | 50% |
| test_controllers.py | 6 | 1 (MID_RUN) + 4 edge | 17% |
| test_SolverKernel.py | 1 | 1 (LONG_RUN) | 0% |
| **TOTAL** | **~80** | **3 standard + ~10 edge = ~13** | **~85%** |

---

## Appendix C: Default Parameter Values Reference

From `tests/conftest.py` (session-scoped `solver_settings` fixture):

```python
DEFAULTS = {
    'algorithm': 'euler',
    'duration': 0.2,
    'dt': 0.01,
    'dt_min': 1e-7,
    'dt_max': 1.0,
    'dt_save': 0.1,
    'dt_summarise': 0.2,
    'atol': 1e-6,
    'rtol': 1e-6,
    'newton_tolerance': 1e-6,
    'krylov_tolerance': 1e-6,
    'step_controller': 'fixed',
    'output_types': ['state'],
}
```

When a parameter is marked as "default" in the tables above, it uses the value shown here.
