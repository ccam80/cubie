# CuBIE Test Parameterization Analysis Report

**Date**: 2025-12-09

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

### Current Parameter Sets (14 total)

| Parameter Set | Duration | Steps | Saves | Tests Using It |
|---------------|----------|-------|-------|----------------|
| solveresult_short | 0.05s | 5 | 2 | 13× SolveResult API tests |
| solveresult_full | 0.06s | 6 | 3 | 1× full instantiation test |
| solveresult_status | 0.05s | 5 | 2 | 1× status codes test |
| solver_basic | 0.05s | 5 | 2 | 1× basic solve test |
| solver_grid_types | 0.05s | 5 | 2 | 1× grid types test |
| solver_prebuilt | 0.05s | 5 | 2 | 2× prebuilt arrays tests |
| output_sizes_default | 0.00s | 0 | 0 | 4× output size calc tests |
| loop_crank_nicolson | 0.05s | 5 | 1 | 1× initial observable seed test |
| loop_float32_small | 0.0001s | 1000 | 5 | 1× float32 accumulation test |
| loop_large_t0 | 0.001s | 1000 | 5 | 1× large t0 test |
| loop_adaptive | 0.0001s | ? | 5 | 1× adaptive controller test |
| controller_matches | 0.2s | 20 | 2 | 1× CPU/GPU match test |
| controller_sequence | 0.2s | 20 | 2 | 1× sequence agreement test |
| output_sizes_realistic | 0.2s | 20 | 20 | 1× realistic scenario test |

### Consolidation Strategy

**Replace all 14 sets with single SHORT_RUN parameter set:**

```python
SHORT_RUN_PARAMS = {
    'duration': 0.05,
    'dt': 0.01,
    'dt_save': 0.05,      # Save only at end
    'dt_summarise': 0.05,  # Summarize only at end
    'output_types': ['state', 'time'],
}
```

**Exceptions to keep separate:**
- `loop_float32_small`: Tests float32 precision accumulation (edge case)
- `loop_large_t0`: Tests large initial time handling (edge case)
- `output_sizes_default`: Zero-duration structural test (edge case)

---

## Category 2: MID RUN Tests

### Purpose
Medium-length runs with frequent saves to allow numerical errors to accumulate across steps. These tests verify:
- Numerical stability over multiple steps
- Error accumulation patterns
- Algorithm correctness
- CPU/GPU result agreement

### Current Parameter Sets (3 total)

| Parameter Set | Duration | Steps | Saves | Tests Using It |
|---------------|----------|-------|-------|----------------|
| step_overrides | 0.2s | 102 | 2 | 3× step algo tests × ~30 algorithms |
| loop_default | 0.2s | 200 | 10 | 1× loop test × ~30 algo/controller combos |
| loop_metrics | 0.2s | 80 | 20 | 1× metrics test × ~15 output types |

### Usage Details

**step_overrides (STEP_OVERRIDES dict)**:
```python
{
    'dt': 0.001953125,  # Exactly-representable float
    'dt_min': 1e-6,
    'newton_tolerance': 1e-6,
    'krylov_tolerance': 1e-6,
    'atol': 1e-6,
    'rtol': 1e-6,
    'output_types': ['state'],
}
```
Used with STEP_CASES (~30 algorithms) in:
- `test_stage_cache_reuse()` - 5 cache reuse cases
- `test_against_euler()` - 30+ algorithm cases
- `test_algorithm()` - 30+ algorithm cases

**loop_default (DEFAULT_OVERRIDES dict)**:
```python
{
    'dt': 0.001,
    'dt_save': 0.02,
    'dt_min': 1e-8,
    'dt_max': 0.5,
    'newton_tolerance': 1e-7,
    'krylov_tolerance': 1e-7,
    'atol': 1e-5,
    'rtol': 1e-6,
    'output_types': ['state', 'time'],
}
```
Used with LOOP_CASES (~30 algorithm/controller combos) in:
- `test_loop()` - Main integration loop test

### Consolidation Strategy

**Standardize to single MID_RUN parameter set:**

```python
MID_RUN_PARAMS = {
    'duration': 0.2,
    'dt': 0.001,
    'dt_save': 0.02,       # 10 save points
    'dt_summarise': 0.1,   # 2 summary points
    'dt_min': 1e-7,
    'dt_max': 0.5,
    'newton_tolerance': 1e-6,
    'krylov_tolerance': 1e-6,
    'atol': 1e-6,
    'rtol': 1e-6,
    'output_types': ['state', 'time', 'mean'],
}
```

**Tests to use MID_RUN:**
- All step algorithm tests (replace STEP_OVERRIDES)
- All loop tests (replace DEFAULT_OVERRIDES)
- Controller tests

---

## Category 3: LONG RUN Tests

### Purpose
Full numerical accuracy and drift validation over extended integration periods. These tests verify:
- Long-term numerical accuracy
- Drift behavior
- Full system integration

### Current Parameter Sets (1 total)

| Parameter Set | Duration | Steps | Saves | Tests Using It |
|---------------|----------|-------|-------|----------------|
| solverkernel_run | 0.3s | 300 | 2 | 1× full SolverKernel integration test |

### Consolidation Strategy

**Keep as LONG_RUN parameter set:**

```python
LONG_RUN_PARAMS = {
    'duration': 0.3,
    'dt': 0.0005,
    'dt_save': 0.05,
    'dt_summarise': 0.15,
    'output_types': ['state', 'observables', 'time', 'mean', 'rms'],
}
```

**Use sparingly:**
- SolverKernel full integration (current use)
- One test per major algorithm family (ERK, DIRK, FIRK, Rosenbrock)
- Mark with `@pytest.mark.slow`

---

## Edge Case Tests (Keep Separate)

### Tests with Unique Requirements

| Test | Reason to Keep Separate | Parameter Set |
|------|-------------------------|---------------|
| `test_float32_small_timestep_accumulation` | Tests float32 precision edge case | duration=0.0001s, dt=1e-7 |
| `test_large_t0_with_small_steps` | Tests large t0 handling | t0=100.0, duration=0.001s |
| `test_adaptive_controller_with_float32` | Tests adaptive with float32 | duration=0.0001s, adaptive |
| `test_from_output_fns_with_nonzero` | Zero-duration structural test | duration=0.0 |
| `test_save_at_settling_time_boundary` | Tests settling_time feature | settling_time=0.1 |
| Controller dt_clamps tests | Tests dt clamping with extreme errors | Special error arrays |
| Controller gain_clamps tests | Tests gain clamping | Special local_mem arrays |

---

## Implementation Plan

### Phase 1: Define Standard Parameter Sets

Add to `tests/conftest.py`:

```python
# Standard parameter sets for common test categories
SHORT_RUN_PARAMS = {
    'duration': 0.05,
    'dt': 0.01,
    'dt_save': 0.05,
    'dt_summarise': 0.05,
    'output_types': ['state', 'time'],
}

MID_RUN_PARAMS = {
    'duration': 0.2,
    'dt': 0.001,
    'dt_save': 0.02,
    'dt_summarise': 0.1,
    'dt_min': 1e-7,
    'dt_max': 0.5,
    'newton_tolerance': 1e-6,
    'krylov_tolerance': 1e-6,
    'atol': 1e-6,
    'rtol': 1e-6,
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

Current: 13 tests use variations of short parameters

Change:
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

Current: STEP_OVERRIDES dict used with ~30 algorithm cases

Change:
```python
# Before
STEP_OVERRIDES = {'dt': 0.001953125, 'dt_min': 1e-6, ...}

# After
STEP_OVERRIDES = MID_RUN_PARAMS
```

#### 2.3 Loop Tests (`test_ode_loop.py`)

Current: DEFAULT_OVERRIDES dict used with ~30 cases

Change:
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

### All Parameter Sets with Usage Count

| ID | Name | Category | Duration | dt | Saves | Summaries | Used By | Usage Count |
|----|------|----------|----------|-----|-------|-----------|---------|-------------|
| 1 | solveresult_short | SHORT | 0.05s | 0.01 | 2 | 1 | test_solveresult.py | 13 |
| 2 | step_overrides | MID | 0.2s | 0.00195 | 2 | 1 | test_step_algorithms.py | 90+ |
| 3 | loop_default | MID | 0.2s | 0.001 | 10 | 1 | test_ode_loop.py | 30+ |
| 4 | solverkernel_run | LONG | 0.3s | 0.001 | 2 | 1 | test_SolverKernel.py | 1 |
| 5 | solver_basic | SHORT | 0.05s | 0.01 | 2 | 1 | test_solver.py | 1 |
| 6 | solver_grid_types | SHORT | 0.05s | 0.01 | 2 | 1 | test_solver.py | 1 |
| 7 | solver_prebuilt | SHORT | 0.05s | 0.01 | 2 | 1 | test_solver.py | 2 |
| 8 | output_sizes_default | SHORT | 0.0s | - | 0 | 0 | test_output_sizes.py | 4 |
| 9 | output_sizes_realistic | SHORT | 0.2s | 0.01 | 20 | 2 | test_output_sizes.py | 1 |
| 10 | loop_metrics | MID | 0.2s | 0.0025 | 20 | 2 | test_ode_loop.py | 15 |
| 11 | loop_float32_small | EDGE | 0.0001s | 1e-7 | 5 | 0 | test_ode_loop.py | 1 |
| 12 | loop_large_t0 | EDGE | 0.001s | 1e-6 | 5 | 0 | test_ode_loop.py | 1 |
| 13 | controller_dt_clamps | EDGE | varies | varies | - | - | test_controllers.py | 2 |
| 14 | controller_gain_clamps | EDGE | varies | varies | - | - | test_controllers.py | 2 |
| 15 | controller_matches | SHORT | 0.2s | 0.01 | 2 | 1 | test_controllers.py | 1 |
| 16 | controller_sequence | SHORT | 0.2s | 0.01 | 2 | 1 | test_controllers.py | 1 |

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
