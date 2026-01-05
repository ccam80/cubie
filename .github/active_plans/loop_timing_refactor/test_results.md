# Test Results Summary

## Overview
- **Tests Run**: 156
- **Passed**: 75
- **Failed**: 7
- **Errors**: 74
- **Skipped**: 0

## Test Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short \
  tests/integrators/loops/test_dt_update_summaries_validation.py \
  tests/integrators/loops/test_ode_loop.py \
  tests/batchsolving/test_solver.py \
  tests/batchsolving/test_solveresult.py \
  tests/outputhandling/test_output_config.py
```

## Root Cause Analysis

The refactoring appears incomplete. The following issues remain:

### Issue 1: Missing property references to old names (74 errors)

Multiple tests error with `AttributeError: 'SingleIntegratorRun' object has no attribute 'dt_save'`

**Cause**: Tests access `solver_settings["dt_save"]` and `solver_settings["dt_summarise"]` but the fixture now provides `save_every` and `summarise_every`.

**Affected Files**:
- `tests/batchsolving/test_solver.py` - References `solver_settings["dt_save"]` and `solver_settings["dt_summarise"]`
- `tests/batchsolving/test_solveresult.py` - Same issue
- `tests/integrators/loops/test_ode_loop.py` - Same issue

**Solution**: Update tests to use `solver_settings["save_every"]` and `solver_settings["summarise_every"]`

### Issue 2: SolveSpec property references (test_solve_info_property)

**Lines 187-196 in test_solver.py**:
```python
assert solve_info.dt_save == pytest.approx(
    solver_settings["dt_save"],  # dt_save doesn't exist in solver_settings
...
assert solve_info.dt_summarise == pytest.approx(
    solver_settings["dt_summarise"],  # dt_summarise doesn't exist in solver_settings
```

**Solution**: Change to `solve_info.save_every` and `solver_settings["save_every"]` (and same for summarise)

### Issue 3: loop_settings parameter using old name

**Lines 482, 506 in test_solver.py**:
```python
loop_settings={"dt_save": solver_settings["dt_save"]},
```

**Solution**: Change to `loop_settings={"save_every": solver_settings["save_every"]}`

### Issue 4: sample_summaries_every validation failures (2 errors)

Tests with certain timing parameter combinations fail with:
```
ValueError: sample_summaries_every (0.00019999...) must be an integer divisor of 
summarise_every (0.05000...). Ratio: 250.00001...
```

**Affected tests**:
- `test_large_t0_with_small_steps[float32]`
- `test_save_at_settling_time_boundary[solver_settings_override0]`

**Cause**: Tests specify `save_every` but not `sample_summaries_every`, and the auto-derived `sample_summaries_every` doesn't evenly divide `summarise_every`.

**Solution**: Either:
1. Add explicit `sample_summaries_every` to test overrides
2. Or adjust `summarise_every` to be a clean multiple of `save_every`

### Issue 5: save_last/summarise_last flags not being set (2 failures)

**Affected tests**:
- `test_save_last_flag_from_config` - `assert config.save_last is True` fails (got False)
- `test_summarise_last_flag_from_config` - `assert config.summarise_last is True` fails (got False)

**Cause**: The test overrides include output_types but the inference logic expects ALL timing params to be None (Case 1 in `_infer_timing_defaults`). However, the test still provides default timing params from `solver_settings` fixture which overrides the None values.

**Solution**: The test needs to explicitly set `save_every=None`, `summarise_every=None`, `sample_summaries_every=None` to trigger the save_last/summarise_last behavior.

## Failures Detail

### test_solve_info_property
**Type**: KeyError
**Message**: 'dt_save' (from `solver_settings["dt_save"]`)

### test_solver_with_different_algorithms  
**Type**: KeyError
**Message**: 'dt_save'

### test_solver_output_types
**Type**: KeyError
**Message**: 'dt_save'

### test_solve_ivp_save_every_param
**Type**: AttributeError
**Message**: 'SingleIntegratorRun' object has no attribute 'dt_save'

### test_solve_ivp_no_dt_save
**Type**: AttributeError
**Message**: 'SingleIntegratorRun' object has no attribute 'dt_save'

### test_save_last_flag_from_config
**Type**: AssertionError
**Message**: assert False is True (config.save_last is False when expected True)

### test_summarise_last_flag_from_config
**Type**: AssertionError
**Message**: assert False is True (config.summarise_last is False when expected True)

## Recommendations

1. **Update test_solver.py**:
   - Replace all `solver_settings["dt_save"]` with `solver_settings["save_every"]`
   - Replace all `solver_settings["dt_summarise"]` with `solver_settings["summarise_every"]`
   - Replace `solve_info.dt_save` with `solve_info.save_every`
   - Replace `solve_info.dt_summarise` with `solve_info.summarise_every`
   - Replace `loop_settings={"dt_save": ...}` with `loop_settings={"save_every": ...}`

2. **Update test_solveresult.py**:
   - Same replacements as above

3. **Fix timing validation tests**:
   - For `test_large_t0_with_small_steps`: Add `sample_summaries_every` or adjust `summarise_every`
   - For `test_save_at_settling_time_boundary`: Ensure `summarise_every` is a clean multiple of `sample_summaries_every`

4. **Fix save_last/summarise_last tests**:
   - Explicitly set timing params to None in test override to trigger Case 1 logic
