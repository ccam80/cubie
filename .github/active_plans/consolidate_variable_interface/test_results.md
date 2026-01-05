# Test Results Summary

## Overview
- **Tests Run**: 138
- **Passed**: 138
- **Failed**: 0
- **Errors**: 0
- **Skipped**: 1

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short tests/batchsolving/test_system_interface.py tests/batchsolving/test_solver.py tests/outputhandling/test_output_config.py
```

## Skipped Tests
- `test_solver.py::test_deprecated_label_parameters_rejected` - System has observables, test requires no observables

## Bug Fixed During Testing

### Issue
Initial test run revealed 55 errors and 19 failures, all with the same error:
```
AttributeError: 'Solver' object has no attribute 'kernel'
```

### Root Cause
In `Solver.__init__`, the method `convert_output_labels()` was called (line 264) before `self.kernel` was initialized (line 284). The `convert_output_labels()` method accessed `self.system_sizes.states` and `self.system_sizes.observables`, which are properties that delegate to `self.kernel.system_sizes`.

### Fix
Modified `Solver.convert_output_labels()` to use `self.system_interface.states.n` and `self.system_interface.observables.n` instead of `self.system_sizes.states` and `self.system_sizes.observables`. The `system_interface` is already initialized before `convert_output_labels()` is called.

**File changed**: `src/cubie/batchsolving/solver.py`

```python
# Before (broken):
self.system_interface.convert_variable_labels(
    output_settings,
    self.system_sizes.states,
    self.system_sizes.observables,
)

# After (fixed):
self.system_interface.convert_variable_labels(
    output_settings,
    self.system_interface.states.n,
    self.system_interface.observables.n,
)
```

## Recommendations
- All tests pass successfully
- The variable interface consolidation feature is working correctly
- No additional action required
