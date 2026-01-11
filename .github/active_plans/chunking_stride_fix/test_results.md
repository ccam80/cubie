# Test Results Summary

## Test Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not cupy and not specific_algos" -v --tb=short tests/memory/test_memmgmt.py tests/batchsolving/arrays/test_chunked_transfers.py tests/batchsolving/test_chunked_solver.py tests/batchsolving/arrays/test_basearraymanager.py tests/batchsolving/arrays/test_batchoutputarrays.py tests/batchsolving/arrays/test_batchinputarrays.py
```

## Overview
- **Tests Run**: 165
- **Passed**: 159
- **Failed**: 6
- **Errors**: 0
- **Skipped**: 0

## Passing Test Files (159 tests)

All tests in these files passed:
- `tests/memory/test_memmgmt.py` - All memory manager tests including chunking
- `tests/batchsolving/arrays/test_chunked_transfers.py` - All chunked transfer tests
- `tests/batchsolving/arrays/test_basearraymanager.py` - All base array manager tests
- `tests/batchsolving/arrays/test_batchoutputarrays.py` - All output array tests
- `tests/batchsolving/arrays/test_batchinputarrays.py` - All input array tests

## Failures (6 tests)

All failures are in `tests/batchsolving/test_chunked_solver.py`:

### test_chunked_solve_produces_valid_output[run]
**Type**: AttributeError
**Message**: `'SolveResult' object has no attribute 'state'`

### test_chunked_solve_produces_valid_output[time]
**Type**: AttributeError
**Message**: `'SolveResult' object has no attribute 'state'`

### test_chunked_solve_with_observables[run]
**Type**: AttributeError
**Message**: `'SolveResult' object has no attribute 'observables'`

### test_chunked_solve_with_observables[time]
**Type**: AttributeError
**Message**: `'SolveResult' object has no attribute 'observables'`

### test_chunked_solve_small_batch[run]
**Type**: AttributeError
**Message**: `'SolveResult' object has no attribute 'state'`

### test_chunked_solve_small_batch[time]
**Type**: AttributeError
**Message**: `'SolveResult' object has no attribute 'state'`

## Analysis

### Root Cause
The test file `tests/batchsolving/test_chunked_solver.py` was created with **incorrect API usage**. The tests access `result.state` and `result.observables` which **do not exist** on the `SolveResult` class.

The correct API uses:
- `result.time_domain_array` - Contains both state and observable data combined
- `result.time_domain_legend` - Contains labels for the time domain array

### Failure Relationship to Changes
These failures are **NOT related to the chunking/stride fixes**. The failures are caused by the test file using an incorrect API that never existed in the `SolveResult` class. The test file was created by a previous agent with incorrect assumptions about the `SolveResult` API.

### Evidence
1. The `SolveResult` class has no `state` property - confirmed by reviewing `src/cubie/batchsolving/solveresult.py`
2. Other test files in the repository correctly use `result.time_domain_array` (e.g., `tests/batchsolving/test_solveresult.py`)
3. The underlying chunking mechanism works correctly - all 159 tests in the other test files pass

## Recommendations

The test file `tests/batchsolving/test_chunked_solver.py` needs to be rewritten to use the correct `SolveResult` API:

1. Replace `result.state` with `result.time_domain_array`
2. Replace `result.observables` with checks on `result.time_domain_legend` for observable presence
3. Use proper stride order handling when accessing array dimensions

Example fix pattern:
```python
# Instead of:
assert result.state is not None

# Use:
assert result.time_domain_array is not None
assert result.time_domain_array.size > 0
```

## Summary

The **core chunking/stride fixes work correctly**. All 159 tests for the actual fix components pass. The 6 failing tests are in a newly created integration test file that was written with incorrect API assumptions and need to be corrected.
