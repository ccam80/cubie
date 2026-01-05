# Test Results Summary (Final Pipeline Exit Verification)

## Overview
- **Tests Run**: 484
- **Passed**: 476
- **Failed**: 7
- **Errors**: 0
- **Skipped**: 1

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short tests/batchsolving/ tests/outputhandling/
```

## Failures

### tests/batchsolving/test_config_plumbing.py::test_comprehensive_config_plumbing[backwards_euler-fixed]
**Type**: AssertionError
**Message**: assert [0, 1, 2] == [0, 1] - Left contains one more item: 2

### tests/batchsolving/test_config_plumbing.py::test_comprehensive_config_plumbing[crank_nicolson-i]
**Type**: AssertionError
**Message**: assert [0, 1, 2] == [0, 1] - Left contains one more item: 2

### tests/batchsolving/test_config_plumbing.py::test_comprehensive_config_plumbing[rk23-gustafsson]
**Type**: AssertionError
**Message**: assert [0, 1, 2] == [0, 1] - Left contains one more item: 2

### tests/batchsolving/test_config_plumbing.py::test_comprehensive_config_plumbing[rk45-pid]
**Type**: AssertionError
**Message**: assert [0, 1, 2] == [0, 1] - Left contains one more item: 2

### tests/batchsolving/test_config_plumbing.py::test_comprehensive_config_plumbing[dopri54-pi]
**Type**: AssertionError
**Message**: assert [0, 1, 2] == [0, 1] - Left contains one more item: 2

### tests/batchsolving/test_config_plumbing.py::test_comprehensive_config_plumbing[tsit5-pi]
**Type**: AssertionError
**Message**: assert [0, 1, 2] == [0, 1] - Left contains one more item: 2

### tests/batchsolving/test_solver.py::test_solve_dict_path_backward_compatible
**Type**: AttributeError
**Message**: tid=[0, 13, 0] ctaid=[0, 0, 0]: module 'numba.cuda' has no attribute 'local'

## Skipped Tests

### tests/batchsolving/test_solver.py:1448
**Reason**: System has observables, test requires no observables

## Analysis

### Failure Pattern 1: test_comprehensive_config_plumbing (6 failures)
All 6 parametrized variants of `test_comprehensive_config_plumbing` fail with the same issue: the test expects a list of length 2 (`[0, 1]`) but receives a list of length 3 (`[0, 1, 2]`). This suggests:
- A new variable or output was added to the variable interface
- The test's expected values need to be updated to reflect the new consolidated variable interface

### Failure Pattern 2: test_solve_dict_path_backward_compatible (1 failure)
This is a CUDA simulation compatibility issue. The error `module 'numba.cuda' has no attribute 'local'` indicates the test uses `cuda.local` which is not available in CUDA simulation mode. This is a known limitation of NUMBA_ENABLE_CUDASIM.

## Recommendations

1. **For test_comprehensive_config_plumbing failures**: Review the test to determine if:
   - The expected values `[0, 1]` need to be updated to `[0, 1, 2]` to match the new variable interface
   - Or if the implementation is incorrectly adding an extra variable index

2. **For test_solve_dict_path_backward_compatible failure**: This test uses `cuda.local` which is not supported in CUDA simulation mode. Either:
   - Mark this test with `@pytest.mark.nocudasim` if it requires actual CUDA
   - Or refactor to avoid `cuda.local` in simulation mode

## Coverage
- Overall coverage: 72%
- Coverage XML written to coverage.xml
