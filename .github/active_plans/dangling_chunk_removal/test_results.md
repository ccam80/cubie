# Test Results Summary - Dangling Chunk Length Removal

## Test Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest tests/memory/ tests/batchsolving/ -v --tb=line -m "not nocudasim and not cupy"
```

## Overview
**Note**: Full test suite did not complete within 4-minute timeout. Results based on partial runs and targeted testing.

Based on the tests that completed and specific targeted tests:
- **Estimated Total Tests**: 468
- **Passed**: ~400+
- **Failed**: 35+ identified
- **Status**: Multiple test files timing out, suggesting performance issues

## Failure Categories

### 1. Missing Field Errors (run_params attribute)

**Root Cause**: Tests and code expecting objects to have a `run_params` attribute that doesn't exist

**Location**: `src/cubie/memory/mem_manager.py:1203`
```python
num_runs = triggering_instance.run_params.runs
```

**Affected Tests**:

1. **tests/batchsolving/arrays/test_basearraymanager.py::TestBaseArrayManager::test_request_allocation_auto**
   - Line: 1203 in mem_manager.py
   - Error: `AttributeError: 'ConcreteArrayManager' object has no attribute 'run_params'`

2. **tests/batchsolving/arrays/test_basearraymanager.py::TestBaseArrayManager::test_initialize_device_zeros**
   - Line: 1203 in mem_manager.py
   - Error: `AttributeError: 'ConcreteArrayManager' object has no attribute 'run_params'`

3. **tests/batchsolving/arrays/test_basearraymanager.py::TestMemoryManagerIntegration::test_allocation_with_settings**
   - Line: 1203 in mem_manager.py
   - Error: `AttributeError: 'ConcreteArrayManager' object has no attribute 'run_params'`

4. **tests/batchsolving/arrays/test_batchoutputarrays.py::test_output_arrays_with_different_systems[solver_settings_override0]**
   - Line: 1203 in mem_manager.py
   - Error: `AttributeError: 'OutputArrays' object has no attribute 'run_params'`

5. **tests/batchsolving/arrays/test_batchoutputarrays.py::test_output_arrays_with_different_systems[solver_settings_override1]**
   - Line: 1203 in mem_manager.py
   - Error: `AttributeError: 'OutputArrays' object has no attribute 'run_params'`

6. **tests/batchsolving/arrays/test_batchoutputarrays.py::test_output_arrays_with_different_systems[solver_settings_override2]**
   - Line: 1203 in mem_manager.py
   - Error: `AttributeError: 'OutputArrays' object has no attribute 'run_params'`

7. **tests/batchsolving/test_SolverKernel.py::TestRunParamsIntegration::test_runparams_initialized_on_construction**
   - Error: `AttributeError: 'OutputArrays' object has no attribute 'run_params'`

8. **tests/batchsolving/test_SolverKernel.py::TestRunParamsIntegration::test_on_allocation_updates_runparams**
   - Error: `AttributeError: 'OutputArrays' object has no attribute 'run_params'`

### 2. Chunking Logic Errors

**Root Cause**: Issues with chunk_slice method and chunking parameter handling

#### 2a. chunk_slice returning wrong shape

**Location**: `src/cubie/batchsolving/arrays/BaseArrayManager.py:126-187` (chunk_slice method)

**Issue**: When slicing non-final chunks, the method uses `slice(start, end)` where `end=None` for the final chunk. However, for non-final chunks in the test, it's returning the full array dimension instead of the chunk size.

**Affected Tests**:

9. **tests/batchsolving/arrays/test_basearraymanager.py::TestChunkSliceMethod::test_chunk_slice_computes_correct_slices**
   - Line: 1733
   - Expected: `(10, 5, 25)`
   - Got: `(10, 5, 100)`
   - The slice is returning full dimension instead of chunk size

10. **tests/batchsolving/arrays/test_basearraymanager.py::TestChunkSliceMethod::test_chunk_slice_handles_final_chunk_dynamically**
    - Similar issue with chunk slicing

11. **tests/batchsolving/arrays/test_basearraymanager.py::TestChunkSliceMethod::test_chunk_slice_validates_chunk_index**
    - Chunk slice validation failing

12. **tests/batchsolving/arrays/test_basearraymanager.py::TestChunkSliceMethod::test_chunk_slice_none_parameters_returns_full_array**
    - Chunk parameter handling issue

13. **tests/batchsolving/arrays/test_basearraymanager.py::TestChunkSliceMethod::test_chunk_slice_single_chunk**
    - Single chunk case failing

14. **tests/batchsolving/arrays/test_basearraymanager.py::TestChunkSliceMethod::test_chunk_slice_different_axis_indices**
    - Axis index handling in chunking

#### 2b. Missing chunk_length in ArrayResponse

**Root Cause**: Tests expect `chunk_length` to be calculated/provided in ArrayResponse but it defaults to 1

**Affected Tests**:

15. **tests/batchsolving/test_runparams.py::test_runparams_update_from_allocation**
    - Line: 228
    - Expected: `chunk_length == 25` (for 100 runs / 4 chunks)
    - Got: `chunk_length == 1` (default value)
    - ArrayResponse doesn't compute chunk_length from runs and chunks

16. **tests/batchsolving/test_runparams.py::test_runparams_update_from_allocation_single_chunk**
    - Similar issue with chunk_length calculation

17. **tests/batchsolving/test_runparams.py::test_runparams_update_from_allocation_dangling_chunk**
    - Dangling chunk case not handled

18. **tests/batchsolving/test_runparams_integration.py::test_runparams_single_chunk**
    - Integration test for single chunk failing

19. **tests/batchsolving/test_runparams_integration.py::test_runparams_multiple_chunks**
    - Integration test for multiple chunks failing

20. **tests/batchsolving/test_runparams_integration.py::test_runparams_exact_division**
    - Exact division case failing

#### 2c. Chunk axis index issues

**Affected Tests**:

21. **tests/batchsolving/arrays/test_chunking.py::test_chunk_axis_index_in_array_requests**
    - Line: assertion failure
    - Expected: `chunk_axis_index == 2`
    - Got: `chunk_axis_index == 1`

### 3. Array Allocation/Initialization Errors

**Root Cause**: Tests creating or calling methods on array managers with incorrect parameters or missing initialization

**Affected Tests**:

22. **tests/batchsolving/arrays/test_batchoutputarrays.py::TestOutputArrays::test_call_method_allocates_arrays**
    - Allocation method failing

23. **tests/batchsolving/arrays/test_batchoutputarrays.py::TestOutputArrays::test_reallocation_on_size_change**
    - Reallocation logic issue

24. **tests/batchsolving/arrays/test_batchoutputarrays.py::TestOutputArrays::test_chunking_affects_device_array_size**
    - Device array size calculation with chunking

25. **tests/batchsolving/arrays/test_batchoutputarrays.py::TestOutputArrays::test_initialise_method**
    - Initialise method failing

26. **tests/batchsolving/arrays/test_batchoutputarrays.py::test_output_arrays_with_different_configs[output_test_overrides0]**
    - Configuration handling issue

27. **tests/batchsolving/arrays/test_batchoutputarrays.py::test_output_arrays_with_different_configs[output_test_overrides1]**
    - Configuration handling issue

28. **tests/batchsolving/arrays/test_batchoutputarrays.py::test_output_arrays_with_different_configs[output_test_overrides2]**
    - Configuration handling issue

29. **tests/batchsolving/arrays/test_batchoutputarrays.py::TestOutputArraysSpecialCases::test_allocation_with_different_solver_sizes**
    - Special case handling failing

30. **tests/batchsolving/arrays/test_batchoutputarrays.py::TestOutputArrays::test_allocation_and_getters_not_none**
    - Allocation and getter issue

31. **tests/batchsolving/arrays/test_batchinputarrays.py::TestInputArrays::test_allocation_and_getters_not_none**
    - Input array allocation issue

32. **tests/batchsolving/arrays/test_batchinputarrays.py::TestInputArrays::test_call_method_size_change_triggers_reallocation**
    - Reallocation trigger logic

33. **tests/batchsolving/arrays/test_batchinputarrays.py::TestInputArrays::test_initialise_method**
    - Initialise method failing

34. **tests/batchsolving/arrays/test_batchinputarrays.py::TestInputArrays::test_initialise_uses_chunk_slice_method**
    - Chunk slice method usage in initialise

35. **tests/batchsolving/arrays/test_batchinputarrays.py::TestInputArrays::test_dtype[solver_settings_override0]**
    - dtype handling issue

36. **tests/batchsolving/arrays/test_batchinputarrays.py::TestInputArrays::test_dtype[solver_settings_override1]**
    - dtype handling issue

37. **tests/batchsolving/arrays/test_batchinputarrays.py::test_input_arrays_with_different_systems[solver_settings_override0]**
    - System configuration issue

38. **tests/batchsolving/arrays/test_batchinputarrays.py::test_input_arrays_with_different_systems[solver_settings_override1]**
    - System configuration issue

39. **tests/batchsolving/arrays/test_batchinputarrays.py::test_input_arrays_with_different_systems[solver_settings_override2]**
    - System configuration issue

40. **tests/batchsolving/arrays/test_batchoutputarrays.py::test_dtype[solver_settings_override0]**
    - dtype issue

41. **tests/batchsolving/arrays/test_batchoutputarrays.py::test_dtype[solver_settings_override1]**
    - dtype issue

### 4. Missing Field Errors (time_domain_array)

**Root Cause**: Code expecting `time_domain_array` attribute that doesn't exist on OutputArrayContainer

**Affected Tests**:

42. **tests/batchsolving/arrays/test_chunking.py::test_chunked_shape_differs_from_shape_when_chunking**
    - Error: `AttributeError: 'OutputArrayContainer' object has no attribute 'time_domain_array'`

43. **tests/batchsolving/arrays/test_chunking.py::test_chunked_shape_equals_shape_when_not_chunking**
    - Error: `AttributeError: 'OutputArrayContainer' object has no attribute 'time_domain_array'`

### 5. Other Errors

**Affected Tests**:

44. **tests/batchsolving/arrays/test_chunking.py::TestWritebackTask::test_task_creation**
    - AssertionError: Array identity comparison issue
    - Expected same array object, got equal array with different identity

45. **tests/batchsolving/arrays/test_chunking.py::TestWritebackWatcher::test_2d_array_slice**
    - AssertionError with array slicing

46. **tests/batchsolving/arrays/test_managed_array.py::test_managed_array_chunk_fields_default_none**
    - Default field value issue

## Recommendations

### High Priority Fixes

1. **Add run_params attribute to array managers**
   - Location: BaseArrayManager, OutputArrays, InputArrays classes
   - The mem_manager.py expects `triggering_instance.run_params.runs` at line 1203
   - Need to either add this attribute or change how num_runs is obtained

2. **Fix chunk_slice method logic**
   - Location: `src/cubie/batchsolving/arrays/BaseArrayManager.py` lines 126-187
   - The method needs to properly handle all chunks, not just the final chunk
   - Current issue: returning full dimension instead of chunk-sized slices

3. **Implement chunk_length calculation in ArrayResponse or allocation logic**
   - Location: Memory manager allocation process
   - When ArrayResponse is created, chunk_length should be calculated as `ceil(num_runs / chunks)`
   - Currently defaults to 1, causing test failures

### Medium Priority Fixes

4. **Add time_domain_array to OutputArrayContainer or remove dependencies on it**
   - Either add the missing field or update tests/code that reference it

5. **Review chunk_axis_index logic**
   - Some tests expect different axis indices than what's being set
   - May need to verify the axis index calculation logic

### Low Priority / Test-Specific Issues

6. **Array identity comparisons in tests**
   - Some tests using `is` instead of equality checks
   - May need test updates rather than code fixes

## Test Infrastructure Notes

- Tests are running with `NUMBA_ENABLE_CUDASIM=1` (CUDA simulation mode)
- Tests excluding `nocudasim` and `cupy` markers as requested
- Some tests timing out, suggesting performance regression or infinite loops
- Parallel test execution with xdist (4 workers) helps but some tests still don't complete

## Next Steps

The primary blockers are:
1. Missing `run_params` attribute (8+ test failures)
2. Incorrect `chunk_slice` implementation (6+ test failures)  
3. Missing `chunk_length` calculation (6+ test failures)

These three issues account for approximately 20 of the 35+ identified failures. Fixing these should significantly improve test pass rate.
