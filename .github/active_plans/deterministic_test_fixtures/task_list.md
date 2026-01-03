# Implementation Task List
# Feature: Deterministic Test Fixtures
# Plan Reference: .github/active_plans/deterministic_test_fixtures/agent_plan.md

## Task Group 1: Clean Up Unused Code
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: tests/odesystems/_utils.py (entire file - to be deleted)
- File: tests/_utils.py (lines 574-662 - random generation functions to remove)

**Input Validation Required**:
- None (deletion only)

**Tasks**:
1. **Delete tests/odesystems/_utils.py**
   - File: tests/odesystems/_utils.py
   - Action: Delete entire file
   - Details:
     - This file contains only unused functions:
       - `get_observables_list()` - only used internally
       - `random_system_values()` - only used internally
       - `create_random_test_set()` - referenced only in commented-out code
       - `create_minimal_input_sets()` - referenced only in commented-out code
       - `instantiate_or_use_instance()` - only used internally
       - `generate_system_tests()` - referenced only in commented-out code
     - The file imports from `tests/_utils.py`, but nothing imports from this file
   - Edge cases: None
   - Integration: None - file is unused

2. **Remove single_scale_float_array() from tests/_utils.py**
   - File: tests/_utils.py
   - Action: Delete function (lines 577-592)
   - Details:
     - Remove the entire function definition:
       ```python
       def single_scale_float_array(
           shape: Union[int, tuple[int]], precision=np.float64, scale=1e6
       ):
           """Generate a random float array of given shape and dtype..."""
           rng = np.random.default_rng()
           return rng.normal(scale=scale, size=shape).astype(precision)
       ```
     - This function uses unseeded `np.random.default_rng()` causing flakiness
   - Edge cases: None
   - Integration: Only called by `random_array()` which is also being removed

3. **Remove mixed_scale_float_array() from tests/_utils.py**
   - File: tests/_utils.py
   - Action: Delete function (lines 595-633)
   - Details:
     - Remove the entire function definition that generates mixed-scale random arrays
     - Uses unseeded `np.random.default_rng()` causing flakiness
   - Edge cases: None
   - Integration: Only called by `random_array()` which is also being removed

4. **Remove random_array() from tests/_utils.py**
   - File: tests/_utils.py
   - Action: Delete function (lines 636-661)
   - Details:
     - Remove the entire function definition:
       ```python
       def random_array(precision, size: Union[int, tuple[int]], scale=1e6):
           """Generate a random float array..."""
           if isinstance(scale, float):
               scale = (scale,)
           if len(scale) == 1:
               randvals = single_scale_float_array(size, precision, scale[0])
           elif len(scale) == 2:
               randvals = mixed_scale_float_array(...)
           else:
               raise ValueError(...)
           return randvals
       ```
   - Edge cases: None
   - Integration: Called by `generate_test_array()` which will be modified in Task Group 2

5. **Remove section header comment for RANDOM GENERATION**
   - File: tests/_utils.py
   - Action: Delete comment block (lines 574-576)
   - Details:
     - Remove the section header:
       ```python
       ### ********************************************************************************************************* ###
       #                                        RANDOM GENERATION
       ### ********************************************************************************************************* ###
       ```
   - Edge cases: None
   - Integration: Cosmetic cleanup

**Tests to Create**:
- None (this is cleanup only)

**Tests to Run**:
- tests/outputhandling/test_output_functions.py (after Task Group 2 completes)

**Outcomes**: 
- Files Modified:
  * tests/odesystems/_utils.py (296 lines removed - file emptied for deletion)
  * tests/_utils.py (89 lines removed)
- Functions/Methods Removed:
  * tests/odesystems/_utils.py: entire file content removed (get_observables_list, random_system_values, create_random_test_set, create_minimal_input_sets, instantiate_or_use_instance, generate_system_tests)
  * tests/_utils.py: single_scale_float_array(), mixed_scale_float_array(), random_array()
- Section Header Removed:
  * tests/_utils.py: "RANDOM GENERATION" section header comment
- Implementation Summary:
  All unused random generation functions have been removed from tests/_utils.py. The tests/odesystems/_utils.py file has been emptied (content removed) as a deletion step - the file should be removed via git rm.
- Issues Flagged: 
  * tests/odesystems/_utils.py was emptied rather than fully deleted due to tool constraints. The file should be removed via `git rm tests/odesystems/_utils.py` or will appear as an empty file.

---

## Task Group 2: Implement Deterministic Generator
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/_utils.py (lines 1-20 for imports, lines 700-724 for generate_test_array)
- File: tests/outputhandling/test_output_functions.py (lines 130-148 for usage patterns, lines 627-639 for scale parameters)
- File: .github/active_plans/deterministic_test_fixtures/agent_plan.md (lines 38-60 for deterministic_array specification)

**Input Validation Required**:
- precision: Must be a valid numpy dtype (np.float32 or np.float64)
- size: Must be int or tuple of ints; empty arrays (size=0) should return empty array
- scale: Must be float, int, list, or tuple; interpret as magnitude guidance

**Tasks**:
1. **Add deterministic_array() function to tests/_utils.py**
   - File: tests/_utils.py
   - Action: Create new function after line 571 (after local_minima function)
   - Details:
     ```python
     def deterministic_array(
         precision,
         size: Union[int, tuple[int]],
         scale=1.0
     ):
         """Generate a deterministic array of numerically challenging values.
         
         Creates reproducible test arrays with values spanning multiple orders
         of magnitude, including edge cases like near-zero values, large values,
         and mathematically interesting constants (π, e).
         
         Parameters
         ----------
         precision : numpy.dtype
             The desired data type of the array (np.float32 or np.float64).
         size : int or tuple of int
             The shape of the array to generate.
         scale : float, int, list, or tuple, optional
             Guidance for value magnitudes. Default is 1.0.
             - Single number: Values centered around that magnitude
             - Tuple/list of two numbers: Interpreted as (min_exp, max_exp)
               for values spanning 10^min_exp to 10^max_exp
         
         Returns
         -------
         numpy.ndarray
             A deterministic array of the specified shape and dtype filled
             with numerically challenging values.
         
         Notes
         -----
         The generated values include:
         - Very small positive values (1e-12, 1e-9, 1e-6, 1e-3)
         - Values near unity (0.1, 0.5, 1.0, 2.0)
         - Mathematical constants (π, e)
         - Large values (1e3, 1e6, 1e9, 1e12)
         - Alternating signs for additional coverage
         
         Values are tiled/broadcast to fill the requested shape and filtered
         based on the scale parameter to stay within appropriate ranges.
         """
         # Handle empty arrays
         if isinstance(size, int):
             total_elements = size
             shape = (size,)
         else:
             shape = tuple(size)
             total_elements = 1
             for dim in shape:
                 total_elements *= dim
         
         if total_elements == 0:
             return np.empty(shape, dtype=precision)
         
         # Interpret scale parameter
         if isinstance(scale, (list, tuple)) and len(scale) == 2:
             min_exp, max_exp = scale
         else:
             # Single scale value: create range centered around it
             if isinstance(scale, (list, tuple)):
                 scale = scale[0]
             scale_exp = math.log10(abs(scale)) if scale != 0 else 0
             min_exp = scale_exp - 6
             max_exp = scale_exp + 6
         
         # Base set of challenging values (positive)
         base_values = [
             1e-12, 1e-9, 1e-6, 1e-3,
             0.1, 0.5, 1.0, 2.0,
             math.pi, math.e,
             1e3, 1e6, 1e9, 1e12,
         ]
         
         # Filter values to be within the scale range
         filtered_values = []
         for v in base_values:
             if v == 0:
                 filtered_values.append(v)
             else:
                 v_exp = math.log10(abs(v))
                 if min_exp <= v_exp <= max_exp:
                     filtered_values.append(v)
         
         # Ensure we have at least some values
         if not filtered_values:
             # Use scale-appropriate values if filter removed everything
             mid_exp = (min_exp + max_exp) / 2
             filtered_values = [
                 10 ** min_exp,
                 10 ** ((min_exp + mid_exp) / 2),
                 10 ** mid_exp,
                 10 ** ((mid_exp + max_exp) / 2),
                 10 ** max_exp,
             ]
         
         # Create array with alternating signs
         values_with_signs = []
         for i, v in enumerate(filtered_values):
             sign = 1 if i % 2 == 0 else -1
             values_with_signs.append(sign * v)
         
         # Tile values to fill the requested size
         num_base = len(values_with_signs)
         result = np.empty(total_elements, dtype=precision)
         for i in range(total_elements):
             result[i] = values_with_signs[i % num_base]
         
         return result.reshape(shape)
     ```
   - Edge cases:
     - Empty arrays (size=0): Return empty array with correct dtype
     - Single element arrays: Return first value from filtered set
     - Very large/small scale values: Clamp to float range via filtering
     - Negative scale values: Treat as magnitude (use abs)
   - Integration: Called by `generate_test_array()` for "random" style

2. **Modify generate_test_array() to use deterministic_array()**
   - File: tests/_utils.py
   - Action: Modify function (lines 700-724)
   - Details:
     - Change the "random" style case to call `deterministic_array()` instead of `random_array()`
     - Before:
       ```python
       if style == "random":
           if scale is None:
               raise ValueError("scale must be specified if type is 'random'.")
           return random_array(precision, size, scale)
       ```
     - After:
       ```python
       if style == "random":
           if scale is None:
               raise ValueError("scale must be specified if type is 'random'.")
           return deterministic_array(precision, size, scale)
       ```
   - Edge cases: None (function signature unchanged)
   - Integration: 
     - Used by tests/outputhandling/test_output_functions.py via `input_arrays` fixture
     - Interface preserved, no changes needed in test files

**Tests to Create**:
- Test file: tests/test_deterministic_array.py
- Test function: test_deterministic_array_reproducible
  - Description: Verify that calling deterministic_array with same parameters returns identical arrays
- Test function: test_deterministic_array_empty
  - Description: Verify that size=0 returns empty array with correct dtype
- Test function: test_deterministic_array_single_element
  - Description: Verify that size=1 returns array with single deterministic value
- Test function: test_deterministic_array_2d_shape
  - Description: Verify that 2D shapes are correctly filled
- Test function: test_deterministic_array_scale_filtering
  - Description: Verify that scale parameter filters values appropriately
- Test function: test_deterministic_array_alternating_signs
  - Description: Verify that generated values have alternating signs

**Tests to Run**:
- tests/test_deterministic_array.py (new tests)
- tests/outputhandling/test_output_functions.py::test_input_value_ranges
- tests/outputhandling/test_output_functions.py::test_all_summaries_long_run

**Outcomes**: 
- Files Modified:
  * tests/_utils.py (112 lines added for deterministic_array(), 1 line modified in generate_test_array())
- Functions/Methods Added/Modified:
  * deterministic_array() added to tests/_utils.py (lines 574-681)
  * generate_test_array() modified to call deterministic_array() instead of random_array() (line 734)
- Implementation Summary:
  Added deterministic_array() function that generates reproducible test arrays with numerically challenging values. The function accepts precision, size, and scale parameters. It creates values including very small positive (1e-12 to 1e-3), near unity (0.1 to 2.0), mathematical constants (π, e), and large values (1e3 to 1e12). Values are filtered based on scale parameter and alternating signs are applied. The generate_test_array() function now calls deterministic_array() for "random" style instead of the removed random_array().
- Issues Flagged: None

---

## Task Group 3: Verification
**Status**: [ ]
**Dependencies**: Task Groups 1 and 2

**Required Context**:
- File: tests/outputhandling/test_output_functions.py (entire file)
- File: tests/_utils.py (entire file after modifications)

**Input Validation Required**:
- None (verification only)

**Tasks**:
1. **Verify no import errors from deleted file**
   - Action: Verify tests can be imported without errors
   - Details:
     - Confirm `tests/odesystems/_utils.py` deletion doesn't cause import errors
     - The file was not imported by any active code
   - Edge cases: None
   - Integration: Full test suite import check

2. **Verify test_output_functions tests work with deterministic arrays**
   - Action: Run test_output_functions.py tests
   - Details:
     - Tests should pass consistently across multiple runs
     - The `input_arrays` fixture uses `generate_test_array()` with "random" style
     - Tests parameterized with different scales:
       - `{"random_scale": 1e-6}` - tiny values
       - `{"random_scale": 1e6}` - large values
       - `{"random_scale": [-12, 12]}` - wide range
   - Edge cases: Verify numerical edge cases are still exercised
   - Integration: Full integration with output functions

3. **Verify determinism by running tests multiple times**
   - Action: Run test_input_value_ranges multiple times
   - Details:
     - Same test with same parameters should produce identical results
     - No flaky failures due to random variation
   - Edge cases: None
   - Integration: Confirms fix for original flaky test issue

**Tests to Create**:
- None (verification uses existing tests)

**Tests to Run**:
- tests/outputhandling/test_output_functions.py::test_input_value_ranges
- tests/outputhandling/test_output_functions.py::test_all_summaries_long_run
- tests/outputhandling/test_output_functions.py::test_all_summaries_long_window
- tests/outputhandling/test_output_functions.py::test_memory_types

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

# Summary

## Task Groups Overview
| Group | Name | Tasks | Dependencies |
|-------|------|-------|--------------|
| 1 | Clean Up Unused Code | 5 | None |
| 2 | Implement Deterministic Generator | 2 | Group 1 |
| 3 | Verification | 3 | Groups 1, 2 |

## Dependency Chain
```
Task Group 1 (Clean Up)
        ↓
Task Group 2 (Implement)
        ↓
Task Group 3 (Verify)
```

## Tests Overview
- **New tests to create**: 6 tests in `tests/test_deterministic_array.py`
- **Existing tests to run**: Multiple tests in `tests/outputhandling/test_output_functions.py`

## Estimated Complexity
- **Task Group 1**: Low - straightforward file/function deletion
- **Task Group 2**: Medium - new function implementation with edge case handling
- **Task Group 3**: Low - verification and test runs

## Files Modified
| File | Action |
|------|--------|
| tests/odesystems/_utils.py | DELETE |
| tests/_utils.py | Remove 3 functions, add 1 function, modify 1 function |
| tests/test_deterministic_array.py | CREATE (new test file) |

## Risk Assessment
- **Low risk**: The deleted functions are confirmed unused
- **Low risk**: `generate_test_array()` interface preserved - no changes needed in test files
- **Medium risk**: Deterministic values must still exercise numerical edge cases effectively
