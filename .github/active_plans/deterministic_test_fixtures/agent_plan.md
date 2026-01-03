# Deterministic Test Fixtures: Agent Plan

## Component Overview

### Component 1: Deterministic Array Generator
**Location:** `tests/_utils.py`

**Purpose:** Generate deterministic, numerically challenging test arrays that replace the random generation.

**Expected Behavior:**
- Given a shape, precision, and scale parameter, produce the same array every time
- Array values should span the indicated scale range
- Include mathematically interesting values (near-zero, negative, large magnitudes)
- Values should tile/cycle if array is larger than the set of base values

**Architectural Changes:**
- Remove: `single_scale_float_array()`, `mixed_scale_float_array()`, `random_array()`
- Modify: `generate_test_array()` to call new deterministic generator for "random" style
- Add: `deterministic_array()` function that constructs challenging test data

### Component 2: Removed ODE System Test Utils
**Location:** `tests/odesystems/_utils.py`

**Purpose:** This file should be deleted entirely.

**Rationale:**
- `random_system_values()` - only used internally by other functions in same file
- `create_random_test_set()` - referenced only in commented-out code
- `create_minimal_input_sets()` - referenced only in commented-out code  
- `generate_system_tests()` - referenced only in commented-out code
- `instantiate_or_use_instance()` - only used internally
- `get_observables_list()` - only used internally

---

## Detailed Component Descriptions

### `deterministic_array()` Function

**Behavior:**
1. Accept precision (dtype), size (int or tuple), and scale parameter
2. Generate a base set of challenging values appropriate to the scale:
   - If scale is a single number: values centered around that magnitude
   - If scale is a tuple (min_exp, max_exp): values spanning 10^min_exp to 10^max_exp
3. Tile/broadcast these values to fill the requested shape
4. Apply alternating signs for additional coverage
5. Return array cast to requested precision

**Value Categories to Include:**
- Very small positive: 1e-15, 1e-12, 1e-9, 1e-6, 1e-3
- Near unity: 0.1, 0.5, 1.0, 2.0, π, e
- Large values: 1e3, 1e6, 1e9, 1e12
- Negative mirrors of all above
- Zero (at least one)

**Scale Interpretation:**
- `scale=1e6`: Values roughly in range [1e-6, 1e6] centered around scale
- `scale=1e-6`: Values roughly in range [1e-12, 1.0] biased toward small
- `scale=[−12, 12]`: Full range from 1e-12 to 1e12

### Modified `generate_test_array()` Function

**Current Signature:**
```python
def generate_test_array(precision, size, style, scale=None):
```

**Modified Behavior:**
- Style "random": Call `deterministic_array(precision, size, scale)` instead of `random_array()`
- Styles "nan", "zero", "ones": Keep unchanged (already deterministic)

**Rationale for keeping function:** 
- Maintains backward compatibility with existing test code
- No changes needed in `test_output_functions.py`

---

## Integration Points

### Test File: `test_output_functions.py`

**Current Usage (lines 140-147):**
```python
states = generate_test_array(
    precision, (num_samples, num_states), "random", scale
)
observables = generate_test_array(
    precision, (num_samples, num_observables), "random", scale
)
```

**Integration:** No changes needed. Function signature preserved.

**Test Parameterization Preserved:**
- `{"random_scale": 1e-6}` → tiny values
- `{"random_scale": 1e6}` → large values  
- `{"random_scale": [-12, 12]}` → wide range

---

## Edge Cases to Consider

1. **Empty arrays (size=0):** Return empty array of correct dtype
2. **Single element arrays:** Return first challenging value
3. **2D arrays with different shapes:** Tile appropriately along both axes
4. **Precision differences:** Ensure values don't overflow float32
5. **Scale parameter edge cases:**
   - Very large scale (1e15+): Clamp to float range
   - Very small scale (1e-15): Handle gracefully
   - Negative scale values: Treat as magnitude

---

## Dependencies and Imports

### Files Importing from `tests/_utils.py`:
1. `tests/outputhandling/test_output_functions.py` - imports `generate_test_array`, `calculate_expected_summaries`
2. `tests/odesystems/_utils.py` - imports `generate_test_array` (file to be deleted)

### Files Importing from `tests/odesystems/_utils.py`:
None (only commented-out imports in `test_symbolic_threeCM.py`)

---

## Task Sequence

### Task Group 1: Clean Up Unused Code
1. Delete `tests/odesystems/_utils.py` entirely
2. Remove from `tests/_utils.py`:
   - `single_scale_float_array()` function
   - `mixed_scale_float_array()` function
   - `random_array()` function

### Task Group 2: Implement Deterministic Generator
1. Add `deterministic_array()` function to `tests/_utils.py`
2. Modify `generate_test_array()` to use `deterministic_array()` for "random" style

### Task Group 3: Verification
1. Run `test_output_functions.py` tests multiple times to verify determinism
2. Confirm no import errors from deleted file
3. Verify tests still exercise numerical edge cases

---

## Data Structures

### Base Challenging Values (conceptual)
```python
# Values should include both positive and negative versions
_BASE_VALUES = [
    0.0,
    1e-15, 1e-12, 1e-9, 1e-6, 1e-3,
    0.1, 0.5, 1.0, 2.0,
    math.pi, math.e,
    1e3, 1e6, 1e9, 1e12, 1e15,
]
```

The actual implementation should:
- Filter values based on requested scale
- Apply sign alternation for coverage
- Tile to fill requested shape
