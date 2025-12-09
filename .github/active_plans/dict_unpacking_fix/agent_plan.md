# Dict Unpacking Fix - Agent Plan

## Problem Statement

The current implementation unpacks dict values at `CUDAFactory.update_compile_settings()`, which is at the leaf level of the update() call chain. This causes incorrect parameter recognition because:

1. When `BatchSolverKernel.update(step_controller_settings={'dt_min': 0.01})` is called
2. The dict `{'step_controller_settings': {'dt_min': 0.01}}` flows down through intermediate update() methods unchanged
3. At the leaf (CUDAFactory), the dict is unpacked to `{'dt_min': 0.01}`
4. Both 'step_controller_settings' and 'dt_min' are marked as recognized
5. This recognized set bubbles back up, but 'dt_min' was never actually present at intermediate levels

This violates the contract that update() methods should only recognize parameters they actually receive in their updates_dict argument.

## Architectural Changes Required

### 1. Create Shared Unpacking Utility

Extract the `_unpack_dict_values()` method from CUDAFactory to a shared utility function that can be used by all update() methods.

**Location:** `src/cubie/_utils.py`

**Function signature:**
```python
def unpack_dict_values(updates_dict: dict) -> Tuple[dict, Set[str]]:
    """Unpack dict values into flat key-value pairs.
    
    Parameters
    ----------
    updates_dict
        Dictionary potentially containing dicts as values
    
    Returns
    -------
    Tuple[dict, Set[str]]
        Flattened dictionary with dict values unpacked, and set of
        original keys that were unpacked dicts
    """
```

**Behavior:**
- Iterate through updates_dict items
- If value is a dict, unpack its contents into result dict
- Track keys whose values were dicts
- Return (flattened_dict, unpacked_keys_set)

### 2. Modify Top-Level update() Methods

Three update() methods need to be modified to unpack dict values before distributing to sub-components:

#### A. BatchSolverKernel.update()
**File:** `src/cubie/batchsolving/BatchSolverKernel.py`  
**Line:** 667

**Changes:**
1. After merging updates_dict and kwargs (line 704)
2. Call `updates_dict, unpacked_keys = unpack_dict_values(updates_dict)`
3. Track unpacked_keys separately from all_unrecognized
4. Before returning recognized set (line 727), add unpacked_keys to recognized

**Integration points:**
- Calls `single_integrator.update()` (line 709)
- Calls `update_compile_settings()` (line 723)

#### B. SingleIntegratorRunCore.update()
**File:** `src/cubie/integrators/SingleIntegratorRunCore.py`  
**Line:** 370

**Changes:**
1. After merging updates_dict and kwargs (line 410)
2. Call `updates_dict, unpacked_keys = unpack_dict_values(updates_dict)`
3. Track unpacked_keys separately from all_unrecognized
4. Before returning recognized set (line 489), add unpacked_keys to recognized

**Integration points:**
- Calls `_system.update()` (line 416)
- Calls `_output_functions.update()` (line 421)
- Calls `_algo_step.update()` (line 431)
- Calls `_step_controller.update()` (line 440)
- Calls `_loop.update()` (line 475)
- Calls `update_compile_settings()` (line 476)

#### C. IVPLoop.update()
**File:** `src/cubie/integrators/loops/ode_loop.py`  
**Line:** 1490

**Changes:**
1. After merging updates_dict and kwargs (line 1516)
2. Call `updates_dict, unpacked_keys = unpack_dict_values(updates_dict)`
3. Track unpacked_keys to add to recognized set
4. Before returning recognized set (line 1527), add unpacked_keys to recognized

**Integration points:**
- Calls `update_compile_settings()` (line 1520)

### 3. Remove Unpacking from CUDAFactory

**File:** `src/cubie/CUDAFactory.py`  
**Method:** `update_compile_settings()`  
**Line:** 547

**Changes:**
1. Remove call to `_unpack_dict_values()` at line 599
2. Remove `unpacked_keys` tracking (lines 623-624)
3. Keep the `_unpack_dict_values()` method but mark it as deprecated or remove it after confirming no other usages
4. Simplify recognition logic since no unpacking occurs

**Expected behavior after change:**
- `update_compile_settings()` receives already-unpacked parameters
- Only marks parameters as recognized if they match compile settings attributes
- No special handling for dict values

### 4. Update Import Statements

All files that call the unpacking utility need to import it from `_utils`:

**Files to update:**
- `src/cubie/batchsolving/BatchSolverKernel.py`
- `src/cubie/integrators/SingleIntegratorRunCore.py`
- `src/cubie/integrators/loops/ode_loop.py`

**Import statement:**
```python
from cubie._utils import unpack_dict_values
```

## Data Structures

### updates_dict Flow

**Before unpacking:**
```python
{
    'step_controller_settings': {'dt_min': 0.01, 'dt_max': 1.0},
    'precision': np.float32,
    'algorithm': 'rk4'
}
```

**After unpacking:**
```python
updates_dict = {
    'dt_min': 0.01,
    'dt_max': 1.0,
    'precision': np.float32,
    'algorithm': 'rk4'
}

unpacked_keys = {'step_controller_settings'}
```

### Recognition Tracking Pattern

Each update() method maintains:
- `all_unrecognized`: Set of all parameter keys (initialized from updates_dict.keys())
- `recognized`: Set of parameters recognized by this method and its sub-components
- `unpacked_keys`: Set of dict keys that were unpacked (marked as recognized immediately)

**Recognition flow:**
```python
# After unpacking
updates_dict, unpacked_keys = unpack_dict_values(updates_dict)
all_unrecognized = set(updates_dict.keys())

# Call sub-components
sub_recognized = sub_component.update(updates_dict, silent=True)
all_unrecognized -= sub_recognized

# Before returning
recognized = (set(updates_dict.keys()) - all_unrecognized) | unpacked_keys
return recognized
```

## Edge Cases to Consider

### 1. Nested Dicts
**Input:** `update(settings={'controller_settings': {'dt_min': 0.01}})`

**Expected behavior:**
- First unpacking: 'settings' key unpacked → `{'controller_settings': {'dt_min': 0.01}}`
- Second unpacking (next level): 'controller_settings' key unpacked → `{'dt_min': 0.01}`
- Recognition: 'settings' marked at first level, 'controller_settings' at second level, 'dt_min' at leaf

**Implementation note:** Each level only unpacks one layer, so nested dicts are handled naturally by the call chain.

### 2. Dict Key Collision
**Input:** `update(dt_min=0.05, step_settings={'dt_min': 0.01})`

**Expected behavior:**
- After unpacking: `{'dt_min': 0.05, 'dt_min': 0.01}` → dict merge rules apply
- Last value wins (Python dict update semantics)
- Both 'dt_min' and 'step_settings' marked as recognized

**Implementation note:** This is existing Python behavior, no special handling needed. Users should be aware of potential conflicts.

### 3. Empty Dict Values
**Input:** `update(step_settings={})`

**Expected behavior:**
- 'step_settings' marked as recognized (it's a valid dict key)
- Empty dict unpacks to nothing, no sub-component parameters

**Implementation note:** Unpacking logic should handle empty dicts gracefully.

### 4. Non-Dict Values
**Input:** `update(dt_min=0.01, algorithm='rk4')`

**Expected behavior:**
- No unpacking occurs
- Parameters passed directly to sub-components
- Recognition tracking works as before

**Implementation note:** Unpacking only affects dict values, not other types.

### 5. Silent Mode with Invalid Dict Contents
**Input:** `update(step_settings={'invalid_param': 123}, silent=True)`

**Expected behavior:**
- 'step_settings' marked as recognized
- 'invalid_param' attempted to pass to sub-components
- Sub-components return it as unrecognized
- With silent=True, no error raised
- 'invalid_param' included in unrecognized set but suppressed

**Implementation note:** Silent mode behavior is preserved, errors only raised when silent=False.

## Dependencies and Imports Required

### New Imports
- All three update() methods need: `from cubie._utils import unpack_dict_values`
- Type hints may need: `from typing import Set, Tuple` (likely already present)

### Existing Dependencies
- No new external dependencies
- Reuses existing dict unpacking logic from CUDAFactory

## Integration Points with Current Codebase

### Component Hierarchy

```
BatchSolverKernel.update()
  ├─> single_integrator.update() [SingleIntegratorRunCore]
  │     ├─> _system.update() [BaseODE]
  │     │     └─> update_compile_settings() [CUDAFactory]
  │     ├─> _output_functions.update() [OutputFunctions]
  │     │     └─> update_compile_settings() [CUDAFactory]
  │     ├─> _algo_step.update() [BaseAlgorithmStep]
  │     │     └─> update_compile_settings() [CUDAFactory]
  │     ├─> _step_controller.update() [BaseStepController]
  │     │     └─> update_compile_settings() [CUDAFactory]
  │     ├─> _loop.update() [IVPLoop]
  │     │     └─> update_compile_settings() [CUDAFactory]
  │     └─> update_compile_settings() [CUDAFactory]
  └─> update_compile_settings() [CUDAFactory]
```

**Unpacking will occur at:**
- BatchSolverKernel.update() (top level)
- SingleIntegratorRunCore.update() (middle level)
- IVPLoop.update() (component level)

**No unpacking at:**
- CUDAFactory.update_compile_settings() (leaf level)
- BaseStepController.update() (component level, delegates to CUDAFactory)
- Other component update() methods (they delegate to CUDAFactory)

### Recognition Flow Example

**Call:** `BatchSolverKernel.update(step_controller_settings={'dt_min': 0.01})`

**Step-by-step:**
1. BatchSolverKernel.update():
   - Unpacks: `{'dt_min': 0.01}`, unpacked_keys={'step_controller_settings'}
   - Passes `{'dt_min': 0.01}` to single_integrator.update()
   - single_integrator returns recognized={'dt_min'}
   - BatchSolverKernel adds 'step_controller_settings' to recognized
   - Returns recognized={'dt_min', 'step_controller_settings'}

2. SingleIntegratorRunCore.update():
   - Receives: `{'dt_min': 0.01}`
   - No unpacking needed (not a dict value)
   - Passes to _step_controller.update()
   - _step_controller returns recognized={'dt_min'}
   - Returns recognized={'dt_min'}

3. BaseStepController.update():
   - Receives: `{'dt_min': 0.01}`
   - Passes to update_compile_settings()
   - update_compile_settings returns recognized={'dt_min'}
   - Returns recognized={'dt_min'}

4. CUDAFactory.update_compile_settings():
   - Receives: `{'dt_min': 0.01}`
   - No unpacking (already done at top)
   - Checks if 'dt_min' is in compile_settings attributes
   - Returns recognized={'dt_min'}

## Testing Considerations

### Existing Tests to Verify

**File:** `tests/test_CUDAFactory.py`
- `test_update_compile_settings_unpack_dict_values()` (line 445)
- `test_update_compile_settings_unpack_multiple_dicts()` (line 474)
- `test_update_compile_settings_unpack_dict_with_regular_updates()` (line 508)

**Expected behavior:** These tests should FAIL after removing unpacking from CUDAFactory, because they test unpacking at the CUDAFactory level. The tests may need to be updated to test the new behavior (unpacking at higher levels) or moved to integration tests.

### New Tests Needed

1. **Test unpacking at BatchSolverKernel level:**
   - Dict value unpacking before distribution to single_integrator
   - Dict key recognition tracking

2. **Test unpacking at SingleIntegratorRunCore level:**
   - Dict value unpacking before distribution to sub-components
   - Mixed dict and direct parameters

3. **Test unpacking at IVPLoop level:**
   - Dict value unpacking before distribution to compile settings

4. **Test full chain:**
   - End-to-end test from BatchSolverKernel to leaf components
   - Verify correct recognition at each level

### Test Modification Strategy

Rather than modifying existing CUDAFactory tests, consider:
1. Moving them to integration tests that test the full update chain
2. Adding new unit tests at each level (BatchSolverKernel, SingleIntegratorRunCore, IVPLoop)
3. Ensuring coverage for all edge cases listed above

## Implementation Sequence

1. Create `unpack_dict_values()` in `_utils.py`
2. Update `BatchSolverKernel.update()` to use new utility
3. Update `SingleIntegratorRunCore.update()` to use new utility
4. Update `IVPLoop.update()` to use new utility
5. Remove unpacking from `CUDAFactory.update_compile_settings()`
6. Run tests and fix any failures
7. Add new tests for unpacking behavior at each level

## Behavior Contracts

### unpack_dict_values()
**Input:** Dictionary (may contain dict values)  
**Output:** (flattened_dict, unpacked_keys_set)  
**Contract:**
- Returns flattened dict with dict values unpacked one level deep
- Returns set of keys whose values were dicts
- Non-dict values pass through unchanged
- Empty dicts are handled gracefully
- Order of unpacking follows dict iteration order

### update() methods (BatchSolverKernel, SingleIntegratorRunCore, IVPLoop)
**Input:** updates_dict (may contain dict values), silent flag, kwargs  
**Output:** set of recognized parameter names  
**Contract:**
- Unpacks dict values before distributing to sub-components
- Marks unpacked dict keys as recognized immediately
- Only marks unpacked contents as recognized if sub-components recognize them
- Returns union of: sub-component recognized params + unpacked dict keys
- Raises KeyError for unrecognized params when silent=False

### CUDAFactory.update_compile_settings()
**Input:** updates_dict (no dict values, already unpacked), silent flag, kwargs  
**Output:** set of recognized parameter names  
**Contract:**
- No unpacking of dict values (done at higher levels)
- Only recognizes parameters that match compile_settings attributes
- Invalidates cache when recognized parameters are updated
- Raises KeyError for unrecognized params when silent=False
