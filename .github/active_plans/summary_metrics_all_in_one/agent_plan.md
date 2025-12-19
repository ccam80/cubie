# Agent Plan: Summary Metrics Integration for all_in_one.py

## Objective

Integrate complete inline implementations of summary metrics into all_in_one.py debug script, matching package source verbatim to enable NVIDIA profiler debugging.

## Component Specifications

### 1. Summary Metric Device Functions

**Location in all_in_one.py**: After output functions section (after line 3475)

**Components to Inline**:

1. **Mean Metric Update Function**
   - Source: `src/cubie/outputhandling/summarymetrics/mean.py` lines 56-87
   - Signature: `update_mean(value, buffer, current_index, customisable_variable)`
   - Behavior: Accumulates value into buffer[0]
   - Buffer size: 1 slot per variable
   - No modifications - copy verbatim

2. **Mean Metric Save Function**
   - Source: `src/cubie/outputhandling/summarymetrics/mean.py` lines 89-123
   - Signature: `save_mean(buffer, output_array, summarise_every, customisable_variable)`
   - Behavior: Divides buffer[0] by summarise_every, saves to output_array[0], resets buffer[0]
   - Output size: 1 value per variable
   - No modifications - copy verbatim

**Integration Notes**:
- Use `@cuda.jit(device=True, inline=True, **compile_kwargs)` decorators
- Precision parameter captured from module-level `precision` variable
- Remove import statements, keep all logic identical

### 2. Chaining Factory Functions

**Location in all_in_one.py**: After metric device functions

**Components to Inline**:

1. **do_nothing Function (Update Version)**
   - Source: `src/cubie/outputhandling/update_summaries.py` lines 29-61
   - Signature: `do_nothing(values, buffer, current_step)`
   - Behavior: No-op base case for empty chains
   - Copy verbatim with decorators

2. **chain_update_metrics Function**
   - Source: Inline simplified version based on `update_summaries.py` chain_metrics
   - Hardcoded for mean metric only
   - Buffer offset: 0
   - Buffer size: 1
   - Param: 0
   - Calls update_mean with appropriate slices

3. **do_nothing Function (Save Version)**
   - Source: `src/cubie/outputhandling/save_summaries.py` lines 29-60
   - Signature: `do_nothing(buffer, output, summarise_every)`
   - Behavior: No-op base case
   - Copy verbatim

4. **chain_save_metrics Function**
   - Source: Inline simplified version based on `save_summaries.py` chain_metrics
   - Hardcoded for mean metric only
   - Buffer offset: 0, size: 1
   - Output offset: 0, size: 1
   - Param: 0
   - Calls save_mean with appropriate slices

**Integration Notes**:
- Pattern follows recursive chaining from package
- Simplified to single metric (mean) - no recursion needed
- Uses same slice syntax as package
- Device function decorators with inline=True

### 3. Summary Update/Save Wrapper Functions

**Location in all_in_one.py**: After chaining functions

**Components**:

1. **update_summaries_inline**
   - Source: Pattern from `update_summaries.py` lines 224-278
   - Signature: `update_summaries_inline(current_state, current_observables, state_summary_buffer, obs_summary_buffer, current_step)`
   - Behavior:
     * Iterates through n_states, calls chain_update_metrics for each
     * Uses buffer stride: `idx * total_buffer_size`
     * Only executes if `summarise_state_bool` is True (compile-time branch)
   - Device function with inline=True

2. **save_summaries_inline**
   - Source: Pattern from `save_summaries.py` lines 257-324
   - Signature: `save_summaries_inline(buffer_state, buffer_obs, output_state, output_obs, summarise_every)`
   - Behavior:
     * Iterates through variables, calls chain_save_metrics
     * Uses buffer and output strides
     * Resets buffers via save function
     * Only executes if summarise flags are True
   - Device function with inline=True

**Integration Notes**:
- n_states32 defined at module level: `n_states32 = int32(n_states)`
- total_buffer_size = 1 (for mean metric)
- total_output_size = 1 (for mean metric)
- Compile-time branching based on summarise_state_bool

### 4. Output Type Configuration System

**Location in all_in_one.py**: In configuration section (around line 147)

**New Configuration Variables**:

```python
# Output types to generate
output_types = ['state', 'summaries', 'counters']  # Configurable list

# Derived boolean toggles
save_state_bool = 'state' in output_types
save_obs_bool = 'observables' in output_types
save_counters_bool = 'counters' in output_types
summarise_state_bool = 'summaries' in output_types and 'state_summaries' not in output_types
summarise_obs_bool = 'summaries' in output_types and 'obs_summaries' not in output_types
save_last = 'last_state' in output_types

# Summary configuration
saves_per_summary = int32(2)  # Already exists
```

**Behavior**:
- Replace existing hardcoded boolean flags
- Support combinations: `['state', 'summaries', 'counters']`
- Explicit summary types: `'state_summaries'`, `'obs_summaries'`
- `'summaries'` alone enables both state and observable summaries

**Integration Notes**:
- Maintains backward compatibility with existing flags
- Clear, list-based configuration
- Easy to see what outputs are enabled

### 5. Buffer Allocation Integration

**Location in all_in_one.py**: Buffer layout section (around line 3870)

**New Buffer Sizing**:

```python
# Summary buffer sizes (conditional on flags)
state_summ_size = int32(n_states) if summarise_state_bool else int32(0)
obs_summ_size = int32(n_observables) if summarise_obs_bool else int32(0)
```

**Already exists in all_in_one.py** - verify correct usage:
- Lines 4180-4210: State/observable summary buffer allocation
- Uses `use_shared_loop_state_summary` flag
- Follows same local/shared pattern as other buffers

**No changes needed** - existing buffer allocation is correct

### 6. Loop Integration Points

**Location in all_in_one.py**: loop_fn device function (around line 4108)

**Integration Points**:

1. **Initial Summary Save** (after initial state save, ~line 4300):
   ```python
   if summarise:
       save_summaries_inline(state_summary_buffer,
                            observable_summary_buffer,
                            state_summaries_output[
                                summary_idx * summarise_state_bool, :
                            ],
                            observable_summaries_output[
                                summary_idx * summarise_obs_bool, :
                            ],
                            saves_per_summary)
   ```

2. **Summary Update and Periodic Save** (in main loop after state save, ~line 4450):
   ```python
   if summarise:
       update_summaries_inline(
           state_buffer,
           observables_buffer,
           state_summary_buffer,
           observable_summary_buffer,
           save_idx,
       )
       
       if (save_idx % saves_per_summary == int32(0)):
           save_summaries_inline(
               state_summary_buffer,
               observable_summary_buffer,
               state_summaries_output[
                   summary_idx * summarise_state_bool, :
               ],
               observable_summaries_output[
                   summary_idx * summarise_obs_bool, :
               ],
               saves_per_summary,
           )
           summary_idx += int32(1)
   ```

**Integration Notes**:
- Pattern matches ode_loop.py exactly (lines 1104-1114, 1271-1291)
- Uses existing `summarise` flag (derived from summarise_state_bool or summarise_obs_bool)
- Predicated array indexing with boolean multiplication
- summary_idx already exists and is tracked

### 7. Output Array Sizing

**Location in all_in_one.py**: run_debug_integration function (around line 4593)

**Existing Code** (verify correct):
- Lines 4610-4621: Summary output array allocation
- Uses `n_summary_samples = int(ceil(n_output_samples / saves_per_summary))`
- Allocates with shape `(n_summary_samples, n_states, n_runs)`

**No changes needed** - existing allocation is correct

## Expected Behavior

### Compilation
- All summary metric functions compile as inline CUDA device functions
- Chaining pattern produces single callable per operation
- No runtime overhead from chain traversal

### Runtime Execution
1. **Initial State**:
   - Buffers initialized to zero
   - First summary save writes zeros (reset state)

2. **During Integration**:
   - After each state save, update_summaries accumulates into buffers
   - Every `saves_per_summary` interval, save_summaries:
     * Computes final metrics (mean = sum/count)
     * Writes to output arrays
     * Resets buffers to zero
   - summary_idx increments after each save

3. **Final State**:
   - Summary output arrays contain mean values over windows
   - Number of summaries = `ceil(n_output_samples / saves_per_summary)`

### Edge Cases

1. **No Summaries Requested**:
   - summarise_state_bool = False, summarise_obs_bool = False
   - summarise = False
   - Summary code paths never execute
   - Buffer allocations are zero-sized

2. **Summaries Only**:
   - output_types = ['summaries']
   - save_state_bool = False
   - Only summary outputs allocated
   - Loop still tracks saves for summary window calculation

3. **First Window Incomplete**:
   - If n_output_samples < saves_per_summary
   - Only initial zero-summary written
   - No subsequent summaries (no window completes)

## Dependencies

### Existing Variables (Must Already Exist)
- `n_states`, `n_observables`
- `n_output_samples`, `saves_per_summary`
- `precision`, `numba_precision`
- `compile_kwargs`
- Buffer location flags (use_shared_loop_state_summary, etc.)

### New Variables (To Add)
- `n_states32 = int32(n_states)` (module level, near line 3363)
- `n_counters32 = int32(n_counters)` (already exists at line 3364)
- `summarise` flag derivation (near line 284): `summarise = summarise_obs_bool or summarise_state_bool`

### Module-Level Constants
- `total_buffer_size = int32(1)` (for mean metric, in wrapper functions)
- `total_output_size = int32(1)` (for mean metric, in wrapper functions)

## Verification Points

1. **Compilation**: Script compiles without errors
2. **Buffer Sizing**: Summary buffer heights match metric requirements (1 for mean)
3. **Output Shape**: Summary output arrays have correct dimensions
4. **Execution**: Integration runs without CUDA errors
5. **Results**: Summary outputs contain non-zero mean values after first window
6. **Verbatim Match**: All copied code matches package source exactly

## Integration Order

1. Add output_types configuration system
2. Inline mean metric device functions
3. Inline chaining factory functions  
4. Add update_summaries_inline wrapper
5. Add save_summaries_inline wrapper
6. Integrate calls into loop_fn
7. Verify buffer allocation
8. Test execution

## Testing Strategy

1. **Minimal Configuration**: `output_types = ['state', 'summaries']`
2. **Run Parameters**: Small n_runs (2^10), short duration (0.2s)
3. **Verify**:
   - No compilation errors
   - No runtime CUDA errors
   - Summary outputs have expected shape
   - Mean values are reasonable (non-zero after warmup)
4. **Compare**: Results should approximate state trajectory means over windows

## Notes for Implementer

- **DO NOT** modify any copied code from package
- **DO** remove import statements from copied code
- **DO** use existing module-level precision variables
- **DO** maintain exact function signatures from package
- **DO** preserve all comments and docstrings from package source
- **DO** use same decorator pattern as package (`@cuda.jit(device=True, inline=True, **compile_kwargs)`)
- **DO** follow existing code style and formatting in all_in_one.py
- **DO NOT** add new imports
- **DO NOT** create helper functions not in package
- **DO NOT** modify existing algorithm or controller code

## File Locations Reference

### Package Source Files
- `/home/runner/work/cubie/cubie/src/cubie/outputhandling/update_summaries.py`
- `/home/runner/work/cubie/cubie/src/cubie/outputhandling/save_summaries.py`
- `/home/runner/work/cubie/cubie/src/cubie/outputhandling/summarymetrics/mean.py`
- `/home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop.py` (integration pattern)

### Target File
- `/home/runner/work/cubie/cubie/tests/all_in_one.py`

## Success Criteria Mapping

### User Story 1: Inline Summary Metrics Factory
- Mean update and save functions inlined ✓
- Chaining functions implemented ✓
- Code matches package verbatim ✓

### User Story 2: Configuration System
- output_types list configuration ✓
- Boolean toggle derivation ✓
- Toggles flow through to loop ✓

### User Story 3: Chaining Integration
- Chained update function created ✓
- Chained save function created ✓
- Loop calls at correct points ✓
- Buffers properly allocated ✓
