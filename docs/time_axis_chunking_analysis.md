# Time Axis Chunking Analysis Report

## Executive Summary

This report analyzes two options for handling the time-axis chunking issue in CuBIE:

| Metric | Option 1: Modify Time-Axis Chunking | Option 2: Remove Time-Axis Chunking |
|--------|-------------------------------------|-------------------------------------|
| **Estimated Work** | ~40-60 hours | ~8-12 hours |
| **Files Modified** | 8-12 files | 12-15 files (mostly deletions) |
| **Risk Level** | High (complex edge cases) | Low (simplification) |
| **Test Changes** | Significant new tests needed | Remove existing tests |
| **Complexity Change** | Increases | Decreases |
| **Lines Added** | ~200-400 | ~0 |
| **Lines Removed** | ~0 | ~150-250 |

**Recommendation**: Remove time-axis chunking (Option 2) unless a specific use case emerges that requires it.

---

## Problem Statement

Time-axis chunking currently assumes all output arrays have the same time-axis length. However:

- **Regular saved outputs** use `save_every` parameter → `output_length = floor(duration / save_every) + initial + final`
- **Summary outputs** use `summarise_every` parameter → `summaries_length = floor(duration / summarise_every)`

These lengths are typically incommensurate (e.g., 1000 samples vs 10 summary intervals), making it difficult to find a common chunk size that divides evenly into both.

---

## Current Time-Axis Chunking Architecture

### Entry Points

```
Solver.solve(chunk_axis="time")
    └── BatchSolverKernel.run(chunk_axis="time")
        └── MemoryManager.allocate_queue(chunk_axis="time")
            └── get_chunk_parameters()
            └── compute_chunked_shapes()
            └── compute_per_chunk_slice()
```

### Key Files Involved

1. **`src/cubie/batchsolving/solver.py`** (lines 351, 384, 450)
   - Exposes `chunk_axis` parameter to users
   - Passes to `BatchSolverKernel.run()`

2. **`src/cubie/batchsolving/BatchSolverKernel.py`** (lines 77-125, 184-203, 506-605)
   - `FullRunParams.chunk_axis` - stores user choice
   - `ChunkParams.__getitem__()` - calculates per-chunk t0, duration, warmup for time-axis
   - `run()` - validates timing parameters after chunking
   - Error handling for time-chunk incompatibility (lines 593-605)

3. **`src/cubie/memory/mem_manager.py`** (lines 1157-1377, 1438-1573)
   - `allocate_queue(chunk_axis)` - dispatches chunking
   - `get_chunk_parameters()` - calculates chunk size
   - `compute_chunked_shapes()` - computes per-array chunked shapes
   - `compute_per_chunk_slice()` - generates slice functions for each chunk
   - `is_request_chunkable()` - determines if array can be chunked on axis
   - `replace_with_chunked_size()` - modifies shape tuple

4. **`src/cubie/memory/array_requests.py`** (lines 111, 123, 141-142)
   - `ArrayResponse.chunk_axis` - communicates chunk axis to callbacks

5. **`src/cubie/batchsolving/arrays/BaseArrayManager.py`** (lines 226-228, 261-262, 368)
   - `_chunk_axis` attribute storage
   - Validator includes "time" option

6. **`src/cubie/batchsolving/arrays/BatchOutputArrays.py`** (lines 45, 52, 59, 66, 80)
   - Output array stride orders include "time" axis

7. **`tests/batchsolving/arrays/test_chunking.py`** (lines 46-52, 64, 78, 116, 142)
   - Parametrized tests for "time" axis chunking

8. **`tests/memory/test_memmgmt.py`** (lines 920, 936)
   - Tests for time-axis chunking in memory manager

---

## Option 1: Modify Time-Axis Chunking for Summary Compatibility

### Required Changes

#### 1. Separate Chunk Size Calculation for Different Array Types

**File**: `src/cubie/memory/mem_manager.py`

**Current**: Single `chunk_length` for all arrays  
**Required**: Track separate chunk lengths for:
- Time-domain outputs (state, observables, iteration_counters)
- Summary outputs (state_summaries, observable_summaries)

```python
# Current (simplified)
chunk_length, num_chunks = get_chunk_parameters(requests, chunk_axis, axis_length)

# Required
chunk_params_regular = get_chunk_parameters(regular_requests, chunk_axis, regular_axis_length)
chunk_params_summary = get_chunk_parameters(summary_requests, chunk_axis, summary_axis_length)
# Need to reconcile these into compatible chunking
```

**Estimated Changes**: ~80-120 lines added/modified in `mem_manager.py`

#### 2. Handle Incommensurate Chunk Sizes

The core problem: if `output_length=1000` and `summaries_length=10`, you can't use the same chunk size.

**Approaches**:

**(A) Compute Common Divisor**:
- Find GCD of output_length and summaries_length
- Divide both by GCD to get compatible chunks
- Problem: GCD may be 1, forcing single-sample chunks (defeats purpose)

**(B) Allow Different Chunk Counts Per Array Type**:
- Regular outputs: 5 chunks of 200 samples
- Summary outputs: 2 chunks of 5 intervals
- Problem: Requires fundamentally different chunk handling per array

**(C) Chunk Only Regular Outputs, Keep Summaries Unchunkable**:
- Mark summary arrays as `unchunkable=True`
- Summaries stay full-size while regular outputs chunk
- Problem: If summaries are large, defeats memory savings

**Estimated Changes**: ~60-100 lines in `mem_manager.py` and `array_requests.py`

#### 3. Update ChunkParams for Dual Chunk Tracking

**File**: `src/cubie/batchsolving/BatchSolverKernel.py`

```python
@define(frozen=True)
class ChunkParams:
    # Add separate tracking for summary outputs
    _summary_chunk_length: int = field(default=0, repr=False)
    _summary_num_chunks: int = field(default=1, repr=False)
```

**Estimated Changes**: ~30-50 lines

#### 4. Modify Slice Functions for Summary Arrays

**File**: `src/cubie/memory/mem_manager.py`

The `compute_per_chunk_slice()` function needs separate logic for summary vs regular arrays.

**Estimated Changes**: ~40-60 lines

#### 5. Coordinate Transfers for Different Chunk Counts

**File**: `src/cubie/batchsolving/arrays/BatchOutputArrays.py`

If regular outputs have 5 chunks but summaries have 2, the `initialise()` and `finalise()` methods need to handle this mismatch.

**Estimated Changes**: ~50-80 lines

#### 6. Update Tests

**File**: `tests/batchsolving/arrays/test_chunking.py`

- Add tests for incommensurate save/summarise intervals
- Add tests for summary-only chunking
- Add edge case tests

**Estimated Changes**: ~100-150 lines

### Total Estimated Work: Option 1

| Category | Lines | Hours |
|----------|-------|-------|
| mem_manager.py modifications | 140-220 | 15-25 |
| ChunkParams updates | 30-50 | 4-6 |
| Slice function changes | 40-60 | 5-8 |
| Transfer coordination | 50-80 | 8-12 |
| New tests | 100-150 | 8-12 |
| **Total** | **360-560** | **40-63** |

### Risks and Challenges

1. **Edge Cases**: Many combinations of save_every, summarise_every, duration
2. **Performance**: Additional bookkeeping overhead
3. **Debugging**: Complex state to track across chunks
4. **Regression Risk**: Existing functionality may break

---

## Option 2: Remove Time-Axis Chunking Entirely

### Required Changes

#### 1. Remove "time" from chunk_axis validators

**Files and Changes**:

| File | Location | Change |
|------|----------|--------|
| `src/cubie/batchsolving/solver.py` | line 351 | Remove "time" option from docstring |
| `src/cubie/batchsolving/BatchSolverKernel.py` | lines 77-78, 112-113 | Remove "time" from docstrings |
| `src/cubie/memory/array_requests.py` | line 142 | Change validator to `in_(["run", "variable"])` |
| `src/cubie/batchsolving/arrays/BaseArrayManager.py` | line 262 | Change validator to `in_(["run", "variable"])` |

**Estimated Changes**: 4 files, ~8 lines modified

#### 2. Remove time-axis specific logic in ChunkParams

**File**: `src/cubie/batchsolving/BatchSolverKernel.py`

```python
# REMOVE (lines 187-196):
elif self._chunk_axis == "time":
    # Calculate per-chunk t0 and duration
    dt_save = float64(_duration / self._axis_length)
    fullchunk_duration = float64(dt_save * self._chunk_length)
    _duration = float64(dt_save * length)
    if index > 0:
        _warmup = 0.0
        _t0 = (
            self._full_params.t0 + float64(index) * fullchunk_duration
        )
```

**Estimated Changes**: ~12 lines removed

#### 3. Remove time-chunk validation logic

**File**: `src/cubie/batchsolving/BatchSolverKernel.py`

```python
# REMOVE (lines 593-605):
if chunks != 1 and chunk_axis == "time":
    self.single_integrator.set_summary_timing_from_duration(duration)
    try:
        self._validate_timing_parameters(duration)
    except ValueError as e:
        raise ValueError(
            f"Your timing parameters were OK for the full duration, "
            f"but the run was divided into multiple time-chunks due "
            f"to GPU memory constraints so they're now invalid. "
            f"Adjust timing parameters OR set chunk_axis='run' to "
            f"avoid this. Time-check exception: {e}"
        ) from e
```

**Estimated Changes**: ~13 lines removed

#### 4. Simplify initialization comment

**File**: `src/cubie/batchsolving/arrays/BatchOutputArrays.py`

```python
# SIMPLIFY (lines 477-479):
# Current:
# No initialization to zeros is needed unless chunk calculations in time
# leave a dangling sample at the end, which is possible but not expected.

# Replace with:
# No initialization to zeros is needed for run-axis chunking.
```

**Estimated Changes**: ~3 lines modified

#### 5. Update/Remove Tests

**File**: `tests/batchsolving/arrays/test_chunking.py`

| Test | Action |
|------|--------|
| `test_chunk_axis_property_raises_on_inconsistency` | Modify to use "variable" instead of "time" |
| `@pytest.mark.parametrize("chunk_axis", ["run", "time"])` | Change to `["run"]` only |
| Lines 46, 52, 64, 78, 116, 142 | Remove "time" from parametrizations |

**Estimated Changes**: ~6 test parametrizations simplified

**File**: `tests/memory/test_memmgmt.py`

| Test | Action |
|------|--------|
| Lines 920, 936 | Remove or modify time-axis specific tests |

**Estimated Changes**: ~20 lines removed/modified

#### 6. Update Documentation

- Remove any references to `chunk_axis="time"` in docstrings
- Update any user-facing documentation

**Estimated Changes**: ~10-15 lines across multiple files

### Total Estimated Work: Option 2

| Category | Lines Removed | Lines Modified | Hours |
|----------|---------------|----------------|-------|
| Validator changes | 0 | 4 | 1 |
| ChunkParams simplification | 12 | 0 | 1 |
| Validation logic removal | 13 | 0 | 1 |
| Comment updates | 0 | 3 | 0.5 |
| Test cleanup | 20 | 10 | 2-3 |
| Documentation | 0 | 15 | 1-2 |
| Testing & verification | 0 | 0 | 2-3 |
| **Total** | **~45** | **~32** | **8-11.5** |

---

## Consolidation Diagram (If Option 2 Chosen)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BEFORE: Time-Axis Chunking Architecture                   │
└─────────────────────────────────────────────────────────────────────────────┘

     solver.py                 BatchSolverKernel.py          mem_manager.py
  ┌─────────────┐            ┌───────────────────────┐    ┌──────────────────┐
  │chunk_axis   │───────────>│FullRunParams          │    │allocate_queue()  │
  │= "run"      │            │  .chunk_axis ─────────│───>│  chunk_axis      │
  │= "time" [X] │            │                       │    │    = "run"       │
  └─────────────┘            │ChunkParams            │    │    = "time" [X]  │
                             │  ._chunk_axis         │    │                  │
                             │  time-specific        │    │is_request_       │
                             │  logic [X]            │    │chunkable()       │
                             │                       │    │  "time" axis [X] │
                             │run() validation       │    │                  │
                             │  time-check [X]       │    │compute_per_      │
                             └───────────────────────┘    │chunk_slice()     │
                                                          │  "time" axis [X] │
     array_requests.py       BaseArrayManager.py          └──────────────────┘
  ┌─────────────────┐       ┌────────────────────┐
  │ArrayResponse    │       │_chunk_axis         │
  │  .chunk_axis    │<──────│  = "run"           │
  │    = "run"      │       │  = "time" [X]      │
  │    = "time" [X] │       │                    │
  └─────────────────┘       └────────────────────┘

  [X] = Components to remove/simplify


┌─────────────────────────────────────────────────────────────────────────────┐
│                    AFTER: Simplified Run-Only Chunking                       │
└─────────────────────────────────────────────────────────────────────────────┘

     solver.py                 BatchSolverKernel.py          mem_manager.py
  ┌─────────────┐            ┌───────────────────────┐    ┌──────────────────┐
  │chunk_axis   │───────────>│FullRunParams          │    │allocate_queue()  │
  │= "run"      │            │  .chunk_axis          │───>│  chunk_axis      │
  │             │            │                       │    │    = "run"       │
  └─────────────┘            │ChunkParams            │    │                  │
                             │  ._chunk_axis         │    │is_request_       │
                             │  (simplified)         │    │chunkable()       │
                             │                       │    │  "run" axis only │
                             │run()                  │    │                  │
                             │  (no time validation) │    │compute_per_      │
                             └───────────────────────┘    │chunk_slice()     │
                                                          │  "run" axis only │
     array_requests.py       BaseArrayManager.py          └──────────────────┘
  ┌─────────────────┐       ┌────────────────────┐
  │ArrayResponse    │       │_chunk_axis         │
  │  .chunk_axis    │<──────│  = "run"           │
  │    = "run"      │       │                    │
  └─────────────────┘       └────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         Complexity Reduction Summary                         │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────┐
  │ ChunkParams.__getitem__()                                              │
  │                                                                        │
  │ BEFORE:                          AFTER:                                │
  │ ┌──────────────────────┐         ┌──────────────────────┐              │
  │ │ if axis == "run":    │         │ if axis == "run":    │              │
  │ │   _runs = length     │    ──>  │   _runs = length     │              │
  │ │ elif axis == "time": │         │ # (removed)          │              │
  │ │   dt_save = ...      │         │                      │              │
  │ │   _duration = ...    │         └──────────────────────┘              │
  │ │   _t0 = ...          │                                               │
  │ │   _warmup = 0.0      │                                               │
  │ └──────────────────────┘                                               │
  │                                                                        │
  │ Removed: 9 lines of complex time-based chunk calculation               │
  └────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────┐
  │ Validation Flow                                                        │
  │                                                                        │
  │ BEFORE:                                                                │
  │ ┌──────────────────────────────────────────────┐                       │
  │ │ run() validates timing parameters            │                       │
  │ │   └──> if time-chunked, revalidate after     │                       │
  │ │        computing per-chunk duration          │                       │
  │ │          └──> raise if invalid               │                       │
  │ └──────────────────────────────────────────────┘                       │
  │                                                                        │
  │ AFTER:                                                                 │
  │ ┌──────────────────────────────────────────────┐                       │
  │ │ run() validates timing parameters            │                       │
  │ │   (single validation, no re-check needed)    │                       │
  │ └──────────────────────────────────────────────┘                       │
  │                                                                        │
  │ Removed: 13 lines of try/except with custom error message              │
  └────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────┐
  │ Validator Simplification                                               │
  │                                                                        │
  │ Files affected: 3                                                      │
  │ ┌───────────────────────────────────────────────────────────────────┐  │
  │ │ array_requests.py:142    ["run", "variable", "time"]              │  │
  │ │ BaseArrayManager.py:262  ["run", "variable", "time"]              │  │
  │ │                                       ▼                           │  │
  │ │ array_requests.py:142    ["run", "variable"]                      │  │
  │ │ BaseArrayManager.py:262  ["run", "variable"]                      │  │
  │ └───────────────────────────────────────────────────────────────────┘  │
  │                                                                        │
  │ Note: "variable" axis chunking retained for potential future use       │
  └────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────┐
  │ Test Simplification                                                    │
  │                                                                        │
  │ test_chunking.py:                                                      │
  │ ┌───────────────────────────────────────────────────────────────────┐  │
  │ │ BEFORE: @pytest.mark.parametrize("chunk_axis", ["run", "time"])   │  │
  │ │         (6 occurrences)                                           │  │
  │ │                                                                   │  │
  │ │ AFTER:  @pytest.mark.parametrize("chunk_axis", ["run"])           │  │
  │ │         OR: Remove parametrization, use "run" directly            │  │
  │ └───────────────────────────────────────────────────────────────────┘  │
  │                                                                        │
  │ Tests removed/simplified: 6+ parametrized test cases                   │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## Consolidation Opportunities (Option 2)

### Modules/Methods That Can Be Simplified

| Component | Current State | After Removal |
|-----------|---------------|---------------|
| `ChunkParams.__getitem__()` | 21 lines with if/elif | 7 lines, simple |
| `BatchSolverKernel.run()` | Time validation block | Removed (13 lines) |
| `is_request_chunkable()` | Checks for "time" in stride_order | Same logic, fewer cases |
| `compute_per_chunk_slice()` | Handles time-axis indices | Simpler, only "run" axis |
| Test parametrizations | ["run", "time"] | ["run"] only |

### Code That Cannot Be Removed (Still Needed)

- Run-axis chunking infrastructure (primary use case)
- Variable-axis chunking validators (future expansion possible)
- Core `MemoryManager.allocate_queue()` (used for run-axis)
- `BaseArrayManager` chunking support (needed for run-axis)

### Net Effect

```
Lines of code removed:     ~45-55
Lines of code simplified:  ~30-40
Test cases removed:        ~6-10 parametrized cases
Complexity reduced:        Significant (removes edge case handling)
```

---

## Use Case Analysis

### When Would Time-Axis Chunking Be Needed?

| Scenario | Likely? | Notes |
|----------|---------|-------|
| Long integration, few runs, dense output | Rare | CPU would be faster |
| Single very long run | Very Rare | Not batch-oriented use case |
| Debugging/visualization of single run | No | Use smaller duration instead |
| Memory-limited device with long runs | Rare | Run-axis chunking usually sufficient |

### When Is Run-Axis Chunking Sufficient?

| Scenario | Common? | Notes |
|----------|---------|-------|
| Large parameter sweeps | Very Common | Primary GPU use case |
| Monte Carlo simulations | Very Common | Many short runs |
| Sensitivity analysis | Common | Many parameter variations |
| Ensemble simulations | Common | Many initial conditions |

**Conclusion**: Run-axis chunking covers >99% of realistic use cases for GPU-accelerated batch ODE integration.

---

## Recommendation

**Remove time-axis chunking (Option 2)** for the following reasons:

1. **Complexity vs. Utility**: High complexity to support a rare edge case
2. **Maintenance Burden**: Time-axis chunking creates ongoing edge case handling
3. **Summary Incompatibility**: The incommensurate chunk size problem has no clean solution
4. **Performance**: Long single runs are better suited to CPU integration
5. **Simplicity**: Removing it makes the codebase easier to understand and maintain
6. **Reversibility**: If needed later, the feature can be re-added with proper design

### Implementation Order (Option 2)

1. Update validators to remove "time" option
2. Remove time-specific logic from `ChunkParams.__getitem__()`
3. Remove time-chunk validation from `BatchSolverKernel.run()`
4. Update/remove tests
5. Update documentation
6. Run full test suite to verify no regressions

---

## Appendix: File-by-File Change Details

### Option 2 Implementation Checklist

- [ ] `src/cubie/batchsolving/solver.py`
  - [ ] Update docstring for `chunk_axis` parameter (line ~384)
  
- [ ] `src/cubie/batchsolving/BatchSolverKernel.py`
  - [ ] Update `FullRunParams` docstring (lines 77-78)
  - [ ] Update `ChunkParams` docstring (lines 112-113)
  - [ ] Remove time-axis logic from `__getitem__()` (lines 187-196)
  - [ ] Remove time-chunk validation (lines 593-605)
  
- [ ] `src/cubie/memory/array_requests.py`
  - [ ] Update `chunk_axis` validator (line 142)
  
- [ ] `src/cubie/batchsolving/arrays/BaseArrayManager.py`
  - [ ] Update `_chunk_axis` validator (line 262)
  - [ ] Update docstring (lines 226-228)
  
- [ ] `src/cubie/batchsolving/arrays/BatchOutputArrays.py`
  - [ ] Simplify comment (lines 477-479)
  
- [ ] `tests/batchsolving/arrays/test_chunking.py`
  - [ ] Update `test_chunk_axis_property_raises_on_inconsistency` (line 46)
  - [ ] Remove "time" from parametrizations (lines 52, 64, 78, 116, 142)
  
- [ ] `tests/memory/test_memmgmt.py`
  - [ ] Update/remove time-axis tests (lines 920, 936)
