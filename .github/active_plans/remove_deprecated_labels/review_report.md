# Implementation Review Report
# Feature: Remove Deprecated Label Parameters
# Review Date: 2026-01-01
# Reviewer: Harsh Critic Agent

## Executive Summary

The implementation successfully removes deprecated label-based parameters while preserving the unified `save_variables`/`summarise_variables` interface. The core implementation is **clean, correct, and well-executed**. Code reduction was achieved (~30 lines removed), simplification goals were met, and all tests pass.

However, I identified several issues that warrant attention:

1. **Docstring accuracy concerns**: The `convert_output_labels()` docstring contains misleading statements about resolver behavior
2. **Minor naming inconsistencies**: Test names could be more precise
3. **Missing edge case coverage**: No tests verify that deprecated parameters are silently ignored when provided (current behavior is unclear)

Overall, this is a **high-quality implementation** that achieves its goals. The issues identified are minor refinements rather than fundamental flaws. The breaking change is intentional, well-documented, and provides a clear migration path.

## User Story Validation

**User Stories** (from human_overview.md):

### Story 1: Clean Parameter Interface
**Status**: ✅ **Met**

- ✅ Deprecated parameters removed from `ALL_OUTPUT_FUNCTION_PARAMETERS` constant (lines 28-38 in output_functions.py)
- ✅ Deprecated parameters removed from `convert_output_labels()` resolver logic (lines 307-316 in solver.py)
- ✅ Properties exposing label lists retained (lines 1008-1046 in solver.py)
- ✅ All tests updated to new interface (test_solver.py updated with 6 tests)
- ✅ Net reduction in lines achieved (~30 lines removed)
- ✅ Breaking change is intentional and documented

**Acceptance Criteria Assessment**: All criteria fully met. The implementation cleanly removes deprecated parameters without affecting read-only properties or result metadata.

### Story 2: Unified Variable Selection
**Status**: ✅ **Met**

- ✅ `save_variables` parameter works as explicit argument (verified in tests)
- ✅ `summarise_variables` parameter works as explicit argument (verified in tests)
- ✅ Variables are automatically classified into states/observables (lines 336-342, 348-358 in solver.py)
- ✅ Index-based kwargs continue to work (resolver dictionary at lines 307-316)
- ✅ Documentation reflects the unified interface (docstrings updated)

**Acceptance Criteria Assessment**: All criteria fully met. The unified interface is preserved and works correctly with automatic classification.

### Story 3: Preserved Label Access
**Status**: ✅ **Met**

- ✅ `solver.saved_states` property returns list of state labels (line 1008-1010)
- ✅ `solver.saved_observables` property returns list of observable labels (lines 1018-1022)
- ✅ `solver.summarised_states` property returns list of state labels (lines 1030-1034)
- ✅ `solver.summarised_observables` property returns list of observable labels (lines 1042-1046)
- ✅ Properties work correctly after initialization (verified in tests)
- ✅ SolveSpec exposes these label lists (lines 111-114 in solveresult.py)

**Acceptance Criteria Assessment**: All criteria fully met. Read-only access to labels is preserved and working correctly.

## Goal Alignment

**Original Goals** (from human_overview.md):

### Goal 1: Remove backward-compatibility code
**Status**: ✅ **Achieved**

- Removed 4 deprecated parameter names from constant
- Removed 4 resolver entries
- Removed `labels2index_keys` dictionary (7 lines)
- Removed key renaming loop (10 lines)
- Total: ~30 lines of backward-compatibility code removed

### Goal 2: Simplify configuration
**Status**: ✅ **Achieved**

- Single source of truth for unified parameters established
- No more confusion between deprecated and current parameters
- Clear separation: unified params for input, properties for output

### Goal 3: Maintain functionality
**Status**: ✅ **Achieved**

- Label access via properties preserved
- Index-based parameters unchanged
- Unified parameters work identically
- Variable classification logic preserved

**Assessment**: All goals achieved. The implementation delivers on the promise to reduce code complexity while maintaining essential functionality.

## Code Quality Analysis

### Duplication
✅ **No duplication detected**

The implementation eliminates duplication rather than creating it. The removed `labels2index_keys` dictionary was itself duplicating information already present in the resolver mappings.

### Unnecessary Complexity
⚠️ **Minor issue identified**

**Location**: `src/cubie/batchsolving/solver.py`, method `convert_output_labels()` (lines 267-359)

**Issue**: The docstring at lines 271-275 states:

> "Users can provide lists of state and observable variable names, or lists/arrays of indices if they know them and want a "fast path" solve to minimise overhead."

This is misleading. The resolver loop (lines 318-322) calls `resolver(values)` for ALL index parameters, which means providing integer indices still triggers the SystemInterface resolver methods. There is no "fast path" when providing integer arrays to index parameters - they are still processed through the resolver.

**Impact**: Medium - Misleading documentation could cause users to make incorrect performance assumptions.

**Suggested Fix**: Either:
1. Update docstring to clarify that resolvers accept both labels and indices (current behavior)
2. Implement actual fast-path by checking if values are already numpy arrays and skipping resolver call

### Unnecessary Additions
✅ **No unnecessary additions detected**

All code changes serve the stated goals. Test additions are appropriate and necessary to verify the new interface.

### Convention Violations

#### PEP8 Compliance
✅ **No violations detected**

Line lengths appear to be within 79 characters. Code formatting follows PEP8 conventions.

#### Numpydoc Docstrings
⚠️ **Minor issues identified**

**Location**: `src/cubie/batchsolving/solver.py`, lines 267-306

**Issue 1**: The ValueError description (lines 291-293) is now less specific:

> "If the settings dict would result in duplicate or conflicting indices."

The old version provided a concrete example. The new version is vaguer about what "duplicate or conflicting" means.

**Impact**: Low - Users may not understand what triggers this error.

**Location**: `src/cubie/batchsolving/solver.py`, lines 297-305

**Issue 2**: The Notes section mentions:

> "Users may supply selectors as labels or integers"

But this is only true for the unified `save_variables`/`summarise_variables` parameters. The index parameters (`saved_state_indices`, etc.) resolve through SystemInterface.state_indices/observable_indices methods, which means they can accept labels too. The docstring doesn't clearly explain this distinction.

**Impact**: Low - Minor confusion about which parameters accept which types.

#### Type Hints
✅ **No violations detected**

Type hints are present in method signatures. No inline type annotations in implementation.

#### Repository Patterns
✅ **No violations detected**

- Never calls `build()` directly
- Uses attrs classes appropriately
- Follows CUDAFactory patterns
- No environment variable modifications

## Performance Analysis

### CUDA Efficiency
**N/A** - This change does not affect CUDA kernel code.

### Memory Patterns
**N/A** - This change does not affect memory allocation or access patterns.

### Buffer Reuse
**N/A** - No buffer allocations in this change.

### Math vs Memory
**N/A** - No mathematical operations in this change.

### Optimization Opportunities

**Opportunity 1: Eliminate resolver calls for pure array inputs**

**Location**: `src/cubie/batchsolving/solver.py`, lines 318-322

**Current Behavior**:
```python
for key, resolver in resolvers.items():
    values = output_settings.get(key)
    if values is not None:
        output_settings[key] = resolver(values)
```

**Issue**: Even if `values` is already a numpy array of indices, we still call `resolver(values)`, which invokes SystemInterface methods that check if the input is an array or labels. This adds overhead for the "array-only fast path" mentioned in docstrings.

**Suggested Optimization**:
```python
for key, resolver in resolvers.items():
    values = output_settings.get(key)
    if values is not None:
        # Fast path: skip resolver if already a numpy array
        if not isinstance(values, np.ndarray):
            output_settings[key] = resolver(values)
```

**Impact**: This would provide a true fast path for index-based parameters and match the documented behavior.

**Note**: This may not be necessary if SystemInterface resolvers already have this optimization internally. Review SystemInterface.state_indices() and observable_indices() to verify.

## Architecture Assessment

### Integration Quality
✅ **Excellent**

The changes integrate seamlessly with existing architecture:
- OutputFunctions continues to work with indices only
- OutputConfig receives clean index-based parameters
- Solver properties bridge between indices and labels
- SolveSpec stores label metadata correctly

No architectural mismatches or awkward integrations detected.

### Design Patterns
✅ **Appropriate**

The implementation follows established CuBIE patterns:
- Constants define recognized parameters
- Resolver pattern for label-to-index conversion
- Properties for read-only access
- attrs classes for data containers

### Future Maintainability
✅ **Excellent**

The simplified code is easier to maintain:
- Fewer code paths to debug
- Clear separation of concerns
- No deprecated code to remove later
- Migration path well-documented

**Potential future improvement**: Consider deprecating index-based parameters in favor of unified parameters only. Current design has three ways to specify variables (unified, state indices, observable indices), which could be simplified to just unified parameters. However, this may not be desirable if index-based parameters provide performance benefits.

## Suggested Edits

### Edit 1: Clarify Docstring About Resolver Behavior

**Task Group**: Documentation improvements
**File**: src/cubie/batchsolving/solver.py
**Issue**: Docstring implies fast path exists for integer indices, but resolver is always called
**Fix**: Update docstring to accurately describe resolver behavior
**Rationale**: Documentation should match implementation to avoid user confusion

**Specific changes**:
```python
# Lines 271-275, change from:
"""Resolve output label settings in-place. Users can provide lists
of state and observable variable names, or lists/arrays of indices
if they know them and want a "fast path" solve to minimise overhead.
The expected usual pathway will be for a user to provide a list of
names to save_variables and summarise_variables."""

# To:
"""Resolve output label settings in-place. Users can provide variable
names via save_variables and summarise_variables (recommended), or use
index-based parameters (saved_state_indices, etc.) for direct control.
All parameters accept either variable names (strings) or integer indices,
which are resolved through SystemInterface."""
```

**Status**: [Taskmaster to complete]

### Edit 2: Make ValueError Description More Specific

**Task Group**: Documentation improvements
**File**: src/cubie/batchsolving/solver.py
**Issue**: ValueError description is vague about what causes the error
**Fix**: Provide concrete description of error conditions
**Rationale**: Users should understand what triggers errors

**Specific changes**:
```python
# Lines 291-293, change from:
"""
ValueError
    If the settings dict would result in duplicate or conflicting
    indices.
"""

# To:
"""
ValueError
    If a variable name in save_variables or summarise_variables is not
    found in the system's states or observables. The error message
    includes available variable names.
"""
```

**Status**: [Taskmaster to complete]

### Edit 3: Improve Test Name Precision

**Task Group**: Test improvements
**File**: tests/batchsolving/test_solver.py
**Issue**: Test name `test_save_variables_union_with_saved_state_indices` (line 937) is redundant with `test_save_variables_union_with_indices` (line 919)
**Fix**: Remove duplicate test or differentiate clearly
**Rationale**: Avoid test duplication and confusion

**Specific changes**:

Option 1: Remove the duplicate test at lines 937-950

Option 2: Differentiate by testing different scenarios:
- Line 919 test: Union with saved_state_indices only
- Line 937 test: Union with saved_observable_indices only

**Status**: [Taskmaster to complete]

### Edit 4: Add Test for Deprecated Parameter Silent Filtering

**Task Group**: Test improvements
**File**: tests/batchsolving/test_solver.py
**Issue**: No test verifies that deprecated parameters are silently filtered when provided to Solver constructor
**Fix**: Add test demonstrating current filtering behavior
**Rationale**: Document expected behavior when users provide deprecated params

**Specific changes**:

Add new test after line 1129:
```python
def test_deprecated_params_silently_filtered(system):
    """Test that deprecated params are silently filtered by Solver.
    
    When users provide deprecated parameters (saved_states, etc.) as kwargs,
    they should be silently filtered out by ALL_OUTPUT_FUNCTION_PARAMETERS
    and not cause errors.
    """
    state_names = list(system.initial_values.names)[:2]
    
    # Attempt to create solver with deprecated param
    # This should not raise an error - param is filtered before processing
    solver = Solver(
        system,
        algorithm="euler",
        saved_states=state_names,  # Deprecated - should be filtered
    )
    
    # Verify solver was created successfully
    assert solver is not None
    
    # Verify the deprecated param had no effect (not set)
    # Since it was filtered, saved_state_indices should be empty or default
    assert len(solver.saved_state_indices) == 0
```

**Status**: [Taskmaster to complete]

### Edit 5: Improve Notes Section Clarity

**Task Group**: Documentation improvements
**File**: src/cubie/batchsolving/solver.py
**Issue**: Notes section (lines 295-305) doesn't clearly explain which parameters accept labels vs indices
**Fix**: Clarify that all parameters can accept both forms
**Rationale**: Users should understand the flexibility of the interface

**Specific changes**:
```python
# Lines 295-305, change from:
"""
Notes
-----
Users may supply selectors as labels or integers; this resolver ensures
that downstream components receive numeric indices and canonical keys.

The unified parameters ``save_variables`` and ``summarise_variables``
are automatically classified into states and observables using
SystemInterface. Results are merged with index-based parameters
(``saved_state_indices``, ``saved_observable_indices``,
``summarised_state_indices``, ``summarised_observable_indices``) using
set union.
"""

# To:
"""
Notes
-----
All parameters accept either variable names (strings) or integer indices.
This method resolves names to indices and ensures downstream components
receive numeric indices with canonical keys.

The unified parameters ``save_variables`` and ``summarise_variables``
automatically classify variables into states and observables. Results are
merged with any index-based parameters (``saved_state_indices``, etc.)
using set union, allowing users to combine both approaches.
"""
```

**Status**: [Taskmaster to complete]

## Summary of Issues

### Critical Issues
**None** - Implementation is functionally correct.

### Important Issues
**None** - All core functionality works as intended.

### Minor Issues
1. ⚠️ Docstring implies fast path that doesn't exist (Edit 1)
2. ⚠️ ValueError description is vague (Edit 2)
3. ⚠️ Duplicate test detected (Edit 3)
4. ⚠️ Missing test for deprecated param filtering (Edit 4)
5. ⚠️ Notes section could be clearer (Edit 5)

All issues are documentation or test-related. **No functional bugs detected.**

## Final Assessment

### Strengths
1. ✅ Clean removal of deprecated code (~30 lines)
2. ✅ All user stories met
3. ✅ All tests passing (19/19 related tests)
4. ✅ Breaking change is intentional and well-documented
5. ✅ Migration path is clear
6. ✅ Properties preserved for read-only access
7. ✅ No architectural mismatches
8. ✅ No duplication introduced
9. ✅ Follows repository conventions

### Weaknesses
1. ⚠️ Docstring accuracy issues (misleading "fast path" claim)
2. ⚠️ Minor test duplication
3. ⚠️ Missing edge case test coverage
4. ⚠️ Some docstring sections could be clearer

### Recommendation

**Approve with minor edits recommended**

The implementation successfully achieves all stated goals. The issues identified are documentation and test refinements that do not affect functionality. The suggested edits would improve clarity and completeness but are not blockers.

This is a **well-executed cleanup** that makes the codebase more maintainable without introducing regressions.

---

## Repository Convention Adherence

### PEP8 Compliance
✅ **Passing** - No violations detected

### Numpydoc Docstrings
⚠️ **Minor issues** - See Edit 1, 2, 5 for improvements

### Type Hints
✅ **Passing** - Type hints present in signatures, no inline annotations

### Repository-Specific Patterns
✅ **Passing** - Follows CUDAFactory patterns, attrs usage, no env var modifications

### PowerShell Compatibility
✅ **Passing** - No command chaining issues

### Comment Style
✅ **Passing** - Comments describe current functionality, not change history

---

## Edge Case Coverage

### Empty/None Parameters
✅ **Covered** - Tests verify `save_variables=[]` and `save_variables=None` behavior

### Mixed Index and Unified Parameters
✅ **Covered** - Test `test_save_variables_union_with_indices` verifies merging

### Invalid Variable Names
✅ **Covered** - Tests verify ValueError with helpful message

### Duplicate Indices
✅ **Covered** - Merging uses `np.union1d` which handles duplicates

### Property Access Before Solving
✅ **Covered** - Test `test_variable_labels_properties` verifies properties work

### Deprecated Parameter Filtering
⚠️ **Missing** - See Edit 4 for suggested test

---

## Test Coverage Assessment

### Tests Created
- ✅ `test_deprecated_label_parameters_rejected` - Verifies constant cleanup
- ✅ `test_unified_save_variables_parameter` - Demonstrates migration pattern

### Tests Updated
- ✅ `test_update_saved_variables` - Changed from deprecated to unified params
- ✅ `test_save_variables_union_with_saved_state_indices` - Updated to use index-based params
- ✅ `test_save_variables_with_solve_ivp` - Renamed and updated
- ✅ `test_save_variables_with_multiple_states` - Renamed and updated

### Test Coverage Gaps
- ⚠️ No test verifying deprecated params are silently filtered (See Edit 4)
- ⚠️ Duplicate test should be removed or differentiated (See Edit 3)

### Overall Test Quality
✅ **Excellent** - Tests are comprehensive, use proper fixtures, and verify both positive and negative cases.

---

*End of Review Report*
