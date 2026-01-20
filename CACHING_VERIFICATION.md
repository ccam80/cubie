# Investigation: Two-Level Caching Architecture

## User's Claims to Verify

1. **Two-level caching**:
   - Level 1: Numba Dispatcher cache (function objects)
   - Level 2: Compiled kernel cache (on disk, keyed by config_hash)

2. **eq=False for callables is correct** because:
   - Callable contents determined by hashed compile settings
   - Same child hashes → identical callable
   - Therefore callables don't need separate hashing

3. **Dispatcher invalidation**:
   - Changes to ANY field trigger dispatcher invalidation
   - eq=False vs eq=True only affects hash calculation, NOT invalidation
   - When field changes: config.update() → changed set → invalidate_cache()

4. **The bug**: Not using `update_compile_settings()` for loop_fn updates

## Code Flow Analysis

### When SingleIntegratorRun.build() Updates IVPLoop

```python
# SingleIntegratorRunCore.build() line 663
self._loop.update(compiled_functions)
```

This calls:
```python
# IVPLoop.update() line 982
recognised = self.update_compile_settings(updates_dict, silent=True)
```

Which calls:
```python
# CUDAFactory.update_compile_settings() line 457
recognized, changed = self._compile_settings.update(updates_dict)

# Line 466-467
if changed:
    self._invalidate_cache()
```

Which calls:
```python
# _CubieConfigBase.update() line 182-186
value_changed = old_value != value
if value_changed:
    setattr(self, fld.name, value)
    changed.add(key)

# Line 196-197
if changed:
    self._values_hash = self._generate_values_hash()
```

### Key Insight

The `changed` set is determined by **value comparison** (line 182), NOT by eq flag.
- Fields with eq=False still participate in change detection
- They're just excluded from values_tuple/values_hash (lines 223, 76-77)

## Testing the Assertion

### Test 1: Does changing callable trigger invalidation?

**Setup**: Update a callable field with eq=False
**Expected**: If user is correct, cache should be invalidated
**Code**:
```python
loop = IVPLoop(...)
assert loop.cache_valid == False  # Initially invalid
loop.device_function  # Force build, cache becomes valid
assert loop.cache_valid == True

# Update callable field
loop.update({'step_function': new_function})
# Should invalidate cache even though eq=False
assert loop.cache_valid == False  # User's assertion
```

### Test 2: Are kernels determined by non-callable settings?

**Setup**: Two configs with same non-callable settings but different callable values
**Expected**: If user is correct, they should produce identical kernels
**Reasoning**: Callables are themselves built from config settings, so same config → same callables → same kernel

## Verification from Code

### Dispatcher Invalidation Flow

1. `update_compile_settings()` called
2. Calls `config.update()` which returns `(recognized, changed)`
3. `changed` set populated based on `old_value != value` comparison
4. If `changed` is non-empty, calls `_invalidate_cache()`
5. `_invalidate_cache()` sets `self._cache_valid = False`

**Conclusion**: eq=False does NOT prevent invalidation!

### Hash Calculation

1. `values_hash` is generated from `values_tuple`
2. `values_tuple` filters by `attribute_is_hashable()`
3. `attribute_is_hashable()` returns False if `eq=False`
4. Therefore eq=False fields excluded from hash

**Conclusion**: eq=False only affects hashing, not invalidation!

## Re-Analysis of the Bug

### What I Got Wrong

I claimed that updating callable fields doesn't trigger rebuilds because they have eq=False. This is **INCORRECT**.

### What the User is Saying

The architecture is:
1. **Dispatcher level**: Invalidated when ANY field changes (including eq=False)
2. **Kernel level**: Cached by config_hash (excludes eq=False fields)

Callables don't need to be in the hash because:
- They're built from child factories
- Child factory hashes ARE included
- Same child hashes → same callables → can reuse compiled kernel

### The Actual Bug

Looking at BatchSolverKernel.update() (line 879):
```python
updates_dict.update({
    "loop_fn": self.single_integrator.device_function,
    ...
})

all_unrecognized -= self.update_compile_settings(
    updates_dict, silent=True
)
```

This DOES call `update_compile_settings()` with loop_fn!

But wait - let me check if loop_fn is actually in the config...

## Findings

### BatchSolverKernel Analysis

`BatchSolverConfig` has `loop_fn` with eq=False (line 117).

In `BatchSolverKernel.update()` (line 879), loop_fn IS updated via `update_compile_settings()`.

However, in `build_kernel()` (line 691):
```python
loopfunction = self.single_integrator.device_function
```

It doesn't use `config.loop_fn`! It goes directly to the child factory.

**This is fine** because single_integrator is a child factory, so its hash is included.

### Real Issue?

The user says "not updating loop function using update_compile_settings" is a bug. But the code DOES update it! 

Let me check if there's somewhere else where update() is called instead of update_compile_settings()...

Actually, looking back at SingleIntegratorRun line 663:
```python
self._loop.update(compiled_functions)
```

This DOES call update_compile_settings internally (via IVPLoop.update → update_compile_settings).

## Empirical Verification

Tested with simple config class:
```python
@define
class TestConfig(_CubieConfigBase):
    normal_field: int = field(default=0)
    excluded_field: int = field(default=0, eq=False)
```

Results:
- Updating normal_field: `changed = {'normal_field'}`, hash changes
- Updating excluded_field: `changed = {'excluded_field'}`, hash DOES NOT change

**Key Finding**: eq=False fields ARE included in the 'changed' set!

This means:
1. `config.update()` returns changed set including eq=False fields
2. `update_compile_settings()` sees non-empty changed set
3. `_invalidate_cache()` is called
4. Dispatcher is invalidated and rebuild happens

## Conclusion

The user is **ABSOLUTELY CORRECT**:

1. **eq=False is appropriate for callables** - they don't need separate hashing because their contents are determined by child factory settings which ARE hashed

2. **Dispatcher invalidation works correctly** - changes to ANY field (eq=False or eq=True) trigger `_invalidate_cache()` and force rebuild

3. **Two-level caching architecture is sound**:
   - **Level 1 (Dispatcher)**: Invalidated when any field changes
   - **Level 2 (Compiled kernel)**: Cached by config_hash which excludes eq=False fields
   - Same non-callable settings + same child hashes → same callables → can reuse compiled kernel

4. **Hash only affects compiled kernel cache**, not dispatcher invalidation

**My original analysis was WRONG** because I conflated:
- Hash calculation (where eq=False excludes fields from values_tuple)
- Change detection (where eq=False fields ARE still tracked)
- Dispatcher invalidation (triggered by non-empty changed set, regardless of eq)

The callables are correctly excluded from hashing because they're fully determined by the hashed compile settings of their child factories.
