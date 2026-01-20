# Corrected Analysis: Chunked Test Issue Investigation

## Summary of User's Correction

The user corrected my misunderstanding of the caching architecture:

### Two-Level Caching System

1. **Dispatcher Cache (Level 1 - In Memory)**:
   - Caches the Python Dispatcher objects returned by `build()`
   - Invalidated when ANY compile setting changes (including eq=False fields)
   - Invalidation triggers: `update_compile_settings()` sees non-empty `changed` set → calls `_invalidate_cache()` → sets `_cache_valid = False`
   - Next access to `device_function` → calls `_build()` → calls `build()`

2. **Compiled Kernel Cache (Level 2 - On Disk)**:
   - Numba's disk cache for compiled CUDA kernels
   - Keyed by config_hash
   - config_hash EXCLUDES eq=False fields (they're not in values_tuple)
   - If config_hash matches a previous run, Numba reuses the compiled kernel

### Why eq=False is Correct for Callables

**User's Assertion**: Device function contents are fully determined by the hashed compile settings of the factory and its children.

**Reasoning**:
1. A device function like `step_function` is built by an algorithm factory (e.g., FIRKStep)
2. FIRKStep is a child factory, so its config_hash is included in parent's hash
3. Same child config_hash → same device function will be built
4. Therefore, the device function itself doesn't need to be hashed separately

**Example**:
```
IVPLoop.config_hash = hash(own_settings + child_factory_hashes)
                    = hash(n_states, precision, ... + FIRKStep.config_hash)
```

If `FIRKStep.config_hash` is the same, the `step_function` it builds will be identical, so we can safely reuse the compiled kernel.

## Verification with Code

### Change Detection (Line 182 in CUDAFactory.py)

```python
value_changed = old_value != value
if value_changed:
    setattr(self, fld.name, value)
    changed.add(key)  # Added to changed set regardless of eq flag!
```

**Empirically verified**: Updating an eq=False field adds it to the `changed` set.

### Invalidation (Lines 466-467 in CUDAFactory.py)

```python
if changed:  # Non-empty if ANY field changed
    self._invalidate_cache()  # Dispatcher invalidated
```

**Result**: Dispatcher IS invalidated when callable fields change.

### Hash Calculation (Line 196-197 in CUDAFactory.py)

```python
if changed:
    self._values_hash = self._generate_values_hash()
```

`_generate_values_hash()` uses `values_tuple` which EXCLUDES eq=False fields.

**Result**: Hash does NOT change when only eq=False fields change.

## Re-examining the Chunked Test Issue

Given this corrected understanding, let's reconsider what could cause infinite loops:

### Scenario 1: Config Changes Between Chunks (INCORRECT ANALYSIS)

My original theory:
- Chunk 1 compiles kernel
- Config changes
- Dispatcher not invalidated (WRONG)
- Stale kernel used

This is **INCORRECT** because:
- Config changes DO invalidate dispatcher
- New build() is called
- New kernel is created

### Scenario 2: Same Config Hash, Different State (POSSIBLE)

If between chunks:
1. Non-callable settings stay the same
2. Only callable fields change
3. config_hash stays same (callables excluded)
4. Dispatcher invalidated, build() called
5. build() creates new dispatcher with new callables
6. But Numba sees same config_hash, reuses compiled kernel from disk?

**Question**: Does Numba's disk cache key on ONLY the config_hash, or does it also consider the actual function closures?

### User's Claim About the Bug

User says: "I do not see the rest of what you presented as bugs. Investigate my claims against what actually happens in the repo."

This suggests:
- The callable tracking is NOT the bug
- There may be a specific case where update() is called instead of update_compile_settings()
- Or there's a different issue entirely

## Next Steps for Investigation

1. **Find where update() is called without going through update_compile_settings()**
   - Could this bypass dispatcher invalidation?

2. **Check BatchSolverKernel.update() flow**
   - Does it properly call update_compile_settings() for loop_fn?

3. **Verify actual chunked test behavior on CUDA**
   - Is there really an infinite loop?
   - What configuration actually changes between chunks?

4. **Examine Numba's caching behavior**
   - How does Numba cache compiled kernels?
   - What determines cache key?
   - Could same config_hash with different closures cause issues?
