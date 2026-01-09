# PR Review Changes - Agent Plan

## Overview

This plan addresses four specific PR review items. Changes should be surgical and precise, following existing patterns in the repository.

---

## Task 1: Move CUDASIM-Conditioned Imports to cuda_simsafe.py

### Current State

In `cubie_cache.py` (lines 25-40):
```python
if not is_cudasim_enabled():
    from numba.cuda.core.caching import (
        _CacheLocator,
        CacheImpl,
        IndexDataCacheFile,
    )
    from numba.cuda.dispatcher import CUDACache

    _CACHING_AVAILABLE = True
else:
    # Stub classes for simulator mode
    _CacheLocator = object
    CacheImpl = object
    IndexDataCacheFile = None
    CUDACache = object
    _CACHING_AVAILABLE = False
```

### Target State

Move conditional imports to `cuda_simsafe.py` following the existing pattern (see lines 116-164 for similar patterns). Then import from `cuda_simsafe.py` in `cubie_cache.py`.

### Components to Modify

1. **cuda_simsafe.py**:
   - Add conditional imports for `_CacheLocator`, `CacheImpl`, `IndexDataCacheFile`, `CUDACache`
   - In CUDASIM mode: provide stub classes (`object` or `None`)
   - In GPU mode: import from `numba.cuda.core.caching` and `numba.cuda.dispatcher`
   - Export `_CACHING_AVAILABLE` boolean
   - Add all new exports to `__all__`

2. **cubie_cache.py**:
   - Remove the conditional import block
   - Import from `cuda_simsafe` instead:
     ```python
     from cubie.cuda_simsafe import (
         _CacheLocator,
         CacheImpl,
         IndexDataCacheFile,
         CUDACache,
         _CACHING_AVAILABLE,
     )
     ```

### Expected Behavior
- No functional change
- Caching classes available in GPU mode
- Stub classes used in CUDASIM mode
- Centralized handling in `cuda_simsafe.py`

---

## Task 2: Modify ODEFile Directory Structure

### Current State

In `odefile.py` (line 38):
```python
self.file_path = GENERATED_DIR / f"{system_name}.py"
```

This creates files directly in `GENERATED_DIR`.

### Target State

```python
self.file_path = GENERATED_DIR / system_name / f"{system_name}.py"
```

With appropriate directory creation.

### Components to Modify

1. **odefile.py**:
   - Change `__init__` to create system-specific subdirectory
   - Update `file_path` to use `GENERATED_DIR / system_name / f"{system_name}.py"`
   - Ensure directory is created in `__init__` before file operations

### Implementation Details

In `ODEFile.__init__`:
```python
def __init__(self, system_name: str, fn_hash: int) -> None:
    system_dir = GENERATED_DIR / system_name
    system_dir.mkdir(parents=True, exist_ok=True)
    self.file_path = system_dir / f"{system_name}.py"
    self.fn_hash = fn_hash
    self._init_file(fn_hash)
```

### Expected Behavior
- Generated code files placed in `generated/{system_name}/{system_name}.py`
- Same directory structure as `CUBIECacheLocator` uses for cache files
- Existing functionality preserved

---

## Task 3: Add Tests for config_hash and _iter_child_factories

### Components to Test

1. **config_hash property** (CUDAFactory.py lines 522-537):
   - Returns hash of compile settings
   - Combines hashes from child factories if present
   - Returns 64-character hex string

2. **_iter_child_factories method** (CUDAFactory.py lines 539-553):
   - Yields CUDAFactory instances from direct attributes
   - Sorted alphabetically by attribute name
   - Each child yielded once (uniqueness by id)

### Test Cases to Add

Add to `tests/test_CUDAFactory.py`:

1. **test_config_hash_no_children**:
   - Create factory with settings, no child factories
   - Verify `config_hash == compile_settings.values_hash`

2. **test_config_hash_with_children**:
   - Create factory with child factory attributes
   - Verify combined hash differs from own settings hash
   - Verify deterministic result

3. **test_iter_child_factories_no_children**:
   - Create factory with no CUDAFactory attributes
   - Verify `_iter_child_factories()` yields nothing

4. **test_iter_child_factories_with_children**:
   - Create factory with multiple child factory attributes
   - Verify all children are yielded
   - Verify alphabetical ordering by attribute name

5. **test_iter_child_factories_uniqueness**:
   - Create factory where same child is referenced by multiple attributes
   - Verify child yielded only once

### Implementation Pattern

```python
def test_config_hash_with_children(factory):
    """Test config_hash combines hashes from child factories."""
    # Create child factory
    child = ConcreteFactory()
    child.setup_compile_settings(...)
    
    # Attach as attribute
    factory._child = child
    factory.setup_compile_settings(...)
    
    # Hash should combine own + child hashes
    own_hash = factory.compile_settings.values_hash
    assert factory.config_hash != own_hash
    assert len(factory.config_hash) == 64
```

---

## Task 4: Update test_cubie_cache.py for config_hash Parameter

### Current State

Tests instantiate `CUBIECache` without `config_hash`:
```python
cache = CUBIECache(system_name="test_system", system_hash="abc123")
```

The `CUBIECache.__init__` signature (line 283-291):
```python
def __init__(
    self,
    system_name: str,
    system_hash: str,
    config_hash: Optional[str] = None,
    max_entries: int = 10,
    mode: str = "hash",
    custom_cache_dir: Optional[Path] = None,
) -> None:
```

### Changes Required

Update test instantiations to pass `config_hash`:

```python
cache = CUBIECache(
    system_name="test_system",
    system_hash="abc123",
    config_hash="def456789...",  # 64-char hex string
)
```

### Tests to Modify

1. **test_cubie_cache_init** (line 131-137)
2. **test_cubie_cache_index_key** (line 140-160)
3. **test_cubie_cache_path** (line 163-169)

### Note
The tests already define `MockCompileSettings` at the top but don't use it in cache instantiation. The fix is to pass an appropriate config hash string.

---

## Integration Points

### Imports Flow
```
cuda_simsafe.py
    ├── exports: _CacheLocator, CacheImpl, IndexDataCacheFile, CUDACache, _CACHING_AVAILABLE
    └── used by: cubie_cache.py

cubie_cache.py
    ├── imports from: cuda_simsafe.py
    └── uses: CUBIECacheLocator, CUBIECacheImpl, CUBIECache
```

### Directory Structure
```
GENERATED_DIR/
    └── {system_name}/
        ├── {system_name}.py      # ODEFile generated code
        └── CUDA_cache/           # CUBIECacheLocator cache files
            ├── *.nbi
            └── *.nbc
```

---

## Dependencies

- Task 1 and Task 2 are independent
- Task 3 requires understanding of existing test patterns in `test_CUDAFactory.py`
- Task 4 requires Task 1 to be complete (to ensure imports work correctly)

---

## Edge Cases

### Task 1
- Ensure stub classes in CUDASIM mode don't break class inheritance
- `_CacheLocator = object` allows `class CUBIECacheLocator(_CacheLocator)` to work

### Task 2
- Handle case where `system_name` contains path-unsafe characters (existing behavior)
- Ensure `parents=True` in `mkdir` for nested structure

### Task 3
- Test with factory that has no `_compile_settings` set (edge case)
- Test with circular references (should not happen but guard against)

### Task 4
- Ensure tests still pass under `@pytest.mark.nocudasim` marker
