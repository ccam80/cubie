# Implementation Task List
# Feature: CUDASIM Caching Refactor
# Plan Reference: .github/active_plans/cudasim_caching_refactor/agent_plan.md

## Task Group 1: Vendor Numba-CUDA Caching Classes in cuda_simsafe.py
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/cuda_simsafe.py (lines 166-313)
- File: active_plans/numba_cuda_caching.py (lines 95-220, 222-354, 356-410, 615-693)

**Input Validation Required**:
- None - this is pure vendoring, no new parameters

**Tasks**:
1. **Add serialization fallback import**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     ```python
     # Add at top of caching infrastructure section (around line 166):
     # --- Caching infrastructure ---
     # Vendored from numba-cuda 2026-01-11 for CUDASIM compatibility.
     # Source: numba_cuda/numba/cuda/core/caching.py
     
     try:
         from numba.cuda.serialize import dumps as _numba_dumps
     except ImportError:
         import pickle
         def _numba_dumps(obj):
             return pickle.dumps(obj)
     ```
   - Edge cases: Import may fail in CUDASIM mode; fallback handles this
   - Integration: Used by vendored IndexDataCacheFile._dump()

2. **Add vendored IndexDataCacheFile class**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     ```python
     class IndexDataCacheFile:
         """Vendored from numba-cuda for CUDASIM compatibility.
         
         Manages index (.nbi) and data (.nbc) files for cache storage.
         Source: numba_cuda/numba/cuda/core/caching.py (2026-01-11)
         """
         
         def __init__(self, cache_path, filename_base, source_stamp):
             import os
             import numba
             self._cache_path = cache_path
             self._index_name = "%s.nbi" % (filename_base,)
             self._index_path = os.path.join(self._cache_path, self._index_name)
             self._data_name_pattern = "%s.{number:d}.nbc" % (filename_base,)
             self._source_stamp = source_stamp
             self._version = numba.__version__
         
         def flush(self):
             self._save_index({})
         
         def save(self, key, data):
             import itertools
             overloads = self._load_index()
             try:
                 data_name = overloads[key]
             except KeyError:
                 existing = set(overloads.values())
                 for i in itertools.count(1):
                     data_name = self._data_name(i)
                     if data_name not in existing:
                         break
                 overloads[key] = data_name
                 self._save_index(overloads)
             self._save_data(data_name, data)
         
         def load(self, key):
             import os
             overloads = self._load_index()
             data_name = overloads.get(key)
             if data_name is None:
                 return
             try:
                 return self._load_data(data_name)
             except OSError:
                 return
         
         def _load_index(self):
             import pickle
             try:
                 with open(self._index_path, "rb") as f:
                     version = pickle.load(f)
                     data = f.read()
             except FileNotFoundError:
                 return {}
             if version != self._version:
                 return {}
             stamp, overloads = pickle.loads(data)
             if stamp != self._source_stamp:
                 return {}
             return overloads
         
         def _save_index(self, overloads):
             import pickle
             data = self._source_stamp, overloads
             data = self._dump(data)
             with self._open_for_write(self._index_path) as f:
                 pickle.dump(self._version, f, protocol=-1)
                 f.write(data)
         
         def _load_data(self, name):
             import pickle
             path = self._data_path(name)
             with open(path, "rb") as f:
                 data = f.read()
             return pickle.loads(data)
         
         def _save_data(self, name, data):
             data = self._dump(data)
             path = self._data_path(name)
             with self._open_for_write(path) as f:
                 f.write(data)
         
         def _data_name(self, number):
             return self._data_name_pattern.format(number=number)
         
         def _data_path(self, name):
             import os
             return os.path.join(self._cache_path, name)
         
         def _dump(self, obj):
             return _numba_dumps(obj)
         
         @contextlib.contextmanager
         def _open_for_write(self, filepath):
             import os
             import uuid
             uid = uuid.uuid4().hex[:16]
             tmpname = "%s.tmp.%s" % (filepath, uid)
             try:
                 with open(tmpname, "wb") as f:
                     yield f
                 os.replace(tmpname, filepath)
             except Exception:
                 try:
                     os.unlink(tmpname)
                 except OSError:
                     pass
                 raise
     ```
   - Edge cases: File operations may fail on Windows (handled by _open_for_write)
   - Integration: Used by CUBIECache to save/load cache entries

3. **Add vendored _CacheLocator abstract base class**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     ```python
     class _CacheLocator(metaclass=ABCMeta):
         """Vendored from numba-cuda for CUDASIM compatibility.
         
         Abstract base for filesystem locators for caching functions.
         Source: numba_cuda/numba/cuda/core/caching.py (2026-01-11)
         """
         
         def ensure_cache_path(self):
             import os
             import tempfile
             path = self.get_cache_path()
             os.makedirs(path, exist_ok=True)
             tempfile.TemporaryFile(dir=path).close()
         
         @abstractmethod
         def get_cache_path(self):
             """Return the directory the function is cached in."""
         
         @abstractmethod
         def get_source_stamp(self):
             """Get a timestamp representing source code freshness."""
         
         @abstractmethod
         def get_disambiguator(self):
             """Get a string disambiguator for this locator's function."""
         
         @classmethod
         def from_function(cls, py_func, py_file):
             """Create a locator instance for the given function."""
             raise NotImplementedError
     ```
   - Edge cases: ensure_cache_path may fail if directory not writable
   - Integration: CUBIECacheLocator inherits from this

4. **Add vendored CacheImpl abstract base class**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     ```python
     class CacheImpl(metaclass=ABCMeta):
         """Vendored from numba-cuda for CUDASIM compatibility.
         
         Provides core machinery for caching serialization.
         Source: numba_cuda/numba/cuda/core/caching.py (2026-01-11)
         """
         
         _locator_classes = []
         
         @property
         def filename_base(self):
             return self._filename_base
         
         @property
         def locator(self):
             return self._locator
         
         @abstractmethod
         def reduce(self, data):
             """Returns the serialized form of the data."""
         
         @abstractmethod
         def rebuild(self, target_context, reduced_data):
             """Returns the de-serialized form of the reduced_data."""
         
         @abstractmethod
         def check_cachable(self, data):
             """Returns True if data is cachable; otherwise False."""
     ```
   - Edge cases: None - abstract base only
   - Integration: CUBIECacheImpl inherits from this

5. **Add vendored Cache base class**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     ```python
     class _Cache(metaclass=ABCMeta):
         """Vendored from numba-cuda for CUDASIM compatibility.
         
         Abstract base for per-function compilation cache.
         Source: numba_cuda/numba/cuda/core/caching.py (2026-01-11)
         """
         
         @property
         @abstractmethod
         def cache_path(self):
             """The base filesystem path of this cache."""
         
         @abstractmethod
         def load_overload(self, sig, target_context):
             """Load an overload for the given signature."""
         
         @abstractmethod
         def save_overload(self, sig, data):
             """Save the overload for the given signature."""
         
         @abstractmethod
         def enable(self):
             """Enable the cache."""
         
         @abstractmethod
         def disable(self):
             """Disable the cache."""
         
         @abstractmethod
         def flush(self):
             """Flush the cache."""
     ```
   - Edge cases: None - abstract base only
   - Integration: CUDACache inherits from this

6. **Add vendored CUDACache class**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     ```python
     class CUDACache(_Cache):
         """Vendored from numba-cuda for CUDASIM compatibility.
         
         Cache that saves and loads CUDA kernels and compile results.
         Source: numba_cuda/numba/cuda/dispatcher.py (2026-01-11)
         """
         
         _impl_class = None
         
         def __init__(self, py_func):
             self._name = repr(py_func)
             self._py_func = py_func
             self._impl = self._impl_class(py_func)
             self._cache_path = self._impl.locator.get_cache_path()
             source_stamp = self._impl.locator.get_source_stamp()
             filename_base = self._impl.filename_base
             self._cache_file = IndexDataCacheFile(
                 cache_path=self._cache_path,
                 filename_base=filename_base,
                 source_stamp=source_stamp,
             )
             self.enable()
         
         def __repr__(self):
             return "<%s py_func=%r>" % (self.__class__.__name__, self._name)
         
         @property
         def cache_path(self):
             return self._cache_path
         
         def enable(self):
             self._enabled = True
         
         def disable(self):
             self._enabled = False
         
         def flush(self):
             self._cache_file.flush()
         
         def load_overload(self, sig, target_context):
             import os
             import errno
             with self._guard_against_spurious_io_errors():
                 return self._load_overload(sig, target_context)
         
         def _load_overload(self, sig, target_context):
             if not self._enabled:
                 return
             key = self._index_key(sig, target_context.codegen())
             data = self._cache_file.load(key)
             if data is not None:
                 data = self._impl.rebuild(target_context, data)
             return data
         
         def save_overload(self, sig, data):
             with self._guard_against_spurious_io_errors():
                 self._save_overload(sig, data)
         
         def _save_overload(self, sig, data):
             if not self._enabled:
                 return
             if not self._impl.check_cachable(data):
                 return
             self._impl.locator.ensure_cache_path()
             key = self._index_key(sig, data.codegen)
             data = self._impl.reduce(data)
             self._cache_file.save(key, data)
         
         @contextlib.contextmanager
         def _guard_against_spurious_io_errors(self):
             import os
             import errno
             if os.name == "nt":
                 try:
                     yield
                 except OSError as e:
                     if e.errno != errno.EACCES:
                         raise
             else:
                 yield
         
         def _index_key(self, sig, codegen):
             """Compute index key for the given signature and codegen."""
             from numba.cuda.serialize import dumps
             import hashlib
             codebytes = self._py_func.__code__.co_code
             if self._py_func.__closure__ is not None:
                 cvars = tuple([x.cell_contents for x in self._py_func.__closure__])
                 cvarbytes = dumps(cvars)
             else:
                 cvarbytes = b""
             hasher = lambda x: hashlib.sha256(x).hexdigest()
             return (
                 sig,
                 codegen.magic_tuple(),
                 (hasher(codebytes), hasher(cvarbytes)),
             )
     ```
   - Edge cases: Windows file permission issues handled by guard
   - Integration: CUBIECache inherits from this base

7. **Remove stub classes and update conditional imports**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     - Delete `_StubCacheLocator` class (lines 170-194)
     - Delete `_StubCacheImpl` class (lines 196-217)
     - Delete `_StubIndexDataCacheFile` class (lines 219-242)
     - Delete `_StubCUDACache` class (lines 244-296)
     - Remove the conditional import block (lines 299-312) that assigns stubs in CUDASIM mode
     - The vendored classes are now used in both modes
     - Keep `_CACHING_AVAILABLE = True` always (or remove the variable if not used elsewhere)
   - Edge cases: Ensure no other code depends on stub classes
   - Integration: cubie_cache.py imports these classes directly

8. **Add required imports at top of caching section**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     ```python
     # Add import for ABCMeta at top of file (before contextlib import)
     from abc import ABCMeta, abstractmethod
     ```
   - Edge cases: None
   - Integration: Required for abstract base classes

9. **Update __all__ exports**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     - Keep the same exports: `_CacheLocator`, `CacheImpl`, `IndexDataCacheFile`, `CUDACache`
     - Remove `_CACHING_AVAILABLE` from `__all__` if it's no longer needed
   - Edge cases: Other modules may check `_CACHING_AVAILABLE`
   - Integration: cubie_cache.py imports from this module

**Tests to Create**:
- None for this task group (existing tests will validate changes)

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_locator_get_cache_path
- tests/test_cubie_cache.py::test_cache_locator_get_source_stamp
- tests/test_cubie_cache.py::test_cache_locator_get_disambiguator
- tests/test_cubie_cache.py::test_cache_locator_from_function_raises
- tests/test_cubie_cache.py::test_cache_impl_locator_property
- tests/test_cubie_cache.py::test_cache_impl_filename_base
- tests/test_cubie_cache.py::test_cache_impl_check_cachable
- tests/test_cubie_cache.py::test_cache_locator_instantiation_works
- tests/test_cubie_cache.py::test_cache_impl_instantiation_works

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: Update cubie_cache.py to Remove CUDASIM Early Returns
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/cubie_cache.py (entire file)
- File: src/cubie/cuda_simsafe.py (lines 459-462 for is_cudasim_enabled)

**Input Validation Required**:
- None - no new parameters

**Tasks**:
1. **Remove CUDASIM check in create_cache function**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     ```python
     # In create_cache function (lines 483-499), remove lines 483-486:
     # REMOVE:
     #     from cubie.cuda_simsafe import is_cudasim_enabled
     #
     #     if is_cudasim_enabled():
     #         return None
     
     # The function should now allow cache creation in CUDASIM mode
     # Cache operations will work for filesystem operations
     # Only serialization of real kernels will fail (expected)
     ```
   - Edge cases: Tests that expect None in CUDASIM need updating
   - Integration: Solvers can now use cache infrastructure in CUDASIM

2. **Remove CUDASIM check in invalidate_cache function**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     ```python
     # In invalidate_cache function (lines 526-529), remove lines 526-529:
     # REMOVE:
     #     from cubie.cuda_simsafe import is_cudasim_enabled
     #
     #     if is_cudasim_enabled():
     #         return
     
     # Allow cache invalidation operations in CUDASIM mode
     ```
   - Edge cases: flush_cache operations should work on filesystem
   - Integration: Cache management works in both modes

**Tests to Create**:
- None for this task group (existing tests will be updated in Task Group 3)

**Tests to Run**:
- tests/test_cubie_cache.py::test_create_cache_returns_none_when_disabled
- tests/test_cubie_cache.py::test_invalidate_cache_no_op_when_hash_mode
- tests/test_cubie_cache.py::test_invalidate_cache_flushes_when_flush_mode

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: Update Cache Tests to Remove nocudasim Markers
**Status**: [ ]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: tests/test_cubie_cache.py (entire file)

**Input Validation Required**:
- None - test modifications only

**Tasks**:
1. **Remove nocudasim marker from test_cubie_cache_init**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     ```python
     # Line 131: Remove @pytest.mark.nocudasim decorator
     # The test should now run in CUDASIM mode
     # CUBIECache instantiation uses vendored classes and works
     ```
   - Edge cases: Test should pass in both CUDA and CUDASIM modes
   - Integration: Validates cache initialization works

2. **Remove nocudasim marker from test_cubie_cache_index_key**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     ```python
     # Line 144: Remove @pytest.mark.nocudasim decorator
     # The test exercises _index_key method which is pure computation
     ```
   - Edge cases: Test should pass in both modes
   - Integration: Validates index key generation

3. **Remove nocudasim marker from test_cubie_cache_path**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     ```python
     # Line 174: Remove @pytest.mark.nocudasim decorator
     # The test verifies cache_path property which is filesystem-based
     ```
   - Edge cases: Test should pass in both modes
   - Integration: Validates path handling

4. **Update test_create_cache_returns_none_in_cudasim test**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     ```python
     # Lines 252-264: Update the test expectation
     # After refactor, create_cache returns a CUBIECache in CUDASIM mode
     # Rename test to test_create_cache_returns_cache_in_cudasim
     # Update assertion:
     
     def test_create_cache_returns_cache_in_cudasim():
         """Verify create_cache returns CUBIECache in CUDASIM mode."""
         from cubie.cubie_cache import create_cache, CUBIECache
         from cubie.cuda_simsafe import is_cudasim_enabled
     
         if is_cudasim_enabled():
             result = create_cache(
                 cache_arg=True,
                 system_name="test_system",
                 system_hash="abc123",
                 config_hash="def456789012345678901234567890123456"
                 "789012345678901234567890abcd",
             )
             assert isinstance(result, CUBIECache)
     ```
   - Edge cases: Only runs assertion in CUDASIM mode
   - Integration: Validates the new behavior

5. **Update test_invalidate_cache_flushes_when_flush_mode test**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     ```python
     # Lines 308-330: Remove the early return for CUDASIM mode
     # The test should now work in both modes since invalidate_cache
     # no longer has a CUDASIM early return
     
     def test_invalidate_cache_flushes_when_flush_mode(tmp_path):
         """Verify invalidate_cache calls flush in flush_on_change mode."""
         from cubie.cubie_cache import invalidate_cache
     
         # Test should work in both CUDA and CUDASIM modes now
         invalidate_cache(
             cache_arg="flush_on_change",
             system_name="test_system",
             system_hash="abc123",
             config_hash="def456789012345678901234567890123456"
             "789012345678901234567890abcd",
         )
         # No exception means success - flush_cache was called
     ```
   - Edge cases: Filesystem operations should work in both modes
   - Integration: Validates cache invalidation works

**Tests to Create**:
- None - existing tests are being updated

**Tests to Run**:
- tests/test_cubie_cache.py (all tests in file)

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 3

**Dependency Chain**:
1. Task Group 1 (vendor classes in cuda_simsafe.py) → independent
2. Task Group 2 (update cubie_cache.py) → depends on Group 1
3. Task Group 3 (update tests) → depends on Groups 1 and 2

**Tests to be Created**: None (existing tests updated)

**Tests to be Run**:
- All tests in tests/test_cubie_cache.py

**Estimated Complexity**:
- Task Group 1: Medium-High (vendoring ~150 lines of numba-cuda code)
- Task Group 2: Low (removing 2 conditional blocks)
- Task Group 3: Low (removing 3 markers, updating 2 tests)

**Key Risk**: The vendored classes must faithfully reproduce numba-cuda behavior for filesystem operations. The serialization fallback to pickle may produce different output than numba.cuda.serialize.dumps, but this only affects actual kernel caching (which fails in CUDASIM anyway).
