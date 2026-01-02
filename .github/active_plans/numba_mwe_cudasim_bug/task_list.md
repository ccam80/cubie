# Implementation Task List
# Feature: Numba CUDASIM Bug MWE
# Plan Reference: .github/active_plans/numba_mwe_cudasim_bug/agent_plan.md

## Task Group 1: Directory Structure and Init
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: tests/ (directory listing only)
- Reference: .github/active_plans/numba_mwe_cudasim_bug/agent_plan.md (lines 178-189)

**Input Validation Required**:
- None (file creation only)

**Tasks**:
1. **Create tests/numba_mwe/ directory**
   - Action: Create directory

2. **Create tests/numba_mwe/__init__.py**
   - File: tests/numba_mwe/__init__.py
   - Action: Create
   - Details:
     ```python
     """Minimal Working Example for Numba CUDASIM bug reproduction."""
     ```

**Tests to Create**:
- None for this group

**Tests to Run**:
- None for this group

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: AllocatorFactory Implementation
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 76-143) - CUDABuffer.build_allocator pattern
- File: .github/active_plans/numba_mwe_cudasim_bug/agent_plan.md (lines 9-34)

**Input Validation Required**:
- buffer_size: Must be positive integer (> 0)

**Tasks**:
1. **Create AllocatorFactory class**
   - File: tests/numba_mwe/allocator_factory.py
   - Action: Create
   - Details:
     ```python
     """Factory for creating buffer allocator device functions."""
     
     import numpy as np
     from numba import cuda
     
     
     class AllocatorFactory:
         """Factory that generates buffer allocator device functions.
         
         Parameters
         ----------
         buffer_size : int
             Size of the local array to allocate.
         
         Attributes
         ----------
         buffer_size : int
             Size of the local array to allocate.
         """
         
         def __init__(self, buffer_size: int):
             """Initialize the allocator factory.
             
             Parameters
             ----------
             buffer_size : int
                 Size of the local array to allocate.
             """
             if buffer_size <= 0:
                 raise ValueError("buffer_size must be positive")
             self.buffer_size = buffer_size
         
         def build(self):
             """Build and return a CUDA device function allocator.
             
             Returns
             -------
             callable
                 CUDA device function that allocates a local array.
                 Signature: () -> local_array
             """
             _buffer_size = self.buffer_size
             
             @cuda.jit(device=True, inline=True)
             def allocate_buffer():
                 """Allocate a local array buffer."""
                 return cuda.local.array(_buffer_size, dtype=np.float32)
             
             return allocate_buffer
     ```
   - Edge cases:
     - buffer_size <= 0 should raise ValueError
   - Integration: Standalone, no CuBIE imports

**Tests to Create**:
- None for this group (tested via kernel in test_mwe.py)

**Tests to Run**:
- None for this group

**Outcomes**:
- Files Modified:
  * tests/numba_mwe/allocator_factory.py (47 lines created)
- Functions/Methods Added/Modified:
  * AllocatorFactory.__init__() in allocator_factory.py
  * AllocatorFactory.build() in allocator_factory.py
- Implementation Summary:
  Created AllocatorFactory class following the CuDABuffer.build_allocator
  pattern from buffer_registry.py. The factory accepts buffer_size,
  validates it is positive, and builds a CUDA device function that
  allocates a local array using cuda.local.array() with the buffer size
  captured in closure.
- Issues Flagged: None

---

## Task Group 3: KernelFactory Implementation
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: tests/numba_mwe/allocator_factory.py (entire file, created in Task Group 2)
- File: src/cubie/CUDAFactory.py (lines 441-522) - Factory pattern reference
- File: .github/active_plans/numba_mwe_cudasim_bug/agent_plan.md (lines 36-66)

**Input Validation Required**:
- allocator_factory: Must be an AllocatorFactory instance

**Tasks**:
1. **Create KernelFactory class**
   - File: tests/numba_mwe/kernel_factory.py
   - Action: Create
   - Details:
     ```python
     """Factory for creating CUDA kernels that use the allocator."""
     
     from numba import cuda
     
     from tests.numba_mwe.allocator_factory import AllocatorFactory
     
     
     class KernelFactory:
         """Factory that builds CUDA kernels using an allocator factory.
         
         Parameters
         ----------
         allocator_factory : AllocatorFactory
             Factory for creating the buffer allocator device function.
         
         Attributes
         ----------
         allocator_factory : AllocatorFactory
             Factory for creating the buffer allocator device function.
         """
         
         def __init__(self, allocator_factory: AllocatorFactory):
             """Initialize the kernel factory.
             
             Parameters
             ----------
             allocator_factory : AllocatorFactory
                 Factory for creating the buffer allocator device function.
             """
             if not isinstance(allocator_factory, AllocatorFactory):
                 raise TypeError(
                     "allocator_factory must be an AllocatorFactory instance"
                 )
             self.allocator_factory = allocator_factory
         
         def build(self):
             """Build and return a CUDA kernel.
             
             The kernel:
             1. Gets thread index via cuda.grid(1)
             2. Returns early if thread index >= n_threads
             3. Calls allocator to get local array
             4. Assigns 1 to first element of local array
             5. Copies value from local array to output array
             
             Returns
             -------
             callable
                 CUDA kernel function.
                 Signature: (output_array, n_threads) -> None
             """
             # Get allocator and capture in closure
             allocator = self.allocator_factory.build()
             
             @cuda.jit
             def kernel(output, n_threads):
                 """CUDA kernel that uses local array allocation.
                 
                 Parameters
                 ----------
                 output : array
                     Output array to write results.
                 n_threads : int
                     Number of threads that should do work.
                 """
                 thread_idx = cuda.grid(1)
                 
                 # Early return for excess threads
                 # BUG: In CUDASIM, threads may continue past this return
                 if thread_idx >= n_threads:
                     return
                 
                 # Allocate local array (fails if thread continues past return)
                 local_buffer = allocator()
                 
                 # Use the local buffer
                 local_buffer[0] = 1.0
                 
                 # Copy to output
                 output[thread_idx] = local_buffer[0]
             
             return kernel
     ```
   - Edge cases:
     - allocator_factory must be correct type
     - Thread index at exact boundary (n_threads) should return
   - Integration: Uses AllocatorFactory from allocator_factory.py

**Tests to Create**:
- None for this group (tested via test_mwe.py)

**Tests to Run**:
- None for this group

**Outcomes**:
- Files Modified:
  * tests/numba_mwe/kernel_factory.py (77 lines created)
- Functions/Methods Added/Modified:
  * KernelFactory.__init__() in kernel_factory.py
  * KernelFactory.build() in kernel_factory.py
- Implementation Summary:
  Created KernelFactory class following the factory pattern from
  CUDAFactory.py. The factory accepts an AllocatorFactory instance,
  validates it is the correct type, and builds a CUDA kernel that
  uses the allocator device function captured in closure. The kernel
  demonstrates the CUDASIM bug where threads may continue past early
  return and fail when calling cuda.local.array().
- Issues Flagged: None

---

## Task Group 4: Conftest Fixtures
**Status**: [x]
**Dependencies**: Task Group 3

**Required Context**:
- File: tests/numba_mwe/allocator_factory.py (entire file)
- File: tests/numba_mwe/kernel_factory.py (entire file)
- File: tests/conftest.py (lines 476-577) - Settings override pattern reference
- File: .github/active_plans/numba_mwe_cudasim_bug/agent_plan.md (lines 68-98)

**Input Validation Required**:
- None (fixtures handle validation via factory classes)

**Tasks**:
1. **Create conftest.py with fixtures**
   - File: tests/numba_mwe/conftest.py
   - Action: Create
   - Details:
     ```python
     """Pytest fixtures for Numba CUDASIM MWE tests."""
     
     import pytest
     
     from tests.numba_mwe.allocator_factory import AllocatorFactory
     from tests.numba_mwe.kernel_factory import KernelFactory
     
     
     @pytest.fixture(scope="function")
     def settings_dict(request):
         """Return settings dictionary for MWE tests.
         
         Default values:
         - array_size: 10 (size of output array)
         - buffer_size: 5 (size of local array in allocator)
         - n_threads: 7 (threads that do work, less than array_size)
         
         Accepts parametrization via request.param for overrides.
         """
         defaults = {
             "array_size": 10,
             "buffer_size": 5,
             "n_threads": 7,
         }
         
         if hasattr(request, "param") and request.param is not None:
             defaults.update(request.param)
         
         return defaults
     
     
     @pytest.fixture(scope="function")
     def allocator_factory(settings_dict):
         """Return a fresh AllocatorFactory instance.
         
         Uses buffer_size from settings_dict.
         """
         return AllocatorFactory(buffer_size=settings_dict["buffer_size"])
     
     
     @pytest.fixture(scope="function")
     def kernel(allocator_factory):
         """Return a compiled CUDA kernel.
         
         Creates a KernelFactory using the allocator_factory and
         returns the result of build().
         """
         kernel_factory = KernelFactory(allocator_factory)
         return kernel_factory.build()
     ```
   - Edge cases:
     - request.param may not exist (hasattr check)
     - request.param may be None
   - Integration: Uses AllocatorFactory and KernelFactory

**Tests to Create**:
- None for this group

**Tests to Run**:
- None for this group

**Outcomes**:
- Files Modified:
  * tests/numba_mwe/conftest.py (47 lines created)
- Functions/Methods Added/Modified:
  * settings_dict() fixture in conftest.py
  * allocator_factory() fixture in conftest.py
  * kernel() fixture in conftest.py
- Implementation Summary:
  Created conftest.py with three function-scoped pytest fixtures following
  the pattern from tests/conftest.py. The settings_dict fixture provides
  default test parameters with support for parametrization overrides. The
  allocator_factory fixture creates an AllocatorFactory using buffer_size
  from settings_dict. The kernel fixture creates a KernelFactory and
  returns the built kernel. All fixtures are function-scoped to maximize
  fresh instance creation for bug reproduction.
- Issues Flagged: None

---

## Task Group 5: Test File Implementation
**Status**: [x]
**Dependencies**: Task Group 4

**Required Context**:
- File: tests/numba_mwe/conftest.py (entire file)
- File: .github/active_plans/numba_mwe_cudasim_bug/agent_plan.md (lines 99-125)
- File: .github/active_plans/numba_mwe_cudasim_bug/agent_plan.md (lines 145-157)

**Input Validation Required**:
- None (tests use fixtures which handle validation)

**Tasks**:
1. **Create test_mwe.py with 15 test functions**
   - File: tests/numba_mwe/test_mwe.py
   - Action: Create
   - Details:
     ```python
     """Tests for reproducing Numba CUDASIM flaky bug.
     
     The bug occurs when threads that should return early continue
     executing and fail when calling cuda.local.array().
     
     Error message: "module 'numba.cuda' has no attribute 'local'"
     
     Having multiple identical tests increases the probability of
     triggering the flaky bug since each test creates fresh fixture
     instances due to function scope.
     """
     
     import numpy as np
     from numba import cuda
     
     
     def _run_kernel_test(kernel, settings_dict):
         """Helper to run kernel and verify output.
         
         Parameters
         ----------
         kernel : callable
             Compiled CUDA kernel.
         settings_dict : dict
             Test settings with array_size and n_threads.
         """
         array_size = settings_dict["array_size"]
         n_threads = settings_dict["n_threads"]
         
         # Create output array
         output = np.zeros(array_size, dtype=np.float32)
         
         # Calculate grid dimensions
         threads_per_block = 32
         blocks = (array_size + threads_per_block - 1) // threads_per_block
         
         # Launch kernel
         kernel[(blocks,), (threads_per_block,)](output, n_threads)
         
         # Synchronize
         cuda.synchronize()
         
         # Verify results
         # First n_threads elements should be 1.0
         assert np.all(output[:n_threads] == 1.0), (
             f"Expected first {n_threads} elements to be 1.0, "
             f"got {output[:n_threads]}"
         )
         # Remaining elements should be 0.0
         if n_threads < array_size:
             assert np.all(output[n_threads:] == 0.0), (
                 f"Expected elements [{n_threads}:] to be 0.0, "
                 f"got {output[n_threads:]}"
             )
     
     
     def test_mwe_case_01(kernel, settings_dict):
         """MWE test case 1."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_02(kernel, settings_dict):
         """MWE test case 2."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_03(kernel, settings_dict):
         """MWE test case 3."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_04(kernel, settings_dict):
         """MWE test case 4."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_05(kernel, settings_dict):
         """MWE test case 5."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_06(kernel, settings_dict):
         """MWE test case 6."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_07(kernel, settings_dict):
         """MWE test case 7."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_08(kernel, settings_dict):
         """MWE test case 8."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_09(kernel, settings_dict):
         """MWE test case 9."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_10(kernel, settings_dict):
         """MWE test case 10."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_11(kernel, settings_dict):
         """MWE test case 11."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_12(kernel, settings_dict):
         """MWE test case 12."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_13(kernel, settings_dict):
         """MWE test case 13."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_14(kernel, settings_dict):
         """MWE test case 14."""
         _run_kernel_test(kernel, settings_dict)
     
     
     def test_mwe_case_15(kernel, settings_dict):
         """MWE test case 15."""
         _run_kernel_test(kernel, settings_dict)
     ```
   - Edge cases:
     - n_threads == array_size (all threads valid)
     - n_threads == 0 (all threads return early)
     - Thread at exact boundary (handled by kernel)
   - Integration: Uses fixtures from conftest.py
   - Note: Tests do NOT have type hints per project conventions

**Tests to Create**:
- tests/numba_mwe/test_mwe.py::test_mwe_case_01 through test_mwe_case_15

**Tests to Run**:
- tests/numba_mwe/test_mwe.py::test_mwe_case_01
- tests/numba_mwe/test_mwe.py::test_mwe_case_02
- tests/numba_mwe/test_mwe.py::test_mwe_case_03
- tests/numba_mwe/test_mwe.py::test_mwe_case_04
- tests/numba_mwe/test_mwe.py::test_mwe_case_05
- tests/numba_mwe/test_mwe.py::test_mwe_case_06
- tests/numba_mwe/test_mwe.py::test_mwe_case_07
- tests/numba_mwe/test_mwe.py::test_mwe_case_08
- tests/numba_mwe/test_mwe.py::test_mwe_case_09
- tests/numba_mwe/test_mwe.py::test_mwe_case_10
- tests/numba_mwe/test_mwe.py::test_mwe_case_11
- tests/numba_mwe/test_mwe.py::test_mwe_case_12
- tests/numba_mwe/test_mwe.py::test_mwe_case_13
- tests/numba_mwe/test_mwe.py::test_mwe_case_14
- tests/numba_mwe/test_mwe.py::test_mwe_case_15

**Outcomes**:
- Files Modified:
  * tests/numba_mwe/test_mwe.py (115 lines created)
- Functions/Methods Added/Modified:
  * _run_kernel_test() helper function in test_mwe.py
  * test_mwe_case_01() through test_mwe_case_15() in test_mwe.py
- Implementation Summary:
  Created test_mwe.py with module docstring explaining the CUDASIM bug
  being reproduced. Implemented _run_kernel_test() helper that creates
  output array, calculates grid dimensions, launches kernel, synchronizes,
  and verifies results. Created 15 identical test functions (test_mwe_case_01
  through test_mwe_case_15) that call the helper with kernel and settings_dict
  fixtures. No type hints on test functions per project conventions.
- Issues Flagged: None

---

# Summary

## Total Task Groups: 5

## Dependency Chain:
```
Task Group 1: Directory Structure
       ↓
Task Group 2: AllocatorFactory
       ↓
Task Group 3: KernelFactory
       ↓
Task Group 4: Conftest Fixtures
       ↓
Task Group 5: Test File
```

## Files to Create:
1. tests/numba_mwe/__init__.py
2. tests/numba_mwe/allocator_factory.py
3. tests/numba_mwe/kernel_factory.py
4. tests/numba_mwe/conftest.py
5. tests/numba_mwe/test_mwe.py

## Tests to Run:
- 15 test functions in tests/numba_mwe/test_mwe.py

## Estimated Complexity: Low
- Self-contained MWE with no CuBIE dependencies
- Simple factory pattern following existing codebase conventions
- Straightforward fixture chain
- Identical test functions to maximize bug reproduction probability

## Notes:
- All fixtures are function-scoped to maximize fresh instance creation
- The bug is flaky; having 15 tests increases probability of triggering
- Tests should be run with NUMBA_ENABLE_CUDASIM=1 to reproduce the bug
- No type hints in test functions per project conventions
