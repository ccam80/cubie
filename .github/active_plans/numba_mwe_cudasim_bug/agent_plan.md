# Numba CUDASIM Bug MWE - Agent Plan

## Overview

Create a Minimal Working Example (MWE) demonstrating a flaky bug in Numba's CUDA simulator where threads that should return early continue executing and fail when calling `cuda.local.array()`.

## Component Descriptions

### 1. allocator_factory.py

**Purpose**: Factory class that generates buffer allocator device functions.

**Behavior**:
- Takes `buffer_size` as initialization parameter
- Has a `build()` method that returns a CUDA device function
- The device function uses `cuda.local.array(buffer_size, dtype)` to allocate memory
- Uses `@cuda.jit(device=True, inline=True)` decorator to match CuBIE's pattern

**Expected Interface**:
```python
class AllocatorFactory:
    def __init__(self, buffer_size):
        # Store buffer size for use in build
        
    def build(self):
        # Return compiled allocator device function
        # Allocator signature: () -> local_array
```

**Key Constraints**:
- Must use `@cuda.jit(device=True, inline=True)` decorator
- Must call `cuda.local.array()` inside the device function
- No dependencies on CuBIE source

### 2. kernel_factory.py

**Purpose**: Factory class that builds CUDA kernels using the allocator factory.

**Behavior**:
- Takes an `AllocatorFactory` instance as initialization parameter
- Has a `build()` method that:
  1. Calls allocator_factory.build() to get the allocator function
  2. Captures the allocator in a closure (like CuBIE does)
  3. Returns a compiled CUDA kernel

**Kernel Logic**:
1. Get thread index using `cuda.grid(1)`
2. If thread index >= n_threads argument, return early
3. Call the allocator to get a local array
4. Assign 1 to an element of the local array
5. Copy the value from local array to the output array at thread index position

**Expected Interface**:
```python
class KernelFactory:
    def __init__(self, allocator_factory):
        # Store allocator factory
        
    def build(self):
        # Get allocator, capture in closure, return kernel
```

**Key Constraints**:
- Allocator must be fetched in build() and captured in closure
- Kernel must check thread index and return early if >= n_threads
- The bug occurs when early-return threads continue executing

### 3. conftest.py

**Purpose**: Pytest fixtures for the MWE tests.

**Fixtures Required**:

#### settings_dict fixture
- Function-scoped
- Accepts parametrization via `request.param` for overrides
- Default values:
  - `array_size`: 10
  - `buffer_size`: 5  
  - `n_threads`: 7 (less than array_size to have excess threads)

#### allocator_factory fixture
- Function-scoped
- Uses settings_dict to get buffer_size
- Returns AllocatorFactory instance

#### kernel fixture
- Function-scoped
- Uses allocator_factory fixture
- Calls allocator_factory to create KernelFactory
- Returns result of kernel_factory.build()

**Key Constraints**:
- All fixtures must be function-scoped (fresh instances maximize bug reproduction)
- Follow CuBIE conftest.py patterns for settings override
- No mocks or patches

### 4. test_mwe.py

**Purpose**: Collection of tests to trigger the flaky bug.

**Test Pattern**:
Each test should:
1. Create a zeros array of size `array_size` (from settings)
2. Call the kernel with appropriate thread count
3. Pass `n_threads` as argument (threads >= this should return)
4. Verify that first `n_threads` elements are 1, rest are 0

**Number of Tests**: 10-20 tests following identical pattern

**Rationale for Multiple Tests**:
- The bug is flaky/intermittent
- Having multiple tests increases probability of triggering
- Each test creates fresh fixture instances due to function scope

**Expected Test Structure**:
```python
def test_mwe_case_1(kernel, settings_dict):
    # Create zeros array
    # Launch kernel
    # Assert expected output pattern
```

## Integration Points

### Fixture Dependencies

```
settings_dict (function scope)
    ↓
allocator_factory (function scope, uses settings_dict.buffer_size)
    ↓
kernel (function scope, uses allocator_factory)
    ↓
test functions (use kernel and settings_dict)
```

### Kernel Launch Pattern

```python
# Block/thread configuration
# Using blockDim of 32 (standard warp size)
# Number of blocks = ceil(array_size / 32)
kernel[(blocks,), (32,)](output_array, n_threads)
```

## Data Structures

### Settings Dict
```python
{
    'array_size': int,   # Size of output array, also determines thread count
    'buffer_size': int,  # Size of local array in allocator
    'n_threads': int,    # Number of threads that should actually do work
}
```

### Output Array
- NumPy array of zeros with dtype=float32
- Size = settings_dict['array_size']
- After kernel execution: first n_threads elements = 1.0, rest = 0.0

## Edge Cases

1. **Thread index exactly at boundary**: Thread with idx == n_threads should return
2. **All threads valid**: n_threads == array_size (no excess threads)
3. **Single thread**: n_threads == 1, array_size > 1
4. **Empty work**: n_threads == 0 (all threads should return)

## Dependencies

### Required Imports
- `numpy` - array creation and assertions
- `numba.cuda` - CUDA primitives
- `pytest` - testing framework

### No CuBIE Dependencies
The MWE must not import anything from cubie source code.

## File Structure

```
tests/numba_mwe/
├── __init__.py          # Empty, makes directory a package
├── allocator_factory.py # AllocatorFactory class
├── kernel_factory.py    # KernelFactory class  
├── conftest.py          # Pytest fixtures
└── test_mwe.py          # 10-20 test functions
```

## Running the MWE

To run the tests and attempt to trigger the bug:

```bash
cd /home/runner/work/cubie/cubie
NUMBA_ENABLE_CUDASIM=1 pytest tests/numba_mwe/ -v
```

To run multiple times to increase chance of triggering:

```bash
for i in {1..10}; do NUMBA_ENABLE_CUDASIM=1 pytest tests/numba_mwe/ -v; done
```

## Success Criteria

1. Tests pass when bug does not manifest
2. When bug manifests, error message contains: `module 'numba.cuda' has no attribute 'local'`
3. Error occurs on a thread with index >= n_threads (proving early return failed)
4. MWE is completely self-contained with no CuBIE imports
