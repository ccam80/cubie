# Agent Plan: Consolidate Memory Sizing Properties

## Overview

This plan consolidates memory sizing properties to the base `CUDAFactory` class, removing redundant implementations distributed throughout the codebase. The `buffer_registry` singleton is the source of truth for memory allocation, and CUDAFactory subclasses should inherit memory sizing properties from the base class.

---

## Component Analysis

### 1. Base CUDAFactory Class (`src/cubie/CUDAFactory.py`)

**Current State**: No memory sizing properties exist on the base class.

**Required Changes**: Add three core properties that delegate to `buffer_registry`:
- `shared_buffer_size` → `buffer_registry.shared_buffer_size(self)`
- `local_buffer_size` → `buffer_registry.local_buffer_size(self)`
- `persistent_local_buffer_size` → `buffer_registry.persistent_local_buffer_size(self)`

**Integration Points**:
- Must import `buffer_registry` from `cubie.buffer_registry`
- Properties should return `int` values
- Properties should handle case where no buffers are registered (returns 0)

---

### 2. IVPLoop (`src/cubie/integrators/loops/ode_loop.py`)

**Current Properties** (lines 875-887):
```python
@property
def shared_memory_elements(self) -> int:
    return buffer_registry.shared_buffer_size(self)

@property
def local_memory_elements(self) -> int:
    return buffer_registry.local_buffer_size(self)

@property
def persistent_local_elements(self) -> int:
    return buffer_registry.persistent_local_buffer_size(self)
```

**Decision**: These can be REMOVED since they will be inherited from CUDAFactory.

**Note**: The naming uses `*_elements` while buffer_registry uses `*_buffer_size`. If we want to maintain backward compatibility, we could keep these as aliases. However, since the issue explicitly requests removing redundant properties, they should be removed.

---

### 3. BaseAlgorithmStep (`src/cubie/integrators/algorithms/base_algorithm_step.py`)

**Current Properties** (lines 641-653):
```python
@property
def shared_memory_elements(self) -> int:
    return buffer_registry.shared_buffer_size(self)

@property
def local_scratch_elements(self) -> int:
    return buffer_registry.local_buffer_size(self)

@property
def persistent_local_elements(self) -> int:
    return buffer_registry.persistent_local_buffer_size(self)
```

**Decision**: These can be REMOVED since they will be inherited from CUDAFactory.

**Note**: `local_scratch_elements` has a different name than the base class property. This is a naming inconsistency that should be resolved.

---

### 4. Step Controllers

**BaseStepController** (`src/cubie/integrators/step_control/base_step_controller.py`):

**Current Abstract Property** (lines 175-180):
```python
@property
@abstractmethod
def local_memory_elements(self) -> int:
    """Return the number of local scratch elements required."""
    return 0
```

**Analysis**: This is different from the buffer_registry pattern. Step controllers define `local_memory_elements` as a compile-time constant that is used by `register_buffers()` to register the timestep buffer. The property returns the buffer SIZE, not the registry-computed size.

**Decision**: KEEP this abstract property. It serves a different purpose - it defines the buffer size requirement, which is then registered with buffer_registry. The base class `shared_buffer_size` etc. would reflect what's registered.

**Concrete Controllers with local_memory_elements**:
- `FixedStepController` (line 162): returns 1
- `AdaptiveIController` (line 50): returns 1  
- `AdaptivePIController` (line 93): returns 2
- `AdaptivePIDController` (line 98): returns 3
- `GustaffssonController` (line 112): returns 2
- `AdaptiveStepController` (line 311): abstract, returns None (base for above)

---

### 5. LinearSolver (`src/cubie/integrators/matrix_free_solvers/linear_solver.py`)

**Current Properties** (lines 575-583):
```python
@property
def local_buffer_size(self) -> int:
    return buffer_registry.local_buffer_size(self)

@property
def persistent_local_buffer_size(self) -> int:
    return buffer_registry.persistent_local_buffer_size(self)
```

**Decision**: REMOVE - will be inherited from CUDAFactory (though naming matches).

---

### 6. NewtonKrylovSolver (`src/cubie/integrators/matrix_free_solvers/newton_krylov.py`)

**Current Properties** (lines 593-605):
```python
@property
def local_buffer_size(self) -> int:
    return buffer_registry.local_buffer_size(self)

@property
def persistent_local_buffer_size(self) -> int:
    return buffer_registry.persistent_local_buffer_size(self)
```

**Decision**: REMOVE - will be inherited from CUDAFactory.

---

### 7. SingleIntegratorRun (`src/cubie/integrators/SingleIntegratorRun.py`)

**Current Properties**:
```python
# Core aggregation (lines 59-80) - delegates to _loop
@property
def shared_memory_elements(self) -> int:
    return self._loop.shared_memory_elements

@property
def shared_memory_bytes(self) -> int:
    element_count = self.shared_memory_elements
    itemsize = np_dtype(self.precision).itemsize
    return element_count * itemsize

@property
def local_memory_elements(self) -> int:
    return self._loop.local_memory_elements

@property
def persistent_local_elements(self) -> int:
    return self._loop.persistent_local_elements

# Child-specific (lines 214-269) - LEGACY, should be removed
@property
def shared_memory_elements_loop(self) -> int:
    return self._loop.shared_memory_elements

@property
def local_memory_elements_loop(self) -> int:
    return self._loop.local_memory_elements

@property
def local_memory_elements_controller(self) -> int:
    return self._step_controller.local_memory_elements
```

**Decision**:
- KEEP `shared_memory_elements`, `local_memory_elements`, `persistent_local_elements`, `shared_memory_bytes` - these are aggregation properties that delegate to the loop
- REMOVE `shared_memory_elements_loop` - redundant, same as `shared_memory_elements`
- REMOVE `local_memory_elements_loop` - redundant, same as `local_memory_elements`
- REMOVE `local_memory_elements_controller` - legacy child-specific property

---

### 8. BatchSolverKernel (`src/cubie/batchsolving/BatchSolverKernel.py`)

**Current Properties** (lines 904-932):
```python
@property
def local_memory_elements(self) -> int:
    return self.compile_settings.local_memory_elements

@property
def shared_memory_elements(self) -> int:
    return self.compile_settings.shared_memory_elements

@property
def shared_memory_bytes(self) -> int:
    return self.single_integrator.shared_memory_bytes
```

**Decision**: 
- KEEP all - these read from compile_settings which is set during initialization from single_integrator. This is correct behavior for BatchSolverKernel as it needs to cache these values.

---

## Test Updates Required

### `tests/batchsolving/test_config_plumbing.py`

**Current References** (lines 233-234, 251, 301):
- `cs.local_memory_elements == kernel.local_memory_elements` - KEEP
- `cs.shared_memory_elements == kernel.shared_memory_elements` - KEEP
- Comment mentions `shared_memory_elements, local_memory_elements: computed` - UPDATE comment

### `tests/integrators/algorithms/test_step_algorithms.py`

**Current References** (lines 510-552):
- `"persistent_local_elements": 0` in expected properties dict - UPDATE key if property renamed
- Line 612: `step_object.persistent_local_elements` - UPDATE if property renamed
- Line 753: `step_object.persistent_local_elements` - UPDATE if property renamed

### `tests/integrators/step_control/test_controllers.py`

**Current References** (lines 274, 296):
- `step_controller.local_memory_elements` - KEEP (this is the abstract property defining buffer size)

### `tests/integrators/step_control/test_controller_equivalence_sequences.py`

**References** (multiple lines):
- `step_controller.local_memory_elements` - KEEP

### `tests/_utils.py`

**References** (lines 774-776):
- `singleintegratorrun.shared_memory_bytes` - KEEP
- `singleintegratorrun.persistent_local_elements` - UPDATE if property renamed

### `tests/integrators/algorithms/instrumented/`

**Files with persistent_local_elements**:
- `generic_dirk.py` (line 777)
- `generic_erk.py` (line 372)
- `conftest.py` (line 252)

These instrumented copies need to be updated if the property name changes.

---

## Architectural Integration

### Import Changes

**CUDAFactory.py** needs:
```python
from cubie.buffer_registry import buffer_registry
```

### Backward Compatibility Considerations

The following property names differ between current implementations:
- `*_elements` vs `*_buffer_size`
- `local_scratch_elements` vs `local_buffer_size`
- `persistent_local_elements` vs `persistent_local_buffer_size`

**Recommendation**: Use `*_buffer_size` naming in CUDAFactory to match buffer_registry. Subclasses that need legacy names can add deprecated aliases.

---

## Expected Interactions

1. **CUDAFactory subclasses** that call `register_buffers()` will have their buffer requirements tracked by buffer_registry
2. **Property access** on any CUDAFactory subclass returns the buffer_registry computed sizes
3. **Aggregation classes** (SingleIntegratorRun, BatchSolverKernel) delegate to their composed children for sizing
4. **Step controllers** still define `local_memory_elements` as the buffer size to register, not as a computed property

---

## Edge Cases

1. **Unregistered objects**: If `shared_buffer_size` is called before `register_buffers()`, buffer_registry returns 0. This is correct behavior.

2. **Multiple inheritance**: Some classes may inherit from multiple CUDAFactory subclasses. The base class properties work correctly regardless.

3. **Property override**: Subclasses that need custom sizing logic can still override the base class properties.

---

## Dependencies

- `buffer_registry` singleton must be importable from `cubie.buffer_registry`
- No circular import concerns - CUDAFactory doesn't import anything that imports CUDAFactory

---

## Files to Modify

### Source Files
1. `src/cubie/CUDAFactory.py` - Add three core properties
2. `src/cubie/integrators/loops/ode_loop.py` - Remove redundant properties
3. `src/cubie/integrators/algorithms/base_algorithm_step.py` - Remove redundant properties
4. `src/cubie/integrators/matrix_free_solvers/linear_solver.py` - Remove redundant properties
5. `src/cubie/integrators/matrix_free_solvers/newton_krylov.py` - Remove redundant properties
6. `src/cubie/integrators/SingleIntegratorRun.py` - Remove child-specific properties, update delegation

### Test Files
1. `tests/integrators/algorithms/test_step_algorithms.py` - Update property references
2. `tests/_utils.py` - Update property references if renamed
3. `tests/integrators/algorithms/instrumented/generic_dirk.py` - Update if renamed
4. `tests/integrators/algorithms/instrumented/generic_erk.py` - Update if renamed
5. `tests/integrators/algorithms/instrumented/conftest.py` - Update if renamed
