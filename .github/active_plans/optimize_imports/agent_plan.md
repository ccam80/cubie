# Agent Plan: Import Optimization for CUDAFactory Subclasses

## Overview

This plan describes the architectural changes needed to convert whole-module imports to explicit imports across all CUDAFactory subclass files. The goal is to reduce the scope captured by Numba during CUDA JIT compilation, potentially improving compilation time.

## Component Descriptions

### 1. CUDAFactory Base Class (`src/cubie/CUDAFactory.py`)

**Current State:**
- Uses `import numpy as np` for array operations and dtype references
- Uses `import attrs` for class introspection
- Uses `import numba` for type checking and signature building

**Required Behavior:**
- Must retain all current functionality
- Import only the specific numpy functions/types used: `array_equal`, `asarray`, `any`
- Import only the specific attrs functions used: `define`, `fields`, `has`
- Import only the specific numba types used: `types`, `float64`, `float32`, etc.

### 2. ODE Systems (`src/cubie/odesystems/`)

#### 2.1 BaseODE (`baseODE.py`)

**Current State:**
- Uses `import numpy as np` for dtype handling and array operations

**Required Behavior:**
- Import explicit numpy types: `float32`, `float64`, `asarray`
- Retain NDArray type hints via `numpy.typing`

#### 2.2 SymbolicODE (`symbolic/symbolicODE.py`)

**Current State:**
- Uses `import numpy as np` for array creation and type handling
- Imports sympy for symbolic manipulation

**Required Behavior:**
- Convert numpy imports to explicit symbols used in module
- Keep sympy imports as needed for symbolic expression handling

### 3. Integrator Components (`src/cubie/integrators/`)

#### 3.1 SingleIntegratorRunCore (`SingleIntegratorRunCore.py`)

**Current State:**
- Minimal numpy usage; primarily orchestration
- Uses attrs for settings classes

**Required Behavior:**
- Import only what's explicitly used
- This file has fewer device function definitions, so impact is lower

#### 3.2 IVPLoop (`loops/ode_loop.py`)

**Current State:**
- Uses `import numpy as np` for dtype and int32 types
- Defines device functions in `build()` method with closures

**Required Behavior:**
- Import specific numpy types: `float32`, `float64`, `int32`
- Critical: Device functions in `build()` capture these in closure
- Explicit imports ensure minimal capture

#### 3.3 ArrayInterpolator (`array_interpolator.py`)

**Current State:**
- Uses `import numpy as np` for array operations and dtypes

**Required Behavior:**
- Import explicit numpy types used in interpolation calculations
- Minimal device function scope capture

### 4. Algorithm Steps (`src/cubie/integrators/algorithms/`)

All algorithm step files follow a similar pattern:

**Common Current State:**
- `import numpy as np` for dtype handling
- `import attrs` or `from attrs import define, field` for configs
- Use of numba types for device function compilation

**Required Behavior for All:**
- Convert to explicit imports of numpy types
- Keep explicit attrs imports (already correct in some files)
- Import specific numba types: `cuda`, `int32`, `float64`, etc.

**Files to Modify:**
- `base_algorithm_step.py`
- `generic_dirk.py`
- `generic_erk.py`
- `generic_firk.py`
- `generic_rosenbrock_w.py`
- `backwards_euler.py`
- `backwards_euler_predict_correct.py`
- `crank_nicolson.py`
- `explicit_euler.py`
- `ode_implicitstep.py`
- `ode_explicitstep.py`

### 5. Matrix-Free Solvers (`src/cubie/integrators/matrix_free_solvers/`)

#### 5.1 LinearSolver (`linear_solver.py`)

**Current State:**
- Uses `import numpy as np` for dtype handling
- Defines complex solver device functions

**Required Behavior:**
- Explicit numpy imports for types used in solver
- Critical for reducing scope in solver closures

#### 5.2 NewtonKrylov (`newton_krylov.py`)

**Current State:**
- Uses `import numpy as np` for dtype handling
- Imports various numba types

**Required Behavior:**
- Explicit numpy imports
- Already has explicit numba imports (`from numba import cuda, int32, from_dtype`)

### 6. Step Controllers (`src/cubie/integrators/step_control/`)

**Common Pattern:**
- Use numpy for precision types
- Define controller device functions in `build()` methods

**Files to Modify:**
- `base_step_controller.py`
- `adaptive_step_controller.py`
- `adaptive_I_controller.py`
- `adaptive_PI_controller.py`
- `adaptive_PID_controller.py`
- `gustafsson_controller.py`
- `fixed_step_controller.py`

**Required Behavior:**
- Convert `import numpy as np` to explicit dtype imports
- Import only specific numpy functions used

### 7. Output Handling (`src/cubie/outputhandling/`)

#### 7.1 OutputFunctions (`output_functions.py`)

**Current State:**
- Uses `import numpy as np` for dtype handling
- Orchestrates save and summary functions

**Required Behavior:**
- Explicit numpy imports for types used
- Minimal device code, so impact is moderate

#### 7.2 Summary Metrics (`summarymetrics/`)

**Files to Modify:**
- `metrics.py` (base class)
- All 18+ individual metric files

**Current State:**
- Each metric file uses numpy for precision handling
- Device functions defined in `build()` methods

**Required Behavior:**
- Base `metrics.py` should use explicit imports
- Individual metric files that use numpy should import explicitly
- Many metric files (e.g., `mean.py`) already have minimal imports

### 8. Batch Solving (`src/cubie/batchsolving/`)

#### 8.1 BatchSolverKernel (`BatchSolverKernel.py`)

**Current State:**
- Uses `import numpy as np` for dtype handling
- Complex kernel orchestration

**Required Behavior:**
- Explicit numpy imports
- Critical file for batch execution performance

## Integration Points

### Import Dependencies

When converting imports, maintain these dependency patterns:

1. **Precision types flow through:**
   - `CUDAFactory` → child components via `compile_settings.precision`
   - Must ensure `float32`, `float64` are available where needed

2. **Buffer registry integration:**
   - Uses numpy types for precision parameters
   - Must maintain explicit imports in registration calls

3. **Device function compilation:**
   - Closures in `build()` methods capture scope
   - Primary target for optimization

## Data Structures

### Import Categories

For each file, categorize imports as:

1. **Type imports** - Used for type hints only (can use TYPE_CHECKING)
2. **Runtime imports** - Used in actual code execution
3. **Device closure imports** - Captured in CUDA device function closures (HIGHEST PRIORITY)

### Import Mapping Table

| Whole-Module Import | Explicit Replacements |
|--------------------|----------------------|
| `import numpy as np` | `from numpy import float32, float64, zeros, ones, array, ndarray, int32, int64, asarray, any` (as needed per file) |
| `import attrs` | `from attrs import define, field, validators` (as needed per file) |
| `import numba` | `from numba import cuda, int32, float64, types` (as needed per file) |

## Edge Cases

### 1. TYPE_CHECKING Imports
Some files use `if TYPE_CHECKING:` blocks for static typing. These do NOT affect runtime or Numba compilation and can remain as whole-module imports.

### 2. Tableaus and Constants
Files like `generic_dirk_tableaus.py` define static data. These are not CUDAFactory subclasses but may be imported by them. Changes here are optional.

### 3. numpy.typing Usage
`NDArray` and other typing constructs should remain imported from `numpy.typing` as they are only used for type hints.

### 4. Dynamic Type Resolution
Some code uses `self.precision` which resolves to numpy dtype at runtime. The conversion must ensure the dtype types are available where these patterns occur.

## Dependencies

### External Dependencies
- `numpy` - Explicit imports of types and functions
- `numba` - Explicit imports of CUDA decorators and types
- `attrs` - Explicit imports of decorators and utilities

### Internal Dependencies
- `cubie._utils` - Already uses explicit imports
- `cubie.cuda_simsafe` - Already uses explicit imports
- `cubie.buffer_registry` - May need update if it uses whole-module imports

## Verification Steps

1. **Syntax Check:** All modified files must pass `flake8` without errors
2. **Import Check:** Verify no unused imports remain (use `ruff` or `autoflake`)
3. **Test Execution:** All existing tests must pass
4. **Compilation Check:** Verify device functions still compile correctly

## Implementation Order

Recommended order to minimize risk:

1. Start with isolated files that have fewer dependencies (e.g., `mean.py`, other metrics)
2. Move to mid-level components (algorithm steps, controllers)
3. Finish with core infrastructure (CUDAFactory, IVPLoop, BatchSolverKernel)

This order allows verification at each step before modifying critical components.
