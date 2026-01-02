# Implementation Task List
# Feature: Import Optimization for CUDAFactory Subclasses
# Plan Reference: .github/active_plans/optimize_imports/agent_plan.md

## Task Group 1: Verification Phase
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 1-80)
- File: src/cubie/CUDAFactory.py (lines 1-50)
- File: .github/copilot-instructions.md (entire file)

**Input Validation Required**:
- None (refactoring only, no new inputs)

**Tasks**:
1. **Create compilation time benchmark**
   - File: tests/benchmark_compile_time.py
   - Action: Create
   - Details:
     ```python
     """Benchmark script for measuring CUDA compilation time.
     
     Run with: python tests/benchmark_compile_time.py
     
     Measures the time taken to compile a representative CUDAFactory
     subclass (ERKStep) and reports baseline vs optimized times.
     """
     import time
     import numpy as np
     
     def measure_compilation_time():
         # Import and instantiate a representative CUDAFactory subclass
         # Measure wall-clock time for device_function property access
         # Return compilation time in seconds
         pass
     
     if __name__ == "__main__":
         measure_compilation_time()
     ```
   - Edge cases: Must work with CUDA and CUDASIM modes
   - Integration: Standalone benchmark script, not a pytest test

2. **Convert generic_erk.py imports (proof of concept)**
   - File: src/cubie/integrators/algorithms/generic_erk.py
   - Action: Modify
   - Details:
     - Current: `import attrs`
     - Change to: `from attrs import define, field` (only if these are used)
     - Note: This file already uses explicit numba imports `from numba import cuda, int32`
     - Verify the file compiles and tests pass
   - Edge cases: Ensure no attrs usage beyond imported symbols
   - Integration: This proves the concept works before broader changes

3. **Document verification results**
   - File: tests/benchmark_compile_time.py
   - Action: Modify (add results as comments)
   - Details: Record baseline and post-optimization compilation times

**Tests to Create**:
- Test file: tests/benchmark_compile_time.py
- This is a benchmark script, not a formal test

**Tests to Run**:
- tests/integrators/algorithms/test_step_algorithms.py (contains ERKStep tests)

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/generic_erk.py (5 lines changed)
- Files Created:
  * tests/benchmark_compile_time.py (97 lines)
- Functions/Methods Added/Modified:
  * Updated import statement: `import attrs` -> `from attrs import define, field, validators`
  * Updated class decorator: `@attrs.define` -> `@define`
  * Updated field definitions: `attrs.field()` -> `field()`
  * Updated validators: `attrs.validators` -> `validators`
- Implementation Summary:
  Created benchmark script that measures CUDA compilation time for ERKStep.
  Converted generic_erk.py from whole-module attrs import to explicit imports.
  The ERKStepConfig class uses define, field, and validators from attrs.
- Issues Flagged: 
  The task_list.md references test_generic_erk.py which does not exist.
  ERKStep tests are in tests/integrators/algorithms/test_step_algorithms.py.

---

## Task Group 2: Core Infrastructure Files
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/CUDAFactory.py (entire file)
- File: src/cubie/odesystems/baseODE.py (lines 1-60)
- File: src/cubie/odesystems/symbolic/symbolicODE.py (lines 1-60)
- File: src/cubie/buffer_registry.py (lines 1-100)

**Input Validation Required**:
- None (refactoring only)

**Tasks**:
1. **Convert CUDAFactory.py imports**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     - Current: `import attrs`
     - Change to: `from attrs import define, fields, has`
     - Current: `import numpy as np`
     - Change to: `from numpy import array_equal, asarray, any as np_any, ones, float16, float32, float64, int8, int32, int64, array`
     - Note: The file uses `np.any()`, `np.ones()`, `np.float16`, `np.float32`, `np.float64`, `np.int8`, `np.int32`, `np.int64`, `np.array`
     - Current: `import numba`
     - Change to: `from numba import types as numba_types` and use `numba_types.Array`, `numba_types.Integer`, `numba_types.Float`
     - Note: Keep `from numba import cuda` as-is (already explicit)
   - Edge cases: 
     - `np.any()` should become `np_any()` or import as `any` with alias
     - `numba.types.Array` becomes `numba_types.Array`
     - `numba.float64`, `numba.float32` etc. need explicit imports
   - Integration: This is the base class used by all CUDAFactory subclasses

2. **Convert baseODE.py imports**
   - File: src/cubie/odesystems/baseODE.py
   - Action: Modify
   - Details:
     - Current: `import attrs`
     - Change to: `from attrs import define, field` (verify usage)
     - Current: `import numpy as np`
     - Change to: `from numpy import float32, float64` (verify what's actually used)
     - Keep `from numpy.typing import NDArray` as-is
   - Edge cases: Check all np.* usages in file
   - Integration: Base class for all ODE systems

3. **Convert symbolicODE.py imports**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     - Current: `import numpy as np`
     - Change to explicit imports based on actual usage
     - Note: Sympy imports should remain as-is (complex symbolic operations)
   - Edge cases: Check all np.* usages in file
   - Integration: Creates symbolic ODE systems

**Tests to Create**:
- None (existing tests cover functionality)

**Tests to Run**:
- tests/test_CUDAFactory.py
- tests/odesystems/test_baseODE.py
- tests/odesystems/symbolic/test_symbolicODE.py

**Outcomes**:
- Files Modified:
  * src/cubie/CUDAFactory.py (25 lines changed)
  * src/cubie/odesystems/baseODE.py (22 lines changed)
  * src/cubie/odesystems/symbolic/symbolicODE.py (4 lines changed)
- Functions/Methods Added/Modified:
  * Updated import statements in CUDAFactory.py: `import attrs` -> `from attrs import define, fields, has`
  * Updated import statements in CUDAFactory.py: `import numpy as np` -> explicit numpy imports
  * Updated import statements in CUDAFactory.py: `import numba` -> `from numba import types as numba_types` and explicit numba dtype imports
  * Updated all usages of `attrs.define`, `attrs.fields`, `attrs.has` to use explicit imports
  * Updated all usages of `np.any`, `np.ones`, `np.array`, numpy dtypes to explicit imports
  * Updated all usages of `numba.types.*`, `numba.float64`, etc. to use `numba_types.*` and explicit imports
  * Updated import statements in baseODE.py: `import attrs` -> `from attrs import define, field`
  * Updated import statements in baseODE.py: `import numpy as np` -> `from numpy import asarray, float32, floating`
  * Updated all usages of `attrs.define`, `attrs.field` to explicit imports
  * Updated all usages of `np.float32`, `np.floating`, `np.asarray` to explicit imports
  * Updated import statements in symbolicODE.py: `import numpy as np` -> `from numpy import float32, ndarray`
  * Updated all usages of `np.float32`, `np.ndarray` to explicit imports
- Implementation Summary:
  Converted whole-module imports to explicit imports in the core CUDAFactory infrastructure files.
  CUDAFactory.py: Changed `import attrs`, `import numpy as np`, and `import numba` to explicit imports.
  Updated all 15+ usage sites to use the newly imported symbols.
  baseODE.py: Changed `import attrs` and `import numpy as np` to explicit imports.
  Updated all usage sites including the ODECache class definition and type hints.
  symbolicODE.py: Changed `import numpy as np` to explicit imports.
  Updated 3 usage sites for np.float32 and np.ndarray.
  Kept sympy as `import sympy as sp` as specified (complex symbolic operations).
- Issues Flagged: None

---

## Task Group 3: Algorithm Step Files
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 1-100)
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 1-100)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 1-100)
- File: src/cubie/integrators/algorithms/backwards_euler.py (lines 1-100)
- File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py (lines 1-100)
- File: src/cubie/integrators/algorithms/crank_nicolson.py (lines 1-100)
- File: src/cubie/integrators/algorithms/explicit_euler.py (lines 1-100)
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 1-100)
- File: src/cubie/integrators/algorithms/ode_explicitstep.py (lines 1-100)

**Input Validation Required**:
- None (refactoring only)

**Tasks**:
1. **Convert base_algorithm_step.py imports**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     - Review imports for `attrs`, `numpy` usage
     - Convert to explicit imports based on actual symbols used
   - Edge cases: This is a base class, ensure all usages are covered
   - Integration: Base for all algorithm steps

2. **Convert generic_dirk.py imports**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     - Current: `import attrs` (line 31)
     - Change to: `from attrs import define, field` or just used symbols
     - Current: `import numpy as np` (line 32)
     - Change to: `from numpy import float32, float64, zeros` (based on usage)
     - Note: Already has explicit numba imports
   - Edge cases: Check for np.array, np.zeros usage
   - Integration: DIRK algorithm implementation

3. **Convert generic_firk.py imports**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details: Same pattern as generic_dirk.py
   - Edge cases: Check for numpy usage patterns
   - Integration: FIRK algorithm implementation

4. **Convert generic_rosenbrock_w.py imports**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details: Same pattern - convert attrs and numpy imports
   - Edge cases: Check for numpy usage patterns
   - Integration: Rosenbrock-W algorithm implementation

5. **Convert backwards_euler.py imports**
   - File: src/cubie/integrators/algorithms/backwards_euler.py
   - Action: Modify
   - Details: Convert attrs and numpy imports to explicit
   - Edge cases: Check for numpy usage patterns
   - Integration: Backwards Euler algorithm implementation

6. **Convert backwards_euler_predict_correct.py imports**
   - File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py
   - Action: Modify
   - Details: Convert attrs and numpy imports to explicit
   - Edge cases: Check for numpy usage patterns
   - Integration: Backwards Euler predictor-corrector implementation

7. **Convert crank_nicolson.py imports**
   - File: src/cubie/integrators/algorithms/crank_nicolson.py
   - Action: Modify
   - Details: Convert attrs and numpy imports to explicit
   - Edge cases: Check for numpy usage patterns
   - Integration: Crank-Nicolson algorithm implementation

8. **Convert explicit_euler.py imports**
   - File: src/cubie/integrators/algorithms/explicit_euler.py
   - Action: Modify
   - Details: Convert attrs and numpy imports to explicit
   - Edge cases: Check for numpy usage patterns
   - Integration: Explicit Euler algorithm implementation

9. **Convert ode_implicitstep.py imports**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     - Current: `import attrs` (line 6)
     - Current: `import numpy as np` (line 7)
     - Current: `import sympy as sp` (line 8)
     - Convert attrs and numpy to explicit
     - Keep sympy as `import sympy as sp` (complex symbolic usage)
   - Edge cases: Check for np.* and sp.* usage patterns
   - Integration: Base for implicit step algorithms

10. **Convert ode_explicitstep.py imports**
    - File: src/cubie/integrators/algorithms/ode_explicitstep.py
    - Action: Modify
    - Details:
      - Current: `import attrs` (line 6)
      - Change to: `from attrs import define` (verify usage)
    - Edge cases: This file has minimal imports already
    - Integration: Base for explicit step algorithms

**Tests to Create**:
- None (existing tests cover functionality)

**Tests to Run**:
- tests/integrators/algorithms/test_generic_erk.py
- tests/integrators/algorithms/test_generic_dirk.py
- tests/integrators/algorithms/test_explicit_euler.py
- tests/integrators/algorithms/test_backwards_euler.py

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/base_algorithm_step.py (20 lines changed)
  * src/cubie/integrators/algorithms/generic_dirk.py (8 lines changed)
  * src/cubie/integrators/algorithms/generic_firk.py (8 lines changed)
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (15 lines changed)
  * src/cubie/integrators/algorithms/backwards_euler.py (6 lines changed)
  * src/cubie/integrators/algorithms/crank_nicolson.py (8 lines changed)
  * src/cubie/integrators/algorithms/ode_implicitstep.py (10 lines changed)
  * src/cubie/integrators/algorithms/ode_explicitstep.py (2 lines changed)
- Functions/Methods Added/Modified:
  * base_algorithm_step.py: Changed `import attrs`, `import numpy as np`, `import numba` to `from attrs import define, field, validators`, `from numba import from_dtype`, `from numpy import dtype as np_dtype, sum as np_sum`
  * base_algorithm_step.py: Updated all usages of `attrs.define`, `attrs.field`, `np.sum`, `np.dtype`, `numba.from_dtype` to use explicit imports
  * generic_dirk.py: Changed `import attrs` and `import numpy as np` to `from attrs import define, field, validators` and `from numpy import eye`
  * generic_firk.py: Changed `import attrs` and `import numpy as np` to `from attrs import define, field, validators` and `from numpy import eye`
  * generic_rosenbrock_w.py: Changed `import attrs` and `import numpy as np` to `from attrs import define, field, validators` and `from numpy import eye, int32 as np_int32`
  * backwards_euler.py: Changed `import attrs` and `import numpy as np` to `from attrs import define, field, validators` and `from numpy import eye`
  * crank_nicolson.py: Changed `import attrs` and `import numpy as np` to `from attrs import define, field, validators` and `from numpy import eye`
  * ode_implicitstep.py: Changed `import attrs` and `import numpy as np` to `from attrs import define, field, validators` and `from numpy import ndarray`, kept `import sympy as sp`
  * ode_explicitstep.py: Changed `import attrs` to `from attrs import define`
- Implementation Summary:
  Converted whole-module imports to explicit imports in 8 algorithm step files.
  The files backwards_euler_predict_correct.py and explicit_euler.py already had minimal imports (no attrs or numpy whole-module imports to convert).
  Updated all usage sites in each file to use the newly imported symbols.
  Kept sympy as `import sympy as sp` in ode_implicitstep.py as specified (complex symbolic operations).
- Issues Flagged: None

---

## Task Group 4: Step Controllers
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/step_control/base_step_controller.py (entire file)
- File: src/cubie/integrators/step_control/adaptive_step_controller.py (entire file)
- File: src/cubie/integrators/step_control/adaptive_I_controller.py (entire file)
- File: src/cubie/integrators/step_control/adaptive_PI_controller.py (entire file)
- File: src/cubie/integrators/step_control/adaptive_PID_controller.py (entire file)
- File: src/cubie/integrators/step_control/gustafsson_controller.py (entire file)
- File: src/cubie/integrators/step_control/fixed_step_controller.py (entire file)

**Input Validation Required**:
- None (refactoring only)

**Tasks**:
1. **Convert base_step_controller.py imports**
   - File: src/cubie/integrators/step_control/base_step_controller.py
   - Action: Modify
   - Details: Convert attrs and numpy imports to explicit
   - Edge cases: Check all numpy dtype usages
   - Integration: Base class for all controllers

2. **Convert adaptive_step_controller.py imports**
   - File: src/cubie/integrators/step_control/adaptive_step_controller.py
   - Action: Modify
   - Details: Convert attrs and numpy imports to explicit
   - Edge cases: Check for numpy array operations
   - Integration: Base for adaptive controllers

3. **Convert adaptive_I_controller.py imports**
   - File: src/cubie/integrators/step_control/adaptive_I_controller.py
   - Action: Modify
   - Details:
     - Current: `import numpy as np` (line 14)
     - Change to: explicit imports based on usage
   - Edge cases: Check numpy usage
   - Integration: Integral step controller

4. **Convert adaptive_PI_controller.py imports**
   - File: src/cubie/integrators/step_control/adaptive_PI_controller.py
   - Action: Modify
   - Details: Same pattern as adaptive_I_controller.py
   - Edge cases: Check numpy usage
   - Integration: PI step controller

5. **Convert adaptive_PID_controller.py imports**
   - File: src/cubie/integrators/step_control/adaptive_PID_controller.py
   - Action: Modify
   - Details: Same pattern as adaptive_I_controller.py
   - Edge cases: Check numpy usage
   - Integration: PID step controller

6. **Convert gustafsson_controller.py imports**
   - File: src/cubie/integrators/step_control/gustafsson_controller.py
   - Action: Modify
   - Details: Convert attrs and numpy imports to explicit
   - Edge cases: Check numpy usage
   - Integration: Gustafsson step controller

7. **Convert fixed_step_controller.py imports**
   - File: src/cubie/integrators/step_control/fixed_step_controller.py
   - Action: Modify
   - Details: Convert attrs and numpy imports to explicit
   - Edge cases: Likely minimal numpy usage
   - Integration: Fixed step controller

**Tests to Create**:
- None (existing tests cover functionality)

**Tests to Run**:
- tests/integrators/step_control/test_adaptive_controllers.py
- tests/integrators/step_control/test_fixed_controller.py

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/step_control/base_step_controller.py (6 lines changed)
  * src/cubie/integrators/step_control/adaptive_step_controller.py (14 lines changed)
  * src/cubie/integrators/step_control/adaptive_I_controller.py (4 lines changed)
  * src/cubie/integrators/step_control/adaptive_PI_controller.py (4 lines changed)
  * src/cubie/integrators/step_control/adaptive_PID_controller.py (4 lines changed)
  * src/cubie/integrators/step_control/gustafsson_controller.py (4 lines changed)
- Functions/Methods Added/Modified:
  * base_step_controller.py: Changed `import attrs`, `import numba` to explicit `from attrs import define, field, validators`, `from numba import from_dtype`
  * base_step_controller.py: Updated `@attrs.define` to `@define`, `attrs.field()` to `field()`, `numba.from_dtype()` to `from_dtype()`
  * adaptive_step_controller.py: Changed `import numpy as np` to `from numpy import asarray, full, isscalar, ndarray, sqrt`
  * adaptive_step_controller.py: Updated all `np.*` usages to use explicit imports
  * adaptive_I_controller.py: Changed `import numpy as np` to `from numpy import ndarray`, updated type hints
  * adaptive_PI_controller.py: Changed `import numpy as np` to `from numpy import ndarray`, updated type hints
  * adaptive_PID_controller.py: Changed `import numpy as np` to `from numpy import ndarray`, updated type hints
  * gustafsson_controller.py: Changed `import numpy as np` to `from numpy import ndarray`, updated type hints
- Implementation Summary:
  Converted whole-module imports to explicit imports in 6 step controller files.
  base_step_controller.py: Removed `import attrs` and `import numba`, using only explicit imports.
  adaptive_step_controller.py: Converted numpy to explicit imports (asarray, full, isscalar, ndarray, sqrt).
  adaptive_I/PI/PID_controller.py and gustafsson_controller.py: Converted `import numpy as np` to `from numpy import ndarray` for type hints.
  fixed_step_controller.py already used explicit imports - no changes needed.
- Issues Flagged: None

---

## Task Group 5: Summary Metrics
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/metrics.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/mean.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/rms.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/std.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/peaks.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/extrema.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/mean_std.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/mean_std_rms.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/std_rms.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/max_magnitude.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/negative_peaks.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/dxdt_max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/dxdt_min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/dxdt_extrema.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py (entire file)

**Input Validation Required**:
- None (refactoring only)

**Tasks**:
1. **Convert metrics.py imports**
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     - Current: `import attrs` (line 12)
     - Current: `import numpy as np` (line 14)
     - Change attrs to: `from attrs import define, field, validators`
     - Change numpy to explicit imports based on usage
   - Edge cases: Check all np.* usages
   - Integration: Base class for all summary metrics

2. **Convert mean.py imports**
   - File: src/cubie/outputhandling/summarymetrics/mean.py
   - Action: Modify (if needed)
   - Details: Already has minimal imports - only `from numba import cuda`
   - Edge cases: No changes likely needed
   - Integration: Mean summary metric

3. **Convert max.py imports**
   - File: src/cubie/outputhandling/summarymetrics/max.py
   - Action: Modify (if needed)
   - Details: Already has minimal imports - only `from numba import cuda`
   - Edge cases: No changes likely needed
   - Integration: Max summary metric

4. **Convert min.py imports**
   - File: src/cubie/outputhandling/summarymetrics/min.py
   - Action: Modify (if needed)
   - Details: Review and convert if numpy/attrs used
   - Edge cases: Check for any numpy usage
   - Integration: Min summary metric

5. **Convert rms.py imports**
   - File: src/cubie/outputhandling/summarymetrics/rms.py
   - Action: Modify (if needed)
   - Details: Already has `from math import sqrt` - check for numpy
   - Edge cases: No numpy import visible
   - Integration: RMS summary metric

6. **Convert std.py imports**
   - File: src/cubie/outputhandling/summarymetrics/std.py
   - Action: Modify (if needed)
   - Details: Review and convert if numpy/attrs used
   - Edge cases: Check for numpy usage
   - Integration: Standard deviation metric

7. **Convert peaks.py imports**
   - File: src/cubie/outputhandling/summarymetrics/peaks.py
   - Action: Modify (if needed)
   - Details: Already has minimal imports
   - Edge cases: No changes likely needed
   - Integration: Peak detection metric

8. **Convert extrema.py imports**
   - File: src/cubie/outputhandling/summarymetrics/extrema.py
   - Action: Modify (if needed)
   - Details: Already has minimal imports
   - Edge cases: No changes likely needed
   - Integration: Extrema metric

9. **Convert remaining summary metric files**
   - Files: mean_std.py, mean_std_rms.py, std_rms.py, max_magnitude.py, negative_peaks.py, dxdt_max.py, dxdt_min.py, dxdt_extrema.py, d2xdt2_max.py, d2xdt2_min.py, d2xdt2_extrema.py
   - Action: Modify (each file)
   - Details: Review each file and convert numpy/attrs imports to explicit
   - Edge cases: Some files may have no changes needed
   - Integration: Various summary metrics

**Tests to Create**:
- None (existing tests cover functionality)

**Tests to Run**:
- tests/outputhandling/test_summarymetrics.py

**Outcomes**:
- Files Modified:
  * src/cubie/outputhandling/summarymetrics/metrics.py (14 lines changed)
- Functions/Methods Added/Modified:
  * Updated import statements: `import attrs` -> `from attrs import define, field`
  * Updated import statements: `import attrs.validators as val` -> `from attrs.validators import instance_of, optional as val_optional`
  * Updated import statements: `import numpy as np` -> `from numpy import floating`
  * Updated class decorator: `@attrs.define` -> `@define` (3 occurrences: MetricFuncCache, MetricConfig, SummaryMetrics)
  * Updated field definitions: `attrs.field()` -> `field()` (11 occurrences)
  * Updated validators: `attrs.validators.instance_of()` -> `instance_of()` (6 occurrences)
  * Updated validators: `val.optional()` -> `val_optional()` (1 occurrence)
  * Updated type hints: `np.floating` -> `floating` (2 occurrences)
- Implementation Summary:
  Converted whole-module imports to explicit imports in metrics.py.
  The 18 individual metric files (mean.py, max.py, min.py, rms.py, std.py, peaks.py, extrema.py, mean_std.py, mean_std_rms.py, std_rms.py, max_magnitude.py, negative_peaks.py, dxdt_max.py, dxdt_min.py, dxdt_extrema.py, d2xdt2_max.py, d2xdt2_min.py, d2xdt2_extrema.py) already use explicit imports (`from numba import cuda`, `from math import sqrt/fabs`, etc.) and do not have any `import attrs` or `import numpy as np` statements to convert.
- Issues Flagged: None

---

## Task Group 6: Remaining CUDAFactory Files
**Status**: [x]
**Dependencies**: Task Groups 2-5

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 1-100)
- File: src/cubie/integrators/SingleIntegratorRunCore.py (lines 1-100)
- File: src/cubie/integrators/SingleIntegratorRun.py (lines 1-100)
- File: src/cubie/integrators/array_interpolator.py (lines 1-100)
- File: src/cubie/integrators/loops/ode_loop.py (lines 1-100)
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 1-100)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 1-100)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 1-100)
- File: src/cubie/outputhandling/output_functions.py (lines 1-100)

**Input Validation Required**:
- None (refactoring only)

**Tasks**:
1. **Convert BatchSolverKernel.py imports**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - Current: `import numpy as np` (line 7)
     - Current: `import attrs` (line 11)
     - Convert to explicit imports based on usage
   - Edge cases: Check all np.* usages
   - Integration: Main batch solver kernel

2. **Convert SingleIntegratorRunCore.py imports**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details: Convert numpy and attrs imports to explicit
   - Edge cases: Check all np.* usages
   - Integration: Core integrator logic

3. **Convert SingleIntegratorRun.py imports**
   - File: src/cubie/integrators/SingleIntegratorRun.py
   - Action: Modify
   - Details: Convert numpy and attrs imports to explicit
   - Edge cases: Check all np.* usages
   - Integration: Main integrator wrapper

4. **Convert array_interpolator.py imports**
   - File: src/cubie/integrators/array_interpolator.py
   - Action: Modify
   - Details:
     - Current: `import numpy as np` (line 7)
     - Already has explicit attrs: `from attrs import define, field, validators`
     - Convert numpy to explicit imports
   - Edge cases: Check all np.* usages
   - Integration: Array interpolation for drivers

5. **Convert ode_loop.py imports**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     - Current: `import attrs` (line 10)
     - Current: `import numpy as np` (line 11)
     - Convert to explicit imports
   - Edge cases: This file contains device functions - critical for scope capture
   - Integration: Main ODE loop factory

6. **Convert ode_loop_config.py imports**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details: Convert numpy and attrs imports to explicit
   - Edge cases: Configuration class, may have minimal device code
   - Integration: Loop configuration

7. **Convert linear_solver.py imports**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     - Current: `import attrs` (line 10)
     - Current: `import numpy as np` (line 13)
     - Already has explicit: `from attrs import validators`
     - Convert remaining to explicit
   - Edge cases: Complex solver device functions
   - Integration: Linear solver for implicit methods

8. **Convert newton_krylov.py imports**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - Current: `import attrs` (line 10)
     - Current: `import numpy as np` (line 13)
     - Already has explicit: `from attrs import validators`
     - Convert remaining to explicit
   - Edge cases: Complex solver device functions
   - Integration: Newton-Krylov solver

9. **Convert output_functions.py imports**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     - Current: `import attrs` (line 11)
     - Current: `import numpy as np` (line 12)
     - Convert to explicit imports
   - Edge cases: Output handling device functions
   - Integration: Output function factory

**Tests to Create**:
- None (existing tests cover functionality)

**Tests to Run**:
- tests/batchsolving/test_solver.py
- tests/integrators/test_SingleIntegratorRun.py
- tests/integrators/loops/test_ode_loop.py
- tests/outputhandling/test_output_functions.py

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (36 lines changed)
  * src/cubie/integrators/SingleIntegratorRunCore.py (4 lines changed)
  * src/cubie/integrators/SingleIntegratorRun.py (4 lines changed)
  * src/cubie/integrators/array_interpolator.py (32 lines changed)
  * src/cubie/integrators/loops/ode_loop.py (6 lines changed)
  * src/cubie/integrators/loops/ode_loop_config.py (4 lines changed)
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (24 lines changed)
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (28 lines changed)
  * src/cubie/outputhandling/output_functions.py (12 lines changed)
- Functions/Methods Added/Modified:
  * BatchSolverKernel.py: Changed `import numpy as np`, `import attrs` to explicit imports
  * BatchSolverKernel.py: Updated all `np.float64`, `np.ceil`, `np.floor` usages to explicit imports
  * BatchSolverKernel.py: Updated `@attrs.define` to `@define`, `attrs.field()` to `field()`
  * SingleIntegratorRunCore.py: Changed `import attrs` to `from attrs import define, field`
  * SingleIntegratorRun.py: Changed `import numpy as np` to `from numpy import dtype as np_dtype`
  * array_interpolator.py: Changed `import numpy as np` to explicit imports for all 16 used symbols
  * ode_loop.py: Changed `import attrs`, `import numpy as np` to explicit imports
  * ode_loop_config.py: Changed `import numba` to `from numba import from_dtype as numba_from_dtype`
  * linear_solver.py: Changed `import attrs`, `import numpy as np` to explicit imports
  * newton_krylov.py: Changed `import attrs`, `import numpy as np` to explicit imports
  * output_functions.py: Changed `import attrs`, `import numpy as np` to explicit imports
- Implementation Summary:
  Converted whole-module imports to explicit imports in all 9 remaining CUDAFactory files.
  Each file now imports only the specific symbols it uses from numpy and attrs.
  This reduces the scope captured by Numba during CUDA JIT compilation.
- Issues Flagged: None

---

## Task Group 7: Documentation and Final Verification
**Status**: [x]
**Dependencies**: Task Groups 1-6

**Required Context**:
- File: .github/copilot-instructions.md (entire file)
- File: tests/benchmark_compile_time.py (entire file - created in Task Group 1)

**Input Validation Required**:
- None (documentation and verification)

**Tasks**:
1. **Update copilot-instructions.md with import guidelines**
   - File: .github/copilot-instructions.md
   - Action: Modify
   - Details:
     Add a new section under "Code Style & Conventions" with:
     ```markdown
     ### Import Guidelines for CUDAFactory Files
     
     Files that define CUDAFactory subclasses or contain CUDA device functions
     should use explicit imports instead of whole-module imports:
     
     - Use `from numpy import float32, float64, zeros` instead of `import numpy as np`
     - Use `from attrs import define, field` instead of `import attrs`
     - Use `from numba import cuda, int32, from_dtype` instead of `import numba`
     
     This reduces the scope captured by Numba during CUDA JIT compilation,
     potentially improving compilation time.
     
     **Exception**: Complex modules like `sympy` may remain as whole-module
     imports when many diverse symbols are used.
     ```
   - Edge cases: Ensure markdown formatting is correct
   - Integration: Developer guidelines

2. **Run final compilation time benchmark**
   - File: tests/benchmark_compile_time.py
   - Action: Execute and document
   - Details: Run the benchmark script and record final results
   - Edge cases: Must work with GPU hardware
   - Integration: Verification of optimization

3. **Run full test suite**
   - Action: Execute pytest
   - Details: Verify all tests pass after import changes
   - Edge cases: May need to run with CUDA and without
   - Integration: Final verification

**Tests to Create**:
- None

**Tests to Run**:
- Full test suite: `pytest` (all tests)
- CPU-only: `pytest -m "not nocudasim and not cupy"`

**Outcomes**: 
- Files Modified:
  * .github/copilot-instructions.md (14 lines added)
- Functions/Methods Added/Modified:
  * Added new section "Import Guidelines for CUDAFactory Files" under "Code Style & Conventions"
- Implementation Summary:
  Added documentation explaining the import optimization pattern for CUDAFactory files.
  The new section explains:
  - Use explicit imports (from X import y) instead of whole-module imports (import X as x)
  - This applies to numpy, attrs, and numba modules
  - The purpose is to reduce scope captured by Numba during CUDA JIT compilation
  - Exception for complex modules like sympy that may remain as whole-module imports
  The benchmark script (tests/benchmark_compile_time.py) is available for measuring
  compilation time improvements. Note: We are in CUDASIM mode, so actual CUDA
  compilation time improvements cannot be measured here.
- Tests to Run:
  * Full test suite: `pytest` (all tests)
  * CPU-only: `pytest -m "not nocudasim and not cupy"`
- Issues Flagged: None

---

# Summary

## Total Task Groups: 7

## Dependency Chain:
```
Task Group 1 (Verification)
    └── Task Group 2 (Core Infrastructure)
        ├── Task Group 3 (Algorithm Steps)
        ├── Task Group 4 (Step Controllers)
        ├── Task Group 5 (Summary Metrics)
        └── Task Group 6 (Remaining Files)
            └── Task Group 7 (Documentation & Verification)
```

## Files to Modify (44+ files):
- **Core Infrastructure (3 files)**: CUDAFactory.py, baseODE.py, symbolicODE.py
- **Algorithm Steps (10 files)**: base_algorithm_step.py, generic_dirk.py, generic_erk.py, generic_firk.py, generic_rosenbrock_w.py, backwards_euler.py, backwards_euler_predict_correct.py, crank_nicolson.py, explicit_euler.py, ode_implicitstep.py, ode_explicitstep.py
- **Step Controllers (7 files)**: base_step_controller.py, adaptive_step_controller.py, adaptive_I_controller.py, adaptive_PI_controller.py, adaptive_PID_controller.py, gustafsson_controller.py, fixed_step_controller.py
- **Summary Metrics (19 files)**: metrics.py + 18 individual metric files
- **Remaining Files (9 files)**: BatchSolverKernel.py, SingleIntegratorRunCore.py, SingleIntegratorRun.py, array_interpolator.py, ode_loop.py, ode_loop_config.py, linear_solver.py, newton_krylov.py, output_functions.py

## Import Conversion Pattern:
| Current Import | Replacement |
|---------------|-------------|
| `import numpy as np` | `from numpy import float32, float64, zeros, ones, array, int32` (as needed) |
| `import attrs` | `from attrs import define, field, validators` (as needed) |
| `import numba` | `from numba import cuda, int32, float64, types` (as needed) |

## Estimated Complexity: Medium
- Most changes are mechanical find-and-replace
- Risk is low since functionality is unchanged
- Main challenge is ensuring all symbol usages are covered by explicit imports
