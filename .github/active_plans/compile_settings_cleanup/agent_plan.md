# Compile Settings Cleanup - Agent Plan

## Objective

Systematically identify and remove redundant variables from all compile_settings attrs classes used by CUDAFactory subclasses. A variable is redundant if it is NOT used anywhere in the build() method chain and does NOT derive parameters that are used in the build() chain.

## Architectural Context

### CUDAFactory Build Chain Pattern

Every CUDAFactory subclass follows this pattern:

1. **Initialization**: Receives parameters and creates a compile_settings attrs class
2. **setup_compile_settings()**: Stores the compile_settings object
3. **build()**: Compiles CUDA device functions using values from compile_settings
4. **Properties**: May expose compile_settings values through properties

The build chain includes:
- The build() method itself
- Any methods called by build()
- Any helper functions that receive compile_settings values
- Child factory build() methods if parent passes settings to children

### Cache Invalidation Mechanism

When `update_compile_settings()` is called:
1. CUDAFactory compares new settings to old settings using `__eq__`
2. If ANY field differs, cache is invalidated
3. Next property access triggers rebuild

**Problem**: Currently, changing ANY compile_settings field invalidates cache, even fields not used in build().

### Build Chain Tracing

To determine if a variable is redundant, trace from build() method:

```python
def build(self):
    config = self.compile_settings
    
    # Variable IS used if:
    x = config.some_var  # 1. Directly accessed
    y = helper(config.derived_var)  # 2. Passed to helper
    
    # Variable is REDUNDANT if:
    # - Never accessed in build() or its callees
    # - Only used to compute OTHER redundant variables
```

**Special Cases**:
- **Buffer location parameters** (`*_location`): Used via buffer_registry.register(), counts as used
- **Device function callbacks**: If stored and passed to compiled kernel, counts as used
- **Properties only**: If variable is only accessed via property but never in build(), it's redundant
- **Base class fields**: If base class has field but only subclass build() uses it, keep in base

## Component-by-Component Analysis Plan

### Phase 1: Core Infrastructure

#### 1.1 BaseODE and ODEData
**Location**: `src/cubie/odesystems/baseODE.py`, `src/cubie/odesystems/ODEData.py`

**Compile Settings**: ODEData attrs class

**Build Chain Analysis**:
- `BaseODE.build()` is abstract, actual implementation in subclasses
- SymbolicODE.build() uses: precision, values, system structure
- Check: Are all fields in ODEData used by SymbolicODE.build()?

**Expected Redundancies**:
- Metadata fields that aren't passed to codegen
- Cached values that aren't part of compilation

#### 1.2 OutputFunctions and OutputConfig
**Location**: `src/cubie/outputhandling/output_functions.py`, `src/cubie/outputhandling/output_config.py`

**Compile Settings**: OutputConfig attrs class

**Build Chain Analysis**:
- `OutputFunctions.build()` calls:
  - `save_state_factory()` 
  - `update_summary_factory()`
  - `save_summary_factory()`
- Each factory uses subset of OutputConfig

**Expected Redundancies**:
- Helper computed properties not used in factories
- Sizing metadata computed but not passed to device code
- sample_summaries_every if not used in metric compilation

**Cross-reference**: Check `ALL_OUTPUT_FUNCTION_PARAMETERS` set

### Phase 2: Integration Components

#### 2.1 IVPLoop and ODELoopConfig
**Location**: `src/cubie/integrators/loops/ode_loop.py`, `src/cubie/integrators/loops/ode_loop_config.py`

**Compile Settings**: ODELoopConfig attrs class

**Build Chain Analysis**:
- `IVPLoop.build()` uses:
  - Device function callbacks (save_state_fn, update_summaries_fn, etc.)
  - Timing parameters (save_every, summarise_every, sample_summaries_every)
  - Size parameters (n_states, n_parameters, etc.)
  - Buffer location parameters (via register_buffers())
  - compile_flags

**Expected Redundancies**:
- controller_local_len, algorithm_local_len (may be sizing only, not used in build)
- Derived properties that aren't passed to device code
- Boolean flags like save_last, save_regularly if not referenced in loop compilation

**Critical Check**: Verify timing parameters are actually captured in device function closures

**Cross-reference**: Check `ALL_LOOP_SETTINGS` set

#### 2.2 BaseAlgorithmStep and Subclasses
**Location**: `src/cubie/integrators/algorithms/`

**Files to Check**:
- `base_algorithm_step.py` - BaseStepConfig, ButcherTableau
- `ode_explicitstep.py` - ExplicitStepConfig
- `ode_implicitstep.py` - ImplicitStepConfig
- All concrete algorithm implementations (explicit_euler.py, backwards_euler.py, etc.)

**Build Chain Analysis**:
- Each algorithm has build_step() method
- Parameters used: evaluate_f, evaluate_observables, evaluate_driver_at_t, numba_precision, n, n_drivers
- Implicit methods also use: solver helpers, Newton/Krylov settings, tableau coefficients
- Buffer locations used via buffer_registry

**Expected Redundancies**:
- Settings used for controller defaults but not in build_step()
- Helper metadata not passed to device compilation

**Cross-reference**: Check `ALL_ALGORITHM_STEP_PARAMETERS` set

#### 2.3 BaseStepController and Subclasses
**Location**: `src/cubie/integrators/step_control/`

**Files to Check**:
- `base_step_controller.py` - BaseStepControllerConfig
- `fixed_step_controller.py` - FixedStepControlConfig
- `adaptive_I_controller.py`, `adaptive_PI_controller.py`, `adaptive_PID_controller.py`

**Build Chain Analysis**:
- Controller build() methods compile control device functions
- Parameters used: precision, n, controller-specific tuning (kp, ki, kd, safety, etc.)
- Buffer locations used via buffer_registry

**Expected Redundancies**:
- dt_min, dt_max, dt0 if not captured in controller device function
- For fixed controller, most parameters are redundant (just returns dt)

**Cross-reference**: Check `ALL_STEP_CONTROLLER_PARAMETERS` set

#### 2.4 SingleIntegratorRunCore
**Location**: `src/cubie/integrators/SingleIntegratorRunCore.py`

**Build Chain Analysis**:
- Composes IVPLoop with algorithm and controller
- May not have explicit compile_settings (uses child factories)
- Check if it stores any redundant configuration

**Expected Redundancies**:
- Coordination metadata not used in actual build()

### Phase 3: Solver Infrastructure

#### 3.1 NewtonKrylov and LinearSolver
**Location**: `src/cubie/integrators/matrix_free_solvers/`

**Build Chain Analysis**:
- These compile iterative solvers
- Parameters: tolerance, max_iters, precision, n
- Buffer locations via buffer_registry

**Expected Redundancies**:
- Reporting/diagnostic fields not used in solver compilation

#### 3.2 BatchSolverKernel
**Location**: `src/cubie/batchsolving/BatchSolverKernel.py`

**Build Chain Analysis**:
- Top-level kernel compilation
- Delegates to SingleIntegratorRun
- May store batch coordination metadata

**Expected Redundancies**:
- Batch sizing that doesn't affect kernel compilation
- Chunking parameters (runtime, not compile-time)

#### 3.3 ArrayInterpolator
**Location**: `src/cubie/integrators/array_interpolator.py`

**Build Chain Analysis**:
- Compiles driver interpolation functions
- Uses: precision, interpolation method, array sizes

**Expected Redundancies**:
- Metadata about driver sources not used in interpolation device code

### Phase 4: Output and Metrics

#### 4.1 SummaryMetric Subclasses
**Location**: `src/cubie/outputhandling/summarymetrics/`

**Files**: `metrics.py`, `mean.py`, `max.py`, `rms.py`, `peaks.py`

**Build Chain Analysis**:
- Each metric compiles update and save device functions
- Parameters: precision, metric-specific tuning

**Expected Redundancies**:
- Helper fields that don't affect metric calculation device code

## Deletion Rules

### Rule 1: Direct Usage
**Keep** if variable appears in build() or methods called by build():
```python
def build(self):
    x = self.compile_settings.my_var  # KEEP my_var
```

### Rule 2: Derived Usage
**Keep** if variable is used to compute another variable that IS used:
```python
def build(self):
    # If my_var is used to compute derived_var, and derived_var is used:
    derived_var = self.compile_settings.my_var * 2  # KEEP my_var
    use_in_device_code(derived_var)
```

**Delete** if variable is used to compute another variable that is NOT used:
```python
class Config:
    my_var: float  # DELETE
    
    @property
    def unused_derived(self):
        return self.my_var * 2  # This property is never called in build()
```

### Rule 3: Base Class Fields
**Keep** if ANY subclass uses it in their build():
```python
class BaseConfig:
    shared_field: int  # KEEP if any subclass build() uses it

class SubclassA(BaseConfig):
    def build(self):
        use(self.compile_settings.shared_field)  # Uses it
```

### Rule 4: Buffer Locations
**Keep** ALL `*_location` parameters - they're used via buffer_registry.register()

### Rule 5: Device Function Callbacks
**Keep** ALL device function references - they're compiled into closures

### Rule 6: Properties Only
**Delete** if variable is only exposed via property but never used in build():
```python
class Config:
    metadata: str  # DELETE
    
    @property
    def get_metadata(self):
        return self.metadata  # Property never called in build()
```

## Property Handling

When deleting a compile_settings variable that has a property wrapper:

### Strategy 1: Reroute to Child
If a child object has equivalent property, reroute parent property:
```python
# Before
@property
def my_value(self):
    return self.compile_settings.my_value

# After (if child has my_value property)
@property
def my_value(self):
    return self._child_object.my_value
```

### Strategy 2: Delete Property
If no child has equivalent and property isn't part of public API, delete it:
```python
# Delete both the compile_settings field AND the property
```

### Strategy 3: Mark as Deprecated
If property is part of public API but should be removed:
- Add deprecation warning
- Schedule for removal in future version
- Document in CHANGELOG

## ALL_*_PARAMETERS Sets Update

After deleting variables from compile_settings, update corresponding ALL_*_PARAMETERS sets:

**Files to update**:
- `src/cubie/integrators/loops/ode_loop.py` - ALL_LOOP_SETTINGS
- `src/cubie/outputhandling/output_functions.py` - ALL_OUTPUT_FUNCTION_PARAMETERS
- `src/cubie/integrators/algorithms/base_algorithm_step.py` - ALL_ALGORITHM_STEP_PARAMETERS
- `src/cubie/integrators/step_control/base_step_controller.py` - ALL_STEP_CONTROLLER_PARAMETERS

**Process**:
1. Identify deleted parameter names
2. Remove from corresponding ALL_*_PARAMETERS set
3. Check for any filtering code that references the set
4. Update filtering logic if needed

## Integration Points

### Buffer Registry
Variables used in buffer_registry.register() calls are NOT redundant:
```python
def register_buffers(self):
    buffer_registry.register(
        'buffer_name',
        self,
        self.compile_settings.buffer_size,  # KEEP buffer_size
        self.compile_settings.buffer_location,  # KEEP buffer_location
        precision=self.compile_settings.precision  # KEEP precision
    )
```

### Child Factory Delegation
Parent factories that create child factories may pass settings:
```python
def build(self):
    child = ChildFactory(
        precision=self.compile_settings.precision,  # KEEP precision
        n=self.compile_settings.n,  # KEEP n
    )
    return child.device_function
```

### Closure Capture
Variables captured in device function closures are NOT redundant:
```python
def build(self):
    threshold = self.compile_settings.threshold  # KEEP threshold
    
    @cuda.jit(device=True)
    def device_func(...):
        if value > threshold:  # Captured in closure
            ...
```

## Testing Strategy

### Pre-Deletion Checks
1. Run full test suite to establish baseline
2. Note any pre-existing failures (not our responsibility to fix)
3. Document test coverage for each affected component

### Post-Deletion Validation
1. Run full test suite - all previously passing tests must pass
2. Check for property access errors
3. Verify cache invalidation still works for kept variables
4. Verify cache NOT invalidated for deleted variables (manually test)

### Edge Cases to Test
- Nested compile_settings objects (e.g., compile_flags inside ODELoopConfig)
- Properties that reference multiple compile_settings fields
- Inheritance hierarchies where parent properties access child compile_settings
- Update paths that try to set deleted parameters (should fail gracefully or be silent depending on silent flag)

## Expected Outcomes

### Files Modified
- 15-30 config/settings attrs classes
- 10-15 CUDAFactory subclasses  
- 4-5 ALL_*_PARAMETERS sets
- Potentially test files if they explicitly set redundant parameters

### Variables Likely Deleted
- Unused buffer sizing helpers
- Metadata fields not passed to device code
- Properties that compute but don't use results
- Coordination fields used by parent but not in build()
- Settings duplicated between parent and child when child is authoritative

### Variables Definitely Kept
- All `*_location` buffer parameters
- All device function callbacks
- All precision-related fields (precision, numba_precision, simsafe_precision)
- All size parameters (n_states, n_parameters, n_observables, etc.)
- All parameters appearing in device function closures
- All controller tuning parameters (kp, ki, kd, safety, etc.)
- All solver tolerance/iteration parameters

## Risk Mitigation

### Risk: Deleting Subtly-Used Variable
**Mitigation**: 
- Manual review of each deletion
- Grep for all uses of variable name across codebase
- Check properties, update methods, and delegation chains

### Risk: Breaking Public API
**Mitigation**:
- Identify public-facing properties before deletion
- Deprecate rather than delete if part of documented API
- Check documentation for property references

### Risk: Test Failures
**Mitigation**:
- Run tests frequently during cleanup
- Fix test issues incrementally
- Separate test updates into distinct commits

### Risk: Build Chain Analysis Errors
**Mitigation**:
- Conservative approach - when in doubt, keep
- Peer review of deletion candidates
- Test cache invalidation behavior manually for borderline cases
