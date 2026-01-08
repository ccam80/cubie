# Agent Plan: Compile Settings Cleanup

## Objective

Systematically analyze every CUDAFactory subclass and its config object to identify and remove compile settings fields that are never used during compilation. This prepares the codebase for efficient caching by minimizing spurious cache invalidations.

## Analysis Methodology

### Isolation Principle

Each CUDAFactory subclass MUST be analyzed in complete isolation:

1. Open only files related to the specific factory being analyzed
2. Do not reference or load files from other factories during analysis
3. Close all files before moving to the next factory
4. Document findings per factory before proceeding

This prevents confusion between different config objects and ensures accurate usage tracking.

### Usage Criteria

A config field is **USED** if:

1. **Direct access**: `self.compile_settings.field_name` appears in build() or helper methods
2. **Property derivation**: Field is accessed by a property that is itself used in build()
3. **Buffer registration**: Field is used in `register_buffers()` which provides allocators to build()
4. **Factory function parameter**: Field is passed to a factory function called in build()
5. **Nested compilation**: Field is passed to child CUDAFactory's config during nested build()
6. **Closure capture**: Field is captured in closure within build() method

A config field is **REDUNDANT** if:

1. Never accessed in build() method or any methods it calls
2. Not used by properties accessed in build() chain
3. Not used in buffer registration that affects build()
4. Not passed to any factory functions or child configs
5. Stored but never read during compilation

### Base Class Config Handling

For config classes shared by multiple CUDAFactory subclasses:

1. Analyze ALL subclasses that use the base config
2. Keep field if used by ANY subclass
3. Remove field only if unused across ALL subclasses
4. Document which subclass(es) use each field

Example: `BaseStepConfig` is used by all algorithm steps - analyze all algorithms before removing any field.

### Build Chain Tracing

For each factory, trace the complete build chain:

1. **Entry point**: `build()` method
2. **Helper methods**: Any methods called by build()
3. **Factory functions**: External factory functions (e.g., `save_state_factory()`)
4. **Nested builds**: Child CUDAFactory instances built within parent's build()
5. **Buffer allocators**: Allocators retrieved from buffer registry (created in `register_buffers()`)
6. **Closures**: Values captured in device function closures

### Property Tracing

Properties may access config fields indirectly:

1. Identify all properties of the factory class
2. For each property used in build(), trace which config fields it accesses
3. Keep all config fields accessed by used properties
4. Do NOT assume all properties are used - verify actual usage in build()

## Factory Analysis Checklist

For each CUDAFactory subclass:

### Step 1: Identify Factory and Config

- [ ] Locate CUDAFactory subclass file
- [ ] Identify config class used in `setup_compile_settings()`
- [ ] Locate config class definition (may be in separate file)
- [ ] List all fields in config class

### Step 2: Trace Build Method

- [ ] Read build() method completely
- [ ] Note all `self.compile_settings.field` accesses
- [ ] Note all property accesses on self
- [ ] Note all helper method calls
- [ ] Note all child factory instantiations

### Step 3: Trace Helper Methods

- [ ] For each helper method, trace config field usage
- [ ] Follow entire call chain from build()

### Step 4: Trace Buffer Registration

- [ ] Read `register_buffers()` method if present
- [ ] Identify which config fields are used for buffer registration
- [ ] Confirm these fields affect allocators used in build()

### Step 5: Trace Properties

- [ ] List all properties accessed in build() chain
- [ ] For each property, identify config fields it reads
- [ ] Mark those config fields as used

### Step 6: Trace Factory Functions

- [ ] List all factory functions called in build()
- [ ] For each factory call, identify which config fields are passed as arguments
- [ ] Mark those config fields as used

### Step 7: Create Usage Map

Create a table:

| Config Field | Used In | Used By | Keep/Remove | Notes |
|--------------|---------|---------|-------------|-------|
| field_name   | build() | Line 123| Keep        | Direct access |
| other_field  | N/A     | N/A     | Remove      | Never accessed |

### Step 8: Validate Removal Decisions

- [ ] Double-check all "Remove" decisions
- [ ] Verify field is not used in derived parameters
- [ ] Verify field is not needed for buffer registry
- [ ] Verify field is not part of shared base config used elsewhere

### Step 9: Document Before Removal

- [ ] Create documentation of removal decisions
- [ ] Note any edge cases or concerns

### Step 10: Test After Removal

- [ ] Run linters (flake8, ruff)
- [ ] Run relevant test file for this factory
- [ ] Fix any import errors or test failures
- [ ] Commit changes with clear message

## Factory Analysis Order

Analyze factories in dependency order (children before parents):

### Tier 1: Leaf Components (No Dependencies)

1. **Summary Metrics**
   - `SummaryMetric` → `MetricConfig`
   - Each metric is independent

2. **BaseODE System Data**
   - `BaseODE` → `ODEData`
   - System data container

### Tier 2: Low-Level Components

3. **Array Interpolator**
   - `ArrayInterpolator` → `ArrayInterpolatorConfig`
   - Used by loops but has no dependencies

4. **Matrix-Free Solvers**
   - `LinearSolver` → `LinearSolverConfig`
   - `NewtonKrylov` → `NewtonKrylovConfig`
   - Used by implicit algorithms

### Tier 3: Algorithm Steps

5. **Base Algorithm**
   - `BaseAlgorithmStep` → `BaseStepConfig`
   - Must analyze ALL subclasses before removing base fields

6. **Explicit Algorithms**
   - ERK variants
   - ExplicitEuler
   - Other explicit methods

7. **Implicit Algorithms**
   - DIRK variants
   - FIRK variants
   - RosenbrockW variants
   - BackwardsEuler
   - CrankNicolson
   - Other implicit methods

### Tier 4: Step Controllers

8. **Base Controller**
   - `BaseStepController` → `BaseStepControllerConfig`
   - Must analyze ALL subclasses before removing base fields

9. **Fixed Controller**
   - `FixedStepController` → `FixedStepControlConfig`

10. **Adaptive Controllers**
    - `AdaptiveIController`
    - `AdaptivePIController`
    - `AdaptivePIDController`
    - `GustafssonController`

### Tier 5: Output Handling

11. **Output Functions**
    - `OutputFunctions` → `OutputConfig`
    - Used by loops

### Tier 6: Integration Loops

12. **IVP Loop**
    - `IVPLoop` → `ODELoopConfig`
    - Uses algorithms, controllers, output functions

### Tier 7: High-Level Integrators

13. **Single Integrator Run Core**
    - `SingleIntegratorRunCore` → Config from settings
    - Composes loops, algorithms, controllers

### Tier 8: Batch Solving

14. **Batch Solver Kernel**
    - `BatchSolverKernel` → `BatchSolverConfig`
    - Uses SingleIntegratorRun

## Expected Interactions and Dependencies

### Component Relationships

```
BaseODE (ODEData)
    ↓
OutputFunctions (OutputConfig)
    ↓
IVPLoop (ODELoopConfig)
    ↓
SingleIntegratorRunCore
    ↓
BatchSolverKernel (BatchSolverConfig)

IVPLoop uses:
- Algorithm Step (BaseStepConfig + subclass configs)
- Step Controller (BaseStepControllerConfig + subclass configs)
- OutputFunctions
- ArrayInterpolator (ArrayInterpolatorConfig)

Implicit algorithms use:
- NewtonKrylov (NewtonKrylovConfig)
- LinearSolver (LinearSolverConfig)

OutputFunctions uses:
- SummaryMetric instances (MetricConfig)
```

### Buffer Registry Integration

Many config objects include `*_location` fields (e.g., `state_location`, `error_location`):

- These fields are used in `register_buffers()` method
- Buffer registry creates allocators based on these locations
- Allocators are retrieved in build() via `buffer_registry.get_allocator()`
- **These fields ARE used** - do not remove

### Data Structures Used

**Config Objects to Analyze:**

1. `ODEData` (BaseODE compile settings)
2. `ArrayInterpolatorConfig`
3. `BaseStepConfig` (base for all algorithms)
4. Algorithm-specific configs (ERKStepConfig, DIRKStepConfig, etc.)
5. `BaseStepControllerConfig` (base for all controllers)
6. Controller-specific configs
7. `LinearSolverConfig`
8. `NewtonKrylovConfig`
9. `MetricConfig` (for each summary metric)
10. `OutputConfig`
11. `ODELoopConfig`
12. Config used by `SingleIntegratorRunCore`
13. `BatchSolverConfig`

### Integration Points to Watch

**Parent-Child Relationships:**

- IVPLoop creates buffer allocations for child components (algorithm, controller)
- Parent's `register_buffers()` may register child's buffer needs
- Child configs may be updated from parent

**Config Parameter Passing:**

- OutputConfig derives sizes passed to loop config
- Loop config receives function references from output, algorithm, controller
- Algorithm configs receive solver configs for implicit methods

**Shared Constants:**

- ALL_*_PARAMETERS sets define recognized config keys
- These must be updated if config fields are removed
- Examples: `ALL_OUTPUT_FUNCTION_PARAMETERS`, `ALL_LOOP_SETTINGS`, `ALL_ALGORITHM_STEP_PARAMETERS`

## Edge Cases to Consider

### 1. Derived Values in Properties

If a config field is accessed ONLY in a property, and that property is used in build(), the field is USED.

Example:
```python
# In config
_value: float = field()

# In factory
@property
def derived(self):
    return self.compile_settings.value * 2

# In build()
def build(self):
    x = self.derived  # Uses 'value' indirectly
```

### 2. Underscore Fields with Properties

Attrs classes often store floats with underscore prefix and expose via property:

```python
@define
class Config:
    _dt0: float = field()
    
    @property
    def dt0(self):
        return self.precision(self._dt0)
```

If `dt0` property is used, `_dt0` field must be kept.

### 3. Function Reference Storage

Some configs store function references:

```python
@define
class Config:
    step_function: Callable = field()
```

If the function is called in build(), the field is used. Do NOT remove.

### 4. Buffer Sizes for Nested Allocations

Configs may store buffer sizes that are used for:
- Allocating arrays
- Iteration bounds
- Stride calculations

These are all USED - do not remove.

### 5. Conditional Compilation Flags

Boolean flags that control `if` statements in build():

```python
if config.save_state:
    # compile state saving logic
```

The flag is USED even if branch is not taken - it affects compilation.

### 6. Shared Memory Calculation

Fields used to calculate shared memory requirements are USED:

```python
shared_size = config.n_states + config.n_parameters
```

Both `n_states` and `n_parameters` are used.

## Removal Process

### For Each Field to Remove:

1. **Delete field definition** from config class
2. **Remove from ALL_*_PARAMETERS** set if present
3. **Update __init__ signature** of CUDAFactory if field was passed to config
4. **Update tests** that reference the field
5. **Update documentation** that mentions the field
6. **Run linters** to catch import/syntax errors
7. **Run tests** to verify functionality preserved

### Test Validation Requirements:

After each factory cleanup:

1. Run `flake8 .` for syntax errors
2. Run `pytest tests/path/to/factory/test_file.py` for functional tests
3. Fix any failures before proceeding to next factory
4. Commit changes with descriptive message

### Rollback Criteria:

If removal causes test failures that cannot be quickly fixed:

1. Revert the removal
2. Mark field as "Keep" with note: "Used in unexpected way - keep for safety"
3. Document the issue for future investigation

## Documentation Updates

### Files to Update:

- Config class docstrings (remove removed fields)
- Factory class __init__ docstrings (remove removed parameters)
- ALL_*_PARAMETERS sets (remove removed parameter names)
- Any tutorials or examples referencing removed fields

### Commit Message Format:

```
chore: remove unused field `field_name` from ConfigClass

Analysis showed `field_name` is never accessed in build() chain.
Verified through:
- Direct access check in build()
- Property usage tracing
- Buffer registration review
- Factory function parameter tracking

Tests pass: pytest tests/path/to/test.py
```

## Success Criteria

For the entire cleanup task:

1. All CUDAFactory subclasses analyzed
2. All redundant config fields removed
3. All tests passing (pytest suite green)
4. All linters passing (flake8, ruff)
5. Documentation updated for all changes
6. Commit history shows systematic, incremental progress

## Behavior Expectations

### Analysis Process:

1. **Be thorough**: Miss no config fields, miss no usage
2. **Be conservative**: When uncertain, keep the field
3. **Be isolated**: Analyze each factory independently
4. **Be systematic**: Follow the checklist for every factory
5. **Be documented**: Record all decisions

### Removal Process:

1. **Be surgical**: Remove only confirmed redundant fields
2. **Be tested**: Verify each removal doesn't break tests
3. **Be committed**: One factory per commit for traceability
4. **Be clear**: Commit messages explain what and why

### Communication:

1. **Ask for guidance** when field usage is ambiguous
2. **Report progress** after each factory completion
3. **Flag concerns** about shared base class fields
4. **Document surprises** when usage is found in unexpected places
