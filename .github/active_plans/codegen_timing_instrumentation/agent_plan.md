# Agent Implementation Plan: Codegen and Parsing Timing Instrumentation

## Component Overview

This plan details the implementation of timing instrumentation for symbolic ODE parsing and code generation functions in the CuBIE project. The implementation integrates with the existing `TimeLogger` infrastructure to provide performance profiling capabilities for the compilation pipeline.

## Files to Modify

### 1. `/home/runner/work/cubie/cubie/src/cubie/odesystems/symbolic/symbolicODE.py`

**Purpose**: Add timing instrumentation to parsing and solver helper generation

**Changes Required:**

#### A. Import TimeLogger
Add import at module level:
```python
from cubie.time_logger import _default_logger
```

#### B. Instrument `SymbolicODE.create()` classmethod
- Register the parsing event before timing starts (conditional on first call)
- Start timing at the beginning of the method (after driver processing)
- Stop timing just before the return statement

**Event Details:**
- Event name: `"symbolic_ode_parsing"`
- Category: `"codegen"`
- Description: `"Codegen time for symbolic ODE parsing: "`

**Timing Span:**
- Start: After driver dictionary processing, before calling `parse_input()`
- Stop: After `cls()` constructor call, before return

#### C. Instrument `get_solver_helper()` method
- Register helper-specific event on first call for each `func_type`
- Start timing at method entry
- Stop timing just before the return statement

**Event Details:**
- Event name: `f"solver_helper_{func_type}"` (e.g., `"solver_helper_linear_operator"`)
- Category: `"codegen"`
- Description: `f"Codegen time for solver helper {func_type}: "`

**Timing Span:**
- Start: After parameter validation, at beginning of helper generation logic
- Stop: Before return of the compiled device function

**Implementation Notes:**
- Use a module-level set to track which helper events have been registered
- Check set membership before calling `_register_event()` to avoid duplicate registrations
- The set should be defined at module level: `_registered_helper_events = set()`

### 2. `/home/runner/work/cubie/cubie/src/cubie/odesystems/symbolic/codegen/dxdt.py`

**Purpose**: Add timing to dxdt and observables code generation functions

**Changes Required:**

#### A. Import TimeLogger
```python
from cubie.time_logger import _default_logger
```

#### B. Register events at module level
Add registration calls after imports (executed once on module import):
```python
_default_logger._register_event(
    "codegen_generate_dxdt_fac_code",
    "codegen",
    "Codegen time for generate_dxdt_fac_code: "
)
_default_logger._register_event(
    "codegen_generate_observables_fac_code",
    "codegen",
    "Codegen time for generate_observables_fac_code: "
)
```

#### C. Instrument `generate_dxdt_fac_code()` function
- Start timing at function entry
- Stop timing before return statement

**Timing Boundaries:**
- Start: First line of function body
- Stop: Before `return` statement

#### D. Instrument `generate_observables_fac_code()` function
- Start timing at function entry
- Stop timing before return statement

**Timing Boundaries:**
- Start: First line of function body
- Stop: Before `return` statement

### 3. `/home/runner/work/cubie/cubie/src/cubie/odesystems/symbolic/codegen/linear_operators.py`

**Purpose**: Add timing to linear operator and Jacobian code generation functions

**Changes Required:**

#### A. Import TimeLogger
```python
from cubie.time_logger import _default_logger
```

#### B. Register events at module level
Register all events for functions in this module:
- `"codegen_generate_operator_apply_code"`
- `"codegen_generate_cached_operator_apply_code"`
- `"codegen_generate_prepare_jac_code"`
- `"codegen_generate_cached_jvp_code"`

Each with category `"codegen"` and description `"Codegen time for {function_name}: "`

#### C. Instrument functions
Add timing calls to:
- `generate_operator_apply_code()`
- `generate_cached_operator_apply_code()`
- `generate_prepare_jac_code()`
- `generate_cached_jvp_code()`

**Timing Pattern:**
```python
def generate_X_code(...):
    _default_logger.start_event("codegen_generate_X_code")
    # ... existing function body ...
    _default_logger.stop_event("codegen_generate_X_code")
    return result
```

### 4. `/home/runner/work/cubie/cubie/src/cubie/odesystems/symbolic/codegen/preconditioners.py`

**Purpose**: Add timing to preconditioner code generation functions

**Changes Required:**

#### A. Import TimeLogger
```python
from cubie.time_logger import _default_logger
```

#### B. Register events at module level
Register events for:
- `"codegen_generate_neumann_preconditioner_code"`
- `"codegen_generate_neumann_preconditioner_cached_code"`
- `"codegen_generate_n_stage_neumann_preconditioner_code"`

#### C. Instrument functions
Add timing to:
- `generate_neumann_preconditioner_code()`
- `generate_neumann_preconditioner_cached_code()`
- `generate_n_stage_neumann_preconditioner_code()`

### 5. `/home/runner/work/cubie/cubie/src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py`

**Purpose**: Add timing to residual code generation functions

**Changes Required:**

#### A. Import TimeLogger
```python
from cubie.time_logger import _default_logger
```

#### B. Register events at module level
Register events for:
- `"codegen_generate_stage_residual_code"`
- `"codegen_generate_n_stage_residual_code"`
- `"codegen_generate_n_stage_linear_operator_code"`

**Note**: `generate_n_stage_linear_operator_code` is mentioned in the task but may be in `linear_operators.py`. Verify location and register in correct module.

#### C. Instrument functions
Add timing to:
- `generate_stage_residual_code()`
- `generate_n_stage_residual_code()`

### 6. `/home/runner/work/cubie/cubie/src/cubie/odesystems/symbolic/codegen/time_derivative.py`

**Purpose**: Add timing to time derivative code generation

**Changes Required:**

#### A. Import TimeLogger
```python
from cubie.time_logger import _default_logger
```

#### B. Register event at module level
Register:
- `"codegen_generate_time_derivative_fac_code"`

#### C. Instrument function
Add timing to:
- `generate_time_derivative_fac_code()`

## Implementation Pattern

### Standard Codegen Function Instrumentation

For all `generate_*_code()` functions:

```python
# At module level after imports
from cubie.time_logger import _default_logger

_default_logger._register_event(
    "codegen_generate_X_code",
    "codegen",
    "Codegen time for generate_X_code: "
)

# In function
def generate_X_code(...):
    _default_logger.start_event("codegen_generate_X_code")
    
    # ... existing function implementation ...
    
    _default_logger.stop_event("codegen_generate_X_code")
    return result
```

### SymbolicODE.create() Instrumentation

```python
# At module level
from cubie.time_logger import _default_logger

# One-time registration flag at module level
_parsing_event_registered = False

@classmethod
def create(cls, dxdt, states, ...):
    global _parsing_event_registered
    if not _parsing_event_registered:
        _default_logger._register_event(
            "symbolic_ode_parsing",
            "codegen",
            "Codegen time for symbolic ODE parsing: "
        )
        _parsing_event_registered = True
    
    # ... driver processing code ...
    
    _default_logger.start_event("symbolic_ode_parsing")
    
    sys_components = parse_input(...)
    symbolic_ode = cls(equations=..., ...)
    
    _default_logger.stop_event("symbolic_ode_parsing")
    return symbolic_ode
```

### get_solver_helper() Instrumentation

```python
# At module level
_registered_helper_events = set()

def get_solver_helper(self, func_type, ...):
    # Register event if not already registered
    event_name = f"solver_helper_{func_type}"
    if event_name not in _registered_helper_events:
        _default_logger._register_event(
            event_name,
            "codegen",
            f"Codegen time for solver helper {func_type}: "
        )
        _registered_helper_events.add(event_name)
    
    # Start timing
    _default_logger.start_event(event_name)
    
    # ... existing helper generation logic ...
    
    # Stop timing before return
    _default_logger.stop_event(event_name)
    return func
```

## Dependencies and Integrations

### TimeLogger Integration
- Uses global `_default_logger` instance from `cubie.time_logger`
- No changes required to `TimeLogger` class itself
- Events automatically respect verbosity settings (no-op when `verbosity=None`)

### Event Registration
- Module-level registration for codegen functions (executed once on import)
- Conditional registration for parsing (executed once per process)
- Lazy registration for solver helpers (executed once per helper type)

### No Breaking Changes
- All timing calls are additions only
- No modification to function signatures or return values
- No changes to core algorithmic logic
- Maintains backward compatibility

## Edge Cases

### Multiple Calls to Same Function
- Codegen functions may be called multiple times for different systems
- Timing events accumulate; use `get_aggregate_durations()` for totals
- Each start/stop pair is tracked individually

### Exception Handling
- If exception occurs between start and stop, event remains "active"
- TimeLogger tracks active starts in `_active_starts` dict
- Caller is responsible for error handling; timing is diagnostic only
- This is consistent with existing timing behavior in CUDAFactory

### Verbosity None
- When `verbosity=None`, all timing calls are no-ops
- Registration still occurs (minimal overhead)
- Zero performance impact during actual timing calls

### Thread Safety
- TimeLogger is not thread-safe by design (single-threaded use assumed)
- CuBIE compilation is single-threaded, so this is acceptable
- Future multi-threaded compilation would require TimeLogger enhancement

## Testing Considerations

### Functional Testing
- Verify events are registered correctly
- Verify timing data is captured when verbosity is enabled
- Verify no-op behavior when verbosity is None
- Verify event names match expected patterns

### Integration Testing
- Test full workflow: parse → codegen → helper generation
- Verify aggregate durations are computed correctly
- Verify timing events appear in chronological order

### Performance Testing
- Measure overhead of timing calls (should be negligible)
- Verify no performance regression when `verbosity=None`

## Expected Behavior

### Event Names Generated
- `"symbolic_ode_parsing"`
- `"codegen_generate_dxdt_fac_code"`
- `"codegen_generate_observables_fac_code"`
- `"codegen_generate_operator_apply_code"`
- `"codegen_generate_cached_operator_apply_code"`
- `"codegen_generate_prepare_jac_code"`
- `"codegen_generate_cached_jvp_code"`
- `"codegen_generate_neumann_preconditioner_code"`
- `"codegen_generate_neumann_preconditioner_cached_code"`
- `"codegen_generate_stage_residual_code"`
- `"codegen_generate_time_derivative_fac_code"`
- `"codegen_generate_n_stage_residual_code"`
- `"codegen_generate_n_stage_linear_operator_code"`
- `"codegen_generate_n_stage_neumann_preconditioner_code"`
- `"solver_helper_{func_type}"` for each helper type called

### Timing Output Example (verbosity='verbose')
```
symbolic_ode_parsing: 0.023s
codegen_generate_dxdt_fac_code: 0.008s
codegen_generate_observables_fac_code: 0.003s
solver_helper_linear_operator: 0.012s
codegen_generate_operator_apply_code: 0.011s
```

### Aggregate Duration Query
```python
# Get all codegen timings
durations = _default_logger.get_aggregate_durations(category='codegen')
# Returns: {'symbolic_ode_parsing': 0.023, 'codegen_generate_dxdt_fac_code': 0.008, ...}
```

## Implementation Order

1. **First**: Add timing to `symbolicODE.py` (parsing and solver helpers)
   - Tests can verify parsing timing immediately
   - Provides framework for testing other components

2. **Second**: Add timing to `dxdt.py` (most commonly used codegen)
   - Validates the module-level registration pattern
   - Provides immediate value for basic ODE compilation

3. **Third**: Add timing to `linear_operators.py`
   - Covers majority of implicit solver helper functions
   - More complex due to multiple functions

4. **Fourth**: Add timing to remaining codegen modules
   - `preconditioners.py`
   - `nonlinear_residuals.py`
   - `time_derivative.py`
   - Lower frequency use, but important for completeness

This order ensures early validation of the implementation pattern while delivering incremental value.
