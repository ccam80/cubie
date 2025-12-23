# Agent Implementation Plan: Default Parameter Rationalization

## Architectural Overview

This plan implements **Option B: Helper Function with Required/Optional Split** to rationalize default parameter handling across cubie's configuration cascade.

### Core Principle

**Single Source of Truth**: All optional parameter defaults live in attrs config classes. Init functions accept required parameters explicitly plus `**kwargs` for optional overrides. A helper function constructs the config object, eliminating verbose `if param is not None` checks.

## Component Architecture

### Current Flow

```
User kwargs → merge_kwargs_into_settings → settings dict → 
get_algorithm_step → split_applicable_settings → 
Algorithm.__init__(15 params) → manually build config dict → 
ConfigClass(**config_dict)
```

### New Flow

```
User kwargs → merge_kwargs_into_settings → settings dict → 
get_algorithm_step → split_applicable_settings →
Algorithm.__init__(precision, n, **kwargs) → build_config →
ConfigClass(with defaults + overrides)
```

## Data Structures

### Config Class Pattern

All config classes follow this pattern:

```python
@attrs.define
class AlgorithmConfig(BaseStepConfig):
    """Configuration for Algorithm.
    
    All optional parameters have defaults defined here.
    This is the single source of truth for parameter defaults.
    """
    
    # Required parameters (no default)
    precision: PrecisionDType = attrs.field(
        converter=precision_converter,
        validator=precision_validator,
    )
    n: int = attrs.field(validator=getype_validator(int, 1))
    
    # Optional parameters (with defaults)
    krylov_tolerance: float = attrs.field(
        default=1e-6,
        validator=gttype_validator(float, 0)
    )
    max_linear_iters: int = attrs.field(
        default=10,
        validator=inrangetype_validator(int, 1, 32767)
    )
    # ... etc
```

### Algorithm Class Pattern

```python
class DIRKStep(ODEImplicitStep):
    """DIRK integration step implementation."""
    
    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        **kwargs
    ):
        """Initialize DIRK step with required parameters and optional overrides.
        
        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for computations.
        n : int
            Number of state variables.
        dxdt_function : Callable, optional
            Device function for derivatives.
        observables_function : Callable, optional
            Device function for observables.
        driver_function : Callable, optional
            Device function for time-varying drivers.
        get_solver_helper_fn : Callable, optional
            Factory for solver helpers.
        **kwargs
            Optional parameters (krylov_tolerance, max_linear_iters, etc.).
            Defaults come from DIRKStepConfig attrs class.
        """
        # Build config using helper - merges defaults with provided values
        config = build_config(
            DIRKStepConfig,
            required={
                'precision': precision,
                'n': n,
                'dxdt_function': dxdt_function,
                'observables_function': observables_function,
                'driver_function': driver_function,
                'get_solver_helper_fn': get_solver_helper_fn,
            },
            **kwargs
        )
        
        # Clear and register buffers using config values
        buffer_registry.clear_parent(self)
        self._register_buffers(config)
        
        # Get controller defaults based on config
        defaults = self._get_controller_defaults(config)
        
        # Initialize parent with config and defaults
        super().__init__(config, defaults)
```

### Factory Function Pattern

Factory functions filter settings and pass to algorithm init:

```python
def get_algorithm_step(precision, settings, **kwargs):
    """Create algorithm step from settings.
    
    Parameters
    ----------
    precision : PrecisionDType
        Numerical precision.
    settings : dict
        Settings dict with 'algorithm' key plus optional overrides.
    **kwargs
        Additional overrides (merged with settings).
    
    Returns
    -------
    BaseAlgorithmStep
        Configured algorithm instance.
    """
    algorithm_settings = {**settings, **kwargs}
    algorithm_value = algorithm_settings.pop('algorithm')
    
    # Resolve algorithm type and tableau
    algorithm_type, tableau = resolve_alias(algorithm_value)
    
    # Inject tableau if resolved
    if tableau is not None:
        algorithm_settings['tableau'] = tableau
    
    # Add required precision
    algorithm_settings['precision'] = precision
    
    # Filter settings for this algorithm's init signature
    filtered, missing, unused = split_applicable_settings(
        algorithm_type,
        algorithm_settings,
        warn_on_unused=False
    )
    
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")
    
    # Instantiate algorithm - it will use build_config internally
    return algorithm_type(**filtered)
```

## Helper Functions

### build_config

New utility function in `_utils.py`:

```python
def build_config(
    config_class: type,
    required: dict,
    **optional
) -> Any:
    """Build attrs config instance from required and optional parameters.
    
    Starts with config class defaults for all optional fields, then applies
    provided required and optional values. This eliminates the verbose pattern
    of checking `if param is not None` for every optional parameter.
    
    Parameters
    ----------
    config_class : type
        Attrs class to instantiate (e.g., DIRKStepConfig).
    required : dict
        Required parameters that must be provided. These are typically
        function parameters like precision, n, dxdt_function.
    **optional
        Optional parameter overrides. Only non-None values override defaults
        from the config class.
    
    Returns
    -------
    config_class instance
        Configured attrs object with defaults + required + optional overrides.
    
    Examples
    --------
    >>> config = build_config(
    ...     DIRKStepConfig,
    ...     required={'precision': np.float32, 'n': 3},
    ...     krylov_tolerance=1e-8
    ... )
    
    Notes
    -----
    The helper automatically:
    - Extracts defaults from config_class attrs fields
    - Merges: defaults <- required <- optional (non-None)
    - Filters out None values from optional kwargs
    - Validates all required config fields are present
    """
    # Start with config class defaults
    defaults = {}
    required_fields = set()
    
    for field in attrs.fields(config_class):
        if field.default != attrs.NOTHING:
            # Has default value
            if isinstance(field.default, attrs.Factory):
                defaults[field.name] = field.default.factory()
            else:
                defaults[field.name] = field.default
        else:
            # Required field (no default)
            required_fields.add(field.name)
    
    # Filter optional kwargs to remove None values
    # (None means "use default", not "set to None")
    filtered_optional = {
        k: v for k, v in optional.items() if v is not None
    }
    
    # Merge: defaults <- required <- filtered_optional
    merged = {**defaults, **required, **filtered_optional}
    
    # Validate all required fields are present
    provided_fields = set(merged.keys())
    missing = required_fields - provided_fields
    if missing:
        raise ValueError(
            f"{config_class.__name__} missing required fields: {missing}"
        )
    
    # Filter to only valid fields (ignore extra keys)
    valid_fields = {f.name for f in attrs.fields(config_class)}
    final = {k: v for k, v in merged.items() if k in valid_fields}
    
    return config_class(**final)
```

### Usage in Factories

Example for `get_algorithm_step` (minimal changes needed):

```python
def get_algorithm_step(precision, settings, **kwargs):
    algorithm_settings = {**settings, **kwargs}
    algorithm_value = algorithm_settings.pop('algorithm', None)
    
    if algorithm_value is None:
        raise ValueError("Algorithm settings must include 'algorithm'.")
    
    # Resolve algorithm and tableau
    if isinstance(algorithm_value, str):
        algorithm_type, resolved_tableau = resolve_alias(algorithm_value)
    elif isinstance(algorithm_value, ButcherTableau):
        algorithm_type, resolved_tableau = resolve_supplied_tableau(algorithm_value)
    else:
        raise TypeError(f"Invalid algorithm type: {type(algorithm_value)}")
    
    # Inject precision (required for all algorithms)
    algorithm_settings['precision'] = precision
    
    # Inject tableau if resolved
    if resolved_tableau is not None:
        algorithm_settings['tableau'] = resolved_tableau
    
    # Filter to algorithm's __init__ signature
    filtered, missing, unused = split_applicable_settings(
        algorithm_type,
        algorithm_settings,
        warn_on_unused=False
    )
    
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(
            f"{algorithm_type.__name__} requires settings for: {missing_keys}"
        )
    
    # Algorithm __init__ will use build_config internally
    return algorithm_type(**filtered)
```

## Integration Points

### BaseAlgorithmStep Changes

Minimal changes to base class - init pattern stays similar:

```python
class BaseAlgorithmStep(CUDAFactory, ABC):
    """Base class for algorithm steps."""
    
    def __init__(self, config: BaseStepConfig, _controller_defaults: StepControlDefaults):
        """Initialize with config object.
        
        Parameters
        ----------
        config : BaseStepConfig
            Configuration object (subclass-specific).
        _controller_defaults : StepControlDefaults
            Per-algorithm controller defaults.
            
        Notes
        -----
        Subclasses construct config using build_config helper before calling
        super().__init__. This keeps config construction internal to each
        algorithm while maintaining single source of truth for defaults.
        """
        super().__init__()
        self._controller_defaults = _controller_defaults.copy()
        self.setup_compile_settings(config)
        self.is_controller_fixed = False
```

### Controller Integration

Similar pattern for step controllers:

```python
class AdaptivePIDController(BaseStepController):
    """Adaptive PID step controller."""
    
    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        algorithm_order: int,
        **kwargs
    ):
        """Initialize PID controller.
        
        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision.
        n : int
            Number of state variables.
        algorithm_order : int
            Order of the integration algorithm.
        **kwargs
            Optional parameters (dt, dt_min, dt_max, kp, ki, kd, etc.).
            Defaults from AdaptivePIDConfig attrs class.
        """
        config = build_config(
            AdaptivePIDConfig,
            required={
                'precision': precision,
                'n': n,
                'algorithm_order': algorithm_order,
            },
            **kwargs
        )
        super().__init__(config)
```

### Loop and Output Functions

These components follow the same pattern:

```python
class IVPLoop(CUDAFactory):
    """Integration loop factory."""
    
    def __init__(
        self,
        precision: PrecisionDType,
        n_states: int,
        n_parameters: int,
        n_observables: int,
        n_drivers: int,
        **kwargs
    ):
        """Initialize integration loop.
        
        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision.
        n_states, n_parameters, n_observables, n_drivers : int
            System dimensions.
        **kwargs
            Optional parameters (dt_save, dt_summarise, buffer locations, etc.).
            Defaults from ODELoopConfig attrs class.
        """
        config = build_config(
            ODELoopConfig,
            required={
                'precision': precision,
                'n_states': n_states,
                'n_parameters': n_parameters,
                'n_observables': n_observables,
                'n_drivers': n_drivers,
            },
            **kwargs
        )
        super().__init__()
        self.setup_compile_settings(config)

class OutputFunctions(CUDAFactory):
    """Output handler factory."""
    
    def __init__(
        self,
        precision: PrecisionDType,
        max_states: int,
        max_observables: int,
        **kwargs
    ):
        """Initialize output functions.
        
        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision.
        max_states, max_observables : int
            Maximum dimensions.
        **kwargs
            Optional parameters (output_types, dt_save, selectors, etc.).
            Defaults from OutputConfig attrs class.
        """
        config = build_config(
            OutputConfig,
            required={
                'precision': precision,
                'max_states': max_states,
                'max_observables': max_observables,
            },
            **kwargs
        )
        super().__init__()
        self.setup_compile_settings(config)
```

## Buffer Registry Integration

Buffer registration happens after config construction:

**Current Pattern (Verbose):**
```python
def __init__(self, precision, n, stage_increment_location=None, ...):
    # Build config with verbose checks
    config_kwargs = {'precision': precision, 'n': n}
    if stage_increment_location is not None:
        config_kwargs['stage_increment_location'] = stage_increment_location
    # ... more checks ...
    
    config = DIRKStepConfig(**config_kwargs)
    
    # Register buffers using config values
    buffer_registry.register('dirk_stage_increment', self, config.n, 
                            config.stage_increment_location, precision=config.precision)
```

**New Pattern (Clean):**
```python
def __init__(self, precision, n, **kwargs):
    # Build config cleanly
    config = build_config(
        DIRKStepConfig,
        required={'precision': precision, 'n': n},
        **kwargs
    )
    
    buffer_registry.clear_parent(self)
    
    # Register buffers using config values
    buffer_registry.register(
        'dirk_stage_increment', 
        self, 
        config.n, 
        config.stage_increment_location,
        precision=config.precision
    )

## Controller Defaults Handling

Algorithm-specific controller defaults remain as class-level constants:

```python
DIRK_ADAPTIVE_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "pid",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
        # ... etc
    }
)

class DIRKStep(ODEImplicitStep):
    def __init__(self, precision, n, **kwargs):
        # Build config using helper
        config = build_config(
            DIRKStepConfig,
            required={'precision': precision, 'n': n},
            **kwargs
        )
        
        # Register buffers...
        
        # Select defaults based on config
        if config.tableau.has_error_estimate:
            defaults = DIRK_ADAPTIVE_DEFAULTS
        else:
            defaults = DIRK_FIXED_DEFAULTS
        
        super().__init__(config, defaults)
```

## SingleIntegratorRunCore Changes

Update to use new factory pattern:

**Before:**
```python
algorithm_settings["n"] = n
algorithm_settings["driver_function"] = driver_function
self._algo_step = get_algorithm_step(
    precision=precision,
    settings=algorithm_settings,
)

controller_settings = self._algo_step.controller_defaults.step_controller.copy()
controller_settings.update(step_control_settings)
controller_settings["n"] = n
self._step_controller = get_controller(
    precision=precision,
    settings=controller_settings,
)
```

**After:**
```python
# Inject required fields into algorithm settings
algorithm_settings["n"] = n
algorithm_settings["driver_function"] = driver_function

# Factory handles config construction
self._algo_step = get_algorithm_step(
    precision=precision,
    settings=algorithm_settings,
)

# Merge defaults with user overrides
controller_settings = self._algo_step.controller_defaults.step_controller.copy()
controller_settings.update(step_control_settings)
controller_settings["n"] = n

# Factory handles config construction
self._step_controller = get_controller(
    precision=precision,
    settings=controller_settings,
)
```

(Minimal changes needed here since factory already handles the complexity)

## Edge Cases

### Required vs Optional Parameters

**Required parameters**: No default in config class, must be provided
- `precision`: Always required
- `n`: System size, always required
- `dxdt_function`: Usually required (None allowed in some cases)

**Optional parameters**: Have defaults in config class
- `krylov_tolerance`: defaults to 1e-6
- `max_linear_iters`: defaults to 10
- All buffer location parameters: default to 'local'

### Tableau Injection

Some algorithms require tableaus:
- Factory resolves tableau from alias or explicit instance
- Factory injects into settings before config construction
- Config class has default tableau for that algorithm type

```python
# In factory
if resolved_tableau is not None:
    algorithm_settings['tableau'] = resolved_tableau

# In config class
@attrs.define
class DIRKStepConfig(ImplicitStepConfig):
    tableau: DIRKTableau = attrs.field(default=DEFAULT_DIRK_TABLEAU)
```

### None vs Missing

attrs distinguishes between `None` (explicit value) and missing (use default):

```python
# User explicitly passes None
config = DIRKStepConfig(precision=np.float32, n=3, dxdt_function=None)

# User omits parameter, default used
config = DIRKStepConfig(precision=np.float32, n=3)  # dxdt_function=None (default)
```

Both work because default is `None` in config class.

### Nested Config Objects

Some configs contain nested attrs objects (e.g., `SystemSizes` in `ODEData`):

```python
@attrs.define
class ODELoopConfig:
    system_sizes: SystemSizes = attrs.field()  # Required, no default
    dt_save: float = attrs.field(default=0.01)  # Optional with default
```

Factory must construct nested objects from flattened settings:

```python
# If settings contain flat keys
settings = {'n_states': 3, 'n_parameters': 2, 'dt_save': 0.02}

# Factory builds nested structure
system_sizes = SystemSizes(states=settings.pop('n_states'), 
                          parameters=settings.pop('n_parameters'))
config = ODELoopConfig(system_sizes=system_sizes, **settings)
```

## Expected Behavior Changes

### Init Signature Changes

**Before (Current Verbose Pattern):**
```python
DIRKStep(
    precision, n, dxdt_function=None, krylov_tolerance=None, 
    max_linear_iters=None, tableau=None, stage_increment_location=None, ...
)
```

**After (Clean Pattern):**
```python
DIRKStep(precision, n, **kwargs)
# with kwargs like: krylov_tolerance=1e-8, stage_increment_location='shared'
```

### User-Facing Changes

**Via Solver (no changes):**
```python
solver = Solver(system, algorithm='dirk', krylov_tolerance=1e-8)
```

**Direct Instantiation:**
```python
# Before (works but verbose)
step = DIRKStep(
    precision=np.float32, n=3, 
    krylov_tolerance=1e-8, max_linear_iters=20
)

# After (cleaner with same capability)
step = DIRKStep(
    precision=np.float32, n=3,
    krylov_tolerance=1e-8, max_linear_iters=20
)

# Or via factory (unchanged)
step = get_algorithm_step(
    precision=np.float32,
    settings={'algorithm': 'dirk', 'n': 3, 'krylov_tolerance': 1e-8}
)
```

Key difference: No more explicit `=None` in signature, but kwargs still work the same way.

## Testing Strategy

### Unit Tests

Test config construction:
```python
def test_config_defaults():
    """Verify config class has expected defaults."""
    config = DIRKStepConfig(precision=np.float32, n=3)
    assert config.krylov_tolerance == 1e-6
    assert config.max_linear_iters == 10

def test_config_overrides():
    """Verify overrides replace defaults."""
    config = DIRKStepConfig(
        precision=np.float32, n=3, krylov_tolerance=1e-8
    )
    assert config.krylov_tolerance == 1e-8

def test_build_config_from_settings():
    """Verify helper constructs config correctly."""
    config = build_config_from_settings(
        DIRKStepConfig,
        settings={'n': 3, 'krylov_tolerance': 1e-8},
        precision=np.float32
    )
    assert config.n == 3
    assert config.krylov_tolerance == 1e-8
    assert config.max_linear_iters == 10  # default
```

### Integration Tests

Test factory functions:
```python
def test_get_algorithm_step_with_defaults():
    """Verify factory applies config defaults."""
    step = get_algorithm_step(
        precision=np.float32,
        settings={'algorithm': 'dirk', 'n': 3}
    )
    assert step.compile_settings.krylov_tolerance == 1e-6

def test_get_algorithm_step_with_overrides():
    """Verify factory applies overrides."""
    step = get_algorithm_step(
        precision=np.float32,
        settings={'algorithm': 'dirk', 'n': 3, 'krylov_tolerance': 1e-8}
    )
    assert step.compile_settings.krylov_tolerance == 1e-8
```

### Regression Tests

Existing integration tests should pass unchanged since Solver API is unchanged.

## Migration Path

1. **Phase 1**: Add `build_config` helper function to `_utils.py`
2. **Phase 2**: Refactor backwards_euler.py to use helper (simplify verbose pattern)
3. **Phase 3**: Refactor other algorithm init functions one at a time
4. **Phase 4**: Refactor controller init functions
5. **Phase 5**: Refactor loop and output function init functions
6. **Phase 6**: Update tests to match new init patterns
7. **Phase 7**: Remove obsolete code and validate all tests pass

Each phase can be tested independently. Breaking changes are acceptable per project guidelines.
