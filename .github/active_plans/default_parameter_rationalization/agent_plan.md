# Agent Implementation Plan: Default Parameter Rationalization

## Architectural Overview

This plan implements **Option A: Attrs-First with Config Object** to rationalize default parameter handling across cubie's configuration cascade.

### Core Principle

**Single Source of Truth**: All optional parameter defaults live in attrs config classes. Init functions accept only required parameters plus kwargs for overrides.

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
get_algorithm_step → build_config_from_settings → 
ConfigClass(with defaults + overrides) → 
Algorithm.__init__(config)
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
    
    # Class attribute linking to config
    CONFIG_CLASS = DIRKStepConfig
    
    def __init__(self, config: DIRKStepConfig, **overrides):
        """Initialize with config object.
        
        Parameters
        ----------
        config : DIRKStepConfig
            Configuration object with all parameters.
        **overrides
            Runtime overrides to config fields.
            Applied via attrs.evolve if provided.
        """
        if overrides:
            # Validate overrides against config fields
            valid_fields = {f.name for f in attrs.fields(type(config))}
            invalid = set(overrides) - valid_fields
            if invalid:
                raise ValueError(f"Invalid config fields: {invalid}")
            config = attrs.evolve(config, **overrides)
        
        # Store config for buffer registration
        self._config = config
        
        # Clear and register buffers using config values
        buffer_registry.clear_parent(self)
        self._register_buffers()
        
        # Get controller defaults based on config
        defaults = self._get_controller_defaults()
        
        # Initialize parent with config and defaults
        super().__init__(config, defaults)
```

### Factory Function Pattern

Factory functions construct config objects and pass to algorithm:

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
    
    # Get config class from algorithm type
    config_class = algorithm_type.CONFIG_CLASS
    
    # Build config with precision + settings
    config = build_config_from_settings(
        config_class,
        precision=precision,
        settings=algorithm_settings,
        tableau=tableau  # Optional: injected by factory
    )
    
    # Instantiate algorithm with config
    return algorithm_type(config)
```

## Helper Functions

### build_config_from_settings

New utility function in `_utils.py`:

```python
def build_config_from_settings(
    config_class: type,
    settings: dict,
    **required_overrides
) -> Any:
    """Build attrs config instance from settings dict.
    
    Starts with config class defaults, merges settings, applies required
    overrides. Validates that required fields (no default) are provided.
    
    Parameters
    ----------
    config_class : type
        Attrs class to instantiate (e.g., DIRKStepConfig).
    settings : dict
        Settings dict with optional parameter overrides.
    **required_overrides
        Required fields that must be provided (e.g., precision=np.float32).
    
    Returns
    -------
    config_class instance
        Configured attrs object with defaults + overrides.
    
    Raises
    ------
    ValueError
        If required field is missing from both settings and required_overrides.
    """
    # Extract defaults from config class
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
            # Required field
            required_fields.add(field.name)
    
    # Merge: defaults <- settings <- required_overrides
    merged = {**defaults, **settings, **required_overrides}
    
    # Check all required fields are present
    missing = required_fields - set(merged.keys())
    if missing:
        raise ValueError(
            f"{config_class.__name__} missing required fields: {missing}"
        )
    
    # Filter to only valid fields (ignore extra keys in settings)
    valid_fields = {f.name for f in attrs.fields(config_class)}
    filtered = {k: v for k, v in merged.items() if k in valid_fields}
    
    return config_class(**filtered)
```

### Usage in Factories

Example for `get_algorithm_step`:

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
    
    # Get config class
    config_class = algorithm_type.CONFIG_CLASS
    
    # Inject tableau if resolved
    if resolved_tableau is not None:
        algorithm_settings['tableau'] = resolved_tableau
    
    # Build config with precision (required) + settings (optional)
    config = build_config_from_settings(
        config_class,
        algorithm_settings,
        precision=precision
    )
    
    # Instantiate algorithm with config
    return algorithm_type(config)
```

## Integration Points

### BaseAlgorithmStep Changes

Modify base class to support new pattern:

```python
class BaseAlgorithmStep(CUDAFactory, ABC):
    """Base class for algorithm steps."""
    
    CONFIG_CLASS = None  # Override in subclasses
    
    def __init__(self, config: BaseStepConfig, _controller_defaults: StepControlDefaults):
        """Initialize with config object.
        
        Parameters
        ----------
        config : BaseStepConfig
            Configuration object (subclass-specific).
        _controller_defaults : StepControlDefaults
            Per-algorithm controller defaults.
        """
        super().__init__()
        self._controller_defaults = _controller_defaults.copy()
        self.setup_compile_settings(config)
        self.is_controller_fixed = False
```

### Controller Integration

Similar changes for step controllers:

```python
class AdaptivePIDController(BaseStepController):
    """Adaptive PID step controller."""
    
    CONFIG_CLASS = AdaptivePIDConfig
    
    def __init__(self, config: AdaptivePIDConfig, **overrides):
        if overrides:
            config = attrs.evolve(config, **overrides)
        super().__init__(config)
```

### Loop and Output Functions

These components also use config pattern:

```python
class IVPLoop(CUDAFactory):
    """Integration loop factory."""
    
    def __init__(self, config: ODELoopConfig, **overrides):
        if overrides:
            config = attrs.evolve(config, **overrides)
        super().__init__()
        self.setup_compile_settings(config)

class OutputFunctions(CUDAFactory):
    """Output handler factory."""
    
    def __init__(self, config: OutputConfig, **overrides):
        if overrides:
            config = attrs.evolve(config, **overrides)
        super().__init__()
        self.setup_compile_settings(config)
```

## Buffer Registry Integration

Buffer registration currently happens in init before config creation. Reorder:

**Before:**
```python
def __init__(self, precision, n, stage_increment_location='local', ...):
    # Register buffers using parameter values
    buffer_registry.register('dirk_stage_increment', self, n, 
                            stage_increment_location, precision=precision)
    
    # Then build config
    config = DIRKStepConfig(
        precision=precision,
        stage_increment_location=stage_increment_location,
        ...
    )
```

**After:**
```python
def __init__(self, config: DIRKStepConfig, **overrides):
    if overrides:
        config = attrs.evolve(config, **overrides)
    
    self._config = config
    buffer_registry.clear_parent(self)
    
    # Register buffers using config values
    buffer_registry.register(
        'dirk_stage_increment', 
        self, 
        config.n, 
        config.stage_increment_location,
        precision=config.precision
    )
```

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
    def __init__(self, config: DIRKStepConfig, **overrides):
        # ... handle config and overrides ...
        
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

**Before:**
```python
DIRKStep(precision, n, dxdt_function=None, krylov_tolerance=1e-6, 
         max_linear_iters=10, tableau=DEFAULT_TABLEAU, ...)
```

**After:**
```python
DIRKStep(config)
# or with overrides
DIRKStep(config, krylov_tolerance=1e-8)
```

### User-Facing Changes

**Before (still works via Solver):**
```python
solver = Solver(system, algorithm='dirk', krylov_tolerance=1e-8)
```

**After (same, but internally uses config objects):**
```python
solver = Solver(system, algorithm='dirk', krylov_tolerance=1e-8)
```

Users of Solver/solve_ivp see no changes. Direct algorithm instantiation requires config:

**Before:**
```python
step = DIRKStep(precision=np.float32, n=3, krylov_tolerance=1e-8)
```

**After:**
```python
config = DIRKStepConfig(precision=np.float32, n=3, krylov_tolerance=1e-8)
step = DIRKStep(config)

# Or use factory
step = get_algorithm_step(
    precision=np.float32,
    settings={'algorithm': 'dirk', 'n': 3, 'krylov_tolerance': 1e-8}
)
```

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

1. **Phase 1**: Add helper function `build_config_from_settings` to `_utils.py`
2. **Phase 2**: Add `CONFIG_CLASS` attribute to all algorithm and controller classes
3. **Phase 3**: Refactor algorithm factory (`get_algorithm_step`)
4. **Phase 4**: Refactor controller factory (`get_controller`)
5. **Phase 5**: Simplify algorithm init functions (one at a time)
6. **Phase 6**: Simplify controller init functions
7. **Phase 7**: Update loop and output function factories
8. **Phase 8**: Remove obsolete code and update tests

Each phase can be tested independently. Breaking changes are acceptable per project guidelines.
