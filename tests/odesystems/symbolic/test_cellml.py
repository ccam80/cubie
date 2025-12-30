import pytest
from pathlib import Path
import numpy as np

from cubie import solve_ivp, SolveResult
from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model
from cubie._utils import is_devfunc

# Note: cellmlmanip import removed - tests should fail if dependency missing
# This ensures critical information about missing dependencies is visible


@pytest.fixture(scope="session")
def cellml_fixtures_dir():
    """Return path to cellml test fixtures directory."""
    return Path(__file__).parent.parent.parent / "fixtures" / "cellml"

@pytest.fixture(scope="session")
def cellml_overrides(request):
    if hasattr(request, "param"):
        return request.param if request.param else {}
    return {}

@pytest.fixture(scope="session")
def cellml_import_settings(cellml_overrides):
    """Return path to basic ODE CellML model."""
    defaults = {}
    defaults.update(cellml_overrides)
    return defaults

@pytest.fixture(scope="session")
def basic_model(cellml_fixtures_dir, cellml_import_settings):
    """Return imported basic ODE CellML model."""
    ode_system = load_cellml_model(
            str(cellml_fixtures_dir/"basic_ode.cellml"),
            **cellml_import_settings
    )
    return ode_system


@pytest.fixture(scope="session")
def beeler_reuter_model(cellml_fixtures_dir, cellml_import_settings):
    """Return imported Beeler-Reuter CellML model."""
    br_path = cellml_fixtures_dir / "beeler_reuter_model_1977.cellml"
    ode_system = load_cellml_model(
            str(br_path),
            **cellml_import_settings
    )
    return ode_system

def test_load_simple_cellml_model(basic_model):
    """Load a simple CellML model successfully."""
    assert basic_model.num_states == 1
    assert is_devfunc(basic_model.dxdt_function)


def test_load_complex_cellml_model(beeler_reuter_model):
    """Load Beeler-Reuter cardiac model successfully."""
    assert beeler_reuter_model.num_states == 8
    assert is_devfunc(beeler_reuter_model.dxdt_function)

@pytest.mark.parametrize("cellml_overrides", [
    {'observables': ['sodium_current_i_Na',
                     'sodium_current_m_gate_alpha_m']}
    ],
    indirect=True,
    ids=[""]
)
def test_algebraic_equations_as_observables(beeler_reuter_model, cellml_overrides):
    """Verify algebraic equations can be assigned as observables."""
    # Load with specific observables (sanitized names from the model)
    observable_names = cellml_overrides['observables']
    
    # Verify the observables were assigned
    obs_map = beeler_reuter_model.indices.observables.index_map
    assert len(obs_map) > 0
    
    # Check that the requested observables are present
    # Keys are symbols, so we need to compare names
    obs_symbol_names = [str(k) for k in obs_map.keys()]
    assert len(obs_map) == 2
    for obs_name in observable_names:
        assert obs_name in obs_symbol_names


def test_invalid_path_type():
    """Verify TypeError raised for non-string path."""
    with pytest.raises(TypeError, match="path must be a string"):
        load_cellml_model(123)


def test_nonexistent_file():
    """Verify FileNotFoundError raised for missing file."""
    with pytest.raises(FileNotFoundError, match="CellML file not found"):
        load_cellml_model("/nonexistent/path/model.cellml")


def test_invalid_extension():
    """Verify ValueError raised for non-.cellml extension."""
    import tempfile
    import os
    
    # Create a temporary file with wrong extension
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml',
                                     delete=False) as f:
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="must have .cellml extension"):
            load_cellml_model(temp_path)
    finally:
        os.unlink(temp_path)

@pytest.mark.parametrize("cellml_overrides", [{'precision': np.float64}],
    indirect=True,
    ids=[""]
)
def test_custom_precision(basic_model):
    """Verify custom precision can be specified."""
    assert basic_model.precision == np.float64

@pytest.mark.parametrize("cellml_overrides", [{'name': "custom_model"}],
    indirect=True,
    ids=[""]
)
def test_custom_name(basic_model):
    """Verify custom name can be specified."""
    assert basic_model.name == "custom_model"


def test_integration_with_solve_ivp(basic_model):
    """Test that loaded model builds and is ready for solve_ivp."""

    
    # Verify the model has the necessary components
    assert is_devfunc(basic_model.dxdt_function)
    assert basic_model.num_states == 1
    # Verify initial values are accessible
    assert basic_model.indices.states.defaults is not None
    results = solve_ivp(basic_model, [1.0], [1.0])
    assert isinstance(results, SolveResult)


def test_initial_values_from_cellml(beeler_reuter_model):
    """Verify initial values from CellML model are preserved."""
    # Check that initial values were set using defaults dict
    assert beeler_reuter_model.indices.states.defaults is not None
    assert len(beeler_reuter_model.indices.states.defaults) == 8

    # Initial values should be non-zero (from the model)
    assert any(v != 0 for v in beeler_reuter_model.indices.states.defaults.values())

def test_units_extracted_from_cellml(basic_model):
    """Verify units are extracted from CellML model."""
    # Check that units are available
    assert hasattr(basic_model, 'state_units')
    assert hasattr(basic_model, 'parameter_units')
    assert hasattr(basic_model, 'observable_units')
    
    # Basic model should have dimensionless units
    assert 'main_x' in basic_model.state_units
    assert basic_model.state_units['main_x'] == 'dimensionless'

def test_default_units_for_symbolic_ode():
    """Verify SymbolicODE defaults to dimensionless units."""
    from cubie import SymbolicODE
    import numpy as np
    
    ode = SymbolicODE.create(
        dxdt="dx = -a * x",
        states={'x': 1.0},
        parameters={'a': 0.5},
        precision=np.float32
    )
    
    assert ode.state_units == {'x': 'dimensionless'}
    assert ode.parameter_units == {'a': 'dimensionless'}
    assert ode.observable_units == {}


def test_cellml_uses_sympy_pathway(basic_model):
    """Verify CellML adapter uses SymPy pathway internally."""
    assert basic_model.num_states == 1
    assert is_devfunc(basic_model.dxdt_function)
    
    initial_vals = basic_model.indices.states.default_values
    assert len(initial_vals) > 0


def test_cellml_timing_events_updated():
    """Verify timing events use new SymPy preparation name."""
    from cubie.time_logger import _default_timelogger
    
    registered_events = _default_timelogger._event_registry
    assert "codegen_cellml_sympy_preparation" in registered_events
    
    assert "codegen_cellml_string_formatting" not in registered_events

def test_custom_units_for_symbolic_ode():
    """Verify custom units can be specified for SymbolicODE."""
    from cubie import SymbolicODE
    import numpy as np
    
    ode = SymbolicODE.create(
        dxdt=["dx = -a * x", "y = 2 * x"],
        states={'x': 1.0},
        parameters={'a': 0.5},
        observables=['y'],
        state_units={'x': 'meters'},
        parameter_units={'a': 'per_second'},
        observable_units={'y': 'meters'},
        precision=np.float32
    )
    
    assert ode.state_units == {'x': 'meters'}
    assert ode.parameter_units == {'a': 'per_second'}
    assert ode.observable_units == {'y': 'meters'}

def test_numeric_assignments_become_constants(basic_model):
    """Verify variables with numeric assignments become constants by default."""
    # Variable 'a' has numeric value 0.5 in the CellML model
    # It should become a constant
    constants_map = basic_model.indices.constants.index_map
    assert len(constants_map) > 0
    
    # Check that 'main_a' is in constants (name is sanitized)
    constant_names = [str(k) for k in constants_map.keys()]
    assert 'main_a' in constant_names
    
    # Check that the default value is correct
    constants_defaults = basic_model.indices.constants.defaults
    assert constants_defaults is not None
    assert 'main_a' in constants_defaults
    assert constants_defaults['main_a'] == 0.5

@pytest.mark.parametrize("cellml_overrides", [{'parameters': ['main_a']}],
    indirect=True,
    ids=[""]
)
def test_numeric_assignments_as_parameters(basic_model):
    """Verify variables with numeric assignments become parameters if specified."""

    # 'main_a' should now be a parameter instead of a constant
    parameters_map = basic_model.indices.parameters.index_map
    parameter_names = [str(k) for k in parameters_map.keys()]
    assert 'main_a' in parameter_names
    
    # Check that the default value is correct
    parameters_defaults = basic_model.indices.parameters.defaults
    assert parameters_defaults is not None
    assert 'main_a' in parameters_defaults
    assert parameters_defaults['main_a'] == 0.5
    
    # Should not be in constants
    constants_map = basic_model.indices.constants.index_map
    constant_names = [str(k) for k in constants_map.keys()]
    assert 'main_a' not in constant_names

@pytest.mark.parametrize("cellml_overrides", [{'parameters': {'main_a': 1.0}}],
    indirect=True,
    ids=[""]
)
def test_parameters_dict_preserves_numeric_values(basic_model):
    """Verify numeric values are preserved when parameters is a dict."""
    # User can provide parameters as dict with custom default values
    # But if the CellML has a numeric value, it should be preserved

    # The user-provided value should take precedence
    parameters_defaults = basic_model.indices.parameters.defaults
    assert parameters_defaults is not None
    assert 'main_a' in parameters_defaults
    assert parameters_defaults['main_a'] == 1.0


def test_non_numeric_algebraic_equations_remain(beeler_reuter_model):
    # The Beeler-Reuter model has complex algebraic equations
    # These should remain as equations, not become constants
    # We can check by ensuring there are equations beyond just the differential ones
    
    # Model has 8 state variables, so 8 differential equations
    # Check that we have state derivatives
    state_derivatives = beeler_reuter_model.equations.state_derivatives
    assert len(state_derivatives) == 8
    
    # Check that we have some observables or auxiliaries
    # (algebraic equations that aren't simple numeric assignments)
    observables = beeler_reuter_model.equations.observables
    auxiliaries = beeler_reuter_model.equations.auxiliaries
    
    # Total algebraic equations should be > 0
    algebraic_eq_count = len(observables) + len(auxiliaries)
    assert algebraic_eq_count > 0

def test_cellml_time_logging_events_registered():
    """Verify time logging events are registered for cellml import."""
    from cubie.time_logger import _default_timelogger
    
    # Check that all cellml events are registered
    expected_events = [
        "codegen_cellml_load_model",
        "codegen_cellml_symbol_conversion",
        "codegen_cellml_equation_processing",
        "codegen_cellml_sympy_preparation",
    ]
    
    for event_name in expected_events:
        assert event_name in _default_timelogger._event_registry
        assert _default_timelogger._event_registry[event_name]["category"] == "codegen"


def test_cellml_time_logging_events_recorded(cellml_fixtures_dir):
    """Verify time logging events are recorded during cellml import."""
    from cubie.time_logger import TimeLogger
    
    # Create a new logger to track just this import
    test_logger = TimeLogger(verbosity='default')
    
    # Register the events
    test_logger.register_event(
        "codegen_cellml_load_model", "codegen",
        "Codegen time for cellmlmanip.load_model()"
    )
    test_logger.register_event(
        "codegen_cellml_symbol_conversion", "codegen",
        "Codegen time for converting Dummy symbols to Symbols"
    )
    test_logger.register_event(
        "codegen_cellml_equation_processing", "codegen",
        "Codegen time for processing differential and algebraic equations"
    )
    test_logger.register_event(
        "codegen_cellml_sympy_preparation", "codegen",
        "Codegen time for formatting equations as strings"
    )
    
    # Temporarily replace the global logger
    from cubie.odesystems.symbolic.parsing import cellml
    original_logger = cellml._default_timelogger
    cellml._default_timelogger = test_logger
    
    try:
        # Load a model
        ode_system = load_cellml_model(
            str(cellml_fixtures_dir / "basic_ode.cellml")
        )

        # Verify model loaded successfully
        assert ode_system is not None

        # Verify events were recorded
        event_names = [event.name for event in test_logger.events]
        
        # Check that start and stop events were recorded for each operation
        assert "codegen_cellml_load_model" in event_names
        assert "codegen_cellml_symbol_conversion" in event_names
        assert "codegen_cellml_equation_processing" in event_names
        assert "codegen_cellml_sympy_preparation" in event_names
        
        # Check that each event has both start and stop
        for event_name in [
            "codegen_cellml_load_model",
            "codegen_cellml_symbol_conversion",
            "codegen_cellml_equation_processing",
            "codegen_cellml_sympy_preparation",
        ]:
            start_events = [e for e in test_logger.events
                          if e.name == event_name and e.event_type == "start"]
            stop_events = [e for e in test_logger.events
                         if e.name == event_name and e.event_type == "stop"]
            assert len(start_events) == 1, f"Expected 1 start event for {event_name}"
            assert len(stop_events) == 1, f"Expected 1 stop event for {event_name}"
        
        # Verify durations can be calculated
        for event_name in [
            "codegen_cellml_load_model",
            "codegen_cellml_symbol_conversion",
            "codegen_cellml_equation_processing",
            "codegen_cellml_sympy_preparation",
        ]:
            duration = test_logger.get_event_duration(event_name)
            assert duration is not None, f"Duration not found for {event_name}"
            assert duration >= 0, f"Duration should be non-negative for {event_name}"
        
    finally:
        # Restore the original logger
        cellml._default_timelogger = original_logger


def test_cellml_time_logging_aggregation(cellml_fixtures_dir):
    """Verify time logging can aggregate cellml import durations."""
    from cubie.time_logger import TimeLogger
    
    # Create a new logger to track just this import
    test_logger = TimeLogger(verbosity='default')
    
    # Register the events
    test_logger.register_event(
        "codegen_cellml_load_model", "codegen",
        "Codegen time for cellmlmanip.load_model()"
    )
    test_logger.register_event(
        "codegen_cellml_symbol_conversion", "codegen",
        "Codegen time for converting Dummy symbols to Symbols"
    )
    test_logger.register_event(
        "codegen_cellml_equation_processing", "codegen",
        "Codegen time for processing differential and algebraic equations"
    )
    test_logger.register_event(
        "codegen_cellml_sympy_preparation", "codegen",
        "Codegen time for formatting equations as strings"
    )
    
    # Temporarily replace the global logger
    from cubie.odesystems.symbolic.parsing import cellml
    original_logger = cellml._default_timelogger
    cellml._default_timelogger = test_logger
    
    try:
        # Load a model
        ode_system = load_cellml_model(
            str(cellml_fixtures_dir / "basic_ode.cellml")
        )

        # Verify model loaded successfully
        assert ode_system is not None

        # Get aggregate durations for codegen category
        durations = test_logger.get_aggregate_durations(category="codegen")
        
        # Verify all cellml events are in the aggregation
        assert "codegen_cellml_load_model" in durations
        assert "codegen_cellml_symbol_conversion" in durations
        assert "codegen_cellml_equation_processing" in durations
        assert "codegen_cellml_sympy_preparation" in durations
        
        # Verify all durations are non-negative
        for event_name, duration in durations.items():
            if event_name.startswith("codegen_cellml_"):
                assert duration >= 0, f"Duration should be non-negative for {event_name}"

    finally:
        # Restore the original logger
        cellml._default_timelogger = original_logger


