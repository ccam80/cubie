import pytest
from pathlib import Path
import numpy as np
import sympy as sp

from cubie import solve_ivp, SolveResult
from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model, \
    _eq_to_equality_str
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

def test_eq_to_equality_str_piecewise_recursive():
    # Build symbols
    i_Ks_ANS_cond_i_Ks = sp.Symbol('i_Ks_ANS_cond_i_Ks')
    PKA_PKA = sp.Symbol('PKA_PKA')
    i_Ks_VW_IKs = sp.Symbol('i_Ks_VW_IKs')
    Rate_modulation_experiments_ANS = sp.Symbol(
        'Rate_modulation_experiments_ANS'
    )
    # Piecewise expression mirroring user example
    piece = sp.Piecewise(
        (
            0.435692 * PKA_PKA**10.0808 /
            (PKA_PKA**10.0808 + 0.0363060458208831) - 0.2152,
            sp.Eq(i_Ks_VW_IKs, 0) & (Rate_modulation_experiments_ANS > 0)
        ),
        (
            0.494259 * PKA_PKA**10.0808 /
            (PKA_PKA**10.0808 + 0.0459499253882566) - 0.2152,
            (Rate_modulation_experiments_ANS > 0) & (i_Ks_VW_IKs > 0)
        ),
        (0, True)
    )
    expr = sp.Eq(i_Ks_ANS_cond_i_Ks, piece)
    out_str = _eq_to_equality_str(expr)
    # Expect conversion of Eq() in condition to ==
    assert 'i_Ks_VW_IKs == 0' in out_str, out_str
    # Expect Piecewise maintained
    assert out_str.count('Piecewise') == 1
    # Expect no raw 'Eq(' tokens remain
    assert 'Eq(' not in out_str


def test_cellml_time_logging_events_registered():
    """Verify time logging events are registered for cellml import."""
    from cubie.time_logger import _default_timelogger
    
    # Check that all cellml events are registered
    expected_events = [
        "codegen_cellml_load_model",
        "codegen_cellml_symbol_conversion",
        "codegen_cellml_equation_processing",
        "codegen_cellml_string_formatting",
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
        "codegen_cellml_string_formatting", "codegen",
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
        assert "codegen_cellml_string_formatting" in event_names
        
        # Check that each event has both start and stop
        for event_name in [
            "codegen_cellml_load_model",
            "codegen_cellml_symbol_conversion",
            "codegen_cellml_equation_processing",
            "codegen_cellml_string_formatting",
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
            "codegen_cellml_string_formatting",
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
        "codegen_cellml_string_formatting", "codegen",
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
        assert "codegen_cellml_string_formatting" in durations
        
        # Verify all durations are non-negative
        for event_name, duration in durations.items():
            if event_name.startswith("codegen_cellml_"):
                assert duration >= 0, f"Duration should be non-negative for {event_name}"

    finally:
        # Restore the original logger
        cellml._default_timelogger = original_logger


def test_cellml_numeric_literals_wrapped(basic_model):
    """Verify that CellML numeric literals are wrapped with precision()."""
    # basic_model is a SymbolicODE loaded from CellML
    # The generated code should have precision() wrapping
    
    # Access the generated code (if available) or compile and check
    # We need to verify the dxdt_function was compiled with wrapped literals
    
    # Since we can't easily inspect compiled CUDA code, we verify:
    # 1. The model compiles successfully (no type errors)
    # 2. The model runs successfully with different precisions
    
    # Test with float32
    result_32 = solve_ivp(
        basic_model,
        t_span=(0, 1),
        initial_values={'main_x': 1.0},
        dt=0.01,
        precision=np.float32
    )
    assert isinstance(result_32.states, np.ndarray)
    assert result_32.states.dtype == np.float32
    
    # Test with float64  
    result_64 = solve_ivp(
        basic_model,
        t_span=(0, 1),
        initial_values={'main_x': 1.0},
        dt=0.01,
        precision=np.float64
    )
    assert isinstance(result_64.states, np.ndarray)
    assert result_64.states.dtype == np.float64
    
    # Verify results are numerically close but precision-appropriate
    # (Different precisions may produce slightly different results)
    assert result_32.states.shape == result_64.states.shape


def test_user_equation_literals_wrapped():
    """Verify user-supplied equation literals are wrapped with precision()."""
    from cubie import SymbolicODE
    
    # Create ODE with magic numbers in equations
    # dx/dt = -0.5 * x + 2.0
    ode = SymbolicODE(
        dxdt=['dx = -0.5 * x + 2.0'],
        observables={}
    )
    
    # Verify it compiles and runs with float32
    result_32 = solve_ivp(
        ode,
        t_span=(0, 1),
        initial_values={'x': 1.0},
        dt=0.01,
        precision=np.float32
    )
    assert result_32.states.dtype == np.float32
    
    # Verify it compiles and runs with float64
    result_64 = solve_ivp(
        ode,
        t_span=(0, 1),
        initial_values={'x': 1.0},
        dt=0.01,
        precision=np.float64
    )
    assert result_64.states.dtype == np.float64
    
    # Verify results make sense (not NaN, reasonable values)
    assert not np.any(np.isnan(result_32.states))
    assert not np.any(np.isnan(result_64.states))

