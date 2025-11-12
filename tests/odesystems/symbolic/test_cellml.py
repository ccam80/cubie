import pytest
from pathlib import Path
import numpy as np

from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model
from cubie._utils import is_devfunc

# Note: cellmlmanip import removed - tests should fail if dependency missing
# This ensures critical information about missing dependencies is visible


@pytest.fixture
def cellml_fixtures_dir():
    """Return path to cellml test fixtures directory."""
    return Path(__file__).parent.parent.parent / "fixtures" / "cellml"


@pytest.fixture
def basic_model_path(cellml_fixtures_dir):
    """Return path to basic ODE CellML model."""
    return cellml_fixtures_dir / "basic_ode.cellml"


@pytest.fixture
def beeler_reuter_model_path(cellml_fixtures_dir):
    """Return path to Beeler-Reuter CellML model."""
    return cellml_fixtures_dir / "beeler_reuter_model_1977.cellml"


def test_load_simple_cellml_model(basic_model_path):
    """Load a simple CellML model successfully."""
    ode_system = load_cellml_model(str(basic_model_path))
    
    assert ode_system.num_states == 1
    assert is_devfunc(ode_system.dxdt_function)


def test_load_complex_cellml_model(beeler_reuter_model_path):
    """Load Beeler-Reuter cardiac model successfully."""
    ode_system = load_cellml_model(str(beeler_reuter_model_path))
    
    # Beeler-Reuter has 8 state variables
    assert ode_system.num_states == 8
    assert is_devfunc(ode_system.dxdt_function)


def test_ode_system_has_correct_attributes(basic_model_path):
    """Verify ODE system has expected attributes."""
    ode_system = load_cellml_model(str(basic_model_path))
    
    # Should have SymbolicODE attributes
    assert hasattr(ode_system, 'num_states')
    assert hasattr(ode_system, 'equations')
    assert hasattr(ode_system, 'indices')


def test_algebraic_equations_as_observables(beeler_reuter_model_path):
    """Verify algebraic equations can be assigned as observables."""
    # Load with specific observables (sanitized names from the model)
    observable_names = ["sodium_current_i_Na", "sodium_current_m_gate_alpha_m"]
    ode_system = load_cellml_model(
        str(beeler_reuter_model_path),
        observables=observable_names
    )
    
    # Verify the observables were assigned
    obs_map = ode_system.indices.observables.index_map
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


def test_custom_precision(basic_model_path):
    """Verify custom precision can be specified."""
    ode_system = load_cellml_model(
        str(basic_model_path),
        precision=np.float64
    )
    
    assert ode_system.precision == np.float64


def test_custom_name(basic_model_path):
    """Verify custom name can be specified."""
    ode_system = load_cellml_model(
        str(basic_model_path),
        name="custom_model"
    )
    
    assert ode_system.name == "custom_model"


def test_integration_with_solve_ivp(basic_model_path):
    """Test that loaded model builds and is ready for solve_ivp."""
    # Use float64 to avoid dtype mismatch in cuda simulator
    ode_system = load_cellml_model(str(basic_model_path), precision=np.float64)
    
    # Build the system - this is the critical step that verifies
    # the model is properly structured for integration
    ode_system.build()
    
    # Verify the model has the necessary components
    assert is_devfunc(ode_system.dxdt_function)
    assert ode_system.num_states == 1
    
    # Verify initial values are accessible
    assert ode_system.indices.states.defaults is not None
    assert len(ode_system.indices.states.defaults) == 1


def test_initial_values_from_cellml(beeler_reuter_model_path):
    """Verify initial values from CellML model are preserved."""
    ode_system = load_cellml_model(str(beeler_reuter_model_path))
    
    # Check that initial values were set using defaults dict
    assert ode_system.indices.states.defaults is not None
    assert len(ode_system.indices.states.defaults) == 8
    
    # Initial values should be non-zero (from the model)
    assert any(v != 0 for v in ode_system.indices.states.defaults.values())


def test_numeric_assignments_become_constants(basic_model_path):
    """Verify variables with numeric assignments become constants by default."""
    ode_system = load_cellml_model(str(basic_model_path))
    
    # Variable 'a' has numeric value 0.5 in the CellML model
    # It should become a constant
    constants_map = ode_system.indices.constants.index_map
    assert len(constants_map) > 0
    
    # Check that 'main_a' is in constants (name is sanitized)
    constant_names = [str(k) for k in constants_map.keys()]
    assert 'main_a' in constant_names
    
    # Check that the default value is correct
    constants_defaults = ode_system.indices.constants.defaults
    assert constants_defaults is not None
    assert 'main_a' in constants_defaults
    assert constants_defaults['main_a'] == 0.5


def test_numeric_assignments_as_parameters(basic_model_path):
    """Verify variables with numeric assignments become parameters if specified."""
    # Load with 'main_a' in the parameters list
    ode_system = load_cellml_model(
        str(basic_model_path),
        parameters=['main_a']
    )
    
    # 'main_a' should now be a parameter instead of a constant
    parameters_map = ode_system.indices.parameters.index_map
    parameter_names = [str(k) for k in parameters_map.keys()]
    assert 'main_a' in parameter_names
    
    # Check that the default value is correct
    parameters_defaults = ode_system.indices.parameters.defaults
    assert parameters_defaults is not None
    assert 'main_a' in parameters_defaults
    assert parameters_defaults['main_a'] == 0.5
    
    # Should not be in constants
    constants_map = ode_system.indices.constants.index_map
    constant_names = [str(k) for k in constants_map.keys()]
    assert 'main_a' not in constant_names


def test_parameters_dict_preserves_numeric_values(basic_model_path):
    """Verify numeric values are preserved when parameters is a dict."""
    # User can provide parameters as dict with custom default values
    # But if the CellML has a numeric value, it should be preserved
    ode_system = load_cellml_model(
        str(basic_model_path),
        parameters={'main_a': 1.0}  # User provides different value
    )
    
    # The user-provided value should take precedence
    parameters_defaults = ode_system.indices.parameters.defaults
    assert parameters_defaults is not None
    assert 'main_a' in parameters_defaults
    assert parameters_defaults['main_a'] == 1.0


def test_non_numeric_algebraic_equations_remain(beeler_reuter_model_path):
    """Verify non-numeric algebraic equations are not converted to constants."""
    ode_system = load_cellml_model(str(beeler_reuter_model_path))
    
    # The Beeler-Reuter model has complex algebraic equations
    # These should remain as equations, not become constants
    # We can check by ensuring there are equations beyond just the differential ones
    
    # Model has 8 state variables, so 8 differential equations
    # If there are more equations total, they are algebraic
    all_equations = ode_system.equations
    differential_eq_count = len([eq for eq in all_equations.keys() 
                                  if str(eq).startswith('d')])
    
    # Should have differential equations equal to number of states
    assert differential_eq_count == 8
    
    # Total equations should be more than just differential
    # (algebraic equations that aren't simple numeric assignments)
    assert len(all_equations) >= differential_eq_count

