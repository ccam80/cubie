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
def basic_model_path(cellml_fixtures_dir):
    """Return path to basic ODE CellML model file."""
    return cellml_fixtures_dir / "basic_ode.cellml"

@pytest.fixture(scope="session")
def beeler_reuter_model_path(cellml_fixtures_dir):
    """Return path to Beeler-Reuter CellML model file."""
    return cellml_fixtures_dir / "beeler_reuter_model_1977.cellml"

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

@pytest.fixture(scope="session")
def fabbri_linder_model(cellml_fixtures_dir, cellml_import_settings):
    """Return path to the whole point of this endea."""
    fl_path = cellml_fixtures_dir / "Fabbri_Linder.cellml"
    ode_system = load_cellml_model(
            str(fl_path),
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
    # Check that we have state derivatives
    state_derivatives = ode_system.equations.state_derivatives
    assert len(state_derivatives) == 8
    
    # Check that we have some observables or auxiliaries
    # (algebraic equations that aren't simple numeric assignments)
    observables = ode_system.equations.observables
    auxiliaries = ode_system.equations.auxiliaries
    
    # Total algebraic equations should be > 0
    algebraic_eq_count = len(observables) + len(auxiliaries)
    assert algebraic_eq_count > 0

def test_import_the_big_boy(fabbri_linder_model):
    assert fabbri_linder_model.num_states != 0
