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

@pytest.fixture(scope="session")
def fabbri_linder_model(cellml_fixtures_dir, cellml_import_settings):
    """Return path to the whole point of this endea."""
    fl_path = cellml_fixtures_dir / "Fabbri_Linder.cellml"
    ode_system = load_cellml_model(
            str(fl_path),
            **cellml_import_settings
    )
    return ode_system


@pytest.fixture(scope="session")
def demir_1999_model(cellml_fixtures_dir, cellml_import_settings):
    """Return path to the whole point of this endea."""
    fl_path = cellml_fixtures_dir / "demir_clark_giles_1999.cellml"
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

def test_import_demir(demir_1999_model):
    assert demir_1999_model.num_states != 0

def test_import_fabbri(fabbri_linder_model):
    assert fabbri_linder_model.num_states != 0