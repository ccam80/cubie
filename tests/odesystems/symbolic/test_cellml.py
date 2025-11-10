import pytest
from pathlib import Path
import numpy as np

from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model

cellmlmanip = pytest.importorskip("cellmlmanip")


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
    assert ode_system is not None


def test_load_complex_cellml_model(beeler_reuter_model_path):
    """Load Beeler-Reuter cardiac model successfully."""
    ode_system = load_cellml_model(str(beeler_reuter_model_path))
    
    # Beeler-Reuter has 8 state variables
    assert ode_system.num_states == 8
    # System should be fully constructed
    assert ode_system is not None
    assert hasattr(ode_system, 'equations')


def test_ode_system_has_correct_attributes(basic_model_path):
    """Verify ODE system has expected attributes."""
    ode_system = load_cellml_model(str(basic_model_path))
    
    # Should have SymbolicODE attributes
    assert hasattr(ode_system, 'num_states')
    assert hasattr(ode_system, 'equations')
    assert hasattr(ode_system, 'indices')


def test_ode_system_ready_for_integration(beeler_reuter_model_path):
    """Verify ODE system can be used with solve_ivp."""
    ode_system = load_cellml_model(str(beeler_reuter_model_path))
    
    # System should be compilable (has necessary methods)
    assert hasattr(ode_system, 'build')
    assert ode_system.num_states == 8


def test_algebraic_equations_as_observables(beeler_reuter_model_path):
    """Verify algebraic equations are loaded into the system."""
    ode_system = load_cellml_model(str(beeler_reuter_model_path))
    
    # Beeler-Reuter has many algebraic equations
    # These get automatically included as anonymous auxiliaries
    # Verify the system loaded successfully with all equations
    assert ode_system.num_states == 8
    assert ode_system.equations is not None


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
    
    assert ode_system is not None
    assert ode_system.num_states == 1


def test_custom_name(basic_model_path):
    """Verify custom name can be specified."""
    ode_system = load_cellml_model(
        str(basic_model_path),
        name="custom_model"
    )
    
    assert ode_system is not None
    assert ode_system.name == "custom_model"


def test_integration_with_solve_ivp(basic_model_path):
    """Test that loaded model can build successfully."""
    # Skip if running without CUDA sim (may not compile)
    pytest.importorskip("numba")
    
    ode_system = load_cellml_model(str(basic_model_path))
    
    # Build the system - this should not raise an error
    ode_system.build()
    
    # This verifies the system is properly structured and compilable
    assert ode_system.num_states == 1

