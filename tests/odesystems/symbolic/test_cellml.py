import pytest
import sympy as sp
from pathlib import Path

from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model

cellmlmanip = pytest.importorskip("cellmlmanip")


@pytest.fixture
def fixtures_dir():
    """Return path to cellml test fixtures directory."""
    return Path(__file__).parent.parent.parent / "fixtures" / "cellml"


@pytest.fixture
def basic_model_path(fixtures_dir):
    """Return path to basic ODE CellML model."""
    return fixtures_dir / "basic_ode.cellml"


@pytest.fixture
def beeler_reuter_model_path(fixtures_dir):
    """Return path to Beeler-Reuter CellML model."""
    return fixtures_dir / "beeler_reuter_model_1977.cellml"


def test_load_simple_cellml_model(basic_model_path):
    """Load a simple CellML model successfully."""
    states, equations = load_cellml_model(str(basic_model_path))
    
    assert len(states) == 1
    assert len(equations) == 1
    # CellML state names include component prefix (e.g., "main$x")
    assert states[0].name.endswith("$x") or states[0].name == "x"


def test_load_complex_cellml_model(beeler_reuter_model_path):
    """Load Beeler-Reuter cardiac model successfully."""
    states, equations = load_cellml_model(str(beeler_reuter_model_path))
    
    # Beeler-Reuter has 8 state variables
    assert len(states) == 8
    assert len(equations) == 8


def test_states_are_symbols(basic_model_path):
    """Verify states are sympy.Symbol instances (not Dummy)."""
    states, _ = load_cellml_model(str(basic_model_path))
    
    for state in states:
        assert isinstance(state, sp.Symbol)
        assert not isinstance(state, sp.Dummy)


def test_equations_are_sympy_eq(basic_model_path):
    """Verify equations are sympy.Eq instances."""
    _, equations = load_cellml_model(str(basic_model_path))
    
    for eq in equations:
        assert isinstance(eq, sp.Eq)


def test_derivatives_in_equation_lhs(basic_model_path):
    """Verify equation LHS contains derivatives."""
    _, equations = load_cellml_model(str(basic_model_path))
    
    for eq in equations:
        assert isinstance(eq.lhs, sp.Derivative)


def test_all_states_have_derivatives(beeler_reuter_model_path):
    """Verify each state variable has a corresponding derivative."""
    states, equations = load_cellml_model(str(beeler_reuter_model_path))
    
    # Extract derivative arguments from equations
    derivative_vars = set()
    for eq in equations:
        if isinstance(eq.lhs, sp.Derivative):
            # Get the function being differentiated
            derivative_vars.add(eq.lhs.args[0])
    
    # Check all states are covered
    state_set = set(states)
    assert derivative_vars == state_set


def test_equation_format_compatibility(basic_model_path):
    """Verify CellML equation format is compatible with cubie."""
    states, equations = load_cellml_model(str(basic_model_path))
    
    # Verify we can extract the RHS expressions
    for eq in equations:
        assert isinstance(eq.lhs, sp.Derivative)
        # RHS should be a valid sympy expression
        assert isinstance(eq.rhs, sp.Expr)
        # RHS should contain symbols
        assert len(eq.rhs.free_symbols) > 0


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

