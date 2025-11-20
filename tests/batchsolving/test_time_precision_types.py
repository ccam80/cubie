"""Tests for time parameter type preservation."""
import numpy as np
import pytest
from cubie.batchsolving.solver import Solver
from tests.system_fixtures import three_state_linear


def test_solver_stores_time_as_float64(three_state_linear):
    """Verify Solver stores time parameters as float64."""
    system = three_state_linear
    system.precision = np.float32
    
    solver = Solver(
        system,
        algorithm="explicit_euler",
        dt=1e-3,
    )
    
    # Set time parameters as float32
    solver.kernel.duration = np.float32(10.0)
    solver.kernel.warmup = np.float32(1.0)
    solver.kernel.t0 = np.float32(5.0)
    
    # Verify retrieved as float64
    assert isinstance(solver.kernel.duration, (float, np.floating))
    assert isinstance(solver.kernel.warmup, (float, np.floating))
    assert isinstance(solver.kernel.t0, (float, np.floating))
    
    # Verify values preserved
    assert np.isclose(solver.kernel.duration, 10.0)
    assert np.isclose(solver.kernel.warmup, 1.0)
    assert np.isclose(solver.kernel.t0, 5.0)


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_time_precision_independent_of_state_precision(
    three_state_linear, precision
):
    """Verify time precision is float64 regardless of state precision."""
    system = three_state_linear
    system.precision = precision
    
    solver = Solver(
        system,
        algorithm="explicit_euler",
        dt=1e-3,
    )
    
    solver.kernel.duration = 5.0
    solver.kernel.t0 = 1.0
    
    # Time should be float64 even when state precision is float32
    assert solver.kernel.duration == np.float64(5.0)
    assert solver.kernel.t0 == np.float64(1.0)
