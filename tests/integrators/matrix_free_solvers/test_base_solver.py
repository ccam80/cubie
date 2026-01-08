import numpy as np
import pytest

from cubie.integrators.matrix_free_solvers.base_solver import (
    MatrixFreeSolverConfig,
)


def test_matrix_free_solver_config_precision_property(precision):
    """Verify numba_precision and simsafe_precision properties work correctly."""
    config = MatrixFreeSolverConfig(precision=precision, n=3)

    # Verify numba_precision returns correct type
    numba_prec = config.numba_precision
    assert numba_prec is not None

    # Verify simsafe_precision returns correct type
    simsafe_prec = config.simsafe_precision
    assert simsafe_prec is not None

    # Verify precision attribute is stored correctly
    assert config.precision == precision


def test_matrix_free_solver_config_validation():
    """Verify precision and n validators work correctly."""
    # Test valid configuration
    config = MatrixFreeSolverConfig(precision=np.float64, n=5)
    assert config.precision == np.float64
    assert config.n == 5

    # Test n must be >= 1
    with pytest.raises(ValueError):
        MatrixFreeSolverConfig(precision=np.float32, n=0)

    # Test invalid precision type raises error
    with pytest.raises(ValueError):
        MatrixFreeSolverConfig(precision=np.int32, n=3)
