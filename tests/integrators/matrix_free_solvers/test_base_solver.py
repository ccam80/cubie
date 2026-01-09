import numpy as np
import pytest

from cubie.integrators.matrix_free_solvers.base_solver import (
    MatrixFreeSolverConfig,
    MatrixFreeSolver,
)
from cubie.integrators.norms import ScaledNorm


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


def test_matrix_free_solver_config_max_iters_default():
    """Verify max_iters defaults to 100."""
    config = MatrixFreeSolverConfig(precision=np.float64, n=3)
    assert config.max_iters == 100


def test_matrix_free_solver_config_max_iters_validation():
    """Verify max_iters rejects values < 1 and > 32767."""
    # Test valid boundary values
    config_min = MatrixFreeSolverConfig(precision=np.float64, n=3, max_iters=1)
    assert config_min.max_iters == 1

    config_max = MatrixFreeSolverConfig(
        precision=np.float64, n=3, max_iters=32767
    )
    assert config_max.max_iters == 32767

    # Test invalid: below minimum
    with pytest.raises((ValueError, TypeError)):
        MatrixFreeSolverConfig(precision=np.float64, n=3, max_iters=0)

    # Test invalid: above maximum
    with pytest.raises((ValueError, TypeError)):
        MatrixFreeSolverConfig(precision=np.float64, n=3, max_iters=32768)

    # Test invalid: wrong type
    with pytest.raises((ValueError, TypeError)):
        MatrixFreeSolverConfig(precision=np.float64, n=3, max_iters=50.5)


def test_matrix_free_solver_config_norm_device_function_field():
    """Verify norm_device_function field exists and accepts None or Callable."""
    # Test default is None
    config = MatrixFreeSolverConfig(precision=np.float64, n=3)
    assert config.norm_device_function is None

    # Test accepts a callable
    def dummy_norm():
        pass

    config_with_fn = MatrixFreeSolverConfig(
        precision=np.float64, n=3, norm_device_function=dummy_norm
    )
    assert config_with_fn.norm_device_function is dummy_norm

    # Verify eq=False behavior: configs with different functions are still equal
    # (since norm_device_function is excluded from equality)
    def another_norm():
        pass

    config_other = MatrixFreeSolverConfig(
        precision=np.float64, n=3, norm_device_function=another_norm
    )
    assert config_with_fn == config_other


def test_matrix_free_solver_creates_norm():
    """Verify MatrixFreeSolver creates ScaledNorm in constructor."""
    # Create a concrete subclass for testing (MatrixFreeSolver is abstract)
    class TestSolver(MatrixFreeSolver):
        def build(self):
            pass

    solver = TestSolver(precision=np.float64, n=3)

    # Verify norm attribute exists and is a ScaledNorm instance
    assert hasattr(solver, 'norm')
    assert isinstance(solver.norm, ScaledNorm)

    # Verify norm has correct precision and n
    assert solver.norm.precision == np.float64
    assert solver.norm.n == 3


def test_matrix_free_solver_extract_prefixed_tolerance():
    """Verify _extract_prefixed_tolerance correctly maps prefixed keys."""
    class TestSolver(MatrixFreeSolver):
        settings_prefix = "krylov_"

        def build(self):
            pass

    solver = TestSolver(precision=np.float64, n=3)

    # Test extraction of prefixed tolerance keys
    updates = {
        'krylov_atol': np.array([1e-8]),
        'krylov_rtol': np.array([1e-6]),
        'other_param': 42,
    }

    norm_updates = solver._extract_prefixed_tolerance(updates)

    # Verify prefixed keys were extracted and mapped to unprefixed
    assert 'atol' in norm_updates
    assert 'rtol' in norm_updates
    assert np.array_equal(norm_updates['atol'], np.array([1e-8]))
    assert np.array_equal(norm_updates['rtol'], np.array([1e-6]))

    # Verify prefixed keys were removed from original dict
    assert 'krylov_atol' not in updates
    assert 'krylov_rtol' not in updates

    # Verify other parameters remain
    assert updates['other_param'] == 42


def test_matrix_free_solver_norm_update_propagates_to_config():
    """Verify _update_norm_and_config sets norm_device_function in config."""
    class TestSolver(MatrixFreeSolver):
        def build(self):
            pass

    solver = TestSolver(precision=np.float64, n=3)

    # Set up compile settings (required by _update_norm_and_config)
    config = MatrixFreeSolverConfig(precision=np.float64, n=3)
    solver.setup_compile_settings(config)

    # Initially norm_device_function is None
    assert solver.compile_settings.norm_device_function is None

    # Call _update_norm_and_config with empty updates
    solver._update_norm_and_config({})

    # Verify config now has the norm device function
    assert solver.compile_settings.norm_device_function is not None
    assert solver.compile_settings.norm_device_function == \
        solver.norm.device_function

    # Test with actual tolerance updates
    new_atol = np.array([1e-10, 1e-10, 1e-10])
    solver._update_norm_and_config({'atol': new_atol})

    # Verify norm was updated and config still has device function
    assert np.array_equal(solver.norm.atol, new_atol)
    assert solver.compile_settings.norm_device_function is not None
