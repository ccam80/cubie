import pytest
import numpy as np
from numpy import float32, float64
from numba.cuda import mapped_array
from cubie.batchsolving.BatchOutputArrays import OutputArrays
from cubie.outputhandling.output_sizes import BatchOutputSizes
from warnings import catch_warnings
import attrs

@attrs.define
class IntegratorRunSettings:
    output_samples: int = attrs.field(default=1000, metadata={'description': 'Number of output samples'})
    summarise_samples: int = attrs.field(default=100, metadata={'description': 'Number of samples to summarise'})
    duration: float = attrs.field(default=10.0, metadata={'description': 'Total simulation duration'})

# Fixtures for run settings
@pytest.fixture(scope="function")
def run_settings_override(request):
    """Override for run settings, if provided."""
    return request.param if hasattr(request, 'param') else {}

@pytest.fixture(scope="function")
def run_settings(run_settings_override):
    """Create a dictionary of run settings for testing"""
    default_settings = IntegratorRunSettings()
    for key, value in run_settings_override.items():
        if hasattr(default_settings, key):
            setattr(default_settings, key, value)
    return default_settings

class TestOutputArrays:
    """Test OutputArrays class for allocation, caching, and array management"""

    def test_init_basic(self):
        """Test basic initialization with mock sizes"""
        # Create a proper BatchOutputSizes object
        sizes = BatchOutputSizes(
            state=(10, 5, 3),
            observables=(10, 5, 4),
            state_summaries=(5, 2, 1),
            observable_summaries=(5, 3, 1)
        )
        arrays = OutputArrays(sizes, precision=float32)

        assert arrays._sizes is sizes
        assert arrays._precision == float32
        assert arrays.state is None
        assert arrays.observables is None

    def test_allocate_new_method(self):
        """Test _allocate_new method creates correctly sized arrays"""
        sizes = BatchOutputSizes(
            state=(10, 5, 3),
            observables=(10, 5, 4),
            state_summaries=(5, 2, 1),
            observable_summaries=(5, 3, 1)
        )
        arrays = OutputArrays(sizes, precision=float32)
        arrays._allocate_new()

        assert arrays.state.shape == (10, 5, 3)
        assert arrays.observables.shape == (10, 5, 4)
        assert arrays.state_summaries.shape == (5, 2, 1)
        assert arrays.observable_summaries.shape == (5, 3, 1)
        assert arrays.state.dtype == float32

    def test_clear_cache_method(self):
        """Test _clear_cache method properly deallocates arrays"""
        sizes = BatchOutputSizes(
            state=(5, 3, 2),
            observables=(5, 3, 2),
            state_summaries=(3, 1, 1),
            observable_summaries=(3, 1, 1)
        )
        arrays = OutputArrays(sizes)
        arrays._allocate_new()

        # Verify arrays exist
        assert arrays.state is not None
        assert arrays.observables is not None

        arrays._clear_cache()

        # Arrays should be None after clearing (del sets them to None in this implementation)

    def test_check_dims_method(self):
        """Test _check_dims method correctly validates dimensions"""
        sizes = BatchOutputSizes(
            state=(10, 5, 3),
            observables=(10, 5, 4),
            state_summaries=(5, 2, 1),
            observable_summaries=(5, 3, 1)
        )
        arrays = OutputArrays(sizes)

        # Create mock arrays with correct dimensions
        state = np.zeros((10, 5, 3))
        observables = np.zeros((10, 5, 4))
        state_summaries = np.zeros((5, 2, 1))
        observable_summaries = np.zeros((5, 3, 1))

        result = arrays._check_dims(state, observables, state_summaries, observable_summaries, sizes)
        assert result

        # Test with incorrect dimensions
        wrong_state = np.zeros((10, 5, 999))
        result = arrays._check_dims(wrong_state, observables, state_summaries, observable_summaries, sizes)
        assert not result

    def test_check_type_method(self):
        """Test _check_type method validates array types"""
        sizes = BatchOutputSizes(
            state=(5, 3, 2),
            observables=(5, 3, 2),
            state_summaries=(3, 1, 1),
            observable_summaries=(3, 1, 1)
        )
        arrays = OutputArrays(sizes, precision=float32)

        # Create arrays with correct type
        state = np.zeros((5, 3, 2), dtype=float32)
        observables = np.zeros((5, 3, 2), dtype=float32)
        state_summaries = np.zeros((3, 1, 1), dtype=float32)
        observable_summaries = np.zeros((3, 1, 1), dtype=float32)

        result = arrays._check_type(state, observables, state_summaries, observable_summaries, float32)
        assert result

        # Test with incorrect type
        wrong_type = np.zeros((5, 3, 2), dtype=float64)
        result = arrays._check_type(wrong_type, observables, state_summaries, observable_summaries, float32)
        assert not result

    def test_check_type_none_precision(self):
        """Test _check_type method with None precision"""
        sizes = BatchOutputSizes(
            state=(5, 3, 2),
            observables=(5, 3, 2),
            state_summaries=(3, 1, 1),
            observable_summaries=(3, 1, 1)
        )
        arrays = OutputArrays(sizes)

        # Any arrays should pass when precision is None
        state = np.zeros((5, 3, 2), dtype=float64)
        observables = np.zeros((5, 3, 2), dtype=float32)
        state_summaries = np.zeros((3, 1, 1), dtype=float32)
        observable_summaries = np.zeros((3, 1, 1), dtype=float32)

        result = arrays._check_type(state, observables, state_summaries, observable_summaries, None)
        assert result

    def test_cache_valid_method(self):
        """Test cache_valid method combines dimension and type checking"""
        sizes = BatchOutputSizes(
            state=(5, 3, 2),
            observables=(5, 3, 2),
            state_summaries=(3, 1, 1),
            observable_summaries=(3, 1, 1)
        )
        arrays = OutputArrays(sizes, precision=float32)
        arrays._allocate_new()

        # Should be valid after allocation
        assert arrays.cache_valid()

        # Clear and reallocate
        arrays._clear_cache()
        arrays._allocate_new()

        # Should be different array objects
        assert arrays.state.shape == (5, 3, 2)

    def test_cache_valid_no_arrays(self):
        """Test cache_valid method when no arrays allocated"""
        sizes = BatchOutputSizes(
            state=(5, 3, 2),
            observables=(5, 3, 2),
            state_summaries=(3, 1, 1),
            observable_summaries=(3, 1, 1)
        )
        arrays = OutputArrays(sizes)

        # Should not be valid without arrays
        try:
            result = arrays.cache_valid()
            assert not result
        except AttributeError:
            # Expected if arrays are None
            pass

    def test_attach_method_valid_arrays(self):
        """Test attach method with valid external arrays"""
        sizes = BatchOutputSizes(
            state=(5, 3, 2),
            observables=(5, 3, 2),
            state_summaries=(3, 1, 1),
            observable_summaries=(3, 1, 1)
        )
        arrays = OutputArrays(sizes, precision=float32)
        arrays.allocate()
        arrays.state[:] = 1.0
        arrays.observables[:] = 1.0
        arrays.state_summaries[:] = 1.0
        arrays.observable_summaries[:] = 1.0

        # Create external arrays with correct dimensions and type
        ext_state = mapped_array(sizes.state, dtype=float32)
        ext_observables = mapped_array(sizes.observables, dtype=float32)
        ext_state_summaries = mapped_array(sizes.state_summaries, dtype=float32)
        ext_observable_summaries = mapped_array(sizes.observable_summaries, dtype=float32)
        ext_state[:] = 2.0
        ext_observables[:] = 2.0
        ext_state_summaries[:] = 2.0
        ext_observable_summaries[:] = 2.0
        arrays.attach(ext_state, ext_observables, ext_state_summaries, ext_observable_summaries)

        assert arrays.state is ext_state
        assert arrays.observables is ext_observables

    def test_attach_method_invalid_arrays(self):
        """Test attach method with invalid external arrays falls back to allocation"""
        sizes = BatchOutputSizes(
            state=(5, 3, 2),
            observables=(5, 3, 2),
            state_summaries=(3, 1, 1),
            observable_summaries=(3, 1, 1)
        )
        arrays = OutputArrays(sizes, precision=float32)

        # Create external arrays with wrong dimensions
        wrong_state = np.ones((999, 3, 2), dtype=float32)
        ext_observables = np.ones((5, 3, 2), dtype=float32)
        ext_state_summaries = np.ones((3, 1, 1), dtype=float32)
        ext_observable_summaries = np.ones((3, 1, 1), dtype=float32)

        with catch_warnings(record=True) as w:
            arrays.attach(wrong_state, ext_observables, ext_state_summaries, ext_observable_summaries)
            assert len(w) == 1
            assert "do not match the expected sizes" in str(w[0].message)

        # Should have allocated new arrays instead
        assert arrays.state.shape == (5, 3, 2)

    def test_initialize_zeros_method(self):
        """Test initialize_zeros method sets all arrays to zero"""
        sizes = BatchOutputSizes(
            state=(3, 2, 1),
            observables=(3, 2, 1),
            state_summaries=(2, 1, 1),
            observable_summaries=(2, 1, 1)
        )
        arrays = OutputArrays(sizes, precision=float32)
        arrays._allocate_new()

        # Set some non-zero values
        arrays.state[:] = 5.0
        arrays.observables[:] = 3.0

        arrays.initialize_zeros()

        assert np.allclose(arrays.state, 0.0)
        assert np.allclose(arrays.observables, 0.0)
        assert np.allclose(arrays.state_summaries, 0.0)
        assert np.allclose(arrays.observable_summaries, 0.0)

    @pytest.mark.parametrize("precision", [float32, float64])
    def test_different_precisions(self, precision):
        """Test OutputArrays works with different precision types"""
        sizes = BatchOutputSizes(
            state=(2, 2, 2),
            observables=(2, 2, 2),
            state_summaries=(2, 1, 1),
            observable_summaries=(2, 1, 1)
        )
        arrays = OutputArrays(sizes, precision=precision)
        arrays._allocate_new()

        assert arrays._precision == precision
        assert arrays.state.dtype == precision
    #
    # def test_summary_views_method(self):
    #     """Test summary_views method returns summary arrays split by type"""
    #     class DummySolver:
    #         class SingleIntegrator:
    #             summary_types = ['mean', 'max']
    #         single_integrator = SingleIntegrator()
    #     class DummySummaryMetrics:
    #         @staticmethod
    #         def output_sizes(types):
    #             return [1 for _ in types]
    #     # Patch summary_metrics in the OutputArrays module
    #     import sys
    #     mod = sys.modules[arrays.__class__.__module__]
    #     setattr(mod, 'summary_metrics', DummySummaryMetrics)
    #
    #     sizes = BatchOutputSizes(
    #         state=(5, 3, 2),
    #         observables=(5, 3, 2),
    #         state_summaries=(3, 1, 2),
    #         observable_summaries=(3, 1, 2)
    #     )
    #     arrays = OutputArrays(sizes)
    #     arrays._allocate_new()
    #     dummy_solver = DummySolver()
    #     state_splits, obs_splits = arrays.summary_views(dummy_solver)
    #     assert set(state_splits.keys()) == {'mean', 'max'}
    #     assert set(obs_splits.keys()) == {'mean', 'max'}
    #     assert state_splits['mean'].shape[-1] == 1
    #     assert state_splits['max'].shape[-1] == 1
    #
    # def test_legend_method(self):
    #     """Test legend method returns correct mapping for state and observable summaries"""
    #     class DummySolver:
    #         class SingleIntegrator:
    #             summary_types = ['mean', 'max']
    #         single_integrator = SingleIntegrator()
    #         class System:
    #             state_names = ['x', 'y']
    #             observable_names = ['a', 'b']
    #         system = System()
    #     class DummySummaryMetrics:
    #         @staticmethod
    #         def output_sizes(types):
    #             return [1 for _ in types]
    #     # Patch summary_metrics in the OutputArrays module
    #     import sys
    #     mod = sys.modules[OutputArrays.__module__]
    #     setattr(mod, 'summary_metrics', DummySummaryMetrics)
    #
    #     sizes = BatchOutputSizes(
    #         state=(5, 3, 2),
    #         observables=(5, 3, 2),
    #         state_summaries=(3, 1, 2),
    #         observable_summaries=(3, 1, 2)
    #     )
    #     arrays = OutputArrays(sizes)
    #     arrays._allocate_new()
    #     dummy_solver = DummySolver()
    #     legend = arrays.legend(dummy_solver, which='state_summaries')
    #     assert legend[0] == ('x', 'mean')
    #     assert legend[1] == ('y', 'mean')
    #     assert legend[2] == ('x', 'max')
    #     assert legend[3] == ('y', 'max')
    #     legend_obs = arrays.legend(dummy_solver, which='observable_summaries')
    #     assert legend_obs[0] == ('a', 'mean')
    #     assert legend_obs[1] == ('b', 'mean')
    #     assert legend_obs[2] == ('a', 'max')
    #     assert legend_obs[3] == ('b', 'max')

    def test_integration_scenario_allocation_reuse(self):
        """Test complete scenario with allocation and cache reuse"""
        sizes = BatchOutputSizes(
            state=(10, 5, 3),
            observables=(10, 5, 4),
            state_summaries=(5, 2, 1),
            observable_summaries=(5, 3, 1)
        )
        arrays = OutputArrays(sizes, precision=float32)

        # First allocation
        arrays._allocate_new()
        first_state = arrays.state

        # Initialize with some values
        arrays.initialize_zeros()
        arrays.state[0, 0, 0] = 42.0

        # Verify cache is valid
        assert arrays.cache_valid()

        # Clear and reallocate
        arrays._clear_cache()
        arrays._allocate_new()

        # Should be different array objects
        assert arrays.state is not first_state
        assert arrays.state.shape == (10, 5, 3)

    def test_allocate_method_with_invalid_cache(self):
        """Test allocate method behavior when cache is invalid"""
        sizes = BatchOutputSizes(
            state=(3, 2, 1),
            observables=(3, 2, 1),
            state_summaries=(2, 1, 1),
            observable_summaries=(2, 1, 1)
        )
        arrays = OutputArrays(sizes)

        # This should trigger allocation since cache is initially invalid
        arrays.allocate()

        assert arrays.state is not None
        assert np.allclose(arrays.state, 0.0)  # Should be initialized to zeros
