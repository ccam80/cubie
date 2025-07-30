import pytest
import numpy as np
from numpy import float32, float64, zeros, ones, array_equal
from CuMC.ForwardSim.BatchInputArrays import InputArrays
from CuMC.ForwardSim.OutputHandling.output_sizes import LoopBufferSizes

@pytest.fixture(scope="function")
def input_arrays():
    """Fixture to create a default InputArrays instance"""
    return InputArrays()

class TestInputArrays:
    """Test InputArrays class for device memory management and caching behavior"""

    def test_init_default(self):
        """Test default initialization creates zero arrays"""
        arrays = InputArrays()

        assert arrays.initial_values.shape == (1, 1, 1)
        assert arrays.parameters.shape == (1, 1, 1)
        assert arrays.forcing_vectors.shape == (1, 1, 1)
        assert arrays._precision == float32
        assert np.all(arrays.initial_values == 0)
        assert np.all(arrays.parameters == 0)
        assert np.all(arrays.forcing_vectors == 0)

    def test_init_with_arrays(self):
        """Test initialization with provided arrays"""
        init_vals = ones((2, 3, 4), dtype=float32)
        params = 2 * ones((2, 3, 5), dtype=float32)
        forcing = 3 * ones((2, 3, 6), dtype=float32)

        arrays = InputArrays(
            precision=float32,
            initial_values=init_vals,
            parameters=params,
            forcing_vectors=forcing
        )

        assert np.array_equal(arrays.initial_values, init_vals)
        assert np.array_equal(arrays.parameters, params)
        assert np.array_equal(arrays.forcing_vectors, forcing)

    def test_precision_setting(self):
        """Test different precision types"""
        arrays_f32 = InputArrays(precision=float32)
        arrays_f64 = InputArrays(precision=float64)

        assert arrays_f32._precision == float32
        assert arrays_f64._precision == float64
        assert arrays_f64.initial_values.dtype == float64

    def test_call_method_basic_update(self):
        """Test __call__ method updates arrays correctly"""
        arrays = InputArrays()

        new_init = ones((2, 3, 4), dtype=float32)
        new_params = 2 * ones((2, 3, 5), dtype=float32)
        new_forcing = 3 * ones((2, 3, 6), dtype=float32)

        arrays(new_init, new_params, new_forcing)

        assert np.array_equal(arrays.initial_values, new_init)
        assert np.array_equal(arrays.parameters, new_params)
        assert np.array_equal(arrays.forcing_vectors, new_forcing)

    def test_call_method_no_change(self):
        """Test __call__ method does nothing when arrays unchanged"""
        arrays = InputArrays()
        original_init = arrays.initial_values.copy()

        # Call with same arrays - should not trigger updates
        arrays(original_init, arrays.parameters, arrays.forcing_vectors)

        # Check no reallocation/overwrite lists populated
        assert len(arrays._needs_reallocation) == 0
        assert len(arrays._needs_overwrite) == 0

    def test_call_method_content_change(self):
        """Test __call__ method detects content changes"""
        arrays = InputArrays()
        original_shape = arrays.initial_values.shape

        # Same shape, different content
        new_init = ones(original_shape, dtype=float32)
        arrays(new_init, arrays.parameters, arrays.forcing_vectors)

        # Should trigger overwrite, not reallocation

        assert arrays._initial_values is new_init
        assert array_equal(arrays._device_inits.copy_to_host(), new_init)


    def test_arrays_equal_method(self):
        """Test _arrays_equal method handles various cases"""
        arrays = InputArrays()

        arr1 = ones((2, 3), dtype=float32)
        arr2 = ones((2, 3), dtype=float32)
        arr3 = zeros((2, 3), dtype=float32)
        arr4 = ones((2, 4), dtype=float32)

        assert arrays._arrays_equal(arr1, arr2)
        assert not arrays._arrays_equal(arr1, arr3)
        assert not arrays._arrays_equal(arr1, arr4)
        assert arrays._arrays_equal(None, None)
        assert not arrays._arrays_equal(arr1, None)

    def test_check_dims_vs_system_no_sizes(self):
        """Test dimension checking when no system sizes set"""
        arrays = InputArrays()
        arrays._sizes = None

        result = arrays._check_dims_vs_system(
            ones((2, 3, 4)), ones((2, 3, 5)), ones((2, 3, 6))
        )
        assert result  # Should return True when no sizes to check against

    def test_check_dims_vs_system_with_sizes(self):
        """Test dimension checking with system sizes"""
        # Mock LoopBufferSizes
        MockSizes = LoopBufferSizes(state = 4,
                                    parameters = 5,
                                    drivers = 6,
                                    )

        arrays = InputArrays()
        arrays._sizes = MockSizes

        # Correct dimensions
        result = arrays._check_dims_vs_system(
            ones((2, 3, 4)), ones((2, 3, 5)), ones((2, 3, 6))
        )
        assert result

        # Incorrect dimensions
        result = arrays._check_dims_vs_system(
            ones((2, 3, 999)), ones((2, 3, 5)), ones((2, 3, 6))
        )
        assert not result

    def test_num_runs_property(self):
        """Test num_runs property calculation"""
        arrays = InputArrays(
            precision=float32,
            initial_values=ones((2, 3, 4), dtype=float32),
            parameters=ones((5, 6, 7), dtype=float32)
        )

        assert arrays.num_runs == 2 * 5  # init_runs * param_runs

    def test_num_runs_property_none_arrays(self):
        """Test num_runs property with None arrays"""
        arrays = InputArrays()
        arrays._initial_values = None
        arrays._parameters = None

        assert arrays.num_runs == 0

    def test_to_device_clears_lists(self):
        """Test to_device method clears pending operation lists"""
        arrays = InputArrays()

        # Manually populate lists to simulate pending operations
        mock_array = ones((2, 3), dtype=float32)
        arrays._needs_reallocation = [(mock_array, arrays._device_inits)]
        arrays._needs_overwrite = [(mock_array, arrays._device_parameters)]

        arrays.to_device()

        assert len(arrays._needs_reallocation) == 0
        assert len(arrays._needs_overwrite) == 0

    @pytest.mark.parametrize("precision", [float32, float64])
    def test_different_precisions(self, precision):
        """Test InputArrays works with different precision types"""
        arrays = InputArrays(precision=precision)

        assert arrays._precision == precision
        assert arrays.initial_values.dtype == precision

    def test_properties_readonly(self):
        """Test that array properties are read-only"""
        arrays = InputArrays()
        init_vals = arrays.initial_values

        # Properties should return the same object
        assert arrays.initial_values is init_vals
        assert arrays.parameters is arrays._parameters
        assert arrays.forcing_vectors is arrays._forcing_vectors

    def test_integration_scenario(self):
        """Test complete integration scenario with multiple updates"""
        arrays = InputArrays()

        # First update - should trigger reallocation
        init1 = ones((2, 3, 4), dtype=float32)
        params1 = 2 * ones((2, 3, 5), dtype=float32)
        forcing1 = 3 * ones((2, 3, 6), dtype=float32)

        arrays(init1, params1, forcing1)

        # Verify arrays updated
        assert np.array_equal(arrays.initial_values, init1)
        assert np.array_equal(arrays.parameters, params1)
        assert np.array_equal(arrays.forcing_vectors, forcing1)

        # Second update - same shape, different content (should overwrite)
        init2 = 4 * ones((2, 3, 4), dtype=float32)
        arrays(init2, params1, forcing1)  # Only change initial values

        assert np.array_equal(arrays.initial_values, init2)

        # Third update - different shape (should reallocate)
        init3 = ones((5, 6, 4), dtype=float32)
        params3 = ones((5, 6, 5), dtype=float32)
        forcing3 = ones((5, 6, 6), dtype=float32)

        arrays(init3, params3, forcing3)

        assert arrays.initial_values.shape == (5, 6, 4)
        assert np.array_equal(arrays.initial_values, init3)

    def test_device_readback(self):
        arrays = InputArrays()

        init = ones((2, 3, 4), dtype=float32)
        params = 2 * ones((2, 3, 5), dtype=float32)
        forcing = 3 * ones((2, 3, 6), dtype=float32)

        arrays(init, params, forcing)

        assert array_equal(arrays._device_inits.copy_to_host(), init)
        assert array_equal(arrays._device_parameters.copy_to_host(), params)
        assert array_equal(arrays._device_forcing.copy_to_host() , forcing)

        init = ones((5, 6, 4), dtype=float32) * 5.0
        params = ones((5, 6, 5), dtype=float32) * 3.0
        forcing = ones((5, 6, 6), dtype=float32) * 2.0
        arrays(init, params, forcing)

        assert array_equal(arrays._device_inits.copy_to_host(), init)
        assert array_equal(arrays._device_parameters.copy_to_host(), params)
        assert array_equal(arrays._device_forcing.copy_to_host(), forcing)

        #reallocate:
        init = ones((5, 2, 4), dtype=float32) * 5.0
        params = ones((5, 2, 5), dtype=float32) * 3.0
        forcing = ones((5, 2, 6), dtype=float32) * 2.0
        arrays(init, params, forcing)

        assert array_equal(arrays._device_inits.copy_to_host(), init)
        assert array_equal(arrays._device_parameters.copy_to_host(), params)
        assert array_equal(arrays._device_forcing.copy_to_host(), forcing)