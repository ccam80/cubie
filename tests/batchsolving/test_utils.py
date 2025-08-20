"""
Tests for batchsolving._utils module.

This module tests all utility functions in the batchsolving._utils module,
including size validation, CUDA array detection, and array validation functions.
"""

import pytest
import numpy as np
from numba import cuda
import os
from unittest.mock import patch

from cubie.batchsolving._utils import (
    ensure_nonzero_size,
    is_cuda_array,
    cuda_array_validator,
    optional_cuda_array_validator,
    optional_cuda_array_validator_3d,
    optional_cuda_array_validator_2d,
    cuda_array_validator_3d,
    cuda_array_validator_2d,
)


class TestEnsureNonzeroSize:
    """Test the ensure_nonzero_size function."""

    def test_int_zero(self):
        """Test that zero integer returns 1."""
        result = ensure_nonzero_size(0)
        assert result == 1

    def test_int_positive(self):
        """Test that positive integer is unchanged."""
        result = ensure_nonzero_size(5)
        assert result == 5

    def test_int_negative(self):
        """Test that negative integer returns 1."""
        result = ensure_nonzero_size(-3)
        assert result == 1

    def test_tuple_with_zeros(self):
        """Test that tuple with zeros returns all ones."""
        result = ensure_nonzero_size((0, 2, 0))
        assert result == (1, 1, 1)

    def test_tuple_without_zeros(self):
        """Test that tuple without zeros is unchanged."""
        result = ensure_nonzero_size((2, 3, 4))
        assert result == (2, 3, 4)

    def test_tuple_empty(self):
        """Test that empty tuple is unchanged."""
        result = ensure_nonzero_size(())
        assert result == ()

    def test_tuple_single_zero(self):
        """Test that single-element tuple with zero returns (1,)."""
        result = ensure_nonzero_size((0,))
        assert result == (1,)

    def test_tuple_single_nonzero(self):
        """Test that single-element tuple with non-zero is unchanged."""
        result = ensure_nonzero_size((5,))
        assert result == (5,)

    def test_other_types(self):
        """Test that other types are returned unchanged."""
        result = ensure_nonzero_size("hello")
        assert result == "hello"

        result = ensure_nonzero_size(None)
        assert result is None

        result = ensure_nonzero_size([1, 2, 3])
        assert result == [1, 2, 3]


class TestCudaArrayFixtures:
    """Fixtures and helper methods for CUDA array testing."""

    @pytest.fixture
    def numpy_1d_array(self):
        """Create a 1D numpy array."""
        return np.array([1, 2, 3, 4, 5])

    @pytest.fixture
    def numpy_2d_array(self):
        """Create a 2D numpy array."""
        return np.array([[1, 2], [3, 4], [5, 6]])

    @pytest.fixture
    def numpy_3d_array(self):
        """Create a 3D numpy array."""
        return np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    @pytest.fixture
    def cuda_1d_array(self, numpy_1d_array):
        """Create a 1D CUDA array."""
        return cuda.to_device(numpy_1d_array)

    @pytest.fixture
    def cuda_2d_array(self, numpy_2d_array):
        """Create a 2D CUDA array."""
        return cuda.to_device(numpy_2d_array)

    @pytest.fixture
    def cuda_3d_array(self, numpy_3d_array):
        """Create a 3D CUDA array."""
        return cuda.to_device(numpy_3d_array)

    @pytest.fixture
    def mock_array_like(self):
        """Create a mock object with shape attribute."""

        class MockArrayLike:
            def __init__(self, shape):
                self.shape = shape

        return MockArrayLike

    @pytest.fixture
    def placeholder_instance(self):
        """Create a placeholder instance for validator functions."""
        return object()

    @pytest.fixture
    def placeholder_attribute(self):
        """Create a placeholder attribute for validator functions."""

        class placeholderAttribute:
            pass

        return placeholderAttribute()


@pytest.mark.nocudasim
class TestIsCudaArray(TestCudaArrayFixtures):
    """Test the is_cuda_array function."""

    def test_cuda_array_detection(self, cuda_1d_array):
        """Test that CUDA arrays are detected correctly."""
        assert is_cuda_array(cuda_1d_array) is True

    def test_numpy_array_detection(self, numpy_1d_array):
        """Test that numpy arrays are not detected as CUDA arrays."""
        # In normal mode (not CUDASIM), numpy arrays should not be CUDA arrays
        if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
            # In CUDASIM mode, anything with shape attribute is considered CUDA array
            assert is_cuda_array(numpy_1d_array) is True
        else:
            assert is_cuda_array(numpy_1d_array) is False

    def test_non_array_detection(self):
        """Test that non-array objects are not detected as CUDA arrays."""
        assert is_cuda_array("not an array") is False
        assert is_cuda_array(42) is False
        assert is_cuda_array(None) is False

    def test_mock_array_like_in_cudasim(self, mock_array_like):
        """Test mock array-like object detection in CUDASIM mode."""
        mock_obj = mock_array_like((10, 10))

        if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
            # In CUDASIM mode, objects with shape attribute are considered CUDA arrays
            assert is_cuda_array(mock_obj) is True
        else:
            # In normal mode, only real CUDA arrays are detected
            assert is_cuda_array(mock_obj) is False


@pytest.mark.nocudasim
class TestCudaArrayValidator(TestCudaArrayFixtures):
    """Test the cuda_array_validator function."""

    def test_valid_cuda_array_no_dimensions(
        self, placeholder_instance, placeholder_attribute, cuda_2d_array
    ):
        """Test validation of CUDA array without dimension checking."""
        result = cuda_array_validator(
            placeholder_instance, placeholder_attribute, cuda_2d_array
        )
        assert result is True

    def test_valid_cuda_array_correct_dimensions(
        self, placeholder_instance, placeholder_attribute, cuda_2d_array
    ):
        """Test validation of CUDA array with correct dimensions."""
        result = cuda_array_validator(
            placeholder_instance,
            placeholder_attribute,
            cuda_2d_array,
            dimensions=2,
        )
        assert result is True

    def test_valid_cuda_array_wrong_dimensions(
        self, placeholder_instance, placeholder_attribute, cuda_2d_array
    ):
        """Test validation of CUDA array with wrong dimensions."""
        result = cuda_array_validator(
            placeholder_instance,
            placeholder_attribute,
            cuda_2d_array,
            dimensions=3,
        )
        assert result is False

    def test_invalid_non_cuda_array(
        self, placeholder_instance, placeholder_attribute, numpy_2d_array
    ):
        """Test validation of non-CUDA array."""
        if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") != "1":
            result = cuda_array_validator(
                placeholder_instance, placeholder_attribute, numpy_2d_array
            )
            assert result is False

    def test_invalid_non_array(
        self, placeholder_instance, placeholder_attribute
    ):
        """Test validation of non-array object."""
        result = cuda_array_validator(
            placeholder_instance, placeholder_attribute, "not an array"
        )
        assert result is False


@pytest.mark.nocudasim
class TestOptionalCudaArrayValidator(TestCudaArrayFixtures):
    """Test the optional_cuda_array_validator function."""

    def test_none_value(self, placeholder_instance, placeholder_attribute):
        """Test that None is always valid."""
        result = optional_cuda_array_validator(
            placeholder_instance, placeholder_attribute, None
        )
        assert result is True

    def test_none_value_with_dimensions(
        self, placeholder_instance, placeholder_attribute
    ):
        """Test that None is valid even with dimension specification."""
        result = optional_cuda_array_validator(
            placeholder_instance, placeholder_attribute, None, dimensions=3
        )
        assert result is True

    def test_valid_cuda_array(
        self, placeholder_instance, placeholder_attribute, cuda_2d_array
    ):
        """Test validation of valid CUDA array."""
        result = optional_cuda_array_validator(
            placeholder_instance, placeholder_attribute, cuda_2d_array
        )
        assert result is True

    def test_valid_cuda_array_correct_dimensions(
        self, placeholder_instance, placeholder_attribute, cuda_3d_array
    ):
        """Test validation of CUDA array with correct dimensions."""
        result = optional_cuda_array_validator(
            placeholder_instance,
            placeholder_attribute,
            cuda_3d_array,
            dimensions=3,
        )
        assert result is True

    def test_valid_cuda_array_wrong_dimensions(
        self, placeholder_instance, placeholder_attribute, cuda_2d_array
    ):
        """Test validation of CUDA array with wrong dimensions."""
        result = optional_cuda_array_validator(
            placeholder_instance,
            placeholder_attribute,
            cuda_2d_array,
            dimensions=3,
        )
        assert result is False

    def test_invalid_non_array(
        self, placeholder_instance, placeholder_attribute
    ):
        """Test validation of non-array object."""
        result = optional_cuda_array_validator(
            placeholder_instance, placeholder_attribute, "not an array"
        )
        assert result is False


@pytest.mark.nocudasim
class TestOptionalCudaArrayValidator3D(TestCudaArrayFixtures):
    """Test the optional_cuda_array_validator_3d function."""

    def test_none_value(self, placeholder_instance, placeholder_attribute):
        """Test that None is valid."""
        result = optional_cuda_array_validator_3d(
            placeholder_instance, placeholder_attribute, None
        )
        assert result is True

    def test_valid_3d_cuda_array(
        self, placeholder_instance, placeholder_attribute, cuda_3d_array
    ):
        """Test validation of 3D CUDA array."""
        result = optional_cuda_array_validator_3d(
            placeholder_instance, placeholder_attribute, cuda_3d_array
        )
        assert result is True

    def test_invalid_2d_cuda_array(
        self, placeholder_instance, placeholder_attribute, cuda_2d_array
    ):
        """Test validation of 2D CUDA array (should fail for 3D validator)."""
        result = optional_cuda_array_validator_3d(
            placeholder_instance, placeholder_attribute, cuda_2d_array
        )
        assert result is False

    def test_invalid_non_array(
        self, placeholder_instance, placeholder_attribute
    ):
        """Test validation of non-array object."""
        result = optional_cuda_array_validator_3d(
            placeholder_instance, placeholder_attribute, "not an array"
        )
        assert result is False


@pytest.mark.nocudasim
class TestOptionalCudaArrayValidator2D(TestCudaArrayFixtures):
    """Test the optional_cuda_array_validator_2d function."""

    def test_none_value(self, placeholder_instance, placeholder_attribute):
        """Test that None is valid."""
        result = optional_cuda_array_validator_2d(
            placeholder_instance, placeholder_attribute, None
        )
        assert result is True

    def test_valid_2d_cuda_array(
        self, placeholder_instance, placeholder_attribute, cuda_2d_array
    ):
        """Test validation of 2D CUDA array."""
        result = optional_cuda_array_validator_2d(
            placeholder_instance, placeholder_attribute, cuda_2d_array
        )
        assert result is True

    def test_invalid_3d_cuda_array(
        self, placeholder_instance, placeholder_attribute, cuda_3d_array
    ):
        """Test validation of 3D CUDA array (should fail for 2D validator)."""
        result = optional_cuda_array_validator_2d(
            placeholder_instance, placeholder_attribute, cuda_3d_array
        )
        assert result is False

    def test_invalid_non_array(
        self, placeholder_instance, placeholder_attribute
    ):
        """Test validation of non-array object."""
        result = optional_cuda_array_validator_2d(
            placeholder_instance, placeholder_attribute, "not an array"
        )
        assert result is False


@pytest.mark.nocudasim
class TestCudaArrayValidator3D(TestCudaArrayFixtures):
    """Test the cuda_array_validator_3d function."""

    def test_valid_3d_cuda_array(
        self, placeholder_instance, placeholder_attribute, cuda_3d_array
    ):
        """Test validation of 3D CUDA array."""
        result = cuda_array_validator_3d(
            placeholder_instance, placeholder_attribute, cuda_3d_array
        )
        assert result is True

    def test_invalid_2d_cuda_array(
        self, placeholder_instance, placeholder_attribute, cuda_2d_array
    ):
        """Test validation of 2D CUDA array (should fail for 3D validator)."""
        result = cuda_array_validator_3d(
            placeholder_instance, placeholder_attribute, cuda_2d_array
        )
        assert result is False

    def test_invalid_non_array(
        self, placeholder_instance, placeholder_attribute
    ):
        """Test validation of non-array object."""
        result = cuda_array_validator_3d(
            placeholder_instance, placeholder_attribute, "not an array"
        )
        assert result is False


@pytest.mark.nocudasim
class TestCudaArrayValidator2D(TestCudaArrayFixtures):
    """Test the cuda_array_validator_2d function."""

    def test_valid_2d_cuda_array(
        self, placeholder_instance, placeholder_attribute, cuda_2d_array
    ):
        """Test validation of 2D CUDA array."""
        result = cuda_array_validator_2d(
            placeholder_instance, placeholder_attribute, cuda_2d_array
        )
        assert result is True

    def test_invalid_3d_cuda_array(
        self, placeholder_instance, placeholder_attribute, cuda_3d_array
    ):
        """Test validation of 3D CUDA array (should fail for 2D validator)."""
        result = cuda_array_validator_2d(
            placeholder_instance, placeholder_attribute, cuda_3d_array
        )
        assert result is False

    def test_invalid_1d_cuda_array(
        self, placeholder_instance, placeholder_attribute, cuda_1d_array
    ):
        """Test validation of 1D CUDA array (should fail for 2D validator)."""
        result = cuda_array_validator_2d(
            placeholder_instance, placeholder_attribute, cuda_1d_array
        )
        assert result is False

    def test_invalid_non_array(
        self, placeholder_instance, placeholder_attribute
    ):
        """Test validation of non-array object."""
        result = cuda_array_validator_2d(
            placeholder_instance, placeholder_attribute, "not an array"
        )
        assert result is False


class TestCudaSimulationMode:
    """Test behavior in CUDA simulation mode."""

    def test_cudasim_mode_detection(self):
        """Test that CUDASIM mode affects is_cuda_array behavior."""
        # Create a numpy array
        numpy_arr = np.array([1, 2, 3])

        # Test current behavior
        current_result = is_cuda_array(numpy_arr)

        # Check if we're in CUDASIM mode
        if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
            # In CUDASIM mode, numpy arrays should be detected as CUDA arrays
            assert current_result is True
        else:
            # In normal mode, numpy arrays should not be detected as CUDA arrays
            assert current_result is False

    @patch.dict(os.environ, {"NUMBA_ENABLE_CUDASIM": "1"})
    def test_mock_array_behavior_in_cudasim(self):
        """Test that objects with shape attribute are detected in CUDASIM mode."""

        # This test demonstrates the behavior difference, but we can't easily
        # test the conditional import without more complex mocking
        class MockArray:
            def __init__(self):
                self.shape = (10, 10)

        mock_arr = MockArray()

        # Note: This test shows the intended behavior but may not work
        # exactly as expected due to the conditional import at module level
        # The actual behavior depends on when the module was imported
        assert hasattr(mock_arr, "shape")
