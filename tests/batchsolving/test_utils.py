"""
Tests for batchsolving._utils module.

This module tests all utility functions in the batchsolving._utils module,
including size validation, CUDA array detection, and array validation functions.
"""

import pytest
import numpy as np
from numba import cuda
import os

from cubie.cuda_simsafe import is_cuda_array
from cubie._utils import ensure_nonzero_size


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


class TestIsCudaArray:
    """Test the is_cuda_array function."""

    @pytest.fixture
    def numpy_1d_array(self):
        """Create a 1D numpy array."""
        return np.array([1, 2, 3, 4, 5])

    @pytest.fixture
    def cuda_1d_array(self, numpy_1d_array):
        """Create a 1D CUDA array."""
        return cuda.to_device(numpy_1d_array)

    @pytest.fixture
    def mock_array_like(self):
        """Create a mock object with shape attribute."""

        class MockArrayLike:
            def __init__(self, shape):
                self.shape = shape

        return MockArrayLike

    def test_cuda_array_detection(self, cuda_1d_array):
        """Test that CUDA arrays are detected correctly."""
        assert is_cuda_array(cuda_1d_array) is True

    def test_numpy_array_detection(self, numpy_1d_array):
        """Test that numpy arrays are not detected as CUDA arrays."""
        if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
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
            assert is_cuda_array(mock_obj) is True
        else:
            assert is_cuda_array(mock_obj) is False
