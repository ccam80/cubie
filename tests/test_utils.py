from unittest.mock import patch

import attrs
import numpy as np
import pytest
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from numpy import float32, float64

from cubie._utils import (
    clamp_factory,
    get_noise_32,
    get_noise_64,
    get_readonly_view,
    in_attr,
    is_attrs_class,
    is_devfunc,
    split_applicable_settings,
    round_list_sf,
    round_sf,
    slice_variable_dimension,
    timing,
)


def clamp_tester(fn, value, low_clip, high_clip, precision):
    out = cuda.device_array(1, dtype=precision)
    d_out = cuda.to_device(out)

    @cuda.jit()
    def clamp_test_kernel(d_value, d_low_clip, d_high_clip, dout):
        dout[0] = fn(d_value, d_low_clip, d_high_clip)

    clamp_test_kernel[1, 1](value, low_clip, high_clip, d_out)
    n_out = d_out.copy_to_host()
    return n_out


@pytest.mark.parametrize("precision", [float64])
def test_clamp_kernel_float64(precision):
    clamp_64 = clamp_factory(precision)
    out = clamp_tester(
        clamp_64,
        precision(-2.0),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == -1.0
    out = clamp_tester(
        clamp_64,
        precision(2.0),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == 1.0
    out = clamp_tester(
        clamp_64,
        precision(0.5),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == 0.5
    out = clamp_tester(
        clamp_64,
        precision(-0.5),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == -0.5


@pytest.mark.parametrize("precision", [float32])
def test_clamp_kernel_float32(precision):
    clamp_32 = clamp_factory(precision)
    out = clamp_tester(
        clamp_32,
        precision(-2.0),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == -1.0
    out = clamp_tester(
        clamp_32,
        precision(2.0),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == 1.0
    out = clamp_tester(
        clamp_32,
        precision(0.5),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == 0.5
    out = clamp_tester(
        clamp_32,
        precision(-0.5),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == -0.5


def noise_tester_64(sigmas, precision):
    """Test helper for get_noise_64 function."""
    n_elements = len(sigmas)
    noise_array = cuda.device_array(n_elements, dtype=precision)
    noise_array[:] = 0.0
    d_sigmas = cuda.to_device(np.array(sigmas, dtype=precision))

    # Create RNG state
    rng_states = create_xoroshiro128p_states(n_elements, seed=42)

    @cuda.jit()
    def noise_test_kernel(noise_arr, sig_arr, rng):
        idx = cuda.grid(1)
        if idx > n_elements:
            return
        if idx < noise_arr.size:
            get_noise_64(noise_arr, sig_arr, idx, rng)

    noise_test_kernel[1, 1](noise_array, d_sigmas, rng_states)
    return noise_array.copy_to_host()


def noise_tester_32(sigmas, precision):
    """Test helper for get_noise_32 function."""
    n_elements = len(sigmas)
    noise_array = cuda.device_array(n_elements, dtype=precision)
    noise_array[:] = 0.0
    d_sigmas = cuda.to_device(np.array(sigmas, dtype=precision))

    # Create RNG state
    rng_states = create_xoroshiro128p_states(n_elements, seed=42)

    @cuda.jit()
    def noise_test_kernel(noise_arr, sig_arr, rng):
        idx = cuda.grid(1)
        if idx > n_elements:
            return
        if idx < noise_arr.size:
            get_noise_32(noise_arr, sig_arr, idx, rng)

    noise_test_kernel[1, n_elements](noise_array, d_sigmas, rng_states)
    return noise_array.copy_to_host()


@pytest.mark.parametrize("precision", [float64])
def test_get_noise_64(precision):
    """Test get_noise_64 CUDA device function."""
    # Test with non-zero sigmas
    sigmas = [1.0, 2.0, 0.5]
    result = noise_tester_64(sigmas, precision)
    assert len(result) == 3
    # Results should be different (random) but finite
    assert all(np.isfinite(result))

    # Test with zero sigma
    sigmas_zero = [0.0, 1.0, 0.0]
    result_zero = noise_tester_64(sigmas_zero, precision)
    assert result_zero[0] == 0.0  # Should be exactly zero
    assert result_zero[2] == 0.0  # Should be exactly zero
    assert result_zero[1] != 0.0  # Should be non-zero


@pytest.mark.parametrize("precision", [float32])
def test_get_noise_32(precision):
    """Test get_noise_32 CUDA device function."""
    # Test with non-zero sigmas
    sigmas = [1.0, 2.0, 0.5]
    result = noise_tester_32(sigmas, precision)
    assert len(result) == 3
    # Results should be different (random) but finite
    assert all(np.isfinite(result))

    # Test with zero sigma
    sigmas_zero = [0.0, 1.0, 0.0]
    result_zero = noise_tester_32(sigmas_zero, precision)
    assert result_zero[0] == 0.0  # Should be exactly zero
    assert result_zero[2] == 0.0  # Should be exactly zero
    assert result_zero[1] != 0.0  # Should be non-zero


# Tests for regular Python functions


def test_slice_variable_dimension():
    """Test slice_variable_dimension function."""
    # Test basic functionality
    result = slice_variable_dimension(slice(1, 3), 0, 3)
    expected = (slice(1, 3), slice(None), slice(None))
    assert result == expected

    # Test multiple slices and indices
    slices = [slice(1, 3), slice(0, 2)]
    indices = [0, 2]
    result = slice_variable_dimension(slices, indices, 4)
    expected = (slice(1, 3), slice(None), slice(0, 2), slice(None))
    assert result == expected

    # Test single values converted to lists
    result = slice_variable_dimension(slice(1, 3), [0], 2)
    expected = (slice(1, 3), slice(None))
    assert result == expected

    # Test error cases
    with pytest.raises(
        ValueError, match="slices and indices must have the same length"
    ):
        slice_variable_dimension([slice(1, 3)], [0, 1], 3)

    with pytest.raises(ValueError, match="indices must be less than ndim"):
        slice_variable_dimension(slice(1, 3), 3, 3)


@attrs.define
class AttrsClasstest:
    field1: int
    _field2: str


class RegularClasstest:
    def __init__(self):
        self.field1 = 1


def test_in_attr():
    """Test in_attr function."""
    attrs_instance = AttrsClasstest(1, "test")

    # Test existing field
    assert in_attr("field1", attrs_instance) == True

    # Test existing private field (with underscore)
    assert in_attr("field2", attrs_instance) == True  # Should find _field2
    assert in_attr("_field2", attrs_instance) == True

    # Test non-existing field
    assert in_attr("nonexistent", attrs_instance) == False


def test_is_attrs_class():
    """Test is_attrs_class function."""
    attrs_instance = AttrsClasstest(1, "test")
    regular_instance = RegularClasstest()

    assert is_attrs_class(attrs_instance) == True
    assert is_attrs_class(regular_instance) == False
    assert is_attrs_class("string") == False
    assert is_attrs_class(42) == False


def test_split_applicable_settings_with_class():
    class Example:
        def __init__(self, required, optional=0):
            self.required = required
            self.optional = optional
    with pytest.warns(UserWarning):
        filtered, missing, unused = split_applicable_settings(
            Example,
            {"required": 1, "optional": 2, "ignored": 3},
        )
    assert filtered == {"required": 1, "optional": 2}
    assert missing == set()
    assert unused == {"ignored"}


def test_split_applicable_settings_with_function():
    def example(required, other=0):
        return required + other
    with pytest.warns(UserWarning):
        filtered, missing, unused = split_applicable_settings(
            example,
            {"other": 4, "extra": 5},
        )
    assert filtered == {"other": 4}
    assert missing == {"required"}
    assert unused == {"extra"}

def dummy_function():
    """Dummy function for timing tests."""
    return 42


def test_timing_decorator():
    """Test timing decorator."""
    # Test with default nruns
    with patch("builtins.print") as mock_print:
        decorated_func = timing(dummy_function)
        result = decorated_func()
        assert result == 42
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "dummy_function" in call_args
        assert "took:" in call_args

    # Test with specified nruns
    with patch("builtins.print") as mock_print:
        decorated_func = timing(nruns=3)(dummy_function)
        result = decorated_func()
        assert result == 42
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "over 3 runs" in call_args


def test_round_sf():
    """Test round_sf function."""
    # Test normal cases
    assert round_sf(123.456, 3) == 123.0
    assert round_sf(0.00123456, 3) == 0.00123
    assert round_sf(1234.56, 3) == 1230.0

    # Test edge cases
    assert round_sf(0.0, 3) == 0.0
    assert round_sf(-123.456, 3) == -123.0

    # Test single significant figure
    assert round_sf(123.456, 1) == 100.0


def test_round_list_sf():
    """Test round_list_sf function."""
    input_list = [123.456, 0.00123456, 1234.56, 0.0]
    result = round_list_sf(input_list, 3)
    expected = [123.0, 0.00123, 1230.0, 0.0]
    assert result == expected

    # Test empty list
    assert round_list_sf([], 3) == []


def test_get_readonly_view():
    """Test get_readonly_view function."""
    original = np.array([1, 2, 3, 4, 5])
    readonly = get_readonly_view(original)

    # Should be a view of the same data
    assert np.array_equal(readonly, original)
    assert readonly.base is original

    # Should be read-only
    assert not readonly.flags.writeable

    # Should raise error when trying to modify
    with pytest.raises(
        ValueError, match="assignment destination is read-only"
    ):
        readonly[0] = 10

    # Original should still be writable
    assert original.flags.writeable
    original[0] = 10
    assert original[0] == 10


def test_is_devfnc():
    """Test is_devfnc function."""

    @cuda.jit(device=True)
    def cuda_device_func(x, y):
        """A simple CUDA device function."""
        return x + y

    @cuda.jit(device=False)
    def cuda_kernel(x, y):
        """A regular Python function."""
        y = x

    def noncuda_func(x, y):
        """A regular Python function."""
        return x + y

    dev_is_device = is_devfunc(cuda_device_func)
    kernel_is_device = is_devfunc(cuda_kernel)
    noncuda_is_device = is_devfunc(noncuda_func)

    assert dev_is_device
    assert not kernel_is_device
    assert not noncuda_is_device