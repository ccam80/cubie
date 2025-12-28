from unittest.mock import patch

import attrs
import numpy as np
import pytest
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from numpy import float32

from cubie._utils import (
    build_config,
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
    unpack_dict_values,
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


def test_clamp_kernel_float64():
    precision = np.float64

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


def test_clamp_kernel_float32():
    precision = np.float32
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


def noise_tester_64(sigmas):
    precision = np.float64
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


def noise_tester_32(sigmas):
    """Test helper for get_noise_32 function."""
    precision=float32
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


def test_get_noise_64():
    """Test get_noise_64 CUDA device function."""
    # Test with non-zero sigmas
    sigmas = [1.0, 2.0, 0.5]
    result = noise_tester_64(sigmas)
    assert len(result) == 3
    # Results should be different (random) but finite
    assert all(np.isfinite(result))

    # Test with zero sigma
    sigmas_zero = [0.0, 1.0, 0.0]
    result_zero = noise_tester_64(sigmas_zero)
    assert result_zero[0] == 0.0  # Should be exactly zero
    assert result_zero[2] == 0.0  # Should be exactly zero
    assert result_zero[1] != 0.0  # Should be non-zero


def test_get_noise_32():
    """Test get_noise_32 CUDA device function."""
    # Test with non-zero sigmas
    sigmas = [1.0, 2.0, 0.5]
    result = noise_tester_32(sigmas)
    assert len(result) == 3
    # Results should be different (random) but finite
    assert all(np.isfinite(result))

    # Test with zero sigma
    sigmas_zero = [0.0, 1.0, 0.0]
    result_zero = noise_tester_32(sigmas_zero)
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


def test_split_applicable_settings_filters_none_values():
    """Verify split_applicable_settings filters out None values."""
    class Example:
        def __init__(self, required, optional=0, another=1):
            self.required = required
            self.optional = optional
            self.another = another

    filtered, missing, unused = split_applicable_settings(
        Example,
        {"required": 1, "optional": None, "another": 5},
        warn_on_unused=False,
    )
    # None value for 'optional' should be filtered out
    assert filtered == {"required": 1, "another": 5}
    assert "optional" not in filtered
    assert missing == set()


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


def test_unpack_dict_values_basic():
    """Test basic dict unpacking functionality."""
    # Basic unpacking: dict values are unpacked, regular values pass through
    input_dict = {
        'step_settings': {'dt_min': 0.01, 'dt_max': 1.0},
        'precision': np.float32
    }
    result, unpacked = unpack_dict_values(input_dict)
    
    assert result == {'dt_min': 0.01, 'dt_max': 1.0, 'precision': np.float32}
    assert unpacked == {'step_settings'}


def test_unpack_dict_values_mixed():
    """Test unpacking with mixed dict and non-dict values."""
    input_dict = {
        'controller': {'atol': 1e-5, 'rtol': 1e-3},
        'algorithm': 'rk4',
        'output': {'save_state': True}
    }
    result, unpacked = unpack_dict_values(input_dict)
    
    assert result == {
        'atol': 1e-5,
        'rtol': 1e-3,
        'algorithm': 'rk4',
        'save_state': True
    }
    assert unpacked == {'controller', 'output'}


def test_unpack_dict_values_empty():
    """Test unpacking with empty dict."""
    result, unpacked = unpack_dict_values({})
    assert result == {}
    assert unpacked == set()


def test_unpack_dict_values_empty_dict_value():
    """Test unpacking when a dict value is empty."""
    input_dict = {'settings': {}, 'value': 42}
    result, unpacked = unpack_dict_values(input_dict)
    
    assert result == {'value': 42}
    assert unpacked == {'settings'}


def test_unpack_dict_values_nested_dicts():
    """Test that only one level deep is unpacked."""
    # Nested dicts within dict values are NOT recursively unpacked
    input_dict = {
        'outer': {'inner': {'nested': 'value'}, 'regular': 5}
    }
    result, unpacked = unpack_dict_values(input_dict)
    
    # Should unpack outer, but leave inner as a dict
    assert result == {'inner': {'nested': 'value'}, 'regular': 5}
    assert unpacked == {'outer'}


def test_unpack_dict_values_no_dicts():
    """Test when there are no dict values to unpack."""
    input_dict = {'a': 1, 'b': 2, 'c': 'test'}
    result, unpacked = unpack_dict_values(input_dict)
    
    assert result == input_dict
    assert unpacked == set()


def test_unpack_dict_values_collision_regular_and_unpacked():
    """Test that key collision between regular entry and unpacked dict raises error."""
    # A key appears both as a regular entry and within an unpacked dict
    input_dict = {
        'dt_min': 0.001,
        'step_settings': {'dt_min': 0.01}
    }
    
    with pytest.raises(ValueError, match="Key collision detected.*dt_min"):
        unpack_dict_values(input_dict)


def test_unpack_dict_values_collision_multiple_unpacked():
    """Test that key collision between multiple unpacked dicts raises error."""
    # Same key appears in two different dict values
    input_dict = {
        'settings1': {'dt_min': 0.01},
        'settings2': {'dt_min': 0.02}
    }
    
    with pytest.raises(ValueError, match="Key collision detected.*dt_min"):
        unpack_dict_values(input_dict)


def test_unpack_dict_values_collision_duplicate_regular():
    """Test that duplicate regular keys raise error."""
    # This shouldn't happen in normal Python dict creation, but test the check
    # Note: Python dicts don't allow duplicate keys, so this tests the robustness
    # of our implementation. We'll test by processing in order.
    input_dict = {'a': 1, 'b': {'a': 2}}
    
    # Should raise error because 'a' appears in result, then 'b' unpacks 'a'
    with pytest.raises(ValueError, match="Key collision detected.*a"):
        unpack_dict_values(input_dict)


def test_unpack_dict_values_preserves_types():
    """Test that unpacking preserves various value types."""
    input_dict = {
        'settings': {
            'int_val': 42,
            'float_val': 3.14,
            'str_val': 'test',
            'bool_val': True,
            'none_val': None,
            'list_val': [1, 2, 3],
        }
    }
    result, unpacked = unpack_dict_values(input_dict)
    
    assert result['int_val'] == 42
    assert result['float_val'] == 3.14
    assert result['str_val'] == 'test'
    assert result['bool_val'] is True
    assert result['none_val'] is None
    assert result['list_val'] == [1, 2, 3]
    assert unpacked == {'settings'}


# =============================================================================
# Tests for build_config helper function
# =============================================================================


@attrs.define
class SimpleTestConfig:
    """Simple attrs config for testing build_config."""
    precision: type = attrs.field()
    n: int = attrs.field()
    optional_float: float = attrs.field(default=1.0)
    optional_str: str = attrs.field(default='default')


@attrs.define
class ConfigWithFactory:
    """Config with attrs.Factory default for testing build_config."""
    precision: type = attrs.field()
    data: dict = attrs.field(factory=dict)
    items: list = attrs.field(factory=list)


@attrs.define
class ConfigWithAlias:
    """Config with underscore-prefixed field (auto-aliased) for testing."""
    precision: type = attrs.field()
    _private_value: float = attrs.field(default=0.5, alias='private_value')


class TestBuildConfig:
    """Tests for build_config helper function."""

    def test_build_config_basic(self):
        """Verify basic config construction with required params only."""
        config = build_config(
            SimpleTestConfig,
            required={'precision': np.float32, 'n': 3},
        )
        assert config.precision == np.float32
        assert config.n == 3
        assert config.optional_float == 1.0
        assert config.optional_str == 'default'

    def test_build_config_optional_override(self):
        """Verify optional parameters override defaults."""
        config = build_config(
            SimpleTestConfig,
            required={'precision': np.float64, 'n': 5},
            optional_float=2.5,
            optional_str='custom',
        )
        assert config.precision == np.float64
        assert config.n == 5
        assert config.optional_float == 2.5
        assert config.optional_str == 'custom'

    def test_build_config_passes_values_directly(self):
        """Verify build_config passes all values directly to attrs.
        
        Note: None filtering happens upstream in split_applicable_settings.
        If None values reach build_config, they are passed through to attrs.
        """
        # This test verifies the pass-through behavior
        config = build_config(
            SimpleTestConfig,
            required={'precision': np.float32, 'n': 2},
            optional_float=3.14,
        )
        assert config.optional_float == 3.14
        assert config.optional_str == 'default'

    def test_build_config_attrs_handles_missing_required(self):
        """Verify attrs raises error on missing required fields."""
        with pytest.raises(TypeError):
            build_config(
                SimpleTestConfig,
                required={'precision': np.float32},
            )

    def test_build_config_extra_kwargs_ignored(self):
        """Verify extra kwargs are silently ignored."""
        config = build_config(
            SimpleTestConfig,
            required={'precision': np.float32, 'n': 4},
            extra_param='ignored',
            another_extra=123,
        )
        assert config.precision == np.float32
        assert config.n == 4
        assert not hasattr(config, 'extra_param')
        assert not hasattr(config, 'another_extra')

    def test_build_config_non_attrs_raises(self):
        """Verify TypeError for non-attrs class."""
        class RegularClass:
            def __init__(self, x):
                self.x = x

        with pytest.raises(TypeError, match="is not an attrs class"):
            build_config(
                RegularClass,
                required={'x': 1},
            )

    def test_build_config_factory_defaults(self):
        """Verify attrs.Factory defaults are handled correctly."""
        config = build_config(
            ConfigWithFactory,
            required={'precision': np.float32},
        )
        assert config.precision == np.float32
        assert config.data == {}
        assert config.items == []
        assert config.data is not ConfigWithFactory.__attrs_attrs__[1].default

    def test_build_config_factory_override(self):
        """Verify attrs.Factory defaults can be overridden."""
        config = build_config(
            ConfigWithFactory,
            required={'precision': np.float32},
            data={'key': 'value'},
            items=[1, 2, 3],
        )
        assert config.data == {'key': 'value'}
        assert config.items == [1, 2, 3]

    def test_build_config_alias_handling(self):
        """Verify underscore-prefixed fields with aliases work correctly."""
        config = build_config(
            ConfigWithAlias,
            required={'precision': np.float32},
            private_value=0.75,
        )
        assert config.precision == np.float32
        assert config._private_value == 0.75

    def test_build_config_alias_default(self):
        """Verify alias fields use defaults when not overridden."""
        config = build_config(
            ConfigWithAlias,
            required={'precision': np.float64},
        )
        assert config._private_value == 0.5

    def test_build_config_with_real_config_class(self):
        """Test build_config with actual cubie config class."""
        from cubie.integrators.step_control.fixed_step_controller import (
            FixedStepControlConfig
        )
        config = build_config(
            FixedStepControlConfig,
            required={'precision': np.float32, 'n': 3, 'dt': 0.01},
        )
        assert config.precision == np.float32
        assert config.n == 3

    def test_build_config_required_in_optional_overrides(self):
        """Verify required fields can also be in optional kwargs."""
        config = build_config(
            SimpleTestConfig,
            required={'precision': np.float32},
            n=7,
        )
        assert config.n == 7

    def test_build_config_empty_required(self):
        """Verify empty required dict works when all fields have defaults."""
        @attrs.define
        class AllOptionalConfig:
            value: int = attrs.field(default=42)
            name: str = attrs.field(default='test')

        config = build_config(
            AllOptionalConfig,
            required={},
        )
        assert config.value == 42
        assert config.name == 'test'
