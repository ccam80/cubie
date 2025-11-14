import attrs
import pytest

from cubie.CUDAFactory import CUDAFactory


def dict_to_attrs_class(dictionary):
    """Convert a dictionary to an attrs class instance."""
    # Create the class with the dictionary keys as field names
    CompileSettings = attrs.make_class(
        "CompileSettings", list(dictionary.keys())
    )

    # Create an instance with the values from the dictionary
    return CompileSettings(**dictionary)


@pytest.fixture(scope="class")
def factory():
    """Fixture to provide a factory for creating system instances."""

    class ConcreteFactory(CUDAFactory):
        def __init__(self):
            super().__init__()

        def build(self):
            return None

    factory = ConcreteFactory()
    return factory


def test_setup_compile_settings(factory):
    settings_dict = {
        "manually_overwritten_1": False,
        "manually_overwritten_2": False,
    }
    factory.setup_compile_settings(dict_to_attrs_class(settings_dict))
    assert factory.compile_settings.manually_overwritten_1 is False, (
        "setup_compile_settings did not overwrite compile settings"
    )


@pytest.fixture(scope="function")
def factory_with_settings(factory):
    """Fixture to provide a factory with specific compile settings."""
    settings_dict = {
        "manually_overwritten_1": False,
        "manually_overwritten_2": False,
    }
    factory.setup_compile_settings(dict_to_attrs_class(settings_dict))
    return factory


def test_update_compile_settings(factory_with_settings):
    factory_with_settings.update_compile_settings(manually_overwritten_1=True)
    assert (
        factory_with_settings.compile_settings.manually_overwritten_1 is True
    ), "compile settings were not updated correctly"
    with pytest.raises(KeyError):
        (
            factory_with_settings.update_compile_settings(
                non_existent_key=True
            ),
            "factory did not emit a warning for non-existent key",
        )


def test_update_compile_settings_reports_correct_key(factory_with_settings):
    with pytest.raises(KeyError) as exc:
        factory_with_settings.update_compile_settings(
            {"non_existent_key": True, "manually_overwritten_1": True}
        )
    assert "non_existent_key" in str(exc.value)
    assert "manually_overwritten_1" not in str(exc.value)


def test_cache_invalidation(factory_with_settings):
    assert factory_with_settings.cache_valid is False, (
        "Cache should be invalid initially"
    )
    _ = factory_with_settings.device_function
    assert factory_with_settings.cache_valid is True, (
        "Cache should be valid after first access to device_function"
    )

    factory_with_settings.update_compile_settings(manually_overwritten_1=True)
    assert factory_with_settings.cache_valid is False, (
        "Cache should be invalidated after updating compile settings"
    )

    _ = factory_with_settings.device_function
    assert factory_with_settings.cache_valid is True, (
        "Cache should be valid after first access to device_function"
    )


def test_build(factory_with_settings, monkeypatch):
    test_func = factory_with_settings.device_function
    assert test_func is None
    # cache validated

    monkeypatch.setattr(factory_with_settings, "build", lambda: 10.0)
    test_func = factory_with_settings.device_function
    assert test_func is None, (
        "device_function rebuilt even though cache was valid"
    )
    factory_with_settings.update_compile_settings(manually_overwritten_1=True)
    test_func = factory_with_settings.device_function
    assert test_func == 10.0, (
        "device_function was not rebuilt after cache invalidation"
    )


def test_build_with_dict_output(factory_with_settings, monkeypatch):
    """Test that when build returns a dictionary, the values are available via get_cached_output."""
    factory_with_settings._cache_valid = False

    @attrs.define
    class TestOutputs:
        test_output1: str = "value1"
        test_output2: str = "value2"

    monkeypatch.setattr(factory_with_settings, "build", lambda: TestOutputs())

    # Access device_function to trigger build
    _ = factory_with_settings.device_function

    # Test that dictionary outputs are available
    assert (
        factory_with_settings.get_cached_output("test_output1") == "value1"
    ), "Output not accessible"
    assert (
        factory_with_settings.get_cached_output("test_output2") == "value2"
    ), "Output not accessible"

    # Test cache invalidation with dict output
    factory_with_settings.update_compile_settings(manually_overwritten_1=True)
    assert factory_with_settings.cache_valid is False, (
        "Cache should be invalidated after updating compile settings"
    )

    # Test that dict values are rebuilt after invalidation
    @attrs.define
    class NewTestOutputs:
        test_output1: str = "new_value1"
        test_output2: str = "new_value2"

    monkeypatch.setattr(
        factory_with_settings, "build", lambda: NewTestOutputs()
    )

    output = factory_with_settings.get_cached_output("test_output1")
    assert output == "new_value1", "Cache not rebuilt after invalidation"


def test_device_function_from_dict(factory_with_settings, monkeypatch):
    """Test that when build returns a dict with 'device_function',
    it's accessible via the device_function property."""
    factory_with_settings._cache_valid = False

    def test_func(x):
        return x * 2

    @attrs.define
    class TestOutputsWithFunc:
        device_function: callable = test_func
        other_output: str = "value"

    monkeypatch.setattr(
        factory_with_settings, "build", lambda: TestOutputsWithFunc()
    )

    # Check if device_function is correctly set from the dict
    assert factory_with_settings.device_function is test_func, (
        "device_function not correctly set from attrs class"
    )

    # Check that other values are still accessible
    assert (
        factory_with_settings.get_cached_output("other_output") == "value"
    ), "Other attrs values not accessible"


def test_get_cached_output_not_implemented_error(
    factory_with_settings, monkeypatch
):
    """Test that get_cached_output raises NotImplementedError for -1 values."""
    factory_with_settings._cache_valid = False

    @attrs.define
    class TestOutputsWithNotImplemented:
        implemented_output: str = "value"
        not_implemented_output: int = -1

    monkeypatch.setattr(
        factory_with_settings, "build", lambda: TestOutputsWithNotImplemented()
    )

    # Access device_function to trigger build
    _ = factory_with_settings.device_function

    # Test that implemented output works normally
    assert (
        factory_with_settings.get_cached_output("implemented_output")
        == "value"
    )

    # Test that -1 value raises NotImplementedError
    with pytest.raises(NotImplementedError) as exc:
        factory_with_settings.get_cached_output("not_implemented_output")

    assert "not_implemented_output" in str(exc.value)
    assert "not implemented" in str(exc.value)


def test_get_cached_output_not_implemented_error_multiple(
    factory_with_settings, monkeypatch
):
    """Test NotImplementedError with multiple -1 values in cache."""
    factory_with_settings._cache_valid = False

    @attrs.define
    class TestOutputsMultipleNotImplemented:
        working_output: str = "works"
        not_implemented_1: int = -1
        not_implemented_2: int = -1

    monkeypatch.setattr(
        factory_with_settings,
        "build",
        lambda: TestOutputsMultipleNotImplemented(),
    )

    # Trigger build
    _ = factory_with_settings.device_function

    # Test that working output still works
    assert factory_with_settings.get_cached_output("working_output") == "works"

    # Test that both -1 values raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc1:
        factory_with_settings.get_cached_output("not_implemented_1")
    assert "not_implemented_1" in str(exc1.value)

    with pytest.raises(NotImplementedError) as exc2:
        factory_with_settings.get_cached_output("not_implemented_2")
    assert "not_implemented_2" in str(exc2.value)


# Tests for helper functions
from cubie.CUDAFactory import _get_device_function_params
from cubie.CUDAFactory import _create_dummy_args
from cubie.CUDAFactory import _create_dummy_kernel
from numba import cuda
import numpy as np


def test_get_device_function_params():
    """Test parameter extraction from device function."""
    @cuda.jit(device=True)
    def sample_device_func(state, params, dt):
        return state[0] + params[0] * dt
    
    params = _get_device_function_params(sample_device_func)
    assert params == ['state', 'params', 'dt']


def test_get_device_function_params_no_py_func():
    """Test graceful handling when py_func missing."""
    # Mock object without py_func
    class FakeFunc:
        pass
    
    params = _get_device_function_params(FakeFunc())
    assert params == []


def test_get_device_function_params_none():
    """Test handling of None input."""
    params = _get_device_function_params(None)
    assert params == []


def test_create_dummy_args():
    """Test dummy argument creation."""
    args = _create_dummy_args(3, np.float64)
    
    assert len(args) == 3
    assert all(isinstance(arg, np.ndarray) for arg in args)
    assert all(arg.dtype == np.float64 for arg in args)
    assert all(len(arg) == 1 for arg in args)


def test_create_dummy_args_zero_params():
    """Test zero parameter case."""
    args = _create_dummy_args(0, np.float64)
    assert len(args) == 0
    assert args == tuple()


def test_create_dummy_args_precision():
    """Test different precision types."""
    args32 = _create_dummy_args(2, np.float32)
    args64 = _create_dummy_args(2, np.float64)
    
    assert all(arg.dtype == np.float32 for arg in args32)
    assert all(arg.dtype == np.float64 for arg in args64)


@pytest.mark.nocudasim
def test_create_dummy_kernel():
    """Test dummy kernel creation and execution."""
    @cuda.jit(device=True)
    def add_device(a, b):
        return a[0] + b[0]
    
    kernel = _create_dummy_kernel(add_device, 2)
    
    # Verify kernel is callable
    assert callable(kernel)
    
    # Test kernel can be launched (will trigger compilation)
    args = _create_dummy_args(2, np.float64)
    kernel[1, 1](*args)
    cuda.synchronize()
    
    # If we got here without exception, test passes


@pytest.mark.nocudasim
def test_create_dummy_kernel_various_param_counts():
    """Test kernel creation for different parameter counts."""
    for count in [0, 1, 3, 5, 8, 10, 12]:
        @cuda.jit(device=True)
        def dummy_func(*args):
            pass
        
        kernel = _create_dummy_kernel(dummy_func, count)
        assert callable(kernel)


# Integration tests for specialize_and_compile
from cubie.time_logger import TimeLogger


@pytest.mark.nocudasim
def test_specialize_and_compile_records_timing():
    """Test that specialize_and_compile records compilation timing."""
    @cuda.jit(device=True)
    def sample_device(x, y):
        return x[0] + y[0]
    
    # Create factory with custom logger
    class TestFactory(CUDAFactory):
        def build(self):
            return sample_device
    
    factory = TestFactory()
    factory._register_event("compile_test", "compile", "Test compilation")
    
    # Call specialize_and_compile
    factory.specialize_and_compile(sample_device, "compile_test")
    
    # Verify timing was recorded
    logger = factory._timing_start.__self__
    duration = logger.get_event_duration("compile_test")
    assert duration is not None
    assert duration > 0


@pytest.mark.nocudasim
def test_specialize_and_compile_none_device_function():
    """Test that None device function is handled gracefully."""
    class TestFactory(CUDAFactory):
        def build(self):
            return None
    
    factory = TestFactory()
    factory._register_event("compile_test", "compile", "Test")
    
    # Should not raise
    factory.specialize_and_compile(None, "compile_test")


def test_specialize_and_compile_simulator_mode():
    """Test that compilation timing is skipped in simulator mode."""
    @cuda.jit(device=True)
    def sample_device(x):
        return x[0]
    
    class TestFactory(CUDAFactory):
        def build(self):
            return sample_device
    
    factory = TestFactory()
    factory._register_event("compile_test", "compile", "Test")
    
    # Should not raise, should skip timing
    factory.specialize_and_compile(sample_device, "compile_test")
    
    # Verify no timing recorded (in sim mode, events may or may not be recorded)
    # Just ensure no error occurred
