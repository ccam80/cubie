from typing import Callable, Union

import attrs
import numba
import pytest
from numba import cuda
import numpy as np
from cubie.cuda_simsafe import DeviceNDArray

from cubie.CUDAFactory import CUDAFactory, CUDAFunctionCache
from cubie.CUDAFactory import _create_placeholder_args
from cubie.CUDAFactory import _run_placeholder_kernel
from cubie.time_logger import TimeLogger


@attrs.define()
class testCache(CUDAFunctionCache):
    """Test cache class."""
    device_function: Union[Callable, int] = attrs.field(default=-1)

def dict_to_attrs_class(dictionary):
    """Convert a dictionary to an attrs class instance."""
    # Create the class with the dictionary keys as field names
    CompileSettings = attrs.make_class(
        "CompileSettings", list(dictionary.keys())
    )

    # Create an instance with the values from the dictionary
    return CompileSettings(**dictionary)


@pytest.fixture(scope="function")
def factory():
    """Fixture to provide a factory for creating system instances."""

    class ConcreteFactory(CUDAFactory):
        def __init__(self):
            super().__init__()

        def build(self):
            return testCache(device_function=lambda: 20.0)

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
def factory_with_settings(factory, precision):
    """Fixture to provide a factory with specific compile settings."""
    settings_dict = {
        "precision": precision,
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
    assert test_func() == 20.0, "device_function not as defined"
    # cache validated

    monkeypatch.setattr(factory_with_settings, "build",
                        lambda: testCache(device_function= lambda: 10.0)
    )
    test_func = factory_with_settings.device_function
    assert test_func() == 20.0, (
        "device_function rebuilt even though cache was valid"
    )
    factory_with_settings.update_compile_settings(manually_overwritten_1=True)
    test_func = factory_with_settings.device_function
    assert test_func() == 10.0, (
        "device_function was not rebuilt after cache invalidation"
    )


def test_build_with_dict_output(factory_with_settings, monkeypatch):
    """Test that when build returns a dictionary, the values are available via get_cached_output."""
    factory_with_settings._cache_valid = False

    @attrs.define
    class TestOutputs(testCache):
        test_output1: str = "value1"
        test_output2: str = "value2"

    monkeypatch.setattr(factory_with_settings, "build",
                        lambda: (TestOutputs())
                        )

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
    class NewTestOutputs(testCache):
        test_output1: str = "new_value1"
        test_output2: str = "new_value2"

    monkeypatch.setattr(
        factory_with_settings, "build", lambda: (NewTestOutputs())
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
    class TestOutputsWithFunc(testCache):
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
    class TestOutputsWithNotImplemented(testCache):
        implemented_output: str = "value"
        not_implemented_output: int = -1

    monkeypatch.setattr(
        factory_with_settings, "build", lambda: TestOutputsWithNotImplemented()
    )

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
    class TestOutputsMultipleNotImplemented(testCache):
        working_output: str = "works"
        not_implemented_1: int = -1
        not_implemented_2: int = -1

    monkeypatch.setattr(
        factory_with_settings,
        "build",
        lambda: TestOutputsMultipleNotImplemented(),
    )
    # Test that working output still works
    assert factory_with_settings.get_cached_output("working_output") == "works"

    # Test that both -1 values raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc1:
        factory_with_settings.get_cached_output("not_implemented_1")
    assert "not_implemented_1" in str(exc1.value)

    with pytest.raises(NotImplementedError) as exc2:
        factory_with_settings.get_cached_output("not_implemented_2")
    assert "not_implemented_2" in str(exc2.value)

@pytest.mark.nocudasim
def test_create_placeholder_args():
    """Test placeholder argument creation."""

    @cuda.jit(
                [numba.float32(numba.float32[:],
                  numba.float32[:],
                  numba.float32[:])],
                device=True)
    def device_func(a, b, c):
        return a[0] + b[0] + c[0]

    args = _create_placeholder_args(device_func, np.float64)
    args = args[0]
    assert len(args) == 3
    assert all(isinstance(arg, DeviceNDArray) for arg in args)
    assert all(arg.dtype == np.float32 for arg in args)

@pytest.mark.nocudasim
def test_create_placeholder_args_zero_params():
    """Test zero parameter case."""
    @cuda.jit(device=True)
    def device_func():
        return 1.0

    args = _create_placeholder_args(device_func, np.float64)
    assert args == tuple()

@pytest.mark.nocudasim
def test_create_placeholder_args_precision():
    """Test different precision types."""
    @cuda.jit(
              [numba.float32(numba.float32[:],
                numba.float32[:]),
               numba.float64(numba.float64[:],
                             numba.float64[:])],
              device=True)
    def device_func(a, b):
        return a[0] + b[0]

    args = _create_placeholder_args(device_func, np.float32)
    args32 = args[0]
    args64 = args[1]

    assert all(arg.dtype == np.float32 for arg in args32)
    assert all(arg.dtype == np.float64 for arg in args64)


@pytest.mark.nocudasim
def test_run_placeholder_kernel():
    """Test placeholder kernel creation and execution."""
    @cuda.jit(device=True)
    def add_device(a, b):
        return a[0] + b[0]

    placeholder_args = _create_placeholder_args(add_device, np.float64)

    # Should not raise
    _run_placeholder_kernel(add_device, placeholder_args)


@pytest.mark.nocudasim
def test_run_placeholder_kernel_various_param_counts():
    """Test kernel creation for different parameter counts."""
    def make_device_func(count):
        """Dynamically create a device function with `count` parameters."""
        params = ", ".join(f"a{i}" for i in range(count))
        src = f"@cuda.jit(device=True)\ndef f({params}):\n    pass"
        loc = {}
        exec(src, {"cuda": cuda}, loc)
        return loc["f"]

    for count in [0, 1, 3, 5, 8, 10, 12]:
        func = make_device_func(count)
        placeholder_args = _create_placeholder_args(func, np.float64)
        # _run_placeholder_kernel now returns None; ensure it completes
        result = _run_placeholder_kernel(func, placeholder_args)
        assert result is None


# Integration tests for specialize_and_compile

@pytest.mark.sim_only
def test_specialize_and_compile_simulator_mode(factory_with_settings):
    """Test that compilation timing is skipped in simulator mode."""
    @cuda.jit(device=True)
    def sample_device(x, y):
        return x[0] + y[0]
    timelogger = TimeLogger(verbosity='verbose')
    factory_with_settings.build = lambda: testCache(device_function=sample_device)
    factory = factory_with_settings

    timelogger.register_event("compile_test", "compile", "Test")

    # Should not raise, should skip timing

    factory.specialize_and_compile(sample_device, "compile_test")



def test_update_compile_settings_nested_attrs(factory):
    """Test that update_compile_settings finds keys in nested attrs classes."""
    @attrs.define
    class NestedSettings:
        nested_value: int = 10
        _underscore_value: int = 20

    @attrs.define
    class TopSettings:
        precision: type = np.float32
        nested: NestedSettings = attrs.Factory(NestedSettings)

    factory.setup_compile_settings(TopSettings())

    # Test updating nested attribute (no underscore)
    recognized = factory.update_compile_settings(nested_value=42)
    assert "nested_value" in recognized
    assert factory.compile_settings.nested.nested_value == 42

    # Test updating nested attribute with underscore
    recognized = factory.update_compile_settings(underscore_value=100)
    assert "underscore_value" in recognized
    assert factory.compile_settings.nested._underscore_value == 100

    # Verify cache was invalidated
    assert factory.cache_valid is False


def test_update_compile_settings_nested_dict(factory):
    """Test that update_compile_settings finds keys in nested dicts."""
    @attrs.define
    class TopSettingsWithDict:
        precision: type = np.float32
        options: dict = attrs.Factory(lambda: {"key1": "value1", "key2": 10})

    factory.setup_compile_settings(TopSettingsWithDict())

    # Test updating dict key
    recognized = factory.update_compile_settings(key1="new_value")
    assert "key1" in recognized
    assert factory.compile_settings.options["key1"] == "new_value"

    recognized = factory.update_compile_settings(key2=99)
    assert "key2" in recognized
    assert factory.compile_settings.options["key2"] == 99

    # Verify cache was invalidated
    assert factory.cache_valid is False


def test_update_compile_settings_nested_not_found(factory):
    """Test that unrecognized nested keys raise KeyError."""
    @attrs.define
    class NestedSettings:
        nested_value: int = 10

    @attrs.define
    class TopSettings:
        precision: type = np.float32
        nested: NestedSettings = attrs.Factory(NestedSettings)

    factory.setup_compile_settings(TopSettings())

    with pytest.raises(KeyError):
        factory.update_compile_settings(nonexistent_key=42)
