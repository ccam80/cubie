import attrs
import pytest

from cubie.CUDAFactory import CUDAFactory


def dict_to_attrs_class(dictionary):
    """Convert a dictionary to an attrs class instance."""
    # Create the class with the dictionary keys as field names
    CompileSettings = attrs.make_class("CompileSettings",
                                       list(dictionary.keys()))

    # Create an instance with the values from the dictionary
    return CompileSettings(**dictionary)


@pytest.fixture(scope='class')
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
        'manually_overwritten_1': False,
        'manually_overwritten_2': False
    }
    factory.setup_compile_settings(dict_to_attrs_class(settings_dict))
    assert factory.compile_settings.manually_overwritten_1 is False, "setup_compile_settings did not overwrite compile settings"


@pytest.fixture(scope='function')
def factory_with_settings(factory):
    """Fixture to provide a factory with specific compile settings."""
    settings_dict = {
        'manually_overwritten_1': False,
        'manually_overwritten_2': False
    }
    factory.setup_compile_settings(dict_to_attrs_class(settings_dict))
    return factory


def test_update_compile_settings(factory_with_settings):
    factory_with_settings.update_compile_settings(manually_overwritten_1=True)
    assert factory_with_settings.compile_settings.manually_overwritten_1 is True, "compile settings were not updated correctly"
    with pytest.raises(KeyError):
        factory_with_settings.update_compile_settings(non_existent_key=True
                                                      ), "factory did not emit a warning for non-existent key"

def test_update_compile_settings_reports_correct_key(factory_with_settings):
    with pytest.raises(KeyError) as exc:
        factory_with_settings.update_compile_settings({'non_existent_key': True,
                                                       'manually_overwritten_1': True})
    assert 'non_existent_key' in str(exc.value)
    assert 'manually_overwritten_1' not in str(exc.value)

def test_cache_invalidation(factory_with_settings):
    assert factory_with_settings.cache_valid is False, "Cache should be invalid initially"
    _ = factory_with_settings.device_function
    assert factory_with_settings.cache_valid is True, "Cache should be valid after first access to device_function"

    factory_with_settings.update_compile_settings(manually_overwritten_1=True)
    assert factory_with_settings.cache_valid is False, "Cache should be invalidated after updating compile settings"

    _ = factory_with_settings.device_function
    assert factory_with_settings.cache_valid is True, "Cache should be valid after first access to device_function"


def test_build(factory_with_settings, monkeypatch):
    test_func = factory_with_settings.device_function
    assert test_func is None
    # cache validated

    monkeypatch.setattr(factory_with_settings, 'build', lambda: 10.0)
    test_func = factory_with_settings.device_function
    assert test_func is None, "device_function rebuilt even though cache was valid"
    factory_with_settings.update_compile_settings(manually_overwritten_1=True)
    test_func = factory_with_settings.device_function
    assert test_func == 10.0, "device_function was not rebuilt after cache invalidation"


def test_build_with_dict_output(factory_with_settings, monkeypatch):
    """Test that when build returns a dictionary, the values are available via get_cached_output."""
    factory_with_settings._cache_valid = False

    @attrs.define
    class TestOutputs:
        test_output1: str = 'value1'
        test_output2: str = 'value2'

    monkeypatch.setattr(factory_with_settings, 'build', lambda: TestOutputs())

    # Access device_function to trigger build
    _ = factory_with_settings.device_function

    # Test that dictionary outputs are available
    assert factory_with_settings.get_cached_output(
        'test_output1') == 'value1', "Output not accessible"
    assert factory_with_settings.get_cached_output(
        'test_output2') == 'value2', "Output not accessible"

    # Test cache invalidation with dict output
    factory_with_settings.update_compile_settings(manually_overwritten_1=True)
    assert factory_with_settings.cache_valid is False, "Cache should be invalidated after updating compile settings"

    # Test that dict values are rebuilt after invalidation
    @attrs.define
    class NewTestOutputs:
        test_output1: str = 'new_value1'
        test_output2: str = 'new_value2'

    monkeypatch.setattr(factory_with_settings, 'build',
                        lambda: NewTestOutputs())

    output = factory_with_settings.get_cached_output('test_output1')
    assert output == 'new_value1', "Cache not rebuilt after invalidation"


def test_device_function_from_dict(factory_with_settings, monkeypatch):
    """Test that when build returns a dict with 'device_function',
     it's accessible via the device_function property."""
    factory_with_settings._cache_valid = False

    def test_func(x): return (x * 2)

    @attrs.define
    class TestOutputsWithFunc:
        device_function: callable = test_func
        other_output: str = 'value'

    monkeypatch.setattr(factory_with_settings, 'build',
                        lambda: TestOutputsWithFunc())

    # Check if device_function is correctly set from the dict
    assert factory_with_settings.device_function is test_func, \
        "device_function not correctly set from attrs class"

    # Check that other values are still accessible
    assert factory_with_settings.get_cached_output(
        'other_output') == 'value', "Other attrs values not accessible"
