import pytest
from CuMC.CUDAFactory import CUDAFactory


@pytest.fixture(scope='class')
def factory():
    """Fixture to provide a factory for creating system instances."""
    factory = CUDAFactory()

    return factory


def test_setup_compile_settings(factory):
    factory.setup_compile_settings({'manually_overwritten_1': False,
                                    'manually_overwritten_2': False
                                    },
                                   )
    assert factory.compile_settings[
               'manually_overwritten_1'] is False, "setup_compile_settings did not overwrite compile settings"


@pytest.fixture(scope='function')
def factory_with_settings(factory):
    """Fixture to provide a factory with specific compile settings."""
    factory.setup_compile_settings({'manually_overwritten_1': False,
                                    'manually_overwritten_2': False
                                    },
                                   )
    return factory


def test_update_compile_settings(factory_with_settings):
    factory_with_settings.update_compile_settings(**{'manually_overwritten_1': True})
    assert factory_with_settings.compile_settings[
               'manually_overwritten_1'] is True, "compile settings were not updated correctly"

    with pytest.warns(UserWarning):
        factory_with_settings.update_compile_settings(**{'non_existent_key': True},
                                                      ), "factory did not emit a warning for non-existent key"


def test_cache_invalidation(factory_with_settings):
    assert factory_with_settings.cache_valid is False, "Cache should be invalid initially"
    _ = factory_with_settings.device_function
    assert factory_with_settings.cache_valid is True, "Cache should be valid after first access to device_function"

    factory_with_settings.update_compile_settings(**{'manually_overwritten_1': True})
    assert factory_with_settings.cache_valid is False, "Cache should be invalidated after updating compile settings"

    _ = factory_with_settings.device_function
    assert factory_with_settings.cache_valid is True, "Cache should be valid after first access to device_function"


def test_build(factory_with_settings, monkeypatch):
    test_func = factory_with_settings.device_function
    assert test_func is None
    #cache validated

    monkeypatch.setattr(factory_with_settings, 'build', lambda: 10.0)
    test_func = factory_with_settings.device_function
    assert test_func is None, "device_function rebuilt even though cache was valid"
    factory_with_settings.update_compile_settings(**{'manually_overwritten_1': True})
    test_func = factory_with_settings.device_function
    assert test_func == 10.0, "device_function was not rebuilt after cache invalidation"


def test_build_with_dict_output(factory_with_settings, monkeypatch):
    """Test that when build returns a dictionary, the values are available via get_cached_output."""
    factory_with_settings._cache_valid = False

    test_dict = {
        'test_output1': 'value1',
        'test_output2': 'value2'
        }

    monkeypatch.setattr(factory_with_settings, 'build', lambda: test_dict)

    # Access device_function to trigger build
    _ = factory_with_settings.device_function

    # Test that dictionary outputs are available
    assert factory_with_settings.get_cached_output('test_output1') == 'value1', "Dictionary output not accessible"
    assert factory_with_settings.get_cached_output('test_output2') == 'value2', "Dictionary output not accessible"

    # Test cache invalidation with dict output
    factory_with_settings.update_compile_settings(**{'manually_overwritten_1': True})
    assert factory_with_settings.cache_valid is False, "Cache should be invalidated after updating compile settings"

    # Test that dict values are rebuilt after invalidation
    new_test_dict = {
        'test_output1': 'new_value1',
        'test_output2': 'new_value2'
        }
    monkeypatch.setattr(factory_with_settings, 'build', lambda: new_test_dict)

    assert factory_with_settings.get_cached_output('test_output1',
                                                   ) == 'new_value1', "Cache not rebuilt after invalidation"


def test_device_function_from_dict(factory_with_settings, monkeypatch):
    """Test that when build returns a dict with 'device_function', it's accessible via the device_function property."""
    factory_with_settings._cache_valid = False

    test_func = lambda x: x * 2
    test_dict = {
        'device_function': test_func,
        'other_output':    'value'
        }

    monkeypatch.setattr(factory_with_settings, 'build', lambda: test_dict)

    # Check if device_function is correctly set from the dict
    assert factory_with_settings.device_function is test_func, "device_function not correctly set from dict"

    # Check that other values are still accessible
    assert factory_with_settings.get_cached_output('other_output') == 'value', "Other dict values not accessible"