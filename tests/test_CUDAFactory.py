from typing import Callable, Union

import attrs
import pytest
import numpy as np

from cubie.CUDAFactory import (
    CUDAFactory,
    CUDADispatcherCache,
    _CubieConfigBase,
    CUDAFactoryConfig,
    MultipleInstanceCUDAFactory,
)


@attrs.define()
class testCache(CUDADispatcherCache):
    """Test cache class."""

    device_function: Union[Callable, int] = attrs.field(default=-1)


def dict_to_attrs_class(dictionary):
    """Convert a dictionary to an attrs class instance."""
    # Create the class with the dictionary keys as field names
    CompileSettings = attrs.make_class(
        "CompileSettings", list(dictionary.keys()), bases=(_CubieConfigBase,)
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

    monkeypatch.setattr(
        factory_with_settings,
        "build",
        lambda: testCache(device_function=lambda: 10.0),
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

    monkeypatch.setattr(
        factory_with_settings, "build", lambda: (TestOutputs())
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


def test_update_compile_settings_nested_attrs(factory):
    """Test that update_compile_settings finds keys in nested attrs classes."""

    @attrs.define
    class NestedSettings(_CubieConfigBase):
        nested_value: int = 10
        _underscore_value: int = 20

    @attrs.define
    class TopSettings(_CubieConfigBase):
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


def test_update_compile_settings_nested_not_found(factory):
    """Test that unrecognized nested keys raise KeyError."""

    @attrs.define
    class NestedSettings(_CubieConfigBase):
        nested_value: int = 10

    @attrs.define
    class TopSettings(_CubieConfigBase):
        precision: type = np.float32
        nested: NestedSettings = attrs.Factory(NestedSettings)

    factory.setup_compile_settings(TopSettings())

    with pytest.raises(KeyError):
        factory.update_compile_settings(nonexistent_key=42)


# --- _CubieConfigBase tests ---


def test_cuda_factory_config_values_hash():
    """Test that _CubieConfigBase produces consistent hashes."""
    from cubie.CUDAFactory import _CubieConfigBase

    @attrs.define
    class TestConfig(_CubieConfigBase):
        value1: int = 10
        value2: str = "test"

    config1 = TestConfig()
    config2 = TestConfig()

    # Same values should produce same hash
    assert config1.values_hash == config2.values_hash
    assert len(config1.values_hash) == 64  # SHA256 hex digest


def test_cuda_factory_config_values_tuple():
    """Test that values_tuple returns tuple of serialized field values."""
    from cubie.CUDAFactory import _CubieConfigBase

    @attrs.define
    class TestConfig(_CubieConfigBase):
        value1: int = 42
        value2: str = "hello"

    config = TestConfig()
    vt = config.values_tuple

    assert isinstance(vt, tuple)
    assert 42 in vt
    assert "hello" in vt


def test_cuda_factory_config_update():
    """Test the update() method on _CubieConfigBase."""
    from cubie.CUDAFactory import _CubieConfigBase

    @attrs.define
    class TestConfig(_CubieConfigBase):
        value1: int = 10
        value2: str = "test"

    config = TestConfig()
    old_hash = config.values_hash

    recognized, changed = config.update({"value1": 20})
    assert "value1" in recognized
    assert "value1" in changed
    assert config.value1 == 20
    assert config.values_hash != old_hash


def test_cuda_factory_config_update_unchanged():
    """Test that update() reports no change when value is same."""
    from cubie.CUDAFactory import _CubieConfigBase

    @attrs.define
    class TestConfig(_CubieConfigBase):
        value1: int = 10

    config = TestConfig()
    old_hash = config.values_hash

    recognized, changed = config.update({"value1": 10})
    assert "value1" in recognized
    assert "value1" not in changed
    assert config.values_hash == old_hash


def test_cuda_factory_config_nested_hash(precision):
    """Test that nested _CubieConfigBase objects are included in hash."""
    from cubie.CUDAFactory import _CubieConfigBase

    @attrs.define
    class InnerConfig(_CubieConfigBase):
        inner_value: int = 5

    @attrs.define
    class OuterConfig(_CubieConfigBase):
        outer_value: int = 10
        nested: InnerConfig = attrs.Factory(InnerConfig)

    # Two configs with identical nested settings should have same hash
    config1 = OuterConfig()
    config2 = OuterConfig()
    assert config1.values_hash == config2.values_hash

    # A config with a different nested value should have a different hash
    config3 = OuterConfig(nested=InnerConfig(inner_value=999))
    assert config1.values_hash != config3.values_hash


def test_cuda_factory_config_hash_property():
    """Test that CUDAFactory.config_hash uses compile_settings.values_hash."""
    from cubie.CUDAFactory import _CubieConfigBase

    @attrs.define
    class TestConfig(_CubieConfigBase):
        value1: int = 10

    class TestFactory(CUDAFactory):
        def __init__(self):
            super().__init__()

        def build(self):
            return testCache(device_function=lambda: 1.0)

    factory = TestFactory()
    factory.setup_compile_settings(TestConfig())

    # config_hash should return the compile_settings.values_hash
    assert factory.config_hash == factory.compile_settings.values_hash
    assert len(factory.config_hash) == 64


def test_cuda_factory_config_eq_false_excluded():
    """Test that fields with eq=False are excluded from hash."""
    from cubie.CUDAFactory import _CubieConfigBase

    @attrs.define
    class TestConfig(_CubieConfigBase):
        value1: int = 10
        callback: object = attrs.field(default=None, eq=False)

    config1 = TestConfig(callback=lambda: 1)
    config2 = TestConfig(callback=lambda: 2)

    # Hash should be same despite different callbacks
    assert config1.values_hash == config2.values_hash


def test_cuda_factory_config_update_applies_converter():
    from numpy import float32, float64

    @attrs.define
    class TestConfig(CUDAFactoryConfig):
        pass

    config = TestConfig(precision=float32)
    # Update with a dtype that needs conversion
    config.update({"precision": "float64"})
    # Verify converter was applied
    assert config.precision == float64


def test_cuda_factory_config_update_nested_applies_converter():
    def x2_converter(value):
        return value * 2

    @attrs.define
    class InnerConfig(_CubieConfigBase):
        a = attrs.field(
            default=1,
            converter=x2_converter,
        )

    @attrs.define
    class OuterConfig(_CubieConfigBase):
        nested: InnerConfig = attrs.field(factory=InnerConfig)
        b = attrs.field(
            default=2,
            converter=x2_converter,
        )

    config = OuterConfig()

    # converters fire on init
    assert config.nested.a == 2
    assert config.b == 4

    config.update({"a": 3, "b": 5})
    # Verify converter was applied in nested config
    assert config.nested.a == 6
    assert config.b == 10


def test_multiple_instance_factory_prefix_mapping(precision):
    """Test that prefixed keys are mapped to unprefixed equivalents."""
    from cubie.integrators.matrix_free_solvers.linear_solver import (
        LinearSolver,
    )

    solver = LinearSolver(precision=precision, n=3)

    # Update with prefixed key
    solver.update({"krylov_max_iters": 50})

    # Verify the unprefixed setting was updated
    assert solver.compile_settings.max_iters == 50


def test_multiple_instance_factory_instance_label_stored(precision):
    """Test that instance_label attribute is correctly stored."""
    from cubie.integrators.matrix_free_solvers.linear_solver import (
        LinearSolver,
    )

    solver = LinearSolver(precision=precision, n=3)

    # Verify instance_label is set correctly
    assert solver.instance_label == "krylov"


def test_multiple_instance_factory_empty_label_allowed():
    """Test that empty instance_label is permitted for standalone use."""

    class TestFactory(MultipleInstanceCUDAFactory):
        def build(self):
            return testCache(device_function=lambda: 1.0)

    # Empty instance_label should be allowed
    factory = TestFactory(instance_label="")
    assert factory.instance_label == ""


def test_multiple_instance_factory_mixed_keys():
    """Test that prefixed keys take precedence over unprefixed."""
    from cubie.CUDAFactory import MultipleInstanceCUDAFactoryConfig

    @attrs.define
    class TestConfig(MultipleInstanceCUDAFactoryConfig):
        value: int = attrs.field(default=10, metadata={"prefixed": True})

    class TestFactory(MultipleInstanceCUDAFactory):
        def __init__(self):
            super().__init__(instance_label="test")
            self.setup_compile_settings(
                TestConfig(precision=np.float32, instance_label="test")
            )

        def build(self):
            return testCache(device_function=lambda: 1.0)

    factory = TestFactory()

    # Update with both prefixed and unprefixed - prefixed should win
    factory.update_compile_settings(
        {"value": 5, "test_value": 20}, silent=True
    )

    assert factory.compile_settings.value == 20


def test_multiple_instance_factory_no_prefix_match():
    """Test that non-matching keys pass through unchanged."""

    @attrs.define
    class TestConfig(CUDAFactoryConfig):
        value: int = 10

    class TestFactory(MultipleInstanceCUDAFactory):
        def __init__(self):
            super().__init__(instance_label="test")
            self.setup_compile_settings(TestConfig(precision=np.float32))

        def build(self):
            return testCache(device_function=lambda: 1.0)

    factory = TestFactory()

    # Update with non-prefixed key
    factory.update_compile_settings({"value": 42})

    assert factory.compile_settings.value == 42


# --- build_config instance_label tests ---


def test_build_config_with_instance_label(precision):
    """Verify prefix transformation works with instance_label parameter."""
    from cubie._utils import build_config
    from cubie.CUDAFactory import MultipleInstanceCUDAFactoryConfig

    @attrs.define
    class TestConfig(MultipleInstanceCUDAFactoryConfig):
        _atol: float = attrs.field(default=1e-6, metadata={"prefixed": True})
        _rtol: float = attrs.field(default=1e-3, metadata={"prefixed": True})

        @property
        def atol(self) -> float:
            return self._atol

        @property
        def rtol(self) -> float:
            return self._rtol

    config = build_config(
        TestConfig,
        required={"precision": precision},
        instance_label="krylov",
        krylov_atol=1e-10,
        krylov_rtol=1e-5,
    )

    # Verify prefixed keys were transformed to unprefixed
    assert config.atol == 1e-10
    assert config.rtol == 1e-5
    # Verify instance_label was set
    assert config.instance_label == "krylov"


def test_build_config_instance_label_prefixed_takes_precedence(precision):
    """Verify prefixed key wins when both prefixed and unprefixed provided."""
    from cubie._utils import build_config
    from cubie.CUDAFactory import MultipleInstanceCUDAFactoryConfig

    @attrs.define
    class TestConfig(MultipleInstanceCUDAFactoryConfig):
        _atol: float = attrs.field(
            default=1e-6, alias="atol", metadata={"prefixed": True}
        )

        @property
        def atol(self) -> float:
            return self._atol

    config = build_config(
        TestConfig,
        required={"precision": precision},
        instance_label="krylov",
        atol=1e-8,  # Unprefixed
        krylov_atol=1e-12,  # Prefixed - should take precedence
    )

    # Prefixed value should win
    assert config.atol == 1e-12


def test_build_config_backward_compatible_no_instance_label(precision):
    """Verify existing behavior unchanged when instance_label not provided."""
    from cubie._utils import build_config
    from cubie.CUDAFactory import CUDAFactoryConfig

    @attrs.define
    class TestConfig(CUDAFactoryConfig):
        _atol: float = attrs.field(default=1e-6, alias="atol")
        value: int = 10

        @property
        def atol(self) -> float:
            return self._atol

    # Without instance_label
    config = build_config(
        TestConfig,
        required={"precision": precision},
        atol=1e-8,
        value=42,
    )

    assert config.atol == 1e-8
    assert config.value == 42


def test_multiple_instance_config_prefix_property(precision):
    """Verify prefix property returns instance_label."""
    from cubie.CUDAFactory import MultipleInstanceCUDAFactoryConfig

    @attrs.define
    class TestConfig(MultipleInstanceCUDAFactoryConfig):
        _atol: float = attrs.field(
            default=1e-6, alias="atol", metadata={"prefixed": True}
        )

    config = TestConfig(precision=precision, instance_label="krylov")

    # prefix property should return instance_label
    assert config.prefix == "krylov"
    assert config.prefix == config.instance_label

    # With empty instance_label
    config_empty = TestConfig(precision=precision, instance_label="")
    assert config_empty.prefix == ""


def test_multiple_instance_config_post_init_populates_prefixed_attrs(
    precision,
):
    """Verify __attrs_post_init__ correctly populates prefixed_attributes."""
    from cubie.CUDAFactory import MultipleInstanceCUDAFactoryConfig

    @attrs.define
    class TestConfig(MultipleInstanceCUDAFactoryConfig):
        _atol: float = attrs.field(default=1e-6, metadata={"prefixed": True})
        _rtol: float = attrs.field(default=1e-3, metadata={"prefixed": True})
        non_prefixed: int = attrs.field(
            default=10,
        )

    # With instance_label set, prefixed_attributes should be populated
    config = TestConfig(precision=precision, instance_label="krylov")

    # prefixed_attributes should include atol and rtol but not non_prefixed
    assert "_atol" in config.prefixed_attributes
    assert "_rtol" in config.prefixed_attributes
    assert "non_prefixed" not in config.prefixed_attributes
    # precision and instance_label are not prefixed (structural parameters)
    assert "precision" not in config.prefixed_attributes
    assert "instance_label" not in config.prefixed_attributes
    assert "prefixed_attributes" not in config.prefixed_attributes

    # With empty instance_label, prefixed_attributes should remain empty
    config_empty = TestConfig(precision=precision, instance_label="")
    assert config_empty.prefixed_attributes == set()


def test_no_manual_key_filtering(precision):
    """Verify factory classes don't manually filter keys.

    All kwargs should pass through to nested objects; each level
    extracts its own via build_config/update and ignores the rest.
    """
    from cubie.integrators.matrix_free_solvers.linear_solver import (
        LinearSolver,
    )

    # Pass unrelated kwargs - they should be silently ignored
    solver = LinearSolver(
        precision=precision,
        n=3,
        unrelated_param=42,
        another_unknown="value",
    )

    # Verify solver was created successfully
    assert solver.n == 3
