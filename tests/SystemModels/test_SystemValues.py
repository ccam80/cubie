import pytest
import numpy as np

from cubie.systemmodels.SystemValues import SystemValues

def test_init_edge_cases():
    """Test edge cases for the __init__ method."""
    # Test with None values_dict
    precision = np.float32
    params = SystemValues(None, precision)
    assert len(params.values_array) == 0
    assert params.values_dict == {}

    # Test with empty dictionary
    params = SystemValues({}, precision)
    assert len(params.values_array) == 0
    assert params.values_dict == {}

    # Test with list of strings as values_dict
    params = SystemValues(["x", "y", "z"], precision)
    assert len(params.values_array) == 3
    assert params.values_dict == {"x": 0.0, "y": 0.0, "z": 0.0}

    # Test with kwargs
    params = SystemValues({"a": 1.0}, precision, x=10.0, y=20.0)
    assert params.values_dict == {"a": 1.0, "x": 10.0, "y": 20.0}
    assert len(params.values_array) == 3

def test_param_array_and_indices_match_supplied_dict():
    """Test that param_array and indices dicts match the supplied dictionary after instantiation."""
    # Test with a simple dictionary
    values_dict = {"a": 1.0, "b": 2.0, "c": 3.0}
    precision = np.float32
    params = SystemValues(values_dict, precision)

    # Check that values_array contains all values from values_dict in the correct order
    assert len(params.values_array) == len(values_dict)
    for i, (key, value) in enumerate(values_dict.items()):
        assert params.indices_dict[key] == i
        assert params.keys_by_index[i] == key
        assert params.values_array[i] == value

    # Test with defaults and overrides
    defaults = {"a": 1.0, "b": 2.0}
    values = {"b": 3.0, "c": 4.0}
    params = SystemValues(values, precision, defaults)

    # Expected combined dictionary
    expected = {"a": 1.0, "b": 3.0, "c": 4.0}

    # Check that values_array contains all values from the combined dictionary
    assert len(params.values_array) == len(expected)
    for key, value in expected.items():
        index = params.indices_dict[key]
        assert params.values_array[index] == value
        assert params.values_dict[key] == value
        assert params.keys_by_index[index] == key

def test_get_index_of_key():
    """Test that getting the index of an individual key works."""
    values_dict = {"foo": 1.0, "bar": 2.0, "baz": 3.0}
    precision = np.float32
    params = SystemValues(values_dict, precision)

    # Test getting index of each key
    for i, key in enumerate(values_dict.keys()):
        assert params.get_index_of_key(key) == i

    # Test KeyError for non-existent key
    with pytest.raises(KeyError, match="not found in this SystemValues object"):
        params.get_index_of_key("nonexistent")

    # Test TypeError for non-string key
    with pytest.raises(TypeError, match="parameter_key must be a string"):
        params.get_index_of_key(123)

def test_get_indices():
    """Test that get_indices works as expected for each input type mentioned in the method."""
    values_dict = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    precision = np.float32
    params = SystemValues(values_dict, precision)

    # Test with a single string
    indices = params.get_indices("a")
    assert np.array_equal(indices, np.asarray([0], dtype=np.int16))


    # Test with a list of strings
    indices = params.get_indices(["a", "c", "e"])
    assert np.array_equal(indices, np.asarray([0, 2, 4], dtype=np.int16))

    # Test with a single integer
    indices = params.get_indices(1)
    assert np.array_equal(indices, np.asarray([1], dtype=np.int16))

    # Test with a list of integers
    indices = params.get_indices([0, 2, 4])
    assert np.array_equal(indices, np.asarray([0, 2, 4], dtype=np.int16))

    # Test with a slice
    indices = params.get_indices(slice(1, 4))
    assert np.array_equal(indices, np.asarray([1, 2, 3], dtype=np.int16))

    # Test with a numpy array
    indices = params.get_indices(np.asarray([0, 2, 4]))
    assert np.array_equal(indices, np.asarray([0, 2, 4], dtype=np.int16))

    # Test error cases
    with pytest.raises(KeyError, match="not found in this SystemValues object"):
        params.get_indices("nonexistent")

    with pytest.raises(TypeError, match="you can provide a list of strings or a list of integers"):
        params.get_indices(["a", 1])

    with pytest.raises(TypeError, match="you can provide strings that match the labels"):
        params.get_indices(1.5)

def test_get_values_edge_cases():
    """Test edge cases and error handling for get_values method."""
    values_dict = {"a": 1.0, "b": 2.0, "c": 3.0}
    precision = np.float32
    params = SystemValues(values_dict, precision)

    # Test with out-of-bounds index
    with pytest.raises(IndexError):
        params.get_values(10)

    # Test with invalid type
    with pytest.raises(TypeError):
        params.get_values(1.5)

    # Test with complex object
    with pytest.raises(TypeError):
        params.get_values(object())

    # Test with None
    with pytest.raises(TypeError):
        params.get_values(None)

def test_set_values_edge_cases():
    """Test edge cases and error handling for set_values method."""
    values_dict = {"a": 1.0, "b": 2.0, "c": 3.0}
    precision = np.float32
    params = SystemValues(values_dict, precision)

    # Test with out-of-bounds index
    with pytest.raises(IndexError):
        params.set_values(10, 100.0)

    # Test with invalid type
    with pytest.raises(TypeError):
        params.set_values(1.5, 100.0)

    # Test with complex object
    with pytest.raises(TypeError):
        params.set_values(object(), 100.0)

    # Test with None
    with pytest.raises(TypeError):
        params.set_values(None, 100.0)

    # MOre keys than values
    with pytest.raises(ValueError):
        params.set_values([1, 2], 100.0)

    # More values than keys
    with pytest.raises(ValueError):
        params.set_values("a", [100.0, 200.0])

    # Test with non-numeric value
    with pytest.raises(TypeError):
        params.set_values("a", ["not a number"])

def test_get_values_and_set_values():
    """Test that get_values and set_values work for all key types."""
    values_dict = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    precision = np.float32
    params = SystemValues(values_dict, precision)

    # Test get_values with a string key
    values = params.get_values("a")
    assert np.array_equal(values, np.asarray(1.0, dtype=precision))

    # Test get_values with a list of string keys
    values = params.get_values(["a", "c", "e"])
    assert np.array_equal(values, np.asarray([1.0, 3.0, 5.0], dtype=precision))

    # Test get_values with an integer key
    values = params.get_values(1)
    assert np.array_equal(values, np.asarray(2.0, dtype=precision))

    # Test get_values with a list of integer keys
    values = params.get_values([0, 2, 4])
    assert np.array_equal(values, np.asarray([1.0, 3.0, 5.0], dtype=precision))

    # Test get_values with a slice
    values = params.get_values(slice(1, 4))
    assert np.array_equal(values, np.asarray([2.0, 3.0, 4.0], dtype=precision))

    # Test set_values with a string key
    params.set_values("a", 10.0)
    assert params.values_dict["a"] == 10.0
    assert params.values_array[0] == 10.0

    # Test set_values with a list of string keys
    params.set_values(["b", "c"], [20.0, 30.0])
    assert params.values_dict["b"] == 20.0
    assert params.values_dict["c"] == 30.0
    assert params.values_array[1] == 20.0
    assert params.values_array[2] == 30.0

    # Test set_values with an integer key
    params.set_values(3, 40.0)
    assert params.values_dict["d"] == 40.0
    assert params.values_array[3] == 40.0

    # Test set_values with a list of integer keys
    params.set_values([0, 4], [100.0, 500.0])
    assert params.values_dict["a"] == 100.0
    assert params.values_dict["e"] == 500.0
    assert params.values_array[0] == 100.0
    assert params.values_array[4] == 500.0


def test_update_from_dict():
    """Test that update_from_dict works when given a single-item or multi-item dict."""
    values_dict = {"a": 1.0, "b": 2.0, "c": 3.0}
    precision = np.float32
    params = SystemValues(values_dict, precision)

    # Test with a single-item dict
    params.update_from_dict({"a": 10.0})
    assert params.values_dict["a"] == 10.0
    assert params.values_array[0] == 10.0
    assert params.values_dict["b"] == 2.0
    assert params.values_dict["c"] == 3.0

    # Test with a multi-item dict
    params.update_from_dict({"b": 20.0, "c": 30.0})
    assert params.values_dict["a"] == 10.0
    assert params.values_dict["b"] == 20.0
    assert params.values_dict["c"] == 30.0
    assert params.values_array[0] == 10.0
    assert params.values_array[1] == 20.0
    assert params.values_array[2] == 30.0

    # Test error on non-existent key
    with pytest.raises(KeyError, match="not found in this SystemValues object"):
        params.update_from_dict({"d": 40.0})


def test_indexing_as_array_or_dict():
    """Test that we can get and set by indexing the object as either an array or a dict."""
    values_dict = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    precision = np.float32
    params = SystemValues(values_dict, precision)

    # Test getting values by string key (dict-like)
    values_a = params["a"]
    assert np.array_equal(values_a, np.asarray(1.0, dtype=precision))

    values_c = params["c"]
    assert np.array_equal(values_c, np.asarray(3.0, dtype=precision))

    # Test getting values by integer index (array-like)
    values_0 = params[0]
    assert np.array_equal(values_0, np.asarray(1.0, dtype=precision))

    values_2 = params[2]
    assert np.array_equal(values_2, np.asarray(3.0, dtype=precision))

    # Test getting values by slice
    values = params[1:4]
    assert np.array_equal(values, np.asarray([2.0, 3.0, 4.0], dtype=precision))

    # Test setting values by string key (dict-like)
    params["a"] = 10.0
    assert params.values_dict["a"] == 10.0
    assert params.values_array[0] == 10.0

    # Test setting values by integer index (array-like)
    params[2] = 30.0
    assert params.values_dict["c"] == 30.0
    assert params.values_array[2] == 30.0

    # Test setting values by slice
    params[3:5] = [40.0, 50.0]
    assert params.values_dict["d"] == 40.0
    assert params.values_dict["e"] == 50.0
    assert params.values_array[3] == 40.0
    assert params.values_array[4] == 50.0

    # Test error cases
    with pytest.raises(KeyError, match="not found in this SystemValues object"):
        params["nonexistent"] = 100.0

    with pytest.raises(IndexError, match="out of bounds"):
        params[10] = 100.0


def test_init_with_invalid_precision():
    """Test initialization with invalid precision types."""
    values_dict = {"a": 1.0, "b": 2.0}

    # Test with unsupported precision type
    with pytest.raises(TypeError):
        SystemValues(values_dict, str)


def test_set_values_with_mismatched_lengths():
    """Test set_values with mismatched lengths of keys and values."""
    values_dict = {"a": 1.0, "b": 2.0, "c": 3.0}
    precision = np.float32
    params = SystemValues(values_dict, precision)

    # Test with more keys than values
    with pytest.raises(ValueError):
        params.set_values(["a", "b", "c"], [10.0, 20.0])

    # Test with more values than keys
    with pytest.raises(ValueError):
        params.set_values(["a", "b"], [10.0, 20.0, 30.0])


def test_set_values_with_non_list_values():
    """Test set_values with non-list values for multiple keys."""
    values_dict = {"a": 1.0, "b": 2.0, "c": 3.0}
    precision = np.float32
    params = SystemValues(values_dict, precision)

    # Test with non-list value for multiple keys
    with pytest.raises(ValueError):
        params.set_values(["a", "b"], 10.0)


def test_update_from_dict_with_empty_dict():
    """Test update_from_dict with empty dictionary."""
    values_dict = {"a": 1.0, "b": 2.0, "c": 3.0}
    precision = np.float32
    params = SystemValues(values_dict, precision)

    # Test with empty dictionary
    params.update_from_dict({})
    # Should not change anything
    assert params.values_dict == values_dict
    assert np.array_equal(params.values_array, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


def test_init_with_conflicting_keys():
    """Test initialization with conflicting keys in defaults, values_dict, and kwargs."""
    defaults = {"a": 1.0, "b": 2.0, "c": 3.0}
    values_dict = {"b": 20.0, "c": 30.0, "d": 40.0}
    precision = np.float32

    # Test with conflicting keys in defaults and values_dict
    params = SystemValues(values_dict, precision, defaults)
    assert params.values_dict["a"] == 1.0
    assert params.values_dict["b"] == 20.0  # values_dict overrides defaults
    assert params.values_dict["c"] == 30.0  # values_dict overrides defaults
    assert params.values_dict["d"] == 40.0

    # Test with conflicting keys in defaults, values_dict, and kwargs
    params = SystemValues(values_dict, precision, defaults, a=10.0, c=300.0)
    assert params.values_dict["a"] == 10.0  # kwargs override defaults and values_dict
    assert params.values_dict["b"] == 20.0  # values_dict overrides defaults
    assert params.values_dict["c"] == 300.0  # kwargs override defaults and values_dict
    assert params.values_dict["d"] == 40.0


def test_with_large_dictionary():
    """Test with a very large dictionary."""
    # Create a large dictionary with 1000 items
    large_dict = {f"param_{i}": float(i) for i in range(1000)}
    precision = np.float32

    # Test initialization with large dictionary
    params = SystemValues(large_dict, precision)
    assert len(params.values_array) == 1000
    assert len(params.values_dict) == 1000
    assert len(params.indices_dict) == 1000
    assert len(params.keys_by_index) == 1000

    # Test get_values with large dictionary
    values = params.get_values([f"param_{i}" for i in range(0, 1000, 100)])
    assert len(values) == 10
    assert values[0] == 0.0
    assert values[9] == 900.0

    # Test update_from_dict with large update
    update_dict = {f"param_{i}": float(i * 2) for i in range(500, 1000)}
    params.update_from_dict(update_dict)
    assert params.values_dict["param_500"] == 1000.0
    assert params.values_dict["param_999"] == 1998.0
