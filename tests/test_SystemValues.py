import pytest
import numpy as np
from CuMC.SystemModels.SystemValues import SystemValues

def test_systemvalues_init_and_array_dtype():
    defaults = {"a": 1.0, "b": 2.0}
    values = {"b": 3.0, "c": 4.0}
    precision = np.float32
    params = SystemValues(values, defaults, precision)
    assert params.values_array.dtype == precision
    # Should be in order: a, b, c
    assert np.allclose(params.values_array, [1.0, 3.0, 4.0])

def test_systemvalues_param_indices_order():
    defaults = {"x": 10, "y": 20}
    params = SystemValues({}, defaults, np.float64)
    keys = list(params.values_dict.keys())
    for k in keys:
        assert params.values_array[params.indices_dict[k]] == params.values_dict[k]

def test_get_param_index_single_and_list():
    defaults = {"foo": 1, "bar": 2}
    params = SystemValues({}, defaults, np.float32)
    idx_foo = params.get_param_index("foo")
    idx_bar = params.get_param_index("bar")
    assert params.values_array[idx_foo] == 1
    assert params.values_array[idx_bar] == 2
    idxs = params.get_param_index(["foo", "bar"])
    assert idxs == [idx_foo, idx_bar]

def test_get_param_index_keyerror():
    defaults = {"a": 1}
    params = SystemValues({}, defaults, np.float32)
    with pytest.raises(KeyError, match="'b' not found in this SystemValues object"):
        params.get_param_index("b")
    with pytest.raises(KeyError, match="Parameter key\\(s\\) \\['b', 'c'\\] not found in this SystemValues object"):
        params.get_param_index(["a", "b", "c"])

def test_get_param_index_typeerror():
    defaults = {"a": 1}
    params = SystemValues({}, defaults, np.float32)
    with pytest.raises(TypeError):
        params.get_param_index(123)

def test_print_param_indices_output(capsys):
    defaults = {"alpha": 1, "beta": 2}
    params = SystemValues({}, defaults, np.float32)
    params.print_param_indices()
    captured = capsys.readouterr()
    for k in defaults:
        assert f"{k}:" in captured.out

def test_systemvalues_get_value():
    defaults = {"a": 1.0, "b": 2.0}
    params = SystemValues({}, defaults, np.float32)

    # Test getting a single value
    assert params.get_value("a") == 1.0
    assert params.get_value("b") == 2.0

    # Test getting multiple values
    values = params.get_value(["a", "b"])
    assert values == [1.0, 2.0]

    # Test error on non-existent parameter
    with pytest.raises(KeyError, match="not found in this SystemValues object"):
        params.get_value("c")

    # Test error on non-existent parameters in a list
    with pytest.raises(KeyError, match="not found in this SystemValues object"):
        params.get_value(["a", "c"])

def test_systemvalues_set_parameters():
    defaults = {"a": 1.0, "b": 2.0}
    params = SystemValues({}, defaults, np.float32)

    # Test updating a parameter
    params.set_values_dict({"a": 10.0})

    # Check that both values_dict and values_array are updated
    assert params.values_dict["a"] == 10.0
    a_index = params.get_param_index("a")
    assert params.values_array[a_index] == 10.0

    # Test updating multiple parameters
    params.set_values_dict({"a": 15.0, "b": 20.0})
    assert params.values_dict["a"] == 15.0
    assert params.values_dict["b"] == 20.0
    a_index = params.get_param_index("a")
    b_index = params.get_param_index("b")
    assert params.values_array[a_index] == 15.0
    assert params.values_array[b_index] == 20.0

    # Test error on non-existent parameter
    with pytest.raises(KeyError):
        params.set_values_dict({"c": 30.0})

def test_update_param_array_and_indices():
    defaults = {"a": 1.0, "b": 2.0}
    params = SystemValues({}, defaults, np.float32)

    # Modify the dictionary directly
    params.values_dict["c"] = 3.0

    # Update the values_array and indices_dict
    params.update_param_array_and_indices()

    # Check that values_array and indices_dict are updated correctly
    assert "c" in params.indices_dict
    c_index = params.get_param_index("c")
    assert params.values_array[c_index] == 3.0

    # Check that the existing values are still correct
    a_index = params.get_param_index("a")
    b_index = params.get_param_index("b")
    assert params.values_array[a_index] == 1.0
    assert params.values_array[b_index] == 2.0

def test_kwargs_override():
    defaults = {"a": 1.0, "b": 2.0}
    values = {"b": 3.0}
    params = SystemValues(values, defaults, np.float32, a=10.0, c=4.0)

    # Check that kwargs override both defaults and values_dict
    assert params.values_dict["a"] == 10.0  # Overridden by kwargs
    assert params.values_dict["b"] == 3.0   # From values_dict
    assert params.values_dict["c"] == 4.0   # Added by kwargs

def test_list_of_strings_as_values_dict():
    defaults = {"a": 1.0, "b": 2.0}
    values = ["c", "d"]  # List of strings
    params = SystemValues(values, defaults, np.float32)

    # Check that all keys from defaults and values are present
    assert "a" in params.values_dict
    assert "b" in params.values_dict
    assert "c" in params.values_dict
    assert "d" in params.values_dict

    # Check values
    assert params.values_dict["a"] == 1.0  # From defaults
    assert params.values_dict["b"] == 2.0  # From defaults
    assert params.values_dict["c"] == 0.0  # From list of strings, default value 0.0
    assert params.values_dict["d"] == 0.0  # From list of strings, default value 0.0

def test_systemvalues_getitem():
    defaults = {"a": 1.0, "b": 2.0, "c": 3.0}
    params = SystemValues({}, defaults, np.float32)

    # Test dictionary-like access
    assert params["a"] == 1.0
    assert params["b"] == 2.0
    assert params["c"] == 3.0

    # Test array-like access
    a_index = params.get_param_index("a")
    b_index = params.get_param_index("b")
    c_index = params.get_param_index("c")

    assert params[a_index] == 1.0
    assert params[b_index] == 2.0
    assert params[c_index] == 3.0

    # Test error on non-existent key
    with pytest.raises(KeyError, match="not found in this SystemValues object"):
        _ = params["d"]

    # Test error on out-of-bounds index
    with pytest.raises(IndexError, match="out of bounds"):
        _ = params[len(params.values_array)]

    # Test error on invalid key type
    with pytest.raises(TypeError, match="key must be a string, integer, or slice"):
        _ = params[1.5]

def test_systemvalues_setitem():
    defaults = {"a": 1.0, "b": 2.0, "c": 3.0}
    params = SystemValues({}, defaults, np.float32)

    # Test dictionary-like update
    params["a"] = 10.0
    assert params.values_dict["a"] == 10.0
    a_index = params.get_param_index("a")
    assert params.values_array[a_index] == 10.0

    # Test array-like update
    b_index = params.get_param_index("b")
    params[b_index] = 20.0
    assert params.values_dict["b"] == 20.0
    assert params.values_array[b_index] == 20.0

    # Test error on non-existent key
    with pytest.raises(KeyError, match="not found in this SystemValues object"):
        params["d"] = 40.0

    # Test error on out-of-bounds index
    with pytest.raises(IndexError, match="out of bounds"):
        params[len(params.values_array)] = 50.0

    # Test error on invalid key type
    with pytest.raises(TypeError, match="key must be a string, integer, or slice"):
        params[1.5] = 60.0

def test_systemvalues_getitem_slice():
    defaults = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    params = SystemValues({}, defaults, np.float32)

    # Test slice access
    slice_values = params[1:4]
    assert isinstance(slice_values, np.ndarray)
    assert len(slice_values) == 3
    assert np.allclose(slice_values, [2.0, 3.0, 4.0])

    # Test slice with step
    slice_values = params[0:5:2]
    assert isinstance(slice_values, np.ndarray)
    assert len(slice_values) == 3
    assert np.allclose(slice_values, [1.0, 3.0, 5.0])

    # Test slice with negative indices
    slice_values = params[-3:]
    assert isinstance(slice_values, np.ndarray)
    assert len(slice_values) == 3
    assert np.allclose(slice_values, [3.0, 4.0, 5.0])

def test_systemvalues_setitem_slice():
    defaults = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    params = SystemValues({}, defaults, np.float32)

    # Test setting a slice with a scalar value
    params[1:4] = 10.0
    assert np.allclose(params.values_array[1:4], [10.0, 10.0, 10.0])
    assert params.values_dict["b"] == 10.0
    assert params.values_dict["c"] == 10.0
    assert params.values_dict["d"] == 10.0

    # Reset for next test
    params = SystemValues({}, defaults, np.float32)

    # Test setting a slice with a sequence value
    params[1:4] = [20.0, 30.0, 40.0]
    assert np.allclose(params.values_array[1:4], [20.0, 30.0, 40.0])
    assert params.values_dict["b"] == 20.0
    assert params.values_dict["c"] == 30.0
    assert params.values_dict["d"] == 40.0

    # Reset for next test
    params = SystemValues({}, defaults, np.float32)

    # Test setting a slice with step
    params[0:5:2] = [10.0, 30.0, 50.0]
    assert np.allclose(params.values_array[0:5:2], [10.0, 30.0, 50.0])
    assert params.values_dict["a"] == 10.0
    assert params.values_dict["c"] == 30.0
    assert params.values_dict["e"] == 50.0

    # Reset for next test
    params = SystemValues({}, defaults, np.float32)

    # Test setting a slice with a scalar value and step
    params[0:5:2] = 100.0
    assert np.allclose(params.values_array[0:5:2], [100.0, 100.0, 100.0])
    assert params.values_dict["a"] == 100.0
    assert params.values_dict["c"] == 100.0
    assert params.values_dict["e"] == 100.0
