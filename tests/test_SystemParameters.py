import pytest
import numpy as np
from CuMC.SystemModels.SystemParameters import SystemParametersDict, SystemParameters

def retrieves_existing_parameter():
    params = SystemParametersDict({"a": 1, "b": 2})
    assert params.get_parameter("a") == 1
    assert params.get_parameter("b") == 2

def raises_key_error_when_retrieving_nonexistent_parameter():
    params = SystemParametersDict({"a": 1})
    with pytest.raises(KeyError, match="Constant c not in parameters dictionary"):
        params.get_parameter("c")

def updates_existing_parameter():
    params = SystemParametersDict({"a": 1})
    params.set_parameter("a", 10)
    assert params["a"] == 10

def raises_key_error_when_updating_nonexistent_parameter():
    params = SystemParametersDict({"a": 1})
    with pytest.raises(KeyError, match="Constant b not in parameters dictionary"):
        params.set_parameter("b", 5)

def test_systemparameters_init_and_array_dtype():
    defaults = {"a": 1.0, "b": 2.0}
    values = {"b": 3.0, "c": 4.0}
    precision = np.float32
    params = SystemParameters(values, defaults, precision)
    assert params.param_array.dtype == precision
    # Should be in order: a, b, c
    assert np.allclose(params.param_array, [1.0, 3.0, 4.0])

def test_systemparameters_param_indices_order():
    defaults = {"x": 10, "y": 20}
    params = SystemParameters({}, defaults, np.float64)
    keys = list(params.parameterDict.keys())
    for k in keys:
        assert params.param_array[params.param_indices[k]] == params.parameterDict[k]

def test_get_param_index_single_and_list():
    defaults = {"foo": 1, "bar": 2}
    params = SystemParameters({}, defaults, np.float32)
    idx_foo = params.get_param_index("foo")
    idx_bar = params.get_param_index("bar")
    assert params.param_array[idx_foo] == 1
    assert params.param_array[idx_bar] == 2
    idxs = params.get_param_index(["foo", "bar"])
    assert idxs == [idx_foo, idx_bar]

def test_get_param_index_keyerror():
    defaults = {"a": 1}
    params = SystemParameters({}, defaults, np.float32)
    with pytest.raises(KeyError, match="Parameter key 'b' not found in param_indices."):
        params.get_param_index("b")
    with pytest.raises(KeyError, match="Parameter key\\(s\\) \\['b', 'c'\\] not found in param_indices."):
        params.get_param_index(["a", "b", "c"])

def test_get_param_index_typeerror():
    defaults = {"a": 1}
    params = SystemParameters({}, defaults, np.float32)
    with pytest.raises(TypeError):
        params.get_param_index(123)

def test_print_param_indices_output(capsys):
    defaults = {"alpha": 1, "beta": 2}
    params = SystemParameters({}, defaults, np.float32)
    params.print_param_indices()
    captured = capsys.readouterr()
    for k in defaults:
        assert f"{k}:" in captured.out
