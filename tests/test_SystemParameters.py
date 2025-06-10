import pytest
from system_models.SystemParameters import SystemParametersDict

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