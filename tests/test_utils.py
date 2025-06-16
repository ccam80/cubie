import pytest
import numpy as np
from CuMC._utils import _convert_to_indices
from CuMC.SystemModels.SystemValues import SystemValues

def test_convert_to_indices_string():
    """Test _convert_to_indices with a single string parameter name."""
    # Create a SystemValues object with some default values
    values = {"param1": 1.0, "param2": 2.0, "param3": 3.0}
    system_values = SystemValues(values, np.float64)

    # Test with a single string
    indices = _convert_to_indices("param1", system_values)
    assert isinstance(indices, np.ndarray)
    assert indices.dtype == np.int16
    assert indices.shape == (1,)
    assert indices[0] == system_values.get_param_index("param1")

def test_convert_to_indices_integer():
    """Test _convert_to_indices with a single integer index."""
    # Create a SystemValues object with some default values
    values = {"param1": 1.0, "param2": 2.0, "param3": 3.0}
    system_values = SystemValues(values, np.float64)

    # Test with a single integer
    indices = _convert_to_indices(0, system_values)
    assert isinstance(indices, np.ndarray)
    assert indices.dtype == np.int16
    assert indices.shape == (1,)
    assert indices[0] == 0

def test_convert_to_indices_string_list():
    """Test _convert_to_indices with a list of string parameter names."""
    # Create a SystemValues object with some default values
    values = {"param1": 1.0, "param2": 2.0, "param3": 3.0}
    system_values = SystemValues(values, np.float64)

    # Test with a list of strings
    indices = _convert_to_indices(["param1", "param3"], system_values)
    assert isinstance(indices, np.ndarray)
    assert indices.dtype == np.int16
    assert indices.shape == (2,)
    assert indices[0] == system_values.get_param_index("param1")
    assert indices[1] == system_values.get_param_index("param3")

def test_convert_to_indices_integer_list():
    """Test _convert_to_indices with a list of integer indices."""
    # Create a SystemValues object with some default values
    values = {"param1": 1.0, "param2": 2.0, "param3": 3.0}
    system_values = SystemValues(values, np.float64)

    # Test with a list of integers
    indices = _convert_to_indices([0, 2], system_values)
    assert isinstance(indices, np.ndarray)
    assert indices.dtype == np.int16
    assert indices.shape == (2,)
    assert indices[0] == 0
    assert indices[1] == 2

def test_convert_to_indices_numpy_array():
    """Test _convert_to_indices with a numpy array of indices."""
    # Create a SystemValues object with some default values
    values = {"param1": 1.0, "param2": 2.0, "param3": 3.0}
    system_values = SystemValues(values, np.float64)

    # Test with a numpy array
    input_array = np.array([0, 2], dtype=np.int32)
    indices = _convert_to_indices(input_array, system_values)
    assert isinstance(indices, np.ndarray)
    assert indices.dtype == np.int16
    assert indices.shape == (2,)
    assert indices[0] == 0
    assert indices[1] == 2

def test_convert_to_indices_float():
    """Test _convert_to_indices with a float (unlisted type)."""
    # Create a SystemValues object with some default values
    values = {"param1": 1.0, "param2": 2.0, "param3": 3.0}
    system_values = SystemValues(values, np.float64)

    # Test with a float (should raise TypeError)
    with pytest.raises(TypeError):
        _convert_to_indices(1.5, system_values)

def test_convert_to_indices_float_list():
    """Test _convert_to_indices with a list containing floats (unlisted type)."""
    # Create a SystemValues object with some default values
    values = {"param1": 1.0, "param2": 2.0, "param3": 3.0}
    system_values = SystemValues(values, np.float64)

    # Test with a list containing floats (should raise TypeError)
    with pytest.raises(TypeError):
        _convert_to_indices([0, 1.5], system_values)

def test_convert_to_indices_mixed_list():
    """Test _convert_to_indices with a list containing mixed types (unlisted type)."""
    # Create a SystemValues object with some default values
    values = {"param1": 1.0, "param2": 2.0, "param3": 3.0}
    system_values = SystemValues(values, np.float64)

    # Test with a list containing mixed types (should raise TypeError)
    with pytest.raises(TypeError):
        _convert_to_indices(["param1", 1.5], system_values)
