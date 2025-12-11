import pytest
from numba import cuda
from cubie.memory.array_requests import ArrayRequest
import numpy as np


@pytest.fixture(scope="session")
def array_request_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="session")
def array_request_settings(array_request_override):
    """Fixture to provide settings for ArrayRequest."""
    defaults = {
        "shape": (1, 1, 1),
        "dtype": np.float32,
        "memory": "device",
        "stride_order": ("time", "variable", "run"),
    }
    if array_request_override:
        for key, value in array_request_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults


@pytest.fixture(scope="session")
def array_request(array_request_settings):
    return ArrayRequest(**array_request_settings)


@pytest.fixture(scope="session")
def expected_single_array(array_request_settings):
    arr_request = array_request_settings
    if arr_request["memory"] == "device":
        arr = cuda.device_array(
            array_request_settings["shape"],
            dtype=array_request_settings["dtype"],
        )
    elif arr_request["memory"] == "pinned":
        arr = cuda.pinned_array(
            array_request_settings["shape"],
            dtype=array_request_settings["dtype"],
        )
    elif arr_request["memory"] == "mapped":
        arr = cuda.mapped_array(
            array_request_settings["shape"],
            dtype=array_request_settings["dtype"],
        )
    elif arr_request["memory"] == "managed":
        raise NotImplementedError("Managed memory not implemented")
    else:
        raise ValueError(f"Invalid memory type: {arr_request['memory']}")
    return arr
