import pytest
import numpy as np
from cubie.memory.array_requests import ArrayRequest, ArrayResponse
from cubie.memory.mem_manager import MemoryManager


class DummyClass:
    def __init__(self, proportion=None, invalidate_all_hook=None):
        self.proportion = proportion
        self.invalidate_all_hook = invalidate_all_hook


class TestArrayRequests:
    @pytest.mark.parametrize(
        "array_request_override",
        [
            {"shape": (20000,), "dtype": np.float64},
            {"memory": "pinned"},
            {"shape": (10, 10, 10, 10, 10), "dtype": np.float32},
        ],
        indirect=True,
    )
    def test_size(self, array_request):
        assert (
            array_request.size
            == np.prod(array_request.shape) * array_request.dtype().itemsize
        ), "Incorrect size calculated"

    @pytest.mark.parametrize(
        "array_request_override",
        [{"shape": (20000,), "dtype": np.float64}],
        indirect=True,
    )
    def test_instantiation(self, array_request):
        assert array_request.shape == (20000,)
        assert array_request.dtype == np.float64
        assert array_request.memory == "device"
        assert array_request.stride_order == ("time", "variable", "run")

    def test_default_stride_ordering(self):
        # 3D shape default _stride_order
        req3 = ArrayRequest(
            shape=(2, 3, 4),
            dtype=np.float32,
            memory="device",
            stride_order=None,
        )
        assert req3.stride_order == ("time", "variable", "run")
        # 2D shape default _stride_order
        req2 = ArrayRequest(
            shape=(5, 6), dtype=np.float32, memory="device", stride_order=None
        )
        assert req2.stride_order == ("variable", "run")
        # 1D shape leaves _stride_order None
        req1 = ArrayRequest(
            shape=(10,), dtype=np.float32, memory="device", stride_order=None
        )
        assert req1.stride_order is None

    @pytest.mark.parametrize(
        "array_request_override",
        [
            {"shape": (20000,), "dtype": np.float64},
            {"memory": "pinned"},
            {"memory": "mapped"},
        ],
        indirect=True,
    )
    def test_array_response(
        self, array_request, array_request_settings, expected_single_array
    ):
        mgr = MemoryManager()
        instance = DummyClass()
        mgr.register(instance)
        resp = mgr.allocate_all(
            {"test": array_request}, id(instance), stream=0
        )
        arr = resp["test"]

        # Can't directly check for equality as they'll be at different addresses
        assert arr.shape == expected_single_array.shape
        assert type(arr) == type(expected_single_array)
        assert arr.nbytes == expected_single_array.nbytes
        assert arr.strides == expected_single_array.strides
        assert arr.dtype == expected_single_array.dtype


class TestArrayResponse:
    def test_array_response_has_chunked_shapes_field(self):
        """Verify ArrayResponse can be instantiated with chunked_shapes dict."""
        chunked_shapes = {
            "output": (100, 3, 50),
            "state": (3, 50),
        }
        response = ArrayResponse(
            arr={},
            chunks=2,
            chunk_axis="run",
            chunked_shapes=chunked_shapes,
        )
        assert response.chunked_shapes == chunked_shapes
        assert response.chunked_shapes["output"] == (100, 3, 50)
        assert response.chunked_shapes["state"] == (3, 50)

    def test_array_response_chunked_shapes_default_empty(self):
        """Verify chunked_shapes defaults to empty dict when not provided."""
        response = ArrayResponse()
        assert response.chunked_shapes == {}
        assert isinstance(response.chunked_shapes, dict)
