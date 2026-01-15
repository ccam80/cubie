import pytest
import numpy as np
from cubie.memory.array_requests import ArrayResponse
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
        """Verify ArrayResponse can be instantiated with chunked_shapes dict.

        Chunking is always performed along the run axis.
        """
        chunked_shapes = {
            "output": (100, 3, 50),
            "state": (3, 50),
        }
        response = ArrayResponse(
            arr={},
            chunks=2,
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


def test_array_request_chunk_axis_index_validation():
    """Verify ArrayRequest validates chunk_axis_index correctly.

    The chunk_axis_index field should:
    - Accept None (no chunking axis)
    - Accept non-negative integers (valid axis indices)
    - Reject negative integers (invalid axis indices)
    - Default to 2 (run axis in standard stride order)
    """
    from cubie.memory.array_requests import ArrayRequest

    # Test default value is 2
    request = ArrayRequest(dtype=np.float64)
    assert request.chunk_axis_index == 2

    # Test None is accepted (no chunking axis)
    request_none = ArrayRequest(dtype=np.float64, chunk_axis_index=None)
    assert request_none.chunk_axis_index is None

    # Test non-negative integers are accepted
    request_0 = ArrayRequest(dtype=np.float64, chunk_axis_index=0)
    assert request_0.chunk_axis_index == 0

    request_1 = ArrayRequest(dtype=np.float64, chunk_axis_index=1)
    assert request_1.chunk_axis_index == 1

    request_5 = ArrayRequest(dtype=np.float64, chunk_axis_index=5)
    assert request_5.chunk_axis_index == 5

    # Test negative values are rejected
    with pytest.raises(ValueError, match="must be >= 0"):
        ArrayRequest(dtype=np.float64, chunk_axis_index=-1)

    with pytest.raises(ValueError, match="must be >= 0"):
        ArrayRequest(dtype=np.float64, chunk_axis_index=-5)


def test_array_request_accepts_total_runs():
    """Verify ArrayRequest can be created with total_runs=100.

    The total_runs field carries the number of runs for chunking calculations
    and should accept positive integers.
    """
    from cubie.memory.array_requests import ArrayRequest

    # Create ArrayRequest with total_runs=100
    request = ArrayRequest(dtype=np.float64, total_runs=100)
    assert request.total_runs == 100

    # Verify other fields still work correctly
    assert request.dtype == np.float64
    assert request.chunk_axis_index == 2


def test_array_request_validates_total_runs_positive():
    """Verify ArrayRequest raises ValueError for total_runs=0 or negative.

    The total_runs field must be >= 1 when provided, as it represents a count
    of runs to process. Zero or negative values should be rejected.
    """
    from cubie.memory.array_requests import ArrayRequest

    # Test zero is rejected
    with pytest.raises(ValueError, match="must be >= 1"):
        ArrayRequest(dtype=np.float64, total_runs=0)

    # Test negative values are rejected
    with pytest.raises(ValueError, match="must be >= 1"):
        ArrayRequest(dtype=np.float64, total_runs=-1)

    with pytest.raises(ValueError, match="must be >= 1"):
        ArrayRequest(dtype=np.float64, total_runs=-100)


def test_array_request_total_runs_defaults_to_none():
    """Verify total_runs defaults to None when not provided.

    When total_runs is None, the array is not intended for run-axis chunking,
    such as driver_coefficients which don't vary per run.
    """
    from cubie.memory.array_requests import ArrayRequest

    # Create ArrayRequest without specifying total_runs
    request = ArrayRequest(dtype=np.float64)
    assert request.total_runs is None

    # Verify None can be explicitly set
    request_explicit = ArrayRequest(dtype=np.float64, total_runs=None)
    assert request_explicit.total_runs is None
