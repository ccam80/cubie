"""Tests for chunked array transfers between host and device."""

import pytest
import numpy as np
from numpy import float32 as np_float32, int32 as np_int32

from cubie.memory.mem_manager import MemoryManager
from cubie.memory.array_requests import ArrayRequest


class MockMemoryManager(MemoryManager):
    """Mock memory manager for testing with controlled memory info."""

    def get_memory_info(self):
        return int(0.1 * 1024**3), int(0.1 * 1024**3)


@pytest.fixture
def test_memory_manager():
    """Create a memory manager for testing."""
    return MockMemoryManager()


class TestChunkArraysSkipsMissingAxis:
    """Test chunk_arrays handles arrays without the chunk axis."""

    def test_chunk_arrays_skips_2d_array_when_chunking_time(
        self, test_memory_manager
    ):
        """2D arrays with (variable, run) should not be chunked on time axis."""
        mgr = test_memory_manager
        requests = {
            "input_2d": ArrayRequest(
                shape=(10, 50),
                dtype=np_float32,
                memory="device",
                stride_order=("variable", "run"),
            ),
            "output_3d": ArrayRequest(
                shape=(100, 10, 50),
                dtype=np_float32,
                memory="device",
                stride_order=("time", "variable", "run"),
            ),
        }

        chunked = mgr.chunk_arrays(requests, numchunks=4, axis="time")

        # 2D array should be unchanged (no time axis)
        assert chunked["input_2d"].shape == (10, 50)
        # 3D array should be chunked on time axis
        assert chunked["output_3d"].shape == (25, 10, 50)

    def test_chunk_arrays_skips_1d_status_codes(self, test_memory_manager):
        """1D status code arrays should not be chunked."""
        mgr = test_memory_manager
        requests = {
            "status_codes": ArrayRequest(
                shape=(100,),
                dtype=np_int32,
                memory="device",
                stride_order=("run",),
                unchunkable=True,
            ),
        }

        chunked = mgr.chunk_arrays(requests, numchunks=4, axis="time")

        # Unchunkable array should be unchanged
        assert chunked["status_codes"].shape == (100,)

    def test_chunk_arrays_handles_run_axis_correctly(
        self, test_memory_manager
    ):
        """Arrays should be correctly chunked on run axis."""
        mgr = test_memory_manager
        requests = {
            "input_2d": ArrayRequest(
                shape=(10, 50),
                dtype=np_float32,
                memory="device",
                stride_order=("variable", "run"),
            ),
            "output_3d": ArrayRequest(
                shape=(100, 10, 50),
                dtype=np_float32,
                memory="device",
                stride_order=("time", "variable", "run"),
            ),
        }

        chunked = mgr.chunk_arrays(requests, numchunks=5, axis="run")

        # Both arrays have run axis, both should be chunked
        assert chunked["input_2d"].shape == (10, 10)  # ceil(50/5)
        assert chunked["output_3d"].shape == (100, 10, 10)


class TestChunkedHostSliceTransfers:
    """Test contiguous slice handling for chunked transfers."""

    def test_noncontiguous_host_slice_detected(self):
        """Verify that slicing a host array creates non-contiguous views."""
        host_array = np.zeros((100, 10, 50), dtype=np_float32)
        # Slice on run axis (last axis)
        host_slice = host_array[:, :, 0:10]
        # This slice is NOT contiguous because the parent has
        # strides based on shape (100, 10, 50), not (100, 10, 10)
        assert not host_slice.flags["C_CONTIGUOUS"]

    def test_contiguous_copy_matches_shape(self):
        """Verify ascontiguousarray creates matching-shape contiguous array."""
        host_array = np.arange(100 * 10 * 50, dtype=np_float32).reshape(
            100, 10, 50
        )
        host_slice = host_array[:, :, 0:10]
        contiguous = np.ascontiguousarray(host_slice)

        assert contiguous.flags["C_CONTIGUOUS"]
        assert contiguous.shape == (100, 10, 10)
        np.testing.assert_array_equal(contiguous, host_slice)
