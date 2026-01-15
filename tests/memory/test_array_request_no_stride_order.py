"""Test that ArrayRequest no longer has stride_order field."""

import pytest
import numpy as np
from cubie.memory.array_requests import ArrayRequest


class TestArrayRequestNoStrideOrder:
    """Verify ArrayRequest does not have stride_order field."""

    def test_array_request_no_stride_order_field(self):
        """Verify ArrayRequest does not have stride_order attribute.

        The stride_order field has been removed from ArrayRequest as part
        of the refactoring. Memory manager now uses axis 0 by convention
        for the run axis, eliminating the need for stride_order metadata.
        """
        request = ArrayRequest(
            shape=(100, 3, 50),
            dtype=np.float64,
            memory="device",
        )
        # Verify the field doesn't exist as an attribute
        assert not hasattr(request, "stride_order"), (
            "ArrayRequest should not have stride_order field"
        )

    def test_array_request_rejects_stride_order_parameter(self):
        """Verify ArrayRequest rejects stride_order as a parameter.

        Attempting to pass stride_order to ArrayRequest should raise
        TypeError since this parameter no longer exists.
        """
        with pytest.raises(TypeError):
            ArrayRequest(
                shape=(100, 3, 50),
                dtype=np.float64,
                memory="device",
                stride_order=("run", "variable", "time"),
            )

    def test_array_request_has_expected_fields(self):
        """Verify ArrayRequest has only expected fields.

        After removing stride_order, ArrayRequest should have only:
        - shape
        - dtype
        - memory
        - unchunkable
        """
        request = ArrayRequest(
            shape=(100, 3, 50),
            dtype=np.float32,
            memory="pinned",
            unchunkable=True,
        )
        assert hasattr(request, "shape")
        assert hasattr(request, "dtype")
        assert hasattr(request, "memory")
        assert hasattr(request, "unchunkable")
        assert request.shape == (100, 3, 50)
        assert request.dtype == np.float32
        assert request.memory == "pinned"
        assert request.unchunkable is True
