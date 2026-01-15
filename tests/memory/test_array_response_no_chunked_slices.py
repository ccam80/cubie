"""Test to verify ArrayResponse does not have chunked_slices field.

This test verifies that the chunked_slices field has been successfully
removed from ArrayResponse as part of the chunking refactor.
"""

import pytest
from cubie.memory.array_requests import ArrayResponse


class TestArrayResponseNoChunkedSlices:
    def test_array_response_no_chunked_slices_field(self):
        """Verify ArrayResponse does not have chunked_slices attribute.
        
        The chunked_slices field has been removed as part of the chunk
        refactoring. This test ensures it doesn't exist.
        """
        response = ArrayResponse()
        
        # Verify the field doesn't exist as an attribute
        assert not hasattr(response, "chunked_slices"), (
            "ArrayResponse should not have chunked_slices field"
        )
        
        # Verify the fields that should exist
        assert hasattr(response, "arr")
        assert hasattr(response, "chunks")
        assert hasattr(response, "axis_length")
        assert hasattr(response, "chunk_length")
        assert hasattr(response, "dangling_chunk_length")
        assert hasattr(response, "chunked_shapes")
