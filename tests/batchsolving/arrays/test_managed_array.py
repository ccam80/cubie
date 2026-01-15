"""Tests for ManagedArray chunk metadata fields."""

from cubie.batchsolving.arrays.BaseArrayManager import ManagedArray


def test_managed_array_chunk_fields_default_none():
    """Verify that chunk_length and num_chunks default to None.
    
    ManagedArray instances should initialize with chunk metadata fields
    set to None, indicating that the array has not been allocated with
    chunking parameters yet.
    """
    # Create a minimal ManagedArray with no chunk parameters
    managed_array = ManagedArray(
        dtype=float,
        stride_order=('run', 'state'),
        default_shape=(100, 3),
        memory_type='device',
        is_chunked=True,
    )
    
    # Verify chunk metadata fields default to None
    assert managed_array.chunk_length is None
    assert managed_array.num_chunks is None


def test_managed_array_chunk_fields_accept_valid_values():
    """Verify that valid chunk_length and num_chunks can be set.
    
    ManagedArray should accept valid integer values for chunk_length
    and num_chunks when provided during initialization or after
    allocation.
    """
    # Create ManagedArray with explicit chunk parameters
    managed_array = ManagedArray(
        dtype=float,
        stride_order=('run', 'state'),
        default_shape=(100, 3),
        memory_type='device',
        is_chunked=True,
        chunk_length=25,
        num_chunks=4,
    )
    
    # Verify chunk metadata fields are set correctly
    assert managed_array.chunk_length == 25
    assert managed_array.num_chunks == 4
    
    # Test updating chunk parameters after initialization
    managed_array.chunk_length = 50
    managed_array.num_chunks = 2
    
    assert managed_array.chunk_length == 50
    assert managed_array.num_chunks == 2
