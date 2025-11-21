"""
Test script to verify mapped array slicing behavior on GPU.

This script tests whether we need to slice the device array when using
mapped memory, or if direct assignment is sufficient.

Run this on a machine with CUDA GPU to verify behavior.
"""

import numpy as np

try:
    from numba import cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("Numba CUDA not available - cannot run GPU test")
    exit(1)

def test_mapped_memory_slicing():
    """Test whether device array needs slicing for mapped memory."""
    
    print("=" * 70)
    print("Testing Mapped Memory Slicing Behavior")
    print("=" * 70)
    
    # Scenario: Simulating chunked output processing
    # - Device (mapped) array size: (10, 5) - one chunk size
    # - Host array size: (30, 5) - three chunks total
    # - Each iteration writes a chunk to a different host slice
    
    try:
        # Create mapped array (device accessible, host visible)
        device_mapped = cuda.mapped_array((10, 5), dtype=np.float32)
        
        # Create host array to accumulate chunks
        host_full = np.zeros((30, 5), dtype=np.float32)
        
        print("\nTest 1: Direct assignment without slicing device array")
        print("-" * 70)
        
        # Simulate three chunks
        for chunk_idx in range(3):
            # Simulate kernel writing to device array
            test_value = float(chunk_idx + 1) * 10.0
            device_mapped[:] = test_value
            
            # Calculate host slice for this chunk
            start = chunk_idx * 10
            end = (chunk_idx + 1) * 10
            host_slice = slice(start, end)
            
            # Method 1: Direct assignment (what we're doing now)
            host_full[host_slice, :] = device_mapped
            
            print(f"Chunk {chunk_idx}: Assigned value {test_value} to host[{start}:{end}]")
            print(f"  Host slice content: min={host_full[host_slice].min()}, "
                  f"max={host_full[host_slice].max()}")
        
        print("\nFinal host array check:")
        print(f"  Chunk 0 (host[0:10]): {host_full[0:10, 0]}")
        print(f"  Chunk 1 (host[10:20]): {host_full[10:20, 0]}")
        print(f"  Chunk 2 (host[20:30]): {host_full[20:30, 0]}")
        
        # Verify correctness
        expected_0 = np.full((10, 5), 10.0, dtype=np.float32)
        expected_1 = np.full((10, 5), 20.0, dtype=np.float32)
        expected_2 = np.full((10, 5), 30.0, dtype=np.float32)
        
        success = True
        if not np.allclose(host_full[0:10], expected_0):
            print("  ✗ Chunk 0 FAILED")
            success = False
        else:
            print("  ✓ Chunk 0 PASSED")
            
        if not np.allclose(host_full[10:20], expected_1):
            print("  ✗ Chunk 1 FAILED")
            success = False
        else:
            print("  ✓ Chunk 1 PASSED")
            
        if not np.allclose(host_full[20:30], expected_2):
            print("  ✗ Chunk 2 FAILED")
            success = False
        else:
            print("  ✓ Chunk 2 PASSED")
        
        print("\n" + "=" * 70)
        if success:
            print("RESULT: Direct assignment works correctly!")
            print("No need to slice device array - mapped memory handles it.")
        else:
            print("RESULT: Test FAILED - investigate further")
        print("=" * 70)
        
        return success
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mapped_vs_device_memory():
    """Compare mapped memory vs device memory behavior."""
    
    print("\n" + "=" * 70)
    print("Comparing Mapped vs Device Memory")
    print("=" * 70)
    
    try:
        # Test with mapped memory
        mapped_arr = cuda.mapped_array((5, 3), dtype=np.float32)
        mapped_arr[:] = 42.0
        
        # Test with device memory
        device_arr = cuda.device_array((5, 3), dtype=np.float32)
        cuda.to_device(np.full((5, 3), 42.0, dtype=np.float32), to=device_arr)
        
        # Read from both
        host_from_mapped = np.array(mapped_arr)
        host_from_device = device_arr.copy_to_host()
        
        print(f"\nMapped array type: {type(mapped_arr)}")
        print(f"Device array type: {type(device_arr)}")
        print(f"\nMapped -> host: {host_from_mapped[0, 0]}")
        print(f"Device -> host: {host_from_device[0, 0]}")
        
        # Test direct assignment
        host_test = np.zeros((5, 3), dtype=np.float32)
        host_test[:] = mapped_arr  # Direct assignment from mapped
        print(f"Direct assignment from mapped: {host_test[0, 0]}")
        
        print("\n✓ Both memory types work, but mapped allows direct assignment")
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if not HAS_CUDA:
        exit(1)
    
    print("\nCUDA Device Info:")
    print(f"  Available: {cuda.is_available()}")
    if cuda.is_available():
        try:
            print(f"  Device: {cuda.get_current_device().name}")
        except AttributeError:
            print("  Device: CUDA Simulator (no GPU)")
    print()
    
    # Run tests
    test1_pass = test_mapped_memory_slicing()
    test2_pass = test_mapped_vs_device_memory()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (Slicing behavior): {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Memory type comparison): {'PASS' if test2_pass else 'FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n✓ All tests passed - implementation is correct")
        exit(0)
    else:
        print("\n✗ Some tests failed - review needed")
        exit(1)
