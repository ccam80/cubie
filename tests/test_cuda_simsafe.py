"""Tests for cuda_simsafe module functionality."""
import os
import pytest


def test_compile_kwargs_in_cudasim_mode():
    """Test that compile_kwargs is empty in CUDASIM mode."""
    # This test only runs when CUDASIM is enabled
    if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") != "1":
        pytest.skip("Test only runs in CUDASIM mode")
    
    from cubie.cuda_simsafe import compile_kwargs, CUDA_SIMULATION
    
    assert CUDA_SIMULATION is True
    assert compile_kwargs == {}


def test_compile_kwargs_without_cudasim():
    """Test that compile_kwargs contains lineinfo when CUDASIM is disabled."""
    # This test only runs when CUDASIM is NOT enabled
    if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
        pytest.skip("Test only runs without CUDASIM mode")
    
    from cubie.cuda_simsafe import compile_kwargs, CUDA_SIMULATION
    
    assert CUDA_SIMULATION is False
    assert compile_kwargs == {"lineinfo": True}


def test_selp_function_in_cudasim():
    """Test that selp function works in CUDASIM mode."""
    if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") != "1":
        pytest.skip("Test only runs in CUDASIM mode")
    
    from cubie.cuda_simsafe import selp
    
    # Test predicated selection
    assert selp(True, 5.0, 3.0) == 5.0
    assert selp(False, 5.0, 3.0) == 3.0


def test_activemask_function_in_cudasim():
    """Test that activemask function works in CUDASIM mode."""
    if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") != "1":
        pytest.skip("Test only runs in CUDASIM mode")
    
    from cubie.cuda_simsafe import activemask
    
    # In CUDASIM mode, activemask always returns 0xFFFFFFFF
    assert activemask() == 0xFFFFFFFF


def test_all_sync_function_in_cudasim():
    """Test that all_sync function works in CUDASIM mode."""
    if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") != "1":
        pytest.skip("Test only runs in CUDASIM mode")
    
    from cubie.cuda_simsafe import all_sync
    
    # In CUDASIM mode, all_sync just returns the predicate
    assert all_sync(0xFFFFFFFF, True) is True
    assert all_sync(0xFFFFFFFF, False) is False
