"""Tests for cuda_simsafe module functionality."""
import pytest

@pytest.mark.sim_only
def test_compile_kwargs_in_cudasim_mode():
    """Test that compile_kwargs is empty in CUDASIM mode."""
    from cubie.cuda_simsafe import compile_kwargs, CUDA_SIMULATION
    
    assert CUDA_SIMULATION is True
    assert compile_kwargs == {}

@pytest.mark.nocudasim
def test_compile_kwargs_without_cudasim():
    """Test that compile_kwargs selects the per-flag fastmath set.

    numba-cuda-mlir accepts per-flag fastmath (natively on patched
    builds, via cubie._mlir_compat's selective-fastmath shim on the
    stock wheel): no signed zeros, FMA contraction and approximate
    division.
    """
    from cubie.cuda_simsafe import CUDA_SIMULATION, compile_kwargs
    assert CUDA_SIMULATION is False
    assert compile_kwargs == {"fastmath": {"nsz", "contract", "arcp"}}

@pytest.mark.sim_only
def test_selp_function_in_cudasim():
    """Test that selp function works in CUDASIM mode."""
    from cubie.cuda_simsafe import selp
    
    # Test predicated selection
    assert selp(True, 5.0, 3.0) == 5.0
    assert selp(False, 5.0, 3.0) == 3.0

@pytest.mark.sim_only
def test_activemask_function_in_cudasim():
    """Test that activemask function works in CUDASIM mode."""
    from cubie.cuda_simsafe import activemask
    
    # In CUDASIM mode, activemask always returns 0xFFFFFFFF
    assert activemask() == 0xFFFFFFFF

@pytest.mark.sim_only
def test_all_sync_function_in_cudasim():
    """Test that all_sync function works in CUDASIM mode."""
    from cubie.cuda_simsafe import all_sync
    
    # In CUDASIM mode, all_sync just returns the predicate
    assert all_sync(0xFFFFFFFF, True) is True
    assert all_sync(0xFFFFFFFF, False) is False
