"""Tests for cuda_simsafe module functionality."""
import pytest

@pytest.mark.sim_only
def test_compile_kwargs_in_cudasim_mode():
    """Test that compile_kwargs is empty in CUDASIM mode."""
    from cubie.cuda_simsafe import compile_kwargs, CUDA_SIMULATION
    
    assert CUDA_SIMULATION is True
    assert compile_kwargs == {}


@pytest.mark.sim_only
def test_get_jit_kwargs_omits_gpu_options_in_cudasim():
    """Test that per-build GPU options are omitted in CUDASIM."""
    from cubie.cuda_simsafe import get_jit_kwargs

    kwargs = get_jit_kwargs(False, fastmath={"afn": True})

    assert kwargs == {}


@pytest.mark.nocudasim
def test_compile_kwargs_without_cudasim():
    """Test that compile_kwargs contains lineinfo when CUDASIM is disabled."""
    from cubie.cuda_simsafe import CUDA_SIMULATION, compile_kwargs
    assert CUDA_SIMULATION is False
    assert compile_kwargs != {}


@pytest.mark.nocudasim
def test_get_jit_kwargs_merges_fastmath_flags():
    """Test that per-build fast-math flags preserve the defaults."""
    from cubie.cuda_simsafe import compile_kwargs, get_jit_kwargs

    kwargs = get_jit_kwargs(False, fastmath={"afn": True})

    assert kwargs["fastmath"] == {
        "nsz": True,
        "contract": True,
        "arcp": True,
        "afn": True,
    }
    assert "afn" not in compile_kwargs["fastmath"]

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
