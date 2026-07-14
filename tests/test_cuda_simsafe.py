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
    """Test the real-GPU compilation defaults.

    The default JITFlags enable the nsz, contract, arcp, and afn
    per-flag fast-math options; lineinfo carries the
    CUBIE_LINEINFO-derived default and LTO enables whole-program
    device optimization.
    """
    from cubie.cuda_simsafe import CUDA_SIMULATION, compile_kwargs
    assert CUDA_SIMULATION is False
    assert compile_kwargs["fastmath"] == {"nsz", "contract", "arcp", "afn"}
    assert "lineinfo" in compile_kwargs
    assert compile_kwargs["lto"] is True


@pytest.mark.nocudasim
def test_jit_flags_render_over_live_defaults():
    """Overrides render without mutating the live default flag set."""
    from cubie.cuda_simsafe import JITFlags, compile_kwargs, get_jit_kwargs

    kwargs = get_jit_kwargs(JITFlags(afn=False, lto=False))

    expected = set(compile_kwargs["fastmath"]) - {"afn"}
    assert kwargs["fastmath"] == expected
    assert "afn" in compile_kwargs["fastmath"]
    assert kwargs["lineinfo"] == compile_kwargs["lineinfo"]
    assert kwargs["lto"] is False
    assert compile_kwargs["lto"] is True

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
