"""Tests for cuda_simsafe module functionality."""
import numpy as np

import pytest

@pytest.mark.sim_only
def test_compile_kwargs_in_cudasim_mode():
    """Test that compile_kwargs is empty in CUDASIM mode."""
    from cubie.cuda_simsafe import compile_kwargs, CUDA_SIMULATION
    
    assert CUDA_SIMULATION is True
    assert compile_kwargs == {}


@pytest.mark.nocudasim
def test_compile_kwargs_without_cudasim():
    """Test that compile_kwargs contains lineinfo when CUDASIM is disabled."""
    from cubie.cuda_simsafe import CUDA_SIMULATION, compile_kwargs
    assert CUDA_SIMULATION is False
    assert compile_kwargs != {}


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


@pytest.mark.nocudasim
@pytest.mark.mlir_only
def test_multiassign_locals_use_semantic_stack_slots():
    """Test compiler locals use value types while arrays use storage."""
    from cubie.cuda_simsafe import compile_kwargs, cuda

    @cuda.jit(**compile_kwargs)
    def stack_slot_kernel(flags, values, output):
        index = cuda.grid(1)
        if index >= values.size:
            return

        value = values[index]
        flag = flags[index]
        flag = flag or value > np.float32(0.0)
        packed = (flag, flag)
        mixed = (flag, value)
        if value > np.float32(0.5):
            flag = not flag
            packed = (flag, not packed[1])
            mixed = (packed[0], value + np.float32(1.0))
        result = mixed[1] if mixed[0] else -mixed[1]
        output[index] = -result if packed[1] else result

    @cuda.jit(**compile_kwargs)
    def optional_stack_kernel(flags, values, output):
        index = cuda.grid(1)
        if index >= values.size:
            return

        optional = None
        if flags[index]:
            optional = values[index]
        output[index] = 1 if optional is None else 2

    flags = np.array([False, False, True])
    values = np.array([-1.0, 0.25, 0.75], dtype=np.float32)
    output = np.zeros_like(values)
    optional_output = np.zeros(3, dtype=np.int32)

    stack_slot_kernel[1, 32](flags, values, output)
    optional_stack_kernel[1, 32](flags, values, optional_output)
    cuda.synchronize()

    np.testing.assert_array_equal(
        output, np.array([1.0, -0.25, -1.75], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        optional_output, np.array([1, 1, 2], dtype=np.int32)
    )

    stack_modules = stack_slot_kernel.inspect_mlir()
    stack_mlir = next(iter(stack_modules.values()))
    assert "memref<?xi8" in stack_mlir
    assert "memref<2xi1>" in stack_mlir
    assert (
        stack_mlir.count("memref.alloca() : memref<1xi1>") >= 2
    )
    assert "memref.alloca() : memref<1xf32>" in stack_mlir

    optional_modules = optional_stack_kernel.inspect_mlir()
    optional_mlir = next(iter(optional_modules.values()))
    assert "llvm.alloca" in optional_mlir
    assert "!llvm.struct<(f32, i1)>" in optional_mlir


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
