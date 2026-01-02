"""Benchmark script for measuring CUDA compilation time.

Run with: python tests/benchmark_compile_time.py

Measures the time taken to compile a representative CUDAFactory
subclass (ERKStep) and reports timing information.

This script works in both real CUDA and CUDASIM modes. The results
help verify that explicit imports reduce the scope captured by Numba
during CUDA JIT compilation.

Results
-------
Baseline (with `import attrs`):
    [To be filled after running benchmark]

After optimization (with `from attrs import define, field, validators`):
    [To be filled after running benchmark]
"""
import time

import numpy as np
from numba import cuda


def create_test_dxdt_function():
    """Create a simple dxdt device function for benchmarking."""
    @cuda.jit(device=True, inline=True)
    def dxdt(state, parameters, drivers, observables, derivatives, t):
        # Simple exponential decay: dx/dt = -k * x
        for i in range(len(state)):
            derivatives[i] = -parameters[0] * state[i]
    return dxdt


def create_test_observables_function():
    """Create a simple observables device function for benchmarking."""
    @cuda.jit(device=True, inline=True)
    def observables(state, parameters, drivers, obs_out, t):
        pass
    return observables


def measure_compilation_time():
    """Measure compilation time for ERKStep device function.

    This function measures the wall-clock time taken to compile the
    device_function property of an ERKStep instance. The compilation
    is triggered by accessing the step property for the first time.

    Returns
    -------
    float
        Compilation time in seconds.
    """
    from cubie.integrators.algorithms.generic_erk import ERKStep

    # Create required device functions
    dxdt_fn = create_test_dxdt_function()
    observables_fn = create_test_observables_function()

    # Create ERKStep instance
    step = ERKStep(
        precision=np.float64,
        n=3,
        dxdt_function=dxdt_fn,
        observables_function=observables_fn,
    )

    # Measure compilation time by accessing the step cache property
    start_time = time.perf_counter()
    _ = step.step_cache
    end_time = time.perf_counter()

    compilation_time = end_time - start_time
    return compilation_time


def main():
    """Run the compilation benchmark and report results."""
    print("CUBiE Compilation Time Benchmark")
    print("=" * 40)
    print()

    # Report CUDA mode
    from cubie.cuda_simsafe import CUDA_SIMULATION
    if CUDA_SIMULATION:
        print("Running in CUDA SIMULATION mode (no GPU)")
    else:
        print("Running with real CUDA hardware")
    print()

    print("Measuring ERKStep compilation time...")
    compilation_time = measure_compilation_time()

    print(f"Compilation time: {compilation_time:.4f} seconds")
    print()
    print("Note: In CUDASIM mode, compilation times may not reflect")
    print("actual CUDA compilation overhead. Run with a real GPU for")
    print("accurate measurements.")


if __name__ == "__main__":
    main()
