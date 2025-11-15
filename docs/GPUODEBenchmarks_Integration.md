# GPUODEBenchmarks Integration - Analysis and Implementation Plan

## Overview

The GPUODEBenchmarks repository (https://github.com/utkarsh530/GPUODEBenchmarks) provides a standardized benchmarking suite for GPU-accelerated ODE solvers. It currently supports:
- **Julia (DiffEqGPU.jl)** - Primary implementation with multiple GPU backends
- **C++ (MPGOS)** - CUDA-based ensemble ODE solver
- **JAX (Diffrax)** - Python-based JAX implementation
- **PyTorch (torchdiffeq)** - PyTorch implementation with vmap

The benchmark suite tests the **Lorenz system** with varying ensemble sizes (from 8 to 2^24 trajectories) and measures performance for both fixed and adaptive time-stepping.

## Requirements to Run Locally on Win11 x86-64 with WSL2

### Environment Prerequisites

Your Win11 environment with WSL2 and Docker is suitable, but you'll need to run the benchmarks **inside WSL2** (not native Windows) for full compatibility:

1. **WSL2 Ubuntu Distribution** (preferred: Ubuntu 20.04 or 22.04)
   - The benchmark suite officially supports Linux
   - WSL2 provides near-native GPU performance through WSL-CUDA

2. **NVIDIA GPU Requirements**
   - GPU with compute capability 6.0+ (GTX 10-series or newer)
   - CUDA Toolkit 11.6+ (11.8 or 12.x recommended)
   - NVIDIA drivers for Windows with WSL support (470.xx or newer)

3. **WSL-CUDA Setup**
   - Install CUDA Toolkit **inside WSL2** (not Windows-side)
   - Verify with: `nvcc --version` and `nvidia-smi`
   - No separate CUDA installation needed on Windows (driver handles GPU access)

### Software Dependencies by Language

#### For Julia (DiffEqGPU.jl)
```bash
# Install Julia 1.8+ (or 1.9+ for AMD GPUs)
wget https://julialang.org/downloads/ # or use juliaup
# Julia packages installed via Project.toml in repo
```

#### For C++ (MPGOS)
```bash
# CUDA Toolkit (includes nvcc)
sudo apt install nvidia-cuda-toolkit  # Or install from NVIDIA
```

#### For JAX (Diffrax)
```bash
# Python 3.9 + conda
conda env create -f GPU_ODE_JAX/environment.yml
conda activate venv_jax
```

#### For PyTorch (torchdiffeq)
```bash
# Python 3.10 + conda
conda env create -f GPU_ODE_PyTorch/environment.yml
conda activate venv_torch
# Custom torchdiffeq version with vmap support
pip install git+https://github.com/utkarsh530/torchdiffeq.git@u/vmap
```

### Repository Setup

```bash
# Inside WSL2
cd ~
git clone https://github.com/utkarsh530/GPUODEBenchmarks.git
cd GPUODEBenchmarks

# Julia setup
julia --project=./GPU_ODE_Julia
julia> using Pkg
julia> Pkg.instantiate()
julia> Pkg.precompile()

# Python environments (if testing JAX/PyTorch)
conda env create -f GPU_ODE_JAX/environment.yml
conda env create -f GPU_ODE_PyTorch/environment.yml
```

### Running Benchmarks

The main script `run_benchmark.sh` orchestrates all benchmarks:

```bash
# Syntax: bash ./run_benchmark.sh -l <language> -d <device> -m <model> [-n <max_trajectories>]
# -l: julia, cpp, jax, pytorch
# -d: gpu, cpu
# -m: ode, sde
# -n: optional, max number of trajectories (default: 2^24)

# Example: Julia GPU ODE benchmark
bash ./run_benchmark.sh -l julia -d gpu -m ode

# Results saved in ./data/<LANGUAGE>/ directory
# Two files: *_adaptive.txt and *_unadaptive.txt
# Format: <num_trajectories> <time_ms>
```

## Requirements to Add Cubie Configuration

To integrate Cubie into the benchmark suite, we need to create a structure similar to the existing Python implementations (JAX/PyTorch).

### 1. Directory Structure

Create a new directory `GPU_ODE_Cubie/` with:
```
GPU_ODE_Cubie/
├── environment.yml          # Conda environment specification
├── requirements.txt         # Pip requirements (cubie + dependencies)
└── bench_cubie.py          # Benchmark script
```

### 2. Benchmark Script Requirements

The `bench_cubie.py` script must:

1. **Accept command-line argument**: Number of trajectories
   ```python
   import sys
   numberOfParameters = int(sys.argv[1])
   ```

2. **Implement Lorenz System**: Match the reference implementation
   ```python
   # Lorenz equations with parameterized rho (28.0 by default)
   dx/dt = sigma * (y - x)         # sigma = 10.0
   dy/dt = x * (rho - z) - y       # rho = parameter (0 to 21)
   dz/dt = x * y - beta * z        # beta = 8/3
   
   # Initial conditions: [1.0, 0.0, 0.0]
   # Time span: 0.0 to 1.0
   ```

3. **Parameter Sweep**: Linear space from 0.0 to 21.0 for rho
   ```python
   parameters = np.linspace(0.0, 21.0, numberOfParameters)
   ```

4. **Two Integration Modes**:
   - **Fixed time-stepping** (unadaptive): Use fixed dt=0.001, no step control
   - **Adaptive time-stepping**: Use rtol=1e-8, atol=1e-8

5. **Benchmark with timeit**: Run 100 repetitions, take minimum time
   ```python
   res = timeit.repeat(lambda: solve_function(), repeat=100, number=1)
   best_time = min(res) * 1000  # Convert to milliseconds
   ```

6. **Save Results**: Append to data files
   ```python
   file = open("./data/CUBIE/Cubie_times_unadaptive.txt", "a+")
   file.write(f'{numberOfParameters} {best_time}\n')
   file.close()
   ```

### 3. Runner Script

Create `runner_scripts/gpu/run_ode_cubie.sh`:
```bash
#!/bin/bash
a=8
max_a=$1
while [ $a -le $max_a ]
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_Cubie/bench_cubie.py $a
    a=$((a*4))
done
```

### 4. Integration with Main Script

Modify `run_benchmark.sh` to add cubie support:
```bash
# Around line 38, add cubie to the condition
elif [[ $lang == "jax" || $lang == "pytorch" || $lang == "cpp" || $lang == "cubie" ]]; then
    # ... existing logic with CUBIE directory
```

### 5. Environment Specification

**environment.yml** for Cubie:
```yaml
name: venv_cubie
channels:
  - nvidia/label/cuda-12.0.0
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy=1.26.4
  - pip
  - pip:
      - cubie
      - numba
      - numba-cuda[cu12]
      - attrs
      - sympy
```

## High-Level Implementation Plan

### Phase 1: Setup and Validation (Local)
**Goal**: Verify Cubie can solve the Lorenz problem correctly

1. **Create Lorenz System in Cubie**
   - Define system using `cubie.create_ODE_system()`
   - Test with single parameter set to verify correctness
   - Compare output with reference implementation (e.g., SciPy)

2. **Implement Batch Solving**
   - Create parameter sweep for rho (0 to 21)
   - Use `cubie.solve_ivp()` with batch parameters
   - Extract final state or save minimal output
   - Verify solutions match expected behavior

3. **Performance Testing**
   - Profile with small ensemble (e.g., 8, 32, 128 trajectories)
   - Ensure GPU utilization is reasonable
   - Check memory usage patterns

### Phase 2: Benchmark Script Development
**Goal**: Create drop-in replacement matching benchmark interface

4. **Implement bench_cubie.py**
   - Fixed time-step integration (RK4 or similar explicit method)
   - Adaptive time-step integration (with error control)
   - Timing harness matching existing implementations
   - Output formatting to match expected format

5. **Create Runner Script**
   - Bash script following existing pattern
   - Handle trajectory scaling (8 → 2^24)
   - Proper error handling and logging

6. **Environment Configuration**
   - Create environment.yml for reproducibility
   - Document CUDA version compatibility
   - Test environment installation from scratch

### Phase 3: Integration and Testing
**Goal**: Integrate into GPUODEBenchmarks suite

7. **Modify Main Benchmark Script**
   - Add cubie option to run_benchmark.sh
   - Create data output directory structure
   - Handle errors gracefully

8. **Validation Testing**
   - Run benchmark suite with small trajectory counts
   - Compare results with other implementations
   - Verify output file format
   - Test error handling

9. **Full Benchmark Run**
   - Run complete benchmark (8 → 2^24 trajectories)
   - Generate performance plots
   - Document any performance characteristics
   - Compare with Julia/JAX/PyTorch/MPGOS

### Phase 4: Documentation and Contribution
**Goal**: Prepare for potential upstream contribution

10. **Documentation**
    - Add Cubie section to README.md
    - Document installation requirements
    - Provide usage examples
    - Note any limitations or differences

11. **Performance Analysis**
    - Create comparison plots
    - Analyze scaling behavior
    - Document optimal trajectory counts
    - Identify performance bottlenecks if any

12. **Optional: Upstream PR**
    - Fork GPUODEBenchmarks repository
    - Create feature branch
    - Submit pull request with Cubie support
    - Address review comments

## Cubie-Specific Implementation Considerations

### 1. Algorithm Selection

Cubie supports multiple integration algorithms. For fair comparison:

- **Fixed time-step**: Use explicit RK4 (matches other implementations)
  ```python
  solution = qb.solve_ivp(system, initial_conditions,
                          duration=1.0,
                          algorithm='RK4',
                          dt=0.001,
                          adaptive=False)
  ```

- **Adaptive time-step**: Use Tsit5 or similar adaptive method
  ```python
  solution = qb.solve_ivp(system, initial_conditions,
                          duration=1.0,
                          algorithm='Tsit5',
                          rtol=1e-8, atol=1e-8,
                          adaptive=True)
  ```

### 2. Batch Configuration

Cubie's batch solving interface allows parameter sweeps:

```python
# Single parameter varying (rho from 0 to 21)
parameters = {
    'sigma': 10.0,           # Fixed
    'beta': 8.0/3.0,        # Fixed
    'rho': np.linspace(0.0, 21.0, num_trajectories)  # Varying
}

initial_conditions = {
    'x': 1.0,
    'y': 0.0,
    'z': 0.0
}

solution = qb.solve_ivp(lorenz_system, initial_conditions,
                        parameters=parameters,
                        duration=1.0)
```

### 3. Output Handling

For benchmarking, we only need timing, not full trajectories:

```python
# Option 1: Save only final state (minimal memory)
solution = qb.solve_ivp(..., save_at=[1.0])

# Option 2: Use summary statistics if available
# (Cubie supports extracting only needed outputs)
```

### 4. Memory Management

For large ensembles (2^24 trajectories):

```python
# May need to batch if memory limited
# Split into chunks if GPU memory is insufficient
max_batch_size = 2**20  # Adjust based on GPU memory
for batch_start in range(0, num_trajectories, max_batch_size):
    batch_end = min(batch_start + max_batch_size, num_trajectories)
    # Solve batch
    # Accumulate timing
```

### 5. Precision Considerations

GPUODEBenchmarks implementations vary in precision:
- Julia: Can use Float32 or Float64
- MPGOS: Configurable (uses Float32 in examples)
- JAX/PyTorch: Typically Float32 on GPU

Cubie should match (likely Float32 for fair comparison, Float64 for accuracy).

## Expected Challenges and Solutions

### Challenge 1: API Differences
**Issue**: Cubie's API may differ from batch-friendly JAX/PyTorch approaches

**Solution**: 
- Carefully structure parameter arrays to match Cubie's expected format
- Use Cubie's batch solving interface appropriately
- May need wrapper functions for cleaner integration

### Challenge 2: Performance Optimization
**Issue**: Initial implementation may not be optimally tuned

**Solution**:
- Profile to identify bottlenecks
- Adjust batch sizes for optimal GPU utilization
- Ensure proper use of Cubie's caching and compilation
- Consider warm-up runs before timing

### Challenge 3: Large Ensemble Memory
**Issue**: 2^24 trajectories may exceed GPU memory

**Solution**:
- Implement batching strategy
- Only save minimal required data (final state or timing only)
- Use Cubie's memory management features if available

### Challenge 4: WSL2 Performance
**Issue**: WSL2 may have overhead compared to native Linux

**Solution**:
- Ensure proper WSL-CUDA setup
- Use latest NVIDIA drivers with WSL support
- Consider dual-boot Linux for production benchmarks
- Document any performance differences

## Success Criteria

The implementation is successful when:

1. **Correctness**: Cubie produces numerically correct solutions (within tolerance of other solvers)
2. **Interface Compatibility**: Integrates cleanly with `run_benchmark.sh` script
3. **Output Format**: Generates data files matching expected format
4. **Scalability**: Successfully runs from 8 to 2^24 trajectories
5. **Performance**: Completes benchmark suite in reasonable time (<30 mins)
6. **Reproducibility**: Can be installed and run from environment.yml
7. **Documentation**: Clear instructions for setup and usage

## Next Steps for Implementation Specialist

The implementation specialist should:

1. **Start Local**: Set up GPUODEBenchmarks in WSL2, verify existing benchmarks run
2. **Develop Incrementally**: 
   - First: Single trajectory Lorenz system
   - Second: Small batch (8-128 trajectories)
   - Third: Full benchmark script
   - Fourth: Integration with suite
3. **Test Thoroughly**: Compare results with reference implementations
4. **Optimize**: Profile and tune for best performance
5. **Document**: Capture setup steps, issues, and solutions

## Appendix: Reference Implementations

### Lorenz System Definition (Mathematical)
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y  
dz/dt = xy - βz

where:
σ = 10.0 (fixed)
β = 8/3 (fixed)
ρ ∈ [0, 21] (parameter varied across ensemble)

Initial conditions: (x₀, y₀, z₀) = (1.0, 0.0, 0.0)
Time span: t ∈ [0, 1]
```

### JAX Implementation (Reference)
```python
class Lorenz(eqx.Module):
    k1: float  # This is rho
    
    def __call__(self, t, y, args):
        f0 = 10.0*(y[1] - y[0])                    # dx/dt
        f1 = self.k1 * y[0] - y[1] - y[0] * y[2]  # dy/dt
        f2 = y[0] * y[1] - (8/3)*y[2]             # dz/dt
        return jnp.stack([f0, f1, f2])
```

### Performance Expectations

Based on existing benchmarks:
- **Julia (DiffEqGPU)**: Fastest, especially at large scales
- **MPGOS (C++)**: Competitive, hand-optimized CUDA
- **JAX**: Good performance, JIT compilation overhead
- **PyTorch**: Slower due to vmap limitations

Cubie should aim for performance between JAX and Julia, given it uses:
- Numba JIT compilation (similar to JAX)
- Direct CUDA kernel generation (similar to MPGOS)
- Python interface (similar to JAX/PyTorch)

Target: Within 2x of Julia performance, competitive with or better than JAX.
