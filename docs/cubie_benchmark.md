# Cubie Benchmark Implementation Guide

This document provides complete instructions for implementing Cubie support in a personal fork of the GPUODEBenchmarks repository.

## Objective

Add Cubie as a benchmarked GPU ODE solver to GPUODEBenchmarks, enabling direct performance comparisons with Julia/DiffEqGPU, JAX/Diffrax, PyTorch/torchdiffeq, and C++/MPGOS implementations.

## Repository Structure Overview

The GPUODEBenchmarks repository follows this pattern for each solver:

```
GPUODEBenchmarks/
├── GPU_ODE_<Language>/          # Solver-specific directory
│   ├── environment.yml          # Conda environment spec
│   └── bench_*.py              # Benchmark script
├── runner_scripts/
│   └── gpu/
│       └── run_ode_<language>.sh  # Runner script
├── run_benchmark.sh             # Main orchestration script
└── data/
    └── <LANGUAGE>/              # Output directory (created at runtime)
```

## Implementation Tasks

### Task 1: Create GPU_ODE_Cubie Directory

**Location**: `GPU_ODE_Cubie/`

Create the following files in this directory:

#### File 1.1: `GPU_ODE_Cubie/environment.yml`

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

#### File 1.2: `GPU_ODE_Cubie/bench_cubie.py`

This is the main benchmark script. It must:
1. Accept trajectory count as command-line argument
2. Define the Lorenz system matching the reference specification
3. Run both fixed and adaptive time-stepping benchmarks
4. Save results in the expected format

```python
#!/usr/bin/env python
# coding: utf-8
"""
Benchmarking Cubie ODE solvers for ensemble problems.
The Lorenz ODE is integrated with fixed and adaptive time-stepping.

Created for GPUODEBenchmarks integration
"""

import sys
import timeit
import numpy as np
import cubie as qb

# Get number of trajectories from command line
numberOfParameters = int(sys.argv[1])

# ========================================
# LORENZ SYSTEM DEFINITION
# ========================================
# Mathematical definition:
#   dx/dt = sigma * (y - x)
#   dy/dt = x * (rho - z) - y
#   dz/dt = x * y - beta * z
#
# Where:
#   sigma = 10.0 (fixed)
#   beta = 8/3 (fixed)  
#   rho = parameter varied from 0 to 21

lorenz_system = qb.create_ODE_system(
    """
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    """,
    states={'x': 1.0, 'y': 0.0, 'z': 0.0},
    parameters={'sigma': 10.0, 'beta': 8.0/3.0, 'rho': 21.0},
    name="Lorenz"
)

# ========================================
# PARAMETER SWEEP SETUP
# ========================================
# Create linear space from 0 to 21 for rho parameter
parameterList = np.linspace(0.0, 21.0, numberOfParameters)

# Build parameter dictionary for batch solve
# All parameters except rho are scalar (same for all trajectories)
# rho varies across the ensemble
parameters = {
    'sigma': 10.0,
    'beta': 8.0/3.0,
    'rho': parameterList
}

# Initial conditions (same for all trajectories)
initial_conditions = {
    'x': 1.0,
    'y': 0.0,
    'z': 0.0
}

# ========================================
# FIXED TIME-STEPPING BENCHMARK
# ========================================
print(f"Running {numberOfParameters} trajectories with fixed time-stepping...")

def solve_fixed():
    """Solve with fixed time step (unadaptive)."""
    solution = qb.solve_ivp(
        lorenz_system,
        initial_conditions,
        duration=1.0,
        parameters=parameters,
        algorithm='RK4',
        dt=0.001,
        adaptive=False,
        save_at=[1.0]  # Only save final state
    )
    return solution

# Warm-up run (JIT compilation)
_ = solve_fixed()

# Benchmark with 100 repetitions
res = timeit.repeat(lambda: solve_fixed(), repeat=100, number=1)

best_time = min(res) * 1000  # Convert to milliseconds
print(f"{numberOfParameters} ODE solves with fixed time-stepping completed in {best_time:.1f} ms")

# Save results
import os
os.makedirs("./data/CUBIE", exist_ok=True)
with open("./data/CUBIE/Cubie_times_unadaptive.txt", "a+") as file:
    file.write(f'{numberOfParameters} {best_time}\n')

# ========================================
# ADAPTIVE TIME-STEPPING BENCHMARK
# ========================================
print(f"Running {numberOfParameters} trajectories with adaptive time-stepping...")

def solve_adaptive():
    """Solve with adaptive time step."""
    solution = qb.solve_ivp(
        lorenz_system,
        initial_conditions,
        duration=1.0,
        parameters=parameters,
        algorithm='Tsit5',  # Tsitouras 5(4) method
        rtol=1e-8,
        atol=1e-8,
        adaptive=True,
        save_at=[1.0]  # Only save final state
    )
    return solution

# Warm-up run (JIT compilation)
_ = solve_adaptive()

# Benchmark with 100 repetitions
res = timeit.repeat(lambda: solve_adaptive(), repeat=100, number=1)

best_time = min(res) * 1000  # Convert to milliseconds
print(f"{numberOfParameters} ODE solves with adaptive time-stepping completed in {best_time:.1f} ms")

# Save results
with open("./data/CUBIE/Cubie_times_adaptive.txt", "a+") as file:
    file.write(f'{numberOfParameters} {best_time}\n')
```

**Important Implementation Notes**:

1. **Algorithm Selection**:
   - Fixed: Use `'RK4'` to match other implementations
   - Adaptive: Use `'Tsit5'` (Tsitouras 5th order) for fair comparison

2. **Tolerance Settings**:
   - Match JAX/Julia adaptive settings: `rtol=1e-8, atol=1e-8`

3. **Output Minimization**:
   - Use `save_at=[1.0]` to only save final state
   - Minimizes memory usage for large ensembles

4. **Warm-up Runs**:
   - First execution triggers JIT compilation
   - Run once before timing to ensure fair benchmarks

5. **Memory Considerations**:
   - For very large ensembles (>2^20), may need batching
   - Monitor GPU memory and implement chunking if needed

### Task 2: Create Runner Script

**Location**: `runner_scripts/gpu/run_ode_cubie.sh`

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

**Make executable**:
```bash
chmod +x runner_scripts/gpu/run_ode_cubie.sh
```

### Task 3: Modify Main Orchestration Script

**Location**: `run_benchmark.sh`

**Modification**: Add cubie to the language options

Find this section (around line 38):
```bash
elif [[ $lang == "jax"  ||  $lang == "pytorch" || $lang == "cpp" ]]; then
```

Change to:
```bash
elif [[ $lang == "jax"  ||  $lang == "pytorch" || $lang == "cpp" || $lang == "cubie" ]]; then
```

**Complete modified section**:
```bash
elif [[ $lang == "jax"  ||  $lang == "pytorch" || $lang == "cpp" || $lang == "cubie" ]]; then
    if [[ $model != "ode" || $dev != "gpu" ]]; then
        echo "The benchmarking of ensemble ${model^^} solvers on ${dev^^} with ${lang} is not supported. Please use -m flag with \"ode\" and -d with \"gpu\"."
        exit 1
    else
        echo "Benchmarking ${lang^^} ${dev^^} accelerated ensemble ${model^^} solvers..."
        if [ -d "./data/${lang^^}" ] 
        then
            rm -rf "./data/${lang^^}"/*
            mkdir -p "./data/${lang^^}"
        else
            mkdir -p "./data/${lang^^}"
        fi
        cmd="./runner_scripts/${dev}/run_${model}_${lang}.sh ${nmax}"
        eval "$cmd"
    fi
fi
```

### Task 4: Create Data Output Directory Structure

The script will create this automatically, but for reference:

```
data/
└── CUBIE/
    ├── Cubie_times_unadaptive.txt
    └── Cubie_times_adaptive.txt
```

**Output Format**: Space-delimited, one line per trajectory count
```
<num_trajectories> <time_milliseconds>
```

Example:
```
8 0.5
32 1.2
128 4.8
512 18.6
```

### Task 5: Optional - Add Plotting Support

To include Cubie in comparison plots, modify plotting scripts.

**Location**: `runner_scripts/plot/plot_ode_comp.jl`

This requires Julia knowledge. The basic approach:

1. Read Cubie data files
2. Add Cubie series to plots
3. Use appropriate color/marker for Cubie

Example addition:
```julia
# Read Cubie data
cubie_unadaptive = readdlm("../data/CUBIE/Cubie_times_unadaptive.txt")
cubie_adaptive = readdlm("../data/CUBIE/Cubie_times_adaptive.txt")

# Add to plot
plot!(cubie_unadaptive[:,1], cubie_unadaptive[:,2], 
      label="Cubie", marker=:diamond, linewidth=2)
```

## Testing and Validation

### Test 1: Small Scale Validation

```bash
# Test with 32 trajectories
bash ./run_benchmark.sh -l cubie -d gpu -m ode -n 32

# Verify output files exist
ls -la data/CUBIE/
cat data/CUBIE/Cubie_times_unadaptive.txt
cat data/CUBIE/Cubie_times_adaptive.txt
```

**Expected output**: Two files with one line each showing trajectory count and time.

### Test 2: Scaling Test

```bash
# Test trajectory scaling: 8, 32, 128, 512
bash ./run_benchmark.sh -l cubie -d gpu -m ode -n 512

# Check that times scale reasonably
cat data/CUBIE/Cubie_times_unadaptive.txt
```

**Expected**: Times should increase roughly linearly with trajectory count (may have overhead for small counts).

### Test 3: Comparison with Other Implementations

```bash
# Run Julia for comparison
bash ./run_benchmark.sh -l julia -d gpu -m ode -n 128

# Run Cubie
bash ./run_benchmark.sh -l cubie -d gpu -m ode -n 128

# Compare
echo "Julia unadaptive:"
grep "128" data/Julia/Julia_times_unadaptive.txt
echo "Cubie unadaptive:"
grep "128" data/CUBIE/Cubie_times_unadaptive.txt
```

**Expected**: Cubie should be within 2-5x of Julia performance.

### Test 4: Full Benchmark Suite

```bash
# Clear previous results
rm -rf data/CUBIE/*

# Run full benchmark (default: up to 2^24 trajectories)
bash ./run_benchmark.sh -l cubie -d gpu -m ode

# This will take 20-30 minutes
# Monitor progress and check for errors
```

## Troubleshooting Guide

### Issue: Import errors for cubie

**Symptom**: `ModuleNotFoundError: No module named 'cubie'`

**Solution**:
```bash
conda activate venv_cubie
pip install cubie
# Or if you have a local development version:
pip install -e /path/to/cubie
```

### Issue: CUDA/GPU not found

**Symptom**: Cubie falls back to CPU or CUDA initialization fails

**Solution**:
```bash
# Verify GPU access
nvidia-smi

# Check CUDA environment
echo $CUDA_HOME
nvcc --version

# Reinstall numba-cuda
pip install --upgrade numba-cuda[cu12]
```

### Issue: Out of memory for large ensembles

**Symptom**: CUDA out of memory error at high trajectory counts

**Solution**: Implement batching in `bench_cubie.py`:

```python
def solve_with_batching(parameters, batch_size=2**18):
    """Solve in batches to manage memory."""
    num_params = len(parameterList)
    results = []
    
    for start_idx in range(0, num_params, batch_size):
        end_idx = min(start_idx + batch_size, num_params)
        
        batch_params = {
            'sigma': 10.0,
            'beta': 8.0/3.0,
            'rho': parameterList[start_idx:end_idx]
        }
        
        result = qb.solve_ivp(
            lorenz_system,
            initial_conditions,
            duration=1.0,
            parameters=batch_params,
            # ... rest of parameters
        )
        results.append(result)
    
    return results
```

### Issue: Performance significantly slower than expected

**Symptom**: Cubie times are >10x slower than Julia

**Diagnostic steps**:
1. Check GPU utilization: `nvidia-smi dmon` during execution
2. Verify JIT compilation is occurring (first run slow, subsequent fast)
3. Profile with smaller batches to isolate overhead
4. Check algorithm selection (RK4 for fixed, Tsit5 for adaptive)

**Potential solutions**:
- Increase batch size for better GPU utilization
- Ensure warm-up run completes before timing
- Check that adaptive tolerances aren't too strict
- Verify CUDA kernels are being used (not CPU fallback)

### Issue: Incorrect numerical results

**Symptom**: Solutions diverge or don't match expected behavior

**Solution**:
1. Verify Lorenz system definition matches reference:
   ```python
   # Test with single trajectory
   params = {'sigma': 10.0, 'beta': 8.0/3.0, 'rho': 28.0}
   ic = {'x': 1.0, 'y': 0.0, 'z': 0.0}
   
   sol = qb.solve_ivp(lorenz_system, ic, duration=1.0, 
                      parameters=params, save_at=np.linspace(0, 1, 101))
   
   # Plot or inspect trajectory
   import matplotlib.pyplot as plt
   plt.plot(sol['x'], sol['z'])
   plt.show()
   ```

2. Compare with SciPy reference implementation
3. Check parameter ordering and names
4. Verify initial conditions

## Performance Expectations

Based on benchmark characteristics and hardware:

### Expected Performance Targets

| Trajectory Count | Expected Time (ms) | Notes |
|-----------------|-------------------|-------|
| 8 | < 1 | Overhead dominates |
| 32 | < 2 | JIT compilation amortized |
| 128 | 2-5 | GPU utilization increasing |
| 512 | 5-15 | Good GPU utilization |
| 2048 | 15-50 | Near-optimal efficiency |
| 8192 | 50-200 | Peak performance range |
| 2^20 | 2000-10000 | Memory considerations |
| 2^24 | May need batching | >16M trajectories |

**Performance relative to other implementations**:
- **vs Julia**: 1-3x slower (Julia highly optimized)
- **vs JAX**: Competitive to 2x faster (depends on JIT)
- **vs PyTorch**: 2-5x faster (PyTorch vmap limitations)
- **vs MPGOS**: 1-2x slower (MPGOS hand-optimized CUDA)

### Hardware Dependencies

- **GPU Memory**: 8GB+ recommended for full benchmark
- **GPU Compute Capability**: 6.0+ required, 7.0+ optimal
- **CPU**: Minimal impact (GPU-bound workload)
- **RAM**: 16GB+ recommended

## Git Workflow

### Committing Changes

```bash
# Stage new files
git add GPU_ODE_Cubie/
git add runner_scripts/gpu/run_ode_cubie.sh
git add run_benchmark.sh

# Commit with descriptive message
git commit -m "feat: Add Cubie GPU ODE benchmark implementation

- Create GPU_ODE_Cubie directory with environment and benchmark script
- Implement bench_cubie.py with Lorenz system and timing harness
- Add runner script for trajectory scaling
- Modify run_benchmark.sh to support cubie language option
- Include both fixed and adaptive time-stepping benchmarks"

# Push to fork
git push origin add-cubie-support
```

### Creating Pull Request (Optional)

If contributing back to upstream GPUODEBenchmarks:

1. Push to your fork
2. Go to GitHub and create Pull Request
3. Title: "Add Cubie GPU ODE solver benchmark"
4. Description should include:
   - Overview of Cubie
   - Performance summary
   - Test results
   - Dependencies added
   - Any limitations or notes

## Verification Checklist

Before considering the implementation complete:

- [ ] Environment file created and tested (`conda env create -f ...`)
- [ ] Benchmark script runs without errors for small trajectory count
- [ ] Runner script executes and iterates through trajectory counts
- [ ] Main script recognizes `cubie` as valid language option
- [ ] Output files generated in correct format
- [ ] Results comparable to other implementations (within expected range)
- [ ] Full benchmark completes successfully (8 → 2^24 or max feasible)
- [ ] Code committed to feature branch
- [ ] Documentation updated (if applicable)
- [ ] Performance analysis completed

## Success Criteria

The implementation is successful when:

1. **Correctness**: Solutions match expected Lorenz system behavior
2. **Integration**: Seamlessly works with `run_benchmark.sh -l cubie -d gpu -m ode`
3. **Output Format**: Generates correctly formatted data files
4. **Scalability**: Successfully runs across trajectory range (8 to max feasible)
5. **Performance**: Completes within reasonable time, performance within expected range
6. **Reproducibility**: Can be set up from scratch using environment.yml
7. **Comparison**: Can be directly compared with Julia/JAX/PyTorch/MPGOS results

## Additional Resources

### Lorenz System Reference

Mathematical definition:
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz

Constants:
σ = 10.0
β = 8/3 ≈ 2.666...

Parameter:
ρ ∈ [0, 21]

Initial conditions:
(x₀, y₀, z₀) = (1.0, 0.0, 0.0)

Integration time:
t ∈ [0, 1]
```

### Cubie API Quick Reference

```python
# Create system
system = qb.create_ODE_system(
    "dx = ... \n dy = ... \n dz = ...",
    states={'x': 0.0, 'y': 0.0, 'z': 0.0},
    parameters={'param': value},
    name="SystemName"
)

# Solve (basic)
solution = qb.solve_ivp(
    system,
    initial_conditions,
    duration=1.0,
    parameters=parameters
)

# Solve (with options)
solution = qb.solve_ivp(
    system,
    initial_conditions,
    duration=1.0,
    parameters=parameters,
    algorithm='RK4',  # or 'Tsit5', 'DOPRI5', etc.
    dt=0.001,  # fixed time step
    adaptive=False,  # or True for adaptive
    rtol=1e-8,  # relative tolerance (adaptive)
    atol=1e-8,  # absolute tolerance (adaptive)
    save_at=[1.0]  # time points to save
)
```

### Benchmark Output Format

Each output file should have format:
```
<num_traj_1> <time_ms_1>
<num_traj_2> <time_ms_2>
...
```

Example `Cubie_times_unadaptive.txt`:
```
8 0.452
32 1.123
128 4.567
512 17.892
2048 68.234
8192 245.678
```

This format allows easy plotting and comparison with other implementations.
