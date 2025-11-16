# GPUODEBenchmarks Environment Setup Guide

This guide provides step-by-step instructions for setting up and running the GPUODEBenchmarks suite on Windows 11 with WSL2.

## Part 1: Setting Up GPUODEBenchmarks (Without Cubie)

This section walks through setting up the GPUODEBenchmarks repository to run the existing benchmark suite (Julia, C++, JAX, PyTorch) on a Windows 11 system using WSL2.

### Prerequisites

Before starting, ensure you have:
- Windows 11 with WSL2 installed
- NVIDIA GPU with compute capability 6.0+ (GTX 10-series or newer)
- NVIDIA drivers for Windows with WSL support (version 470.xx or newer)
- At least 16GB RAM recommended
- 20GB+ free disk space for environments and dependencies

### Step 1: Set Up WSL2 Ubuntu

1. **Install WSL2 with Ubuntu** (if not already installed):
   ```powershell
   # In PowerShell (as Administrator)
   wsl --install -d Ubuntu-22.04
   ```

2. **Restart your computer** if this is your first WSL2 installation.

3. **Launch Ubuntu** from the Start menu and complete the initial setup (create username/password).

4. **Update Ubuntu packages**:
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

### Step 2: Verify GPU Access in WSL2

1. **Check that your GPU is accessible**:
   ```bash
   nvidia-smi
   ```
   
   You should see output showing your NVIDIA GPU. If you get an error:
   - Ensure your Windows NVIDIA driver supports WSL2 (version 470.xx or newer)
   - Update your Windows NVIDIA driver from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx)

### Step 3: Install CUDA Toolkit in WSL2

1. **Install CUDA Toolkit 12.0** (or 11.8):
   ```bash
   # Download and install CUDA keyring
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   
   # Update package lists
   sudo apt update
   
   # Install CUDA toolkit
   sudo apt install cuda-toolkit-12-0 -y
   ```

2. **Add CUDA to your PATH**:
   ```bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Verify CUDA installation**:
   ```bash
   nvcc --version
   ```
   
   You should see CUDA compiler version information.

### Step 4: Install Julia

1. **Download and install Julia 1.8 or later**:
   ```bash
   # Download Julia 1.8.5 (or latest 1.8.x)
   wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz
   
   # Extract
   tar -xzf julia-1.8.5-linux-x86_64.tar.gz
   
   # Move to /opt
   sudo mv julia-1.8.5 /opt/julia
   
   # Create symlink
   sudo ln -s /opt/julia/bin/julia /usr/local/bin/julia
   ```

2. **Verify Julia installation**:
   ```bash
   julia --version
   ```

### Step 5: Install Conda (for Python Environments)

1. **Download and install Miniconda**:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. **Follow the installer prompts**:
   - Press Enter to review the license
   - Type "yes" to accept
   - Accept default installation location or specify custom path
   - Type "yes" when asked to initialize conda

3. **Close and reopen your terminal**, or run:
   ```bash
   source ~/.bashrc
   ```

4. **Verify conda installation**:
   ```bash
   conda --version
   ```

### Step 6: Clone GPUODEBenchmarks Repository

1. **Navigate to your home directory**:
   ```bash
   cd ~
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/utkarsh530/GPUODEBenchmarks.git
   cd GPUODEBenchmarks
   ```

### Step 7: Set Up Julia Environment

1. **Navigate to the repository**:
   ```bash
   cd ~/GPUODEBenchmarks
   ```

2. **Start Julia with the project**:
   ```bash
   julia --project=./GPU_ODE_Julia
   ```

3. **In the Julia REPL, install dependencies**:
   ```julia
   using Pkg
   Pkg.instantiate()
   Pkg.precompile()
   exit()
   ```
   
   This may take 5-10 minutes.

### Step 8: Set Up JAX Environment (Optional)

**Note**: The original conda environment files may fail due to outdated CUDA dependency management. We provide both the original conda method and an alternative venv method.

#### Option A: Using Conda (Original Method)

1. **Create the JAX conda environment**:
   ```bash
   cd ~/GPUODEBenchmarks
   conda env create -f GPU_ODE_JAX/environment.yml
   ```
   
   This may take 10-15 minutes.

2. **Verify the environment**:
   ```bash
   conda activate venv_jax
   python -c "import jax; print(jax.devices())"
   conda deactivate
   ```

#### Option B: Using Python venv (Alternative if Conda Fails)

1. **Install Python 3.10 and venv**:
   ```bash
   sudo apt install python3.10 python3.10-venv python3-pip -y
   ```

2. **Create and activate venv**:
   ```bash
   cd ~/GPUODEBenchmarks/GPU_ODE_JAX
   python3.10 -m venv venv_jax
   source venv_jax/bin/activate
   ```

3. **Install JAX with CUDA support** (using latest versions):
   ```bash
   pip install --upgrade pip
   pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   pip install diffrax equinox jaxtyping numpy scipy typeguard
   ```

4. **Verify the installation**:
   ```bash
   python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
   deactivate
   ```

5. **Update runner script** to use venv activation:
   ```bash
   # Edit runner_scripts/gpu/run_ode_jax.sh
   # Add after the shebang: source ./GPU_ODE_JAX/venv_jax/bin/activate
   # Add at the end: deactivate
   ```

### Step 9: Set Up PyTorch Environment (Optional)

**Note**: The original conda environment files may fail due to outdated CUDA dependency management. We provide both the original conda method and an alternative venv method.

#### Option A: Using Conda (Original Method)

1. **Create the PyTorch conda environment**:
   ```bash
   cd ~/GPUODEBenchmarks
   conda env create -f GPU_ODE_PyTorch/environment.yml
   ```

2. **Activate environment and install custom torchdiffeq**:
   ```bash
   conda activate venv_torch
   pip uninstall torchdiffeq -y
   pip install git+https://github.com/utkarsh530/torchdiffeq.git@u/vmap
   conda deactivate
   ```

#### Option B: Using Python venv (Alternative if Conda Fails)

1. **Install Python 3.10 and venv** (if not already done):
   ```bash
   sudo apt install python3.10 python3.10-venv python3-pip -y
   ```

2. **Create and activate venv**:
   ```bash
   cd ~/GPUODEBenchmarks/GPU_ODE_PyTorch
   python3.10 -m venv venv_torch
   source venv_torch/bin/activate
   ```

3. **Install PyTorch with CUDA support** (using latest versions):
   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install numpy scipy
   ```

4. **Install custom torchdiffeq**:
   ```bash
   pip install git+https://github.com/utkarsh530/torchdiffeq.git@u/vmap
   ```

5. **Verify the installation**:
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
   deactivate
   ```

6. **Update runner script** to use venv activation:
   ```bash
   # Edit runner_scripts/gpu/run_ode_pytorch.sh
   # Add after the shebang: source ./GPU_ODE_PyTorch/venv_torch/bin/activate
   # Add at the end: deactivate
   ```

### Modifying Runner Scripts for venv (If Using Option B)

If you chose to use Python venv instead of conda for JAX or PyTorch, you need to modify the corresponding runner scripts to activate the venv before running the benchmark script.

#### For JAX (if using venv):

Edit `runner_scripts/gpu/run_ode_jax.sh`:

```bash
#!/bin/bash
# Activate venv
source ./GPU_ODE_JAX/venv_jax/bin/activate

a=8
max_a=$1
XLA_PYTHON_CLIENT_PREALLOCATE=false
while [ $a -le $max_a ]
do
    # Print the values
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_JAX/bench_diffrax.py $a	
    # increment the value
    a=$((a*4))
done

# Deactivate venv
deactivate
```

#### For PyTorch (if using venv):

Edit `runner_scripts/gpu/run_ode_pytorch.sh`:

```bash
#!/bin/bash
# Activate venv
source ./GPU_ODE_PyTorch/venv_torch/bin/activate

a=8
max_a=$1
while [ $a -le $max_a ]
do
    # Print the values
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py $a	
    # increment the value
    a=$((a*4))
done

# Deactivate venv
deactivate
```

**Important**: These modifications are only needed if you used venv instead of conda. If you successfully used conda, no changes are needed.

### Step 10: Run Your First Benchmark

1. **Test Julia GPU ODE benchmark** (small scale):
   ```bash
   cd ~/GPUODEBenchmarks
   bash ./run_benchmark.sh -l julia -d gpu -m ode -n 128
   ```
   
   This runs the benchmark with up to 128 trajectories. You should see output showing trajectory counts and timing.

2. **Check the results**:
   ```bash
   cat data/Julia/Julia_times_unadaptive.txt
   cat data/Julia/Julia_times_adaptive.txt
   ```

3. **Optional: Test other implementations**:
   ```bash
   # JAX (requires JAX environment)
   bash ./run_benchmark.sh -l jax -d gpu -m ode -n 128
   
   # PyTorch (requires PyTorch environment)
   bash ./run_benchmark.sh -l pytorch -d gpu -m ode -n 128
   
   # C++ MPGOS
   bash ./run_benchmark.sh -l cpp -d gpu -m ode -n 128
   ```

### Troubleshooting Common Issues

**Issue: `nvidia-smi` not found**
- Update Windows NVIDIA drivers to latest version with WSL2 support
- Restart Windows after driver installation

**Issue: CUDA compilation errors**
- Ensure CUDA toolkit version matches what the benchmark expects (11.6+)
- Check that PATH and LD_LIBRARY_PATH are set correctly

**Issue: Julia package installation fails**
- Try: `julia --project=./GPU_ODE_Julia -e 'using Pkg; Pkg.update()'`
- Check internet connection and Julia package registry access

**Issue: Conda environment creation fails**
- Update conda: `conda update -n base -c defaults conda`
- Try creating environment with `--force` flag
- **If conda continues to fail**: Use the venv alternative (Option B) described in Steps 8 and 9

**Issue: JAX/PyTorch conda environments fail with CUDA dependency errors**
- This is a known issue with the original environment.yml files due to outdated CUDA dependency specifications
- **Solution**: Use the Python venv alternative (Option B) in Steps 8 and 9, which uses current package versions
- After setting up venv, modify the runner scripts as described in "Modifying Runner Scripts for venv"

**Issue: Python not finding JAX/PyTorch after venv setup**
- Ensure you activated the venv: `source ./GPU_ODE_JAX/venv_jax/bin/activate` (or venv_torch)
- Check installation: `pip list | grep jax` (or `grep torch`)
- Verify runner script has activation line at the beginning

**Issue: Out of memory errors**
- Start with smaller trajectory counts (-n 32 or -n 64)
- Monitor GPU memory with `nvidia-smi` in another terminal

---

## Part 2: Adding Cubie to the Benchmark Configuration

This section explains how to add Cubie to your local GPUODEBenchmarks setup after completing Part 1.

### Prerequisites

- Completed Part 1 (GPUODEBenchmarks installed and working)
- Cubie installed in a conda environment or system Python
- Familiarity with basic Python and Git

### Step 1: Fork and Clone Your Own Copy

Since you'll be modifying the benchmark suite, work with your own fork:

1. **Fork the repository on GitHub** (if you want to preserve changes):
   - Go to https://github.com/utkarsh530/GPUODEBenchmarks
   - Click "Fork" in the upper right

2. **Add your fork as a remote** (if working from existing clone):
   ```bash
   cd ~/GPUODEBenchmarks
   git remote add myfork https://github.com/YOUR_USERNAME/GPUODEBenchmarks.git
   ```
   
   Replace `YOUR_USERNAME` with your GitHub username.

3. **Create a feature branch**:
   ```bash
   git checkout -b add-cubie-support
   ```

### Step 2: Create Cubie Environment

**Choose either conda (Option A) or Python venv (Option B)**. If JAX/PyTorch conda failed, use venv for consistency.

#### Option A: Using Conda

1. **Create a conda environment for Cubie**:
   ```bash
   cd ~/GPUODEBenchmarks
   mkdir -p GPU_ODE_Cubie
   ```

2. **Create `GPU_ODE_Cubie/environment.yml`**:
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

3. **Create the environment**:
   ```bash
   conda env create -f GPU_ODE_Cubie/environment.yml
   ```

4. **Test Cubie installation**:
   ```bash
   conda activate venv_cubie
   python -c "import cubie; print('Cubie version:', cubie.__version__)"
   conda deactivate
   ```

#### Option B: Using Python venv (Recommended if conda failed)

1. **Create directory and venv**:
   ```bash
   cd ~/GPUODEBenchmarks
   mkdir -p GPU_ODE_Cubie
   cd GPU_ODE_Cubie
   python3.10 -m venv venv_cubie
   source venv_cubie/bin/activate
   ```

2. **Install Cubie and dependencies**:
   ```bash
   pip install --upgrade pip
   pip install cubie
   pip install numpy==1.26.4
   pip install numba numba-cuda[cu12]
   pip install attrs sympy
   ```

3. **Test Cubie installation**:
   ```bash
   python -c "import cubie; print('Cubie version:', cubie.__version__)"
   python -c "import numba.cuda; print('CUDA available:', numba.cuda.is_available())"
   deactivate
   ```

4. **Remember**: If using venv, you'll need to modify `runner_scripts/gpu/run_ode_cubie.sh` to include venv activation (see cubie_benchmark.md).

### Step 3: Install Benchmark Scripts

The benchmark scripts and runner should be added to your fork per the detailed instructions in `cubie_benchmark.md`.

Key files to add:
- `GPU_ODE_Cubie/bench_cubie.py` - Main benchmark script
- `runner_scripts/gpu/run_ode_cubie.sh` - Runner script
- Modifications to `run_benchmark.sh` - Add cubie option

See `cubie_benchmark.md` for complete implementation details.

### Step 4: Verify Cubie Benchmark

1. **Test with small trajectory count**:
   ```bash
   cd ~/GPUODEBenchmarks
   bash ./run_benchmark.sh -l cubie -d gpu -m ode -n 32
   ```

2. **Check output files**:
   ```bash
   cat data/CUBIE/Cubie_times_unadaptive.txt
   cat data/CUBIE/Cubie_times_adaptive.txt
   ```

3. **Compare with other implementations**:
   ```bash
   # Run Julia for comparison
   bash ./run_benchmark.sh -l julia -d gpu -m ode -n 32
   
   # Compare results
   echo "Julia:"
   tail -1 data/Julia/Julia_times_unadaptive.txt
   echo "Cubie:"
   tail -1 data/CUBIE/Cubie_times_unadaptive.txt
   ```

### Step 5: Run Full Benchmark Suite

1. **Run complete benchmark** (takes ~20-30 minutes):
   ```bash
   bash ./run_benchmark.sh -l cubie -d gpu -m ode
   ```

2. **Generate comparison plots** (if plotting is set up):
   ```bash
   # Requires modifications to plotting scripts
   julia --project=. ./runner_scripts/plot/plot_ode_comp.jl
   ```

### Summary: Key Changes When Using venv Instead of Conda

If you used Python venv (Option B) for JAX, PyTorch, or Cubie, the following changes are required to the benchmark system:

**1. Runner Scripts Must Activate/Deactivate venv**

Each affected runner script needs:
- Add `source ./GPU_ODE_<Language>/venv_<name>/bin/activate` at the start
- Add `deactivate` at the end

Example for JAX (`runner_scripts/gpu/run_ode_jax.sh`):
```bash
#!/bin/bash
source ./GPU_ODE_JAX/venv_jax/bin/activate
# ... existing benchmark loop ...
deactivate
```

**2. No Changes to Main run_benchmark.sh**

The main `run_benchmark.sh` script does NOT need modification. It calls the runner scripts, which handle environment activation.

**3. No Changes to Benchmark Python Scripts**

The actual benchmark scripts (`bench_*.py`) remain unchanged. They rely on the activated environment.

**4. Directory Structure Difference**

- Conda: Environment managed by conda (external to repo)
- venv: Each `GPU_ODE_<Language>/` contains its own `venv_<name>/` directory

**5. Environment Recreation**

- Conda: `conda env create -f environment.yml`
- venv: Run setup script or manually create/activate/pip install

**What Doesn't Change:**
- Output format and data directory structure
- Benchmark logic and timing methodology  
- Integration with plotting scripts
- Main orchestration via run_benchmark.sh

### Step 6: Commit and Push Changes

1. **Review your changes**:
   ```bash
   git status
   git diff
   ```

2. **Stage and commit**:
   ```bash
   git add GPU_ODE_Cubie/
   git add runner_scripts/gpu/run_ode_cubie.sh
   git add run_benchmark.sh
   git commit -m "feat: Add Cubie GPU ODE benchmark implementation"
   ```

3. **Push to your fork**:
   ```bash
   git push myfork add-cubie-support
   ```

### Troubleshooting Cubie Integration

**Issue: Cubie import fails**
- Verify conda environment: `conda activate venv_cubie`
- Reinstall cubie: `pip install --upgrade cubie`
- Check CUDA toolkit compatibility

**Issue: Benchmark script errors**
- Check that Lorenz system definition matches reference
- Verify parameter sweep range (0 to 21 for rho)
- Ensure output directory exists: `mkdir -p data/CUBIE`

**Issue: Performance significantly slower than expected**
- Profile with small batches first
- Check GPU utilization: `nvidia-smi dmon`
- Verify JIT compilation is occurring (first run may be slow)
- Consider batch size adjustments for memory efficiency

**Issue: Out of memory at high trajectory counts**
- Implement batching in benchmark script
- Reduce maximum trajectory count
- Monitor memory usage during execution

### Next Steps

- Run comprehensive benchmarks across all implementations
- Analyze performance characteristics
- Consider contributing back to GPUODEBenchmarks (optional)
- Generate comparison plots and performance analysis

For detailed implementation of the benchmark scripts themselves, see `cubie_benchmark.md`.
