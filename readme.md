# As-yet-unnamed CuNODE derivative (Tentatively: MCHammer)
[![docs](https://github.com/ccam80/smc/actions/workflows/documentation.yml/badge.svg)](https://github.com/ccam80/smc/actions/workflows/documentation.yml) [![CUDA-enabled tests](https://github.com/ccam80/smc/actions/workflows/cuda_test_lightning.yml/badge.svg)](https://github.com/ccam80/smc/actions/workflows/cuda_test_lightning.yml) [![Python Tests](https://github.com/ccam80/smc/actions/workflows/python-package.yml/badge.svg)](https://github.com/ccam80/smc/actions/workflows/python-package.yml)

A GPU-shaped hammer with which to hit likelihood-free Monte Carlo methods. Most components are written for execution on
NVIDIA GPUs, using CUDA through Numba, so this library will not be very helpful on non-NVIDIA hardware.

This library comprises four main modules:
- ForwardSim: A batch-integrating system for simulating ODEs and SDEs, with live processing/summarising of results to 
    allow generation of only the information required for inference, permitting large batches by minimising memory overhead.
- Sampling: Routines for estimating posterior distributions and generating parameter sets from them, utilising CUDA for
    parallelisation.
- SystemModels: The format for defining ODE systems, and the biggest barrier to entry for using this library. Although 
the intention was to make it easy to port existing equations from MATLAB or SciPy, the Numba/CUDA system does not feature
full support for all Python features, so some work is required to "simplify" (by making more comlicated to read), existing
code. 
- MonteCarlo: Monte Carlo algorithms, using the other modules to perform inference.

I am using this library as a way to experiment with and learn about some better software practice than I have used in 
past, including testing, CI/CD, and other helpful tactics I stumble upon. As such, while it's in development, there will
be some clunky bits.

The interface is not yet stable, and the documentation is in its infancy, so it will be hard to use, however some of the 
CUDA components are functional if you're looking for a head start on a similar simulation project.

