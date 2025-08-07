#CuBIE
## CUDA batch integration engine for python

[![docs](https://github.com/ccam80/smc/actions/workflows/documentation.yml/badge.svg)](https://github.com/ccam80/smc/actions/workflows/documentation.yml) [![CUDA-enabled tests](https://github.com/ccam80/smc/actions/workflows/cuda_test_lightning.yml/badge.svg)](https://github.com/ccam80/smc/actions/workflows/cuda_test_lightning.yml) [![Python Tests](https://github.com/ccam80/smc/actions/workflows/python-package.yml/badge.svg)](https://github.com/ccam80/smc/actions/workflows/python-package.yml)

A batch integration system for systems of ODEs and SDEs, for when elegant solutions fail and you would like to simulate 
1,000,000 systems, fast. This package was designed to simulate a large electrophysiological model as part of a 
likelihood-free inference method (see package [cubism]), but the machinery is domain-agnostic.

The most basic use case is to define a system of ODEs or SDEs, and then call cubie.solve(system, inits, params, duration)
with a description of the "batch" in the form of initial conditions and system parameters. There are a few seconds of  
overhead in the first call to Solve - cubie really shines when dealing with large problems or repeated calls with a similarly sized batch.

Defining a system of ODEs is the most cumbersome part of using this library. Like in MATLAB or SciPy, we need to create 
a dxdt function that takes the current state and parameters, and returns the rate of change of the state. Unlike MATLAB 
and SciPy, this function needs to be CUDA-compatible, which means it cannot use some of the features of Python and numpy.
Creating a system is done by subclassing cubie.SystemModel, and implementing the dxdt method. See ThreeCM.py for an 
example of a small system. Fabbri_linder.py for an example of a large system.

This library comprises four main modules:
- ForwardSim: A batch-integrating system for simulating ODEs and SDEs, with live processing/summarising of results to 
    allow generation of only the information required for inference, permitting large batches by minimising memory overhead.
- SystemModels: The format for defining ODE systems, and the biggest barrier to entry for using this library. Although 
the intention was to make it easy to port existing equations from MATLAB or SciPy, the Numba/CUDA system does not feature
full support for all Python features, so some work is required to "simplify" (by making more comlicated to read), existing
code.

## Installation:
pip install cubie

## System Requirements:
- Python 3.8 or later
- CUDA Toolkit 12.9 or later
- NVIDIA GPU with compute capability 6.0 or higher (i.e. GTX10-series or newer)


I am using this library as a way to experiment with and learn about some better software practice than I have used in 
past, including testing, CI/CD, and other helpful tactics I stumble upon. As such, while it's in development, there will
be some clunky bits.

The interface is not yet stable, and the documentation is in its infancy, so it will be hard to use, however some of the 
CUDA components are functional if you're looking for a head start on a similar simulation project.