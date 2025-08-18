# CuBIE
## CUDA batch integration engine for python

[![docs](https://github.com/ccam80/smc/actions/workflows/documentation.yml/badge.svg)](https://github.com/ccam80/smc/actions/workflows/documentation.yml) [![CUDA tests](https://github.com/ccam80/cubie/actions/workflows/ci_cuda_tests.yml/badge.svg)](https://github.com/ccam80/cubie/actions/workflows/ci_cuda_tests.yml)  [![Python Tests](https://github.com/ccam80/cubie/actions/workflows/ci_nocuda_tests.yml/badge.svg)](https://github.com/ccam80/cubie/actions/workflows/ci_nocuda_tests.yml)  [![test build](https://github.com/ccam80/cubie/actions/workflows/test_pypi.yml/badge.svg)](https://github.com/ccam80/cubie/actions/workflows/test_pypi.yml)

A batch integration system for systems of ODEs and SDEs, for when elegant solutions fail and you would like to simulate 
1,000,000 systems, fast. This package was designed to simulate a large electrophysiological model as part of a 
likelihood-free inference method (eventually, package [cubism]), but the machinery is domain-agnostic.

The most basic use case is to define a system of ODEs or SDEs, and then call cubie.solve(system, inits, params, duration) with a description of the "batch" in the form of initial conditions and system parameters. There are a few seconds of overhead in the first call to Solve - cubie really shines when dealing with large problems or repeated calls with a similarly sized batch.

Defining a system of ODEs is the most cumbersome part of using this library. Like in MATLAB or SciPy, we need to create a dxdt function that takes the current state and parameters, and returns the rate of change of the state. Unlike MATLAB and SciPy, this function needs to be CUDA-compatible, which means it cannot use some of the features of Python and numpy. Creating a system is done by subclassing cubie.SystemModel.GenericODE, and implementing the dxdt method. See ThreeCM.py for an example of a small system. Fabbri_linder.py for an example of a large system.

## Installation:
pip install cubie

## System Requirements:
- Python 3.8 or later
- CUDA Toolkit 12.9 or later
- NVIDIA GPU with compute capability 6.0 or higher (i.e. GTX10-series or newer)

I am using this library as a way to experiment with and learn about some better software practice than I have used in 
past, including testing, CI/CD, and other helpful tactics I stumble upon. As such, while it's in development, there will
be some clunky bits.

The interface is not yet stable, and the documentation is currently non-working AI-generated slop, but the library now works roughly as you might expect, and can get up and running quickly by reading docstrings. Documentation and SymPy integration (as a means to get Jacobians to use implicit algorithms) are on the hit list for v0.1.0.
## Project Goals:

- Make an engine and interface for batch integration that is close enough to MATLAB or SciPy that a Python beginner can
  get integrating with the documentation alone in an hour or two. This also means staying Windows-compatible.
- Perform integrations of 10 or more parallel systems faster than MATLAB or SciPy can
- Enable extraction of summary variables only (rather than saving time-domain outputs) to facilitate use in algorithms 
  like likelihood-free inference.
- Be extensible enough that users can add their own systems and algorithms without needing to go near the core machinery.
- Don't be greedy - allow the user to control VRAM usage so that cubie can run alongside other applications.

## Non-Goals:
- Have the full set of integration algorithms that SciPy and MATLAB have.
  The full set of known and trusted algorithms is long, and it includes many wrappers for old Fortran libraries that the Numba compiler can't touch. If a problem requires a specific algorithm, we can add it as a feature request, but we won't set out to implement them all.
- Have a GUI.
  MATLABs toolboxes are excellent, but from previous projects (specifically CuNODE, the precursor to cubie), GUI development becomes all-consuming and distracts from the purpose of the project.