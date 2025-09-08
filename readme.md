# CuBIE
## CUDA batch integration engine for python

[![docs](https://github.com/ccam80/smc/actions/workflows/documentation.yml/badge.svg)](https://github.com/ccam80/smc/actions/workflows/documentation.yml) [![CUDA tests](https://github.com/ccam80/cubie/actions/workflows/ci_cuda_tests.yml/badge.svg)](https://github.com/ccam80/cubie/actions/workflows/ci_cuda_tests.yml)    [![Python Tests](https://github.com/ccam80/cubie/actions/workflows/ci_nocuda_tests.yml/badge.svg)](https://github.com/ccam80/cubie/actions/workflows/ci_nocuda_tests.yml)    [![codecov](https://codecov.io/gh/ccam80/cubie/graph/badge.svg?token=VG6SFXJ3MW)](https://codecov.io/gh/ccam80/cubie)
![PyPI - Version](https://img.shields.io/pypi/v/cubie)]    [![test build](https://github.com/ccam80/cubie/actions/workflows/test_pypi.yml/badge.svg)](https://github.com/ccam80/cubie/actions/workflows/test_pypi.yml)

A batch integration system for systems of ODEs and SDEs, for when elegant solutions fail and you would like to simulate 
1,000,000 systems, fast. This package was designed to simulate a large electrophysiological model as part of a 
likelihood-free inference method (eventually, package [cubism]), but the machinery is domain-agnostic.

While in early development, using this library as a way to experiment with and learn about some better software practice than I have used in 
past, including testing, CI/CD, and other helpful tactics I stumble upon. As such, there will
be some clunky bits.

The interface is not yet stable. As of v0.0.3, the symbolic interface for creating problems is up and running, and batch 
solves can be performed using Euler's method only, with a slightly clumsy API and some disorganised documentation.

### Roadmap:
-v0.0.4: Implicit integration methods.
  - Currently in development: Matrix-free solvers
  - Next up: 
    - Adaptive time-stepping loops and abstraction of the integrator loop base class.
    - Backward Euler method
    - Rosenbrock methods
    - Radau methods
    - Runge-Kutta methods
- v0.0.5: API improvements. This version should be stable enough for use in research - I will be using it in mine.
- v0.1.0: Documentation to match the API, organised in the sane way that a robot does not.

I'm completing this project to use it to finish my PhD, so I've got a pretty solid driver to get to v0.0.5 as fast as my
little fingers can type. I am motivated to get v0.1.0 out soon after to see if there is interest in this tool from the 
wider community.

## Documentation:

https://ccam80.github.io/cubie/

## Installation:
```
pip install cubie
```

## System Requirements:
- Python 3.8 or later
- CUDA Toolkit 12.9 or later
- NVIDIA GPU with compute capability 6.0 or higher (i.e. GTX10-series or newer)

## Contributing:
Pull requests are very, very welcome! Please open an issue if you would like to discuss a feature or bug before doing a 
bunch of work on it.

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