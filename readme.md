# CuBIE
## CUDA batch integration engine for python

[![docs](https://github.com/ccam80/smc/actions/workflows/documentation.yml/badge.svg)](https://github.com/ccam80/smc/actions/workflows/documentation.yml) [![CUDA tests](https://github.com/ccam80/cubie/actions/workflows/ci_cuda_tests.yml/badge.svg)](https://github.com/ccam80/cubie/actions/workflows/ci_cuda_tests.yml)    [![Python Tests](https://github.com/ccam80/cubie/actions/workflows/ci_nocuda_tests.yml/badge.svg)](https://github.com/ccam80/cubie/actions/workflows/ci_nocuda_tests.yml)    [![codecov](https://codecov.io/gh/ccam80/cubie/graph/badge.svg?token=VG6SFXJ3MW)](https://codecov.io/gh/ccam80/cubie)
![PyPI - Version](https://img.shields.io/pypi/v/cubie)    [![test build](https://github.com/ccam80/cubie/actions/workflows/test_pypi.yml/badge.svg)](https://github.com/ccam80/cubie/actions/workflows/test_pypi.yml)

A batch integration system for systems of ODEs and SDEs, for when elegant solutions fail and you would like to simulate 
1,000,000 systems, fast. This package was designed to simulate a large electrophysiological model as part of a 
likelihood-free inference method (eventually, package [cubism]), but the machinery is domain-agnostic.

This library uses Numba to JIT-compile CUDA kernels, allowing you the speed of compiled CUDA code without the headache
of writing CUDA code. It is designed to have a reasonably MATLAB- or SciPy-like interface, so that you can get up and 
running without having to figure out the intricacies of the internal mechanics.

The batch solving interface is unstable, and is likely to change further through to v0.0.6 when I focus on how to 
actually use all of the componentry I've built. The core (per-parameter-set) machinery is reasonably stable. As of v0.0.5,
you can:

- Set up and solve large parameter/initial condition sweeps of a system defined by a set of ODEs
- Use any of a large set of explicit or implicit runge-kutta or rosenbrock methods to integrate the problem.
- Extract the solution for any variable or ``observable`` at any time point, or extract summary statistics only to speed 
  things up.
- Provide ``forcing terms`` by including a function of _t_ in your equations, or by providing an array of values for the
  system to interpolate.
- Select from a handful of step-size control algortihms when using an adaptive-step algorithm like Crank-Nicolson.
  Control over this is likely to lessen as I dial in per-algorithm defaults.


### Roadmap:
- v0.0.6: API improvements. This version should be stable enough for use in research - I will be using it in mine.
  - Batchsolving interface made more user-friendly
  - Batchsolving results delivered in a more sensible format
  - Per-algorithm defaults for step-size control and granular detail to remove the burden from you, friendly user.
- v0.1.0: Documentation to match the API, organised in the sane way that a robot does not.

I'm completing this project to use it to finish my PhD, so I've got a pretty solid driver to get to v0.0.6 as fast as my
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