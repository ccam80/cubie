# CuBIE
## CUDA batch integration engine for python

[![docs](https://github.com/ccam80/smc/actions/workflows/documentation.yml/badge.svg)](https://github.com/ccam80/smc/actions/workflows/documentation.yml) [![CUDA tests](https://github.com/ccam80/cubie/actions/workflows/ci_cuda_tests.yml/badge.svg)](https://github.com/ccam80/cubie/actions/workflows/ci_cuda_tests.yml)    [![Python Tests](https://github.com/ccam80/cubie/actions/workflows/ci_nocuda_tests.yml/badge.svg)](https://github.com/ccam80/cubie/actions/workflows/ci_nocuda_tests.yml)    [![codecov](https://codecov.io/gh/ccam80/cubie/graph/badge.svg?token=VG6SFXJ3MW)](https://codecov.io/gh/ccam80/cubie)
![PyPI - Version](https://img.shields.io/pypi/v/cubie)    [![test build](https://github.com/ccam80/cubie/actions/workflows/test_pypi.yml/badge.svg)](https://github.com/ccam80/cubie/actions/workflows/test_pypi.yml)

A batch integration system for numerically integarating many systems of ODEs in parallel, for when elegant solutions fail and you would like to simulate 
1,000,000 systems, fast. Cubie is a tool that performs the equivalent of MATLABs ODE functions (ode45 and the like), Scipy's solve_ivp function,
or some of the functions in Julia's SciML/OrdinaryDiffEq. This package was designed to simulate a large electrophysiological model as part of a 
likelihood-free inference method (eventually, package [cubism]), but the machinery is domain-agnostic.

This library uses Numba to JIT-compile CUDA kernels, allowing you the speed of compiled CUDA code without the headache
of writing CUDA code. It is designed to have a reasonably MATLAB- or SciPy-like interface, so that you can get up and 
running without having to figure out the intricacies of the internal mechanics.

The batch solving interface is not yet completely stable, and some parameters/arguments are likely to change further through to v1.0.
The core (per-parameter-set) machinery is reasonably stable. As of v0.0.6, you can:

- Set up and solve large parameter/initial condition sweeps of a system defined by a set of ODEs, entered either as:
  - A string or list of strings containing the equations of the system
  - A python function (not well tested yet)
  - A CellML model (tested on a subset of models in the CellML library so far)
- Use any of a large set of explicit or implicit runge-kutta or rosenbrock methods to integrate the problem.
- Extract the solution for any variable or ``observable`` at any time point, or extract summary statistics only to speed 
  things up.
- Provide ``forcing terms`` by including a function of _t_ in your equations, or by providing an array of values for the
  system to interpolate.
- Select from a handful of step-size control algorithms when using an adaptive-step algorithm like RK45 or RadauIIA5.

v0.0.6 also includes a few major milestones, the most significant of which is numerical validation of the methods against
results from the OrdinaryDiffEq suite of solvers. All algorithms are now validated and have been partially optimized to compete
with the really fast solvers in Julia DiffEqGPU.jl and some other GPU-accelerated packages that are either written in a 
tricky language or use some pretty rigid interfaces. Other changes are largely internal, but all contribute to a more cohesive
and abuse-tolerant package. I intend to publish some comparisons moving forward, watch this space.

### Roadmap:
- v0.0.7: API improvements. This version should be stable enough for use in research - I will be using it in mine.
  - Remove the distinction between 'observables' and 'state' from the user-facing arguments requesting variable saving.
  - Separate summary variable timing from save timing.
  - Add time-logging for user visibility of codegen-compile-run times.
- v0.1.0: Documentation to match the API, organised in the sane way that a robot does not.
  - User guide brought up-to-date with API, tracing an example through a few integration scenarios.


## Documentation:

https://ccam80.github.io/cubie/

## Installation:
We recommend that you use a python virtual environment to install Cubie - some dependencies are pinned to a specific version,
so installing it in it's own environment will avoid downgrading your system-wide packages and interfering with other projects.

```
python -m venv cubie_env
./cubie_env/Scripts/activate # Windows
# source cubie_env/bin/activate # Linux/Mac
pip install cubie
```

Then, when you fire up your Cubie project, run

```
source cubie_env/bin/activate
```

Or set up your IDE to use the `python.exe` in `cubie_env/Scripts/activate` (Windows) or `cubie_env/bin/activate` (Linux/Mac) 
as the project's interpreter so you don't have to worry about it.

## System Requirements:
- Python 3.10 or later
- CUDA Toolkit 12.9 or later
- NVIDIA GPU with compute capability 6.0 or higher (i.e. GTX10-series or newer)

## Python Requirements

* Python >= 3.10
* NumPy==1.26.4
* Numba
* Numba-CUDA
* attrs
* SymPy >= 1.13.0

## Optional Dependencies

Install these using `pip install cubie[optional]`

* Cupy-cu12x: For pool-based memory management (if you're doing a lot of
  consecutive batches of different sizes)
* Pandas: For DataFrame output support
* Matplotlib: For plotting support. Only used to plot an interpolated driver function for sanity-checks (see
  :doc:`Drivers <user_guide/drivers>`), but generally useful for visualizing results.


## Contributing:
Pull requests are very, very welcome! Please open an issue if you would like to discuss a feature or bug before doing a 
bunch of work on it, as I may have already partially implemented it or at least figured out where it might fit. For v0.1.0 I
will get some more descriptive documentation of the architecture in place.

## Project Goals:

- Make an engine and interface for batch integration that is close enough to MATLAB or SciPy that a Python beginner can
  get integrating with the documentation alone in an hour or two. This also means staying Windows-compatible.
- Perform integrations of 10 or more parallel systems faster than MATLAB or SciPy can
- Enable extraction of summary variables only (rather than saving time-domain outputs) to facilitate use in algorithms 
  like likelihood-free inference.
- Be extensible enough that users can add their own systems and algorithms without needing to go near the core machinery.