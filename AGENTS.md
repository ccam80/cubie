## Style
Follow PEP8; max line length 79 characters, comment length 71 characters. Do not add commits that explain what you are doing 
to the user; write comments that explain unconventional or complex operations to future developers. Write numpydocs-style
docstrings for all functions and classes. Write type hints for all functions and methods.
Use descriptive variable names rather than minimal ones.
Use descriptive function names rather than minimal ones.
The repository is in development, do not enforce backwards compatibility; breaking changes are expected.
Type hints are compulsory, in PEP484 format in function definitions, rather than in docstrings.

## Tests
To run tests, use "pytest" from the command line. The dev environment is in Windows, so format terminal commands for powershell.
Create tests using pytests. Always use pytest fixtures, parameterised by settings dictionaries that can be indirectly overriden by "override" fixtures. Observe this pattern in tests/conftest.py.
Do not use mock or patch in tests.
A test which fails is a good test. Do not design tests to work around bugs or quirks in the code. Design tests to test 
that the code works as intended.
Never shortcut "is_device" or implement patches to get around other cuda-related checks that fail - this defeats the purpose.

## Environment
Install from workspace/cubie with pip install -e .[dev]
To run tests from an environment without CUDA drivers, set the environment variable NUMBA_ENABLE_CUDASIM="1".
If running tests without CUDA drivers, then omit pytests marked nocudasim and cupy.
- Never modify environment variables, as a monkeypatch or otherwise. Set CUDASIM in the environment external to the python source, never in python source.

## Attrs usage
For any floating-point attributes in an attrs class, save the attributes with a leading underscore, then add a property
which returns self.precision(self._attribute). Never add an alias to these underscored variables. Never include the underscore
in calls to __init__. Attrs handles both internally.

## Project structure

### src
#### src/cubie
##### src/cubie/batchsolving
###### src/cubie/batchsolving/arrays
##### src/cubie/integrators
###### src/cubie/integrators/algorithms
Labelled:
- Hosts explicit Euler and implicit Newton–Krylov step factories that
  share ``BaseStepConfig`` precision handling and ``StepCache`` outputs.
- Implicit algorithms depend on ``cubie.integrators.matrix_free_solvers``
  helpers surfaced through ``get_solver_helper_fn`` closures.
- Keep numpydoc docstrings aligned with precision casting rules and
  expose autosummary entries for configs and exported factory classes.
###### src/cubie/integrators/loops
###### src/cubie/integrators/matrix_free_solvers
Labelled:
- CUDA device solver factories that return matrix-free linear and
  Newton--Krylov iterations built with :mod:`numba.cuda`.
- Relies on warp-vote helpers for convergence checks and expects
  caller-supplied operator and residual callbacks plus preallocated
  device buffers.
- Public API exposes ``linear_solver_factory``,
  ``newton_krylov_solver_factory``, and ``SolverRetCodes`` status codes.

###### src/cubie/integrators/step_control
##### src/cubie/memory
##### src/cubie/odesystems
###### src/cubie/odesystems/symbolic
###### src/cubie/odesystems/systems
##### src/cubie/outputhandling
###### src/cubie/outputhandling/summarymetrics
#### src/cubie.egg-info

## Labelled

### src/cubie/integrators/step_control
Controllers share attrs-based configuration classes paired with CUDAFactory
builders that JIT Numba CUDA device functions. Adaptive controllers depend on
precision-controlled tolerance vectors rather than external error-norm helpers
and respect precision choices surfaced by the configuration. Keep docstrings
numpydoc-style with explicit parameter and return sections, and ensure module
documentation highlights the fixed and adaptive controller hierarchy.