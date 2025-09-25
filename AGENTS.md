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
- Package root highlights ``SingleIntegratorRun`` and
  ``IntegratorReturnCodes``. Support modules such as
  ``IntegratorRunSettings`` and ``SingleIntegratorRunCore`` live beside
  the subpackages and are imported directly when needed.
- Subdirectories provide algorithm factories, CUDA loop builders,
  matrix-free solver helpers, and adaptive/fixed controllers. Their public
  objects are surfaced through the package namespace for convenience, but
  solver status enums remain in ``matrix_free_solvers``.
- Implicit steps call into ``matrix_free_solvers`` for Newton--Krylov
  helpers, loops pull compile flags from ``cubie.outputhandling``, and
  controllers coordinate with algorithm instances via shared settings.
###### src/cubie/integrators/algorithms
- Hosts explicit Euler and implicit Newton–Krylov-based step factories that
  share ``BaseStepConfig`` precision handling and ``StepCache`` outputs.
- All step functions share a common signature and return a status code
  indicating success or failure.
- Implicit algorithms depend on ``cubie.integrators.matrix_free_solvers``
  helpers surfaced through ``get_solver_helper_fn`` closures.
- Public API exposes ``get_algorithm_step``, ``ExplicitStepConfig``,
  ``ImplicitStepConfig``, ``ExplicitEulerStep``, ``BackwardsEulerStep``,
  ``BackwardsEulerPCStep``, and ``CrankNicolsonStep``
###### src/cubie/integrators/loops
- Houses the :class:`IVPLoop` factory that compiles CUDA device loops via
  :class:`cubie.CUDAFactory`.
- Keeps ``LoopSharedIndices``, ``LoopLocalIndices``, and ``ODELoopConfig`` in
  ``ode_loop_config`` to describe shared and persistent buffer layouts plus
  loop metadata without re-exporting them from the package root.
- Depends on ``cubie.integrators.algorithms`` and
  ``cubie.integrators.step_control`` for device callbacks and on
  ``cubie.outputhandling`` for save and summary flag configuration.
###### src/cubie/integrators/matrix_free_solvers
- CUDA device solver factories that return matrix-free linear and
  Newton--Krylov iterations built with :mod:`numba.cuda`.
- Relies on warp-vote helpers for convergence checks and expects
  caller-supplied operator and residual callbacks plus preallocated
  device buffers.
- Public API exposes ``linear_solver_factory``,
  ``newton_krylov_solver_factory``, and ``SolverRetCodes`` status codes.

###### src/cubie/integrators/step_control
- CUDA device step control factories that return step control
  functions built with :mod:`numba.cuda`.
- fixed-step controller does nothing; adaptive step controllers return a 
  function that takes a step size and returns a new proposed step size.
- Public API exposes: ``AdaptiveIController``, ``AdaptivePIController``, ``AdaptivePIDController``,
  ``GustafssonController``, ``FixedStepController``, and ``get_controller``.
##### src/cubie/memory
GPU memory subsystem. ``mem_manager`` exposes the ``MemoryManager`` singleton
and default instance that orchestrate chunked allocations, instance registry
hooks, and stream grouping. ``array_requests`` defines request/response
containers that describe shapes, precision factories, and chunk metadata for
allocations. ``stream_groups`` groups host instances onto shared CUDA streams
using :mod:`numba.cuda`. ``cupy_emm`` integrates optional CuPy memory pools via
Numba's External Memory Manager interface and provides a context manager to
adopt Numba streams inside CuPy. CuPy is optional; without it the package falls
back to Numba's default allocator.
##### src/cubie/odesystems
###### src/cubie/odesystems/symbolic
###### src/cubie/odesystems/systems
##### src/cubie/outputhandling
###### src/cubie/outputhandling/summarymetrics
#### src/cubie.egg-info


### src/cubie/integrators/step_control
