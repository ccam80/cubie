## Style
Follow PEP8; max line length 79 characters, comment length 71 characters. Do not add commits that explain what you are doing 
to the user; write comments that explain unconventional or complex operations to future developers. Write numpydocs-style
docstrings for all functions and classes. Write type hints for all functions and methods.
Use descriptive variable names rather than minimal ones.
Use descriptive function names rather than minimal ones.
Don't type hint variables inside functions.
Never call build() directly on a CUDAFactory subclass. These objects automatially cache or build when you request the result through the object's property.
Don't import from __future__ import annotations, assume Python 3.7+.
The repository is in development, do not enforce backwards compatibility; breaking changes are expected.
Type hints are compulsory, in PEP484 format in function definitions, rather than in docstrings.
Never add comments or docstrings that form part of the conversation with the user, e.g. "this part now does this". Comments only serve
to explain what a complex piece of code is doing. They are always addressed to future developers.

## Tests
To run tests, use "pytest" from the command line. The dev environment is in Windows, so format terminal commands for powershell.
Create tests using pytests. Always use pytest fixtures, parameterised by settings dictionaries that can be indirectly overriden by "override" fixtures. Observe this pattern in tests/conftest.py.
Do not use mock or patch in tests.
A test which fails is a good test. Do not design tests to work around bugs or quirks in the code. Design tests to test 
that the code works as intended.
Never shortcut "is_device" or implement patches to get around other cuda-related checks that fail - this defeats the purpose.
Don't type hint tests.
## Environment
Install from workspace/cubie with pip install -e .[dev]
To run tests from an environment without CUDA drivers, set the environment variable NUMBA_ENABLE_CUDASIM="1".
If running tests without CUDA drivers, then omit pytests marked nocudasim and cupy.
- Never modify environment variables, as a monkeypatch or otherwise. Set CUDASIM in the environment external to the python source, never in python source.
- There are no lower level AGENTS.md files, this is the only one.
## Attrs usage
For any floating-point attributes in an attrs class, save the attributes with a leading underscore, then add a property
which returns self.precision(self._attribute). Never add an alias to these underscored variables. Never include the underscore
in calls to __init__. Attrs handles both internally.

## Project structure

### src
#### src/cubie
##### src/cubie/batchsolving
 - Package root exposes :class:`cubie.batchsolving.Solver` and
   :func:`cubie.batchsolving.solve_ivp` for GPU batch IVP runs, bundling grid
   construction, kernel compilation, and result aggregation helpers.
 - ``BatchSolverKernel``, ``BatchSolverConfig``, and ``BatchGridBuilder`` wire
   integrator loops, CUDA factories, and batch grids together using
   :mod:`cubie.integrators` components plus :mod:`cubie.memory` for allocations.
 - ``SystemInterface`` adapts :class:`cubie.odesystems.baseODE.BaseODE`
   instances for kernels, while ``solveresult`` collects output arrays and
   summary flags registered through :mod:`cubie.outputhandling.summary_metrics`.
###### src/cubie/batchsolving/arrays
- Base utilities in ``BaseArrayManager`` register array containers with
  :mod:`cubie.memory` and centralise chunk-aware transfer helpers that concrete
  input and output managers reuse for host/device copies.
- ``BatchInputArrays`` and ``BatchOutputArrays`` wrap attrs containers so
  solvers expose host views while device buffers stay synchronised via the
  memory manager and stream groups.
- ``BatchInputArrays`` defines input containers plus ``InputArrays`` managers
  that size initial values, parameters, and driver tables from solver metadata.
- ``BatchOutputArrays`` provides output containers for state, observables, state 
- summaries, and observable summaries, ``ActiveOutputs`` flags,
  and ``OutputArrays`` managers that collect device trajectories and summaries.
- Modules lean on :mod:`cubie.outputhandling.output_sizes` for stride metadata
  and :mod:`cubie._utils` helpers for slicing variable dimensions.

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
Base classes and data containers that describe CUDA-ready ODE systems.
``baseODE`` extends :class:`cubie.CUDAFactory` to manage compile settings and
provide solver helper caches. ``ODEData`` and ``SystemValues`` capture precision
aware metadata that integrator factories consume when wiring algorithms to
generated kernels.
###### src/cubie/odesystems/symbolic
SymPy-driven code generation pipeline. Parses symbolic system definitions,
emits CUDA ``dxdt`` kernels, and produces Newton--Krylov helpers consumed by
integrator loops. Depends on SymPy, :mod:`numba.cuda`, and the base classes in
the parent package.
##### src/cubie/outputhandling
CUDA output subsystem. ``OutputFunctions`` wraps :class:`cubie.CUDAFactory` to
compile state-saving and summary callbacks from an ``OutputConfig`` instance,
while ``output_sizes`` exposes sizing helpers for host/device buffer planning.
``summarymetrics`` instantiates the shared registry and re-exports
``register_metric`` so CUDA metric modules can self-register.
###### src/cubie/outputhandling/summarymetrics
Summary metric registry plus CUDA implementations. ``__init__`` instantiates
``SummaryMetrics`` and imports the built-in ``mean``, ``max``, ``rms``, and
``peaks`` modules so their classes register themselves. ``metrics`` defines the
registry, decorator, and cache helpers, relying on :class:`cubie.CUDAFactory`
and :mod:`numba.cuda` to compile device update/save callables. Each metric file
returns callable pairs that the registry hands to output-handling loops.
#### src/cubie.egg-info


### src/cubie/integrators/step_control
