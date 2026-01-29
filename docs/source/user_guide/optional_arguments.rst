Optional Arguments Reference
============================

CuBIE uses a cascading configuration system where optional parameters flow
through to underlying components. When you call ``solver.solve()`` or create
integration components directly, you can customize behaviour by passing
optional keyword arguments.

How Optional Arguments Work
---------------------------

When you pass an optional argument, it flows down through the configuration
hierarchy:

1. **Algorithm level**: Parameters like ``preconditioner_order`` control how
   implicit algorithms solve their internal equations.
2. **Controller level**: Parameters like ``atol`` and ``rtol`` control how
   the step size adapts to maintain accuracy.
3. **Loop level**: Parameters like ``dt_save`` control output timing.
4. **Output level**: Parameters like ``output_types`` control what data is
   saved.

Any parameter you don't specify uses a sensible default from the component's
configuration class. Passing ``None`` for any optional parameter means "use
the default" — it does not set the parameter to ``None``.

Algorithm Options
-----------------

Algorithm parameters control how integration steps are computed. Implicit
algorithms (Backwards Euler, DIRK, FIRK, Rosenbrock-W, Crank-Nicolson) use
internal solvers that have their own tuning parameters.

Implicit Solver Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

These parameters control the Newton-Krylov solver used by implicit methods.
The solver works in two layers: an outer Newton loop that handles
nonlinearity, and an inner Krylov loop that solves the linear system at each
Newton step.

**newton_atol**
    The absolute tolerance for the Newton solver convergence check. Together
    with ``newton_rtol``, this defines when the Newton loop exits. The scaled
    norm of the residual must fall below 1 (where the norm uses these
    tolerances for scaling) for convergence. Use tighter tolerances for more
    accurate implicit solves.

    - Default: ``1e-6``
    - Type: ``float`` or ``ndarray`` (must be positive)

**newton_rtol**
    The relative tolerance for the Newton solver convergence check. Works
    together with ``newton_atol`` to scale the residual norm. The relative
    tolerance scales based on the magnitude of the current solution.

    - Default: ``1e-6``
    - Type: ``float`` or ``ndarray`` (must be positive)

**max_newton_iters**
    Maximum number of Newton iterations before the solver gives up. If the
    Newton loop hasn't converged after this many iterations, the step is
    marked as failed. Increase this if you have a very stiff system that
    converges slowly but eventually succeeds.

    - Default: ``100``
    - Type: ``int`` (1 to 32767)

**newton_damping**
    The fraction by which the step is shrunk during backtracking. When a
    Newton step doesn't reduce the residual, the solver backtracks by
    multiplying the step by this factor and trying again. Values closer to
    1.0 try smaller corrections first, while values closer to 0 make more
    aggressive corrections.

    - Default: ``0.5``
    - Type: ``float`` (0 to 1)

**newton_max_backtracks**
    Maximum number of backtracking attempts per Newton step. If the solver
    cannot find a step that reduces the residual after this many attempts,
    it accepts the current step anyway and continues. Increase this for
    systems where finding good descent directions is difficult.

    - Default: ``8``
    - Type: ``int`` (1 to 32767)

**krylov_atol**
    The absolute tolerance for the linear solver convergence check. Together
    with ``krylov_rtol``, this defines when the Krylov loop exits. The inner
    Krylov loop solves a linear system at each Newton step and exits when
    the scaled residual norm falls below 1.

    - Default: ``1e-6``
    - Type: ``float`` or ``ndarray`` (must be positive)

**krylov_rtol**
    The relative tolerance for the linear solver convergence check. Works
    together with ``krylov_atol`` to scale the residual norm.

    - Default: ``1e-6``
    - Type: ``float`` or ``ndarray`` (must be positive)

**max_linear_iters**
    Maximum number of linear solver iterations per Newton step. If the
    Krylov loop hasn't converged after this many iterations, it returns
    with its current best estimate. Increase for ill-conditioned systems.

    - Default: ``100``
    - Type: ``int`` (1 to 32767)

**linear_correction_type**
    The line search strategy used within the linear solver. Choose between:

    - ``"steepest_descent"``: Moves in the direction of steepest decrease.
      Robust but can be slow for ill-conditioned problems.
    - ``"minimal_residual"``: Minimises the residual along the search
      direction. Often converges faster but may be less stable.

    - Default: ``"minimal_residual"``
    - Type: ``str``

**preconditioner_order**
    Order of the truncated Neumann series preconditioner. Higher orders give
    better preconditioning (faster convergence) but cost more per iteration.
    For most problems, order 1-3 works well.

    - Default: ``1``
    - Type: ``int`` (1 to 32)

Implicit Algorithm Applicability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 10 10 10 10 10

   * - Parameter
     - Backwards Euler
     - Crank-Nicolson
     - DIRK
     - FIRK
     - Rosenbrock-W
   * - newton_atol
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗
   * - newton_rtol
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗
   * - max_newton_iters
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗
   * - newton_damping
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗
   * - newton_max_backtracks
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗
   * - krylov_atol
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - krylov_rtol
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - max_linear_iters
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - linear_correction_type
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - preconditioner_order
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

Tableau Selection
~~~~~~~~~~~~~~~~~

Multi-stage algorithms (ERK, DIRK, FIRK, Rosenbrock-W) use Butcher tableaus
that define the coefficients for each stage. CuBIE provides several built-in
tableaus, and you can also define custom ones.

**tableau**
    The Butcher tableau defining the Runge-Kutta method. Different tableaus
    offer different trade-offs between accuracy, stability, and computational
    cost. Tableaus with embedded error estimates enable adaptive stepping;
    tableaus without them require fixed stepping.

    - ERK defaults: ``DOPRI54`` (Dormand-Prince 5(4), adaptive)
    - DIRK defaults: ``ESDIRK32`` (3rd order, adaptive)
    - FIRK defaults: ``RadauIIA`` (5th order, stiff problems)
    - Rosenbrock-W defaults: ``ROS34PW2`` (4th order, stiff problems)

Controller Options
------------------

Step controllers determine how the integration proceeds: whether to accept or
reject a step, and how to adjust the step size for the next attempt. CuBIE
provides several controllers for different use cases.

Common Adaptive Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

These parameters apply to all adaptive step controllers (I, PI, PID,
Gustafsson).

**atol**
    Absolute tolerance for error control. The error at each step is compared
    against ``atol + rtol * |state|``. Absolute tolerance dominates when state
    values are small. Can be a scalar (same for all variables) or an array
    (one per state variable).

    - Default: ``1e-6``
    - Type: ``float`` or array of ``float``

**rtol**
    Relative tolerance for error control. Scales with the magnitude of the
    state, so larger values have proportionally larger acceptable errors.
    Can be a scalar or array.

    - Default: ``1e-6``
    - Type: ``float`` or array of ``float``

**dt_min**
    Minimum allowable step size. The controller will not shrink the step
    below this limit. If the error is still too large at the minimum step,
    the integration will continue but flag this condition.

    - Default: ``1e-6``
    - Type: ``float`` (must be positive)

**dt_max**
    Maximum allowable step size. The controller will not grow the step beyond
    this limit, even if the error estimate suggests a larger step would be
    acceptable. Set this to prevent jumping over important dynamics.

    - Default: ``dt_min * 100`` (if not specified)
    - Type: ``float`` (must be greater than ``dt_min``)

**algorithm_order**
    Order of the integration algorithm, used to calculate optimal step size
    adjustment. Usually determined automatically from the algorithm, but can
    be overridden if needed.

    - Default: determined by algorithm
    - Type: ``int`` (must be at least 1)

**min_gain**
    Minimum factor by which the step size can shrink in one adjustment.
    Prevents overly aggressive step reduction. A value of 0.3 means the step
    can shrink to at most 30% of its previous size.

    - Default: ``0.3``
    - Type: ``float`` (0 to 1)

**max_gain**
    Maximum factor by which the step size can grow in one adjustment.
    Prevents overly aggressive step growth. A value of 2.0 means the step can
    at most double.

    - Default: ``2.0``
    - Type: ``float`` (must be at least 1)

**safety**
    Safety factor applied to step size predictions. A value less than 1.0
    makes the controller more conservative, preferring smaller steps than
    the error estimate suggests. Helps prevent rejected steps due to
    optimistic predictions.

    - Default: ``0.9``
    - Type: ``float`` (0 to 1)

**deadband_min** / **deadband_max**
    Range within which step size changes are suppressed. If the calculated
    gain falls between ``deadband_min`` and ``deadband_max``, the step size
    is left unchanged. This prevents small oscillations in step size when
    the error is near the tolerance. Set both to 1.0 to disable the deadband.

    - Default: ``deadband_min=1.0``, ``deadband_max=1.2``
    - Type: ``float``

PI Controller Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

The PI (proportional-integral) controller uses both current and previous
error estimates to calculate step size adjustments.

**kp**
    Proportional gain coefficient. Controls how strongly the current error
    estimate affects the step size. Higher values react more aggressively to
    the current error.

    - Default: ``1/18``
    - Type: ``float``

**ki**
    Integral gain coefficient. Controls how strongly the accumulated error
    history affects the step size. Higher values provide more smoothing but
    slower response.

    - Default: ``1/9``
    - Type: ``float``

PID Controller Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

The PID controller adds a derivative term to the PI controller for faster
response to rapidly changing errors.

**kp**
    Proportional gain coefficient (same as PI controller).

    - Default: ``1/18``
    - Type: ``float``

**ki**
    Integral gain coefficient (same as PI controller).

    - Default: ``1/9``
    - Type: ``float``

**kd**
    Derivative gain coefficient. Controls how the rate of change of error
    affects step size. Helps anticipate changes but can amplify noise.

    - Default: ``0.0`` (derivative term disabled)
    - Type: ``float``

Gustafsson Controller Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Gustafsson controller is designed for implicit methods, incorporating
information about Newton iteration convergence.

**gamma**
    Damping factor applied to the gain calculation. Lower values produce
    more conservative step size changes.

    - Default: ``0.9``
    - Type: ``float`` (0 to 1)

**max_newton_iters**
    Expected maximum Newton iterations, used to scale the step size
    prediction. Should match or slightly exceed your actual Newton iteration
    limit.

    - Default: ``20``
    - Type: ``int``

Fixed Step Controller
~~~~~~~~~~~~~~~~~~~~~

The fixed step controller maintains a constant step size throughout
integration.

**dt**
    The fixed step size to use. Required parameter for fixed stepping.

    - No default (must be specified)
    - Type: ``float`` (must be positive)

Controller Applicability
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 10 10 10

   * - Parameter
     - Fixed
     - I
     - PI
     - PID
     - Gustafsson
   * - dt
     - ✓
     - ✗
     - ✗
     - ✗
     - ✗
   * - atol
     - ✗
     - ✓
     - ✓
     - ✓
     - ✓
   * - rtol
     - ✗
     - ✓
     - ✓
     - ✓
     - ✓
   * - dt_min
     - ✗
     - ✓
     - ✓
     - ✓
     - ✓
   * - dt_max
     - ✗
     - ✓
     - ✓
     - ✓
     - ✓
   * - min_gain
     - ✗
     - ✓
     - ✓
     - ✓
     - ✓
   * - max_gain
     - ✗
     - ✓
     - ✓
     - ✓
     - ✓
   * - safety
     - ✗
     - ✓
     - ✓
     - ✓
     - ✓
   * - deadband_min/max
     - ✗
     - ✓
     - ✓
     - ✓
     - ✓
   * - kp
     - ✗
     - ✗
     - ✓
     - ✓
     - ✗
   * - ki
     - ✗
     - ✗
     - ✓
     - ✓
     - ✗
   * - kd
     - ✗
     - ✗
     - ✗
     - ✓
     - ✗
   * - gamma
     - ✗
     - ✗
     - ✗
     - ✗
     - ✓
   * - max_newton_iters (ctrl)
     - ✗
     - ✗
     - ✗
     - ✗
     - ✓

Loop Options
------------

Loop parameters control the overall integration process, including timing,
initial conditions, and output cadence.

Timing Parameters
~~~~~~~~~~~~~~~~~

**dt0**
    Initial step size at the start of integration. For adaptive controllers,
    this is used only for the first step; subsequent steps are determined
    by the controller. For fixed stepping, this is ignored (``dt`` is used).

    - Default: computed as ``sqrt(dt_min * dt_max)`` for adaptive
    - Type: ``float`` (must be positive)

**dt_save**
    Time interval between saved output samples. Determines how often state
    and observable values are recorded to the output arrays. Smaller values
    give higher resolution output but require more memory.

    - Default: ``0.1``
    - Type: ``float`` (must be positive)

**dt_summarise**
    Time interval between summary statistic calculations. Summary metrics
    (mean, max, RMS, etc.) are accumulated over this interval before being
    written to output. Should be at least as large as ``dt_save``.

    - Default: ``1.0``
    - Type: ``float`` (must be positive)

Output Options
--------------

Output parameters control what data is saved during integration.

Output Type Selection
~~~~~~~~~~~~~~~~~~~~~

**output_types**
    List of output types to save. Valid options include:

    - ``"state"``: Save state variable time series
    - ``"observables"``: Save observable time series
    - ``"time"``: Save time stamps for each output sample
    - ``"iteration_counters"``: Save Newton and Krylov iteration counts

    Summary metrics can also be specified:

    - ``"mean"``: Arithmetic mean over summary interval
    - ``"max"``: Maximum value over summary interval
    - ``"min"``: Minimum value over summary interval
    - ``"rms"``: Root-mean-square over summary interval
    - ``"std"``: Standard deviation over summary interval
    - ``"var"``: Variance over summary interval

    - Default: ``["state"]``
    - Type: list of ``str``

Index Selection
~~~~~~~~~~~~~~~

**saved_state_indices**
    Indices of state variables to include in time-domain output. Use this
    to save only the variables you need, reducing memory requirements.
    If not specified, all state variables are saved.

    - Default: all states
    - Type: list or array of ``int``

**saved_observable_indices**
    Indices of observable variables to include in time-domain output.

    - Default: all observables
    - Type: list or array of ``int``

**summarised_state_indices**
    Indices of state variables to include in summary calculations.
    Defaults to the same as ``saved_state_indices`` if not specified.

    - Default: same as saved indices
    - Type: list or array of ``int``

**summarised_observable_indices**
    Indices of observable variables to include in summary calculations.

    - Default: same as saved indices
    - Type: list or array of ``int``

Memory Location Options
-----------------------

Advanced users can control where intermediate buffers are allocated in GPU
memory. These options affect performance but not correctness.

**Buffer location parameters** (e.g., ``state_location``,
``preconditioned_vec_location``, etc.)

Control whether a buffer is allocated in local memory (``"local"``) or
shared memory (``"shared"``). Local memory is private to each thread and
larger; shared memory is faster but limited and shared across a thread
block. The defaults are tuned for typical use cases.

- Default: ``"local"`` for most buffers
- Type: ``str`` (``"local"`` or ``"shared"``)

These parameters are primarily useful for performance tuning on specific GPU
architectures and can generally be left at their defaults.
