DIRKStep
========

.. currentmodule:: cubie.integrators.algorithms

:class:`DIRKStep` provides a diagonally implicit Runge--Kutta integrator that
solves stage systems with the cached Newton--Krylov helpers supplied by
:mod:`cubie.integrators.matrix_free_solvers`. The factory consumes
:class:`~cubie.integrators.algorithms.generic_dirk.DIRKTableau` instances,
exposing L-stable SDIRK and ESDIRK schemes for stiff problems while preserving
adaptive error control through embedded weights.

Defaults
--------

* Default tableau: ``lobatto_iiic_3`` (three-stage Lobatto IIIC,
  order 4; ``DEFAULT_DIRK_TABLEAU``,
  ``generic_dirk_tableaus.py:127-136,290-291``). Implicit — each
  stage runs one Newton–Krylov solve.
* Selection rule: when the resolved tableau carries an embedded error
  estimate the step controller defaults to ``DIRK_ADAPTIVE_DEFAULTS``;
  errorless tableaus default to ``DIRK_FIXED_DEFAULTS``
  (``generic_dirk.py:210-212``). ``check_compatibility``
  (``SingleIntegratorRunCore.py:394-457``) additionally forces
  ``FixedStepController`` (with a ``UserWarning``) whenever an
  adaptive controller is paired with an errorless tableau.
* ``DIRK_ADAPTIVE_DEFAULTS`` (``generic_dirk.py:64-74``): ``"pid"``
  controller, ``kp=0.7``, ``ki=-0.4``, ``deadband_min=1.0``,
  ``deadband_max=1.1``, ``min_gain=0.1``, ``max_gain=5.0``.
* ``DIRK_FIXED_DEFAULTS`` (``generic_dirk.py:91-95``): ``"fixed"``
  controller.
* Nonlinear/linear solver defaults (Newton–Krylov,
  ``ode_implicitstep.py:52-124``), shared by every implicit algorithm
  except Rosenbrock-W:

  .. list-table::
     :header-rows: 1
     :widths: 45 55

     * - Parameter
       - Default
     * - ``newton_atol`` / ``newton_rtol``
       - ``1e-6``
     * - ``newton_max_iters``
       - ``100``
     * - ``newton_damping``
       - ``0.5``
     * - ``newton_max_backtracks``
       - ``8``
     * - ``krylov_atol`` / ``krylov_rtol``
       - ``1e-6``
     * - ``krylov_max_iters``
       - ``100``
     * - ``linear_correction_type``
       - ``"minimal_residual"``
     * - ``preconditioner_type``
       - ``"neumann"``
     * - ``preconditioner_order``
       - ``2``

* Tableau aliases resolving to :class:`DIRKStep`: every entry in
  ``DIRK_TABLEAU_REGISTRY`` (``algorithms/__init__.py:73-74``),
  including the implicit-midpoint, SDIRK, and Hairer–Wanner L-stable
  schemes documented on the
  :doc:`DIRK tableau registry <generic_dirk_tableaus>` page.

.. autoclass:: DIRKStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_dirk.DIRKStepConfig
    :members:
