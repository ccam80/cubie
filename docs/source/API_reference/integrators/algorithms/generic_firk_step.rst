FIRKStep
========

.. currentmodule:: cubie.integrators.algorithms

:class:`FIRKStep` provides a fully implicit Runge--Kutta integrator that solves
coupled stage systems with the cached Newton--Krylov helpers supplied by
:mod:`cubie.integrators.matrix_free_solvers`. The factory consumes
:class:`~cubie.integrators.algorithms.generic_firk_tableaus.FIRKTableau`
instances, exposing high-order Gauss--Legendre and Radau IIA schemes for stiff
problems while preserving adaptive error control through embedded weights.

Defaults
--------

* Default tableau: ``firk_gauss_legendre_2`` (two-stage
  Gauss–Legendre, order 4; ``DEFAULT_FIRK_TABLEAU``,
  ``generic_firk_tableaus.py:64-72,149``). Implicit — all stages are
  solved together as one coupled ``n * stages`` Newton–Krylov system.
* Selection rule: when the resolved tableau carries an embedded error
  estimate the step controller defaults to ``FIRK_ADAPTIVE_DEFAULTS``;
  errorless tableaus default to ``FIRK_FIXED_DEFAULTS``
  (``generic_firk.py:220-222``). ``check_compatibility``
  (``SingleIntegratorRunCore.py:394-457``) additionally forces
  ``FixedStepController`` (with a ``UserWarning``) whenever an
  adaptive controller is paired with an errorless tableau.
* ``FIRK_ADAPTIVE_DEFAULTS`` (``generic_firk.py:62-72``): ``"pid"``
  controller, ``kp=0.6``, ``ki=-0.4``, ``deadband_min=1.0``,
  ``deadband_max=1.1``, ``min_gain=0.5``, ``max_gain=2.0``.
* ``FIRK_FIXED_DEFAULTS`` (``generic_firk.py:89-93``): ``"fixed"``
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

* Tableau aliases resolving to :class:`FIRKStep`: every entry in
  ``FIRK_TABLEAU_REGISTRY`` (``algorithms/__init__.py:76-77``),
  including the Radau IIA-5 scheme documented on the
  :doc:`FIRK tableau registry <generic_firk_tableaus>` page.

.. autoclass:: FIRKStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_firk.FIRKStepConfig
    :members:
