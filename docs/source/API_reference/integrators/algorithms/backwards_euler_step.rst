BackwardsEulerStep
==================

.. currentmodule:: cubie.integrators

Defaults
--------

* No Butcher tableau — a single-stage, implicit, order-1 backward-Euler
  update (``beta=1.0``, ``gamma=1.0``, ``M=eye`` — ``ALGO_CONSTANTS``,
  ``backwards_euler.py:54-56``).
* Step controller: ``"fixed"`` (``BE_DEFAULTS``,
  ``backwards_euler.py:58-62``). Backward Euler has no embedded error
  estimate, so an adaptive controller can never be selected: an
  algorithm with ``is_adaptive == False`` paired with an adaptive
  controller is silently replaced with ``FixedStepController`` and a
  ``UserWarning`` is raised by ``check_compatibility``
  (``SingleIntegratorRunCore.py:394-457``).
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

.. autoclass:: BackwardsEulerStep
    :members:
    :show-inheritance:
