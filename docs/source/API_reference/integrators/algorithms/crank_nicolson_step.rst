CrankNicolsonStep
=================

.. currentmodule:: cubie.integrators

Defaults
--------

* No Butcher tableau — a single-stage, implicit, order-2 update
  (``beta=1.0``, ``gamma=1.0``, ``M=eye`` — ``ALGO_CONSTANTS``,
  ``crank_nicolson.py:39-41``). Two implicit solves run per step (the
  Crank–Nicolson update and a backward-Euler update); their difference
  supplies the embedded error estimate, so the algorithm reports
  ``is_adaptive == True`` (``crank_nicolson.py:356-360``).
* Step controller: ``"pid"`` with ``kp=0.7``, ``ki=-0.4``,
  ``deadband_min=1.0``, ``deadband_max=1.1``, ``min_gain=0.5``,
  ``max_gain=2.0`` (``CN_DEFAULTS``, ``crank_nicolson.py:43-53``).
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

.. autoclass:: CrankNicolsonStep
    :members:
    :show-inheritance:
