BackwardsEulerPCStep
====================

.. currentmodule:: cubie.integrators

Defaults
--------

:class:`BackwardsEulerPCStep` subclasses
:class:`~cubie.integrators.algorithms.backwards_euler.BackwardsEulerStep`
(``backwards_euler_predict_correct.py``) and inherits its configuration
unchanged, adding only an explicit forward-Euler predictor evaluated
before the implicit Newton–Krylov corrector runs.

* No Butcher tableau — a single-stage, implicit, order-1 update, same
  as backward Euler.
* Step controller: ``"fixed"`` (inherited ``BE_DEFAULTS``,
  ``backwards_euler.py:58-62``); the algorithm reports no embedded
  error estimate, so ``check_compatibility``
  (``SingleIntegratorRunCore.py:394-457``) silently replaces any
  adaptive controller with ``FixedStepController``.
* Nonlinear/linear solver defaults: identical to
  :doc:`BackwardsEulerStep <backwards_euler_step>` — ``newton_atol``/
  ``newton_rtol=1e-6``, ``newton_max_iters=100``,
  ``newton_damping=0.5``, ``newton_max_backtracks=8``,
  ``krylov_atol``/``krylov_rtol=1e-6``, ``krylov_max_iters=100``,
  ``linear_correction_type="minimal_residual"``,
  ``preconditioner_type="neumann"``, ``preconditioner_order=2``
  (``ode_implicitstep.py:52-124``).

.. autoclass:: BackwardsEulerPCStep
    :members:
    :show-inheritance:
