ExplicitEulerStep
=================

.. currentmodule:: cubie.integrators

Defaults
--------

* No Butcher tableau — a single-stage, explicit, order-1 forward-Euler
  update (``explicit_euler.py``).
* Step controller: ``"fixed"`` (``EE_DEFAULTS``,
  ``explicit_euler.py:41-45``). Forward Euler has no embedded error
  estimate, so an adaptive controller can never be selected: an
  algorithm with ``is_adaptive == False`` paired with an adaptive
  controller is silently replaced with ``FixedStepController`` and a
  ``UserWarning`` is raised by ``check_compatibility``
  (``SingleIntegratorRunCore.py:394-457``).
* No nonlinear or linear solver — explicit methods evaluate the RHS
  directly.

.. autoclass:: ExplicitEulerStep
    :members:
    :show-inheritance:
