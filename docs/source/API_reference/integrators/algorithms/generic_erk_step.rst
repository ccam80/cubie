ERKStep
=======

.. currentmodule:: cubie.integrators.algorithms

The :class:`ERKStep` factory wraps a configurable explicit Runge--Kutta
integrator. It accepts any :class:`~cubie.integrators.algorithms.generic_erk.ERKTableau`
and ships with PI step-control defaults tuned for the embedded Dormand--Prince
pair. The factory performs staged right-hand-side evaluations on the GPU and
supports optional driver and observable callbacks.

Defaults
--------

* Default tableau: ``dormand-prince-54`` (Dormand–Prince 5(4), 7-stage,
  order 5 with an order-4 embedded estimate;
  ``DEFAULT_ERK_TABLEAU``, ``generic_erk_tableaus.py:99-169,748``).
  Explicit, so it needs no nonlinear or linear solver.
* Selection rule: when the resolved tableau carries an embedded error
  estimate (``tableau.has_error_estimate``, true for
  Dormand–Prince 5(4)) the step controller defaults to
  ``ERK_ADAPTIVE_DEFAULTS``; tableaus without one default to
  ``ERK_FIXED_DEFAULTS`` (``generic_erk.py:240-243``).
  ``check_compatibility`` (``SingleIntegratorRunCore.py:394-457``)
  additionally forces ``FixedStepController`` (with a ``UserWarning``)
  whenever an adaptive controller is paired with an errorless tableau.
* ``ERK_ADAPTIVE_DEFAULTS`` (``generic_erk.py:77-88``): ``"pid"``
  controller, ``kp=0.7``, ``ki=-0.4``, ``deadband_min=1.0``,
  ``deadband_max=1.1``, ``min_gain=0.1``, ``max_gain=5.0``,
  ``safety=0.9``.
* ``ERK_FIXED_DEFAULTS`` (``generic_erk.py:96-100``): ``"fixed"``
  controller.
* Tableau aliases resolving to :class:`ERKStep`: every entry in
  ``ERK_TABLEAU_REGISTRY`` (``algorithms/__init__.py:70-71``) — e.g.
  ``"dopri54"``, ``"rk45"``, ``"ode45"`` are aliases for
  ``dormand-prince-54``.

.. autoclass:: ERKStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_erk.ERKStepConfig
    :members:
