GenericRosenbrockWStep
=====================

.. currentmodule:: cubie.integrators.algorithms

:class:`GenericRosenbrockWStep` provides a linearly implicit Rosenbrock-W
integrator that requires only one Jacobian factorisation per step. The factory
consumes
:class:`~cubie.integrators.algorithms.generic_rosenbrockw_tableaus.RosenbrockTableau`
instances and couples user-supplied device callbacks with matrix-free linear
solves from :mod:`cubie.integrators.matrix_free_solvers`.

Defaults
--------

* Default tableau: ``ros3p`` (three-stage ROS3P, order 3;
  ``DEFAULT_ROSENBROCK_TABLEAU``,
  ``generic_rosenbrockw_tableaus.py:87-132,426-428``). Rosenbrock-W is
  **linearly implicit**: each stage solves one linear system with a
  cached Jacobian factorisation, never a Newton iteration.
* Selection rule: when the resolved tableau carries an embedded error
  estimate the step controller defaults to
  ``ROSENBROCK_ADAPTIVE_DEFAULTS``; errorless tableaus default to
  ``ROSENBROCK_FIXED_DEFAULTS`` (``generic_rosenbrock_w.py:235-238``).
  ``check_compatibility`` (``SingleIntegratorRunCore.py:394-457``)
  additionally forces ``FixedStepController`` (with a
  ``UserWarning``) whenever an adaptive controller is paired with an
  errorless tableau.
* ``ROSENBROCK_ADAPTIVE_DEFAULTS`` (``generic_rosenbrock_w.py:67-77``):
  ``"pid"`` controller, ``kp=0.6``, ``ki=-0.4``, ``deadband_min=1.0``,
  ``deadband_max=1.1``, ``min_gain=0.5``, ``max_gain=2.0``.
* Because Rosenbrock-W is linearly implicit, only the **linear**
  solver defaults apply (no ``newton_*`` parameters):
  ``krylov_atol``/``krylov_rtol=1e-6``, ``krylov_max_iters=100``,
  ``linear_correction_type="minimal_residual"``,
  ``preconditioner_type="neumann"``, ``preconditioner_order=2``
  (``ode_implicitstep.py:52-124``).
* Tableau aliases resolving to :class:`GenericRosenbrockWStep`: every
  entry in ``ROSENBROCK_TABLEAUS``
  (``algorithms/__init__.py:79-83``), documented on the
  :doc:`Rosenbrock tableau registry <generic_rosenbrock_tableaus>`
  page.

.. autoclass:: GenericRosenbrockWStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_rosenbrock_w.RosenbrockWStepConfig
    :members:
