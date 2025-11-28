ERKStep
=======

.. currentmodule:: cubie.integrators.algorithms

The :class:`ERKStep` factory wraps a configurable explicit Runge--Kutta
integrator. It accepts any :class:`~cubie.integrators.algorithms.generic_erk.ERKTableau`
and ships with PI step-control defaults tuned for the embedded Dormand--Prince
pair. The factory performs staged right-hand-side evaluations on the GPU and
supports optional driver and observable callbacks.

.. autoclass:: ERKStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_erk.ERKStepConfig
    :members:
