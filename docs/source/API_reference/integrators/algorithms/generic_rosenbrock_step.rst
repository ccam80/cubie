GenericRosenbrockWStep
=====================

.. currentmodule:: cubie.integrators.algorithms

:class:`GenericRosenbrockWStep` provides a linearly implicit Rosenbrock-W
integrator that requires only one Jacobian factorisation per step. The factory
consumes
:class:`~cubie.integrators.algorithms.generic_rosenbrockw_tableaus.RosenbrockTableau`
instances and couples user-supplied device callbacks with matrix-free linear
solves from :mod:`cubie.integrators.matrix_free_solvers`.

.. autoclass:: GenericRosenbrockWStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_rosenbrock_w.RosenbrockWStepConfig
    :members:
