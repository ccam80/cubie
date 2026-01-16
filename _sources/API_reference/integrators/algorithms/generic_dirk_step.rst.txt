DIRKStep
========

.. currentmodule:: cubie.integrators.algorithms

:class:`DIRKStep` provides a diagonally implicit Runge--Kutta integrator that
solves stage systems with the cached Newton--Krylov helpers supplied by
:mod:`cubie.integrators.matrix_free_solvers`. The factory consumes
:class:`~cubie.integrators.algorithms.generic_dirk.DIRKTableau` instances,
exposing L-stable SDIRK and ESDIRK schemes for stiff problems while preserving
adaptive error control through embedded weights.

.. autoclass:: DIRKStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_dirk.DIRKStepConfig
    :members:
