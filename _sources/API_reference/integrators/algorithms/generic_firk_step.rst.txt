FIRKStep
========

.. currentmodule:: cubie.integrators.algorithms

:class:`FIRKStep` provides a fully implicit Runge--Kutta integrator that solves
coupled stage systems with the cached Newton--Krylov helpers supplied by
:mod:`cubie.integrators.matrix_free_solvers`. The factory consumes
:class:`~cubie.integrators.algorithms.generic_firk_tableaus.FIRKTableau`
instances, exposing high-order Gauss--Legendre and Radau IIA schemes for stiff
problems while preserving adaptive error control through embedded weights.

.. autoclass:: FIRKStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_firk.FIRKStepConfig
    :members:
