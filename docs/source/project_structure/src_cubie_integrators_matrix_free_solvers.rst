src/cubie/integrators/matrix_free_solvers
=========================================

.. module:: cubie.integrators.matrix_free_solvers

The ``matrix_free_solvers`` package gathers factories that build CUDA device
functions for matrix-free linear and nonlinear solves. These factories are
used by the integrator loops to update implicit states without materialising
Jacobian matrices. The solvers rely on :mod:`numba.cuda` for device kernels
and perform warp-synchronisation via lightweight vote helpers.

Public API
----------

.. autosummary::
   :nosignatures:

   cubie.integrators.matrix_free_solvers.linear_solver_factory
   cubie.integrators.matrix_free_solvers.newton_krylov_solver_factory
   cubie.integrators.matrix_free_solvers.SolverRetCodes

Modules
-------

linear_solver
^^^^^^^^^^^^^

.. module:: cubie.integrators.matrix_free_solvers.linear_solver

Constructs preconditioned steepest-descent and minimal-residual solvers that
operate on matrix-free operators. The factory emits CUDA device functions that
maintain only vector workspaces, which keeps GPU memory usage predictable.

Dependencies
~~~~~~~~~~~~

- Warp synchronisation helpers for collective convergence tests.
- :mod:`numba.cuda` for compiling device functions.

Public factory
~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   cubie.integrators.matrix_free_solvers.linear_solver.linear_solver_factory

newton_krylov
^^^^^^^^^^^^^

.. module:: cubie.integrators.matrix_free_solvers.newton_krylov

Wraps the linear solver factory to assemble damped Newton--Krylov iterations
for implicit integration steps. The solver encodes completion status using
the :class:`cubie.integrators.matrix_free_solvers.SolverRetCodes` enumeration.

Dependencies
~~~~~~~~~~~~

- :mod:`cubie.integrators.matrix_free_solvers.linear_solver` for the inner
  linear solver.
- Warp-level vote operations supplied by the runtime.
- :mod:`numba.cuda` for device execution.

Public factory
~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   cubie.integrators.matrix_free_solvers.newton_krylov.newton_krylov_solver_factory

