Matrix-free solvers
===================

``cubie.integrators.matrix_free_solvers``
-----------------------------------------

.. module:: cubie.integrators.matrix_free_solvers

The ``matrix_free_solvers`` package gathers factories that build CUDA device
functions for matrix-free linear and nonlinear solves. These factories are used
by the integrator loops to update implicit states without materialising Jacobian
matrices. The solvers rely on :mod:`numba.cuda` for device kernels and perform
warp-synchronisation via lightweight vote helpers.

Public API
----------

* :func:`linear_solver_factory` – emits steepest-descent/minimal-residual CUDA
  solvers that operate on matrix-free operators.
* :func:`newton_krylov_solver_factory` – wraps the linear solver to construct
  damped Newton–Krylov iterations for implicit steps.
* :class:`SolverRetCodes` – enumerates solver completion status codes.

Modules
-------

``linear_solver``
^^^^^^^^^^^^^^^^^

Constructs preconditioned steepest-descent and minimal-residual solvers that
operate on matrix-free operators. The factory emits CUDA device functions that
maintain only vector workspaces, which keeps GPU memory usage predictable.

``newton_krylov``
^^^^^^^^^^^^^^^^^

Wraps the linear solver factory to assemble damped Newton–Krylov iterations for
implicit integration steps. The solver encodes completion status using the
:class:`SolverRetCodes` enumeration.

Dependencies
------------

* Warp synchronisation helpers implement collective convergence tests.
* :mod:`numba.cuda` compiles device functions and manages kernel launches.
* :mod:`cubie.integrators.matrix_free_solvers.linear_solver` provides the inner
  linear solver used by Newton–Krylov iterations.
