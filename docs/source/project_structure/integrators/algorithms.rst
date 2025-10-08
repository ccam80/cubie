Algorithms
==========

``cubie.integrators.algorithms``
--------------------------------

.. currentmodule:: cubie.integrators.algorithms

Factories in :mod:`cubie.integrators.algorithms` assemble explicit and implicit
step implementations that plug into the CUDA-based integrator loop. Explicit
steps wrap direct right-hand-side evaluations, while implicit steps couple
user-supplied device callbacks with matrix-free Newton–Krylov helpers to
satisfy nonlinear solves. Precision handling is central: every factory
propagates the configured precision through compiled device helpers and the
shared linear and nonlinear solver stack.

Base infrastructure
-------------------

* :class:`BaseStepConfig <cubie.integrators.algorithms.base_algorithm_step.BaseStepConfig>` –
  attrs configuration shared by explicit and implicit steps.
* :class:`BaseAlgorithmStep <cubie.integrators.algorithms.base_algorithm_step.BaseAlgorithmStep>` –
  base class that wires precision and CUDA compilation helpers.
* :class:`StepCache <cubie.integrators.algorithms.base_algorithm_step.StepCache>` –
  container storing compiled kernels and loop scratch buffers.

Explicit steps
--------------

* :class:`ExplicitStepConfig` – configuration for explicit step factories.
* :class:`ExplicitEulerStep` – Euler step that invokes the RHS device function.

Implicit steps
--------------

* :class:`ImplicitStepConfig` – configuration for implicit step factories.
* :class:`BackwardsEulerStep` – backward Euler implicit algorithm.
* :class:`BackwardsEulerPCStep` – predictor-corrector backward Euler variant.
* :class:`CrankNicolsonStep` – Crank–Nicolson implicit algorithm.

Factory helpers
---------------

* :func:`get_algorithm_step` – resolves step factories by enum or name.

Dependencies
------------

Implicit steps depend on :mod:`cubie.integrators.matrix_free_solvers` for the
linear and Newton–Krylov factories and reuse :class:`cubie.CUDAFactory`
utilities for JIT compilation and caching. All algorithms expect caller
supplied device callbacks for time derivatives, operator assembly, and
observable evaluation, operating on preallocated device buffers whose precision
matches the configured integration precision.
