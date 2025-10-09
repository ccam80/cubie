Algorithms
==========

``cubie.integrators.algorithms``
--------------------------------

.. currentmodule:: cubie.integrators.algorithms

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   base_step_config
   base_algorithm_step
   step_cache
   ../explicit_step_config
   ../explicit_euler_step
   ../implicit_step_config
   ../backwards_euler_step
   ../backwards_euler_pc_step
   ../crank_nicolson_step
   ../get_algorithm_step

Factories in :mod:`cubie.integrators.algorithms` assemble explicit and implicit
step implementations that plug into the CUDA-based integrator loop. Explicit
steps wrap direct right-hand-side evaluations, while implicit steps couple
user-supplied device callbacks with matrix-free Newton–Krylov helpers to
satisfy nonlinear solves. Precision handling is central: every factory
propagates the configured precision through compiled device helpers and the
shared linear and nonlinear solver stack.

Base infrastructure
-------------------

* :doc:`BaseStepConfig <base_step_config>` – attrs configuration shared by explicit and implicit steps.
* :doc:`BaseAlgorithmStep <base_algorithm_step>` – base class that wires precision and CUDA compilation helpers.
* :doc:`StepCache <step_cache>` – container storing compiled kernels and loop scratch buffers.

Explicit steps
--------------

* :doc:`ExplicitStepConfig <../explicit_step_config>` – configuration for explicit step factories.
* :doc:`ExplicitEulerStep <../explicit_euler_step>` – Euler step that invokes the RHS device function.

Implicit steps
--------------

* :doc:`ImplicitStepConfig <../implicit_step_config>` – configuration for implicit step factories.
* :doc:`BackwardsEulerStep <../backwards_euler_step>` – backward Euler implicit algorithm.
* :doc:`BackwardsEulerPCStep <../backwards_euler_pc_step>` – predictor-corrector backward Euler variant.
* :doc:`CrankNicolsonStep <../crank_nicolson_step>` – Crank–Nicolson implicit algorithm.

Factory helpers
---------------

* :doc:`get_algorithm_step <../get_algorithm_step>` – resolves step factories by enum or name.

Dependencies
------------

Implicit steps depend on :mod:`cubie.integrators.matrix_free_solvers` for the
linear and Newton–Krylov factories and reuse :class:`cubie.CUDAFactory`
utilities for JIT compilation and caching. All algorithms expect caller
supplied device callbacks for time derivatives, operator assembly, and
observable evaluation, operating on preallocated device buffers whose precision
matches the configured integration precision.
