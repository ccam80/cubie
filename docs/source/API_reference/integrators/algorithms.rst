Algorithms
==========

``cubie.integrators.algorithms``
--------------------------------

.. currentmodule:: cubie.integrators.algorithms

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   algorithms/base_step_config
   algorithms/base_algorithm_step
   algorithms/step_cache
   algorithms/explicit_step_config
   algorithms/explicit_euler_step
   algorithms/implicit_step_config
   algorithms/backwards_euler_step
   algorithms/backwards_euler_pc_step
   algorithms/crank_nicolson_step
   algorithms/generic_erk_step
   algorithms/generic_erk_tableaus
   algorithms/generic_dirk_step
   algorithms/generic_dirk_tableaus
   algorithms/get_algorithm_step

Factories in :mod:`cubie.integrators.algorithms` assemble explicit and implicit
step implementations that plug into the CUDA-based integrator loop. Explicit
steps wrap direct right-hand-side evaluations, while implicit steps couple
user-supplied device callbacks with matrix-free Newton–Krylov helpers to
satisfy nonlinear solves. Precision handling is central: every factory
propagates the configured precision through compiled device helpers and the
shared linear and nonlinear solver stack.

Base infrastructure
-------------------

* :doc:`BaseStepConfig <algorithms/base_step_config>` – attrs configuration shared by explicit and implicit steps.
* :doc:`BaseAlgorithmStep <algorithms/base_algorithm_step>` – base class that wires precision and CUDA compilation helpers.
* :doc:`StepCache <algorithms/step_cache>` – container storing compiled kernels and loop scratch buffers.

Explicit steps
--------------

* :doc:`ExplicitStepConfig <algorithms/explicit_step_config>` – configuration for explicit step factories.
* :doc:`ExplicitEulerStep <algorithms/explicit_euler_step>` – Euler step that invokes the RHS device function.
* :doc:`ERKStep <algorithms/generic_erk_step>` – multistage explicit Runge–Kutta step with adaptive control defaults.
* :doc:`ERK tableau registry <algorithms/generic_erk_tableaus>` – named explicit Runge–Kutta tableaus and references.

Implicit steps
--------------

* :doc:`ImplicitStepConfig <algorithms/implicit_step_config>` – configuration for implicit step factories.
* :doc:`BackwardsEulerStep <algorithms/backwards_euler_step>` – backward Euler implicit algorithm.
* :doc:`BackwardsEulerPCStep <algorithms/backwards_euler_pc_step>` – predictor-corrector backward Euler variant.
* :doc:`CrankNicolsonStep <algorithms/crank_nicolson_step>` – Crank–Nicolson implicit algorithm.
* :doc:`DIRKStep <algorithms/generic_dirk_step>` – diagonally implicit Runge–Kutta family with embedded error estimates.
* :doc:`DIRK tableau registry <algorithms/generic_dirk_tableaus>` – named diagonally implicit Runge–Kutta tableaus and references.

Factory helpers
---------------

* :doc:`get_algorithm_step <algorithms/get_algorithm_step>` – resolves step factories by enum or name.

Dependencies
------------

Implicit steps depend on :mod:`cubie.integrators.matrix_free_solvers` for the
linear and Newton–Krylov factories and reuse :class:`cubie.CUDAFactory`
utilities for JIT compilation and caching. All algorithms expect caller
supplied device callbacks for time derivatives, operator assembly, and
observable evaluation, operating on preallocated device buffers whose precision
matches the configured integration precision.
