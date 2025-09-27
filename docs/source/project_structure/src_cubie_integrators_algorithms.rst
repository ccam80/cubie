src/cubie/integrators/algorithms
================================

.. currentmodule:: cubie.integrators.algorithms

Overview
--------

Factories in :mod:`cubie.integrators.algorithms` assemble explicit and
implicit step implementations that plug into the CUDA-based integrator loop.
Explicit steps wrap direct right-hand-side evaluations, while implicit steps
couple user-supplied device callbacks with matrix-free Newton–Krylov helpers
to satisfy nonlinear solves. Precision handling is central: every factory
propagates the configured precision through compiled device helpers and the
shared linear and nonlinear solver stack.

Public API
----------

Base infrastructure
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   base_algorithm_step.BaseStepConfig
   base_algorithm_step.BaseAlgorithmStep
   base_algorithm_step.StepCache

Explicit steps
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ExplicitStepConfig
   ExplicitEulerStep

Implicit steps
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ImplicitStepConfig
   BackwardsEulerStep
   BackwardsEulerPCStep
   CrankNicolsonStep

Factory helpers
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   get_algorithm_step

Dependencies
------------

Implicit steps depend on :mod:`cubie.integrators.matrix_free_solvers` for the
linear and Newton–Krylov factories and reuse :class:`cubie.CUDAFactory`
utilities for JIT compilation and caching. All algorithms expect caller
supplied device callbacks for time derivatives, operator assembly, and
observable evaluation, and they operate on preallocated device buffers whose
precision matches the configured integration precision.
