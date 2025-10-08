src/cubie/batchsolving
======================

.. currentmodule:: cubie.batchsolving

The :class:`Solver` façade is the main entry point. It wires batch grids,
array managers, kernel compilation, and system metadata into a coordinated GPU
integration pipeline while :func:`solve_ivp` provides a convenience wrapper for
common workflows. Supporting modules configure compiled kernels, describe
solver outputs, and expose validators used by attrs-based containers.

Subpackages
-----------

.. autosummary::
   :toctree: generated/

   cubie.batchsolving.arrays

Core API
--------

.. autosummary::
   :toctree: generated/

   Solver
   solve_ivp
   SolveResult
   SolveSpec

Supporting infrastructure
-------------------------

.. autosummary::
   :toctree: generated/

   BatchGridBuilder
   BatchSolverConfig
   BatchSolverKernel
   SystemInterface
   ActiveOutputs
   InputArrays
   OutputArrays
   ArrayContainer
   BaseArrayManager
   ManagedArray
   InputArrayContainer
   OutputArrayContainer

Validation helpers
------------------

.. autosummary::
   :toctree: generated/

   cuda_array_validator
   cuda_array_validator_2d
   cuda_array_validator_3d
   optional_cuda_array_validator
   optional_cuda_array_validator_2d
   optional_cuda_array_validator_3d

Dependencies
------------

- Batch kernel compilation extends :class:`cubie.CUDAFactory` and relies on
  :mod:`cubie.integrators.algorithms`,
  :mod:`cubie.integrators.step_control`, and
  :mod:`cubie.integrators.array_interpolator` for loop callables and driver
  interpolation.
- Array managers size buffers with
  :mod:`cubie.outputhandling.output_sizes` and allocate memory through
  :mod:`cubie.memory` managers.
- Solver results surface summary metrics registered via
  :mod:`cubie.outputhandling.summary_metrics` and adopt validators from
  :mod:`cubie.batchsolving._utils`.

