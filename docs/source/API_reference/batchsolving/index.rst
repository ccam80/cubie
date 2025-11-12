Batch solving
=============

``cubie.batchsolving``
----------------------

.. currentmodule:: cubie.batchsolving

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   solver
   solve_ivp
   solve_result
   solve_spec
   batch_grid_builder
   batch_solver_config
   batch_solver_kernel
   system_interface
   cuda_array_validator
   cuda_array_validator_2d
   cuda_array_validator_3d
   optional_cuda_array_validator
   optional_cuda_array_validator_2d
   optional_cuda_array_validator_3d

The :class:`Solver` interface is the main entry point. It brings together batch
grids, array managers, GPU program builds, and system details into a coordinated
integration pipeline, while :func:`solve_ivp` offers a shortcut for common
workflows. Supporting modules prepare the GPU programs, describe solver
outputs, and provide data checks used by attrs-based containers.

.. toctree::
   :maxdepth: 2
   :caption: Subpackages
   :titlesonly:

   arrays

Core API
--------

* :doc:`Solver <solver>` – high-level manager that drives CUDA kernel launches.
* :doc:`solve_ivp <solve_ivp>` – convenience wrapper for single-run solver configuration.
* :doc:`SolveResult <solve_result>` – captures state, summaries, and diagnostic metadata.
* :doc:`SolveSpec <solve_spec>` – checked configuration describing a solver invocation.

Supporting infrastructure
-------------------------

* :doc:`BatchGridBuilder <batch_grid_builder>` – prepares chunked integration grids.
* :doc:`BatchSolverConfig <batch_solver_config>` – attrs container that tracks solver metadata.
* :doc:`BatchSolverKernel <batch_solver_kernel>` – compiles device kernels for batch integration.
* :doc:`SystemInterface <system_interface>` – adapts :class:`cubie.odesystems.baseODE.BaseODE`
  instances for CUDA execution.

Input checks
------------

* :doc:`cuda_array_validator <cuda_array_validator>` – checks required 1D CUDA arrays.
* :doc:`cuda_array_validator_2d <cuda_array_validator_2d>` – checks required 2D CUDA arrays.
* :doc:`cuda_array_validator_3d <cuda_array_validator_3d>` – checks required 3D CUDA arrays.
* :doc:`optional_cuda_array_validator <optional_cuda_array_validator>` – optional 1D array check.
* :doc:`optional_cuda_array_validator_2d <optional_cuda_array_validator_2d>` – optional 2D array check.
* :doc:`optional_cuda_array_validator_3d <optional_cuda_array_validator_3d>` – optional 3D array check.

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
  :mod:`cubie.outputhandling.summarymetrics` and adopt data checks from
  :mod:`cubie.batchsolving._utils`.
