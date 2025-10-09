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
   active_outputs
   input_arrays
   output_arrays
   array_container
   base_array_manager
   managed_array
   input_array_container
   output_array_container
   cuda_array_validator
   cuda_array_validator_2d
   cuda_array_validator_3d
   optional_cuda_array_validator
   optional_cuda_array_validator_2d
   optional_cuda_array_validator_3d

The :class:`Solver` façade is the main entry point. It wires batch grids,
array managers, kernel compilation, and system metadata into a coordinated GPU
integration pipeline while :func:`solve_ivp` provides a convenience wrapper for
common workflows. Supporting modules configure compiled kernels, describe
solver outputs, and expose validators used by attrs-based containers.

.. toctree::
   :maxdepth: 1
   :caption: Subpackages

   arrays

Core API
--------

* :doc:`Solver <solver>` – high-level manager that drives CUDA kernel launches.
* :doc:`solve_ivp <solve_ivp>` – convenience wrapper for single-run solver configuration.
* :doc:`SolveResult <solve_result>` – captures state, summaries, and diagnostic metadata.
* :doc:`SolveSpec <solve_spec>` – validated configuration describing a solver invocation.

Supporting infrastructure
-------------------------

* :doc:`BatchGridBuilder <batch_grid_builder>` – prepares chunked integration grids.
* :doc:`BatchSolverConfig <batch_solver_config>` – attrs container that tracks solver metadata.
* :doc:`BatchSolverKernel <batch_solver_kernel>` – compiles device kernels for batch integration.
* :doc:`SystemInterface <system_interface>` – adapts :class:`cubie.odesystems.baseODE.BaseODE`
  instances for CUDA execution.
* :doc:`ActiveOutputs <active_outputs>` – flags requested outputs and summaries.
* :doc:`InputArrays <input_arrays>` – hosts input buffers and arranges GPU copies.
* :doc:`OutputArrays <output_arrays>` – collects device trajectories and summaries.
* :doc:`ArrayContainer <array_container>` – base attrs container for array descriptors.
* :doc:`BaseArrayManager <base_array_manager>` – shared logic for host/device array management.
* :doc:`ManagedArray <managed_array>` – descriptor that pairs host buffers with CUDA handles.
* :doc:`InputArrayContainer <input_array_container>` – typed attrs container for solver inputs.
* :doc:`OutputArrayContainer <output_array_container>` – typed attrs container for solver outputs.

Validation helpers
------------------

* :doc:`cuda_array_validator <cuda_array_validator>` – validates required 1D CUDA arrays.
* :doc:`cuda_array_validator_2d <cuda_array_validator_2d>` – validates required 2D CUDA arrays.
* :doc:`cuda_array_validator_3d <cuda_array_validator_3d>` – validates required 3D CUDA arrays.
* :doc:`optional_cuda_array_validator <optional_cuda_array_validator>` – optional 1D array validator.
* :doc:`optional_cuda_array_validator_2d <optional_cuda_array_validator_2d>` – optional 2D array validator.
* :doc:`optional_cuda_array_validator_3d <optional_cuda_array_validator_3d>` – optional 3D array validator.

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
  :mod:`cubie.outputhandling.summarymetrics` and adopt validators from
  :mod:`cubie.batchsolving._utils`.
