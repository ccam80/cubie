Batch solving
=============

``cubie.batchsolving``
----------------------

.. currentmodule:: cubie.batchsolving

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

* :class:`Solver` – high-level manager that drives CUDA kernel launches.
* :func:`solve_ivp` – convenience wrapper for single-run solver configuration.
* :class:`SolveResult` – captures state, summaries, and diagnostic metadata.
* :class:`SolveSpec` – validated configuration describing a solver invocation.

Supporting infrastructure
-------------------------

* :class:`BatchGridBuilder` – prepares chunked integration grids.
* :class:`BatchSolverConfig` – attrs container that tracks solver metadata.
* :class:`BatchSolverKernel` – compiles device kernels for batch integration.
* :class:`SystemInterface` – adapts :class:`cubie.odesystems.baseODE.BaseODE`
  instances for CUDA execution.
* :class:`ActiveOutputs` – flags requested outputs and summaries.
* :class:`InputArrays` – hosts input buffers and arranges GPU copies.
* :class:`OutputArrays` – collects device trajectories and summaries.
* :class:`ArrayContainer` – base attrs container for array descriptors.
* :class:`BaseArrayManager` – shared logic for host/device array management.
* :class:`ManagedArray` – descriptor that pairs host buffers with CUDA handles.
* :class:`InputArrayContainer` – typed attrs container for solver inputs.
* :class:`OutputArrayContainer` – typed attrs container for solver outputs.

Validation helpers
------------------

* :func:`cuda_array_validator` – validates required 1D CUDA arrays.
* :func:`cuda_array_validator_2d` – validates required 2D CUDA arrays.
* :func:`cuda_array_validator_3d` – validates required 3D CUDA arrays.
* :func:`optional_cuda_array_validator` – optional 1D array validator.
* :func:`optional_cuda_array_validator_2d` – optional 2D array validator.
* :func:`optional_cuda_array_validator_3d` – optional 3D array validator.

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
