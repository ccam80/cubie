Output handling
===============

``cubie.outputhandling``
------------------------

.. currentmodule:: cubie.outputhandling

The output handling package coordinates CUDA device callbacks that persist state
trajectories and summary reductions from integration loops. It translates loop
settings into validated configuration, compiles device functions through the
CUDA factory infrastructure, and exposes sizing helpers so callers can allocate
host and device buffers without duplicating dimension logic.

.. toctree::
   :maxdepth: 1
   :caption: Subpackages

   summarymetrics

Entry point
-----------

:class:`OutputFunctions` is the main interface. Instantiate it with loop
dimensions and requested outputs to compile CUDA functions that save solver
state, refresh summary metrics, and write reductions back to host arrays. The
factory retains an :class:`OutputConfig` instance and regenerates compiled
callbacks when configuration changes.

Configuration
-------------

* :class:`OutputConfig` – attrs container describing requested outputs and
  summaries.
* :class:`OutputCompileFlags` – compile-time flags for CUDA output kernels.

Sizing utilities
----------------

* :class:`SummariesBufferSizes` – computes summary buffer dimensions.
* :class:`LoopBufferSizes` – describes per-loop buffer heights and widths.
* :class:`OutputArrayHeights` – calculates height metadata for output arrays.
* :class:`SingleRunOutputSizes` – shapes for single-run solver outputs.
* :class:`BatchInputSizes` – input buffer sizes for batch runs.
* :class:`BatchOutputSizes` – output buffer sizes for batch runs.

Runtime factories and registries
--------------------------------

* :class:`OutputFunctions` – compiles CUDA callbacks for saving state and
  summaries.
* :class:`OutputFunctionCache` – caches compiled device callables keyed by
  configuration.
* :data:`summary_metrics` – shared registry of registered summary metrics.
* :func:`register_metric` – decorator used by metric implementations to
  register with the shared registry.

Dependencies
------------

* Compiles CUDA callables through :class:`cubie.CUDAFactory` and
  :mod:`numba.cuda`.
* Loop buffers and output slices align with expectations from
  :mod:`cubie.integrators.loops` and related algorithm factories.
