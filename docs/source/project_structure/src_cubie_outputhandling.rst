src/cubie/outputhandling
========================

.. currentmodule:: cubie.outputhandling

Overview
--------

The output handling package coordinates CUDA device callbacks that persist
state trajectories and summary reductions from integration loops. It translates
loop settings into validated configuration, compiles device functions through
the CUDA factory infrastructure, and exposes sizing helpers so callers can
allocate host and device buffers without duplicating dimension logic.

Entry point
-----------

:class:`OutputFunctions` is the main interface. Instantiate it with loop
dimensions and requested outputs to compile CUDA functions that save solver
state, refresh summary metrics, and write reductions back to host arrays. The
factory retains an :class:`OutputConfig` instance and regenerates compiled
callbacks when configuration changes.

Subpackages
-----------

``summarymetrics`` houses the metric registry and built-in reductions. Import
the subpackage to register CUDA metric implementations and access the shared
registry via :data:`summary_metrics`.

.. toctree::
   :maxdepth: 1

   src_cubie_outputhandling_summarymetrics

Public API
----------

Configuration
~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../../api
   :nosignatures:

   OutputConfig
   OutputCompileFlags

Sizing utilities
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../../api
   :nosignatures:

   SummariesBufferSizes
   LoopBufferSizes
   OutputArrayHeights
   SingleRunOutputSizes
   BatchInputSizes
   BatchOutputSizes

Runtime factories and registries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../../api
   :nosignatures:

   OutputFunctions
   OutputFunctionCache
   summary_metrics
   register_metric

Dependencies
------------

* Compiles CUDA callables through :class:`cubie.CUDAFactory.CUDAFactory` and
  :mod:`numba.cuda`.
* Loop buffers and output slices align with expectations from
  :mod:`cubie.integrators.loops` and related algorithm factories.
