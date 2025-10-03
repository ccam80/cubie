src/cubie/batchsolving/arrays
=============================

.. currentmodule:: cubie.batchsolving.arrays



Public API
----------
The batch solver constructs :class:`~cubie.batchsolving.arrays.BatchInputArrays.InputArrays`
and its companion output manager during initialisation. These managers expose
host-side containers for user data and handle device bindings in step loops.

The ``batchsolving.arrays`` package coordinates host and device array
management for batch solver runs. ``InputArrays`` and ``OutputArrays`` are the
main entry points: they collect stride metadata, request GPU allocations via
:mod:`cubie.memory`, and expose helpers for copying data between the CPU and
CUDA kernels.

The batch array managers stage host and device buffers for solver runs.
``OutputArrays`` is the primary entry point: a solver kernel instantiates it
to mirror requested state, observable, and summary buffers on the host and the
CUDA device before each launch.

Base utilities
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   cubie.batchsolving.arrays.BaseArrayManager.ArrayContainer
   cubie.batchsolving.arrays.BaseArrayManager.BaseArrayManager

Input arrays
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   cubie.batchsolving.arrays.BatchInputArrays.InputArrayContainer
   cubie.batchsolving.arrays.BatchInputArrays.InputArrays

Output arrays
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   cubie.batchsolving.arrays.BatchOutputArrays.OutputArrayContainer
   cubie.batchsolving.arrays.BatchOutputArrays.ActiveOutputs
   cubie.batchsolving.arrays.BatchOutputArrays.OutputArrays


Dependencies
------------

* :mod:`cubie.outputhandling.output_sizes` supplies ``BatchOutputSizes``
  metadata for array shapes and active buffers.
* :mod:`cubie.batchsolving.ArrayTypes` defines the array typing shared with
  solver kernels.
* :mod:`cubie._utils` offers ``slice_variable_dimension`` to convert chunk
  indices into multidimensional slices.
