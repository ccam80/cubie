Arrays
======

``cubie.batchsolving.arrays``
-----------------------------

.. currentmodule:: cubie.batchsolving.arrays

The ``batchsolving.arrays`` package coordinates host and device array
management for batch solver runs. ``InputArrays`` and ``OutputArrays`` are the
main entry points: they collect stride metadata, request GPU allocations via
:mod:`cubie.memory`, and expose helpers for copying data between the CPU and
CUDA kernels. ``OutputArrays`` mirrors requested state, observable, and summary
buffers on the host and GPU for every solver launch.

Base utilities
^^^^^^^^^^^^^^

* :class:`ArrayContainer <cubie.batchsolving.arrays.BaseArrayManager.ArrayContainer>` –
  attrs container shared by input and output managers.
* :class:`BaseArrayManager <cubie.batchsolving.arrays.BaseArrayManager.BaseArrayManager>` –
  helper that manages host/device array bindings and synchronisation.

Input arrays
^^^^^^^^^^^^

* :class:`InputArrayContainer <cubie.batchsolving.arrays.BatchInputArrays.InputArrayContainer>` –
  validated attrs container describing solver inputs.
* :class:`InputArrays <cubie.batchsolving.arrays.BatchInputArrays.InputArrays>` –
  orchestrates allocation, host buffers, and CUDA copies for input data.

Output arrays
^^^^^^^^^^^^^

* :class:`OutputArrayContainer <cubie.batchsolving.arrays.BatchOutputArrays.OutputArrayContainer>` –
  attrs container for state, observables, and summary outputs.
* :class:`ActiveOutputs <cubie.batchsolving.arrays.BatchOutputArrays.ActiveOutputs>` –
  boolean flags describing which outputs and summaries are requested.
* :class:`OutputArrays <cubie.batchsolving.arrays.BatchOutputArrays.OutputArrays>` –
  prepares host/device arrays, handles CUDA transfers, and exposes host views.

Dependencies
------------

* :mod:`cubie.outputhandling.output_sizes` supplies ``BatchOutputSizes``
  metadata for array shapes and active buffers.
* :mod:`cubie.batchsolving.ArrayTypes` defines the array typing shared with
  solver kernels.
* :mod:`cubie._utils` offers ``slice_variable_dimension`` to convert chunk
  indices into multidimensional slices.
