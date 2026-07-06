GPU Memory Management
=====================

CuBIE manages GPU memory (VRAM) automatically, but understanding the
available options lets you run larger batches and avoid out-of-memory
errors.

Default Behaviour
-----------------

By default CuBIE uses the Numba CUDA allocator.  Each call to
:meth:`~cubie.batchsolving.solver.Solver.solve` allocates the required
device arrays, runs the kernel, copies results back, then frees the
memory.

CuPy Memory Pools
------------------

Repeatedly allocating and freeing GPU memory has overhead.  CuPy provides
memory pools that recycle allocations across calls:

.. code-block:: python

   solver = qb.Solver(system, algorithm="dormand_prince_54",
                       memory_settings={"allocator": "cupy"})

Available allocators:

``"default"``
   Numba's built-in allocator.

``"cupy"``
   CuPy synchronous memory pool.  Reduces allocation overhead between
   successive solves.

``"cupy_async"``
   CuPy asynchronous memory pool.  Can overlap allocation with
   computation on supported hardware.

CuPy must be installed separately (``pip install cupy-cuda12x``).

VRAM Limits
-----------

CuBIE estimates the available VRAM and sizes the batch accordingly.  You
can override the proportion of VRAM that CuBIE is allowed to use:

.. code-block:: python

   solver = qb.Solver(system, algorithm="dormand_prince_54",
                       memory_settings={"mem_proportion": 0.7})

Set a lower proportion if other processes share the GPU.

Automatic Chunking
------------------

When a batch is too large to fit in VRAM in one go, CuBIE automatically
splits it into *chunks* and processes them sequentially.  The results are
concatenated transparently---you always get a single
:class:`~cubie.batchsolving.solveresult.SolveResult`.

Chunking is triggered automatically when the estimated memory requirement
exceeds the available VRAM.

Stream Groups
-------------

For advanced use, CuBIE supports running multiple chunks concurrently on
different CUDA streams via *stream groups*.  This can hide data-transfer
latency behind compute.  Configure via
``memory_settings={"stream_group": ...}`` on the ``Solver``.
