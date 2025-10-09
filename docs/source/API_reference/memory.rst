Memory
======

``cubie.memory``
----------------

.. currentmodule:: cubie.memory

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   memory/default_memmgr
   memory/memory_manager
   memory/array_request
   memory/array_response
   memory/stream_groups
   memory/current_cupy_stream
   memory/cupy_async_numba_manager
   memory/cupy_sync_numba_manager

The memory package coordinates GPU allocations across cubie. It exposes the
package-level :class:`~cubie.memory.mem_manager.MemoryManager` through
``default_memmgr`` so integrators can request array buffers, register CUDA
streams, and plug in alternative allocators without reimplementing the control
logic. Supporting modules describe allocation requests, track chunked response
metadata, manage stream groups, and expose CuPy-backed External Memory Manager
plugins for Numba contexts.

Entry point
-----------

``default_memmgr`` instantiates :class:`cubie.memory.mem_manager.MemoryManager`
with stream grouping and CuPy integration ready to configure. Typical callers
obtain this singleton, register their instance identifier, and submit
:class:`cubie.memory.array_requests.ArrayRequest` objects that describe the
arrays they need.

Core manager
------------

* :doc:`default_memmgr <memory/default_memmgr>` – default :class:`MemoryManager` instance shared across
  the package.
* :doc:`MemoryManager <memory/memory_manager>` – orchestrates allocation requests, stream registration, and External Memory Managers.

Array specifications
--------------------

* :doc:`ArrayRequest <memory/array_request>` – describes requested buffers, precision factories, and chunking.
* :doc:`ArrayResponse <memory/array_response>` – returns allocated buffers and metadata for callers.

Stream coordination
-------------------

* :doc:`StreamGroups <memory/stream_groups>` – assigns host instances to CUDA streams and manages synchronisation policies.

CuPy External Memory Managers
-----------------------------

* :doc:`current_cupy_stream <memory/current_cupy_stream>` – context manager that binds a Numba stream to a CuPy stream.
* :doc:`CuPyAsyncNumbaManager <memory/cupy_async_numba_manager>` – asynchronous CuPy External Memory Manager integration.
* :doc:`CuPySyncNumbaManager <memory/cupy_sync_numba_manager>` – synchronous CuPy External Memory Manager integration.

Dependencies
------------

The package requires :mod:`numba.cuda` for stream management and context access.
CuPy is optional but enables the External Memory Manager plugins and stream
interoperability helpers. These plugins are registered through
:class:`MemoryManager` when available and fall back to Numba's native allocator
otherwise.
