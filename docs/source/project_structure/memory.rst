Memory
======

``cubie.memory``
----------------

.. currentmodule:: cubie.memory

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

* :data:`default_memmgr` – default :class:`MemoryManager` instance shared across
  the package.
* :class:`MemoryManager <cubie.memory.mem_manager.MemoryManager>` – orchestrates
  allocation requests, stream registration, and External Memory Managers.

Array specifications
--------------------

* :class:`ArrayRequest <cubie.memory.array_requests.ArrayRequest>` – describes
  requested buffers, precision factories, and chunking.
* :class:`ArrayResponse <cubie.memory.array_requests.ArrayResponse>` – returns
  allocated buffers and metadata for callers.

Stream coordination
-------------------

* :class:`StreamGroups <cubie.memory.stream_groups.StreamGroups>` – assigns host
  instances to CUDA streams and manages synchronisation policies.

CuPy External Memory Managers
-----------------------------

* :func:`current_cupy_stream <cubie.memory.cupy_emm.current_cupy_stream>` –
  context manager that binds a Numba stream to a CuPy stream.
* :class:`CuPyAsyncNumbaManager <cubie.memory.cupy_emm.CuPyAsyncNumbaManager>` –
  asynchronous CuPy External Memory Manager integration.
* :class:`CuPySyncNumbaManager <cubie.memory.cupy_emm.CuPySyncNumbaManager>` –
  synchronous CuPy External Memory Manager integration.

Dependencies
------------

The package requires :mod:`numba.cuda` for stream management and context access.
CuPy is optional but enables the External Memory Manager plugins and stream
interoperability helpers. These plugins are registered through
:class:`MemoryManager` when available and fall back to Numba's native allocator
otherwise.
