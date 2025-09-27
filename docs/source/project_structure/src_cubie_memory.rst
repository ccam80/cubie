src/cubie/memory
================

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

``default_memmgr`` instantiates :class:`~cubie.memory.mem_manager.MemoryManager`
with stream grouping and CuPy integration ready to configure. Typical callers
obtain this singleton, register their instance identifier, and submit
:class:`~cubie.memory.array_requests.ArrayRequest` objects that describe the
arrays they need.

Public API
----------

Core manager
~~~~~~~~~~~~

.. autosummary::
   :toctree: ../../api

   cubie.memory.default_memmgr
   cubie.memory.mem_manager.MemoryManager

Array specifications
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../../api

   cubie.memory.array_requests.ArrayRequest
   cubie.memory.array_requests.ArrayResponse

Stream coordination
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../../api

   cubie.memory.stream_groups.StreamGroups

CuPy External Memory Managers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../../api

   cubie.memory.cupy_emm.current_cupy_stream
   cubie.memory.cupy_emm.CuPyAsyncNumbaManager
   cubie.memory.cupy_emm.CuPySyncNumbaManager

Dependencies
------------

The package requires :mod:`numba.cuda` for stream management and context access.
CuPy is optional but enables the External Memory Manager plugins and stream
interoperability helpers. These plugins are registered through
``MemoryManager`` when available and fall back to Numba's native allocator
otherwise.

