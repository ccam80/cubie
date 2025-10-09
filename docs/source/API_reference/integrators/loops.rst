Loops
=====

``cubie.integrators.loops``
---------------------------

The loop package supplies the CUDA-oriented orchestration layer for Cubie's
integrators. The :class:`~cubie.integrators.loops.IVPLoop` factory compiles a
device function that drives per-step algorithms, manages shared-memory buffers,
and coordinates summary collection through the provided callbacks. Supporting
configuration classes in :mod:`cubie.integrators.loops.ode_loop_config` describe
shared and persistent local memory layouts expected during kernel launches.

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ../ivp_loop
   ode_loop_config
   loop_shared_indices
   loop_local_indices

Loop factory
------------

* :doc:`IVPLoop <../ivp_loop>` – builds compiled CUDA loops that step through IVP integrations.

Configuration helpers
---------------------

* :doc:`ODELoopConfig <ode_loop_config>` – captures metadata describing loop memory layout and dimensions.
* :doc:`LoopSharedIndices <loop_shared_indices>` – describes offsets for shared-memory buffers.
* :doc:`LoopLocalIndices <loop_local_indices>` – indexes persistent local buffers and device scratch space.

Dependencies
------------

* Relies on :class:`cubie.CUDAFactory` for caching compiled device functions.
* Consumes CUDA device callbacks supplied by
  :mod:`cubie.integrators.algorithms` and
  :mod:`cubie.integrators.step_control`.
* Shares precision helpers with :mod:`cubie._utils` to standardise float
  handling across the integrator stack.
