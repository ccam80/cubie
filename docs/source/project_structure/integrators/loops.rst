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

Loop factory
------------

* :class:`IVPLoop <cubie.integrators.loops.IVPLoop>` – builds compiled CUDA
  loops that step through IVP integrations.

Configuration helpers
---------------------

* :class:`ODELoopConfig <cubie.integrators.loops.ode_loop_config.ODELoopConfig>` –
  captures metadata describing loop memory layout and dimensions.
* :class:`LoopSharedIndices <cubie.integrators.loops.ode_loop_config.LoopSharedIndices>` –
  describes offsets for shared-memory buffers.
* :class:`LoopLocalIndices <cubie.integrators.loops.ode_loop_config.LoopLocalIndices>` –
  indexes persistent local buffers and device scratch space.

Dependencies
------------

* Relies on :class:`cubie.CUDAFactory` for caching compiled device functions.
* Consumes CUDA device callbacks supplied by
  :mod:`cubie.integrators.algorithms` and
  :mod:`cubie.integrators.step_control`.
* Shares precision helpers with :mod:`cubie._utils` to standardise float
  handling across the integrator stack.
