src/cubie/integrators/loops
===========================

Overview
--------

The loop package supplies the CUDA-oriented orchestration layer for Cubie's
integrators. The :class:`~cubie.integrators.loops.IVPLoop` factory compiles a
device function that drives per-step algorithms, manages shared-memory
buffers, and coordinates summary collection through the provided callbacks.
Supporting configuration classes in :mod:`cubie.integrators.loops.ode_loop_config`
describe the shared and persistent local memory layouts that the loop expects
when launching kernels.

API reference
-------------

Loop factory
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   cubie.integrators.loops.IVPLoop

Configuration helpers
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   cubie.integrators.loops.ode_loop_config.ODELoopConfig
   cubie.integrators.loops.ode_loop_config.LoopSharedIndices
   cubie.integrators.loops.ode_loop_config.LoopLocalIndices

Dependencies
------------

* Relies on :class:`cubie.CUDAFactory` for caching compiled device functions.
* Consumes CUDA device callbacks supplied by
  :mod:`cubie.integrators.algorithms` and
  :mod:`cubie.integrators.step_control`.
* Shares precision helpers with :mod:`cubie._utils` to standardise float
  handling across the integrator stack.

