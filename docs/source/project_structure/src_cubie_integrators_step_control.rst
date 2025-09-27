src/cubie/integrators/step_control
==================================

The step control package encapsulates the configuration and compilation
machinery used to build CUDA device functions that manage integrator time-step
selection. Fixed, integral, proportional–integral, proportional–integral–
derivative, and Gustafsson-style predictive controllers share a common
configuration interface that feeds the :class:`~cubie.CUDAFactory.CUDAFactory`
compilation pipeline.

.. currentmodule:: cubie.integrators.step_control

Controller interfaces
---------------------

.. autosummary::
   :toctree: generated/

   BaseStepController
   BaseAdaptiveStepController
   FixedStepController
   AdaptiveIController
   AdaptivePIController
   AdaptivePIDController
   GustafssonController
   get_controller

Configuration objects
---------------------

.. autosummary::
   :toctree: generated/

   BaseStepControllerConfig
   AdaptiveStepControlConfig
   FixedStepControlConfig
   PIStepControlConfig
   PIDStepControlConfig
   GustafssonStepControlConfig

Dependencies
------------

* Relies on :mod:`cubie.CUDAFactory` for compile-time caching of CUDA device
  functions.
* Requires Numba CUDA support during runtime to JIT the device controllers.
* Pulls validators and clamp helpers from :mod:`cubie._utils` when building
  device functions.

