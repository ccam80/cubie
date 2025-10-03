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

Suggested controller parameters
-------------------------------

The default proportional, integral, and derivative gains mirror the
recommendations from Söderlind and Wang while matching the guidance in
`OrdinaryDiffEq.jl <https://github.com/SciML/OrdinaryDiffEq.jl>`_.
Common choices include

.. list-table::
   :header-rows: 1

   * - Controller
     - ``beta1``
     - ``beta2``
     - ``beta3``
   * - basic
     - 1.00
     - 0.00
     - 0
   * - PI42
     - 0.60
     - -0.20
     - 0
   * - PI33
     - 2/3
     - -1/3
     - 0
   * - PI34
     - 0.70
     - -0.40
     - 0
   * - H211PI
     - 1/6
     - 1/6
     - 0
   * - H312PID
     - 1/18
     - 1/9
     - 1/18

