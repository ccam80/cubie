Step control
===========

``cubie.integrators.step_control``
---------------------------------

.. currentmodule:: cubie.integrators.step_control

The step control package encapsulates the configuration and compilation
machinery used to build CUDA device functions that manage integrator time-step
selection. Fixed, integral, proportional–integral, proportional–integral–
derivative, and Gustafsson-style predictive controllers share a common
configuration interface that feeds the :class:`cubie.CUDAFactory` compilation
pipeline.

Controller interfaces
---------------------

* :class:`BaseStepController` – abstract base for all controllers.
* :class:`BaseAdaptiveStepController` – adds adaptive gain handling.
* :class:`FixedStepController` – returns constant time steps.
* :class:`AdaptiveIController` – integral-only adaptive controller.
* :class:`AdaptivePIController` – proportional–integral controller.
* :class:`AdaptivePIDController` – proportional–integral–derivative controller.
* :class:`GustafssonController` – Gustafsson PI controller variant.
* :func:`get_controller` – resolves controller implementations from settings.

Configuration objects
---------------------

* :class:`BaseStepControllerConfig` – base attrs configuration shared by all
  controllers.
* :class:`AdaptiveStepControlConfig` – configuration used by adaptive
  controllers.
* :class:`FixedStepControlConfig` – fixed-step configuration container.
* :class:`PIStepControlConfig` – proportional–integral gain configuration.
* :class:`PIDStepControlConfig` – proportional–integral–derivative configuration.
* :class:`GustafssonStepControlConfig` – Gustafsson controller configuration.

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
`OrdinaryDiffEq.jl <https://github.com/SciML/OrdinaryDiffEq.jl>`_. Common
choices include:

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
