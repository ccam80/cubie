src/cubie/odesystems
====================

.. currentmodule:: cubie.odesystems

The :func:`create_ODE_system` helper is the main entry point. It consumes
symbolic :mod:`sympy` equations and materialises :class:`SymbolicODE`
instances that inherit CUDA compilation behaviour from
:class:`cubie.CUDAFactory`. Base classes and data containers expose the
precision-aware metadata required by integrator factories.

Subpackages
-----------

.. autosummary::
   :toctree: generated/
   :recursive:

   cubie.odesystems.symbolic

Core API
--------

.. autosummary::
   :toctree: generated/

   create_ODE_system
   SymbolicODE
   BaseODE

Data containers and caches
--------------------------

.. autosummary::
   :toctree: generated/

   ODEData
   SystemValues
   SystemSizes
   ODECache

Dependencies
------------

- :class:`SymbolicODE` subclasses :class:`cubie.CUDAFactory` so integrator
  loops can request compiled CUDA device functions directly.
- Precision handling relies on :mod:`cubie._utils` helpers and
  :mod:`cubie.cuda_simsafe` to provide simulator-safe coercions.
- Generated kernels are consumed by :mod:`cubie.integrators` factories during
  loop construction.

