ODE systems
==========

``cubie.odesystems``
--------------------

.. currentmodule:: cubie.odesystems

The :func:`create_ODE_system` helper is the main entry point. It consumes
symbolic :mod:`sympy` equations and materialises :class:`SymbolicODE` instances
that inherit CUDA compilation behaviour from :class:`cubie.CUDAFactory`.
:class:`BaseODE` is the abstract contract, and ``SymbolicODE`` is currently its
concrete implementation. Base classes and data containers expose the
precision-aware metadata required by integrator factories.

.. toctree::
   :maxdepth: 1
   :caption: Subpackages

   symbolic
   systems

Core API
--------

* :func:`create_ODE_system` – builds :class:`SymbolicODE` instances from SymPy
  expressions.
* :class:`SymbolicODE` – concrete :class:`BaseODE` subclass for symbolic systems.
* :class:`BaseODE` – abstract base exposing CUDA compilation helpers.

Data containers and caches
--------------------------

* :class:`ODEData` – captures compile-time metadata such as precision and state
  sizes.
* :class:`SystemValues` – runtime container for system-specific constants.
* :class:`SystemSizes` – records shape information for state and observable
  vectors.
* :class:`ODECache` – caches compiled device functions and solver helpers.

Dependencies
------------

- :class:`SymbolicODE` subclasses :class:`cubie.CUDAFactory` so integrator loops
  can request compiled CUDA device functions directly.
- Precision handling relies on :mod:`cubie._utils` helpers and
  :mod:`cubie.cuda_simsafe` to provide simulator-safe coercions.
- Generated kernels are consumed by :mod:`cubie.integrators` factories during
  loop construction.
