Systems
=======

``cubie.odesystems.systems``
----------------------------

.. currentmodule:: cubie.odesystems.systems

The ``systems`` module collects ready-to-use ODE system definitions that ship
with Cubie. Each system subclasses :class:`cubie.odesystems.BaseODE` and
therefore inherits CUDA compilation behaviour, cached helper management, and
precision handling utilities.

Available systems
-----------------

* ``lorenz63`` – canonical Lorenz system useful for testing chaotic dynamics.
* ``lotka_volterra`` – predator-prey model demonstrating coupled oscillators.

Each module provides factory helpers that build configured :class:`BaseODE`
subclasses ready to integrate through :mod:`cubie.batchsolving` and
:mod:`cubie.integrators` entry points.
