"""
Numerical integration algorithms and settings for ODE solving.

This module provides a collection of numerical integration algorithms and
configuration classes for solving ordinary differential equations (ODEs) on
CUDA-enabled devices. It includes both the high-level configuration classes
and low-level algorithm implementations.

The module contains:
- Integration algorithm implementations (Euler method, etc.)
- Runtime and timing configuration classes
- Single integrator run coordination
- CUDA device function management

Examples
--------
The integrators are typically used through higher-level solver classes,
but can be accessed directly through the ImplementedAlgorithms registry.
"""

from cubie.integrators.algorithms import *

__all__ = ["ImplementedAlgorithms"]
