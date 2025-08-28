"""System models package for ODE systems with CUDA support.

This package provides classes and utilities for defining and solving
ordinary differential equation systems on CUDA devices.
"""

from cubie.systemmodels.systems import Decays, ThreeChamberModel

__all__ = ["ThreeChamberModel", "Decays"]
