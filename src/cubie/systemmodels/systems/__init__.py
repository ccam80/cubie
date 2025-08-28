"""ODE system implementations with CUDA support.

This module contains specific implementations of ordinary differential equation
systems that can be compiled and executed on CUDA devices.
"""

from cubie.systemmodels.systems.decays import Decays
from cubie.systemmodels.systems.threeCM import ThreeChamberModel

__all__ = ["Decays", "ThreeChamberModel"]
