"""
CuMC: CUDA-accelerated Monte Carlo simulations
"""

from .ForwardSim import *
from .MonteCarlo import *
from .Sampling import *
from .SystemModels import *
from ._utils import *

from importlib.metadata import version

try:
    __version__ = version("CuMC")
except ImportError:
    # Package is not installed
    __version__ = "unknown"