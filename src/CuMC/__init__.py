"""
CuMC: CUDA-accelerated Monte Carlo simulations
"""

from importlib.metadata import version

from .ForwardSim import *
from .MonteCarlo import *
from .Sampling import *
from .SystemModels import *
from ._utils import *

try:
    __version__ = version("CuMC")
except ImportError:
    # Package is not installed
    __version__ = "unknown"
