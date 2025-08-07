"""
cubie: CUDA-accelerated Monte Carlo simulations
"""

from importlib.metadata import version

from cubie.batchsolving import *
from cubie.integrators import  *
from cubie.outputhandling import *
from cubie.systemmodels import *
from cubie._utils import *

try:
    __version__ = version("cubie")
except ImportError:
    # Package is not installed
    __version__ = "unknown"
