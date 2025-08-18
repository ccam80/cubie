"""
cubie: CUDA Batch Integration Engine
"""

from importlib.metadata import version

from cubie.batchsolving import *
from cubie.integrators import  *
from cubie.outputhandling import *
from cubie.memory import *
import cubie.systemmodels as systems
from cubie._utils import *

__all__ = ["summary_metrics", "default_memmgr", "systems", "ArrayTypes",
           "Solver", "solve_ivp"]

try:
    __version__ = version("cubie")
except ImportError:
    # Package is not installed
    __version__ = "unknown"
