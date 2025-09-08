"""
cubie: CUDA Batch Integration Engine
"""

from importlib.metadata import version

from cubie.batchsolving import *        # noqa
from cubie.integrators import *         # noqa
from cubie.outputhandling import *      # noqa
from cubie.memory import *              # noqa
import cubie.odesystems as systems    # noqa
from cubie.odesystems import *        # noqa
from cubie._utils import *              # noqa

__all__ = [
    "summary_metrics",
    "default_memmgr",
    "systems",
    "ArrayTypes",
    "Solver",
    "solve_ivp",
    "SymbolicODE",
    "create_ODE_system"
]

try:
    __version__ = version("cubie")
except ImportError:
    # Package is not installed
    __version__ = "unknown"
