"""
cubie: CUDA Batch Integration Engine
"""

from importlib.metadata import version

# Suppress Numba performance warnings for library users. The warnings are
# emitted from Numba internals when kernels are dispatched with an
# inefficient batch size.
# These are not actionable for CuBIE users,
# so they are filtered at import time.
import os
os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "0"

from cubie.batchsolving import *        # noqa
from cubie.integrators import *         # noqa
from cubie.outputhandling import *      # noqa
from cubie.memory import *              # noqa
import cubie.odesystems as systems    # noqa
from cubie.odesystems import *        # noqa
from cubie._utils import *              # noqa
from cubie.time_logger import TimeLogger  # noqa

__all__ = [
    "summary_metrics",
    "default_memmgr",
    "systems",
    "ArrayTypes",
    "Solver",
    "solve_ivp",
    "SymbolicODE",
    "create_ODE_system",
    "TimeLogger",
]

try:
    __version__ = version("cubie")
except ImportError:
    # Package is not installed
    __version__ = "unknown"
