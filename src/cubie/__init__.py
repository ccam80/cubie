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

# Apply compile-time performance patches to stock numba-cuda before
# anything can compile a kernel. No-op on the cubie_patch fork, under
# CUDASIM, and for any patch already accepted upstream.
import cubie._numba_cuda_compat  # noqa: F401

from cubie.result_codes import CUBIE_RESULT_CODES
from cubie.batchsolving import *  # noqa
from cubie.integrators import *  # noqa
from cubie.outputhandling import *  # noqa
from cubie.memory import *  # noqa
from cubie.odesystems import *  # noqa
from cubie._utils import *  # noqa
from cubie.time_logger import TimeLogger, default_timelogger

__all__ = [
    "summary_metrics",
    "default_memmgr",
    "ArrayTypes",
    "Solver",
    "solve_ivp",
    "SymbolicODE",
    "create_ODE_system",
    "TimeLogger",
    "default_timelogger",
    "load_cellml_model",
    "CUBIE_RESULT_CODES",
]

try:
    __version__ = version("cubie")
except ImportError:
    # Package is not installed
    __version__ = "unknown"
