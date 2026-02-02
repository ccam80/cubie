"""Tests for __init__.py re-export modules across cubie."""

from __future__ import annotations

import os

import numpy as np
import pytest

import cubie
import cubie.batchsolving
import cubie.batchsolving.arrays
import cubie.integrators.loops
import cubie.memory
import cubie.odesystems
import cubie.odesystems.symbolic
import cubie.odesystems.symbolic.codegen
import cubie.odesystems.symbolic.parsing
import cubie.outputhandling
import cubie.outputhandling.summarymetrics
import cubie.vendored
from cubie.batchsolving import ArrayTypes as _ArrayTypes
from cubie.batchsolving.solver import Solver as _Solver
from cubie.integrators.loops.ode_loop import IVPLoop as _IVPLoop
from cubie.memory import cupy_emm as _cupy_emm
from cubie.memory import default_memmgr as _default_memmgr
from cubie.memory.mem_manager import MemoryManager as _MemoryManager
from cubie.odesystems.symbolic.symbolicODE import SymbolicODE as _SymbolicODE
from cubie.outputhandling.output_functions import (
    OutputFunctions as _OutputFunctions,
)
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetrics,
    register_metric as _register_metric,
)
from cubie.time_logger import (
    TimeLogger as _TimeLogger,
    default_timelogger as _default_timelogger,
)


# ── cubie/__init__.py ───────────────────────────────────── #


def test_numba_low_occupancy_warning_suppressed():
    """Importing cubie sets NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS to '0'."""
    assert os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] == "0"


@pytest.mark.parametrize(
    "name",
    [
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
    ],
)
def test_cubie_all_contains(name):
    """cubie.__all__ lists expected public names."""
    assert name in cubie.__all__


def test_cubie_all_length():
    """cubie.__all__ has exactly 10 entries."""
    assert len(cubie.__all__) == 10


def test_cubie_star_reexports_accessible():
    """Star-imported names resolve to the canonical source objects."""
    assert cubie.Solver is _Solver
    assert cubie.SymbolicODE is _SymbolicODE
    assert cubie.OutputFunctions is _OutputFunctions
    assert cubie.default_memmgr is _default_memmgr
    assert cubie.TimeLogger is _TimeLogger
    assert cubie.default_timelogger is _default_timelogger


def test_cubie_version_is_string():
    """__version__ is a non-empty string (semver or 'unknown')."""
    # isinstance + value check: type IS the functionality (version metadata)
    assert isinstance(cubie.__version__, str)
    assert len(cubie.__version__) > 0


# ── cubie/batchsolving/__init__.py ──────────────────────── #

def test_batchsolving_arraytypes_is_type_alias():
    """ArrayTypes references the same alias defined in __init__."""
    assert cubie.batchsolving.ArrayTypes is _ArrayTypes


BATCHSOLVING_ALL_EXPECTED = [
    "ActiveOutputs",
    "ArrayContainer",
    "ArrayTypes",
    "BatchInputHandler",
    "BatchSolverConfig",
    "BatchSolverKernel",
    "BaseArrayManager",
    "InputArrayContainer",
    "InputArrays",
    "ManagedArray",
    "OutputArrayContainer",
    "OutputArrays",
    "Solver",
    "SolveResult",
    "SolveSpec",
    "SystemInterface",
    "solve_ivp",
    "summary_metrics",
]


@pytest.mark.parametrize("name", BATCHSOLVING_ALL_EXPECTED)
def test_batchsolving_all_contains(name):
    """batchsolving.__all__ lists expected public names."""
    assert name in cubie.batchsolving.__all__


def test_batchsolving_all_length():
    """batchsolving.__all__ has exactly 18 entries."""
    assert len(cubie.batchsolving.__all__) == 18


@pytest.mark.parametrize(
    "name, origin_module",
    [
        ("BatchInputHandler", "cubie.batchsolving.BatchInputHandler"),
        ("BatchSolverConfig", "cubie.batchsolving.BatchSolverConfig"),
        ("ActiveOutputs", "cubie.batchsolving.BatchSolverConfig"),
        ("BatchSolverKernel", "cubie.batchsolving.BatchSolverKernel"),
        ("SystemInterface", "cubie.batchsolving.SystemInterface"),
        ("ArrayContainer", "cubie.batchsolving.arrays.BaseArrayManager"),
        ("BaseArrayManager", "cubie.batchsolving.arrays.BaseArrayManager"),
        ("ManagedArray", "cubie.batchsolving.arrays.BaseArrayManager"),
        ("InputArrayContainer", "cubie.batchsolving.arrays.BatchInputArrays"),
        ("InputArrays", "cubie.batchsolving.arrays.BatchInputArrays"),
        ("OutputArrayContainer", "cubie.batchsolving.arrays.BatchOutputArrays"),
        ("OutputArrays", "cubie.batchsolving.arrays.BatchOutputArrays"),
        ("Solver", "cubie.batchsolving.solver"),
        ("solve_ivp", "cubie.batchsolving.solver"),
        ("SolveResult", "cubie.batchsolving.solveresult"),
        ("SolveSpec", "cubie.batchsolving.solveresult"),
    ],
)
def test_batchsolving_reexport_origin(name, origin_module):
    """Re-exported names come from their expected origin module."""
    obj = getattr(cubie.batchsolving, name)
    assert obj.__module__ == origin_module


def test_batchsolving_summary_metrics_is_singleton():
    """summary_metrics from batchsolving is the outputhandling singleton."""
    assert (
        cubie.batchsolving.summary_metrics
        is cubie.outputhandling.summary_metrics
    )


# ── cubie/batchsolving/arrays/__init__.py ───────────────── #

def test_batchsolving_arrays_is_package():
    """arrays subpackage is importable (empty __init__)."""
    assert cubie.batchsolving.arrays.__name__ == "cubie.batchsolving.arrays"


# ── cubie/integrators/loops/__init__.py ─────────────────── #

def test_loops_reexports_ivploop():
    """IVPLoop is accessible from integrators.loops."""
    assert cubie.integrators.loops.IVPLoop is _IVPLoop


def test_loops_all_contents():
    """__all__ contains exactly ['IVPLoop']."""
    assert cubie.integrators.loops.__all__ == ["IVPLoop"]


# ── cubie/odesystems/__init__.py ────────────────────────── #

@pytest.mark.parametrize(
    "name, origin_module",
    [
        ("ODEData", "cubie.odesystems.ODEData"),
        ("SystemSizes", "cubie.odesystems.ODEData"),
        ("SystemValues", "cubie.odesystems.SystemValues"),
        ("BaseODE", "cubie.odesystems.baseODE"),
        ("ODECache", "cubie.odesystems.baseODE"),
    ],
)
def test_odesystems_reexport_origin(name, origin_module):
    """Re-exported names come from their expected origin module."""
    obj = getattr(cubie.odesystems, name)
    assert obj.__module__ == origin_module


def test_odesystems_symbolic_reexports():
    """SymbolicODE, create_ODE_system, load_cellml_model accessible."""
    assert cubie.odesystems.SymbolicODE is _SymbolicODE
    assert cubie.odesystems.create_ODE_system is (
        cubie.odesystems.symbolic.create_ODE_system
    )
    assert cubie.odesystems.load_cellml_model is (
        cubie.odesystems.symbolic.load_cellml_model
    )


def test_odesystems_all_length():
    """odesystems.__all__ has 8 names."""
    assert len(cubie.odesystems.__all__) == 8


# ── cubie/odesystems/symbolic/__init__.py ───────────────── #

def test_symbolic_all_contents():
    """__all__ is exactly the 3 public names."""
    assert set(cubie.odesystems.symbolic.__all__) == {
        "SymbolicODE", "create_ODE_system", "load_cellml_model",
    }


def test_symbolic_star_imports_accessible():
    """Star-imported submodules make their names available."""
    assert cubie.odesystems.symbolic.SymbolicODE is _SymbolicODE


# ── cubie/odesystems/symbolic/codegen/__init__.py ───────── #

def test_codegen_all_is_empty():
    """codegen __all__ is an empty list."""
    assert cubie.odesystems.symbolic.codegen.__all__ == []


def test_codegen_star_imports_load():
    """Star imports from submodules execute without error."""
    assert cubie.odesystems.symbolic.codegen.__name__ == (
        "cubie.odesystems.symbolic.codegen"
    )


# ── cubie/odesystems/symbolic/parsing/__init__.py ───────── #

def test_parsing_all_contents():
    """parsing __all__ contains load_cellml_model."""
    assert cubie.odesystems.symbolic.parsing.__all__ == [
        "load_cellml_model",
    ]


def test_parsing_load_cellml_accessible():
    """load_cellml_model is accessible from parsing package."""
    assert cubie.odesystems.symbolic.parsing.load_cellml_model is (
        cubie.odesystems.load_cellml_model
    )


# ── cubie/outputhandling/__init__.py ────────────────────── #

@pytest.mark.parametrize(
    "name",
    [
        "OutputCompileFlags",
        "OutputConfig",
        "OutputFunctionCache",
        "OutputFunctions",
        "OutputArrayHeights",
        "SingleRunOutputSizes",
        "BatchInputSizes",
        "BatchOutputSizes",
        "summary_metrics",
        "register_metric",
    ],
)
def test_outputhandling_all_contains(name):
    """outputhandling.__all__ lists expected public names."""
    assert name in cubie.outputhandling.__all__


def test_outputhandling_all_length():
    """outputhandling.__all__ has 10 names."""
    assert len(cubie.outputhandling.__all__) == 10


# ── cubie/outputhandling/summarymetrics/__init__.py ─────── #

def test_summarymetrics_singleton_type():
    """summary_metrics is a SummaryMetrics instance with float32."""
    sm = cubie.outputhandling.summarymetrics.summary_metrics
    # isinstance + value: type IS the functionality (singleton creation)
    assert isinstance(sm, SummaryMetrics)
    assert sm.precision == np.float32


def test_summarymetrics_default_precision():
    """Default precision is float32."""
    sm = cubie.outputhandling.summarymetrics.summary_metrics
    assert sm.precision == np.float32


def test_summarymetrics_metrics_registered():
    """All 18 built-in metrics are registered after import."""
    sm = cubie.outputhandling.summarymetrics.summary_metrics
    expected_metrics = [
        "mean", "max", "rms", "peaks", "std", "min",
        "max_magnitude", "extrema", "negative_peaks",
        "mean_std_rms", "mean_std", "std_rms",
        "dxdt_max", "dxdt_min", "dxdt_extrema",
        "d2xdt2_max", "d2xdt2_min", "d2xdt2_extrema",
    ]
    registered = set(sm.implemented_metrics)
    for name in expected_metrics:
        assert name in registered, f"{name} not registered"


def test_summarymetrics_all_contents():
    """__all__ contains exactly the 2 public names."""
    assert set(cubie.outputhandling.summarymetrics.__all__) == {
        "summary_metrics", "register_metric",
    }


def test_summarymetrics_register_metric_accessible():
    """register_metric re-exported from summarymetrics matches source."""
    assert (
        cubie.outputhandling.summarymetrics.register_metric
        is _register_metric
    )


# ── cubie/vendored/__init__.py ──────────────────────────── #

def test_vendored_is_package():
    """vendored is importable and has no exports."""
    assert cubie.vendored.__name__ == "cubie.vendored"
    assert not hasattr(cubie.vendored, "__all__") or len(
        getattr(cubie.vendored, "__all__", [])
    ) == 0


# ── cubie/memory/__init__.py ───────────────────────────── #

@pytest.mark.parametrize(
    "name",
    [
        "current_cupy_stream",
        "CuPySyncNumbaManager",
        "CuPyAsyncNumbaManager",
        "default_memmgr",
        "MemoryManager",
    ],
)
def test_memory_all_contains(name):
    """memory.__all__ lists expected names."""
    assert name in cubie.memory.__all__


def test_memory_all_length():
    """memory.__all__ has 5 entries."""
    assert len(cubie.memory.__all__) == 5


def test_memory_default_memmgr_type():
    """default_memmgr is a MemoryManager instance."""
    # isinstance + identity: type IS the functionality (singleton creation)
    assert isinstance(cubie.memory.default_memmgr, _MemoryManager)
    assert cubie.memory.default_memmgr is cubie.default_memmgr


def test_memory_reexports_from_cupy_emm():
    """CuPy EMM names come from the cupy_emm module."""
    assert cubie.memory.current_cupy_stream is _cupy_emm.current_cupy_stream
    assert (
        cubie.memory.CuPySyncNumbaManager
        is _cupy_emm.CuPySyncNumbaManager
    )
    assert (
        cubie.memory.CuPyAsyncNumbaManager
        is _cupy_emm.CuPyAsyncNumbaManager
    )
