"""Tests for cubie.integrators.matrix_free_solvers.__init__."""

from __future__ import annotations

import pytest

import cubie.integrators.matrix_free_solvers as mfs
from cubie.integrators.matrix_free_solvers import CUBIE_RESULT_CODES


# ── Re-exports (items 1-3) ───────────────────────────────── #

_EXPECTED_EXPORTS = [
    ("MatrixFreeSolverConfig", "cubie.integrators.matrix_free_solvers"),
    ("LinearSolverBase", "cubie.integrators.matrix_free_solvers"),
    ("LinearSolverBaseConfig", "cubie.integrators.matrix_free_solvers"),
    ("LinearSolverCache", "cubie.integrators.matrix_free_solvers"),
    ("MRLinearSolver", "cubie.integrators.matrix_free_solvers"),
    ("MRLinearSolverConfig", "cubie.integrators.matrix_free_solvers"),
    ("BiCGSTABSolver", "cubie.integrators.matrix_free_solvers"),
    ("BiCGSTABSolverConfig", "cubie.integrators.matrix_free_solvers"),
    ("NewtonKrylov", "cubie.integrators.matrix_free_solvers"),
    ("NewtonKrylovConfig", "cubie.integrators.matrix_free_solvers"),
    ("NewtonKrylovCache", "cubie.integrators.matrix_free_solvers"),
]


@pytest.mark.parametrize("name, mod_prefix", _EXPECTED_EXPORTS)
def test_reexport_available(name, mod_prefix):
    """Each expected symbol is importable from the package."""
    obj = getattr(mfs, name)
    assert obj.__module__.startswith(mod_prefix)


# ── CUBIE_RESULT_CODES (re-export) ───────────────────────── #

@pytest.mark.parametrize(
    "member, value",
    [
        ("SUCCESS", 0),
        ("NEWTON_BACKTRACKING_NO_SUITABLE_STEP", 1),
        ("MAX_NEWTON_ITERATIONS_EXCEEDED", 2),
        ("MAX_LINEAR_ITERATIONS_EXCEEDED", 4),
        ("BICGSTAB_BREAKDOWN", 128),
    ],
)
def test_solver_ret_code_values(member, value):
    """CUBIE_RESULT_CODES members have correct integer values."""
    assert CUBIE_RESULT_CODES[member] == value
    assert int(CUBIE_RESULT_CODES[member]) == value
