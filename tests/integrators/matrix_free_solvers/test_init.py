"""Tests for cubie.integrators.matrix_free_solvers.__init__."""

from __future__ import annotations

import pytest

import cubie.integrators.matrix_free_solvers as mfs
from cubie.integrators.matrix_free_solvers import SolverRetCodes


# ── Re-exports (items 1-3) ───────────────────────────────── #

_EXPECTED_EXPORTS = [
    ("MatrixFreeSolverConfig", "cubie.integrators.matrix_free_solvers"),
    ("LinearSolver", "cubie.integrators.matrix_free_solvers"),
    ("LinearSolverConfig", "cubie.integrators.matrix_free_solvers"),
    ("LinearSolverCache", "cubie.integrators.matrix_free_solvers"),
    ("NewtonKrylov", "cubie.integrators.matrix_free_solvers"),
    ("NewtonKrylovConfig", "cubie.integrators.matrix_free_solvers"),
    ("NewtonKrylovCache", "cubie.integrators.matrix_free_solvers"),
]


@pytest.mark.parametrize("name, mod_prefix", _EXPECTED_EXPORTS)
def test_reexport_available(name, mod_prefix):
    """Each expected symbol is importable from the package."""
    obj = getattr(mfs, name)
    assert obj.__module__.startswith(mod_prefix)


# ── SolverRetCodes (items 4-7) ───────────────────────────── #

@pytest.mark.parametrize(
    "member, value",
    [
        ("SUCCESS", 0),
        ("NEWTON_BACKTRACKING_NO_SUITABLE_STEP", 1),
        ("MAX_NEWTON_ITERATIONS_EXCEEDED", 2),
        ("MAX_LINEAR_ITERATIONS_EXCEEDED", 4),
    ],
)
def test_solver_ret_code_values(member, value):
    """SolverRetCodes members have correct integer values."""
    assert SolverRetCodes[member] == value
    assert int(SolverRetCodes[member]) == value
