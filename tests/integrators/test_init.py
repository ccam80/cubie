"""Tests for cubie.integrators.__init__."""

from __future__ import annotations

import pytest

import cubie.integrators as integrators
from cubie.integrators import CUBIE_RESULT_CODES


# ── Re-exports ────────────────────────────────────────────── #

_EXPECTED_EXPORTS = [
    "SingleIntegratorRun",
    "BackwardsEulerPCStep",
    "BackwardsEulerStep",
    "CrankNicolsonStep",
    "ExplicitEulerStep",
    "ExplicitStepConfig",
    "ImplicitStepConfig",
    "get_algorithm_step",
    "IVPLoop",
    "MRLinearSolver",
    "MRLinearSolverConfig",
    "LinearSolverCache",
    "BiCGSTABSolver",
    "BiCGSTABSolverConfig",
    "NewtonKrylov",
    "NewtonKrylovConfig",
    "NewtonKrylovCache",
    "AdaptiveIController",
    "AdaptivePIController",
    "AdaptivePIDController",
    "FixedStepController",
    "GustafssonController",
    "get_controller",
]


@pytest.mark.parametrize("name", _EXPECTED_EXPORTS)
def test_reexport_available(name):
    """Each expected symbol is importable from cubie.integrators."""
    obj = getattr(integrators, name)
    # Verify the object originates from the expected submodule
    assert obj.__module__.startswith("cubie.integrators")


# ── CUBIE_RESULT_CODES ────────────────────────────────────── #

@pytest.mark.parametrize(
    "member, value",
    [
        ("SUCCESS", 0),
        ("NEWTON_BACKTRACKING_NO_SUITABLE_STEP", 1),
        ("MAX_NEWTON_ITERATIONS_EXCEEDED", 2),
        ("MAX_LINEAR_ITERATIONS_EXCEEDED", 4),
        ("STEP_TOO_SMALL", 8),
        ("DT_EFF_EFFECTIVELY_ZERO", 16),
        ("MAX_LOOP_ITERS_EXCEEDED", 32),
        ("STAGNATION", 64),
    ],
)
def test_return_code_values(member, value):
    """CUBIE_RESULT_CODES members have correct integer values."""
    assert CUBIE_RESULT_CODES[member] == value
    assert int(CUBIE_RESULT_CODES[member]) == value
