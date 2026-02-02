"""Tests for cubie.integrators.__init__."""

from __future__ import annotations

import pytest

import cubie.integrators as integrators
from cubie.integrators import IntegratorReturnCodes


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
    "LinearSolver",
    "LinearSolverConfig",
    "LinearSolverCache",
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


# ── IntegratorReturnCodes ─────────────────────────────────── #

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
    ],
)
def test_return_code_values(member, value):
    """IntegratorReturnCodes members have correct integer values."""
    assert IntegratorReturnCodes[member] == value
    assert int(IntegratorReturnCodes[member]) == value
