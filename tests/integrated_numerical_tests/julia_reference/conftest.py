"""Fixtures for the Julia reference gate.

Solvers and the Lorenz system flow through the shared
``solver_settings`` hierarchy: each algorithm under test is one
``solver_settings_override`` parameter set carrying its pinned dt or
tolerance (see ``test_julia_reference.py``). The fixtures here pin the
vendored DifferentialEquations.jl data and run the single per-algorithm
solve the numerical-equivalence protocol requires.
"""

import numpy as np
import pytest

from tests.integrated_numerical_tests.julia_reference.ne_gate import (
    N_NE,
    load_reference,
)
from tests.system_fixtures import LORENZ_JULIA_STATES


@pytest.fixture(scope="session")
def julia_reference():
    """Vendored Julia sweep arrays, loaded into memory."""
    with load_reference() as archive:
        return {key: archive[key] for key in archive.files}


@pytest.fixture(scope="session")
def golden_ensemble(julia_reference):
    """Float64 golden reference: (rho grid, final states, rms scale)."""
    rho = julia_reference["golden_rho"]
    states = julia_reference["golden_states"]
    scale = float(np.sqrt(np.mean(states ** 2)))
    return rho, states, scale


@pytest.fixture(scope="function")
def gate_final(solver, solver_settings, golden_ensemble):
    """One cubie solve at the pinned settings; (alias, finals).

    The pinned dt or tolerance arrives through the solver settings
    override, so the session solver compiles once and solves once.
    Returns (alias, (N_NE, 3) float32 finals).
    """
    golden_rho, _, _ = golden_ensemble
    initials_array, parameter_array = solver.build_grid(
        initial_values=dict(LORENZ_JULIA_STATES),
        parameters={'rho': golden_rho},
    )
    solution = solver.solve(
        initial_values=initials_array,
        parameters=parameter_array,
        blocksize=solver_settings["blocksize"],
        duration=solver_settings["duration"],
    )
    # Copy: time_domain_array views cubie's output buffer (state only,
    # no time column), which the next solve overwrites in place. The
    # solve default leaves errored runs (nonzero status) as NaN, so a
    # trajectory cubie could not solve reads as a failure here rather
    # than as a raw diverged final.
    finals = np.array(
        solution.time_domain_array[-1, :, :].T, copy=True)
    if finals.dtype != np.float32:
        raise TypeError(
            "expected float32 output, got {0}".format(finals.dtype))
    if finals.shape != (N_NE, 3):
        raise ValueError(
            "expected ({0}, 3) finals, got {1}".format(N_NE, finals.shape))
    return solver_settings["algorithm"], finals
