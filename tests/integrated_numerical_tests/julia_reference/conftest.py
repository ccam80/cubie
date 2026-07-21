"""Fixtures for the Julia reference gate.

Solvers and the Lorenz system flow through the shared
``solver_settings`` hierarchy: each algorithm under test is one
``solver_settings_override`` parameter set (see
``test_julia_reference.py``). The fixtures here pin the vendored
DifferentialEquations.jl data and run the per-algorithm dt and
tolerance sweeps the numerical-equivalence protocol requires.
"""

import numpy as np
import pytest

from tests.integrated_numerical_tests.julia_reference.ne_gate import (
    DT0_NE,
    DTS_NE,
    N_NE,
    TOLS_NE,
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


def _solve_finals(solver, solver_settings, initials_array, parameter_array):
    """One solve; returns a copied (N_NE, 3) float32 finals array."""
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
    return finals


def _gate_grid(solver, golden_ensemble):
    """Julia-ensemble input arrays: default initials over the rho grid."""
    golden_rho, _, _ = golden_ensemble
    return solver.build_grid(
        initial_values=dict(LORENZ_JULIA_STATES),
        parameters={'rho': golden_rho},
    )


@pytest.fixture(scope="function")
def fixed_sweep(solver_mutable, solver_settings, golden_ensemble):
    """Cubie fixed-step finals over DTS_NE for the configured algorithm.

    Returns (alias, {dt: (N_NE, 3) float32 finals}).
    """
    alias = solver_settings["algorithm"]
    initials_array, parameter_array = _gate_grid(
        solver_mutable, golden_ensemble)
    finals_by_dt = {}
    for dt in DTS_NE:
        solver_mutable.update(dt=dt)
        finals_by_dt[dt] = _solve_finals(
            solver_mutable, solver_settings, initials_array,
            parameter_array)
    return alias, finals_by_dt


@pytest.fixture(scope="function")
def adaptive_matched_sweep(solver_mutable, solver_settings, golden_ensemble):
    """Cubie adaptive finals over TOLS_NE under Julia-matched control.

    Returns (alias, {tol: (N_NE, 3) float32 finals}).
    """
    alias = solver_settings["algorithm"]
    initials_array, parameter_array = _gate_grid(
        solver_mutable, golden_ensemble)
    finals_by_tol = {}
    for tol in TOLS_NE:
        solver_mutable.update(atol=tol, rtol=tol, dt=DT0_NE)
        finals_by_tol[tol] = _solve_finals(
            solver_mutable, solver_settings, initials_array,
            parameter_array)
    return alias, finals_by_tol
