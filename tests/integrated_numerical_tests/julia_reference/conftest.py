"""Fixtures for the Julia reference gate.

This directory is a sanctioned exception to the root fixture hierarchy
(see tests/README.md): the gate mirrors the numerical-equivalence
protocol shared with GPUODEBenchmarks, so it pins its own Lorenz
ensemble, solver settings, and dt/tolerance grids instead of deriving
them from ``solver_settings``. Routing it through the shared fixture
tree would break bit-identical-input parity with the vendored Julia
sweeps.
"""

import numpy as np
import pytest

import cubie as qb

from tests.integrated_numerical_tests.julia_reference.ne_gate import (
    DT0_NE,
    DT_MIN_NE,
    DT_MAX_NE,
    DTS_NE,
    INNER_SOLVER_SETTINGS,
    N_NE,
    TOLS_NE,
    load_controller_constants,
    load_reference,
    load_algorithms,
    matched_controller_settings,
)

LORENZ_EQUATIONS = """
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
"""

LORENZ_INITIAL = {'x': 1.0, 'y': 0.0, 'z': 0.0}


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


@pytest.fixture(scope="session")
def lorenz_gate_system():
    """The ne-protocol Lorenz system in Float32."""
    return qb.create_ODE_system(
        LORENZ_EQUATIONS,
        states=dict(LORENZ_INITIAL),
        parameters={'rho': 21.0},
        constants={'sigma': 10.0, 'beta': 8.0 / 3.0},
        name="LorenzJuliaGate",
        precision=np.float32,
    )


def _solve_finals(solver, initials_array, parameter_array):
    """One solve; returns a copied (N_NE, 3) float32 finals array."""
    solution = solver.solve(
        initial_values=initials_array,
        parameters=parameter_array,
        blocksize=64,
        results_type='raw',
        duration=1.0,
    )
    # Copy: the returned array views cubie's output buffer, which the
    # next solve overwrites in place.
    finals = np.array(solution['state'][-1, :, :].T, copy=True)
    if finals.dtype != np.float32:
        raise TypeError(
            "expected float32 output, got {0}".format(finals.dtype))
    if finals.shape != (N_NE, 3):
        raise ValueError(
            "expected ({0}, 3) finals, got {1}".format(N_NE, finals.shape))
    return finals


@pytest.fixture(scope="session")
def fixed_sweep(request, lorenz_gate_system, golden_ensemble):
    """Cubie fixed-step finals over DTS_NE for one algorithm alias.

    Returns (alias, {dt: (N_NE, 3) float32 finals}).
    """
    alias = request.param
    golden_rho, _, _ = golden_ensemble
    solver = qb.Solver(
        lorenz_gate_system,
        algorithm=alias,
        dt=DTS_NE[0],
        save_every=1.0,
        step_controller='fixed',
        output_types=['state'],
        time_logging_level=None,
    )
    # With a fixed step controller cubie does not derive the inner
    # Newton/Krylov tolerances from atol/rtol, so pin them to match what
    # OrdinaryDiffEq enforced in the paired Julia run.
    solver.update(dict(INNER_SOLVER_SETTINGS), silent=True)
    initials_array, parameter_array = solver.build_grid(
        initial_values=dict(LORENZ_INITIAL),
        parameters={'rho': golden_rho},
    )
    finals_by_dt = {}
    for dt in DTS_NE:
        solver.update(dt=dt)
        finals_by_dt[dt] = _solve_finals(
            solver, initials_array, parameter_array)
    return alias, finals_by_dt


@pytest.fixture(scope="session")
def adaptive_matched_sweep(request, lorenz_gate_system, golden_ensemble):
    """Cubie adaptive finals over TOLS_NE under Julia-matched control.

    Returns (alias, {tol: (N_NE, 3) float32 finals}).
    """
    alias = request.param
    golden_rho, _, _ = golden_ensemble
    order = {row["cubie_alias"]: row["order"]
             for row in load_algorithms()}[alias]
    matched = matched_controller_settings(
        load_controller_constants(), alias, order)
    solver = qb.Solver(
        lorenz_gate_system,
        algorithm=alias,
        dt=DT0_NE,
        dt_min=DT_MIN_NE,
        dt_max=DT_MAX_NE,
        atol=TOLS_NE[0],
        rtol=TOLS_NE[0],
        save_every=1.0,
        step_controller=matched["step_controller"],
        output_types=['state'],
        time_logging_level=None,
    )
    extra = {key: value for key, value in matched.items()
             if key != "step_controller"}
    if extra:
        solver.update(extra, silent=True)
    initials_array, parameter_array = solver.build_grid(
        initial_values=dict(LORENZ_INITIAL),
        parameters={'rho': golden_rho},
    )
    finals_by_tol = {}
    for tol in TOLS_NE:
        solver.update(atol=tol, rtol=tol, dt=DT0_NE)
        finals_by_tol[tol] = _solve_finals(
            solver, initials_array, parameter_array)
    return alias, finals_by_tol
