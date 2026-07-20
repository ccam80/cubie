"""Golden-reference gate: cubie vs DifferentialEquations.jl.

Fixed-step Float32 convergence study and adaptive matched-controller
study on the Lorenz ensemble (N=1024, rho in [0, 21], t in [0, 1]).
Cubie integrates the same bit-identical Float32 inputs as the vendored
DifferentialEquations.jl sweeps; final states are judged against the
Float64 golden reference (Vern9, tol 1e-13) and against the Julia
implementation per algorithm. Protocol, thresholds, and provenance:
``ne_gate.py`` and ``data/README.md``.
"""

import pytest

from cubie.integrators.algorithms import resolve_alias

from tests.integrated_numerical_tests.julia_reference.ne_gate import (
    adaptive_table,
    fixed_points,
    fixed_table,
    fixed_verdict,
    julia_adaptive_finals,
    julia_fixed_finals,
    load_algorithms,
    load_controller_constants,
    load_reference,
    matched_controller_settings,
    matched_verdict,
)

pytestmark = pytest.mark.nocudasim

ALGORITHMS = {row["cubie_alias"]: row for row in load_algorithms()}

# Known deviations from DifferentialEquations.jl, tracked in issue #641:
# the matrix-free Krylov inner solves (truncated-Neumann preconditioner,
# fixed inner tolerances) leave a per-step residual floor that Julia's
# dense-LU stage solves do not have, so cubie's error stops converging
# and then grows as dt shrinks. Pinned with xfail(strict=True): fixing
# the implicit stack turns these XPASS, forcing removal of the marks.
KNOWN_FIXED_MISMATCH = {
    "backwards_euler",
    "crank_nicolson",
    "trapezoidal_dirk",
    "implicit_midpoint",
    "sdirk_2_2",
    "l_stable_sdirk_4",
    "radau_iia_5",
    "ros3p",
    "rodas3p",
    "rosenbrock23_sciml",
}
KNOWN_ADAPTIVE_DIVERGENT = {
    "l_stable_sdirk_4",
    "radau_iia_5",
    "rodas3p",
}


def _marks(alias, known_bad):
    marks = [pytest.mark.xdist_group("julia-ne-{0}".format(alias))]
    if alias in known_bad:
        marks.append(pytest.mark.xfail(
            strict=True,
            reason="implicit inner-solve error floor vs Julia dense LU; "
                   "issue #641",
        ))
    return marks


def _cubie_is_adaptive(alias):
    """Whether cubie's implementation carries an embedded error estimate."""
    _, tableau = resolve_alias(alias)
    if tableau is not None:
        return tableau.has_error_estimate
    # Bespoke (non-tableau) steps: crank_nicolson derives an embedded
    # estimate; explicit/backwards euler do not.
    return {"crank_nicolson": True}.get(alias, False)


def _adaptive_aliases():
    """Aliases in the mutual adaptive set with a mappable controller."""
    constants = load_controller_constants()
    with load_reference() as archive:
        keys = set(archive.files)
    aliases = []
    for alias, row in ALGORITHMS.items():
        if not _cubie_is_adaptive(alias):
            continue
        if "adaptive_{0}_tols".format(alias) not in keys:
            continue
        if matched_controller_settings(
                constants, alias, row["order"]) is None:
            continue
        aliases.append(alias)
    return aliases


FIXED_PARAMS = [
    pytest.param(alias, id=alias, marks=_marks(alias, KNOWN_FIXED_MISMATCH))
    for alias in ALGORITHMS
]

ADAPTIVE_PARAMS = [
    pytest.param(alias, id=alias,
                 marks=_marks(alias, KNOWN_ADAPTIVE_DIVERGENT))
    for alias in _adaptive_aliases()
]


@pytest.mark.parametrize("fixed_sweep", FIXED_PARAMS, indirect=True)
def test_fixed_step_matches_julia(fixed_sweep, julia_reference,
                                  golden_ensemble):
    """Fixed-step sweep converges like, and agrees with, the Julia side."""
    alias, cubie_finals = fixed_sweep
    row = ALGORITHMS[alias]
    _, golden_states, scale = golden_ensemble
    julia_finals = julia_fixed_finals(julia_reference, alias)

    points = fixed_points(cubie_finals, julia_finals, golden_states)
    verdict, order_c, order_j = fixed_verdict(points, row["exact"], scale)

    expected = "EQUIVALENT" if row["exact"] else "CONSISTENT"
    assert verdict == expected, (
        "{0} (order {1}, {2}): verdict {3}, expected {4}.\n"
        "Observed order cubie={5:.2f} julia={6:.2f}.\n{7}".format(
            alias, row["order"], row["julia_expr"], verdict, expected,
            order_c, order_j, fixed_table(points)))


@pytest.mark.parametrize(
    "adaptive_matched_sweep", ADAPTIVE_PARAMS, indirect=True)
def test_adaptive_matched_controller_tracks_julia(
        adaptive_matched_sweep, julia_reference, golden_ensemble):
    """Adaptive solve under Julia-matched control tracks the Julia side."""
    alias, cubie_by_tol = adaptive_matched_sweep
    _, golden_states, scale = golden_ensemble
    julia_by_tol = julia_adaptive_finals(julia_reference, alias)

    verdict, worst = matched_verdict(
        cubie_by_tol, julia_by_tol, golden_states, scale)

    assert verdict == "TRACKING", (
        "{0}: matched-controller verdict {1} (worst rms/err ratio "
        "{2:.3g}).\n{3}".format(
            alias, verdict, worst,
            adaptive_table(cubie_by_tol, julia_by_tol, golden_states)))
