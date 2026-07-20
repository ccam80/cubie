"""Protocol constants and verdict logic for the Julia reference gate.

Ports the numerical-equivalence protocol shared with GPUODEBenchmarks
(``runner_scripts/numerical_equivalence/ne_common.py`` and
``compare_numerical_equivalence.py``): cubie and raw
DifferentialEquations.jl both integrate the same Float32 Lorenz ensemble
and their final states are judged against a Float64 golden reference
(Vern9 at tol 1e-13) and against each other. The Julia and golden data
are vendored under ``data/`` by ``benchmarks/vendor_julia_reference.py``.

Keep the constants in sync with the GPUODEBenchmarks protocol; the gate
is only meaningful while both sides integrate bit-identical inputs on
the same grids.
"""

import csv
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Dyadic dt grid, 2^-1 .. 2^-13: every dt, save and end boundary is exact
# in binary floating point (Float32 included). Order >= 5 methods hit the
# float32 error floor by dt ~ 1/32 on this problem, so the convergence
# region is only observable at coarse dt.
DTS_NE = [2.0 ** -k for k in range(1, 14)]

# Adaptive sweep: atol = rtol grid, 1e-2 .. 1e-6. Below 1e-6 a float32
# solve cannot honor the request (the error floor is ~1e-6 relative).
TOLS_NE = [10.0 ** -k for k in range(2, 7)]

# Shared adaptive-run pins (both stacks): initial dt and the dt clamps.
DT0_NE = 0.01
DT_MIN_NE = 1e-6
DT_MAX_NE = 0.5

N_NE = 1024

# Solve-request tolerances for the fixed sweep: OrdinaryDiffEq's
# defaults (abstol = 1e-6, reltol = 1e-3), which the paired Julia
# fixed-step runs used for their nonlinear stage solves.
ATOL_FIXED_NE = 1e-6
RTOL_FIXED_NE = 1e-3

# Error bounds (relative to the golden scale) delimiting where each check
# applies. Below FLOOR_REL the error is float32 roundoff, not truncation.
# Above ORDER_CAP_REL the solve has left the asymptotic regime and the
# observed order is meaningless. The equivalence check tolerates a much
# looser cap: two implementations of the same tableau still track each
# other closely at coarse dt where the error is large but finite.
FLOOR_REL = 4e-6
ORDER_CAP_REL = 3e-2
EQ_CAP_REL = 5e-1

# Exact pairs (identical tableau): mutual rms distance must stay well
# below the truncation error once above the roundoff floor allowance.
EQ_FRACTION = 0.25
EQ_FLOOR_MULT = 3.0

# Non-exact pairs (same method class, different tableau): same observed
# order and errors of comparable magnitude.
ORDER_TOL = 0.6
RATIO_LIM = 4.0

# Adaptive matched-controller tier: the mutual rms distance must not
# exceed this multiple of the Julia run's own error at any in-range
# tolerance.
ADAPTIVE_EQ_FRACTION = 2.0


def load_algorithms():
    """Return the mutual algorithm table as a list of dicts.

    Keys: ``cubie_alias``, ``julia_expr``, ``order`` (int), ``family``,
    ``exact`` (bool — identical tableau on both sides, so per-trajectory
    equivalence is expected rather than just matching order/error
    magnitude), ``notes``.
    """
    path = os.path.join(DATA_DIR, "algorithms.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["order"] = int(row["order"])
        row["exact"] = row["exact"].strip().lower() == "true"
    return rows


def load_controller_constants():
    """Julia's resolved default-controller constants by cubie alias."""
    path = os.path.join(DATA_DIR, "controller_constants.csv")
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            entry = {"controller": row["controller"]}
            for key in ("beta1", "beta2", "qmin", "qmax", "gamma",
                        "qsteady_min", "qsteady_max", "order"):
                raw = row.get(key, "")
                entry[key] = float(raw) if raw not in ("", None) else None
            out[row["cubie_alias"]] = entry
    return out


def load_reference():
    """Load the vendored npz; returns the raw archive mapping."""
    path = os.path.join(DATA_DIR, "julia_reference_ne.npz")
    return np.load(path)


def julia_fixed_finals(reference, alias):
    """Vendored Julia fixed-sweep finals; dict {dt: (N_NE, 3) f8}."""
    dts = reference["fixed_{0}_dts".format(alias)]
    finals = reference["fixed_{0}_finals".format(alias)]
    return {float(dt): finals[i].astype(np.float64)
            for i, dt in enumerate(dts)}


def julia_adaptive_finals(reference, alias):
    """Vendored Julia adaptive finals; dict {tol: (N_NE, 3) f8}."""
    key = "adaptive_{0}_tols".format(alias)
    if key not in reference:
        return None
    tols = reference[key]
    finals = reference["adaptive_{0}_finals".format(alias)]
    return {float(tol): finals[i].astype(np.float64)
            for i, tol in enumerate(tols)}


def matched_controller_settings(constants, alias, order):
    """Cubie controller kwargs mirroring Julia's resolved defaults.

    Julia's PI updates dt*gamma*EEst^(-beta1)*errold^(+beta2); cubie's PI
    gain is safety*EEst^(-kp/(order+1))*errold^(-ki/(order+1)) with order
    the classical order it feeds the exponent, so kp = beta1*(order+1)
    and ki = -beta2*(order+1). qmin/qmax bound the same gain quantity as
    cubie's min_gain/max_gain, and Julia's qsteady deadband acts on
    q = 1/gain, hence the inverted bounds. Julia's PredictiveController
    (Radau) maps to cubie's gustafsson controller — same Gustafsson
    family, matched safety only (documented approximate match).
    """
    c = constants.get(alias)
    if c is None:
        return None
    if c["controller"] == "PIController":
        required = ("beta1", "beta2", "gamma", "qmin", "qmax",
                    "qsteady_min", "qsteady_max")
        missing = [key for key in required if c[key] is None]
        if missing:
            raise ValueError(
                "controller constants for '{0}' are missing {1}; "
                "regenerate data/ with "
                "benchmarks/vendor_julia_reference.py".format(
                    alias, ", ".join(missing)))
        return {
            "step_controller": "pi",
            "kp": c["beta1"] * (order + 1),
            "ki": -c["beta2"] * (order + 1),
            "safety": c["gamma"],
            "min_gain": c["qmin"],
            "max_gain": c["qmax"],
            "deadband_min": 1.0 / c["qsteady_max"],
            "deadband_max": 1.0 / c["qsteady_min"],
        }
    if c["controller"] == "PredictiveController":
        return {
            "step_controller": "gustafsson",
            "gamma": c["gamma"],
        }
    return None


def ensemble_error(final_states, golden_states):
    """l2-at-final error over the ensemble, computed in float64."""
    diff = np.asarray(final_states, dtype=np.float64) - golden_states
    return float(np.sqrt(np.mean(diff ** 2)))


def observed_orders(errs_by_dt, floor, cap):
    """Median log2 error ratio between successive dt halvings in-region."""
    orders = []
    for k in range(len(DTS_NE) - 1):
        d1, d2 = DTS_NE[k], DTS_NE[k + 1]
        if d1 in errs_by_dt and d2 in errs_by_dt:
            e1, e2 = errs_by_dt[d1], errs_by_dt[d2]
            if (np.isfinite(e1) and np.isfinite(e2)
                    and floor < e1 < cap and floor < e2 < cap and e2 > 0):
                orders.append(np.log2(e1 / e2))
    return float(np.median(orders)) if orders else float("nan")


def fixed_points(cubie_finals, julia_finals, golden_states):
    """Per-dt error table: (dt, err_cubie, err_julia, rms_diff)."""
    points = []
    for dt in DTS_NE:
        fc = cubie_finals.get(dt)
        fj = julia_finals.get(dt)
        err_c = ensemble_error(fc, golden_states) if fc is not None else None
        err_j = ensemble_error(fj, golden_states) if fj is not None else None
        rms_diff = None
        if fc is not None and fj is not None:
            with np.errstate(invalid="ignore", over="ignore"):
                d = np.asarray(fc, dtype=np.float64) - fj
            rms_diff = float(np.sqrt(np.mean(d ** 2)))
        points.append((dt, err_c, err_j, rms_diff))
    return points


def fixed_verdict(points, exact, scale):
    """Verdict for one algorithm's fixed sweep.

    Returns (verdict, order_cubie, order_julia). EQUIVALENT: identical
    tableau and mutual rms difference below EQ_FRACTION of the truncation
    error at every in-region dt. CONSISTENT: different tableau of the
    same order; observed orders and error magnitudes agree. MISMATCH:
    neither holds. INSUFFICIENT OVERLAP: fewer than two dts where both
    sides landed in the comparable-error region.
    """
    floor = FLOOR_REL * scale
    order_cap = ORDER_CAP_REL * scale
    eq_cap = EQ_CAP_REL * scale

    order_c = observed_orders(
        {dt: e for dt, e, _, _ in points if e is not None},
        floor, order_cap)
    order_j = observed_orders(
        {dt: e for dt, _, e, _ in points if e is not None},
        floor, order_cap)

    checked = 0
    julia_valid = 0
    ok = True
    for dt, err_c, err_j, rms_diff in points:
        if err_c is None or err_j is None:
            continue
        if np.isfinite(err_j) and floor < err_j < eq_cap:
            julia_valid += 1
        if not np.isfinite(err_c) or not np.isfinite(err_j):
            continue
        emax = max(err_c, err_j)
        if not (floor < emax < eq_cap):
            continue
        checked += 1
        if exact:
            if rms_diff > max(EQ_FRACTION * emax, EQ_FLOOR_MULT * floor):
                ok = False
        else:
            ratio = err_c / err_j if err_j > 0 else float("inf")
            if not (1.0 / RATIO_LIM < ratio < RATIO_LIM):
                ok = False
    if not exact and np.isfinite(order_c) and np.isfinite(order_j):
        if abs(order_c - order_j) > ORDER_TOL:
            ok = False

    if checked < 2:
        # Julia produced sane in-region errors that cubie never matched
        # anywhere (out-of-region / non-finite): that is divergence, not
        # missing data.
        verdict = ("MISMATCH" if julia_valid >= 2
                   else "INSUFFICIENT OVERLAP")
    elif ok:
        verdict = "EQUIVALENT" if exact else "CONSISTENT"
    else:
        verdict = "MISMATCH"
    return verdict, order_c, order_j


def matched_verdict(cubie_by_tol, julia_by_tol, golden_states, scale):
    """Verdict for the adaptive matched-controller tier.

    Returns (verdict, worst_ratio). TRACKING when the mutual rms distance
    stays within ADAPTIVE_EQ_FRACTION of the Julia run's own error at
    every tolerance where the Julia solve is in a sane range.
    """
    floor = FLOOR_REL * scale
    eq_cap = EQ_CAP_REL * scale
    checked = 0
    worst = 0.0
    for tol in TOLS_NE:
        fj = julia_by_tol.get(tol)
        fc = cubie_by_tol.get(tol)
        if fj is None or fc is None:
            continue
        err_j = ensemble_error(fj, golden_states)
        if not np.isfinite(err_j) or not (floor < err_j < eq_cap):
            continue
        with np.errstate(invalid="ignore", over="ignore"):
            d = np.asarray(fc, dtype=np.float64) - fj
        rms_diff = float(np.sqrt(np.mean(d ** 2)))
        checked += 1
        worst = max(worst, rms_diff / max(err_j, EQ_FLOOR_MULT * floor))
    if checked == 0:
        return "INSUFFICIENT OVERLAP", worst
    if worst <= ADAPTIVE_EQ_FRACTION:
        return "TRACKING", worst
    return "DIVERGENT", worst


def _fmt(value, spec="{0:.3e}"):
    return spec.format(value) if value is not None else "-"


def fixed_table(points):
    """Human-readable per-dt error table for assertion messages."""
    lines = ["dt           err_cubie   err_julia   rms_diff"]
    for dt, err_c, err_j, rms_diff in points:
        lines.append("{0:<12.10g} {1:<11s} {2:<11s} {3}".format(
            dt, _fmt(err_c), _fmt(err_j), _fmt(rms_diff)))
    return "\n".join(lines)


def adaptive_table(cubie_by_tol, julia_by_tol, golden_states):
    """Human-readable per-tolerance error table for assertion messages."""
    lines = ["tol      err_cubie   err_julia   rms_diff"]
    for tol in TOLS_NE:
        fc = cubie_by_tol.get(tol)
        fj = julia_by_tol.get(tol)
        err_c = ensemble_error(fc, golden_states) if fc is not None else None
        err_j = ensemble_error(fj, golden_states) if fj is not None else None
        rms_diff = None
        if fc is not None and fj is not None:
            with np.errstate(invalid="ignore", over="ignore"):
                d = np.asarray(fc, dtype=np.float64) - fj
            rms_diff = float(np.sqrt(np.mean(d ** 2)))
        lines.append("{0:<8.0e} {1:<11s} {2:<11s} {3}".format(
            tol, _fmt(err_c), _fmt(err_j), _fmt(rms_diff)))
    return "\n".join(lines)
