#!/usr/bin/env python
"""Vendor the Julia numerical-equivalence reference into the test suite.

Reads the machine-independent DifferentialEquations.jl sweep outputs and
the Float64 golden reference from a GPUODEBenchmarks checkout and writes
them as a compressed ``.npz`` plus the two small protocol CSVs into
``tests/integrated_numerical_tests/julia_reference/data/``.

The gate tests (``tests/integrated_numerical_tests/julia_reference/``)
consume only the vendored copies, so the benchmark checkout is not needed
to run the test suite. Re-run this script whenever the Julia sweeps in
GPUODEBenchmarks are regenerated:

    python benchmarks/vendor_julia_reference.py [path-to-GPUODEBenchmarks]
"""

import csv
import os
import shutil
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BENCH_ROOT = os.path.join(os.path.dirname(REPO_ROOT),
                                  "GPUODEBenchmarks")
DATA_DIR = os.path.join(REPO_ROOT, "tests", "integrated_numerical_tests",
                        "julia_reference", "data")

N_NE = 1024


def read_ne_csv(path):
    """Read a fixed-sweep file; returns dict {dt: (N_NE, 3) f8 array}."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for dt in sorted({float(row["dt"]) for row in rows}, reverse=True):
        sel = [row for row in rows if float(row["dt"]) == dt]
        sel.sort(key=lambda row: int(row["traj"]))
        out[dt] = np.array(
            [[float(row["x"]), float(row["y"]), float(row["z"])]
             for row in sel], dtype=np.float64)
    return out


def read_ne_adaptive_csv(path):
    """Read an adaptive-sweep file; returns dict {tol: (N_NE, 3) f8}."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for tol in sorted({float(row["tol"]) for row in rows}, reverse=True):
        sel = [row for row in rows if float(row["tol"]) == tol]
        sel.sort(key=lambda row: int(row["traj"]))
        out[tol] = np.array(
            [[float(row["x"]), float(row["y"]), float(row["z"])]
             for row in sel], dtype=np.float64)
    return out


def main():
    bench_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BENCH_ROOT
    julia_dir = os.path.join(bench_root, "data", "numerical_equivalence",
                             "julia")
    golden_path = os.path.join(bench_root, "data", "numerical",
                               "golden_ne_lorenz_1024.csv")
    algorithms_csv = os.path.join(bench_root, "runner_scripts",
                                  "numerical_equivalence", "algorithms.csv")
    constants_csv = os.path.join(julia_dir, "controller_constants.csv")
    for path in (julia_dir, golden_path, algorithms_csv, constants_csv):
        if not os.path.exists(path):
            sys.exit("not found: {0}".format(path))

    os.makedirs(DATA_DIR, exist_ok=True)

    golden = np.loadtxt(golden_path, delimiter=",")
    if golden.shape != (N_NE, 4):
        sys.exit("golden reference has shape {0}, expected ({1}, 4)"
                 .format(golden.shape, N_NE))
    arrays = {
        "golden_rho": golden[:, 0],
        "golden_states": golden[:, 1:],
    }

    with open(algorithms_csv, newline="", encoding="utf-8") as f:
        aliases = [row["cubie_alias"] for row in csv.DictReader(f)]

    for alias in aliases:
        fixed_path = os.path.join(julia_dir, "{0}.csv".format(alias))
        if os.path.isfile(fixed_path):
            per_dt = read_ne_csv(fixed_path)
            dts = sorted(per_dt, reverse=True)
            arrays["fixed_{0}_dts".format(alias)] = np.array(
                dts, dtype=np.float64)
            arrays["fixed_{0}_finals".format(alias)] = np.stack(
                [per_dt[dt] for dt in dts]).astype(np.float32)
            print("fixed    {0}: {1} dts".format(alias, len(dts)))
        else:
            print("fixed    {0}: NO JULIA DATA".format(alias))

        adaptive_path = os.path.join(julia_dir,
                                     "{0}_adaptive.csv".format(alias))
        if os.path.isfile(adaptive_path):
            per_tol = read_ne_adaptive_csv(adaptive_path)
            tols = sorted(per_tol, reverse=True)
            arrays["adaptive_{0}_tols".format(alias)] = np.array(
                tols, dtype=np.float64)
            arrays["adaptive_{0}_finals".format(alias)] = np.stack(
                [per_tol[tol] for tol in tols]).astype(np.float32)
            print("adaptive {0}: {1} tols".format(alias, len(tols)))
        else:
            print("adaptive {0}: no julia adaptive data".format(alias))

    out_npz = os.path.join(DATA_DIR, "julia_reference_ne.npz")
    np.savez_compressed(out_npz, **arrays)
    shutil.copy2(algorithms_csv, os.path.join(DATA_DIR, "algorithms.csv"))
    shutil.copy2(constants_csv,
                 os.path.join(DATA_DIR, "controller_constants.csv"))
    print("wrote {0} ({1:.1f} MiB)".format(
        out_npz, os.path.getsize(out_npz) / 2 ** 20))


if __name__ == "__main__":
    main()
