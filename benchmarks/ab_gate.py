#!/usr/bin/env python
"""Interleaved A/B kernel-runtime gate: ``main`` vs the active worktree.

For each installed CUDA backend the gate runs ``lorenz_mean_runtime.py``
several times, interleaving the A (``main``) and B (working-tree) sides
in a drift-balancing ABBA order, and compares the mean-of-lowest-``k``
kernel time each run reports.

Why interleave: the per-run statistic is precise, but the GPU's floor
drifts upward as it warms (measured ~0.8% over back-to-back runs of
the adaptive config). Running all of A then all of B turns that drift
into a false regression on the later side; ABBA places each side at a
balanced point in time so the drift cancels in the side medians. One
throwaway warm-up run per side absorbs the cold-start transient and
the compile cost.

The input grid is written to disk once and shared across sides; each
side keeps its own compile cache directory, so only the warm-up runs
compile.

The only input is the current worktree: A is an ephemeral ``git
worktree`` at ``--main`` (default ``origin/main``; removed afterwards),
B is this repository. Both sides run this repository's benchmark
script but import ``cubie`` from their own ``src`` via ``PYTHONPATH``.

Usage::

    python benchmarks/ab_gate.py [--main REF] [--backends numba-cuda,mlir]
        [--repeats M] [--min-count K] [--pairs P] [--threshold PCT]
        [--n-runs N] [--calibrate] [--keep]

``--calibrate`` points B at ``main`` too (A-vs-A); rerun it a few
times to measure this machine's null |delta| and set ``--threshold``
to about twice the worst of it. On the reference machine (RTX 4070
SUPER driving a live desktop) the four-pair null reaches ~0.26% on
the fixed config, hence the 0.50% default. Exit status is non-zero if
any config regresses past ``--threshold``.
"""
import argparse
import importlib.util
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent / "lorenz_mean_runtime.py"
RESULT_RE = re.compile(r"^RESULT (\S+) ([\d.]+)")

# label -> (importable spec, CUBIE_CUDA_BACKEND value)
BACKENDS = {
    "numba-cuda": ("numba.cuda", "numba-cuda"),
    "mlir": ("numba_cuda_mlir", "mlir"),
}


def installed_backends():
    return [
        label
        for label, (spec, _) in BACKENDS.items()
        if importlib.util.find_spec(spec) is not None
    ]


def run_side(tree, backend, cache_dir, grid_dir, args):
    """Run one benchmark invocation; return {config_key: ms}."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(tree) / "src")
    env["CUBIE_CUDA_BACKEND"] = BACKENDS[backend][1]
    env["CUBIE_CACHE_DIR"] = str(cache_dir)
    cmd = [sys.executable, str(BENCH),
           "--repeats", str(args.repeats),
           "--min-count", str(args.min_count),
           "--grid-cache", str(grid_dir), "--no-clear-cache"]
    if args.n_runs is not None:
        cmd.append(str(args.n_runs))
    proc = subprocess.run(
        cmd, env=env, cwd=str(tree), capture_output=True, text=True,
    )
    out = {}
    for line in proc.stdout.splitlines():
        m = RESULT_RE.match(line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    if proc.returncode != 0 or not out:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"benchmark exited {proc.returncode} with {len(out)} "
            f"RESULT line(s) for {backend} in {tree}"
        )
    return out


def add_main_worktree(main_ref):
    tree = Path(tempfile.mkdtemp(prefix="cubie_main_"))
    subprocess.run(
        ["git", "-C", str(REPO), "worktree", "add", "--detach",
         str(tree), main_ref],
        check=True,
    )
    return tree


def remove_worktree(tree):
    subprocess.run(
        ["git", "-C", str(REPO), "worktree", "remove", "--force",
         str(tree)],
        check=False,
    )


def abba_order(pairs):
    """Side sequence ABBA BAAB ... balancing time between A and B.

    Even ``pairs`` cancels linear drift in the side means; ``pairs``
    divisible by 4 also cancels quadratic drift (plain ABBA repetition
    always leaves B on the inner positions, which biases B under
    curving drift).
    """
    seq = []
    for i in range(pairs):
        seq.extend(("A", "B") if i % 4 in (0, 3) else ("B", "A"))
    return seq


def run_backend(backend, main_tree, b_tree, base, args):
    """Interleaved A/B for one backend; return list of result rows."""
    grid_dir = base / "grid"
    sides = {
        "A": (main_tree, base / f"A_{backend}"),
        "B": (b_tree, base / f"B_{backend}"),
    }

    def invoke(side, note=""):
        print(f"[{backend}] {side}{note}", file=sys.stderr)
        tree, cache_dir = sides[side]
        return run_side(tree, backend, cache_dir, grid_dir, args)

    # Throwaway warm-ups: heat the GPU, fill the grid cache and both
    # compile caches so no measured run pays a compile.
    invoke("A", " (warm-up)")
    invoke("B", " (warm-up)")
    collected = {"A": defaultdict(list), "B": defaultdict(list)}
    for side in abba_order(args.pairs):
        for key, val in invoke(side).items():
            collected[side][key].append(val)

    if set(collected["A"]) != set(collected["B"]):
        raise SystemExit(
            f"config keys differ between sides: "
            f"A={sorted(collected['A'])} B={sorted(collected['B'])}"
        )
    rows = []
    for key in sorted(collected["A"]):
        a = statistics.median(collected["A"][key])
        b = statistics.median(collected["B"][key])
        delta = 100.0 * (b - a) / a
        if delta > args.threshold:
            verdict = "REGRESSION"
        elif delta < -args.threshold:
            verdict = "improvement"
        else:
            verdict = "ok"
        rows.append((backend, key, a, b, delta, verdict,
                     collected["A"][key], collected["B"][key]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default="origin/main",
                        help="A-side ref (default origin/main; the "
                             "local main branch is often stale).")
    parser.add_argument("--backends", default=None,
                        help="Comma-separated subset of: "
                             + ", ".join(BACKENDS))
    parser.add_argument("--repeats", type=int, default=300,
                        help="Solves per invocation (default 300; 100 "
                             "gives the same null on a quiet GPU in "
                             "under half the wall time, but a longer "
                             "window rides out bursty contention).")
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=4,
                        help="A/B pairs per backend; multiples of 4 "
                             "cancel linear and quadratic drift.")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="Trajectory count for both configs "
                             "(small values smoke-test the harness).")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Regression threshold in percent (about "
                             "twice the calibrated null).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Point B at main too (A-vs-A null).")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the ephemeral main tree and caches.")
    args = parser.parse_args()

    backends = installed_backends()
    if args.backends:
        wanted = [b.strip() for b in args.backends.split(",")]
        missing = [b for b in wanted if b not in backends]
        if missing:
            raise SystemExit(
                f"backend(s) not installed: {', '.join(missing)}"
            )
        backends = wanted
    if not backends:
        raise SystemExit("no CUDA backend is installed")

    branch = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    a_sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", args.main],
        capture_output=True, text=True,
    ).stdout.strip()
    b_label = args.main if args.calibrate else f"{branch} (working tree)"
    print(f"A = {args.main} ({a_sha})   B = {b_label}   "
          f"backends: {', '.join(backends)}   "
          f"({args.repeats} solves x {args.pairs} runs/side + warm-ups)"
          f"\n")

    main_tree = add_main_worktree(args.main)
    base = Path(tempfile.mkdtemp(prefix="cubie_abgate_"))
    b_tree = main_tree if args.calibrate else REPO
    regressed = False
    try:
        for backend in backends:
            for row in run_backend(backend, main_tree, b_tree, base,
                                   args):
                bk, key, a, b, delta, verdict, av, bv = row
                if verdict == "REGRESSION":
                    regressed = True
                spread = (f"  [A {min(av):.3f}-{max(av):.3f} "
                          f"B {min(bv):.3f}-{max(bv):.3f}]")
                print(f"{bk:<11}{key:<10}A {a:8.3f}  B {b:8.3f}  "
                      f"{delta:+6.2f}%  {verdict}{spread}")
    finally:
        if not args.keep:
            remove_worktree(main_tree)
            shutil.rmtree(base, ignore_errors=True)

    print(f"\nGATE: {'REGRESSION' if regressed else 'PASS'} "
          f"(threshold {args.threshold:.2f}%)")
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
