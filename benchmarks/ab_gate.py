#!/usr/bin/env python
"""A/B kernel-runtime gate: ``main`` vs the active worktree.

Runs ``lorenz_mean_runtime.py`` for each installed CUDA backend, first
against a clean ``main`` tree (A) then the active working tree (B), and
prints one pass/regression line per backend and config from the
mean-of-lowest-``k`` kernel time each run reports. A and B run
back-to-back with the backend held fixed, so the comparison sees only
the source difference.

The only input is the current worktree: A is derived by adding an
ephemeral ``git worktree`` at ``main`` (removed afterwards), B is this
repository. Both sides import ``cubie`` from their own ``src`` via
``PYTHONPATH`` and compile into an isolated, unique cache directory, so
neither the installed editable package nor a warm cache can leak across
the comparison.

Usage::

    python benchmarks/ab_gate.py [--main REF] [--backends numba-cuda,mlir]
        [--n-runs N] [--repeats R] [--min-count K] [--threshold PCT]
        [--keep-main-tree]

Exit status is non-zero if any config regresses beyond ``--threshold``
(default 0.5%).
"""
import argparse
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent / "lorenz_mean_runtime.py"
RESULT_RE = re.compile(r"^RESULT (\S+) ([\d.]+)")

# label -> (spec to test for install, CUBIE_CUDA_BACKEND value)
BACKENDS = {
    "numba-cuda": ("numba.cuda", "numba-cuda"),
    "mlir": ("numba_cuda_mlir", "mlir"),
}


def installed_backends():
    """Return the backend labels whose package is importable."""
    return [
        label
        for label, (spec, _) in BACKENDS.items()
        if importlib.util.find_spec(spec) is not None
    ]


def run_side(tree, backend, cache_dir, passthrough):
    """Run the benchmark in ``tree`` on ``backend``; return {key: ms}."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(tree) / "src")
    env["CUBIE_CUDA_BACKEND"] = BACKENDS[backend][1]
    env["CUBIE_CACHE_DIR"] = str(cache_dir)
    proc = subprocess.run(
        [sys.executable, str(BENCH), *passthrough],
        env=env,
        cwd=str(tree),
        capture_output=True,
        text=True,
    )
    results = {}
    for line in proc.stdout.splitlines():
        match = RESULT_RE.match(line.strip())
        if match:
            results[match.group(1)] = float(match.group(2))
    if not results:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"no RESULT lines from {backend} in {tree}"
        )
    return results


def add_main_worktree(main_ref):
    """Create a detached ephemeral worktree at ``main_ref``."""
    tree = Path(tempfile.mkdtemp(prefix="cubie_main_"))
    subprocess.run(
        ["git", "-C", str(REPO), "worktree", "add", "--detach",
         str(tree), main_ref],
        check=True,
    )
    return tree


def remove_worktree(tree):
    """Remove an ephemeral worktree created by ``add_main_worktree``."""
    subprocess.run(
        ["git", "-C", str(REPO), "worktree", "remove", "--force",
         str(tree)],
        check=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default="main",
                        help="Git ref for the A side (default: main).")
    parser.add_argument("--backends", default=None,
                        help="Comma-separated subset of the installed "
                             "backends (default: all installed).")
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Regression threshold in percent "
                             "(default: 0.5).")
    parser.add_argument("--keep-main-tree", action="store_true",
                        help="Do not remove the ephemeral main worktree "
                             "(keeps its warm compile cache for reruns).")
    args = parser.parse_args()

    backends = installed_backends()
    if args.backends:
        wanted = [b.strip() for b in args.backends.split(",")]
        backends = [b for b in wanted if b in backends]
    if not backends:
        raise SystemExit("no requested CUDA backend is installed")

    # The benchmark takes n_runs then repeats positionally; repeats can
    # only be forwarded alongside n_runs. --min-count is independent.
    passthrough = []
    if args.n_runs is not None:
        passthrough.append(str(args.n_runs))
        passthrough.append(str(args.repeats if args.repeats else 100))
    if args.min_count is not None:
        passthrough.extend(["--min-count", str(args.min_count)])

    branch = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"A = {args.main}   B = {branch} (working tree)   "
          f"backends: {', '.join(backends)}\n")

    main_tree = add_main_worktree(args.main)
    regressed = False
    try:
        cache_base = Path(tempfile.mkdtemp(prefix="cubie_abcache_"))
        for backend in backends:
            a = run_side(main_tree, backend,
                         cache_base / f"A_{backend}", passthrough)
            b = run_side(REPO, backend,
                         cache_base / f"B_{backend}", passthrough)
            for key in a:
                if key not in b:
                    continue
                delta = 100.0 * (b[key] - a[key]) / a[key]
                if delta > args.threshold:
                    verdict = "REGRESSION"
                    regressed = True
                elif delta < -args.threshold:
                    verdict = "improvement"
                else:
                    verdict = "ok"
                print(f"{backend:<11}{key:<10}"
                      f"A {a[key]:8.3f}  B {b[key]:8.3f}  "
                      f"{delta:+6.2f}%  {verdict}")
    finally:
        if not args.keep_main_tree:
            remove_worktree(main_tree)

    print(f"\nGATE: {'REGRESSION' if regressed else 'PASS'} "
          f"(threshold {args.threshold:.2f}%)")
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
