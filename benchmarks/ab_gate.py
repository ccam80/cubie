#!/usr/bin/env python
"""Block-interleaved A/B kernel-runtime gate: ``main`` vs the worktree.

For each installed CUDA backend the gate starts two persistent
``lorenz_mean_runtime.py --worker`` processes — A imports ``cubie``
from an ephemeral ``git worktree`` at ``--main`` (default
``origin/main``; removed afterwards), B from this repository — and
ping-pongs short solve blocks between them in a drift-balancing ABBA
order. Each block's statistic is the mean of its lowest ``k``
per-solve kernel times (CUDA events, kernel only); the two blocks of
an ABBA pair ran back to back, so the verdict statistic is the
median of the paired percent deltas.

Why blocks: the floor of the kernel-time distribution tracks the
compiled kernel's intrinsic cost but wanders a few tenths of a
percent with GPU clock state. The blocks of a pair run seconds
apart, so they sample nearly the same clock state and the wander
cancels inside each pair — and each side pays its process startup,
JIT compile, and grid build once instead of once per sample. Verdict
reliability is judged from the per-pair deltas themselves: when
their median absolute deviation exceeds half the threshold the row
is marked DISTRUST — rerun, with more ``--pairs`` or on a quieter
GPU, before acting on it.
Constant background load inflates absolute times but cancels out of
the deltas. Both workers hold their device pools concurrently, which
is fine at the default sizes (~0.7 GB each); much larger ``--n-runs``
could change chunking between sides.

Usage::

    python benchmarks/ab_gate.py [--main REF] [--backends numba-cuda,mlir]
        [--block-solves N] [--pairs P] [--min-count K]
        [--threshold PCT] [--n-runs N] [--calibrate] [--keep]

``--calibrate`` points B at ``main`` too (A-vs-A); rerun it a few
times to measure this machine's null |delta| and set ``--threshold``
to two to three times the worst of it. Exit status is non-zero if any
config regresses past ``--threshold``.
"""
import argparse
import importlib.util
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent / "lorenz_mean_runtime.py"

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


def start_worker(tree, backend, cache_dir, grid_dir, args):
    """Start one persistent benchmark worker; return the process."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(tree) / "src")
    env["CUBIE_CUDA_BACKEND"] = BACKENDS[backend][1]
    env["CUBIE_CACHE_DIR"] = str(cache_dir)
    cmd = [sys.executable, str(BENCH), "--worker",
           "--grid-cache", str(grid_dir), "--no-clear-cache"]
    if args.n_runs is not None:
        cmd.append(str(args.n_runs))
    return subprocess.Popen(
        cmd, env=env, cwd=str(tree), stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True, bufsize=1,
    )


def read_reply(proc, prefix, side):
    """Read worker stdout lines until one starts with ``prefix``."""
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit(
                f"worker {side} exited (code {proc.poll()})"
            )
        line = line.strip()
        if line.startswith(prefix):
            return line


def run_block(proc, side, key, count):
    """Ask one worker for ``count`` solves; return per-solve ms."""
    try:
        proc.stdin.write(f"run {key} {count}\n")
        proc.stdin.flush()
    except OSError:
        raise SystemExit(
            f"worker {side} pipe closed (exit {proc.poll()})"
        )
    reply = read_reply(proc, "@TIMES ", side).split()
    if reply[1] != key:
        raise SystemExit(
            f"worker {side} answered for {reply[1]!r}, "
            f"expected {key!r}"
        )
    return [float(v) for v in reply[2:]]


def stop_worker(proc):
    try:
        proc.stdin.write("quit\n")
        proc.stdin.flush()
    except OSError:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


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
    """Block-interleaved A/B for one backend; return result rows."""
    grid_dir = base / "grid"
    workers = {}
    try:
        for side, tree in (("A", main_tree), ("B", b_tree)):
            workers[side] = start_worker(
                tree, backend, base / f"{side}_{backend}", grid_dir,
                args)
        ready = {}
        for side in ("A", "B"):
            ready[side] = read_reply(
                workers[side], "@READY", side).split()[1:]
            print(f"[{backend}] {side} ready", file=sys.stderr)
        if ready["A"] != ready["B"]:
            raise SystemExit(
                f"config keys differ between sides: "
                f"A={ready['A']} B={ready['B']}"
            )
        keys = ready["A"]

        def block(side, store=None):
            for key in keys:
                vals = run_block(workers[side], side, key,
                                 args.block_solves)
                if store is not None:
                    store[side][key].append(vals)
            # Idle gap: without it the GPU sits pinned at its power
            # limit and the kernel-time floor dithers block to block;
            # a rest lets it re-enter the repeatable boost state. The
            # duration is drawn fresh per block so a concurrent
            # periodic GPU load cannot phase-lock with the ping-pong
            # rhythm and bias one side coherently.
            time.sleep(random.uniform(*args.gap))

        # One throwaway ABBA round rides out the post-setup ramp.
        for side in ("A", "B", "B", "A"):
            block(side)
        order = abba_order(args.pairs)
        print(f"[{backend}] measuring {''.join(order)}",
              file=sys.stderr)
        collected = {s: {k: [] for k in keys} for s in ("A", "B")}
        for side in order:
            block(side, collected)
    finally:
        for proc in workers.values():
            stop_worker(proc)

    rows = []
    for key in keys:
        floors = {
            side: [
                statistics.fmean(sorted(blk)[:args.min_count])
                for blk in collected[side][key]
            ]
            for side in ("A", "B")
        }
        a = statistics.fmean(floors["A"])
        b = statistics.fmean(floors["B"])
        # The i-th A and B blocks ran back to back (one ABBA pair),
        # so they share clock state; the median paired delta cancels
        # wander that survives in the side means.
        deltas = [
            100.0 * (bf - af) / af
            for af, bf in zip(floors["A"], floors["B"])
        ]
        delta = statistics.median(deltas)
        if delta > args.threshold:
            verdict = "REGRESSION"
        elif delta < -args.threshold:
            verdict = "improvement"
        else:
            verdict = "ok"
        # The verdict is only as good as the pairs' agreement: a
        # median absolute deviation past half the threshold means
        # the reported delta could plausibly sit on the other side
        # of the verdict boundary.
        mad = statistics.median(abs(d - delta) for d in deltas)
        distrust = mad > 0.5 * args.threshold
        rows.append((backend, key, a, b, delta, verdict, distrust))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default="origin/main",
                        help="A-side ref (default origin/main; the "
                             "local main branch is often stale).")
    parser.add_argument("--backends", default=None,
                        help="Comma-separated subset of: "
                             + ", ".join(BACKENDS))
    parser.add_argument("--block-solves", type=int, default=25,
                        help="Solves per block per config.")
    parser.add_argument("--pairs", type=int, default=4,
                        help="A/B block pairs per backend; even "
                             "cancels linear drift, multiples of 4 "
                             "also quadratic.")
    parser.add_argument("--min-count", type=int, default=5,
                        help="Lowest per-solve kernel times averaged "
                             "into each block's statistic.")
    parser.add_argument("--gap", type=float, nargs=2, default=(1.5, 3.5),
                        metavar=("MIN", "MAX"),
                        help="Idle seconds after each block, drawn "
                             "uniformly per block. The rest lets the "
                             "GPU re-enter its repeatable boost state; "
                             "the jitter stops concurrent periodic GPU "
                             "loads phase-locking with the rhythm.")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="Trajectory count for both configs "
                             "(small values smoke-test the harness).")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Regression threshold in percent (two "
                             "to three times the calibrated null).")
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
          f"({args.pairs} block pairs x {args.block_solves} "
          f"solves/block/side)\n")

    main_tree = add_main_worktree(args.main)
    base = Path(tempfile.mkdtemp(prefix="cubie_abgate_"))
    b_tree = main_tree if args.calibrate else REPO
    regressed = False
    distrusted = False
    try:
        for backend in backends:
            for row in run_backend(backend, main_tree, b_tree, base,
                                   args):
                bk, key, a, b, delta, verdict, distrust = row
                regressed = regressed or verdict == "REGRESSION"
                distrusted = distrusted or distrust
                flag = "  DISTRUST" if distrust else ""
                print(f"{bk:<11}{key:<10}A {a:8.3f}  B {b:8.3f}  "
                      f"{delta:+6.2f}%  {verdict}{flag}")
    finally:
        if not args.keep:
            remove_worktree(main_tree)
            shutil.rmtree(base, ignore_errors=True)

    print(f"\nGATE: {'REGRESSION' if regressed else 'PASS'} "
          f"(threshold {args.threshold:.2f}%)")
    if distrusted:
        print("DISTRUST = the per-pair deltas disagree, so the "
              "verdict is unreliable; rerun, with more --pairs or a "
              "quieter GPU, before acting on it.")
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
