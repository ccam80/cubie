#!/usr/bin/env python
"""Block-interleaved A/B kernel-runtime gate: ``main`` vs the worktree.

For each installed CUDA backend the gate starts two persistent
``lorenz_mean_runtime.py --worker`` processes — A imports ``cubie``
from an ephemeral ``git worktree`` at ``--main`` (default
``origin/main``; removed afterwards), B from this repository — and
ping-pongs short solve blocks between them in a drift-balancing ABBA
order. Each block yields two statistics per config: the mean of its
lowest ``k`` per-solve kernel times (CUDA events, kernel only) and
the median per-solve wall time (host clock around ``Solver.solve``,
so changes that lengthen its end-to-end critical path or destroy
chunk-transfer overlap show up in it). The two blocks of an ABBA pair ran
back to
back, so each verdict statistic is the median of the paired percent
deltas — kernel deltas gate at ``--threshold``, wall deltas at the
coarser ``--wall-threshold``.

The workers also exchange compile metrics read from each config's
loaded cufunc and exact cubin link (``@META`` lines: registers, ptxas
spill-store/load byte counts, shared and constant memory, actual launch
geometry, occupancy in blocks/SM, run and chunk counts). The gate
prints an A-vs-B metrics table per backend and fails outright on an
occupancy decrease, a spill increase, or a chunk-count mismatch;
register deltas that leave occupancy and spill unchanged are
reported but not gated (their runtime effect, if any, is caught by
the timing rows).

Four configs run (see ``lorenz_mean_runtime.py``): ``fixed`` and
``adaptive`` as before; ``chunked``, whose small VRAM cap forces a
few run-axis chunks so chunk-path regressions surface in its wall
delta; and ``wave``, sized by the gate to exactly two full waves of
side A's occupancy (``wave <n_runs>`` handshake after startup) so
any occupancy decrease on B forces a third wave and a step kernel
regression instead of hiding behind compute saturation.

Why blocks: the floor of the kernel-time distribution tracks the
compiled kernel's intrinsic cost but wanders a few tenths of a
percent with GPU clock state. The blocks of a pair run seconds
apart, so they sample nearly the same clock state and the wander
cancels inside each pair — and each side pays its process startup,
JIT compile, and grid build once instead of once per sample. Verdict
reliability is judged from the per-pair deltas themselves: when pair
verdicts disagree across the configured threshold boundaries, the row is
marked DISTRUST — rerun, with more ``--pairs`` or on a quieter GPU,
before acting on it.
Constant background load inflates absolute times but cancels out of
the deltas. Both workers hold their device pools concurrently, which
is fine at the default sizes (~0.7 GB each); much larger ``--n-runs``
could change chunking between sides.

Usage::

    python benchmarks/ab_gate.py [--main REF] [--backends numba-cuda,mlir]
        [--block-solves N] [--chunked-solves N] [--pairs P]
        [--min-count K] [--threshold PCT] [--wall-threshold PCT]
        [--n-runs N] [--chunked-runs N] [--chunked-proportion P]
        [--calibrate] [--keep]

``--calibrate`` points B at ``main`` too (A-vs-A); rerun it a few
times to measure this machine's null |delta| for both statistics and
set ``--threshold``/``--wall-threshold`` to two to three times the
worst of it. Exit status is 1 for a regression, 2 for an otherwise
inconclusive DISTRUST result, and 0 for a trusted pass.
"""
import argparse
import importlib.util
import math
import os
import queue
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent / "lorenz_mean_runtime.py"

# label -> (importable spec, CUBIE_CUDA_BACKEND value)
BACKENDS = {
    "numba-cuda": ("numba.cuda", "numba-cuda"),
    "mlir": ("numba_cuda_mlir", "mlir"),
}

META_FIELDS = (
    "regs",
    "spill_store_bytes",
    "spill_load_bytes",
    "shared",
    "dynshared",
    "const",
    "blocks_per_sm",
    "sms",
    "blocksize",
    "runs_per_block",
    "runs",
    "chunks",
)
READY_KEYS = ("fixed", "adaptive", "chunked")


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
    if args.chunked_runs is not None:
        cmd.extend(("--chunked-runs", str(args.chunked_runs)))
    if args.chunked_proportion is not None:
        cmd.extend((
            "--chunked-proportion", str(args.chunked_proportion)
        ))
    if args.n_runs is not None:
        cmd.append(str(args.n_runs))
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(tree), stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True, bufsize=1,
    )
    proc.output_queue = queue.Queue()

    def read_output():
        for line in proc.stdout:
            proc.output_queue.put(line)
        proc.output_queue.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    return proc


def _worker_timeout(proc, side, timeout):
    """Terminate one worker and report its protocol timeout."""
    if proc.poll() is None:
        proc.kill()
    proc.wait()
    raise SystemExit(
        f"worker {side} timed out after {timeout:g} seconds"
    )


def read_worker_line(proc, side, timeout, deadline=None):
    """Read one worker line with a Windows-compatible timeout."""
    wait = timeout
    if deadline is not None:
        wait = deadline - time.monotonic()
        if wait <= 0:
            _worker_timeout(proc, side, timeout)
    try:
        line = proc.output_queue.get(timeout=wait)
    except queue.Empty:
        _worker_timeout(proc, side, timeout)
    if line is None:
        raise SystemExit(
            f"worker {side} exited (code {proc.poll()})"
        )
    return line.strip()


def read_reply(proc, prefix, side, timeout):
    """Read worker stdout lines until one starts with ``prefix``."""
    deadline = time.monotonic() + timeout
    while True:
        line = read_worker_line(proc, side, timeout, deadline)
        if line.startswith("@ERROR"):
            raise SystemExit(f"worker {side}: {line}")
        if line.startswith(prefix):
            return line
        if line.startswith("@"):
            raise SystemExit(
                f"worker {side} sent unexpected protocol line: {line}"
            )


def parse_meta(line):
    """Parse an ``@META`` line into its config key and metric dict."""
    parts = line.split()
    if len(parts) != 2 + len(META_FIELDS) or parts[0] != "@META":
        raise ValueError("malformed @META line")
    meta = {}
    for expected, token in zip(META_FIELDS, parts[2:]):
        name, separator, value = token.partition("=")
        if separator != "=" or name != expected or name in meta:
            raise ValueError("invalid @META schema")
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError("@META values must be integers") from exc
        if parsed < 0:
            raise ValueError("@META values cannot be negative")
        meta[name] = parsed
    for field in (
        "blocks_per_sm", "sms", "blocksize", "runs_per_block",
        "runs", "chunks",
    ):
        if meta[field] <= 0:
            raise ValueError(f"@META {field} must be positive")
    return parts[1], meta


def read_startup(proc, side, timeout):
    """Collect ``@META`` lines until ``@READY``; return both."""
    metas = {}
    deadline = time.monotonic() + timeout
    while True:
        line = read_worker_line(proc, side, timeout, deadline)
        if line.startswith("@ERROR"):
            raise SystemExit(f"worker {side}: {line}")
        if line.startswith("@META "):
            try:
                key, meta = parse_meta(line)
            except ValueError as exc:
                raise SystemExit(f"worker {side}: {exc}") from exc
            if key not in READY_KEYS or key in metas:
                raise SystemExit(
                    f"worker {side} sent duplicate/unknown meta {key!r}"
                )
            metas[key] = meta
        elif line.startswith("@READY "):
            keys = tuple(line.split()[1:])
            if keys != READY_KEYS or tuple(metas) != READY_KEYS:
                raise SystemExit(
                    f"worker {side} startup schema mismatch"
                )
            return list(keys), metas
        elif line.startswith("@"):
            raise SystemExit(
                f"worker {side} sent unexpected protocol line: {line}"
            )


def parse_times(line, key, count):
    """Strictly parse one worker timing reply."""
    parts = line.split()
    expected_length = 4 + 2 * count
    if len(parts) != expected_length:
        raise ValueError(
            f"expected {count} kernel and wall values"
        )
    if parts[:3] != ["@TIMES", key, "kernel"]:
        raise ValueError("invalid @TIMES prefix")
    wall_at = 3 + count
    if parts[wall_at] != "wall":
        raise ValueError("invalid @TIMES wall delimiter")
    try:
        kernel = [float(value) for value in parts[3:wall_at]]
        wall = [float(value) for value in parts[wall_at + 1:]]
    except ValueError as exc:
        raise ValueError("@TIMES values must be numbers") from exc
    if any(
        not math.isfinite(value) or value <= 0
        for value in kernel + wall
    ):
        raise ValueError("@TIMES values must be finite and positive")
    return kernel, wall


def run_block(proc, side, key, count, timeout):
    """Ask one worker for ``count`` solves; return kernel/wall ms."""
    try:
        proc.stdin.write(f"run {key} {count}\n")
        proc.stdin.flush()
    except OSError:
        raise SystemExit(
            f"worker {side} pipe closed (exit {proc.poll()})"
        )
    reply = read_reply(proc, "@TIMES ", side, timeout)
    try:
        return parse_times(reply, key, count)
    except ValueError as exc:
        raise SystemExit(f"worker {side}: {exc}") from exc


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
        proc.wait()


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


def compare_meta(backend, metas, keys):
    """Print the A-vs-B compile-metrics table; return regression."""
    regressed = False
    print(f"\n[{backend}] compile metrics, A -> B")
    print(f"{'config':<10}{'metric':<20}{'A':>12}{'B':>12}  flags")
    for key in keys:
        a, b = metas["A"][key], metas["B"][key]
        flags = []
        # Fewer resident blocks per SM means less latency hiding
        # and, at the wave config's sizes, a whole extra wave.
        if b["blocks_per_sm"] < a["blocks_per_sm"]:
            flags.append("OCCUPANCY REGRESSION")
        # Spilled registers turn register traffic into local-memory
        # traffic; any increase is a compiled-code regression even
        # when the timing rows absorb it.
        if (
            b["spill_store_bytes"] > a["spill_store_bytes"]
            or b["spill_load_bytes"] > a["spill_load_bytes"]
        ):
            flags.append("SPILL REGRESSION")
        # A chunk-count mismatch means the sides ran different
        # transfer schedules, so their timings are not comparable
        # and the memory footprint itself changed.
        if b["chunks"] != a["chunks"]:
            flags.append("CHUNK MISMATCH")
        if b["sms"] != a["sms"] or b["runs"] != a["runs"]:
            flags.append("WORKLOAD MISMATCH")
        regressed = regressed or bool(flags)
        for index, field in enumerate(META_FIELDS):
            suffix = "  ".join(flags) if index == 0 else ""
            print(
                f"{key if index == 0 else '':<10}{field:<20}"
                f"{a[field]:>12}{b[field]:>12}  {suffix}"
            )
    return regressed


def block_statistic(stat, values, k):
    """Reduce one block's per-solve times to its statistic."""
    if stat == "kernel":
        return statistics.fmean(sorted(values)[:k])
    return statistics.median(values)


def classify_deltas(deltas, threshold):
    """Classify paired deltas and whether pair verdicts disagree."""
    if not deltas or any(not math.isfinite(value) for value in deltas):
        raise ValueError("paired deltas must be finite and nonempty")

    def classify(value):
        if value > threshold:
            return "REGRESSION"
        if value < -threshold:
            return "improvement"
        return "ok"

    delta = statistics.median(deltas)
    pair_verdicts = {classify(value) for value in deltas}
    return delta, classify(delta), len(pair_verdicts) > 1


def run_backend(backend, main_tree, b_tree, base, args):
    """Block-interleaved A/B for one backend.

    Returns (timing rows, compile-metrics regression flag).
    """
    grid_dir = base / "grid"
    workers = {}
    metas = {}
    try:
        for side, tree in (("A", main_tree), ("B", b_tree)):
            workers[side] = start_worker(
                tree, backend, base / f"{side}_{backend}", grid_dir,
                args)
        ready = {}
        for side in ("A", "B"):
            ready[side], metas[side] = read_startup(
                workers[side], side, args.worker_timeout)
            print(f"[{backend}] {side} ready", file=sys.stderr)
        if ready["A"] != ready["B"]:
            raise SystemExit(
                f"config keys differ between sides: "
                f"A={ready['A']} B={ready['B']}"
            )

        # Size the wave config from side A's occupancy — "two full
        # waves at current levels" — and impose that count on both
        # sides, so a B-side occupancy drop must add a third wave.
        fixed_a = metas["A"]["fixed"]
        wave_runs = (2 * fixed_a["sms"] * fixed_a["blocks_per_sm"]
                     * fixed_a["runs_per_block"])
        print(
            f"[{backend}] wave config: {wave_runs} runs = 2 waves x "
            f"{fixed_a['sms']} SMs x {fixed_a['blocks_per_sm']} "
            f"blocks/SM x {fixed_a['runs_per_block']} runs/block",
            file=sys.stderr,
        )
        for side in ("A", "B"):
            workers[side].stdin.write(f"wave {wave_runs}\n")
            workers[side].stdin.flush()
        for side in ("A", "B"):
            line = read_reply(
                workers[side], "@META wave", side,
                args.worker_timeout,
            )
            try:
                meta_key, meta = parse_meta(line)
            except ValueError as exc:
                raise SystemExit(f"worker {side}: {exc}") from exc
            if meta_key != "wave":
                raise SystemExit(
                    f"worker {side} answered with meta {meta_key!r}"
                )
            metas[side]["wave"] = meta
        keys = ready["A"] + ["wave"]
        counts = {
            key: (args.chunked_solves if key == "chunked"
                  else args.block_solves)
            for key in keys
        }

        def block(side, store=None):
            for key in keys:
                vals = run_block(
                    workers[side], side, key, counts[key],
                    args.worker_timeout,
                )
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

    meta_regressed = compare_meta(backend, metas, keys)

    rows = []
    thresholds = {
        "kernel": args.threshold,
        "wall": args.wall_threshold,
    }
    for key in keys:
        for stat_idx, stat in enumerate(("kernel", "wall")):
            if stat == "kernel":
                k = min(args.min_count, counts[key])
                floors = {
                    side: [
                        block_statistic(stat, blk[stat_idx], k)
                        for blk in collected[side][key]
                    ]
                    for side in ("A", "B")
                }
            else:
                floors = {
                    side: [
                        statistics.median(blk[stat_idx])
                        for blk in collected[side][key]
                    ]
                    for side in ("A", "B")
                }
            if any(
                not math.isfinite(value) or value <= 0
                for side in floors.values() for value in side
            ):
                raise SystemExit(
                    f"{backend}/{key}/{stat}: block statistics must "
                    "be finite and positive"
                )
            a = statistics.fmean(floors["A"])
            b = statistics.fmean(floors["B"])
            if any(not math.isfinite(value) or value <= 0
                   for value in (a, b)):
                raise SystemExit(
                    f"{backend}/{key}/{stat}: side means must be "
                    "finite and positive"
                )
            # The i-th A and B blocks ran back to back (one ABBA
            # pair), so they share clock state; the median paired
            # delta cancels wander that survives in the side means.
            deltas = [
                100.0 * (bf - af) / af
                for af, bf in zip(floors["A"], floors["B"])
            ]
            threshold = thresholds[stat]
            try:
                delta, verdict, distrust = classify_deltas(
                    deltas, threshold
                )
            except ValueError as exc:
                raise SystemExit(
                    f"{backend}/{key}/{stat}: {exc}"
                ) from exc
            rows.append(
                (backend, key, stat, a, b, delta, verdict, distrust)
            )
    return rows, meta_regressed


def positive_int(value):
    """Argparse type requiring a positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_finite(value):
    """Argparse type requiring a positive finite float."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def nonnegative_finite(value):
    """Argparse type requiring a nonnegative finite float."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            "must be finite and nonnegative"
        )
    return parsed


def finite_proportion(value):
    """Argparse type requiring a finite proportion in (0, 1]."""
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be finite and in (0, 1]")
    return parsed


def build_parser():
    """Build the command-line parser for tests and ``main``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default="origin/main",
                        help="A-side ref (default origin/main; the "
                             "local main branch is often stale).")
    parser.add_argument("--backends", default=None,
                        help="Comma-separated subset of: "
                             + ", ".join(BACKENDS))
    parser.add_argument("--block-solves", type=positive_int, default=25,
                        help="Solves per block per config.")
    parser.add_argument("--chunked-solves", type=positive_int, default=10,
                        help="Solves per block for the chunked "
                             "config (its solves are slow and its "
                             "signal coarse, so fewer suffice).")
    parser.add_argument("--pairs", type=positive_int, default=4,
                        help="A/B block pairs per backend; even "
                             "cancels linear drift, multiples of 4 "
                             "also quadratic.")
    parser.add_argument("--min-count", type=positive_int, default=5,
                        help="Lowest per-solve kernel times averaged "
                             "into each block's kernel statistic.")
    parser.add_argument("--gap", type=nonnegative_finite, nargs=2,
                        default=(1.5, 3.5),
                        metavar=("MIN", "MAX"),
                        help="Idle seconds after each block, drawn "
                             "uniformly per block. The rest lets the "
                             "GPU re-enter its repeatable boost state; "
                             "the jitter stops concurrent periodic GPU "
                             "loads phase-locking with the rhythm.")
    parser.add_argument("--n-runs", type=positive_int, default=None,
                        help="Trajectory count for the fixed and "
                             "adaptive configs (small values "
                             "smoke-test the harness).")
    parser.add_argument("--chunked-runs", type=positive_int,
                        default=None,
                        help="Trajectory count for the chunked "
                             "config (defaults to --n-runs when "
                             "supplied, otherwise 2**22).")
    parser.add_argument("--chunked-proportion", type=finite_proportion,
                        default=None,
                        help="Manual VRAM proportion for each of the "
                             "chunked config's three memory-manager "
                             "instances; the batch chunks once it "
                             "exceeds three times this share of "
                             "total VRAM.")
    parser.add_argument("--threshold", type=positive_finite, default=0.50,
                        help="Kernel-delta regression threshold in "
                             "percent (two to three times the "
                             "calibrated null).")
    parser.add_argument("--wall-threshold", type=positive_finite,
                        default=5.0,
                        help="Wall-delta regression threshold in "
                             "percent; wall time carries host "
                             "scatter, so it is far coarser than "
                             "--threshold.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Point B at main too (A-vs-A null).")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the ephemeral main tree and caches.")
    parser.add_argument(
        "--worker-timeout", type=positive_finite, default=300.0,
        help="Maximum seconds to wait for any worker protocol reply.",
    )
    return parser


def parse_args(argv=None):
    """Parse and validate command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gap[0] > args.gap[1]:
        parser.error("--gap MIN cannot exceed MAX")
    return args


def main(argv=None):
    """Run the A/B gate and return its process exit status."""
    args = parse_args(argv)

    backends = installed_backends()
    if args.backends:
        wanted = [b.strip() for b in args.backends.split(",")]
        if len(set(wanted)) != len(wanted):
            raise SystemExit("--backends cannot contain duplicates")
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
          f"solves/block/side, {args.chunked_solves} for chunked)\n")

    main_tree = add_main_worktree(args.main)
    base = Path(tempfile.mkdtemp(prefix="cubie_abgate_"))
    b_tree = main_tree if args.calibrate else REPO
    regressed = False
    distrusted = False
    try:
        for backend in backends:
            rows, meta_regressed = run_backend(
                backend, main_tree, b_tree, base, args)
            regressed = regressed or meta_regressed
            print()
            for row in rows:
                bk, key, stat, a, b, delta, verdict, distrust = row
                regressed = regressed or verdict == "REGRESSION"
                distrusted = distrusted or distrust
                flag = "  DISTRUST" if distrust else ""
                print(f"{bk:<11}{key:<10}{stat:<8}"
                      f"A {a:9.3f}  B {b:9.3f}  "
                      f"{delta:+6.2f}%  {verdict}{flag}")
    finally:
        if not args.keep:
            remove_worktree(main_tree)
            shutil.rmtree(base, ignore_errors=True)

    status = (
        "REGRESSION" if regressed
        else "INCONCLUSIVE" if distrusted
        else "PASS"
    )
    print(f"\nGATE: {status} "
          f"(kernel threshold {args.threshold:.2f}%, wall threshold "
          f"{args.wall_threshold:.2f}%)")
    if distrusted:
        print("DISTRUST = the per-pair deltas disagree, so that "
              "verdict is unreliable; rerun, with more --pairs or a "
              "quieter GPU, before acting on it.")
    if regressed:
        return 1
    return 2 if distrusted else 0


if __name__ == "__main__":
    raise SystemExit(main())
