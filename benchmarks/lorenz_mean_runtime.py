#!/usr/bin/env python
"""Mean-runtime benchmark for a large Lorenz ensemble.

Solves the same Lorenz ensemble as the GPUODEBenchmarks cubie runner
(``GPU_ODE_CUBIE/bench_cubie.py``) with the same solver settings and
drive pattern: one warm-up solve to absorb JIT compilation, then
``timeit.repeat`` with garbage collection enabled and one solve per
repeat. The first 20 post-warm-up solves are discarded — the GPU has
not reached steady state and they run slow — and the statistic covers
the following ``repeats`` solves.

The reported statistic is the **mean of the lowest ``min_count``
per-solve kernel times** — the per-chunk ``kernel_chunk_i`` CUDA events
recorded on the GPU timeline by ``BatchSolverKernel``. The lowest
solves are those that ran at full boost clock with no on-GPU
contention, so they track the compiled kernel's intrinsic cost and
drift far less between invocations than the mean (which the upper tail
pulls around). Wall-clock and transfer times are not reported: they
carry host and d2h scatter irrelevant to the compiled kernel.

Usage::

    python benchmarks/lorenz_mean_runtime.py [n_runs]
        [--repeats R] [--min-count K] [--grid-cache DIR]
        [--no-clear-cache] [--worker]

Defaults: ``2**22`` trajectories for the fixed config and ``2**24``
for the adaptive config (the adaptive kernel is fast enough at
``2**22`` that launch effects blur small deltas); ``repeats = 100``;
``min_count = 5``. An explicit ``n_runs`` argument applies to both
configs. Each config prints a human-readable line and a parseable
``RESULT <key> <ms>`` line.

``--worker`` turns the process into a persistent solve server for
``ab_gate.py``: after the solvers are built, the grids loaded, and one
JIT warm-up solve run per config, it prints ``@READY <config>...``
and then answers ``run <config> <n>`` lines on stdin with
``@TIMES <config> <ms>...`` (per-solve kernel totals) until ``quit``
or EOF.

The generated-code and compiled-kernel caches are cleared on every
invocation unless ``--no-clear-cache`` is given; the recompile lands
in the warm-up solve, outside the timed region. Both cache layers key
on a content hash of the imported cubie source, so cache reuse is
safe even across source trees.
"""

import argparse
import contextlib
import io
import os
import shutil
import sys
import timeit

import numpy as np

import cubie as qb
from cubie.cache_root import get_cache_root
from cubie.time_logger import default_timelogger

discarded_solves = 20
min_count = 5
precision = np.float32
initial_conditions = {"x": 1.0, "y": 0.0, "z": 0.0}


def collect_kernel_time(solver, kernel_ms):
    """Append one solve's kernel CUDA-event total (ms) to ``kernel_ms``.

    Sums the per-chunk ``kernel_chunk_i`` GPU-timeline times, which
    exclude host<->device transfer traffic.
    """
    events = solver.kernel._cuda_events
    kernel_ms.append(
        sum(
            event.elapsed_time_ms()
            for event in events
            if event.name.startswith("kernel_chunk")
        )
    )


def resolve_n_runs(n_runs):
    """Return (fixed, adaptive) trajectory counts."""
    if n_runs is not None:
        return n_runs, n_runs
    return 2**22, 2**24


def build_solvers(n_fixed, n_adaptive):
    """Build the Lorenz system and both benchmark solvers.

    Returns a dict of ``key -> (label, solver, n_runs)`` in the order
    the configs run.
    """
    lorenz_system = qb.create_ODE_system(
        """
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        """,
        states={"x": 1.0, "y": 0.0, "z": 0.0},
        parameters={"rho": 21.0},
        constants={"sigma": 10.0, "beta": 8.0 / 3.0},
        name="Lorenz",
        precision=precision,
    )

    fixed_solver = qb.Solver(
        lorenz_system,
        algorithm="classical-rk4",
        dt=0.001,
        save_every=1.0,
        step_controller="fixed",
        output_types=["state"],
        time_logging_level="default",
    )

    adaptive_solver = qb.Solver(
        lorenz_system,
        algorithm="tsit5",
        atol=1e-08,
        rtol=1e-08,
        save_every=1.0,
        dt_min=1e-12,
        dt_max=1e3,
        step_controller="pid",
        kp=6 / 5,
        kd=0.0,
        ki=0.0,
        max_gain=5.0,
        min_gain=0.1,
        output_types=["state"],
        time_logging_level="default",
    )

    return {
        "fixed": ("fixed (classical-rk4)", fixed_solver, n_fixed),
        "adaptive": ("adaptive (tsit5)", adaptive_solver, n_adaptive),
    }


def load_grid(solver, key, n_runs, grid_cache):
    """Load the input grid from cache, or build and save it.

    The save is atomic (write then ``os.replace``) so concurrent
    workers sharing a cache directory cannot read a partial file.
    """
    gfile = (
        os.path.join(grid_cache, f"grid_{key}.npz")
        if grid_cache is not None
        else None
    )
    if gfile is not None and os.path.exists(gfile):
        with np.load(gfile) as grid:
            return grid["inits"], grid["params"]
    parameters = {"rho": np.linspace(0.0, 21.0, n_runs)}
    inits, params = solver.build_grid(
        initial_values=initial_conditions, parameters=parameters
    )
    if gfile is not None:
        os.makedirs(grid_cache, exist_ok=True)
        tmp = f"{gfile}.{os.getpid()}.npz"
        np.savez(tmp, inits=inits, params=params)
        try:
            os.replace(tmp, gfile)
        except OSError:
            # Another worker raced us to the same grid (Windows
            # locks open files); ours is in memory, drop the temp.
            with contextlib.suppress(OSError):
                os.remove(tmp)
    return inits, params


def solve_once(solver, inits, params, kernel_ms):
    """Run one solve and record its kernel time in ``kernel_ms``."""
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve(
            initial_values=inits,
            parameters=params,
            blocksize=64,
            duration=1.0,
        )
    collect_kernel_time(solver, kernel_ms)


def benchmark(key, label, solver, n_runs, repeats, k, grid_cache=None):
    """Run ``repeats`` solves after a warm-up; print the mean-of-mins.

    Prints the mean of the ``k`` lowest per-solve kernel times (the
    module docstring explains the choice of statistic), then a
    parseable ``RESULT <key> <ms>`` line.
    """
    inits, params = load_grid(solver, key, n_runs, grid_cache)
    kernel_ms = []

    def run():
        solve_once(solver, inits, params, kernel_ms)

    run()  # warm-up (JIT compilation)
    kernel_ms.clear()
    timeit.repeat(
        run,
        setup="gc.enable()",
        repeat=discarded_solves + repeats,
        number=1,
    )
    kernel_arr = np.sort(np.asarray(kernel_ms[discarded_solves:]))
    stat = kernel_arr[:k].mean()
    print(
        f"{label}: {stat:.3f} ms (mean of {k} lowest kernel times "
        f"over {repeats} solves of {n_runs} trajectories)"
    )
    print(f"RESULT {key} {stat:.4f}")


def worker_loop(solvers, grid_cache):
    """Serve solve blocks over stdin/stdout for ``ab_gate.py``.

    Prints ``@READY <config>...`` once every config has its grid and
    a JIT warm-up solve behind it, then answers ``run <config> <n>``
    with ``@TIMES <config> <ms>...`` until ``quit`` or EOF.
    """
    ready = {}
    for key, (label, solver, n_runs) in solvers.items():
        inits, params = load_grid(solver, key, n_runs, grid_cache)
        solve_once(solver, inits, params, [])
        ready[key] = (solver, inits, params)
    print("@READY " + " ".join(ready), flush=True)
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "quit":
            break
        if parts[0] != "run":
            continue
        key, count = parts[1], int(parts[2])
        solver, inits, params = ready[key]
        kernel_ms = []
        for _ in range(count):
            solve_once(solver, inits, params, kernel_ms)
        print(
            "@TIMES " + key + " "
            + " ".join(f"{v:.4f}" for v in kernel_ms),
            flush=True,
        )


def main():
    """Parse arguments and run the benchmark or the worker mode."""
    args = _parse_args()

    cache_root = get_cache_root()
    if not args.no_clear_cache:
        shutil.rmtree(cache_root, ignore_errors=True)
    os.makedirs(cache_root, exist_ok=True)

    # Per-chunk CUDA events are recorded only while the time logger
    # is armed; Solver construction re-arms it from each solver's
    # time_logging_level argument.
    default_timelogger.set_verbosity("default")

    n_fixed, n_adaptive = resolve_n_runs(args.n_runs)
    solvers = build_solvers(n_fixed, n_adaptive)

    if args.worker:
        worker_loop(solvers, args.grid_cache)
        return

    for key, (label, solver, n_runs) in solvers.items():
        benchmark(
            key,
            label,
            solver,
            n_runs,
            args.repeats,
            args.min_count,
            args.grid_cache,
        )


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Kernel-runtime benchmark (Lorenz ensemble). Prints "
        "the mean of the lowest kernel times per config; drive A/B "
        "comparisons with ab_gate.py."
    )
    parser.add_argument("n_runs", nargs="?", type=int, default=None)
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Solves to time per config after the discard window.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=min_count,
        help="Number of lowest per-solve kernel times to average.",
    )
    parser.add_argument(
        "--grid-cache",
        default=None,
        metavar="DIR",
        help="Cache/load the input grid here to skip rebuilding it.",
    )
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="Reuse the compile cache instead of clearing it (for a "
        "fixed source tree across repeated invocations).",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Serve solve blocks over stdin/stdout for ab_gate.py.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
