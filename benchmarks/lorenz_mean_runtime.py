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
        [--no-clear-cache]

Defaults: ``2**22`` trajectories for the fixed config and ``2**24``
for the adaptive config (the adaptive kernel is fast enough at
``2**22`` that launch effects blur small deltas); ``repeats = 100``;
``min_count = 5``. An explicit ``n_runs`` argument applies to both
configs. Each config prints a human-readable line and a parseable
``RESULT <key> <ms>`` line; ``ab_gate.py`` drives the A/B comparison
from those.

The generated-code and compiled-kernel caches are cleared on every
invocation unless ``--no-clear-cache`` is given; the recompile lands
in the warm-up solve, outside the timed region. Both cache layers key
on a content hash of the imported cubie source, so cache reuse is
safe even across source trees — ``ab_gate.py`` passes
``--no-clear-cache`` to compile only once per side.
"""

import argparse
import contextlib
import io
import os
import shutil
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


def benchmark(key, label, solver, n_runs, repeats, k, grid_cache=None):
    """Run ``repeats`` solves after a warm-up; print the mean-of-mins.

    Prints the mean of the ``k`` lowest per-solve kernel times (the
    module docstring explains the choice of statistic), then a
    parseable ``RESULT <key> <ms>`` line for ``ab_gate.py``.

    When ``grid_cache`` is a directory, the input grid (identical
    every invocation) is loaded from it if present, else built once
    and saved there.
    """
    gfile = (
        os.path.join(grid_cache, f"grid_{key}.npz")
        if grid_cache is not None
        else None
    )
    if gfile is not None and os.path.exists(gfile):
        grid = np.load(gfile)
        initials_array, parameter_array = grid["inits"], grid["params"]
    else:
        parameters = {"rho": np.linspace(0.0, 21.0, n_runs)}
        initials_array, parameter_array = solver.build_grid(
            initial_values=initial_conditions, parameters=parameters
        )
        if gfile is not None:
            os.makedirs(grid_cache, exist_ok=True)
            np.savez(
                gfile, inits=initials_array, params=parameter_array
            )
    kernel_ms = []

    def run():
        with contextlib.redirect_stdout(io.StringIO()):
            solution = solver.solve(
                initial_values=initials_array,
                parameters=parameter_array,
                blocksize=64,
                results_type="raw",
                duration=1.0,
            )
        collect_kernel_time(solver, kernel_ms)
        return solution

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


def main():
    """Parse arguments, build the solvers, and run both configs."""
    args = _parse_args()

    cache_root = get_cache_root()
    if not args.no_clear_cache:
        shutil.rmtree(cache_root, ignore_errors=True)
    os.makedirs(cache_root, exist_ok=True)

    # Per-chunk CUDA events are recorded only while the time logger
    # is armed; Solver construction re-arms it from each solver's
    # time_logging_level argument.
    default_timelogger.set_verbosity("default")

    repeats = args.repeats
    if args.n_runs is not None:
        n_fixed = n_adaptive = args.n_runs
    else:
        n_fixed = 2**22
        n_adaptive = 2**24

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

    benchmark(
        "fixed",
        "fixed (classical-rk4)",
        fixed_solver,
        n_fixed,
        repeats,
        args.min_count,
        args.grid_cache,
    )
    benchmark(
        "adaptive",
        "adaptive (tsit5)",
        adaptive_solver,
        n_adaptive,
        repeats,
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
