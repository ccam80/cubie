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

    python benchmarks/lorenz_mean_runtime.py [n_runs] [repeats]
        [--min-count K]

Defaults: ``2**22`` trajectories for the fixed config and ``2**24``
for the adaptive config (the adaptive kernel is fast enough at
``2**22`` that launch effects blur small deltas); ``repeats = 100``;
``min_count = 5``. An explicit ``n_runs`` argument applies to both
configs. Each config prints a human-readable line and a parseable
``RESULT <key> <ms>`` line; ``ab_gate.py`` drives the A/B comparison
(main vs the active worktree, per backend) from those.

The generated-code and compiled-kernel caches are cleared on every
invocation. The kernel cache is keyed by config hash, which does not
include the package source, so a warm cache can serve kernels
compiled from a different source tree — poisonous for A/B runs where
only ``PYTHONPATH`` differs. Clearing forces each invocation to
compile from the tree under benchmark; the compile cost is absorbed
by the warm-up solve, outside the timed region.
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

    Reads the kernel's per-chunk ``kernel_chunk_i`` event objects
    after the solve has synchronised the stream; the GPU-timeline
    elapsed times exclude host<->device transfer traffic.
    """
    events = solver.kernel._cuda_events
    kernel_ms.append(
        sum(
            event.elapsed_time_ms()
            for event in events
            if event.name.startswith("kernel_chunk")
        )
    )


def benchmark(key, label, solver, n_runs, repeats, k):
    """Run ``repeats`` solves after a warm-up; print the mean-of-mins.

    The reported statistic is the mean of the ``k`` lowest per-solve
    kernel times. The lowest solves are the ones that ran at full boost
    clock with no on-GPU contention, so they track the compiled
    kernel's intrinsic cost and drift far less between invocations than
    the mean. A machine-parseable ``RESULT <key> <ms>`` line follows the
    human-readable line for the A/B driver (``ab_gate.py``) to consume.
    """
    parameters = {"rho": np.linspace(0.0, 21.0, n_runs)}
    initials_array, parameter_array = solver.build_grid(
        initial_values=initial_conditions, parameters=parameters
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
    cache_root = get_cache_root()
    shutil.rmtree(cache_root, ignore_errors=True)
    os.makedirs(cache_root, exist_ok=True)

    # CUDA events (kernel / h2d / d2h per chunk) are recorded only
    # when the time logger is armed. Solver construction resets the
    # global verbosity from its time_logging_level argument, so
    # arming happens through that argument below; solve's printed
    # summaries are swallowed by the stdout redirect inside the
    # timed callable.
    default_timelogger.set_verbosity("default")

    parser = argparse.ArgumentParser(
        description="Kernel-runtime benchmark (Lorenz ensemble). Prints "
        "the mean of the lowest kernel times per config; drive A/B "
        "comparisons with ab_gate.py."
    )
    parser.add_argument("n_runs", nargs="?", type=int, default=None)
    parser.add_argument("repeats", nargs="?", type=int, default=100)
    parser.add_argument(
        "--min-count",
        type=int,
        default=min_count,
        help="Number of lowest per-solve kernel times to average.",
    )
    args = parser.parse_args()

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
        args.repeats,
        args.min_count,
    )
    benchmark(
        "adaptive",
        "adaptive (tsit5)",
        adaptive_solver,
        n_adaptive,
        args.repeats,
        args.min_count,
    )


if __name__ == "__main__":
    main()
