#!/usr/bin/env python
"""Mean-runtime benchmark for a large Lorenz ensemble.

Solves the same Lorenz ensemble as the GPUODEBenchmarks cubie runner
(``GPU_ODE_CUBIE/bench_cubie.py``) with the same solver settings and
drive pattern: one warm-up solve to absorb JIT compilation, then
``timeit.repeat`` with garbage collection enabled and one solve per
repeat. The first 20 post-warm-up solves are discarded — the GPU has
not reached steady state and they run slow, inflating the standard
deviation — and statistics cover the following ``repeats`` solves.
The output is the **kernel runtime only** — the per-chunk
``kernel_chunk_i`` CUDA events recorded on the GPU timeline by
``BatchSolverKernel`` — as mean, sample standard deviation, and
minimum over the repeats. Wall-clock and transfer times are not
reported: per-process d2h/host-memory scatter makes them useless for
A/B comparison, and kernel runtime is the gate metric.

Usage::

    python benchmarks/lorenz_mean_runtime.py [n_runs] [repeats]
        [--ref-fixed MEAN STD] [--ref-adaptive MEAN STD]

Defaults: ``2**22`` trajectories for the fixed config and ``2**24``
for the adaptive config (the adaptive kernel is fast enough at
``2**22`` that launch effects blur small deltas); ``repeats = 100``.
An explicit ``n_runs`` argument applies to both configs.

For the B side of an A/B gate, pass the A side's printed mean and
std per config via ``--ref-fixed`` / ``--ref-adaptive``. The script
then also prints a Welch two-sample z statistic,
``z = (mean - ref_mean) / sqrt(std**2/n + ref_std**2/n)``, and a
verdict: ``|z| >= 3`` flags the means as different (a regression
when positive). The textbook 95% bound is 1.96, but solve samples
are mildly autocorrelated (thermal state), so the gate uses 3.

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
import math
import os
import shutil
import timeit

import numpy as np

import cubie as qb
from cubie.cache_root import get_cache_root
from cubie.time_logger import default_timelogger

discarded_solves = 20
z_threshold = 3.0
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


def benchmark(label, solver, n_runs, repeats, reference):
    """Run ``repeats`` solves after a warm-up; print kernel runtime.

    When ``reference`` holds the A side's (mean, std), a Welch
    two-sample z statistic against it is printed with a verdict.
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
    kernel_arr = np.asarray(kernel_ms[discarded_solves:])
    mean = kernel_arr.mean()
    std = kernel_arr.std(ddof=1)
    print(
        f"{label}: kernel mean {mean:.2f} ms over {repeats} "
        f"solves of {n_runs} trajectories "
        f"(std {std:.2f} ms, min {kernel_arr.min():.2f} ms)"
    )
    if reference is not None:
        ref_mean, ref_std = reference
        z = (mean - ref_mean) / math.sqrt(
            (std**2 + ref_std**2) / repeats
        )
        if z >= z_threshold:
            verdict = "means differ - REGRESSION"
        elif z <= -z_threshold:
            verdict = "means differ - improvement"
        else:
            verdict = "no significant difference"
        print(f"{label}: z = {z:+.2f} vs reference ({verdict})")


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
        description="Kernel-runtime A/B gate benchmark (Lorenz ensemble)."
    )
    parser.add_argument("n_runs", nargs="?", type=int, default=None)
    parser.add_argument("repeats", nargs="?", type=int, default=100)
    parser.add_argument(
        "--ref-fixed",
        nargs=2,
        type=float,
        metavar=("MEAN", "STD"),
        default=None,
        help="A-side kernel mean and std (ms) for the fixed config.",
    )
    parser.add_argument(
        "--ref-adaptive",
        nargs=2,
        type=float,
        metavar=("MEAN", "STD"),
        default=None,
        help="A-side kernel mean and std (ms) for the adaptive config.",
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
        "fixed (classical-rk4)",
        fixed_solver,
        n_fixed,
        args.repeats,
        args.ref_fixed,
    )
    benchmark(
        "adaptive (tsit5)",
        adaptive_solver,
        n_adaptive,
        args.repeats,
        args.ref_adaptive,
    )


if __name__ == "__main__":
    main()
