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

Defaults: ``n_runs = 2**22`` trajectories, ``repeats = 100``.

The generated-code and compiled-kernel caches are cleared on every
invocation. The kernel cache is keyed by config hash, which does not
include the package source, so a warm cache can serve kernels
compiled from a different source tree — poisonous for A/B runs where
only ``PYTHONPATH`` differs. Clearing forces each invocation to
compile from the tree under benchmark; the compile cost is absorbed
by the warm-up solve, outside the timed region.
"""

import contextlib
import io
import os
import shutil
import sys
import timeit

import numpy as np

import cubie as qb
from cubie.odesystems.symbolic.odefile import GENERATED_DIR
from cubie.time_logger import default_timelogger

shutil.rmtree(GENERATED_DIR, ignore_errors=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

# CUDA events (kernel / h2d / d2h per chunk) are recorded only when
# the time logger is armed. Solver construction resets the global
# verbosity from its time_logging_level argument, so arming happens
# through that argument below; solve's printed summaries are
# swallowed by the stdout redirect inside the timed callable.
default_timelogger.set_verbosity("default")

n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2**22
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 100
discarded_solves = 20

precision = np.float32

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

parameters = {"rho": np.linspace(0.0, 21.0, n_runs)}
initial_conditions = {"x": 1.0, "y": 0.0, "z": 0.0}

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

initials_array, parameter_array = fixed_solver.build_grid(
    initial_values=initial_conditions, parameters=parameters
)


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


def benchmark(label, solver):
    """Run ``repeats`` solves after a warm-up; print kernel runtime."""
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
    print(
        f"{label}: kernel mean {kernel_arr.mean():.2f} ms over {repeats} "
        f"solves of {n_runs} trajectories "
        f"(std {kernel_arr.std(ddof=1):.2f} ms, "
        f"min {kernel_arr.min():.2f} ms)"
    )


benchmark("fixed (classical-rk4)", fixed_solver)
benchmark("adaptive (tsit5)", adaptive_solver)
