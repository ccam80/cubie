#!/usr/bin/env python
"""Mean-runtime benchmark for a large Lorenz ensemble.

Solves the same Lorenz ensemble as the GPUODEBenchmarks cubie runner
(``GPU_ODE_CUBIE/bench_cubie.py``) with the same solver settings, and
times ``Solver.solve`` the same way: one warm-up solve to absorb JIT
compilation, then ``timeit.repeat`` with garbage collection enabled
and one solve per repeat. The headline number is the mean runtime
over the repeats (the cross-package comparison scripts report the
minimum); the minimum and sample standard deviation are printed
alongside for reference.

Usage::

    python benchmarks/lorenz_mean_runtime.py [n_runs] [repeats]

Defaults: ``n_runs = 2**24`` trajectories, ``repeats = 100``.
"""

import sys
import timeit

import numpy as np

import cubie as qb
from cubie.time_logger import default_timelogger

default_timelogger.set_verbosity(None)

n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2**24
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 100

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
    time_logging_level=None,
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
    time_logging_level=None,
)

initials_array, parameter_array = fixed_solver.build_grid(
    initial_values=initial_conditions, parameters=parameters
)


def benchmark(label, solver):
    """Time ``repeats`` solves after a warm-up and print the mean."""

    def run():
        return solver.solve(
            initial_values=initials_array,
            parameters=parameter_array,
            blocksize=64,
            results_type="raw",
            duration=1.0,
        )

    run()  # warm-up (JIT compilation)
    res = timeit.repeat(run, setup="gc.enable()", repeat=repeats, number=1)
    times_ms = np.asarray(res) * 1000.0
    print(
        f"{label}: mean {times_ms.mean():.2f} ms over {repeats} solves of "
        f"{n_runs} trajectories (std {times_ms.std(ddof=1):.2f} ms, "
        f"min {times_ms.min():.2f} ms)"
    )


benchmark("fixed (classical-rk4)", fixed_solver)
benchmark("adaptive (tsit5)", adaptive_solver)
