#!/usr/bin/env python
"""Measure Lorenz kernel runtime for the performance gate."""
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


DISCARDED_SOLVES = 20
Z_THRESHOLD = 3.0
PRECISION = np.float32
INITIAL_CONDITIONS = {"x": 1.0, "y": 0.0, "z": 0.0}


def collect_kernel_time(solver, kernel_ms):
    """Append one solve's kernel time in milliseconds."""
    kernel_ms.append(
        sum(
            event.elapsed_time_ms()
            for event in solver.kernel._cuda_events
            if event.name.startswith("kernel_chunk")
        )
    )


def load_grid(solver, key, n_runs, grid_cache):
    """Load or build the input grid."""
    grid_file = (
        os.path.join(grid_cache, f"grid_{key}_{n_runs}.npz")
        if grid_cache is not None
        else None
    )
    if grid_file is not None and os.path.exists(grid_file):
        with np.load(grid_file) as grid:
            return grid["inits"], grid["params"]

    parameters = {"rho": np.linspace(0.0, 21.0, n_runs)}
    arrays = solver.build_grid(
        initial_values=INITIAL_CONDITIONS,
        parameters=parameters,
    )
    if grid_file is not None:
        os.makedirs(grid_cache, exist_ok=True)
        np.savez(grid_file, inits=arrays[0], params=arrays[1])
    return arrays


def benchmark(key, label, solver, n_runs, repeats, reference, grid_cache):
    """Measure and print one solver configuration."""
    initials_array, parameter_array = load_grid(
        solver, key, n_runs, grid_cache
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

    run()
    kernel_ms.clear()
    timeit.repeat(
        run,
        setup="gc.enable()",
        repeat=DISCARDED_SOLVES + repeats,
        number=1,
    )
    samples = np.asarray(kernel_ms[DISCARDED_SOLVES:])
    mean = float(samples.mean())
    std = float(samples.std(ddof=1))
    print(
        f"{label}: {mean:.3f} ms +/- {std:.3f} ms "
        f"over {repeats} solves of {n_runs} trajectories "
        f"(min {samples.min():.3f} ms)"
    )
    print(f"RESULT {key} {mean:.6f} {std:.6f}")
    if reference is not None:
        ref_mean, ref_std = reference
        denominator = math.sqrt((std**2 + ref_std**2) / repeats)
        if denominator == 0.0:
            z_value = 0.0 if mean == ref_mean else math.inf
        else:
            z_value = (mean - ref_mean) / denominator
        if z_value >= Z_THRESHOLD:
            verdict = "REGRESSION"
        elif z_value <= -Z_THRESHOLD:
            verdict = "improvement"
        else:
            verdict = "no significant difference"
        print(f"{label}: z = {z_value:+.2f} ({verdict})")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_runs", nargs="?", type=int)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--ref-fixed", nargs=2, type=float, metavar=("MEAN", "STD")
    )
    parser.add_argument(
        "--ref-adaptive", nargs=2, type=float, metavar=("MEAN", "STD")
    )
    parser.add_argument("--grid-cache", metavar="DIR")
    parser.add_argument("--no-clear-cache", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if args.n_runs is not None and args.n_runs < 1:
        parser.error("n_runs must be positive")
    return args


def main():
    """Build both solvers and run the benchmark."""
    args = parse_args()
    cache_root = get_cache_root()
    if not args.no_clear_cache:
        shutil.rmtree(cache_root, ignore_errors=True)
    os.makedirs(cache_root, exist_ok=True)
    default_timelogger.set_verbosity("default")

    if args.n_runs is None:
        n_fixed = 2**22
        n_adaptive = 2**24
    else:
        n_fixed = n_adaptive = args.n_runs

    system = qb.create_ODE_system(
        """
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        """,
        states=INITIAL_CONDITIONS,
        parameters={"rho": 21.0},
        constants={"sigma": 10.0, "beta": 8.0 / 3.0},
        name="Lorenz",
        precision=PRECISION,
    )
    fixed_solver = qb.Solver(
        system,
        algorithm="classical-rk4",
        dt=0.001,
        save_every=1.0,
        step_controller="fixed",
        output_types=["state"],
        time_logging_level="default",
    )
    adaptive_solver = qb.Solver(
        system,
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
        "fixed", "fixed (classical-rk4)", fixed_solver, n_fixed,
        args.repeats, args.ref_fixed, args.grid_cache,
    )
    benchmark(
        "adaptive", "adaptive (tsit5)", adaptive_solver, n_adaptive,
        args.repeats, args.ref_adaptive, args.grid_cache,
    )


if __name__ == "__main__":
    main()
