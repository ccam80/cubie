#!/usr/bin/env python
"""Overnight sweep: buffer memory locations across sizes and algorithms.

Measures kernel runtime for placement variants of every registered
buffer group (loop arrays, algorithm work buffers, nonlinear/linear
solver vectors) over a grid of system sizes (states, parameters,
drivers, observables, baked constants) and algorithm families. The
results feed the size-based location heuristics tracked by issue
#329.

Each configuration runs in a fresh subprocess so the kernel is
compiled for exactly that placement. Results append to a JSONL file;
re-running skips configurations already recorded, so the sweep is
resumable.

Usage::

    python benchmarks/memory_location_sweep.py            # full sweep
    python benchmarks/memory_location_sweep.py --phase core
    python benchmarks/memory_location_sweep.py --single '<config json>'

The single-config mode is internal: the driver invokes it in a
subprocess per configuration.

Timing follows benchmarks/lorenz_mean_runtime.py: kernel-only CUDA
event times (per-chunk ``kernel_chunk_i`` events), one warm-up solve
to absorb JIT compilation, then ``repeats`` timed solves.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "memory_location_sweep_results.jsonl",
)
REPEATS = 5
NRUNS = 32768
BLOCKSIZE = 256
DURATION = 0.05
DT = 1e-3

# Buffer-location kwargs per placement group. Loop groups apply to
# every algorithm; algorithm and solver groups are keyed by family.
LOOP_GROUPS = {
    "state": ["state_location", "proposed_state_location"],
    "params": ["parameters_location"],
    "drivers": ["drivers_location", "proposed_drivers_location"],
    "obs": ["observables_location", "proposed_observables_location"],
}
ALGO_GROUPS = {
    "euler": [],
    "tsit5": ["stage_rhs_location", "stage_accumulator_location"],
    "dirk": [
        "stage_increment_location",
        "stage_base_location",
        "accumulator_location",
        "stage_rhs_location",
    ],
    "firk": [
        "stage_increment_location",
        "stage_driver_stack_location",
        "stage_state_location",
    ],
    "rosenbrock": [
        "stage_rhs_location",
        "stage_store_location",
        "cached_auxiliaries_location",
    ],
    "backwards_euler": ["increment_cache_location"],
    "crank_nicolson": ["dxdt_location"],
}
SOLVER_GROUPS = {
    "euler": [],
    "tsit5": [],
    "dirk": [
        "delta_location",
        "residual_location",
        "residual_temp_location",
        "stage_base_bt_location",
        "preconditioned_vec_location",
        "temp_location",
    ],
    "firk": [
        "delta_location",
        "residual_location",
        "residual_temp_location",
        "stage_base_bt_location",
        "preconditioned_vec_location",
        "temp_location",
    ],
    "rosenbrock": ["preconditioned_vec_location", "temp_location"],
    "backwards_euler": [
        "delta_location",
        "residual_location",
        "residual_temp_location",
        "stage_base_bt_location",
        "preconditioned_vec_location",
        "temp_location",
    ],
    "crank_nicolson": [
        "delta_location",
        "residual_location",
        "residual_temp_location",
        "stage_base_bt_location",
        "preconditioned_vec_location",
        "temp_location",
    ],
}
IMPLICIT = {"dirk", "firk", "rosenbrock", "backwards_euler",
            "crank_nicolson"}


def variant_kwargs(algorithm, variant, n_drivers, n_observables):
    """Return the ``*_location`` kwargs for a named placement variant.

    Returns None when the variant does not apply to this
    configuration (e.g. solver placement on an explicit algorithm).
    """
    groups = []
    if variant == "local":
        return {}
    elif variant == "state_shared":
        groups = [LOOP_GROUPS["state"]]
    elif variant == "params_shared":
        groups = [LOOP_GROUPS["params"]]
    elif variant == "drivers_shared":
        if n_drivers == 0:
            return None
        groups = [LOOP_GROUPS["drivers"]]
    elif variant == "obs_shared":
        if n_observables == 0:
            return None
        groups = [LOOP_GROUPS["obs"]]
    elif variant == "state_params_shared":
        groups = [LOOP_GROUPS["state"], LOOP_GROUPS["params"]]
    elif variant == "algo_shared":
        if not ALGO_GROUPS[algorithm]:
            return None
        groups = [ALGO_GROUPS[algorithm]]
    elif variant == "solver_shared":
        if algorithm not in IMPLICIT:
            return None
        groups = [SOLVER_GROUPS[algorithm]]
    elif variant == "all_shared":
        groups = [
            LOOP_GROUPS["state"],
            LOOP_GROUPS["params"],
            ALGO_GROUPS[algorithm],
            SOLVER_GROUPS[algorithm],
        ]
        if n_drivers:
            groups.append(LOOP_GROUPS["drivers"])
        if n_observables:
            groups.append(LOOP_GROUPS["obs"])
    else:
        raise ValueError(f"unknown variant {variant}")
    return {key: "shared" for group in groups for key in group}


def make_system(cfg):
    """Build the synthetic benchmark system for a configuration.

    Nonlinear nearest-neighbour chain: every state couples to its
    ring neighbours with one nonlinear term, parameters cycle over
    the equations, drivers force the first equations additively, and
    observables are pairwise products. Baked constants are unique
    literals so codegen cannot fold them together.
    """
    from cubie import create_ODE_system

    n = cfg["n_states"]
    n_params = cfg["n_params"]
    n_drivers = cfg["n_drivers"]
    n_obs = cfg["n_observables"]
    consts_per_eq = cfg["consts_per_eq"]
    precision = np.float32 if cfg["precision"] == "f32" else np.float64

    rng = np.random.default_rng(1234)
    eqs = []
    constants = {}
    for i in range(n):
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
        terms = [f"0.2*x{im1} + 0.3*x{ip1}"]
        for c in range(consts_per_eq):
            cname = f"k{i}_{c}"
            constants[cname] = float(rng.uniform(0.5, 5.0))
            if c == 0:
                terms.append(f"-{cname}*x{i}")
            else:
                terms.append(f"+ {cname}*x{ip1}/(1.0 + x{i}*x{i})")
        if n_params:
            pj = i % n_params
            terms.append(f"+ 0.05*p{pj}*x{i}*x{ip1}")
        if n_drivers:
            dj = i % n_drivers
            terms.append(f"+ 0.1*d{dj}")
        eqs.append(f"dx{i} = " + " ".join(terms))
    for j in range(n_obs):
        a = j % n
        b = (j * 7 + 1) % n
        eqs.append(f"o{j} = x{a}*x{b}")

    system = create_ODE_system(
        dxdt=eqs,
        states={f"x{i}": 0.5 for i in range(n)},
        parameters={f"p{j}": 1.0 for j in range(n_params)},
        constants=constants,
        drivers=[f"d{j}" for j in range(n_drivers)] or None,
        observables=[f"o{j}" for j in range(n_obs)] or None,
        precision=precision,
        name=(
            f"sweep_{n}s_{n_params}p_{n_drivers}d_{n_obs}o_"
            f"{consts_per_eq}c_{cfg['precision']}"
        ),
    )
    return system, precision


def solver_stats(solver):
    """Return launch-geometry and register stats for a built solver."""
    bsk = solver.kernel
    pad = 4 if bsk.shared_memory_needs_padding else 0
    bytes_per_run = bsk.shared_memory_bytes + pad
    smem = int(bytes_per_run * min(NRUNS, BLOCKSIZE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eff_blocksize, smem = bsk.limit_blocksize(
            BLOCKSIZE, smem, bytes_per_run, NRUNS
        )
    regs = lmem = None
    try:
        kernel = bsk.kernel
        regs = list(kernel.get_regs_per_thread().values())[0]
        lmem = list(kernel.get_local_mem_per_thread().values())[0]
    except Exception:
        pass
    return dict(
        regs_per_thread=regs,
        local_mem_per_thread=lmem,
        shared_bytes_per_run=int(bytes_per_run),
        eff_blocksize=int(eff_blocksize),
        dynamic_shared_per_block=int(smem),
    )


def timed_solve(solver, y0, p, solve_kwargs):
    """Run one solve and return its kernel CUDA-event total (ms)."""
    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(y0, p, **solve_kwargs)
    events = solver.kernel._cuda_events
    ms = sum(
        e.elapsed_time_ms()
        for e in events
        if e.name.startswith("kernel_chunk")
    )
    return ms, result


def run_single(cfg):
    """Run one paired baseline/variant configuration; print JSON.

    The benchmark GPU also drives a desktop, and background load can
    inflate whole windows of measurements uniformly, which no
    per-config variance filter can detect. Timing is therefore
    paired: the all-local baseline solver and the variant solver
    alternate solves within the same process, and the reported ratio
    compares minima over interleaved samples, so slow drifts in
    outside load cancel instead of masquerading as placement
    effects.
    """
    from cubie.batchsolving.solver import Solver
    from cubie.time_logger import default_timelogger

    default_timelogger.set_verbosity("default")
    system, precision = make_system(cfg)

    loc_kwargs = variant_kwargs(
        cfg["algorithm"], cfg["variant"], cfg["n_drivers"],
        cfg["n_observables"],
    )
    if loc_kwargs is None or cfg["variant"] == "local":
        print(json.dumps({"skip": True}))
        return

    # auto_memory off on both sides: the sweep measures pure
    # placements, and the heuristics under calibration must never
    # leak into their own baseline.
    base_kwargs = dict(
        dt=DT,
        step_controller="fixed",
        save_every=DURATION,
        output_types=["state"],
        time_logging_level="default",
        algorithm=cfg["algorithm"],
        auto_memory=False,
    )
    if cfg["algorithm"] in IMPLICIT:
        base_kwargs["krylov_max_iters"] = 100

    n = cfg["n_states"]
    y0 = {
        f"x{i}": np.full(NRUNS, 0.5, dtype=precision) for i in range(n)
    }
    p = {"p0": np.full(NRUNS, 1.0, dtype=precision)}
    solve_kwargs = dict(
        duration=DURATION, grid_type="verbatim", blocksize=BLOCKSIZE,
    )
    if cfg["n_drivers"]:
        t_samples = np.linspace(
            0.0, DURATION, 16, dtype=precision
        )
        drivers = {
            f"d{j}": np.sin(
                2 * np.pi * (j + 1) * t_samples / DURATION
            ).astype(precision)
            for j in range(cfg["n_drivers"])
        }
        drivers["dt"] = precision(DURATION / 15)
        drivers["wrap"] = False
        solve_kwargs["drivers"] = drivers

    caught = []
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        solver_local = Solver(system, **base_kwargs)
        solver_variant = Solver(system, **base_kwargs, **loc_kwargs)
        # Warm-up both (JIT compilation / cache load) and capture
        # outputs for a placement-invariance check: buffer location
        # must never change results.
        _, result_local = timed_solve(
            solver_local, y0, p, solve_kwargs
        )
        _, result_variant = timed_solve(
            solver_variant, y0, p, solve_kwargs
        )
        caught = sorted({str(w.message)[:80] for w in wlist})
    compile_s = time.perf_counter() - t0

    out_local = np.asarray(result_local.time_domain_array)
    out_variant = np.asarray(result_variant.time_domain_array)
    if out_local.shape == out_variant.shape:
        diff = float(
            np.nanmax(np.abs(out_local - out_variant))
        )
    else:
        diff = float("nan")

    local_ms = []
    variant_ms = []
    while True:
        for _ in range(REPEATS):
            ms, _ = timed_solve(solver_local, y0, p, solve_kwargs)
            local_ms.append(ms)
            ms, _ = timed_solve(solver_variant, y0, p, solve_kwargs)
            variant_ms.append(ms)
        pair_ratios = np.asarray(variant_ms) / np.asarray(local_ms)
        if len(local_ms) >= 3 * REPEATS:
            break
        if pair_ratios.std(ddof=1) <= 0.05 * pair_ratios.mean():
            break

    local_arr = np.asarray(local_ms)
    variant_arr = np.asarray(variant_ms)
    record = dict(cfg)
    record.update(
        ratio_min=float(variant_arr.min() / local_arr.min()),
        ratio_median_pairwise=float(
            np.median(variant_arr / local_arr)
        ),
        local_ms_min=float(local_arr.min()),
        variant_ms_min=float(variant_arr.min()),
        local_ms=list(map(float, local_arr)),
        variant_ms=list(map(float, variant_arr)),
        compile_s=round(compile_s, 2),
        local_stats=solver_stats(solver_local),
        variant_stats=solver_stats(solver_variant),
        out_maxdiff=diff,
        warnings=caught,
    )
    print(json.dumps(record))


def core_matrix():
    """Yield the core sweep: sizes x algorithms x placement variants."""
    variants = [
        "local",
        "state_shared",
        "params_shared",
        "state_params_shared",
        "algo_shared",
        "solver_shared",
        "all_shared",
    ]
    # rosenbrock is excluded: auxiliary-cache planning
    # (_search_group_combinations) does not terminate for chain
    # systems at n_states >= 16, so its kernels cannot be built.
    algorithms = [
        "euler",
        "tsit5",
        "dirk",
        "firk",
        "backwards_euler",
    ]
    for n in [4, 16, 64, 128, 256]:
        for algorithm in algorithms:
            if algorithm == "firk" and n > 64:
                continue
            for variant in variants:
                yield dict(
                    phase="core",
                    algorithm=algorithm,
                    n_states=n,
                    n_params=2,
                    n_drivers=0,
                    n_observables=0,
                    consts_per_eq=3,
                    precision="f32",
                    variant=variant,
                )


def dims_matrix():
    """Yield the size-dimension sweep on two representative algorithms."""
    for algorithm in ["tsit5", "backwards_euler"]:
        for n in [16, 64, 128]:
            for n_params in [16, 64, 128]:
                for variant in ["local", "params_shared"]:
                    yield dict(
                        phase="dims",
                        algorithm=algorithm,
                        n_states=n,
                        n_params=n_params,
                        n_drivers=0,
                        n_observables=0,
                        consts_per_eq=3,
                        precision="f32",
                        variant=variant,
                    )
            for n_drivers in [2, 8]:
                for variant in ["local", "drivers_shared"]:
                    yield dict(
                        phase="dims",
                        algorithm=algorithm,
                        n_states=n,
                        n_params=2,
                        n_drivers=n_drivers,
                        n_observables=0,
                        consts_per_eq=3,
                        precision="f32",
                        variant=variant,
                    )
            for n_obs in [8, 32]:
                for variant in ["local", "obs_shared"]:
                    yield dict(
                        phase="dims",
                        algorithm=algorithm,
                        n_states=n,
                        n_params=2,
                        n_drivers=0,
                        n_observables=n_obs,
                        consts_per_eq=3,
                        precision="f32",
                        variant=variant,
                    )


def consts_matrix():
    """Yield the baked-constants spot check (register pressure)."""
    for algorithm in ["tsit5", "backwards_euler"]:
        for n in [16, 64]:
            for consts_per_eq in [1, 8]:
                for variant in ["local", "state_params_shared"]:
                    yield dict(
                        phase="consts",
                        algorithm=algorithm,
                        n_states=n,
                        n_params=2,
                        n_drivers=0,
                        n_observables=0,
                        consts_per_eq=consts_per_eq,
                        precision="f32",
                        variant=variant,
                    )


def f64_matrix():
    """Yield float64 spot checks."""
    for algorithm in ["euler", "tsit5", "backwards_euler"]:
        for n in [16, 128]:
            for variant in [
                "local",
                "state_shared",
                "params_shared",
                "all_shared",
            ]:
                yield dict(
                    phase="f64",
                    algorithm=algorithm,
                    n_states=n,
                    n_params=2,
                    n_drivers=0,
                    n_observables=0,
                    consts_per_eq=3,
                    precision="f64",
                    variant=variant,
                )


PHASES = {
    "core": core_matrix,
    "dims": dims_matrix,
    "consts": consts_matrix,
    "f64": f64_matrix,
}


def config_key(cfg):
    """Return the deduplication key for a configuration."""
    fields = [
        "algorithm", "n_states", "n_params", "n_drivers",
        "n_observables", "consts_per_eq", "precision", "variant",
    ]
    return tuple(cfg[f] for f in fields)


def drive(phases):
    """Run every configuration not yet in the results file."""
    done = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "ratio_min" in rec or rec.get("error"):
                    done.add(config_key(rec))

    configs = [
        cfg for phase in phases for cfg in PHASES[phase]()
        if cfg["variant"] != "local"
    ]
    todo = [c for c in configs if config_key(c) not in done]
    print(f"{len(configs)} configs, {len(todo)} to run")

    env = dict(os.environ)
    for idx, cfg in enumerate(todo):
        label = "/".join(str(v) for v in config_key(cfg))
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--single",
                 json.dumps(cfg)],
                capture_output=True, text=True, timeout=900, env=env,
            )
        except subprocess.TimeoutExpired:
            record = dict(cfg)
            record["error"] = "timeout after 900 s"
            with open(RESULTS_FILE, "a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"[{idx + 1}/{len(todo)}] {label}: TIMEOUT")
            continue
        elapsed = time.perf_counter() - t0
        record = None
        for line in reversed(proc.stdout.strip().splitlines()):
            try:
                candidate = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(candidate, dict):
                record = candidate
                break
        if record is None:
            record = dict(cfg)
            record["error"] = (proc.stdout + proc.stderr)[-2000:]
        if record.get("skip"):
            print(f"[{idx + 1}/{len(todo)}] {label}: skipped")
            continue
        with open(RESULTS_FILE, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        if "ratio_min" in record:
            vstats = record["variant_stats"]
            print(
                f"[{idx + 1}/{len(todo)}] {label}: "
                f"ratio {record['ratio_min']:.3f} "
                f"(local {record['local_ms_min']:.2f} ms, "
                f"variant {record['variant_ms_min']:.2f} ms, "
                f"bs {vstats['eff_blocksize']}, "
                f"sh {vstats['shared_bytes_per_run']} B/run, "
                f"maxdiff {record['out_maxdiff']:.2e}) "
                f"[{elapsed:.0f} s]"
            )
        else:
            print(f"[{idx + 1}/{len(todo)}] {label}: ERROR")
    print("SWEEP DONE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", default=None)
    parser.add_argument(
        "--phase", default="all",
        choices=["all"] + list(PHASES),
    )
    args = parser.parse_args()
    if args.single:
        run_single(json.loads(args.single))
    else:
        phases = list(PHASES) if args.phase == "all" else [args.phase]
        drive(phases)


if __name__ == "__main__":
    main()
