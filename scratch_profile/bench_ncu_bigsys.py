r"""Nsight harness: buffer placement at production scale (n=200).

Synthetic system shaped like the real workloads: 200 states in a
nonlinear nearest-neighbour chain, 5 constants per equation (1000
total, baked into source as literals), 2 global parameters. The
Jacobian is tridiagonal-ish so codegen stays tractable while the
solver working set matches a real n=200 model.

Usage::

    python bench_ncu_bigsys.py <variant> [nruns] [blocksize] [f32|f64]

Variants:

- ``mr_local``       MR/Neumann baseline, all local
- ``bicg_local``     BiCGSTAB, all local (default placement)
- ``bicg_shared``    BiCGSTAB, all 5 work vectors shared
                     (32 KiB cap collapses blocksize to ~4)
- ``bicg_r0_shared`` only the read-mostly witness vector shared
- ``bicg_mixed``     r0_hat, p, v shared; tmp, s_hat local

Blocksize matters on the LOCAL side here too: fewer resident threads
per SM leaves more L1/L2 per thread for the ~4-16 KB per-run working
set. Sweep [256, 128, 64, 32] for bicg_local.
"""
import sys
import time

sys.path.insert(0, r"C:\local_working_projects\cubie-bicgstab-review\src")

import numpy as np  # noqa: E402
from cubie import create_ODE_system  # noqa: E402
from cubie.batchsolving.solver import Solver  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "bicg_local"
NRUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
BLOCKSIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 256
PRECISION = np.float64 if "f64" in sys.argv[4:] else np.float32

N = 200
for arg in sys.argv[4:]:
    if arg.startswith("n") and arg[1:].isdigit():
        N = int(arg[1:])
rng = np.random.default_rng(42)

eqs = []
constants = {}
for i in range(N):
    im1 = (i - 1) % N
    ip1 = (i + 1) % N
    constants[f"k{i}"] = float(rng.uniform(1.0, 10.0))
    constants[f"a{i}"] = float(rng.uniform(0.1, 0.5))
    constants[f"b{i}"] = float(rng.uniform(0.1, 0.5))
    constants[f"c{i}"] = float(rng.uniform(0.01, 0.1))
    constants[f"d{i}"] = float(rng.uniform(0.1, 0.3))
    eqs.append(
        f"dx{i} = -k{i}*x{i} + a{i}*x{im1} + b{i}*x{ip1}"
        f" + p0*c{i}*x{i}*x{ip1}"
        f" + d{i}*p1/(1.0 + x{i}*x{i})"
    )

system = create_ODE_system(
    dxdt=eqs,
    states={f"x{i}": 0.5 for i in range(N)},
    parameters={"p0": 1.0, "p1": 0.5},
    constants=constants,
    precision=PRECISION,
    name=f"ncu_big_{N}_{PRECISION.__name__}",
)

solver_kwargs = {
    "algorithm": "backwards_euler",
    "dt": 1e-3,
    "krylov_max_iters": 200,
    "output_types": ["state"],
    "time_logging_level": None,
}

BICG_VECS = ("r0_hat", "p", "v", "tmp", "s_hat")

if VARIANT == "mr_local":
    solver_kwargs["linear_correction_type"] = "minimal_residual"
elif VARIANT == "bicg_local":
    solver_kwargs["linear_correction_type"] = "bicgstab"
elif VARIANT == "bicg_shared":
    solver_kwargs["linear_correction_type"] = "bicgstab"
    for name in BICG_VECS:
        solver_kwargs[f"{name}_location"] = "shared"
elif VARIANT == "bicg_r0_shared":
    solver_kwargs["linear_correction_type"] = "bicgstab"
    solver_kwargs["r0_hat_location"] = "shared"
elif VARIANT == "bicg_p_shared":
    solver_kwargs["linear_correction_type"] = "bicgstab"
    solver_kwargs["p_location"] = "shared"
elif VARIANT == "bicg_r0_p_shared":
    solver_kwargs["linear_correction_type"] = "bicgstab"
    solver_kwargs["r0_hat_location"] = "shared"
    solver_kwargs["p_location"] = "shared"
elif VARIANT == "bicg_mixed":
    solver_kwargs["linear_correction_type"] = "bicgstab"
    for name in ("r0_hat", "p", "v"):
        solver_kwargs[f"{name}_location"] = "shared"
else:
    raise SystemExit(f"unknown variant {VARIANT}")

t0 = time.perf_counter()
solver = Solver(system, **solver_kwargs)
print(f"[{VARIANT}] Solver construction: "
      f"{time.perf_counter() - t0:.1f} s")

y0 = {
    f"x{i}": np.full(NRUNS, 0.5, dtype=PRECISION) for i in range(N)
}
p = {"p0": np.full(NRUNS, 1.0, dtype=PRECISION)}

t0 = time.perf_counter()
solver.solve(
    y0, p, duration=0.02, grid_type="verbatim", blocksize=BLOCKSIZE
)
print(f"[{VARIANT}] compile+first: {time.perf_counter() - t0:.1f} s")

bsk = solver.kernel
pad = 4 if bsk.shared_memory_needs_padding else 0
bytes_per_run = bsk.shared_memory_bytes + pad
smem = int(bytes_per_run * min(NRUNS, BLOCKSIZE))
eff_blocksize, smem = bsk.limit_blocksize(
    BLOCKSIZE, smem, bytes_per_run, NRUNS
)
blocks = max(1, -(-NRUNS // eff_blocksize))
print(
    f"[{VARIANT}] launch shape: {NRUNS} runs, "
    f"blocksize {BLOCKSIZE}->{eff_blocksize}, "
    f"{blocks} blocks, {bytes_per_run} B shared/run, "
    f"{smem} B dynamic shared/block"
)

times = []
for _ in range(3):
    t0 = time.perf_counter()
    solver.solve(
        y0, p, duration=0.02, grid_type="verbatim",
        blocksize=BLOCKSIZE,
    )
    times.append(time.perf_counter() - t0)
print(f"[{VARIANT}] hot: "
      + ", ".join(f"{t:.3f}" for t in times)
      + f" (min {min(times):.3f} s)")

kernel = solver.kernel.kernel
try:
    regs = list(kernel.get_regs_per_thread().values())[0]
    lmem = list(kernel.get_local_mem_per_thread().values())[0]
    print(f"[{VARIANT}] regs/thread: {regs}, "
          f"local mem/thread: {lmem}")
except Exception as exc:
    print(f"[{VARIANT}] dispatcher stats unavailable: {exc!r}")
