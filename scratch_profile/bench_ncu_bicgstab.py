r"""Nsight Compute harness: BiCGSTAB occupancy / register / spill study.

Usage
-----
Run one variant per ncu invocation (fresh process per variant so the
kernel is compiled for exactly that configuration)::

    ncu --set full -f -o bicg_<variant> `
        --kernel-name regex:integration_kernel `
        --launch-count 1 --launch-skip 1 `
        <venv>\python.exe scratch_profile\bench_ncu_bicgstab.py `
        <variant> [nruns] [blocksize]

``--launch-skip 1`` skips the compile-warmup launch and profiles the
hot launch. Variants:

- ``mr_local``          MR/Neumann baseline, all buffers local
- ``bicg_local``        BiCGSTAB, all buffers local (current default)
- ``bicg_shared``       BiCGSTAB, all 5 work vectors in shared
- ``bicg_mixed``        BiCGSTAB, long-lived vectors (r0_hat, p, v)
                        shared; transients (tmp, s_hat) local
- ``bicg_newton_shared``  bicg_shared plus Newton's 4 vectors shared

``nruns`` defaults to 65536 (256 blocks at blocksize 256 — enough to
saturate the 56 SMs on an RTX 4070 SUPER). ``blocksize`` defaults to
256; note that ``limit_blocksize`` halves it until the dynamic shared
allocation fits under cubie's 32 KiB-per-block ceiling, so the
*effective* block size for shared-heavy variants is smaller. The
script prints the effective launch shape so runs are comparable.

Key metrics to compare: sm__warps_active.avg.pct_of_peak (achieved
occupancy), launch__registers_per_thread,
l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum (spill traffic),
launch__shared_mem_per_block, launch__occupancy_limit_* (what caps
residency: registers, shared, blocks), and gpu__time_duration.sum.
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
PRECISION = np.float32 if "f32" in sys.argv[4:] else np.float64

eqs = [
    "dx0 = -k0*x0 + a*x1*x7 / (1.0 + x0*x0)",
    "dx1 = -k1*x1 + b*exp(-c*x0) * x2",
    "dx2 = -k2*x2 + x1*x3 - d*x2*x4",
    "dx3 = -k3*x3 + x2*x2 / (1.0 + x5)",
    "dx4 = -k4*x4 + e*x3*x5 - x4*x6",
    "dx5 = -k5*x5 + exp(-x4) + f*x6",
    "dx6 = -k6*x6 + x5*x7 - g*x6*x0",
    "dx7 = -k7*x7 + x6 / (1.0 + x7*x7)",
]
params = {
    "k0": 1.0, "k1": 10.0, "k2": 100.0, "k3": 1.0,
    "k4": 50.0, "k5": 5.0, "k6": 20.0, "k7": 2.0,
    "a": 0.5, "b": 0.3, "c": 0.2, "d": 0.1,
    "e": 0.4, "f": 0.6, "g": 0.7,
}
system = create_ODE_system(
    dxdt=eqs,
    states={f"x{i}": 0.5 for i in range(8)},
    parameters=params,
    precision=PRECISION,
    name=f"ncu_{VARIANT}_{PRECISION.__name__}",
)

solver_kwargs = {
    "algorithm": "backwards_euler",
    "dt": 1e-3,
    "krylov_max_iters": 100,
    "output_types": ["state"],
    "time_logging_level": None,
}

BICG_VECS = ("r0_hat", "p", "v", "tmp", "s_hat")
NEWTON_VECS = (
    "delta", "residual",
)

if VARIANT == "mr_local":
    solver_kwargs["linear_correction_type"] = "minimal_residual"
elif VARIANT == "bicg_local":
    solver_kwargs["linear_correction_type"] = "bicgstab"
elif VARIANT == "bicg_shared":
    solver_kwargs["linear_correction_type"] = "bicgstab"
    for name in BICG_VECS:
        solver_kwargs[f"{name}_location"] = "shared"
elif VARIANT == "bicg_mixed":
    solver_kwargs["linear_correction_type"] = "bicgstab"
    for name in ("r0_hat", "p", "v"):
        solver_kwargs[f"{name}_location"] = "shared"
elif VARIANT == "bicg_newton_shared":
    solver_kwargs["linear_correction_type"] = "bicgstab"
    for name in BICG_VECS:
        solver_kwargs[f"{name}_location"] = "shared"
    for name in NEWTON_VECS:
        solver_kwargs[f"{name}_location"] = "shared"
else:
    raise SystemExit(f"unknown variant {VARIANT}")

solver = Solver(system, **solver_kwargs)

y0 = {f"x{i}": np.full(NRUNS, 0.5, dtype=PRECISION) for i in range(8)}
p = {"k0": np.full(NRUNS, 1.0, dtype=PRECISION)}

# Launch 1: compile warmup (skipped by --launch-skip 1)
t0 = time.perf_counter()
solver.solve(
    y0, p, duration=0.05, grid_type="verbatim", blocksize=BLOCKSIZE
)
print(f"[{VARIANT}] compile+first: {time.perf_counter() - t0:.2f} s")

# Effective launch shape: mirror BatchSolverKernel.run / the
# limit_blocksize 32 KiB dynamic-shared ceiling so runs with
# different buffer placements are compared on their real geometry.
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

# Hot launches (profiled by ncu; timed here for a no-ncu sanity run)
times = []
for _ in range(5):
    t0 = time.perf_counter()
    solver.solve(
        y0, p, duration=0.05, grid_type="verbatim", blocksize=BLOCKSIZE
    )
    times.append(time.perf_counter() - t0)
print(f"[{VARIANT}] hot: "
      + ", ".join(f"{t:.3f}" for t in times)
      + f" (min {min(times):.3f} s)")

kernel = solver.kernel.kernel
try:
    regs = list(kernel.get_regs_per_thread().values())[0]
    lmem = list(kernel.get_local_mem_per_thread().values())[0]
    print(f"[{VARIANT}] regs/thread: {regs}, local mem/thread: {lmem}")
    # Residency arithmetic for RTX 4070 SUPER (CC 8.9):
    # 65536 regs/SM, 100 KiB shared/SM, max 24 blocks & 1536 thr/SM.
    reg_limited_threads = (65536 // max(regs, 1)) // 32 * 32
    smem_limited_blocks = (
        (100 * 1024) // smem if smem > 0 else 24
    )
    smem_limited_threads = smem_limited_blocks * eff_blocksize
    print(
        f"[{VARIANT}] threads/SM ceiling: "
        f"regs {min(reg_limited_threads, 1536)}, "
        f"shared {min(smem_limited_threads, 1536)}"
    )
except Exception as exc:
    print(f"[{VARIANT}] dispatcher stats unavailable: {exc!r}")
