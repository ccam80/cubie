"""Matrix-free BiCGSTAB linear solver.

This module builds CUDA device functions that implement the
Bi-Conjugate Gradient Stabilized algorithm without forming
Jacobian matrices explicitly.

Published Classes
-----------------
:class:`BiCGSTABSolverConfig`
    Attrs configuration for the BiCGSTAB solver factory.

:class:`BiCGSTABSolver`
    CUDAFactory subclass that compiles a preconditioned BiCGSTAB
    solver for use inside Newton--Krylov iterations.

See Also
--------
:class:`~cubie.integrators.matrix_free_solvers.linear_solver_base.LinearSolverBase`
    Abstract parent providing shared infrastructure.
:class:`~cubie.integrators.matrix_free_solvers.newton_krylov.NewtonKrylov`
    Newton--Krylov solver that wraps a linear solver.
"""

from typing import Dict, Any

from attrs import define, field, validators
from numba import cuda, int32, from_dtype
from numpy import (
    dtype as np_dtype,
    float32 as np_float32,
    float64 as np_float64,
)

from cubie._utils import PrecisionDType
from cubie.integrators.matrix_free_solvers.linear_solver_base import (
    LinearSolverBaseConfig,
    LinearSolverBase,
    LinearSolverCache,
)
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import activemask, all_sync, compile_kwargs, selp
from cubie.result_codes import CUBIE_RESULT_CODES


@define
class BiCGSTABSolverConfig(LinearSolverBaseConfig):
    """Configuration for BiCGSTABSolver compilation.

    Attributes
    ----------
    r0_hat_location : str
        Memory location for r0_hat buffer (witness vector).
    p_location : str
        Memory location for p buffer (search direction).
    v_location : str
        Memory location for v buffer (operator product).
    tmp_location : str
        Memory location for tmp buffer (preconditioned/scratch).
    s_hat_location : str
        Memory location for s_hat buffer (preconditioned s).
    """

    r0_hat_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    p_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    v_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    tmp_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    s_hat_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return BiCGSTAB solver configuration as dictionary.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        return {
            "krylov_max_iters": self.max_iters,
            "linear_correction_type": "bicgstab",
            "r0_hat_location": self.r0_hat_location,
            "p_location": self.p_location,
            "v_location": self.v_location,
            "tmp_location": self.tmp_location,
            "s_hat_location": self.s_hat_location,
        }


class BiCGSTABSolver(LinearSolverBase):
    """Factory for BiCGSTAB linear solver device functions.

    Implements the Bi-Conjugate Gradient Stabilized algorithm
    for solving linear systems without forming Jacobian matrices.
    Uses 5 work vectors: r0_hat, p, v, tmp, s_hat.

    Parameters
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    n : int
        Length of residual and search-direction vectors.
    **kwargs
        Forwarded to :class:`BiCGSTABSolverConfig` and the norm
        factory.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        **kwargs,
    ) -> None:
        super().__init__(
            config_class=BiCGSTABSolverConfig,
            precision=precision,
            n=n,
            **kwargs,
        )

    def register_buffers(self) -> None:
        """Register 5 device buffers with buffer_registry."""
        config = self.compile_settings
        prec = config.precision
        for name, loc in [
            ("bicg_r0_hat", config.r0_hat_location),
            ("bicg_p", config.p_location),
            ("bicg_v", config.v_location),
            ("bicg_tmp", config.tmp_location),
            ("bicg_s_hat", config.s_hat_location),
        ]:
            buffer_registry.register(
                name, self, config.n, loc, precision=prec
            )

    @property
    def linear_correction_type(self) -> str:
        """Return 'bicgstab' as the correction strategy."""
        return "bicgstab"

    def build(self) -> LinearSolverCache:
        """Compile BiCGSTAB solver device function.

        Returns
        -------
        LinearSolverCache
            Container with compiled linear_solver device function.
        """
        config = self.compile_settings

        if config.use_cached_auxiliaries:
            raise NotImplementedError(
                "BiCGSTAB does not yet support cached auxiliaries "
                "(required by Rosenbrock-W methods)."
            )

        # Device Functions
        operator_apply = config.operator_apply
        preconditioner = config.preconditioner
        scaled_norm_fn = config.norm_device_function

        # Config parameters
        n = config.n
        max_iters = config.max_iters
        precision = config.precision

        preconditioned = preconditioner is not None

        # Convert types for device function
        n_val = int32(n)
        max_iters_val = int32(max_iters)
        precision_numba = from_dtype(np_dtype(precision))
        typed_zero = precision_numba(0.0)
        typed_one = precision_numba(1.0)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        max_linear_iters_exceeded = int32(
            CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED
        )
        bicgstab_breakdown = int32(CUBIE_RESULT_CODES.BICGSTAB_BREAKDOWN)

        # Breakdown thresholds (~ sqrt(eps) for rho, ~ eps for omega)
        # float32: eps ~ 1.2e-7, sqrt(eps) ~ 3.5e-4
        # float64: eps ~ 2.2e-16, sqrt(eps) ~ 1.5e-8
        if precision == np_float32:
            breakdown_tol_rho = precision_numba(1e-6)
            breakdown_tol_omega = precision_numba(1.2e-7)
        elif precision == np_float64:
            breakdown_tol_rho = precision_numba(1e-14)
            breakdown_tol_omega = precision_numba(2.3e-16)
        else:
            breakdown_tol_rho = precision_numba(1e-14)
            breakdown_tol_omega = precision_numba(2.3e-16)

        # Get allocators from buffer_registry
        get_alloc = buffer_registry.get_allocator
        alloc_r0_hat = get_alloc("bicg_r0_hat", self)
        alloc_p = get_alloc("bicg_p", self)
        alloc_v = get_alloc("bicg_v", self)
        alloc_tmp = get_alloc("bicg_tmp", self)
        alloc_s_hat = get_alloc("bicg_s_hat", self)

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **compile_kwargs,
        )
        def bicgstab_solver(
            state,
            parameters,
            drivers,
            base_state,
            t,
            h,
            a_ij,
            rhs,
            x,
            shared,
            persistent_local,
            krylov_iters_out,
        ):
            """Run one preconditioned BiCGSTAB solve.

            Parameters
            ----------
            state
                State vector forwarded to operator/preconditioner.
            parameters
                Model parameters forwarded to operator/preconditioner.
            drivers
                External drivers forwarded to operator/preconditioner.
            base_state
                Base state for n-stage operators.
            t
                Stage time.
            h
                Step size used by the operator evaluation.
            a_ij
                Stage coefficient.
            rhs
                Right-hand side; overwritten with running residual.
            x
                Initial guess; overwritten with final solution.
            shared
                Shared memory pool.
            persistent_local
                Persistent local memory pool.
            krylov_iters_out
                Single-element int32 array receiving iteration count.

            Returns
            -------
            int32
                ``0`` on convergence, ``4`` when iteration limit
                reached, ``128`` on BiCGSTAB breakdown.
            """

            # Allocate buffers
            r0_hat = alloc_r0_hat(shared, persistent_local)
            p = alloc_p(shared, persistent_local)
            v = alloc_v(shared, persistent_local)
            tmp = alloc_tmp(shared, persistent_local)
            s_hat = alloc_s_hat(shared, persistent_local)

            # ── INIT ────────────────────────────────────
            # I1-I2: r = rhs - A(x)
            operator_apply(
                state, parameters, drivers, base_state,
                t, h, a_ij, x, tmp,
            )
            for i in range(n_val):
                rhs[i] = rhs[i] - tmp[i]

            # I3-I4: freeze witness, init search direction
            for i in range(n_val):
                r0_hat[i] = rhs[i]
                p[i] = rhs[i]

            # I5: rho_prev = <r0, r0> (always >= 0)
            rho_prev = typed_zero
            for i in range(n_val):
                rho_prev += r0_hat[i] * rhs[i]
            rho_0 = rho_prev

            # I6: initial convergence check
            acc = scaled_norm_fn(rhs, x)
            mask = activemask()
            converged = acc <= typed_one
            broken = False
            finished = converged

            iter_count = int32(0)

            for _ in range(max_iters_val):
                if all_sync(mask, finished):
                    break

                iter_count = selp(
                    not finished,
                    int32(iter_count + int32(1)),
                    iter_count,
                )

                # ── Step 1: tmp = P(p), scratch = v ─────
                if preconditioned:
                    preconditioner(
                        state, parameters, drivers, base_state,
                        t, h, a_ij, p, tmp, v,
                    )
                else:
                    for i in range(n_val):
                        tmp[i] = p[i]

                # ── Step 2: v = A(tmp) ──────────────────
                operator_apply(
                    state, parameters, drivers, base_state,
                    t, h, a_ij, tmp, v,
                )

                # ── Step 3: alpha = rho_prev / <r0_hat, v>
                dot_r0v = typed_zero
                for i in range(n_val):
                    dot_r0v += r0_hat[i] * v[i]
                alpha = selp(
                    dot_r0v != typed_zero,
                    rho_prev / dot_r0v,
                    typed_zero,
                )

                # ── Step 4: x += alpha * tmp (predicated)
                for i in range(n_val):
                    x[i] = selp(
                        not finished,
                        x[i] + alpha * tmp[i],
                        x[i],
                    )

                # ── Step 5: s = r - alpha*v (in-place) ──
                for i in range(n_val):
                    rhs[i] = selp(
                        not finished,
                        rhs[i] - alpha * v[i],
                        rhs[i],
                    )

                # ── Step 6: half-step convergence check ─
                acc = scaled_norm_fn(rhs, x)
                converged = converged or (acc <= typed_one)
                finished = converged or broken

                # ── Step 7: s_hat = P(s), scratch = tmp ─
                if preconditioned:
                    preconditioner(
                        state, parameters, drivers, base_state,
                        t, h, a_ij, rhs, s_hat, tmp,
                    )
                else:
                    for i in range(n_val):
                        s_hat[i] = rhs[i]

                # ── Step 8: tmp = A(s_hat) ──────────────
                operator_apply(
                    state, parameters, drivers, base_state,
                    t, h, a_ij, s_hat, tmp,
                )

                # ── Step 9: omega = <tmp,s>/<tmp,tmp> ───
                dot_ts = typed_zero
                dot_tt = typed_zero
                for i in range(n_val):
                    ti = tmp[i]
                    dot_ts += ti * rhs[i]
                    dot_tt += ti * ti
                omega = selp(
                    dot_tt != typed_zero,
                    dot_ts / dot_tt,
                    typed_zero,
                )

                # ── Step 10: x += omega * s_hat ─────────
                for i in range(n_val):
                    x[i] = selp(
                        not finished,
                        x[i] + omega * s_hat[i],
                        x[i],
                    )

                # ── Step 11: r = s - omega*tmp ──────────
                for i in range(n_val):
                    rhs[i] = selp(
                        not finished,
                        rhs[i] - omega * tmp[i],
                        rhs[i],
                    )

                # ── Step 12: full-step convergence check ─
                acc = scaled_norm_fn(rhs, x)
                converged = converged or (acc <= typed_one)

                # ── Step 13: rho_new = <r0_hat, r> ──────
                rho_new = typed_zero
                for i in range(n_val):
                    rho_new += r0_hat[i] * rhs[i]

                # ── Step 14-15: breakdown detection ──────
                rho_bad = abs(rho_new) < breakdown_tol_rho * rho_0
                omega_bad = abs(omega) < breakdown_tol_omega
                broken = broken or (
                    (not converged) and (rho_bad or omega_bad)
                )
                finished = converged or broken

                # ── Step 16: beta ────────────────────────
                beta = selp(
                    not finished,
                    (rho_new / rho_prev) * (alpha / omega),
                    typed_zero,
                )

                # ── Step 17: p = r + beta*(p - omega*v) ──
                for i in range(n_val):
                    p[i] = selp(
                        not finished,
                        rhs[i] + beta * (p[i] - omega * v[i]),
                        p[i],
                    )

                # ── Step 18: rho_prev = rho_new ─────────
                rho_prev = selp(
                    not finished, rho_new, rho_prev
                )

            # ── Exit ────────────────────────────────────
            final_status = selp(
                converged, success,
                selp(broken, bicgstab_breakdown, max_linear_iters_exceeded),
            )
            krylov_iters_out[0] = iter_count
            return final_status

        # no cover: end
        return LinearSolverCache(linear_solver=bicgstab_solver)
