"""Newton--Krylov solver factories for matrix-free integrators.

The helpers in this module wrap the linear solver provided by
:mod:`cubie.integrators.matrix_free_solvers.linear_solver` to build damped
Newton iterations suitable for CUDA device execution.
"""

from typing import Callable, Optional

import attrs
from attrs import validators
from numba import cuda, int32, from_dtype
import numpy as np
from cubie._utils import ALLOWED_PRECISIONS, PrecisionDType, getype_validator
from cubie.cuda_simsafe import activemask, all_sync, selp, any_sync
from cubie.integrators.matrix_free_solvers.linear_solver import (
    LinearSolverBufferSettings, LocalSizes, SliceIndices
)


@attrs.define
class NewtonLocalSizes(LocalSizes):
    """Local array sizes for Newton solver buffers."""

    delta: int = attrs.field(validator=getype_validator(int, 0))
    residual: int = attrs.field(validator=getype_validator(int, 0))
    residual_temp: int = attrs.field(validator=getype_validator(int, 0))
    stage_base_bt: int = attrs.field(validator=getype_validator(int, 0))
    krylov_iters: int = attrs.field(validator=getype_validator(int, 0))


@attrs.define
class NewtonSliceIndices(SliceIndices):
    """Slice container for Newton solver shared memory layouts."""

    delta: slice = attrs.field()
    residual: slice = attrs.field()
    residual_temp: slice = attrs.field()
    stage_base_bt: slice = attrs.field()
    local_end: int = attrs.field()
    lin_solver_start: int = attrs.field()


@attrs.define
class NewtonBufferSettings:
    """Configuration for Newton solver buffer sizes and locations."""

    n: int = attrs.field(validator=getype_validator(int, 1))
    delta_location: str = attrs.field(
        default='shared', validator=validators.in_(["local", "shared"])
    )
    residual_location: str = attrs.field(
        default='shared', validator=validators.in_(["local", "shared"])
    )
    residual_temp_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    stage_base_bt_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    linear_solver_buffer_settings: Optional[LinearSolverBufferSettings] = (
        attrs.field(default=None)
    )

    @property
    def use_shared_delta(self) -> bool:
        """Return True if delta buffer uses shared memory."""
        return self.delta_location == 'shared'

    @property
    def use_shared_residual(self) -> bool:
        """Return True if residual buffer uses shared memory."""
        return self.residual_location == 'shared'

    @property
    def use_shared_residual_temp(self) -> bool:
        """Return True if residual_temp buffer uses shared memory."""
        return self.residual_temp_location == 'shared'

    @property
    def use_shared_stage_base_bt(self) -> bool:
        """Return True if stage_base_bt buffer uses shared memory."""
        return self.stage_base_bt_location == 'shared'

    @property
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required."""
        total = 0
        if self.use_shared_delta:
            total += self.n
        if self.use_shared_residual:
            total += self.n
        if self.use_shared_residual_temp:
            total += self.n
        if self.use_shared_stage_base_bt:
            total += self.n
        if self.linear_solver_buffer_settings is not None:
            total += self.linear_solver_buffer_settings.shared_memory_elements
        return total

    @property
    def local_memory_elements(self) -> int:
        """Return total local memory elements required."""
        total = 0
        if not self.use_shared_delta:
            total += self.n
        if not self.use_shared_residual:
            total += self.n
        if not self.use_shared_residual_temp:
            total += self.n
        if not self.use_shared_stage_base_bt:
            total += self.n
        total += 1  # krylov_iters
        if self.linear_solver_buffer_settings is not None:
            total += self.linear_solver_buffer_settings.local_memory_elements
        return total

    @property
    def local_sizes(self) -> NewtonLocalSizes:
        """Return NewtonLocalSizes instance with buffer sizes."""
        return NewtonLocalSizes(
            delta=self.n,
            residual=self.n,
            residual_temp=self.n,
            stage_base_bt=self.n,
            krylov_iters=1,
        )

    @property
    def shared_indices(self) -> NewtonSliceIndices:
        """Return NewtonSliceIndices instance with shared memory layout."""
        ptr = 0
        if self.use_shared_delta:
            delta_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            delta_slice = slice(0, 0)

        if self.use_shared_residual:
            residual_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            residual_slice = slice(0, 0)

        if self.use_shared_residual_temp:
            residual_temp_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            residual_temp_slice = slice(0, 0)

        if self.use_shared_stage_base_bt:
            stage_base_bt_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            stage_base_bt_slice = slice(0, 0)

        return NewtonSliceIndices(
            delta=delta_slice,
            residual=residual_slice,
            residual_temp=residual_temp_slice,
            stage_base_bt=stage_base_bt_slice,
            local_end=ptr,
            lin_solver_start=ptr,
        )


def newton_krylov_solver_factory(
    residual_function: Callable,
    linear_solver: Callable,
    n: int,
    tolerance: float,
    max_iters: int,
    damping: float = 0.5,
    max_backtracks: int = 8,
    precision: PrecisionDType = np.float32,
    buffer_settings: Optional[NewtonBufferSettings] = None,
) -> Callable:
    """Create a damped Newton--Krylov solver device function.

    Parameters
    ----------
    residual_function
        Matrix-free residual evaluator with signature
        ``(stage_increment, parameters, drivers, t, h, a_ij, base_state,
        residual)``.
    linear_solver
        Matrix-free linear solver created by :func:`linear_solver_factory`.
    n
        Size of the flattened residual and state vectors.
    tolerance
        Residual norm threshold for convergence.
    max_iters
        Maximum number of Newton iterations performed.
    damping
        Step shrink factor used during backtracking.
    max_backtracks
        Maximum number of damping attempts per Newton step.
    precision
        Floating-point precision used when compiling the device function.
    buffer_settings
        Optional buffer settings controlling memory allocation. When provided,
        the solver uses selective allocation between shared and local memory.
        When None (default), all buffers use shared memory.

    Returns
    -------
    Callable
        CUDA device function implementing the damped Newton--Krylov scheme.
        The return value encodes the iteration count in the upper 16 bits and
        a :class:`~cubie.integrators.matrix_free_solvers.SolverRetCodes`
        value in the lower 16 bits. Iteration counts are also returned via
        the counters parameter.

    Notes
    -----
    The lower 16 bits of the returned status report the convergence outcome:
    ``0`` for success, ``1`` when backtracking cannot find a suitable step,
    ``2`` when the Newton iteration limit is exceeded, and ``4`` when the
    inner linear solver signals failure. The upper 16 bits hold the number of
    Newton iterations performed. Iteration counts are also written to
    the counters array: counters[0] holds Newton iterations and counters[1]
    holds total Krylov iterations.
    """

    precision_dtype = np.dtype(precision)
    if precision_dtype not in ALLOWED_PRECISIONS:
        raise ValueError("precision must be float16, float32, or float64.")

    # Default buffer settings - shared delta/residual (current behavior)
    if buffer_settings is None:
        buffer_settings = NewtonBufferSettings(n=n)

    # Extract compile-time flags
    delta_shared = buffer_settings.use_shared_delta
    residual_shared = buffer_settings.use_shared_residual
    shared_indices = buffer_settings.shared_indices
    delta_slice = shared_indices.delta
    residual_slice = shared_indices.residual
    lin_solver_start = shared_indices.lin_solver_start
    local_sizes = buffer_settings.local_sizes
    delta_local_size = local_sizes.nonzero('delta')
    residual_local_size = local_sizes.nonzero('residual')
    residual_temp_shared = buffer_settings.use_shared_residual_temp
    residual_temp_slice = shared_indices.residual_temp
    residual_temp_local_size = local_sizes.nonzero('residual_temp')
    stage_base_bt_shared = buffer_settings.use_shared_stage_base_bt
    stage_base_bt_slice = shared_indices.stage_base_bt
    stage_base_bt_local_size = local_sizes.nonzero('stage_base_bt')

    numba_precision = from_dtype(precision_dtype)
    tol_squared = numba_precision(tolerance * tolerance)
    typed_zero = numba_precision(0.0)
    typed_one = numba_precision(1.0)
    typed_damping = numba_precision(damping)
    n_val = int32(n)
    max_iters = int32(max_iters)
    max_backtracks = int32(max_backtracks)
    # no cover: start

    @cuda.jit(device=True, inline=True)
    def newton_krylov_solver(
        stage_increment,
        parameters,
        drivers,
        t,
        h,
        a_ij,
        base_state,
        shared_scratch,
        counters,
    ):
        """Solve a nonlinear system with a damped Newton--Krylov iteration.

        Parameters
        ----------
        stage_increment
            Current Newton iterate representing the stage increment.
        parameters
            Model parameters forwarded to the residual evaluation.
        drivers
            External drivers forwarded to the residual evaluation.
        t
            Stage time forwarded to the residual and linear solver.
        h
            Timestep scaling factor supplied by the outer integrator.
        a_ij
            Stage weight used by multi-stage integrators.
        base_state
            Reference state used when evaluating the residual.
        shared_scratch
            Shared scratch buffer providing Newton direction, residual,
            and linear solver storage.
        counters
            Size (2,) int32 array for iteration counters.

        Returns
        -------
        int
            Status word with convergence information and iteration count.
        """

        # Selective allocation based on buffer_settings
        if delta_shared:
            delta = shared_scratch[delta_slice]
        else:
            delta = cuda.local.array(delta_local_size, numba_precision)
            for _i in range(delta_local_size):
                delta[_i] = typed_zero

        if residual_shared:
            residual = shared_scratch[residual_slice]
        else:
            residual = cuda.local.array(residual_local_size, numba_precision)
            for _i in range(residual_local_size):
                residual[_i] = typed_zero

        if residual_temp_shared:
            residual_temp = shared_scratch[residual_temp_slice]
        else:
            residual_temp = cuda.local.array(
                residual_temp_local_size, numba_precision
            )

        residual_function(
            stage_increment,
            parameters,
            drivers,
            t,
            h,
            a_ij,
            base_state,
            residual,
        )
        norm2_prev = typed_zero
        for i in range(n_val):
            residual_value = residual[i]
            residual[i] = -residual_value
            delta[i] = typed_zero
            norm2_prev += residual_value * residual_value

        converged = norm2_prev <= tol_squared
        has_error = False
        final_status = int32(0)

        krylov_iters_local = cuda.local.array(1, int32)

        iters_count = int32(0)
        total_krylov_iters = int32(0)
        mask = activemask()
        for _ in range(max_iters):
            done = converged or has_error
            if all_sync(mask, done):
                break

            active = not done
            iters_count = selp(
                active, int32(iters_count + int32(1)), iters_count
            )

            lin_shared = shared_scratch[lin_solver_start:]
            krylov_iters_local[0] = int32(0)
            lin_status = linear_solver(
                stage_increment,
                parameters,
                drivers,
                base_state,
                t,
                h,
                a_ij,
                residual,
                delta,
                lin_shared,
                krylov_iters_local,
            )

            lin_failed = lin_status != int32(0)
            has_error = has_error or lin_failed
            final_status = selp(
                lin_failed, int32(final_status | lin_status), final_status
            )
            total_krylov_iters += selp(active, krylov_iters_local[0], int32(0))

            if stage_base_bt_shared:
                stage_base_bt = shared_scratch[stage_base_bt_slice]
            else:
                stage_base_bt = cuda.local.array(stage_base_bt_local_size,
                                                 numba_precision)
            for i in range(n_val):
                stage_base_bt[i] = stage_increment[i]
            found_step = False
            alpha = typed_one

            for _ in range(max_backtracks):
                active_bt = active and (not found_step) and (not converged)
                if not any_sync(mask, active_bt):
                    break

                if active_bt:
                    for i in range(n_val):
                        stage_increment[i] = stage_base_bt[i] + alpha * delta[i]

                    residual_function(
                        stage_increment,
                        parameters,
                        drivers,
                        t,
                        h,
                        a_ij,
                        base_state,
                        residual_temp,
                    )

                    norm2_new = typed_zero
                    for i in range(n_val):
                        residual_value = residual_temp[i]
                        norm2_new += residual_value * residual_value

                    if norm2_new <= tol_squared:
                        converged = True
                        found_step = True

                    if norm2_new < norm2_prev:
                        for i in range(n_val):
                            residual[i] = -residual_temp[i]
                        norm2_prev = norm2_new
                        found_step = True

                alpha *= typed_damping

            backtrack_failed = active and (not found_step) and (not converged)
            has_error = has_error or backtrack_failed
            final_status = selp(
                backtrack_failed, int32(final_status | int32(1)), final_status
            )

            if backtrack_failed:
                for i in range(n_val):
                    stage_increment[i] = stage_base_bt[i]

        max_iters_exceeded = (not converged) and (not has_error)
        final_status = selp(
            max_iters_exceeded, int32(final_status | int32(2)), final_status
        )

        counters[0] = iters_count
        counters[1] = total_krylov_iters

        return final_status

    # no cover: end
    return newton_krylov_solver
