"""Matrix-free preconditioned linear solver.

This module builds CUDA device functions that implement steepest-descent or
minimal-residual iterations without forming Jacobian matrices explicitly.
The helpers interact with the nonlinear solvers in :mod:`cubie.integrators`
and expect caller-supplied operator and preconditioner callbacks.

Buffer settings classes for memory allocation configuration are also defined
here, providing selective allocation between shared and local memory.
"""

from typing import Callable, Optional

import attrs
from attrs import validators
from numba import cuda, int32, from_dtype
import numpy as np

from cubie._utils import getype_validator, PrecisionDType
from cubie.BufferSettings import BufferSettings, LocalSizes, SliceIndices
from cubie.cuda_simsafe import activemask, all_sync, compile_kwargs, selp


@attrs.define
class LinearSolverLocalSizes(LocalSizes):
    """Local array sizes for linear solver buffers with nonzero guarantees.

    Attributes
    ----------
    preconditioned_vec : int
        Preconditioned vector buffer size.
    temp : int
        Temporary vector buffer size.
    """

    preconditioned_vec: int = attrs.field(validator=getype_validator(int, 0))
    temp: int = attrs.field(validator=getype_validator(int, 0))


@attrs.define
class LinearSolverSliceIndices(SliceIndices):
    """Slice container for linear solver shared memory buffer layouts.

    Attributes
    ----------
    preconditioned_vec : slice
        Slice covering the preconditioned vector buffer (empty if local).
    temp : slice
        Slice covering the temporary vector buffer.
    local_end : int
        Offset of the end of solver-managed shared memory.
    """

    preconditioned_vec: slice = attrs.field()
    temp: slice = attrs.field()
    local_end: int = attrs.field()


@attrs.define
class LinearSolverBufferSettings(BufferSettings):
    """Configuration for linear solver buffer sizes and memory locations.

    Controls whether preconditioned_vec and temp buffers use shared or
    local memory during Krylov iteration.

    Attributes
    ----------
    n : int
        Number of state variables (length of vectors).
    preconditioned_vec_location : str
        Memory location for preconditioned vector: 'local' or 'shared'.
    temp_location : str
        Memory location for temporary vector: 'local' or 'shared'.
    """

    n: int = attrs.field(validator=getype_validator(int, 1))
    preconditioned_vec_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    temp_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )

    @property
    def use_shared_preconditioned_vec(self) -> bool:
        """Return True if preconditioned_vec uses shared memory."""
        return self.preconditioned_vec_location == 'shared'

    @property
    def use_shared_temp(self) -> bool:
        """Return True if temp buffer uses shared memory."""
        return self.temp_location == 'shared'

    @property
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required."""
        total = 0
        if self.use_shared_preconditioned_vec:
            total += self.n
        if self.use_shared_temp:
            total += self.n
        return total

    @property
    def local_memory_elements(self) -> int:
        """Return total local memory elements required."""
        total = 0
        if not self.use_shared_preconditioned_vec:
            total += self.n
        if not self.use_shared_temp:
            total += self.n
        return total

    @property
    def local_sizes(self) -> LinearSolverLocalSizes:
        """Return LinearSolverLocalSizes instance with buffer sizes.

        The returned object provides nonzero sizes suitable for
        cuda.local.array allocation.
        """
        return LinearSolverLocalSizes(
            preconditioned_vec=self.n,
            temp=self.n,
        )

    @property
    def shared_indices(self) -> LinearSolverSliceIndices:
        """Return LinearSolverSliceIndices instance with shared memory layout.

        The returned object contains slices for each buffer's region
        in shared memory. Local buffers receive empty slices.
        """
        ptr = 0

        if self.use_shared_preconditioned_vec:
            preconditioned_vec_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            preconditioned_vec_slice = slice(0, 0)

        if self.use_shared_temp:
            temp_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            temp_slice = slice(0, 0)

        return LinearSolverSliceIndices(
            preconditioned_vec=preconditioned_vec_slice,
            temp=temp_slice,
            local_end=ptr,
        )


def linear_solver_factory(
    operator_apply: Callable,
    n: int,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
    precision: PrecisionDType = np.float64,
    buffer_settings: Optional[LinearSolverBufferSettings] = None,
) -> Callable:
    """Create a CUDA device function implementing steepest-descent or MR.

    Parameters
    ----------
    operator_apply
        Callback that overwrites its output vector with ``F @ v``.
    n
        Length of the one-dimensional residual and search-direction vectors.
    preconditioner
        Approximate inverse preconditioner invoked as ``(state, parameters,
        drivers, t, h, residual, z, scratch)``. ``scratch`` can be overwritten.
        If ``None`` the identity preconditioner is used.
    correction_type
        Line-search strategy. Must be ``"steepest_descent"`` or
        ``"minimal_residual"``.
    tolerance
        Target on the squared residual norm that signals convergence.
    max_iters
        Maximum number of iterations permitted.
    precision
        Floating-point precision used when building the device function.
    buffer_settings
        Optional buffer settings controlling memory allocation. When provided,
        the solver uses selective allocation between shared and local memory.
        When None (default), all buffers use local memory.

    Returns
    -------
    Callable
        CUDA device function returning ``0`` on convergence and ``4`` when the
        iteration limit is reached. When buffer_settings specifies shared
        memory, the function signature includes a shared memory array
        parameter.

    Notes
    -----
    The operator typically has the form ``F = β M - γ h J`` where ``M`` is the
    mass matrix (often the identity), ``J`` is the Jacobian, ``h`` is the step
    size, and ``β`` and ``γ`` are scalar parameters captured in the closure.
    The solver instantiates its own local scratch buffers so callers only need
    to provide the residual and correction vectors.
    """

    sd_flag = 1 if correction_type == "steepest_descent" else 0
    mr_flag = 1 if correction_type == "minimal_residual" else 0
    if correction_type not in ("steepest_descent", "minimal_residual"):
        raise ValueError(
            "Correction type must be 'steepest_descent' or 'minimal_residual'."
        )
    preconditioned = 1 if preconditioner is not None else 0
    n_val = int32(n)
    max_iters = int32(max_iters)
    precision = from_dtype(precision)
    typed_zero = precision(0.0)
    tol_squared = precision(tolerance * tolerance)

    # Extract buffer settings as compile-time constants
    if buffer_settings is None:
        buffer_settings = LinearSolverBufferSettings(n=n)

    # Unpack boolean flags for selective allocation (compile-time constants)
    precond_vec_shared = buffer_settings.use_shared_preconditioned_vec
    temp_shared = buffer_settings.use_shared_temp

    # Unpack slice indices for shared memory layout
    slice_indices = buffer_settings.shared_indices
    precond_vec_slice = slice_indices.preconditioned_vec
    temp_slice = slice_indices.temp

    # Unpack local sizes for local array allocation
    local_sizes = buffer_settings.local_sizes
    precond_vec_local_size = local_sizes.nonzero('preconditioned_vec')
    temp_local_size = local_sizes.nonzero('temp')

    # no cover: start
    @cuda.jit(
        [
            (precision[::1],
             precision[::1],
             precision[::1],
             precision[::1],
             precision,
             precision,
             precision,
             precision[::1],
             precision[::1],
             precision[::1],
            )
        ],
        device=True,
        inline=True,
        **compile_kwargs,
    )
    def linear_solver(
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
    ):
        """Run one preconditioned steepest-descent or minimal-residual solve.

        Parameters
        ----------
        state
            State vector forwarded to the operator and preconditioner.
        parameters
            Model parameters forwarded to the operator and preconditioner.
        drivers
            External drivers forwarded to the operator and preconditioner.
        base_state
            Base state for n-stage operators (unused for single-stage).
        t
            Stage time forwarded to the operator and preconditioner.
        h
            Step size used by the operator evaluation.
        a_ij
            Stage coefficient forwarded to the operator and preconditioner.
        rhs
            Right-hand side of the linear system. Overwritten with the current
            residual.
        x
            Iterand provided as the initial guess and overwritten with the
            final solution.
        shared
            Shared memory array for selective buffer allocation.

        Returns
        -------
        int
            ``0`` on convergence or ``4`` when the iteration limit is reached.

        Notes
        -----
        ``rhs`` is updated in place to hold the running residual, and ``temp``
        is reused as the scratch vector passed to the preconditioner. The
        iteration therefore keeps just two auxiliary vectors of length ``n``.
        The operator, preconditioner behaviour, and correction strategy are
        fixed by the factory closure, while ``state``, ``parameters``, and
        ``drivers`` are treated as read-only context values.
        """

        # Selective memory allocation based on buffer_settings
        if precond_vec_shared:
            preconditioned_vec = shared[precond_vec_slice]
        else:
            preconditioned_vec = cuda.local.array(precond_vec_local_size,
                                                  precision)

        if temp_shared:
            temp = shared[temp_slice]
        else:
            temp = cuda.local.array(temp_local_size, precision)

        operator_apply(state, parameters, drivers, base_state, t, h, a_ij, x, temp)
        acc = typed_zero
        for i in range(n_val):
            residual_value = rhs[i] - temp[i]
            rhs[i] = residual_value
            acc += residual_value * residual_value
        mask = activemask()
        converged = acc <= tol_squared

        iter_count = int32(0)
        for _ in range(max_iters):
            iter_count += int32(1)
            if preconditioned:
                preconditioner(
                    state,
                    parameters,
                    drivers,
                    base_state,
                    t,
                    h,
                    a_ij,
                    rhs,
                    preconditioned_vec,
                    temp,
                )
            else:
                for i in range(n_val):
                    preconditioned_vec[i] = rhs[i]

            operator_apply(
                state,
                parameters,
                drivers,
                base_state,
                t,
                h,
                a_ij,
                preconditioned_vec,
                temp,
            )
            numerator = typed_zero
            denominator = typed_zero
            if sd_flag:
                for i in range(n_val):
                    zi = preconditioned_vec[i]
                    numerator += rhs[i] * zi
                    denominator += temp[i] * zi
            elif mr_flag:
                for i in range(n_val):
                    ti = temp[i]
                    numerator += ti * rhs[i]
                    denominator += ti * ti

            alpha = selp(
                denominator != typed_zero,
                numerator / denominator,
                typed_zero,
            )
            alpha_effective = selp(converged, precision(0.0), alpha)

            acc = typed_zero
            for i in range(n_val):
                x[i] += alpha_effective * preconditioned_vec[i]
                rhs[i] -= alpha_effective * temp[i]
                residual_value = rhs[i]
                acc += residual_value * residual_value
            converged = converged or (acc <= tol_squared)

            if all_sync(mask, converged):
                return_status = int32(0)
                return_status |= (iter_count + int32(1)) << 16
                return return_status
        return_status = int32(4)
        return_status |= (iter_count + int32(1)) << 16
        return return_status

    # no cover: end
    return linear_solver


def linear_solver_cached_factory(
    operator_apply: Callable,
    n: int,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
    precision: PrecisionDType = np.float64,
    buffer_settings: Optional[LinearSolverBufferSettings] = None,
) -> Callable:
    """Create a CUDA linear solver that forwards cached auxiliaries.

    Parameters
    ----------
    operator_apply
        Callback that overwrites its output vector with ``F @ v``.
    n
        Length of the one-dimensional residual and search-direction vectors.
    preconditioner
        Approximate inverse preconditioner. If ``None`` the identity
        preconditioner is used.
    correction_type
        Line-search strategy. Must be ``"steepest_descent"`` or
        ``"minimal_residual"``.
    tolerance
        Target on the squared residual norm that signals convergence.
    max_iters
        Maximum number of iterations permitted.
    precision
        Floating-point precision used when building the device function.
    buffer_settings
        Optional buffer settings controlling memory allocation. When provided,
        the solver uses selective allocation between shared and local memory.
        When None (default), all buffers use local memory.

    Returns
    -------
    Callable
        CUDA device function that runs the linear solver with cached aux.
    """

    sd_flag = 1 if correction_type == "steepest_descent" else 0
    mr_flag = 1 if correction_type == "minimal_residual" else 0
    if correction_type not in ("steepest_descent", "minimal_residual"):
        raise ValueError(
            "Correction type must be 'steepest_descent' or 'minimal_residual'."
        )
    preconditioned = 1 if preconditioner is not None else 0
    n_val = int32(n)
    max_iters = int32(max_iters)

    precision_dtype = np.dtype(precision)
    precision_scalar = from_dtype(precision_dtype)
    typed_zero = precision_scalar(0.0)
    tol_squared = tolerance * tolerance

    # Extract buffer settings as compile-time constants
    if buffer_settings is None:
        buffer_settings = LinearSolverBufferSettings(n=n)

    # Unpack boolean flags for selective allocation (compile-time constants)
    precond_vec_shared = buffer_settings.use_shared_preconditioned_vec
    temp_shared = buffer_settings.use_shared_temp

    # Unpack slice indices for shared memory layout
    slice_indices = buffer_settings.shared_indices
    precond_vec_slice = slice_indices.preconditioned_vec
    temp_slice = slice_indices.temp

    # Unpack local sizes for local array allocation
    local_sizes = buffer_settings.local_sizes
    precond_vec_local_size = local_sizes.nonzero('preconditioned_vec')
    temp_local_size = local_sizes.nonzero('temp')

    # no cover: start
    @cuda.jit(
        device=True,
        **compile_kwargs,
    )
    def linear_solver_cached(
        state,
        parameters,
        drivers,
        base_state,
        cached_aux,
        t,
        h,
        a_ij,
        rhs,
        x,
        shared,
    ):
        """Run one cached preconditioned steepest-descent or MR solve."""

        # Selective memory allocation based on buffer_settings
        if precond_vec_shared:
            preconditioned_vec = shared[precond_vec_slice]
        else:
            preconditioned_vec = cuda.local.array(precond_vec_local_size,
                                                  precision_scalar)

        if temp_shared:
            temp = shared[temp_slice]
        else:
            temp = cuda.local.array(temp_local_size, precision_scalar)

        operator_apply(
            state, parameters, drivers, base_state, cached_aux, t, h, a_ij,
                x, temp
        )
        acc = typed_zero
        for i in range(n_val):
            residual_value = rhs[i] - temp[i]
            rhs[i] = residual_value
            acc += residual_value * residual_value
        mask = activemask()
        converged = acc <= tol_squared

        iter_count = int32(0)
        for _ in range(max_iters):
            iter_count += int32(1)
            if preconditioned:
                preconditioner(
                    state,
                    parameters,
                    drivers,
                    base_state,
                    cached_aux,
                    t,
                    h,
                    a_ij,
                    rhs,
                    preconditioned_vec,
                    temp,
                )
            else:
                for i in range(n_val):
                    preconditioned_vec[i] = rhs[i]

            operator_apply(
                state,
                parameters,
                drivers,
                base_state,
                cached_aux,
                t,
                h,
                a_ij,
                preconditioned_vec,
                temp,
            )
            numerator = typed_zero
            denominator = typed_zero
            if sd_flag:
                for i in range(n_val):
                    zi = preconditioned_vec[i]
                    numerator += rhs[i] * zi
                    denominator += temp[i] * zi
            elif mr_flag:
                for i in range(n_val):
                    ti = temp[i]
                    numerator += ti * rhs[i]
                    denominator += ti * ti

            alpha = selp(
                denominator != typed_zero,
                numerator / denominator,
                typed_zero,
            )
            alpha_effective = selp(converged, precision_scalar(0.0), alpha)

            acc = typed_zero
            for i in range(n_val):
                x[i] += alpha_effective * preconditioned_vec[i]
                rhs[i] -= alpha_effective * temp[i]
                residual_value = rhs[i]
                acc += residual_value * residual_value
            converged = converged or (acc <= tol_squared)

            if all_sync(mask, converged):
                return_status = int32(0)
                return_status |= (iter_count + int32(1)) << 16
                return return_status
        return_status = int32(4)
        return_status |= (iter_count + int32(1)) << 16
        return return_status

    # no cover: end
    return linear_solver_cached
