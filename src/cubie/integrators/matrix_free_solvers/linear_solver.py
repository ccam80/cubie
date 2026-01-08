"""Matrix-free preconditioned linear solver.

This module builds CUDA device functions that implement steepest-descent or
minimal-residual iterations without forming Jacobian matrices explicitly.
The helpers interact with the nonlinear solvers in :mod:`cubie.integrators`
and expect caller-supplied operator and preconditioner callbacks.
"""

from typing import Callable, Optional, Set, Dict, Any, Union

from attrs import Converter, define, field, validators
from numba import cuda, int32, from_dtype
from numpy import asarray, dtype as np_dtype, full, isscalar, ndarray
from numpy.typing import ArrayLike

from cubie._utils import (
    PrecisionDType,
    build_config,
    float_array_validator,
    getype_validator,
    gttype_validator,
    inrangetype_validator,
    is_device_validator,
    precision_converter,
    precision_validator,
)
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import CUDAFactory, CUDAFunctionCache
from cubie.cuda_simsafe import activemask, all_sync, compile_kwargs, selp
from cubie.cuda_simsafe import from_dtype as simsafe_dtype


def tol_converter(
    value: Union[float, ArrayLike],
    self_: "LinearSolverConfig",
) -> ndarray:
    """Convert tolerance input into an array with solver precision.

    Parameters
    ----------
    value
        Scalar or array-like tolerance specification.
    self_
        Configuration instance providing precision and dimension info.

    Returns
    -------
    numpy.ndarray
        Tolerance array with one value per state variable.

    Raises
    ------
    ValueError
        Raised when ``value`` cannot be broadcast to the expected shape.
    """
    if isscalar(value):
        tol = full(self_.n, value, dtype=self_.precision)
    else:
        tol = asarray(value, dtype=self_.precision)
        if tol.shape[0] == 1 and self_.n > 1:
            tol = full(self_.n, tol[0], dtype=self_.precision)
        elif tol.shape[0] != self_.n:
            raise ValueError("tol must have shape (n,).")
    return tol


@define
class LinearSolverConfig:
    """Configuration for LinearSolver compilation.

    Attributes
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    n : int
        Length of residual and search-direction vectors.
    operator_apply : Optional[Callable]
        Device function applying operator F @ v.
    preconditioner : Optional[Callable]
        Device function for approximate inverse preconditioner.
    linear_correction_type : str
        Line-search strategy ('steepest_descent' or 'minimal_residual').
    krylov_tolerance : float
        Target on squared residual norm for convergence.
    max_linear_iters : int
        Maximum iterations permitted.
    preconditioned_vec_location : str
        Memory location for preconditioned_vec buffer ('local' or 'shared').
    temp_location : str
        Memory location for temp buffer ('local' or 'shared').
    use_cached_auxiliaries : bool
        Whether to use cached auxiliary arrays (determines signature).
    """

    precision: PrecisionDType = field(
        converter=precision_converter,
        validator=precision_validator
    )
    n: int = field(validator=getype_validator(int, 1))
    operator_apply: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
    )
    preconditioner: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
    )
    linear_correction_type: str = field(
        default="minimal_residual",
        validator=validators.in_(["steepest_descent", "minimal_residual"])
    )
    _krylov_tolerance: float = field(
        default=1e-6,
        validator=gttype_validator(float, 0)
    )
    krylov_atol: ndarray = field(
        default=asarray([1e-6]),
        validator=float_array_validator,
        converter=Converter(tol_converter, takes_self=True)
    )
    krylov_rtol: ndarray = field(
        default=asarray([1e-6]),
        validator=float_array_validator,
        converter=Converter(tol_converter, takes_self=True)
    )
    max_linear_iters: int = field(
        default=100,
        validator=inrangetype_validator(int, 1, 32767)
    )
    preconditioned_vec_location: str = field(
        default='local',
        validator=validators.in_(["local", "shared"])
    )
    temp_location: str = field(
        default='local',
        validator=validators.in_(["local", "shared"])
    )
    use_cached_auxiliaries: bool = field(default=False)

    @property
    def krylov_tolerance(self) -> float:
        """Return tolerance in configured precision."""
        return self.precision(self._krylov_tolerance)

    @property
    def numba_precision(self) -> type:
        """Return Numba type for precision."""
        return from_dtype(np_dtype(self.precision))

    @property
    def simsafe_precision(self) -> type:
        """Return CUDA-sim-safe type for precision."""
        return simsafe_dtype(np_dtype(self.precision))

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return linear solver configuration as dictionary.

        Returns
        -------
        dict
            Configuration dictionary containing:
            - krylov_tolerance: Convergence tolerance for linear solver
            - krylov_atol: Absolute tolerance array for scaled norm
            - krylov_rtol: Relative tolerance array for scaled norm
            - max_linear_iters: Maximum iterations permitted
            - linear_correction_type: Line-search strategy
            - preconditioned_vec_location: Buffer location for preconditioned vector
            - temp_location: Buffer location for temporary vector
        """
        return {
            'krylov_tolerance': self.krylov_tolerance,
            'krylov_atol': self.krylov_atol,
            'krylov_rtol': self.krylov_rtol,
            'max_linear_iters': self.max_linear_iters,
            'linear_correction_type': self.linear_correction_type,
            'preconditioned_vec_location': self.preconditioned_vec_location,
            'temp_location': self.temp_location,
        }


@define
class LinearSolverCache(CUDAFunctionCache):
    """Cache container for LinearSolver outputs.

    Attributes
    ----------
    linear_solver : Callable
        Compiled CUDA device function for linear solving.
    """

    linear_solver: Callable = field(
        validator=is_device_validator
    )


class LinearSolver(CUDAFactory):
    """Factory for linear solver device functions.

    Implements steepest-descent or minimal-residual iterations
    for solving linear systems without forming Jacobian matrices.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        **kwargs,
    ) -> None:
        """Initialize LinearSolver with parameters.

        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for computations.
        n : int
            Length of residual and search-direction vectors.
        **kwargs
            Optional parameters passed to LinearSolverConfig. See
            LinearSolverConfig for available parameters. None values
            are ignored.
        """
        super().__init__()

        config = build_config(
            LinearSolverConfig,
            required={
                'precision': precision,
                'n': n,
            },
            **kwargs
        )
        self.setup_compile_settings(config)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register device buffers with buffer_registry."""

        config = self.compile_settings
        buffer_registry.register(
            'preconditioned_vec',
            self,
            config.n,
            config.preconditioned_vec_location,
            precision=config.precision
        )
        buffer_registry.register(
            'temp',
            self,
            config.n,
            config.temp_location,
            precision=config.precision
        )

    def build(self) -> LinearSolverCache:
        """Compile linear solver device function.

        Returns
        -------
        LinearSolverCache
            Container with compiled linear_solver device function.

        Raises
        ------
        ValueError
            If operator_apply is None when build() is called.
        """
        config = self.compile_settings

        # Extract parameters from config
        operator_apply = config.operator_apply
        preconditioner = config.preconditioner
        n = config.n
        linear_correction_type = config.linear_correction_type
        krylov_tolerance = config.krylov_tolerance
        max_linear_iters = config.max_linear_iters
        precision = config.precision
        use_cached_auxiliaries = config.use_cached_auxiliaries

        # Compute flags for correction type
        sd_flag = linear_correction_type == "steepest_descent"
        mr_flag = linear_correction_type == "minimal_residual"
        preconditioned = preconditioner is not None

        # Extract tolerance arrays
        krylov_atol = config.krylov_atol
        krylov_rtol = config.krylov_rtol

        # Convert types for device function
        n_val = int32(n)
        max_iters_val = int32(max_linear_iters)
        precision_numba = from_dtype(np_dtype(precision))
        typed_zero = precision_numba(0.0)
        typed_one = precision_numba(1.0)
        inv_n = precision_numba(1.0 / n)
        tol_floor = precision_numba(1e-16)

        # Get allocators from buffer_registry
        get_alloc = buffer_registry.get_allocator
        alloc_precond = get_alloc('preconditioned_vec', self)
        alloc_temp = get_alloc('temp', self)

        # Build device function based on cached auxiliaries flag
        if use_cached_auxiliaries:
            # no cover: start
            @cuda.jit(
                device=True,
                inline=True,
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
                persistent_local,
                krylov_iters_out,
            ):
                """Run one cached preconditioned steepest-descent or MR solve."""

                # Allocate buffers from registry
                preconditioned_vec = alloc_precond(shared, persistent_local)
                temp = alloc_temp(shared, persistent_local)

                operator_apply(
                    state, parameters, drivers, cached_aux, base_state, t, h,
                    a_ij, x, temp
                )
                acc = typed_zero
                for i in range(n_val):
                    residual_value = rhs[i] - temp[i]
                    rhs[i] = residual_value
                    ref_val = x[i]
                    abs_ref = ref_val if ref_val >= typed_zero else -ref_val
                    tol_i = krylov_atol[i] + krylov_rtol[i] * abs_ref
                    tol_i = tol_i if tol_i > tol_floor else tol_floor
                    abs_res = residual_value if residual_value >= typed_zero \
                        else -residual_value
                    ratio = abs_res / tol_i
                    acc += ratio * ratio
                acc = acc * inv_n
                mask = activemask()
                converged = acc <= typed_one

                iter_count = int32(0)
                for _ in range(max_iters_val):
                    if all_sync(mask, converged):
                        break

                    iter_count += int32(1)
                    if preconditioned:
                        preconditioner(
                            state,
                            parameters,
                            drivers,
                            cached_aux,
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
                        cached_aux,
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

                    if denominator != typed_zero:
                        alpha = numerator / denominator
                    else:
                        alpha = typed_zero

                    acc = typed_zero
                    if not converged:
                        for i in range(n_val):
                            x[i] += alpha * preconditioned_vec[i]
                            rhs[i] -= alpha * temp[i]
                            residual_value = rhs[i]
                            ref_val = x[i]
                            abs_ref = ref_val if ref_val >= typed_zero \
                                else -ref_val
                            tol_i = krylov_atol[i] + krylov_rtol[i] * abs_ref
                            tol_i = tol_i if tol_i > tol_floor else tol_floor
                            abs_res = residual_value if residual_value >= \
                                typed_zero else -residual_value
                            ratio = abs_res / tol_i
                            acc += ratio * ratio
                        acc = acc * inv_n
                    else:
                        for i in range(n_val):
                            residual_value = rhs[i]
                            ref_val = x[i]
                            abs_ref = ref_val if ref_val >= typed_zero \
                                else -ref_val
                            tol_i = krylov_atol[i] + krylov_rtol[i] * abs_ref
                            tol_i = tol_i if tol_i > tol_floor else tol_floor
                            abs_res = residual_value if residual_value >= \
                                typed_zero else -residual_value
                            ratio = abs_res / tol_i
                            acc += ratio * ratio
                        acc = acc * inv_n

                    converged = converged or (acc <= typed_one)

                # Log "exceeded linear iters" status if still not converged
                final_status = selp(converged, int32(0), int32(4))
                krylov_iters_out[0] = iter_count
                return final_status

            # no cover: end
            return LinearSolverCache(linear_solver=linear_solver_cached)


        else:
            # Device function for non-cached variant
            # no cover: start
            @cuda.jit(
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
                persistent_local,
                krylov_iters_out,
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
                persistent_local
                    Persistent local memory array for selective buffer allocation.
                krylov_iters_out
                    Single-element int32 array to receive the iteration count.

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

                # Allocate buffers from registry
                preconditioned_vec = alloc_precond(shared, persistent_local)
                temp = alloc_temp(shared, persistent_local)

                operator_apply(
                    state, parameters, drivers, base_state, t, h, a_ij, x, temp
                )
                acc = typed_zero
                for i in range(n_val):
                    residual_value = rhs[i] - temp[i]
                    rhs[i] = residual_value
                    ref_val = x[i]
                    abs_ref = ref_val if ref_val >= typed_zero else -ref_val
                    tol_i = krylov_atol[i] + krylov_rtol[i] * abs_ref
                    tol_i = tol_i if tol_i > tol_floor else tol_floor
                    abs_res = residual_value if residual_value >= typed_zero \
                        else -residual_value
                    ratio = abs_res / tol_i
                    acc += ratio * ratio
                acc = acc * inv_n
                mask = activemask()
                converged = acc <= typed_one

                iter_count = int32(0)
                for _ in range(max_iters_val):
                    if all_sync(mask, converged):
                        break

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

                    if denominator != typed_zero:
                        alpha = numerator / denominator
                    else:
                        alpha = typed_zero

                    acc = typed_zero
                    if not converged:
                        for i in range(n_val):
                            x[i] += alpha * preconditioned_vec[i]
                            rhs[i] -= alpha * temp[i]
                            residual_value = rhs[i]
                            ref_val = x[i]
                            abs_ref = ref_val if ref_val >= typed_zero \
                                else -ref_val
                            tol_i = krylov_atol[i] + krylov_rtol[i] * abs_ref
                            tol_i = tol_i if tol_i > tol_floor else tol_floor
                            abs_res = residual_value if residual_value >= \
                                typed_zero else -residual_value
                            ratio = abs_res / tol_i
                            acc += ratio * ratio
                        acc = acc * inv_n
                    else:
                        for i in range(n_val):
                            residual_value = rhs[i]
                            ref_val = x[i]
                            abs_ref = ref_val if ref_val >= typed_zero \
                                else -ref_val
                            tol_i = krylov_atol[i] + krylov_rtol[i] * abs_ref
                            tol_i = tol_i if tol_i > tol_floor else tol_floor
                            abs_res = residual_value if residual_value >= \
                                typed_zero else -residual_value
                            ratio = abs_res / tol_i
                            acc += ratio * ratio
                        acc = acc * inv_n

                    converged = converged or (acc <= typed_one)

                # Log "exceeded linear iters" status if still not converged
                final_status = selp(converged, int32(0), int32(4))
                krylov_iters_out[0] = iter_count
                return final_status

            # no cover: end
            return LinearSolverCache(linear_solver=linear_solver)

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs
    ) -> Set[str]:
        """Update compile settings and invalidate cache if changed.

        Parameters
        ----------
        updates_dict : dict, optional
            Dictionary of settings to update.
        silent : bool, default False
            If True, suppress warnings about unrecognized keys.
        **kwargs
            Additional settings as keyword arguments.

        Returns
        -------
        set
            Set of recognized parameter names that were updated.
        """
        # Merge updates
        all_updates = {}
        if updates_dict:
            all_updates.update(updates_dict)
        all_updates.update(kwargs)

        recognized = set()

        if not all_updates:
            return recognized

        recognized |= self.update_compile_settings(updates_dict=all_updates, silent=True)

        # Buffer locations will trigger cache invalidation in compile settings
        buffer_registry.update(self, updates_dict=all_updates, silent=True)
        self.register_buffers()
        return recognized

    @property
    def device_function(self) -> Callable:
        """Return cached linear solver device function."""
        return self.get_cached_output("linear_solver")

    @property
    def precision(self) -> PrecisionDType:
        """Return configured precision."""
        return self.compile_settings.precision

    @property
    def n(self) -> int:
        """Return vector size."""
        return self.compile_settings.n

    @property
    def linear_correction_type(self) -> str:
        """Return correction strategy."""
        return self.compile_settings.linear_correction_type

    @property
    def krylov_tolerance(self) -> float:
        """Return convergence tolerance."""
        return self.compile_settings.krylov_tolerance

    @property
    def krylov_atol(self) -> ndarray:
        """Return absolute tolerance array."""
        return self.compile_settings.krylov_atol

    @property
    def krylov_rtol(self) -> ndarray:
        """Return relative tolerance array."""
        return self.compile_settings.krylov_rtol

    @property
    def max_linear_iters(self) -> int:
        """Return maximum iterations."""
        return self.compile_settings.max_linear_iters

    @property
    def use_cached_auxiliaries(self) -> bool:
        """Return whether cached auxiliaries are used."""
        return self.compile_settings.use_cached_auxiliaries

    @property
    def shared_buffer_size(self) -> int:
        """Return total shared memory elements required."""
        return buffer_registry.shared_buffer_size(self)

    @property
    def local_buffer_size(self) -> int:
        """Return total local memory elements required."""
        return buffer_registry.local_buffer_size(self)

    @property
    def persistent_local_buffer_size(self) -> int:
        """Return total persistent local memory elements required."""
        return buffer_registry.persistent_local_buffer_size(self)

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return linear solver configuration as dictionary.

        Delegates to compile_settings for configuration state.

        Returns
        -------
        dict
            Configuration dictionary from LinearSolverConfig.settings_dict
        """
        return self.compile_settings.settings_dict
