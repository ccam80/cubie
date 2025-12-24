"""Matrix-free preconditioned linear solver.

This module builds CUDA device functions that implement steepest-descent or
minimal-residual iterations without forming Jacobian matrices explicitly.
The helpers interact with the nonlinear solvers in :mod:`cubie.integrators`
and expect caller-supplied operator and preconditioner callbacks.
"""

from typing import Callable, Optional, Set, Dict, Any

import attrs
from attrs import validators
from numba import cuda, int32, from_dtype
import numpy as np

from cubie._utils import (
    PrecisionDType,
    build_config,
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


@attrs.define
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

    precision: PrecisionDType = attrs.field(
        converter=precision_converter,
        validator=precision_validator
    )
    n: int = attrs.field(validator=getype_validator(int, 1))
    operator_apply: Optional[Callable] = attrs.field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
    )
    preconditioner: Optional[Callable] = attrs.field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
    )
    linear_correction_type: str = attrs.field(
        default="minimal_residual",
        validator=validators.in_(["steepest_descent", "minimal_residual"])
    )
    _krylov_tolerance: float = attrs.field(
        default=1e-6,
        validator=gttype_validator(float, 0)
    )
    max_linear_iters: int = attrs.field(
        default=100,
        validator=inrangetype_validator(int, 1, 32767)
    )
    preconditioned_vec_location: str = attrs.field(
        default='local',
        validator=validators.in_(["local", "shared"])
    )
    temp_location: str = attrs.field(
        default='local',
        validator=validators.in_(["local", "shared"])
    )
    use_cached_auxiliaries: bool = attrs.field(default=False)

    @property
    def krylov_tolerance(self) -> float:
        """Return tolerance in configured precision."""
        return self.precision(self._krylov_tolerance)

    @property
    def numba_precision(self) -> type:
        """Return Numba type for precision."""
        return from_dtype(np.dtype(self.precision))

    @property
    def simsafe_precision(self) -> type:
        """Return CUDA-sim-safe type for precision."""
        return simsafe_dtype(np.dtype(self.precision))

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return linear solver configuration as dictionary.

        Returns
        -------
        dict
            Configuration dictionary containing:
            - krylov_tolerance: Convergence tolerance for linear solver
            - max_linear_iters: Maximum iterations permitted
            - linear_correction_type: Line-search strategy
            - preconditioned_vec_location: Buffer location for preconditioned vector
            - temp_location: Buffer location for temporary vector
        """
        return {
            'krylov_tolerance': self.krylov_tolerance,
            'max_linear_iters': self.max_linear_iters,
            'linear_correction_type': self.linear_correction_type,
            'preconditioned_vec_location': self.preconditioned_vec_location,
            'temp_location': self.temp_location,
        }


@attrs.define
class LinearSolverCache(CUDAFunctionCache):
    """Cache container for LinearSolver outputs.

    Attributes
    ----------
    linear_solver : Callable
        Compiled CUDA device function for linear solving.
    """

    linear_solver: Callable = attrs.field(
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

        # Convert types for device function
        n_val = int32(n)
        max_iters_val = int32(max_linear_iters)
        precision_numba = from_dtype(np.dtype(precision))
        typed_zero = precision_numba(0.0)
        tol_squared = precision_numba(krylov_tolerance * krylov_tolerance)

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
                    acc += residual_value * residual_value
                mask = activemask()
                converged = acc <= tol_squared

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
                            acc += residual_value * residual_value
                    else:
                        for i in range(n_val):
                            residual_value = rhs[i]
                            acc += residual_value * residual_value

                    converged = converged or (acc <= tol_squared)

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
                    acc += residual_value * residual_value
                mask = activemask()
                converged = acc <= tol_squared

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
                            acc += residual_value * residual_value
                    else:
                        for i in range(n_val):
                            residual_value = rhs[i]
                            acc += residual_value * residual_value

                    converged = converged or (acc <= tol_squared)

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
