"""Newton--Krylov solver factories for matrix-free integrators.

The helpers in this module wrap the linear solver provided by
:mod:`cubie.integrators.matrix_free_solvers.linear_solver` to build damped
Newton iterations suitable for CUDA device execution.
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
from cubie.cuda_simsafe import activemask, all_sync, selp, any_sync, compile_kwargs
from cubie.cuda_simsafe import from_dtype as simsafe_dtype

from cubie.integrators.matrix_free_solvers.linear_solver import LinearSolver


@attrs.define
class NewtonKrylovConfig:
    """Configuration for NewtonKrylov solver compilation.

    Attributes
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    n : int
        Size of state vectors.
    residual_function : Optional[Callable]
        Device function evaluating residuals.
    linear_solver_function : Optional[CUDA Device Function]
        LinearSolver instance for solving linear systems.
    newton_tolerance : float
        Residual norm threshold for convergence.
    max_newton_iters : int
        Maximum Newton iterations permitted.
    newton_damping : float
        Step shrink factor for backtracking.
    newton_max_backtracks : int
        Maximum damping attempts per Newton step.
    delta_location : str
        Memory location for delta buffer.
    residual_location : str
        Memory location for residual buffer.
    residual_temp_location : str
        Memory location for residual_temp buffer.
    stage_base_bt_location : str
        Memory location for stage_base_bt buffer.
    """

    precision: PrecisionDType = attrs.field(
        converter=precision_converter,
        validator=precision_validator
    )
    n: int = attrs.field(validator=getype_validator(int, 1))
    residual_function: Optional[Callable] = attrs.field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
    )
    linear_solver_function: Optional[Callable] = attrs.field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
    )
    _newton_tolerance: float = attrs.field(
        default=1e-3,
        validator=gttype_validator(float, 0)
    )
    max_newton_iters: int = attrs.field(
        default=100,
        validator=inrangetype_validator(int, 1, 32767)
    )
    _newton_damping: float = attrs.field(
        default=0.5,
        validator=inrangetype_validator(float, 0, 1)
    )
    newton_max_backtracks: int = attrs.field(
        default=8,
        validator=inrangetype_validator(int, 1, 32767)
    )
    delta_location: str = attrs.field(
        default='local',
        validator=validators.in_(["local", "shared"])
    )
    residual_location: str = attrs.field(
        default='local',
        validator=validators.in_(["local", "shared"])
    )
    residual_temp_location: str = attrs.field(
        default='local',
        validator=validators.in_(["local", "shared"])
    )
    stage_base_bt_location: str = attrs.field(
        default='local',
        validator=validators.in_(["local", "shared"])
    )
    krylov_iters_local_location: str = attrs.field(
        default='local',
        validator=validators.in_(["local", "shared"])
    )

    @property
    def newton_tolerance(self) -> float:
        """Return tolerance in configured precision."""
        return self.precision(self._newton_tolerance)

    @property
    def newton_damping(self) -> float:
        """Return damping factor in configured precision."""
        return self.precision(self._newton_damping)

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
        """Return Newton-Krylov configuration as dictionary.

        Returns
        -------
        dict
            Configuration dictionary containing:
            - newton_tolerance: Residual norm threshold for convergence
            - max_newton_iters: Maximum Newton iterations permitted
            - newton_damping: Step shrink factor for backtracking
            - newton_max_backtracks: Maximum damping attempts per Newton step
            - delta_location: Buffer location for delta
            - residual_location: Buffer location for residual
            - residual_temp_location: Buffer location for residual_temp
            - stage_base_bt_location: Buffer location for stage_base_bt
        """
        return {
            'newton_tolerance': self.newton_tolerance,
            'max_newton_iters': self.max_newton_iters,
            'newton_damping': self.newton_damping,
            'newton_max_backtracks': self.newton_max_backtracks,
            'delta_location': self.delta_location,
            'residual_location': self.residual_location,
            'residual_temp_location': self.residual_temp_location,
            'stage_base_bt_location': self.stage_base_bt_location,
            'krylov_iters_local_location': self.krylov_iters_local_location,
        }


@attrs.define
class NewtonKrylovCache(CUDAFunctionCache):
    """Cache container for NewtonKrylov outputs.

    Attributes
    ----------
    newton_krylov_solver : Callable
        Compiled CUDA device function for Newton-Krylov solving.
    """

    newton_krylov_solver: Callable = attrs.field(
        validator=is_device_validator
    )


class NewtonKrylov(CUDAFactory):
    """Factory for Newton-Krylov solver device functions.

    Implements damped Newton iteration using a matrix-free
    linear solver for the correction equation.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        linear_solver: LinearSolver,
        **kwargs,
    ) -> None:
        """Initialize NewtonKrylov with parameters.

        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for computations.
        n : int
            Size of state vectors.
        linear_solver : LinearSolver
            LinearSolver instance for solving linear systems.
        **kwargs
            Optional parameters passed to NewtonKrylovConfig. See
            NewtonKrylovConfig for available parameters. None values
            are ignored.
        """
        super().__init__()

        self.linear_solver = linear_solver

        config = build_config(
            NewtonKrylovConfig,
            required={
                'precision': precision,
                'n': n,
            },
            **kwargs
        )

        self.setup_compile_settings(config)
        self.register_buffers()


    def register_buffers(self) -> None:
        """Register buffers according to locations in compile settings."""
        # Register buffers with buffer_registry
        config = self.compile_settings
        precision = config.precision

        buffer_registry.register(
            'delta',
            self,
            config.n,
            config.delta_location,
            precision=precision
        )
        buffer_registry.register(
            'residual',
            self,
            config.n,
            config.residual_location,
            precision=precision
        )
        buffer_registry.register(
            'residual_temp',
            self,
            config.n,
            config.residual_temp_location,
            precision=precision
        )
        buffer_registry.register(
            'stage_base_bt',
            self,
            config.n,
            config.stage_base_bt_location,
            precision=precision
        )
        buffer_registry.register(
            'krylov_iters_local',
            self,
            1,
            config.krylov_iters_local_location,
            precision=np.int32
        )

    def build(self) -> NewtonKrylovCache:
        """Compile Newton-Krylov solver device function.

        Returns
        -------
        NewtonKrylovCache
            Container with compiled newton_krylov_solver device function.

        Raises
        ------
        ValueError
            If residual_function or linear_solver is None when build() is called.
        """
        config = self.compile_settings

        # Extract parameters from config
        residual_function = config.residual_function
        linear_solver_fn = config.linear_solver_function

        n = config.n
        newton_tolerance = config.newton_tolerance
        max_newton_iters = config.max_newton_iters
        newton_damping = config.newton_damping
        newton_max_backtracks = config.newton_max_backtracks

        numba_precision = config.numba_precision
        tol_squared = numba_precision(newton_tolerance * newton_tolerance)
        typed_zero = numba_precision(0.0)
        typed_one = numba_precision(1.0)
        typed_damping = numba_precision(newton_damping)
        n_val = int32(n)
        max_iters_val = int32(max_newton_iters)
        max_backtracks_val = int32(newton_max_backtracks + 1)

        # Get allocators from buffer_registry
        get_alloc = buffer_registry.get_allocator
        alloc_delta = get_alloc('delta', self)
        alloc_residual = get_alloc('residual', self)
        alloc_residual_temp = get_alloc('residual_temp', self)
        alloc_stage_base_bt = get_alloc('stage_base_bt', self)
        alloc_krylov_iters_local = get_alloc('krylov_iters_local', self)

        # Get child allocators for linear solver
        alloc_lin_shared, alloc_lin_persistent = (
            buffer_registry.get_child_allocators(self, self.linear_solver)
        )

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **compile_kwargs
        )
        def newton_krylov_solver(
            stage_increment,
            parameters,
            drivers,
            t,
            h,
            a_ij,
            base_state,
            shared_scratch,
            persistent_scratch,
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
            persistent_scratch
                Persistent local scratch buffer for Newton and linear solver.
            counters
                Size (2,) int32 array for iteration counters.

            Returns
            -------
            int
                Status word with convergence information and iteration count.
            """

            # Allocate buffers from registry
            delta = alloc_delta(shared_scratch, persistent_scratch)
            residual = alloc_residual(shared_scratch, persistent_scratch)
            residual_temp = alloc_residual_temp(shared_scratch, persistent_scratch)
            stage_base_bt = alloc_stage_base_bt(shared_scratch, persistent_scratch)
            lin_shared = alloc_lin_shared(shared_scratch, persistent_scratch)
            lin_persistent = alloc_lin_persistent(shared_scratch, persistent_scratch)

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

            krylov_iters_local = alloc_krylov_iters_local(
                shared_scratch, persistent_scratch
            )

            iters_count = int32(0)
            total_krylov_iters = int32(0)
            mask = activemask()
            for _ in range(max_iters_val):
                done = converged or has_error
                if all_sync(mask, done):
                    break

                active = not done
                iters_count = selp(
                    active, int32(iters_count + int32(1)), iters_count
                )

                krylov_iters_local[0] = int32(0)
                lin_status = linear_solver_fn(
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
                    lin_persistent,
                    krylov_iters_local,
                )

                lin_failed = lin_status != int32(0)
                has_error = has_error or lin_failed
                final_status = selp(
                    lin_failed, int32(final_status | lin_status), final_status
                )
                total_krylov_iters += selp(active, krylov_iters_local[0], int32(0))

                for i in range(n_val):
                    stage_base_bt[i] = stage_increment[i]
                found_step = False
                alpha = typed_one

                for _ in range(max_backtracks_val):
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
                            # Negate residual for return
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
                    # Revert increment to unscaled value for another go
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
        return NewtonKrylovCache(newton_krylov_solver=newton_krylov_solver)

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

        if not all_updates:
            return set()

        recognized = set()
        recognized |= self.linear_solver.update(all_updates, silent=True)

        # Update device function, so that cache invalidates if it's changed
        all_updates['linear_solver_function'] = self.linear_solver.device_function
        recognized |= self.update_compile_settings(
                updates_dict=all_updates, silent=True
            )

        # Buffer locations will trigger cache invalidation in compile settings
        buffer_registry.update(self, updates_dict=all_updates, silent=True)
        self.register_buffers()

        return recognized

    @property
    def device_function(self) -> Callable:
        """Return cached Newton-Krylov solver device function."""
        return self.get_cached_output("newton_krylov_solver")

    @property
    def precision(self) -> PrecisionDType:
        """Return configured precision."""
        return self.compile_settings.precision

    @property
    def n(self) -> int:
        """Return vector size."""
        return self.compile_settings.n

    @property
    def newton_tolerance(self) -> float:
        """Return convergence tolerance."""
        return self.compile_settings.newton_tolerance

    @property
    def max_newton_iters(self) -> int:
        """Return maximum Newton iterations."""
        return self.compile_settings.max_newton_iters

    @property
    def newton_damping(self) -> float:
        """Return damping factor."""
        return self.compile_settings.newton_damping

    @property
    def newton_max_backtracks(self) -> int:
        """Return maximum backtracking steps."""
        return self.compile_settings.newton_max_backtracks

    @property
    def krylov_tolerance(self) -> float:
        """Return krylov tolerance from nested linear solver."""
        return self.linear_solver.krylov_tolerance

    @property
    def max_linear_iters(self) -> int:
        """Return max linear iterations from nested linear solver."""
        return self.linear_solver.max_linear_iters

    @property
    def linear_correction_type(self) -> str:
        """Return correction type from nested linear solver."""
        return self.linear_solver.linear_correction_type

    @property
    def shared_buffer_size(self) -> int:
        """Return total shared memory elements required.

        Includes both Newton buffers and nested LinearSolver buffers.
        """
        return buffer_registry.shared_buffer_size(self)

    @property
    def local_buffer_size(self) -> int:
        """Return total local memory elements required.

        Includes both Newton buffers and nested LinearSolver buffers.
        """
        return buffer_registry.local_buffer_size(self)

    @property
    def persistent_local_buffer_size(self) -> int:
        """Return total persistent local memory elements required.

        Includes both Newton buffers and nested LinearSolver buffers.
        """
        return buffer_registry.persistent_local_buffer_size(self)

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return merged Newton and linear solver configuration.

        Combines Newton-level settings from compile_settings with
        linear solver settings from nested linear_solver instance.

        Returns
        -------
        dict
            Merged configuration dictionary containing both Newton
            parameters and linear solver parameters
        """
        combined = dict(self.linear_solver.settings_dict)
        combined.update(self.compile_settings.settings_dict)
        return combined
