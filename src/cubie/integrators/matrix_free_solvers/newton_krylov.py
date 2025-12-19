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
import numba

from cubie._utils import (
    ALLOWED_PRECISIONS,
    PrecisionDType,
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
    linear_solver : Optional[LinearSolver]
        LinearSolver instance for solving linear systems.
    _tolerance : float
        Residual norm threshold for convergence.
    max_iters : int
        Maximum Newton iterations permitted.
    _damping : float
        Step shrink factor for backtracking.
    max_backtracks : int
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
        validator=validators.optional(is_device_validator)
    )
    linear_solver: Optional['LinearSolver'] = attrs.field(
        default=None,
        validator=validators.optional(
            validators.instance_of(LinearSolver)
        )
    )
    _tolerance: float = attrs.field(
        default=1e-3,
        validator=gttype_validator(float, 0)
    )
    max_iters: int = attrs.field(
        default=100,
        validator=inrangetype_validator(int, 1, 32767)
    )
    _damping: float = attrs.field(
        default=0.5,
        validator=inrangetype_validator(float, 0, 1)
    )
    max_backtracks: int = attrs.field(
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
    
    def __attrs_post_init__(self):
        """Validate precision consistency with linear_solver."""
        if self.linear_solver is not None:
            if self.linear_solver.precision != self.precision:
                raise ValueError(
                    f"NewtonKrylov precision ({self.precision}) must match "
                    f"LinearSolver precision ({self.linear_solver.precision})"
                )
    
    @property
    def tolerance(self) -> float:
        """Return tolerance in configured precision."""
        return self.precision(self._tolerance)
    
    @property
    def damping(self) -> float:
        """Return damping factor in configured precision."""
        return self.precision(self._damping)
    
    @property
    def numba_precision(self) -> type:
        """Return Numba type for precision."""
        return from_dtype(np.dtype(self.precision))
    
    @property
    def simsafe_precision(self) -> type:
        """Return CUDA-sim-safe type for precision."""
        return simsafe_dtype(np.dtype(self.precision))
    
    @property
    def settings_dict(self) -> dict:
        """Return configuration fields as dictionary."""
        return {
            'precision': self.precision,
            'n': self.n,
            'residual_function': self.residual_function,
            'linear_solver': self.linear_solver,
            'tolerance': self.tolerance,
            'max_iters': self.max_iters,
            'damping': self.damping,
            'max_backtracks': self.max_backtracks,
            'delta_location': self.delta_location,
            'residual_location': self.residual_location,
            'residual_temp_location': self.residual_temp_location,
            'stage_base_bt_location': self.stage_base_bt_location,
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
    
    def __init__(self, config: NewtonKrylovConfig) -> None:
        """Initialize NewtonKrylov with configuration.
        
        Parameters
        ----------
        config : NewtonKrylovConfig
            Configuration containing all compile-time parameters.
        """
        super().__init__()
        self.setup_compile_settings(config)
        
        # Register buffers with buffer_registry
        buffer_registry.register(
            'newton_delta',
            self,
            config.n,
            config.delta_location,
            precision=config.precision
        )
        buffer_registry.register(
            'newton_residual',
            self,
            config.n,
            config.residual_location,
            precision=config.precision
        )
        buffer_registry.register(
            'newton_residual_temp',
            self,
            config.n,
            config.residual_temp_location,
            precision=config.precision
        )
        buffer_registry.register(
            'newton_stage_base_bt',
            self,
            config.n,
            config.stage_base_bt_location,
            precision=config.precision
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
        
        # Validate required components are set
        if config.residual_function is None:
            raise ValueError(
                "residual_function must be set before building NewtonKrylov"
            )
        if config.linear_solver is None:
            raise ValueError(
                "linear_solver must be set before building NewtonKrylov"
            )
        
        # Extract parameters from config
        residual_function = config.residual_function
        linear_solver = config.linear_solver
        n = config.n
        tolerance = config.tolerance
        max_iters = config.max_iters
        damping = config.damping
        max_backtracks = config.max_backtracks
        precision = config.precision
        
        # Get linear solver device function (may trigger LinearSolver.build())
        linear_solver_fn = linear_solver.device_function
        
        # Convert types for device function
        precision_dtype = np.dtype(precision)
        numba_precision = from_dtype(precision_dtype)
        tol_squared = numba_precision(tolerance * tolerance)
        typed_zero = numba_precision(0.0)
        typed_one = numba_precision(1.0)
        typed_damping = numba_precision(damping)
        n_val = int32(n)
        max_iters_val = int32(max_iters)
        max_backtracks_val = int32(max_backtracks + 1)
        
        # Get allocators from buffer_registry
        alloc_delta = buffer_registry.get_allocator('newton_delta', self)
        alloc_residual = buffer_registry.get_allocator('newton_residual', self)
        alloc_residual_temp = buffer_registry.get_allocator(
            'newton_residual_temp', self
        )
        alloc_stage_base_bt = buffer_registry.get_allocator(
            'newton_stage_base_bt', self
        )
        
        # Compute offset for linear solver shared buffers.
        # NewtonKrylov registers its buffers first (delta, residual,
        # residual_temp, stage_base_bt), so shared_buffer_size(self) returns
        # the total size of Newton buffers. LinearSolver buffers start
        # immediately after.
        lin_shared_offset = buffer_registry.shared_buffer_size(self)
        
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
            
            # Allocate buffers from registry
            delta = alloc_delta(shared_scratch, shared_scratch)
            residual = alloc_residual(shared_scratch, shared_scratch)
            residual_temp = alloc_residual_temp(shared_scratch, shared_scratch)
            stage_base_bt = alloc_stage_base_bt(shared_scratch, shared_scratch)
            
            # Initialize local arrays
            for _i in range(n_val):
                delta[_i] = typed_zero
                residual[_i] = typed_zero
            
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
            for _ in range(max_iters_val):
                done = converged or has_error
                if all_sync(mask, done):
                    break
                
                active = not done
                iters_count = selp(
                    active, int32(iters_count + int32(1)), iters_count
                )
                
                # Linear solver uses remaining shared space after Newton buffers
                lin_shared = shared_scratch[lin_shared_offset:]
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
        
        Notes
        -----
        If linear_solver is updated, cache is invalidated even if
        the LinearSolver instance reference hasn't changed, because
        the LinearSolver's internal state may have changed.
        """
        return self.update_compile_settings(
            updates_dict=updates_dict,
            silent=silent,
            **kwargs
        )
    
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
    def tolerance(self) -> float:
        """Return convergence tolerance."""
        return self.compile_settings.tolerance
    
    @property
    def max_iters(self) -> int:
        """Return maximum Newton iterations."""
        return self.compile_settings.max_iters
    
    @property
    def damping(self) -> float:
        """Return damping factor."""
        return self.compile_settings.damping
    
    @property
    def max_backtracks(self) -> int:
        """Return maximum backtracking steps."""
        return self.compile_settings.max_backtracks
    
    @property
    def linear_solver(self) -> Optional['LinearSolver']:
        """Return nested LinearSolver instance."""
        return self.compile_settings.linear_solver
    
    @property
    def shared_buffer_size(self) -> int:
        """Return total shared memory elements required.
        
        Includes both Newton buffers and nested LinearSolver buffers.
        """
        newton_size = buffer_registry.shared_buffer_size(self)
        if self.compile_settings.linear_solver is not None:
            linear_size = self.compile_settings.linear_solver.shared_buffer_size
        else:
            linear_size = 0
        return newton_size + linear_size
    
    @property
    def local_buffer_size(self) -> int:
        """Return total local memory elements required.
        
        Includes both Newton buffers and nested LinearSolver buffers.
        """
        newton_size = buffer_registry.local_buffer_size(self)
        if self.compile_settings.linear_solver is not None:
            linear_size = self.compile_settings.linear_solver.local_buffer_size
        else:
            linear_size = 0
        return newton_size + linear_size
