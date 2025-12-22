"""Instrumented matrix-free solver factories for CUDA device kernels."""

import attrs
from typing import Callable

import numpy as np
from numba import cuda, int32, from_dtype

from cubie._utils import is_device_validator
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import CUDAFunctionCache
from cubie.cuda_simsafe import (
    activemask, all_sync, selp, any_sync, compile_kwargs
)
from cubie.integrators.matrix_free_solvers.linear_solver import (
    LinearSolver,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
)


@attrs.define
class InstrumentedLinearSolverCache(CUDAFunctionCache):
    """Cache container for InstrumentedLinearSolver outputs.
    
    Attributes
    ----------
    linear_solver : Callable
        Compiled CUDA device function with logging signature.
    """
    
    linear_solver: Callable = attrs.field(
        validator=is_device_validator
    )


class InstrumentedLinearSolver(LinearSolver):
    """Factory for instrumented linear solver device functions.
    
    Inherits from LinearSolver and adds iteration logging to device function.
    Logging arrays are passed as device function parameters and populated
    during iteration. Uses buffer_registry for production buffers
    (preconditioned_vec, temp) but logging arrays are caller-allocated.
    """
    
    def register_buffers(self) -> None:
        """Register device buffers with lin_ prefix for instrumented solver."""
        config = self.compile_settings
        use_cached = config.use_cached_auxiliaries
        
        if use_cached:
            buffer_registry.register(
                'lin_cached_preconditioned_vec',
                self,
                config.n,
                config.preconditioned_vec_location,
                precision=config.precision
            )
            buffer_registry.register(
                'lin_cached_temp',
                self,
                config.n,
                config.temp_location,
                precision=config.precision
            )
        else:
            buffer_registry.register(
                'lin_preconditioned_vec',
                self,
                config.n,
                config.preconditioned_vec_location,
                precision=config.precision
            )
            buffer_registry.register(
                'lin_temp',
                self,
                config.n,
                config.temp_location,
                precision=config.precision
            )
    
    def build(self) -> InstrumentedLinearSolverCache:
        """Compile instrumented linear solver device function.
        
        Returns
        -------
        InstrumentedLinearSolverCache
            Container with compiled linear_solver device function including
            logging parameters.
        
        Logging Parameters (added to device function signature)
        ---------------------------------------------------
        slot_index : int32
            Index into first dimension of logging arrays.
        linear_initial_guesses : array[num_slots, n]
            Records initial guess x values.
        linear_iteration_guesses : array[num_slots, max_iters, n]
            Records x values at each iteration.
        linear_residuals : array[num_slots, max_iters, n]
            Records residual values at each iteration.
        linear_squared_norms : array[num_slots, max_iters]
            Records squared residual norms at each iteration.
        linear_preconditioned_vectors : array[num_slots, max_iters, n]
            Records preconditioned search direction at each iteration.
        
        Raises
        ------
        ValueError
            If operator_apply is None when build() is called.
        """
        config = self.compile_settings
        
        # Validate required device functions are set
        if config.operator_apply is None:
            raise ValueError(
                "operator_apply must be set before building "
                "InstrumentedLinearSolver"
            )
        
        # Extract parameters from config
        operator_apply = config.operator_apply
        preconditioner = config.preconditioner
        n = config.n
        correction_type = config.correction_type
        tolerance = config.krylov_tolerance
        max_iters = config.max_linear_iters
        precision = config.precision
        use_cached_auxiliaries = config.use_cached_auxiliaries
        
        # Compute flags for correction type
        sd_flag = correction_type == "steepest_descent"
        mr_flag = correction_type == "minimal_residual"
        preconditioned = preconditioner is not None
        
        # Convert types for device function
        n_val = int32(n)
        max_iters_val = int32(max_iters)
        precision_numba = from_dtype(np.dtype(precision))
        typed_zero = precision_numba(0.0)
        tol_squared = precision_numba(tolerance * tolerance)
        
        # Get allocators from buffer_registry
        if use_cached_auxiliaries:
            alloc_precond = buffer_registry.get_allocator(
                'lin_cached_preconditioned_vec', self
            )
            alloc_temp = buffer_registry.get_allocator(
                'lin_cached_temp', self
            )
        else:
            alloc_precond = buffer_registry.get_allocator(
                'lin_preconditioned_vec', self
            )
            alloc_temp = buffer_registry.get_allocator('lin_temp', self)
        
        # Branch on use_cached_auxiliaries flag
        if use_cached_auxiliaries:
            # Device function for cached auxiliaries variant with logging
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
                krylov_iters_out,
                # Logging parameters:
                slot_index,
                linear_initial_guesses,
                linear_iteration_guesses,
                linear_residuals,
                linear_squared_norms,
                linear_preconditioned_vectors,
            ):
                """Run one cached preconditioned solve with logging."""
                
                # Allocate buffers from registry
                preconditioned_vec = alloc_precond(shared, shared)
                temp = alloc_temp(shared, shared)
                
                # Evaluate operator and compute initial residual
                operator_apply(
                    state, parameters, drivers, cached_aux, base_state,
                    t, h, a_ij, x, temp
                )
                acc = typed_zero
                for i in range(n_val):
                    residual_value = rhs[i] - temp[i]
                    rhs[i] = residual_value
                    acc += residual_value * residual_value
                mask = activemask()
                converged = acc <= tol_squared
                
                # Log initial guess
                log_slot = int32(slot_index)
                for i in range(n_val):
                    linear_initial_guesses[log_slot, i] = x[i]
                
                iteration = int32(0)
                for _ in range(max_iters_val):
                    if all_sync(mask, converged):
                        break
                    
                    iteration += int32(1)
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
                    
                    alpha = selp(
                        denominator != typed_zero,
                        numerator / denominator,
                        typed_zero,
                    )
                    alpha_effective = selp(
                        converged, precision_numba(0.0), alpha
                    )
                    
                    acc = typed_zero
                    for i in range(n_val):
                        x[i] += alpha_effective * preconditioned_vec[i]
                        rhs[i] -= alpha_effective * temp[i]
                        residual_value = rhs[i]
                        acc += residual_value * residual_value
                    converged = converged or (acc <= tol_squared)
                    
                    # Log iteration state (uses 0-based indexing)
                    log_iter = iteration - int32(1)
                    for i in range(n_val):
                        linear_iteration_guesses[log_slot, log_iter, i] = x[i]
                        linear_residuals[log_slot, log_iter, i] = rhs[i]
                        linear_preconditioned_vectors[
                            log_slot, log_iter, i
                        ] = preconditioned_vec[i]
                    linear_squared_norms[log_slot, log_iter] = acc
                
                # Single exit point - status based on converged flag
                final_status = selp(converged, int32(0), int32(4))
                krylov_iters_out[0] = iteration
                return final_status
            
            # no cover: end
            return InstrumentedLinearSolverCache(
                linear_solver=linear_solver_cached
            )
        else:
            # Device function for non-cached variant with logging
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
                krylov_iters_out,
                # Logging parameters:
                slot_index,
                linear_initial_guesses,
                linear_iteration_guesses,
                linear_residuals,
                linear_squared_norms,
                linear_preconditioned_vectors,
            ):
                """Run one preconditioned solve with logging."""
                
                # Allocate buffers from registry
                preconditioned_vec = alloc_precond(shared, shared)
                temp = alloc_temp(shared, shared)
                
                # Evaluate operator and compute initial residual
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
                
                # Log initial guess
                log_slot = int32(slot_index)
                for i in range(n_val):
                    linear_initial_guesses[log_slot, i] = x[i]
                
                iteration = int32(0)
                for _ in range(max_iters_val):
                    if all_sync(mask, converged):
                        break
                    
                    iteration += int32(1)
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
                    alpha_effective = selp(
                        converged, precision_numba(0.0), alpha
                    )
                    
                    acc = typed_zero
                    for i in range(n_val):
                        x[i] += alpha_effective * preconditioned_vec[i]
                        rhs[i] -= alpha_effective * temp[i]
                        residual_value = rhs[i]
                        acc += residual_value * residual_value
                    converged = converged or (acc <= tol_squared)
                    
                    # Log iteration state (uses 0-based indexing)
                    log_iter = iteration - int32(1)
                    for i in range(n_val):
                        linear_iteration_guesses[log_slot, log_iter, i] = x[i]
                        linear_residuals[log_slot, log_iter, i] = rhs[i]
                        linear_preconditioned_vectors[
                            log_slot, log_iter, i
                        ] = preconditioned_vec[i]
                    linear_squared_norms[log_slot, log_iter] = acc
                
                # Single exit point - status based on converged flag
                final_status = selp(converged, int32(0), int32(4))
                krylov_iters_out[0] = iteration
                return final_status
            
            # no cover: end
            return InstrumentedLinearSolverCache(linear_solver=linear_solver)


@attrs.define
class InstrumentedNewtonKrylovCache(CUDAFunctionCache):
    """Cache container for InstrumentedNewtonKrylov outputs.
    
    Attributes
    ----------
    newton_krylov_solver : Callable
        Compiled CUDA device function with logging signature.
    """
    
    newton_krylov_solver: Callable = attrs.field(
        validator=is_device_validator
    )


class InstrumentedNewtonKrylov(NewtonKrylov):
    """Factory for instrumented Newton-Krylov solver device functions.
    
    Inherits from NewtonKrylov and adds iteration logging to device function.
    Logging arrays are passed as device function parameters and populated
    during Newton iteration. Embeds InstrumentedLinearSolver for nested
    linear solve logging.
    """
    
    def register_buffers(self) -> None:
        """Register device buffers with newton_ prefix for instrumented solver."""
        config = self.compile_settings
        precision = config.precision
        n = config.n
        
        buffer_registry.register(
            'newton_delta',
            self,
            n,
            config.delta_location,
            precision=precision
        )
        buffer_registry.register(
            'newton_residual',
            self,
            n,
            config.residual_location,
            precision=precision
        )
        buffer_registry.register(
            'newton_residual_temp',
            self,
            n,
            config.residual_temp_location,
            precision=precision
        )
        buffer_registry.register(
            'newton_stage_base_bt',
            self,
            n,
            config.stage_base_bt_location,
            precision=precision
        )
    
    def build(self) -> InstrumentedNewtonKrylovCache:
        """Compile instrumented Newton-Krylov solver device function.
        
        Returns
        -------
        InstrumentedNewtonKrylovCache
            Container with compiled newton_krylov_solver device function
            including logging parameters.
        
        Logging Parameters (added to device function signature)
        ---------------------------------------------------
        stage_index : int32
            Index into first dimension of Newton logging arrays.
        newton_initial_guesses : array[num_stages, n]
            Records initial guess values for each stage.
        newton_iteration_guesses : array[num_stages, max_iters+1, n]
            Records state values at each Newton iteration.
        newton_residuals : array[num_stages, max_iters+1, n]
            Records residual values at each Newton iteration.
        newton_squared_norms : array[num_stages, max_iters+1]
            Records squared residual norms at each Newton iteration.
        newton_iteration_scale : array[num_stages, max_iters]
            Records alpha scaling factor at each Newton iteration.
        linear_initial_guesses : array[total_linear_slots, n]
            Records initial guess x values for embedded linear solves.
        linear_iteration_guesses : array[total_linear_slots, max_linear_iters, n]
            Records x values at each linear solver iteration.
        linear_residuals : array[total_linear_slots, max_linear_iters, n]
            Records residual values at each linear solver iteration.
        linear_squared_norms : array[total_linear_slots, max_linear_iters]
            Records squared residual norms at each linear solver iteration.
        linear_preconditioned_vectors : array[total_linear_slots, max_linear_iters, n]
            Records preconditioned search direction at each linear iteration.
        
        Raises
        ------
        ValueError
            If residual_function or linear_solver is None when build() is
            called.
        TypeError
            If linear_solver is not InstrumentedLinearSolver instance.
        """
        config = self.compile_settings
        
        # Extract parameters from config
        residual_function = config.residual_function
        linear_solver = self.linear_solver
        n = config.n
        tolerance = config.newton_tolerance
        max_iters = config.max_newton_iters
        damping = config.newton_damping
        max_backtracks = config.newton_max_backtracks
        precision = config.precision
        
        # Get linear solver device function (triggers build() if needed)
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
        alloc_residual = buffer_registry.get_allocator(
            'newton_residual', self
        )
        alloc_residual_temp = buffer_registry.get_allocator(
            'newton_residual_temp', self
        )
        alloc_stage_base_bt = buffer_registry.get_allocator(
            'newton_stage_base_bt', self
        )
        
        # Compute offset for linear solver shared buffers
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
            # Logging parameters:
            stage_index,
            newton_initial_guesses,
            newton_iteration_guesses,
            newton_residuals,
            newton_squared_norms,
            newton_iteration_scale,
            linear_initial_guesses,
            linear_iteration_guesses,
            linear_residuals,
            linear_squared_norms,
            linear_preconditioned_vectors,
        ):
            """Solve a nonlinear system with damped Newton-Krylov and logging."""
            
            # Allocate buffers from registry
            delta = alloc_delta(shared_scratch, shared_scratch)
            residual = alloc_residual(shared_scratch, shared_scratch)
            residual_temp = alloc_residual_temp(shared_scratch, shared_scratch)
            stage_base_bt = alloc_stage_base_bt(shared_scratch, shared_scratch)
            
            # Initialize local arrays
            for _i in range(n_val):
                delta[_i] = typed_zero
                residual[_i] = typed_zero
            
            # Evaluate initial residual
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
            linear_slot_base = int32(stage_index * max_iters_val)
            log_index = int32(0)
            residual_copy = cuda.local.array(n, numba_precision)
            for i in range(n_val):
                residual_value = residual[i]
                norm2_prev += residual_value * residual_value
                delta[i] = typed_zero
                residual[i] = -residual_value
                residual_copy[i] = residual_value
                newton_initial_guesses[stage_index, i] = stage_increment[i]
            
            # Log first iteration (initial state)
            for i in range(n_val):
                newton_iteration_guesses[stage_index, log_index, i] = (
                    stage_increment[i]
                )
                newton_residuals[stage_index, log_index, i] = residual_copy[i]
            newton_squared_norms[stage_index, log_index] = norm2_prev
            log_index += int32(1)
            
            converged = norm2_prev <= tol_squared
            has_error = False
            final_status = int32(0)
            
            krylov_iters_local = cuda.local.array(1, int32)
            
            iters_count = int32(0)
            total_krylov_iters = int32(0)
            mask = activemask()
            stage_increment_snapshot = cuda.local.array(n, numba_precision)
            residual_snapshot = cuda.local.array(n, numba_precision)
            
            for _ in range(max_iters_val):
                done = converged or has_error
                if all_sync(mask, done):
                    break
                
                active = not done
                iters_count = selp(
                    active, int32(iters_count + int32(1)), iters_count
                )
                
                iter_slot = int(iters_count) - 1
                
                # Linear solver uses remaining shared space after Newton buffers
                lin_shared = shared_scratch[lin_shared_offset:]
                krylov_iters_local[0] = int32(0)
                # Compute flat index: slot_index * max_iters + iteration
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
                    # Logging parameters for linear solver:
                    linear_slot_base + iter_slot,
                    linear_initial_guesses,
                    linear_iteration_guesses,
                    linear_residuals,
                    linear_squared_norms,
                    linear_preconditioned_vectors,
                )
                
                lin_failed = lin_status != int32(0)
                has_error = has_error or lin_failed
                final_status = selp(
                    lin_failed, int32(final_status | lin_status), final_status
                )
                total_krylov_iters += selp(
                    active, krylov_iters_local[0], int32(0)
                )
                
                for i in range(n_val):
                    stage_base_bt[i] = stage_increment[i]
                
                alpha = typed_one
                found_step = False
                snapshot_ready = False
                
                for _ in range(max_backtracks_val):
                    active_bt = active and (not found_step) and (not converged)
                    if not any_sync(mask, active_bt):
                        break
                    
                    if active_bt:
                        for i in range(n_val):
                            stage_increment[i] = (
                                stage_base_bt[i] + alpha * delta[i]
                            )
                        
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
                            stage_increment_snapshot[i] = stage_increment[i]
                            residual_snapshot[i] = residual_value
                        snapshot_ready = True
                        
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
                    backtrack_failed,
                    int32(final_status | int32(1)),
                    final_status
                )
                
                if backtrack_failed:
                    for i in range(n_val):
                        stage_increment[i] = stage_base_bt[i]
                
                # Log iteration state if snapshot is ready
                iter_slot = int(iters_count) - 1
                if iter_slot >= 0:
                    if snapshot_ready:
                        for i in range(n_val):
                            newton_iteration_guesses[
                                stage_index, log_index, i
                            ] = stage_increment_snapshot[i]
                            newton_residuals[
                                stage_index, log_index, i
                            ] = residual_snapshot[i]
                        newton_squared_norms[stage_index, log_index] = norm2_new
                        log_index += int32(1)
                    newton_iteration_scale[stage_index, iter_slot] = alpha
            
            max_iters_exceeded = (not converged) and (not has_error)
            final_status = selp(
                max_iters_exceeded,
                int32(final_status | int32(2)),
                final_status
            )
            
            counters[0] = iters_count
            counters[1] = total_krylov_iters
            
            return final_status
        
        # no cover: end
        return InstrumentedNewtonKrylovCache(
            newton_krylov_solver=newton_krylov_solver
        )


__all__ = [
    "InstrumentedLinearSolver",
    "InstrumentedLinearSolverCache",
    "InstrumentedNewtonKrylov",
    "InstrumentedNewtonKrylovCache",
]

