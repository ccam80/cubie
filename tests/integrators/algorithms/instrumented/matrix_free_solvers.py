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

        # Get scaled norm device function from config
        scaled_norm_fn = config.norm_device_function

        # Convert types for device function
        n_val = int32(n)
        max_iters_val = int32(max_linear_iters)
        precision_numba = from_dtype(np.dtype(precision))
        typed_zero = precision_numba(0.0)
        typed_one = precision_numba(1.0)
        
        # Get allocators from buffer_registry using production buffer names
        # (registered by parent LinearSolver.register_buffers)
        get_alloc = buffer_registry.get_allocator
        alloc_precond = get_alloc('preconditioned_vec', self)
        alloc_temp = get_alloc('temp', self)
        
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
                persistent_local,
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
                preconditioned_vec = alloc_precond(shared, persistent_local)
                temp = alloc_temp(shared, persistent_local)
                
                # Evaluate operator and compute initial residual
                operator_apply(
                    state, parameters, drivers, cached_aux, base_state,
                    t, h, a_ij, x, temp
                )
                for i in range(n_val):
                    rhs[i] = rhs[i] - temp[i]
                acc = scaled_norm_fn(rhs, x)
                mask = activemask()
                converged = acc <= typed_one
                
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

                    if denominator != typed_zero:
                        alpha = numerator / denominator
                    else:
                        alpha = typed_zero

                    if not converged:
                        for i in range(n_val):
                            x[i] += alpha * preconditioned_vec[i]
                            rhs[i] -= alpha * temp[i]
                    acc = scaled_norm_fn(rhs, x)

                    converged = converged or (acc <= typed_one)
                    
                    # Log iteration state (uses 0-based indexing)
                    log_iter = iteration - int32(1)
                    for i in range(n_val):
                        linear_iteration_guesses[log_slot, log_iter, i] = x[i]
                        linear_residuals[log_slot, log_iter, i] = rhs[i]
                        linear_preconditioned_vectors[
                            log_slot, log_iter, i
                        ] = preconditioned_vec[i]
                    linear_squared_norms[log_slot, log_iter] = acc
                
                # Log "exceeded linear iters" status if still not converged
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
                persistent_local,
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
                preconditioned_vec = alloc_precond(shared, persistent_local)
                temp = alloc_temp(shared, persistent_local)
                
                # Evaluate operator and compute initial residual
                operator_apply(
                    state, parameters, drivers, base_state, t, h, a_ij, x,
                        temp
                )
                for i in range(n_val):
                    rhs[i] = rhs[i] - temp[i]
                acc = scaled_norm_fn(rhs, x)
                mask = activemask()
                converged = acc <= typed_one
                
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
                    
                    for i in range(n_val):
                        x[i] += alpha_effective * preconditioned_vec[i]
                        rhs[i] -= alpha_effective * temp[i]
                    acc = scaled_norm_fn(rhs, x)
                    converged = converged or (acc <= typed_one)
                    
                    # Log iteration state (uses 0-based indexing)
                    log_iter = iteration - int32(1)
                    for i in range(n_val):
                        linear_iteration_guesses[log_slot, log_iter, i] = x[i]
                        linear_residuals[log_slot, log_iter, i] = rhs[i]
                        linear_preconditioned_vectors[
                            log_slot, log_iter, i
                        ] = preconditioned_vec[i]
                    linear_squared_norms[log_slot, log_iter] = acc
                
                # Log "exceeded linear iters" status if still not converged
                final_status = selp(converged, int32(0), int32(4))
                krylov_iters_out[0] = iteration
                return final_status
            
            # no cover: end
            return InstrumentedLinearSolverCache(linear_solver=linear_solver)

    @property
    def device_function(self) -> Callable:
        return self.get_cached_output('linear_solver')

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
        linear_solver_fn = config.linear_solver_function

        n = config.n
        tolerance = config.newton_tolerance
        max_iters = config.max_newton_iters
        damping = config.newton_damping
        max_backtracks = config.newton_max_backtracks
        precision = config.precision

        # Get scaled norm device function from config
        scaled_norm_fn = config.norm_device_function

        # Convert types for device function
        precision_dtype = np.dtype(precision)
        numba_precision = from_dtype(precision_dtype)
        typed_zero = numba_precision(0.0)
        typed_one = numba_precision(1.0)
        typed_damping = numba_precision(damping)
        n_val = int32(n)
        max_iters_val = int32(max_iters)
        max_backtracks_val = int32(max_backtracks + 1)
        
        # Get allocators from buffer_registry using production buffer names
        # (registered by parent NewtonKrylov.register_buffers)
        alloc_delta = buffer_registry.get_allocator('delta', self)
        alloc_residual = buffer_registry.get_allocator('residual', self)
        alloc_residual_temp = buffer_registry.get_allocator(
            'residual_temp', self
        )
        alloc_stage_base_bt = buffer_registry.get_allocator(
            'stage_base_bt', self
        )
        alloc_krylov_iters_local = buffer_registry.get_allocator(
            'krylov_iters_local', self
        )
        
        # Get child allocators for linear solver (same pattern as production)
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
            delta = alloc_delta(shared_scratch, persistent_scratch)
            residual = alloc_residual(shared_scratch, persistent_scratch)
            residual_temp = alloc_residual_temp(shared_scratch, persistent_scratch)
            stage_base_bt = alloc_stage_base_bt(shared_scratch, persistent_scratch)
            lin_shared = alloc_lin_shared(shared_scratch, persistent_scratch)
            lin_persistent = alloc_lin_persistent(shared_scratch, persistent_scratch)
            
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
            
            linear_slot_base = int32(stage_index * max_iters_val)
            log_index = int32(0)
            residual_copy = cuda.local.array(n, numba_precision)

            # Compute norm BEFORE negation for correct scaling
            norm2_prev = scaled_norm_fn(residual, stage_increment)

            for i in range(n_val):
                residual_value = residual[i]
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
            
            converged = norm2_prev <= typed_one
            has_error = False
            final_status = int32(0)
            
            krylov_iters_local = alloc_krylov_iters_local(
                shared_scratch, persistent_scratch
            )
            
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
                        
                        norm2_new = scaled_norm_fn(residual_temp, stage_increment)
                        for i in range(n_val):
                            stage_increment_snapshot[i] = stage_increment[i]
                            residual_snapshot[i] = residual_temp[i]
                        snapshot_ready = True
                        
                        if norm2_new <= typed_one:
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
                    # Revert stage increment for another round
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

    @property
    def device_function(self) -> Callable:
        return self.get_cached_output('newton_krylov_solver')


__all__ = [
    "InstrumentedLinearSolver",
    "InstrumentedLinearSolverCache",
    "InstrumentedNewtonKrylov",
    "InstrumentedNewtonKrylovCache",
]

