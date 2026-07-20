"""Instrumented matrix-free solver factories for CUDA device kernels."""

import attrs
from math import sqrt as math_sqrt
from typing import Callable

import numpy as np
from cubie.cuda_simsafe import cuda, int32, numba_from_dtype as from_dtype

from cubie._utils import is_device_validator
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import CUDADispatcherCache
from cubie.cuda_simsafe import (
    activemask, all_sync, selp, compile_kwargs
)
from cubie.result_codes import CUBIE_RESULT_CODES
from cubie.integrators.matrix_free_solvers.linear_solver import (
    MRLinearSolver,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
)
from cubie.integrators.norms import ScaledNorm


@attrs.define
class InstrumentedMRLinearSolverCache(CUDADispatcherCache):
    """Cache container for InstrumentedMRLinearSolver outputs.
    
    Attributes
    ----------
    linear_solver : Callable
        Compiled CUDA device function with logging signature.
    """
    
    linear_solver: Callable = attrs.field(
        validator=is_device_validator
    )


class InstrumentedMRLinearSolver(MRLinearSolver):
    """Factory for instrumented linear solver device functions.
    
    Inherits from MRLinearSolver and adds iteration logging to device function.
    Logging arrays are passed as device function parameters and populated
    during iteration. Uses buffer_registry for production buffers
    (preconditioned_vec, temp) but logging arrays are caller-allocated.
    """
    
    def build(self) -> InstrumentedMRLinearSolverCache:
        """Compile instrumented linear solver device function.
        
        Returns
        -------
        InstrumentedMRLinearSolverCache
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
        max_iters = config.max_iters
        precision = config.precision
        use_cached_auxiliaries = config.use_cached_auxiliaries
        
        # Compute flags for correction type
        sd_flag = linear_correction_type == "steepest_descent"
        mr_flag = linear_correction_type == "minimal_residual"
        preconditioned = preconditioner is not None
        chained_precond = config.preconditioner_is_chained

        # Get scaled norm device function from config
        scaled_norm_fn = config.norm_device_function

        # Convert types for device function
        n_val = int32(n)
        max_iters_val = int32(max_iters)
        precision_numba = from_dtype(np.dtype(precision))
        typed_zero = precision_numba(0.0)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        max_linear_iters_exceeded = int32(
            CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED
        )
        typed_one = precision_numba(1.0)
        
        # Get allocators from buffer_registry using production buffer names
        # (registered by parent MRLinearSolver.register_buffers)
        get_alloc = buffer_registry.get_allocator
        alloc_precond = get_alloc('preconditioned_vec', self)
        alloc_temp = get_alloc('temp', self)
        alloc_precond_scratch = get_alloc('mr_precond_scratch', self)
        alloc_chain_scratch = get_alloc('mr_chain_scratch', self)
        
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
                precond_scratch = alloc_precond_scratch(
                    shared, persistent_local
                )
                if chained_precond:
                    chain_scratch = alloc_chain_scratch(
                        shared, persistent_local
                    )
                else:
                    chain_scratch = precond_scratch

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
                        if chained_precond:
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
                                precond_scratch,
                                chain_scratch,
                            )
                        else:
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
                                precond_scratch,
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
                final_status = selp(
                    converged, success, max_linear_iters_exceeded
                )
                krylov_iters_out[0] = iteration
                return final_status
            
            # no cover: end
            return InstrumentedMRLinearSolverCache(
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
                precond_scratch = alloc_precond_scratch(
                    shared, persistent_local
                )
                if chained_precond:
                    chain_scratch = alloc_chain_scratch(
                        shared, persistent_local
                    )
                else:
                    chain_scratch = precond_scratch

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
                        if chained_precond:
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
                                precond_scratch,
                                chain_scratch,
                            )
                        else:
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
                                precond_scratch,
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
                final_status = selp(
                    converged, success, max_linear_iters_exceeded
                )
                krylov_iters_out[0] = iteration
                return final_status
            
            # no cover: end
            return InstrumentedMRLinearSolverCache(linear_solver=linear_solver)

    @property
    def device_function(self) -> Callable:
        return self.get_cached_output('linear_solver')

@attrs.define
class InstrumentedNewtonKrylovCache(CUDADispatcherCache):
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
    """Build a Newton solver that records its iterations."""
    
    def build(self) -> InstrumentedNewtonKrylovCache:
        """Compile the instrumented Newton solver.

        Returns
        -------
        InstrumentedNewtonKrylovCache
            Compiled device function with logging arguments.
        """
        config = self.compile_settings

        # Extract parameters from config
        residual_function = config.residual_function
        linear_solver_fn = config.linear_solver_function

        n = config.n
        max_iters = config.max_iters
        precision = config.precision

        # The production config carries only the correction norm; the
        # residual norm logged for parity is built here from the same
        # tolerances.
        correction_norm_fn = config.norm_device_function
        scaled_norm_fn = ScaledNorm(
            precision=precision,
            n=n,
            atol=self.norm.atol,
            rtol=self.norm.rtol,
        ).device_function

        # Convert types for device function
        precision_dtype = np.dtype(precision)
        numba_precision = from_dtype(precision_dtype)
        typed_zero = numba_precision(0.0)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        max_newton_iters_exceeded = int32(
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED
        )
        newton_divergence = int32(CUBIE_RESULT_CODES.NEWTON_DIVERGENCE)
        typed_one = numba_precision(1.0)
        typed_huge = numba_precision(float(np.finfo(precision_dtype).max))
        kappa = numba_precision(0.01)
        first_iteration_bound = numba_precision(1.0e-5)
        theta_decay = numba_precision(0.3)
        theta_divergence_bound = numba_precision(2.0)
        stagnation_eps = numba_precision(
            100.0 * math_sqrt(float(np.finfo(precision_dtype).eps))
        )
        n_val = int32(n)
        max_iters_val = int32(max_iters)

        # NewtonKrylov registers these buffers.
        alloc_delta = buffer_registry.get_allocator('delta', self)
        alloc_residual = buffer_registry.get_allocator('residual', self)
        alloc_krylov_iters_local = buffer_registry.get_allocator(
            'krylov_iters_local', self
        )
        alloc_prev_theta = buffer_registry.get_allocator(
            'prev_theta', self
        )
        # Allocate the linear solver's buffers.
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
            step_start,
            shared_scratch,
            persistent_scratch,
            counters,
            # Logging parameters:
            stage_index,
            newton_initial_guesses,
            newton_iteration_guesses,
            newton_residuals,
            newton_squared_norms,
            linear_initial_guesses,
            linear_iteration_guesses,
            linear_residuals,
            linear_squared_norms,
            linear_preconditioned_vectors,
        ):
            """Solve a nonlinear system and record each iteration."""

            # Allocate buffers from registry
            delta = alloc_delta(shared_scratch, persistent_scratch)
            residual = alloc_residual(shared_scratch, persistent_scratch)
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
            
            # Warm-started contraction estimate; zero marks a fresh
            # scratch buffer or a failed previous solve. Callers zero
            # the persistent scratch before the first solve.
            prev_theta_store = alloc_prev_theta(
                shared_scratch, persistent_scratch
            )
            stored_theta = prev_theta_store[0]
            prev_theta = selp(
                stored_theta > typed_zero, stored_theta, typed_one
            )

            # RMS norm of the previous accepted full-step correction.
            ndz_prev = typed_zero

            converged = False
            failed = False

            krylov_iters_local = alloc_krylov_iters_local(
                shared_scratch, persistent_scratch
            )

            iters_count = int32(0)
            total_krylov_iters = int32(0)
            iteration = int32(0)
            last_lin_status = success
            mask = activemask()

            for _ in range(max_iters_val):
                if all_sync(mask, converged | failed):
                    break
                iteration += int32(1)
                active = (not converged) & (not failed)

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
                for i in range(n_val):
                    residual[i] = -residual[i]
                    delta[i] = typed_zero

                iter_slot = int32(iteration - int32(1))

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

                total_krylov_iters += selp(
                    active, krylov_iters_local[0], int32(0)
                )
                last_lin_status = selp(
                    active, lin_status, last_lin_status
                )
                iters_count = selp(
                    active, int32(iters_count + int32(1)), iters_count
                )

                norm2_dz = correction_norm_fn(
                    delta,
                    stage_increment,
                    base_state,
                    step_start,
                    a_ij,
                )
                ndz = numba_precision(math_sqrt(norm2_dz))

                judged = active & (lin_status == success)
                history = ndz_prev > typed_zero
                ndz_prev_safe = selp(history, ndz_prev, typed_one)
                theta = selp(
                    history,
                    max(theta_decay * prev_theta, ndz / ndz_prev_safe),
                    prev_theta,
                )
                small_first_step = (iteration == int32(1)) & (
                    ndz < first_iteration_bound
                )
                eta_accept = (theta < typed_one) & (
                    theta * ndz < kappa * (typed_one - theta)
                )

                nonfinite = not (norm2_dz <= typed_huge)
                stagnant = (
                    judged
                    & history
                    & (abs(theta - typed_one) <= stagnation_eps)
                )
                diverging = judged & (
                    (history & (theta > theta_divergence_bound))
                    | nonfinite
                )
                converged_stagnant = (
                    stagnant & (ndz <= typed_one) & (not diverging)
                )
                failed_now = diverging | (
                    stagnant & (ndz > typed_one)
                )
                failed = failed | failed_now

                commit = (
                    judged
                    & (not failed_now)
                    & (not converged_stagnant)
                )
                for i in range(n_val):
                    stage_increment[i] = selp(
                        commit,
                        stage_increment[i] + delta[i],
                        stage_increment[i],
                    )
                converged = (
                    converged
                    | converged_stagnant
                    | (commit & (eta_accept | small_first_step))
                )
                ndz_prev = selp(commit, ndz, typed_zero)
                prev_theta = selp(
                    judged & history, theta, prev_theta
                )

                # Log the committed iterate and its residual.
                if commit:
                    residual_function(
                        stage_increment,
                        parameters,
                        drivers,
                        t,
                        h,
                        a_ij,
                        base_state,
                        residual_copy,
                    )
                    norm2_new = scaled_norm_fn(
                        residual_copy, stage_increment
                    )
                    for i in range(n_val):
                        newton_iteration_guesses[
                            stage_index, log_index, i
                        ] = stage_increment[i]
                        newton_residuals[
                            stage_index, log_index, i
                        ] = residual_copy[i]
                    newton_squared_norms[stage_index, log_index] = norm2_new
                    log_index += int32(1)

            # Persist contraction history for the next solve; a failed
            # solve resets it to the conservative estimate.
            prev_theta_store[0] = selp(converged, prev_theta, typed_one)

            fail_bits = selp(
                failed, newton_divergence, max_newton_iters_exceeded
            )
            fail_bits = selp(
                last_lin_status != success,
                int32(fail_bits | last_lin_status),
                fail_bits,
            )
            final_status = selp(converged, success, fail_bits)

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
    "InstrumentedMRLinearSolver",
    "InstrumentedMRLinearSolverCache",
    "InstrumentedNewtonKrylov",
    "InstrumentedNewtonKrylovCache",
]

