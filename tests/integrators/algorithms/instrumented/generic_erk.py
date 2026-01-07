"""Generic explicit Runge--Kutta integration step with streamed accumulators."""

from typing import Callable, Optional

from numba import cuda, int32

from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import all_sync, activemask
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
)
from cubie.integrators.algorithms.generic_erk import (
    ERKStep)



class InstrumentedERKStep(ERKStep):
    """Generic explicit Runge--Kutta step with configurable tableaus."""

    def build_step(
        self,
        evaluate_f: Callable,
        evaluate_observables: Callable,
        evaluate_driver_at_t: Optional[Callable],
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the explicit Runge--Kutta device step."""

        config = self.compile_settings
        tableau = config.tableau

        typed_zero = numba_precision(0.0)
        n = int32(n)
        stage_count = int32(tableau.stage_count)
        stages_except_first = stage_count - int32(1)

        accumulator_length = (tableau.stage_count - 1) * n

        has_evaluate_driver_at_t = evaluate_driver_at_t is not None
        first_same_as_last = self.first_same_as_last
        multistage = stage_count > 1
        has_error = self.is_adaptive

        stage_rhs_coeffs = tableau.typed_columns(tableau.a, numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        stage_nodes = tableau.typed_vector(tableau.c, numba_precision)

        if has_error:
            error_weights = tableau.error_weights(numba_precision)
        else:
            error_weights = tuple(typed_zero for _ in range(stage_count))

        # Replace streaming accumulation with direct assignment when
        # stage matches b or b_hat row in coupling matrix.
        accumulates_output = tableau.accumulates_output
        accumulates_error = tableau.accumulates_error

        b_row = tableau.b_matches_a_row
        b_hat_row = tableau.b_hat_matches_a_row
        if b_row is not None:
            b_row = int32(b_row)
        if b_hat_row is not None:
            b_hat_row = int32(b_hat_row)

        # Get allocators from buffer registry
        getalloc = buffer_registry.get_allocator
        alloc_stage_rhs = getalloc('stage_rhs', self)
        alloc_stage_accumulator = getalloc('stage_accumulator', self)

        # no cover: start
        @cuda.jit(
            # (
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision,
            #     numba_precision,
            #     int32,
            #     int32,
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     int32[::1],
            # ),
            device=True,
            inline=True,
        )
        def step(
            state,
            proposed_state,
            parameters,
            driver_coeffs,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,
            residuals,
            jacobian_updates,
            stage_states,
            stage_derivatives,
            stage_observables,
            stage_drivers_out,
            stage_increments,
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
            dt_scalar,
            time_scalar,
            first_step_flag,
            accepted_flag,
            shared,
            persistent_local,
            counters,
        ):

            stage_rhs = alloc_stage_rhs(shared, persistent_local)
            stage_accumulator = alloc_stage_accumulator(shared, persistent_local)

            observable_count = proposed_observables.shape[0]

            current_time = time_scalar
            end_time = current_time + dt_scalar

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] = typed_zero
                if has_error and accumulates_error:
                    error[idx] = typed_zero

            # LOGGING: Record stage 0 initial state
            for idx in range(n):
                stage_states[0, idx] = state[idx]
                residuals[0, idx] = typed_zero
                jacobian_updates[0, idx] = typed_zero
                stage_increments[0, idx] = typed_zero
            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = observables[obs_idx]
            for driver_idx in range(stage_drivers_out.shape[1]):
                stage_drivers_out[0, driver_idx] = drivers_buffer[driver_idx]

            # ----------------------------------------------------------- #
            #            Stage 0: may use cached values                   #
            # ----------------------------------------------------------- #
            # Only use cache if all threads in warp can - otherwise no gain.
            use_cached_rhs = False
            if first_same_as_last and multistage:
                if not first_step_flag:
                    mask = activemask()
                    all_threads_accepted = all_sync(
                            mask,
                            accepted_flag != int32(0))
                    use_cached_rhs = all_threads_accepted
            else:
                use_cached_rhs = False

            # Keep cached rhs if able to, otherwise recalculate.
            if not multistage or not use_cached_rhs:
                evaluate_f(
                    state,
                    parameters,
                    drivers_buffer,
                    observables,
                    stage_rhs,
                    current_time,
                )

            # LOGGING: Record stage 0 derivatives
            for idx in range(n):
                stage_derivatives[0, idx] = stage_rhs[idx]
                stage_increments[0, idx] = dt_scalar * stage_rhs[idx]

            # b weights can't match a rows for erk, as they would return 0
            # So we include ifs to skip accumulating but do no direct assign.
            for idx in range(n):
                increment = stage_rhs[idx]
                if accumulates_output:
                    proposed_state[idx] += solution_weights[0] * increment
                if has_error:
                    if accumulates_error:
                        error[idx] = error[idx] + error_weights[0] * increment

            for idx in range(accumulator_length):
                stage_accumulator[idx] = typed_zero

            # ----------------------------------------------------------- #
            #            Stages 1-s: refresh observables and drivers       #
            # ----------------------------------------------------------- #
            for prev_idx in range(stages_except_first):
                stage_offset = prev_idx * n
                stage_idx = prev_idx + int32(1)
                matrix_col = stage_rhs_coeffs[prev_idx]

                for successor_idx in range(stages_except_first):
                    coeff = matrix_col[successor_idx + int32(1)]
                    row_offset = successor_idx * n
                    for idx in range(n):
                        increment = stage_rhs[idx]
                        stage_accumulator[row_offset + idx] += (
                            coeff * increment
                        )

                base = stage_offset
                dt_stage = dt_scalar * stage_nodes[stage_idx]
                stage_time = current_time + dt_stage

                # Convert accumulated gradients sum(f(y_nj) into a state y_j
                for idx in range(n):
                    stage_accumulator[base] = (
                        stage_accumulator[base] * dt_scalar + state[idx]
                    )
                    base += int32(1)

                # LOGGING: Record stage state
                for idx in range(n):
                    stage_states[stage_idx, idx] = stage_accumulator[
                        stage_offset + idx
                    ]
                    residuals[stage_idx, idx] = typed_zero
                    jacobian_updates[stage_idx, idx] = typed_zero

                # get rhs for next stage
                stage_drivers = proposed_drivers
                if has_evaluate_driver_at_t:
                    evaluate_driver_at_t(
                        stage_time,
                        driver_coeffs,
                        stage_drivers,
                    )

                # LOGGING: Record driver values
                for driver_idx in range(stage_drivers_out.shape[1]):
                    stage_drivers_out[stage_idx, driver_idx] = (
                        stage_drivers[driver_idx]
                    )

                evaluate_observables(
                    stage_accumulator[stage_offset : stage_offset + n],
                    parameters,
                    stage_drivers,
                    proposed_observables,
                    stage_time,
                )

                # LOGGING: Record observables
                for obs_idx in range(observable_count):
                    stage_observables[stage_idx, obs_idx] = (
                        proposed_observables[obs_idx]
                    )

                evaluate_f(
                    stage_accumulator[stage_offset : stage_offset + n],
                    parameters,
                    stage_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                # LOGGING: Record derivatives and increments
                for idx in range(n):
                    stage_derivatives[stage_idx, idx] = stage_rhs[idx]
                    stage_increments[stage_idx, idx] = dt_scalar * stage_rhs[idx]

                # Accumulate f(y_jn) terms or capture direct stage state
                solution_weight = solution_weights[stage_idx]
                error_weight = error_weights[stage_idx]
                for idx in range(n):
                    if accumulates_output:
                        increment = stage_rhs[idx]
                        proposed_state[idx] += solution_weight * increment

                    if has_error:
                        if accumulates_error:
                            increment = stage_rhs[idx]
                            error[idx] += error_weight * increment

            if b_row is not None:
                for idx in range(n):
                    proposed_state[idx] = stage_accumulator[
                        (b_row - 1) * n + idx
                    ]
            if b_hat_row is not None:
                for idx in range(n):
                    error[idx] = stage_accumulator[(b_hat_row - 1) * n + idx]
            # ----------------------------------------------------------- #
            for idx in range(n):
                # Scale and shift f(Y_n) value if accumulated
                if accumulates_output:
                    proposed_state[idx] = (
                        proposed_state[idx] * dt_scalar + state[idx]
                    )
                if has_error:
                    # Scale error if accumulated
                    if accumulates_error:
                        error[idx] *= dt_scalar
                    # Or form error from difference if captured from a-row
                    else:
                        error[idx] = proposed_state[idx] - error[idx]

            if has_evaluate_driver_at_t:
                evaluate_driver_at_t(
                    end_time,
                    driver_coeffs,
                    proposed_drivers,
                )

            evaluate_observables(
                    proposed_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    end_time,
            )

            return int32(0)

        return StepCache(step=step)

    @property
    def is_multistage(self) -> bool:
        """Return ``True`` when the method has multiple stages."""
        return self.tableau.stage_count > 1

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` if algorithm calculates an error estimate."""
        return self.tableau.has_error_estimate

    @property
    def shared_memory_elements(self) -> int:
        """Return the number of precision entries required in shared memory."""
        return buffer_registry.shared_buffer_size(self)

    @property
    def local_scratch_elements(self) -> int:
        """Return the number of local precision entries required."""
        return buffer_registry.local_buffer_size(self)

    @property
    def persistent_local_elements(self) -> int:
        """Return the number of persistent local entries required."""
        return buffer_registry.persistent_local_buffer_size(self)

    @property
    def order(self) -> int:
        """Return the classical order of accuracy."""

        return self.tableau.order

    @property
    def threads_per_step(self) -> int:
        """Return the number of CUDA threads that advance one state."""

        return 1
