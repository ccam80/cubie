"""Backward Euler predictor–corrector step with instrumentation."""

from typing import Callable, Optional

from numba import cuda, int32

from cubie.integrators.algorithms.base_algorithm_step import StepCache

from .backwards_euler import BackwardsEulerStep


class BackwardsEulerPCStep(BackwardsEulerStep):
    """Backward Euler with a predictor–corrector refinement."""

    def build_step(
        self,
        solver_fn: Callable,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
        dt: Optional[float],
    ) -> StepCache:  # pragma: no cover - cuda code
        """Build the device function for the predictor–corrector scheme."""

        a_ij = numba_precision(1.0)
        has_driver_function = driver_function is not None
        solver_shared_elements = self.solver_shared_elements

        @cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:, :, :],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                int32[:],
                int32[:],
                numba_precision,
                numba_precision,
                numba_precision[:],
                numba_precision[:],
            ),
            device=True,
            inline=True,
        )
        def step(
            state,
            proposed_state,
            parameters,
            driver_coefficients,
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
            solver_initial_guesses,
            solver_solutions,
            solver_iterations,
            solver_status,
            dt_scalar,
            time_scalar,
            shared,
            persistent_local,
        ):
            typed_zero = numba_precision(0.0)
            typed_int_zero = int32(0)
            status_mask = int32(0xFFFF)
            stage_rhs = cuda.local.array(n, numba_precision)

            instrument = stage_states.shape[0] > 0
            observable_count = proposed_observables.shape[0]

            predictor = shared[:n]
            dxdt_fn(
                state,
                parameters,
                drivers_buffer,
                observables,
                predictor,
                time_scalar,
            )

            fixed_dt = dt if dt is not None else dt_scalar
            for idx in range(n):
                increment_guess = fixed_dt * predictor[idx]
                proposed_state[idx] = increment_guess
                if instrument:
                    solver_initial_guesses[0, idx] = increment_guess
                    solver_solutions[0, idx] = typed_zero
                    residuals[0, idx] = typed_zero
                    jacobian_updates[0, idx] = typed_zero
                    stage_states[0, idx] = typed_zero
                    stage_derivatives[0, idx] = typed_zero

            if instrument:
                for obs_idx in range(observable_count):
                    stage_observables[0, obs_idx] = typed_zero
                solver_iterations[0] = typed_int_zero
                solver_status[0] = typed_int_zero

            next_time = time_scalar + fixed_dt

            if has_driver_function:
                driver_function(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            solver_scratch = shared[:solver_shared_elements]

            status = solver_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                fixed_dt,
                a_ij,
                state,
                solver_scratch,
            )

            if instrument:
                solver_iterations[0] = status >> 16
                solver_status[0] = status & status_mask
                for idx in range(n):
                    solver_solutions[0, idx] = proposed_state[idx]

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )

            for idx in range(n):
                proposed_state[idx] += state[idx]
                if instrument:
                    stage_states[0, idx] = proposed_state[idx]

            if instrument:
                for obs_idx in range(observable_count):
                    stage_observables[0, obs_idx] = proposed_observables[obs_idx]

                dxdt_fn(
                    proposed_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    next_time,
                )

                for idx in range(n):
                    rhs_value = stage_rhs[idx]
                    stage_derivatives[0, idx] = rhs_value
                    residuals[0, idx] = solver_solutions[0, idx] - fixed_dt * rhs_value
                    jacobian_updates[0, idx] = typed_zero

            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)
