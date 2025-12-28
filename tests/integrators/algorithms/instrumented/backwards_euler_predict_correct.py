"""Backward Euler step with an explicit predictor and implicit corrector."""

from typing import Callable, Optional

from numba import cuda, int32

from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.base_algorithm_step import StepCache
from tests.integrators.algorithms.instrumented.backwards_euler import (
    InstrumentedBackwardsEulerStep
)


class InstrumentedBackwardsEulerPCStep(InstrumentedBackwardsEulerStep):
    """Backward Euler with a predictor-corrector refinement."""

    def build_step(
        self,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        solver_function: Callable,
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - cuda code
        """Build the device function for the predictor-corrector scheme.

        Parameters
        ----------
        dxdt_fn
            Device derivative function for the ODE system.
        observables_function
            Device observable computation helper.
        driver_function
            Optional device function evaluating drivers at arbitrary times.
        numba_precision
            Numba precision corresponding to the configured precision.
        n
            Dimension of the state vector.
        n_drivers
            Number of driver signals provided to the system.

        Returns
        -------
        StepCache
            Container holding the compiled predictor-corrector step.
        """
        a_ij = numba_precision(1.0)
        has_driver_function = driver_function is not None
        n = int32(n)

        # Get child allocators for Newton solver
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(self, self.solver,
                                                 name='solver')
        )
        alloc_increment_cache = buffer_registry.get_allocator('increment_cache', self)
        
        n = int32(n)

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
            driver_coefficients,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,  # Non-adaptive algorithms receive a zero-length slice.
            residuals,
            jacobian_updates,
            stage_states,
            stage_derivatives,
            stage_observables,
            stage_drivers,
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
            """Advance the state using an explicit predictor and implicit corrector.

            Parameters
            ----------
            state
                Device array storing the current state.
            proposed_state
                Device array receiving the updated state.
            parameters
                Device array of static model parameters.
            driver_coefficients
                Device array containing spline driver coefficients.
            drivers_buffer
                Device array of time-dependent drivers.
            proposed_drivers
                Device array receiving proposed driver samples.
            observables
                Device array storing accepted observable outputs.
            proposed_observables
                Device array receiving proposed observable outputs.
            error
                Device array capturing solver diagnostics. Fixed-step
                algorithms receive a zero-length slice that can be repurposed
                as scratch when available.
            dt_scalar
                Scalar containing the proposed step size.
            time_scalar
                Scalar containing the current simulation time.
            shared
                Device array providing shared scratch buffers.
            persistent_local
                Device array for persistent local storage (unused here).

            Returns
            -------
            int
                Status code returned by the nonlinear solver.
            """
            typed_zero = numba_precision(0.0)
            stage_rhs = cuda.local.array(n, numba_precision)

            solver_scratch = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(shared,
                                                        persistent_local)
            observable_count = proposed_observables.shape[0]
            predictor = alloc_increment_cache(shared, persistent_local)

            dxdt_fn(
                state,
                parameters,
                drivers_buffer,
                observables,
                predictor,
                time_scalar,
            )

            # LOGGING: Initialize instrumentation arrays
            for i in range(n):
                proposed_state[i] = dt_scalar * predictor[i]
                residuals[0, i] = typed_zero
                jacobian_updates[0, i] = typed_zero
                stage_states[0, i] = typed_zero
                stage_derivatives[0, i] = typed_zero
                stage_increments[0, i] = typed_zero

            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = typed_zero
            for driver_idx in range(stage_drivers.shape[1]):
                stage_drivers[0, driver_idx] = typed_zero

            next_time = time_scalar + dt_scalar
            if has_driver_function:
                driver_function(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            # LOGGING: Record stage drivers
            for driver_idx in range(stage_drivers.shape[1]):
                stage_drivers[0, driver_idx] = proposed_drivers[driver_idx]


            status = solver_function(
                proposed_state,
                parameters,
                proposed_drivers,
                next_time,
                dt_scalar,
                a_ij,
                state,
                solver_scratch,
                solver_persistent,
                counters,
                int32(0),
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
            )

            # LOGGING: Record increment and residual values
            for i in range(n):
                predictor[i] = proposed_state[i]
                stage_increments[0, i] = proposed_state[i]

            for i in range(n):
                proposed_state[i] += state[i]
                stage_states[0, i] = proposed_state[i]

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )

            # LOGGING: Record observables
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

            # LOGGING: Record derivatives
            for i in range(n):
                stage_derivatives[0, i] = stage_rhs[i]
                jacobian_updates[0, i] = typed_zero

            return status

        return StepCache(step=step, nonlinear_solver=solver_function)

    @property
    def local_scratch_elements(self) -> int:
        """Local scratch usage expressed in precision-sized entries."""

        return 0
