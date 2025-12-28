"""Backward Euler step with an explicit predictor and implicit corrector."""

from typing import Callable, Optional

from numba import cuda, int32

from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.backwards_euler import BackwardsEulerStep
from cubie.integrators.algorithms.base_algorithm_step import StepCache

class BackwardsEulerPCStep(BackwardsEulerStep):
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
        solver_function
            Device function for the Newton-Krylov nonlinear solver.
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
        solver_fn = solver_function

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

            predictor = alloc_increment_cache(shared, persistent_local)
            solver_scratch = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(shared,
                                                        persistent_local)
            dxdt_fn(
                state,
                parameters,
                drivers_buffer,
                observables,
                predictor,
                time_scalar,
            )
            for i in range(n):
                proposed_state[i] = dt_scalar * predictor[i]

            next_time = time_scalar + dt_scalar
            if has_driver_function:
                driver_function(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            status = solver_fn(
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
            )

            for i in range(n):
                proposed_state[i] += state[i]

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )


            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)
