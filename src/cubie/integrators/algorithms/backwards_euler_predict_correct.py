"""Backward Euler step with an explicit predictor and implicit corrector."""

from typing import Callable, Optional

from numba import cuda

from cubie.integrators.algorithms.backwards_euler import BackwardsEulerStep
from cubie.integrators.algorithms.base_algorithm_step import StepCache


class BackwardsEulerPCStep(BackwardsEulerStep):
    """Backward Euler with a predictor-corrector refinement."""

    def build_step(
        self,
        solver_fn: Callable,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
    ) -> StepCache:  # pragma: no cover - cuda code
        """Build the device function for the predictor-corrector scheme.

        Parameters
        ----------
        solver_fn
            Device nonlinear solver produced by the implicit helper chain.
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

        Returns
        -------
        StepCache
            Container holding the compiled predictor-corrector step.
        """

        a_ij = numba_precision(1.0)
        has_driver_function = driver_function is not None
        driver_function = driver_function

        @cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:, :, :],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
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
            work_buffer,
            parameters,
            driver_coefficients,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,
            dt_scalar,
            time_scalar,
            shared,
            persistent_local,
        ):
            """Advance the state using an explicit predictor and implicit corrector.

            Parameters
            ----------
            state
                Device array storing the current state.
            proposed_state
                Device array receiving the updated state.
            work_buffer
                Device array used as temporary storage for derivatives.
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
                Device array capturing solver diagnostics.
            dt_scalar
                Scalar containing the proposed step size.
            time_scalar
                Scalar containing the current simulation time.
            shared
                Device array used for shared memory (unused here).
            persistent_local
                Device array for persistent local storage (unused here).

            Returns
            -------
            int
                Status code returned by the nonlinear solver.
            """

            dxdt_fn(
                state,
                parameters,
                drivers_buffer,
                observables,
                work_buffer,
                time_scalar,
            )
            for i in range(n):
                proposed_state[i] = state[i] + dt_scalar * work_buffer[i]

            next_time = time_scalar + dt_scalar
            if has_driver_function:
                driver_function(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            resid = cuda.local.array(n, numba_precision)
            z = cuda.local.array(n, numba_precision)

            status = solver_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                dt_scalar,
                a_ij,
                state,
                work_buffer,
                resid,
                z,
                error,  # fixed-step loop reuses error as scratch
            )

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )
            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)
