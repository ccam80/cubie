from cubie.integrators.algorithms_.backwards_euler import BackwardsEulerStep
from numba import cuda

from cubie.integrators.algorithms_.base_algorithm_step import StepCache


class BackwardsEulerPCStep(BackwardsEulerStep):
    """Backwards Euler with a predictor-corrector approach.

    This method uses an explicit Euler step to predict the next state,
    then refines this prediction using the implicit Backwards Euler method.
    """

    def build_step(self,
                   solver_fn,
                   dxdt_fn,
                   obs_fn,
                   numba_precision,
                   n):  # pragma: no cover - complex
        """Build the device function for a backward Euler step."""

        a_ij = numba_precision(1.0)

        @cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
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
            drivers,
            observables,
            error,
            dt_scalar,
            shared,
            persistent_local,
        ):
            # Only difference to backwards euler is here:
            dxdt_fn(state, parameters, drivers, observables, work_buffer)
            for i in range(n):
                proposed_state[i] = state[i] + dt_scalar * work_buffer[i]

            resid = cuda.local.array(n, numba_precision)
            z = cuda.local.array(n, numba_precision)

            status = solver_fn(
                proposed_state,
                parameters,
                drivers,
                dt_scalar,
                a_ij,
                state,
                work_buffer,
                resid,
                z,
                error, # fixed-step loop doesn't use error, reuse as scratch
            )
            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)
