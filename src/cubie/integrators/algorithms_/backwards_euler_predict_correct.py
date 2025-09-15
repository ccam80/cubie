from cubie.integrators.algorithms_.backwards_euler import BackwardsEulerStep
from numba import cuda

class BackwardsEulerPredictCorrectStep(BackwardsEulerStep):
    """Backwards Euler with a predictor-corrector approach.

    This method uses an explicit Euler step to predict the next state,
    then refines this prediction using the implicit Backwards Euler method.
    """

    def build_step(self,
                   solver_fn,
                   dxdt_fn,
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
                numba_precision[:],
                numba_precision[:],
            ),
            device=True,
            inline=True,
        )
        def step(
            state,
            parameters,
            drivers,
            observables,
            proposed_state,
            error,
            dt,
            shared,
            persistent_local,
        ):
            dt_scalar = dt[0]

            # Only difference to backwards euler is here:
            dxdt = dxdt_fn(state, parameters, drivers, observables)
            for i in range(n):
                proposed_state[i] = state[i] + dt_scalar * dxdt[i]

            # The rest is unaltered

            delta = cuda.local.array(n, numba_precision)
            resid = cuda.local.array(n, numba_precision)
            z = cuda.local.array(n, numba_precision)
            temp = cuda.local.array(n, numba_precision)

            status = solver_fn(
                proposed_state,
                parameters,
                drivers,
                dt_scalar,
                a_ij,
                state,
                delta,
                resid,
                z,
                temp,
            )

            for i in range(n):
                state[i] = proposed_state[i]
            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)