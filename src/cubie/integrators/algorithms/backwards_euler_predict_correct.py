"""Backward Euler step with an explicit predictor and implicit corrector."""

from typing import Callable

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
            drivers,
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
            drivers
                Device array of time-dependent drivers.
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
                drivers,
                observables,
                work_buffer,
                time_scalar,
            )
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
                error,  # fixed-step loop reuses error as scratch
            )

            observables_function(
                proposed_state,
                parameters,
                drivers,
                proposed_observables,
                time_scalar,
            )
            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)
