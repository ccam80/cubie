"""Explicit Euler step implementation."""

from typing import Callable, Optional

from numba import cuda, int32

from cubie.integrators.algorithms.base_algorithm_step import StepCache
from cubie.integrators.algorithms.ode_explicitstep import (
    ExplicitStepConfig,
    ODEExplicitStep,
)

class ExplicitEulerStep(ODEExplicitStep):
    """Forward Euler integration step for explicit ODE updates."""

    def __init__(
        self,
        dxdt_function: Callable,
        observables_function: Callable,
        precision: type,
        n: int,
        dt: float,
        get_solver_helper_fn: Optional[Callable] = None,
    ) -> None:
        """Initialise the explicit Euler step configuration.

        Parameters
        ----------
        dxdt_function
            Device derivative function evaluating ``dx/dt``.
        observables_function
            Device function computing system observables.
        precision
            Precision applied to device buffers.
        n
            Number of state entries advanced per step.
        dt
            Fixed step size.
        solver_function_getter
            Present for interface parity with implicit steps and ignored here.
        """

        config = ExplicitStepConfig(
            dt=dt,
            precision=precision,
            dxdt_function=dxdt_function,
            observables_function=observables_function,
            n=n,
        )

        super().__init__(config)

    def build_step(
        self,
        dxdt_function: Callable,
        observables_function: Callable,
        numba_precision: type,
        n: int,
        fixed_step_size: float,
    ) -> StepCache:
        """Build the device function for an explicit Euler step.

        Parameters
        ----------
        dxdt_function
            Device derivative function used within the update.
        numba_precision
            Numba precision corresponding to the configured precision.
        n
            Dimension of the state vector.
        fixed_step_size
            Step size used for integration.

        Returns
        -------
        StepCache
            Container holding the compiled step function.
        """

        step_size = numba_precision(fixed_step_size)

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
            """Advance the state with a single explicit Euler update.

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
                Device array reserved for error estimates (unused here).
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
                Status code indicating successful completion.
            """

            dxdt_function(
                state,
                parameters,
                drivers,
                observables,
                work_buffer,
                time_scalar,
            )
            for i in range(n):
                proposed_state[i] = state[i] + step_size * work_buffer[i]
            observables_function(
                proposed_state,
                parameters,
                drivers,
                proposed_observables,
                time_scalar,
            )
            return int32(0)

        return StepCache(step=step, nonlinear_solver=None)

    @property
    def shared_memory_required(self) -> int:
        """Shared memory usage expressed in precision-sized entries."""

        return 0

    @property
    def local_scratch_required(self) -> int:
        """Local scratch usage expressed in precision-sized entries."""

        return 0

    @property
    def persistent_local_required(self) -> int:
        """Persistent local storage expressed in precision-sized entries."""

        return 0

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""
        return 1

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because explicit Euler is a single-stage method."""
        return False

    @property
    def is_adaptive(self) -> bool:
        """Return ``False`` because explicit Euler has no error estimator."""
        return False

    @property
    def order(self) -> int:
        """Return the classical order of the explicit Euler method."""
        return 1
