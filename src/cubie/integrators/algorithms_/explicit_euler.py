"""Explicit Euler step implementation."""

from typing import Callable

from numba import cuda, int32

from cubie.integrators.algorithms_.base_algorithm_step import StepCache
from cubie.integrators.algorithms_.ode_explicitstep import ODEExplicitStep


class ExplicitEulerStep(ODEExplicitStep):
    """Simple forward Euler integration step."""

    def build_step(
        self,
        dxdt_function: Callable,
        numba_precision: type,
        n: int,
        fixed_step_size: float,
    ) -> StepCache:
        """Build the device function for an explicit Euler step.

        Parameters
        ----------
        dxdt_function : Callable
            Device function computing state derivatives.
        numba_precision : type
            Numba-compatible precision type.
        n : int
            Dimension of the state vector.
        fixed_step_size : float
            Step size used for integration.

        Returns
        -------
        StepCache
            Container holding the compiled step function.
        """

        step_size = numba_precision(fixed_step_size)

        @cuda.jit(
            (
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
            ),
            device=True,
            inline=True,
        )
        def step(
            state,
            parameters,
            drivers,
            observables,
            dxdt_buffer,
            error,
            dt,
            shared,
            persistent_local,
        ):
            dxdt_function(state, parameters, drivers, observables, dxdt_buffer)
            for i in range(n):
                state[i] += step_size * dxdt_buffer[i]
            return int32(0)

        return StepCache(step=step, nonlinear_solver=None)

    @property
    def shared_memory_required(self) -> int:  # pragma: no cover - simple
        return 0

    @property
    def local_memory_required(self) -> int:  # pragma: no cover - simple
        return 0

    @property
    def threads_per_step(self) -> int:
        """Threads required for the step."""
        return 1

    @property
    def is_multistage(self) -> bool:
        """Whether algorithm has multiple stages."""
        return False

    @property
    def is_adaptive(self) -> bool:
        """Whether algorithm adjusts step size adaptively."""
        return False