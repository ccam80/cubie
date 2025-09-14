"""Explicit Euler step implementation."""
from typing import Callable

from numba import cuda, int32

from cubie.integrators.algorithms_.base_algorithm_step import StepCache
from cubie.integrators.algorithms_.ode_explicitstep import (
    ODEExplicitStep,
)


class ExplicitEulerStep(ODEExplicitStep):
    """Simple forward Euler integration step."""

    def build_step(self,
                   dxdt_function: Callable,
                   numba_precision: type,
                   n: int,
                   fixed_step_size: float
                   ):
        fixed_step_size = numba_precision(fixed_step_size)
        # no cover: start
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
        def step(state,
                 parameters,
                 drivers,
                 observables,
                 dxdt_buffer,
                 error, dt,
                 shared, persistent_local):
            dxdt_function(state, parameters, drivers, observables, dxdt_buffer)
            for i in range(n):
                state[i] += fixed_step_size * dxdt_buffer[i]
            return int32(0)
        # no cover: end

        return StepCache(step=step,
                         nonlinear_solver=None)

    @property
    def shared_memory_required(self) -> int:  # pragma: no cover - simple
        return 0

    @property
    def local_memory_required(self) -> int:  # pragma: no cover - simple
        return 0

    @property
    def threads_per_step(self) -> int:
        return 1

    @property
    def is_multistage(self) -> bool:
        return False

    @property
    def is_adaptive(self) -> bool:
        return False