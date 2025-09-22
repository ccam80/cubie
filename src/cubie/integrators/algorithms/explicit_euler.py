"""Explicit Euler step implementation."""

from typing import Callable, Optional

from numba import cuda, int32

from cubie.integrators.algorithms.base_algorithm_step import StepCache
from cubie.integrators.algorithms.ode_explicitstep import ODEExplicitStep, \
    ExplicitStepConfig

class ExplicitEulerStep(ODEExplicitStep):
    """Simple forward Euler integration step."""

    def __init__(
        self,
        dxdt_function: Callable,
        precision: type,
        n: int,
        dt: float,
        solver_function_getter: Optional[Callable] = None,
    ):
        config = ExplicitStepConfig(dt=dt,
                                    precision=precision,
                                    dxdt_function=dxdt_function,
                                    n=n)

        super().__init__(config)


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
            dxdt_function(state, parameters, drivers, observables, work_buffer)
            for i in range(n):
                proposed_state[i] = state[i] + step_size * work_buffer[i]
            return int32(0)

        return StepCache(step=step, nonlinear_solver=None)

    @property
    def shared_memory_required(self) -> int:  # pragma: no cover - simple
        return 0

    @property
    def local_scratch_required(self) -> int:  # pragma: no cover - simple
        return 0

    @property
    def persistent_local_required(self) -> int:
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

    @property
    def order(self) -> int:
        """Order of the algorithm."""
        return 1