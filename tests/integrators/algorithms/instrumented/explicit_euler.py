"""Explicit Euler integration step with instrumentation support."""

from typing import Callable, Optional

from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.ode_explicitstep import (
    ExplicitStepConfig,
    ODEExplicitStep,
)


EE_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)


class ExplicitEulerStep(ODEExplicitStep):
    """Forward Euler step instrumented to expose intermediate buffers."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dt: Optional[float],
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
    ) -> None:
        """Initialise the instrumented explicit Euler configuration."""

        if dt is None:
            dt = EE_DEFAULTS.step_controller["dt"]
        config = ExplicitStepConfig(
            precision=precision,
            n=n,
            dt=dt,
            dxdt_function=dxdt_function,
            observables_function=observables_function,
            driver_function=driver_function,
            get_solver_helper_fn=get_solver_helper_fn,
        )
        super().__init__(config, EE_DEFAULTS.copy())

    def build_step(
        self,
        dxdt_function: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
        dt: float,
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the explicit Euler device step with instrumentation."""

        cached_dt = numba_precision(dt)
        dt_is_dynamic = dt is None
        has_driver_function = driver_function is not None

        @cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:, :, :],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :, :],
                numba_precision[:, :, :],
                numba_precision[:, :, :],
                numba_precision[:, :, :],
                numba_precision[:, :, :],
                numba_precision[:, :, :],
                numba_precision[:, :, :],
                numba_precision[:, :],
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
            parameters,
            driver_coefficients,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,
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
            shared,
            persistent_local,
        ):
            typed_zero = numba_precision(0.0)
            step_size = cached_dt

            for idx in range(n):
                residuals[0, idx] = typed_zero
                jacobian_updates[0, idx] = typed_zero
                stage_states[0, idx] = state[idx]
                stage_increments[0, idx] = typed_zero
            for obs_idx in range(proposed_observables.shape[0]):
                stage_observables[0, obs_idx] = observables[obs_idx]
            for driver_idx in range(stage_drivers.shape[1]):
                stage_drivers[0, driver_idx] = drivers_buffer[driver_idx]

            dxdt_function(
                state,
                parameters,
                drivers_buffer,
                observables,
                proposed_state,
                time_scalar,
            )

            for idx in range(n):
                stage_derivatives[0, idx] = proposed_state[idx]
                stage_increments[0, idx] = step_size * proposed_state[idx]

            for idx in range(n):
                proposed_state[idx] = (
                    state[idx] + step_size * proposed_state[idx]
                )

            next_time = time_scalar + step_size
            if has_driver_function:
                driver_function(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )

            for idx in range(error.shape[0]):
                error[idx] = typed_zero

            return int32(0)

        return StepCache(step=step, nonlinear_solver=None)

    @property
    def stage_count(self) -> int:
        """Return the single-stage structure of explicit Euler."""

        return 1

    @property
    def shared_memory_required(self) -> int:
        """Explicit Euler reuses the base shared memory calculation."""

        return super().shared_memory_required

    @property
    def local_scratch_required(self) -> int:
        """Explicit Euler does not require persistent local scratch."""

        return 0

    @property
    def algorithm_shared_elements(self) -> int:
        """Explicit Euler allocates no additional shared elements."""

        return 0

    @property
    def algorithm_local_elements(self) -> int:
        """Explicit Euler does not reserve persistent local storage."""

        return 0

    @property
    def threads_per_step(self) -> int:
        """Return the number of CUDA threads required per step."""

        return 1

    @property
    def is_multistage(self) -> bool:
        """Explicit Euler advances a single stage per step."""

        return False

    @property
    def is_adaptive(self) -> bool:
        """Explicit Euler does not compute an error estimate."""

        return False

    @property
    def order(self) -> int:
        """Classical order of accuracy for explicit Euler."""

        return 1
