"""Crank–Nicolson integration step with instrumentation."""

from typing import Callable, Optional

import numpy as np
from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms import ImplicitStepConfig
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.ode_implicitstep import ODEImplicitStep


ALGO_CONSTANTS = {
    "beta": 1.0,
    "gamma": 1.0,
    "M": np.eye,
}

CN_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "pi",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
        "kp": 0.6,
        "kd": 0.4,
        "deadband_min": 1.0,
        "deadband_max": 1.1,
        "min_gain": 0.5,
        "max_gain": 2.0,
    }
)


class CrankNicolsonStep(ODEImplicitStep):
    """Crank–Nicolson step with embedded backward Euler error estimation."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dt: Optional[float],
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        preconditioner_order: int = 1,
        krylov_tolerance: float = 1e-6,
        max_linear_iters: int = 100,
        linear_correction_type: str = "minimal_residual",
        newton_tolerance: float = 1e-6,
        max_newton_iters: int = 1000,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 10,
    ) -> None:
        """Initialise the Crank–Nicolson step configuration."""

        beta = ALGO_CONSTANTS["beta"]
        gamma = ALGO_CONSTANTS["gamma"]
        mass_matrix = ALGO_CONSTANTS["M"](n, dtype=precision)

        config = ImplicitStepConfig(
            get_solver_helper_fn=get_solver_helper_fn,
            beta=beta,
            gamma=gamma,
            M=mass_matrix,
            n=n,
            preconditioner_order=preconditioner_order,
            krylov_tolerance=krylov_tolerance,
            max_linear_iters=max_linear_iters,
            linear_correction_type=linear_correction_type,
            newton_tolerance=newton_tolerance,
            max_newton_iters=max_newton_iters,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
            dxdt_function=dxdt_function,
            observables_function=observables_function,
            driver_function=driver_function,
            precision=precision,
        )
        super().__init__(config, CN_DEFAULTS)

    def build_step(
        self,
        solver_fn: Callable,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
        dt: Optional[float],
    ) -> StepCache:  # pragma: no cover - cuda code
        """Build the device function for the Crank–Nicolson step."""

        stage_coefficient = numba_precision(0.5)
        be_coefficient = numba_precision(1.0)
        has_driver_function = driver_function is not None
        solver_shared_elements = self.solver_shared_elements

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
                numba_precision[:, :],
                numba_precision[:, :],
                int32[:],
                int32[:],
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
            solver_initial_guesses,
            solver_solutions,
            solver_iterations,
            solver_status,
            dt_scalar,
            time_scalar,
            shared,
            persistent_local,
        ):
            typed_zero = numba_precision(0.0)
            typed_int_zero = int32(0)
            status_mask = int32(0xFFFF)
            stage_rhs = cuda.local.array(n, numba_precision)
            cn_increment = cuda.local.array(n, numba_precision)

            instrument = stage_states.shape[0] > 0
            observable_count = proposed_observables.shape[0]

            for idx in range(n):
                proposed_state[idx] = typed_zero
                cn_increment[idx] = typed_zero
                if instrument:
                    solver_initial_guesses[0, idx] = typed_zero
                    solver_solutions[0, idx] = typed_zero
                    residuals[0, idx] = typed_zero
                    jacobian_updates[0, idx] = typed_zero
                    stage_states[0, idx] = typed_zero
                    stage_derivatives[0, idx] = typed_zero
                    stage_increments[0, idx] = typed_zero

            if instrument:
                for obs_idx in range(observable_count):
                    stage_observables[0, obs_idx] = typed_zero
                for driver_idx in range(stage_drivers.shape[1]):
                    stage_drivers[0, driver_idx] = typed_zero
                solver_iterations[0] = typed_int_zero
                solver_status[0] = typed_int_zero

            solver_scratch = shared[:solver_shared_elements]
            dxdt_buffer = solver_scratch[:n]
            base_state = error

            dxdt_fn(
                state,
                parameters,
                drivers_buffer,
                observables,
                dxdt_buffer,
                time_scalar,
            )

            fixed_dt = dt if dt is not None else dt_scalar
            half_dt = fixed_dt * numba_precision(0.5)
            end_time = time_scalar + fixed_dt

            for idx in range(n):
                base_state[idx] = state[idx] + half_dt * dxdt_buffer[idx]

            if has_driver_function:
                driver_function(
                    end_time,
                    driver_coefficients,
                    proposed_drivers,
                )
            if instrument:
                for driver_idx in range(stage_drivers.shape[1]):
                    stage_drivers[0, driver_idx] = proposed_drivers[driver_idx]

            status = solver_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                fixed_dt,
                stage_coefficient,
                base_state,
                solver_scratch,
            )

            if instrument:
                solver_iterations[0] = status >> 16
                solver_status[0] = status & status_mask

            for idx in range(n):
                increment_value = proposed_state[idx]
                residual_value = solver_scratch[idx + n]
                final_state = base_state[idx] + stage_coefficient * increment_value
                trapezoidal_increment = final_state - state[idx]
                proposed_state[idx] = final_state
                base_state[idx] = increment_value
                cn_increment[idx] = trapezoidal_increment
                if instrument:
                    solver_solutions[0, idx] = trapezoidal_increment
                    stage_increments[0, idx] = trapezoidal_increment
                    residuals[0, idx] = residual_value
                    stage_states[0, idx] = final_state

            be_status = solver_fn(
                base_state,
                parameters,
                proposed_drivers,
                fixed_dt,
                be_coefficient,
                state,
                solver_scratch,
            )
            status |= be_status & status_mask

            if instrument:
                solver_status[0] = status & status_mask

            for idx in range(n):
                error[idx] = proposed_state[idx] - (state[idx] + base_state[idx])

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                end_time,
            )

            if instrument:
                for obs_idx in range(observable_count):
                    stage_observables[0, obs_idx] = proposed_observables[obs_idx]

                dxdt_fn(
                    proposed_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    end_time,
                )

                for idx in range(n):
                    rhs_value = stage_rhs[idx]
                    stage_derivatives[0, idx] = rhs_value
                    jacobian_updates[0, idx] = typed_zero

            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because Crank–Nicolson is a single-stage method."""

        return False

    @property
    def shared_memory_required(self) -> int:
        """Shared memory usage expressed in precision-sized entries."""

        return super().shared_memory_required

    @property
    def local_scratch_required(self) -> int:
        """Local scratch usage expressed in precision-sized entries."""

        return 0

    @property
    def algorithm_shared_elements(self) -> int:
        """Crank–Nicolson does not reserve extra shared scratch."""

        return 0

    @property
    def algorithm_local_elements(self) -> int:
        """Crank–Nicolson does not require persistent local storage."""

        return 0

    @property
    def stage_count(self) -> int:
        """Crank–Nicolson evaluates a single implicit stage."""

        return 1

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` because the embedded error estimate enables adaptivity."""

        return True

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""

        return 1

    @property
    def order(self) -> int:
        """Return the classical order of the Crank–Nicolson method."""

        return 2
