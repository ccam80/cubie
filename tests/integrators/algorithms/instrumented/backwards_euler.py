"""Backward Euler step implementation using Newton–Krylov with instrumentation."""

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

BE_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)


class BackwardsEulerStep(ODEImplicitStep):
    """Backward Euler step solved with matrix-free Newton–Krylov."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dt: float,
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        preconditioner_order: int = 1,
        krylov_tolerance: float = 1e-5,
        max_linear_iters: int = 100,
        linear_correction_type: str = "minimal_residual",
        newton_tolerance: float = 1e-5,
        max_newton_iters: int = 100,
        newton_damping: float = 0.85,
        newton_max_backtracks: int = 25,
    ) -> None:
        """Initialise the backward Euler step configuration."""

        if dt is None:
            dt = BE_DEFAULTS.step_controller["dt"]

        beta = ALGO_CONSTANTS["beta"]
        gamma = ALGO_CONSTANTS["gamma"]
        mass_matrix = ALGO_CONSTANTS["M"](n, dtype=precision)
        config = ImplicitStepConfig(
            get_solver_helper_fn=get_solver_helper_fn,
            beta=beta,
            gamma=gamma,
            M=mass_matrix,
            n=n,
            dt=dt,
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
        super().__init__(config, BE_DEFAULTS.copy())

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
        """Build the device function for a backward Euler step."""

        a_ij = numba_precision(1.0)
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

            solver_scratch = shared[:solver_shared_elements]
            instrument = stage_states.shape[0] > 0
            observable_count = proposed_observables.shape[0]

            for idx in range(n):
                guess_value = solver_scratch[idx]
                proposed_state[idx] = guess_value
                if instrument:
                    solver_initial_guesses[0, idx] = guess_value
                    residuals[0, idx] = typed_zero
                    jacobian_updates[0, idx] = typed_zero
                    solver_solutions[0, idx] = typed_zero
                    stage_states[0, idx] = typed_zero
                    stage_derivatives[0, idx] = typed_zero

            if instrument:
                for obs_idx in range(observable_count):
                    stage_observables[0, obs_idx] = typed_zero
                solver_iterations[0] = typed_int_zero
                solver_status[0] = typed_int_zero

            fixed_dt = dt if dt is not None else dt_scalar
            next_time = time_scalar + fixed_dt

            if has_driver_function:
                driver_function(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            status = solver_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                fixed_dt,
                a_ij,
                state,
                solver_scratch,
            )

            if instrument:
                solver_iterations[0] = status >> 16
                solver_status[0] = status & status_mask

            for idx in range(n):
                solver_scratch[idx] = proposed_state[idx]
                proposed_state[idx] += state[idx]

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )

            if instrument:
                for idx in range(n):
                    increment_value = solver_scratch[idx]
                    solver_solutions[0, idx] = increment_value
                    stage_states[0, idx] = proposed_state[idx]
                for obs_idx in range(observable_count):
                    stage_observables[0, obs_idx] = proposed_observables[obs_idx]

                dxdt_fn(
                    proposed_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    next_time,
                )

                for idx in range(n):
                    rhs_value = stage_rhs[idx]
                    stage_derivatives[0, idx] = rhs_value
                    residuals[0, idx] = solver_solutions[0, idx] - fixed_dt * rhs_value
                    jacobian_updates[0, idx] = typed_zero

            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because backward Euler is a single-stage method."""

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
        """Backward Euler has no additional shared-memory requirements."""

        return 0

    @property
    def algorithm_local_elements(self) -> int:
        """Backward Euler does not reserve persistent local storage."""

        return 0

    @property
    def stage_count(self) -> int:
        """Backward Euler advances a single implicit stage."""

        return 1

    @property
    def is_adaptive(self) -> bool:
        """Return ``False`` because backward Euler is fixed step."""

        return False

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""

        return 1

    @property
    def settings_dict(self) -> dict:
        """Return the configuration dictionary for the step."""

        return self.compile_settings.settings_dict

    @property
    def order(self) -> int:
        """Return the classical order of the backward Euler method."""

        return 1

    @property
    def dxdt_function(self) -> Optional[Callable]:
        """Return the derivative device function."""

        return self.compile_settings.dxdt_function

    @property
    def identifier(self) -> str:
        """Return the identifier describing this algorithm."""

        return "backwards_euler"
