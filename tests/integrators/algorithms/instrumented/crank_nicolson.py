"""Crank–Nicolson integration step with instrumentation."""

from typing import Callable, Optional

import numpy as np
from numba import cuda, int16, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms import ImplicitStepConfig
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.ode_implicitstep import ODEImplicitStep

from .matrix_free_solvers import (
    linear_solver_factory,
    newton_krylov_solver_factory,
)


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

    def build_implicit_helpers(self) -> Callable:
        """Return the instrumented nonlinear solver for Crank–Nicolson."""

        config = self.compile_settings
        beta = config.beta
        gamma = config.gamma
        mass = config.M
        preconditioner_order = config.preconditioner_order
        n = config.n
        get_fn = config.get_solver_helper_fn

        preconditioner = get_fn(
            "neumann_preconditioner",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )
        residual = get_fn(
            "stage_residual",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )
        operator = get_fn(
            "linear_operator",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        linear_solver = linear_solver_factory(
            operator,
            n=n,
            preconditioner=preconditioner,
            correction_type=config.linear_correction_type,
            tolerance=config.krylov_tolerance,
            max_iters=config.max_linear_iters,
            precision=config.precision,
        )

        nonlinear_solver = newton_krylov_solver_factory(
            residual_function=residual,
            linear_solver=linear_solver,
            n=n,
            tolerance=config.newton_tolerance,
            max_iters=config.max_newton_iters,
            damping=config.newton_damping,
            max_backtracks=config.newton_max_backtracks,
            precision=config.precision,
        )
        return nonlinear_solver

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
        config = self.compile_settings
        stage_coefficient = numba_precision(0.5)
        be_coefficient = numba_precision(1.0)
        has_driver_function = driver_function is not None
        solver_shared_elements = self.solver_shared_elements
        stage_count = self.stage_count
        newton_iters = int(config.max_newton_iters)
        newton_backtracks = int(config.newton_max_backtracks)
        newton_slots = newton_iters * (newton_backtracks + 1) + 1
        linear_iters = int(config.max_linear_iters)
        linear_slots = max(1, stage_count * newton_iters)

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
                numba_precision[:, :, :],
                numba_precision[:, :, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :, :],
                numba_precision[:, :, :],
                numba_precision[:, :],
                numba_precision[:, :, :],
                numba_precision,
                numba_precision,
                int16,
                int16,
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
            first_step_flag,
            accepted_flag,
            shared,
            persistent_local,
        ):
            typed_zero = numba_precision(0.0)
            status_mask = int32(0xFFFF)
            stage_rhs = cuda.local.array(n, numba_precision)
            cn_increment = cuda.local.array(n, numba_precision)
            dummy_newton_initial_guesses = cuda.local.array(
                (stage_count, n),
                numba_precision,
            )
            dummy_newton_iteration_guesses = cuda.local.array(
                (stage_count, newton_slots, n),
                numba_precision,
            )
            dummy_newton_residuals = cuda.local.array(
                (stage_count, newton_slots, n),
                numba_precision,
            )
            dummy_newton_squared_norms = cuda.local.array(
                (stage_count, newton_slots),
                numba_precision,
            )
            dummy_newton_iteration_scale = cuda.local.array(
                (stage_count, newton_iters),
                numba_precision,
            )
            dummy_linear_initial_guesses = cuda.local.array(
                (linear_slots, n),
                numba_precision,
            )
            dummy_linear_iteration_guesses = cuda.local.array(
                (linear_slots, linear_iters, n),
                numba_precision,
            )
            dummy_linear_residuals = cuda.local.array(
                (linear_slots, linear_iters, n),
                numba_precision,
            )
            dummy_linear_squared_norms = cuda.local.array(
                (linear_slots, linear_iters),
                numba_precision,
            )
            dummy_linear_preconditioned_vectors = cuda.local.array(
                (linear_slots, linear_iters, n),
                numba_precision,
            )

            observable_count = proposed_observables.shape[0]

            for idx in range(n):
                proposed_state[idx] = typed_zero
                cn_increment[idx] = typed_zero
                residuals[0, idx] = typed_zero
                jacobian_updates[0, idx] = typed_zero
                stage_states[0, idx] = typed_zero
                stage_derivatives[0, idx] = typed_zero
                stage_increments[0, idx] = typed_zero

            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = typed_zero
            for driver_idx in range(stage_drivers.shape[1]):
                stage_drivers[0, driver_idx] = typed_zero
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
            for driver_idx in range(stage_drivers.shape[1]):
                stage_drivers[0, driver_idx] = proposed_drivers[driver_idx]

            status = solver_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                end_time,
                fixed_dt,
                stage_coefficient,
                base_state,
                solver_scratch,
                int32(0),
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
            )

            for idx in range(n):
                increment_value = proposed_state[idx]
                residual_value = solver_scratch[idx + n]
                final_state = base_state[idx] + (
                    stage_coefficient * increment_value
                )
                trapezoidal_increment = final_state - state[idx]
                proposed_state[idx] = final_state
                base_state[idx] = increment_value
                cn_increment[idx] = trapezoidal_increment
                stage_increments[0, idx] = trapezoidal_increment
                residuals[0, idx] = residual_value
                stage_states[0, idx] = final_state

            be_status = solver_fn(
                base_state,
                parameters,
                proposed_drivers,
                end_time,
                fixed_dt,
                be_coefficient,
                state,
                solver_scratch,
                int32(0),
                dummy_newton_initial_guesses,
                dummy_newton_iteration_guesses,
                dummy_newton_residuals,
                dummy_newton_squared_norms,
                dummy_newton_iteration_scale,
                dummy_linear_initial_guesses,
                dummy_linear_iteration_guesses,
                dummy_linear_residuals,
                dummy_linear_squared_norms,
                dummy_linear_preconditioned_vectors,
            )
            status |= be_status & status_mask

            for idx in range(n):
                error[idx] = (
                    proposed_state[idx] - (state[idx] + base_state[idx])
                )

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                end_time,
            )

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
        """Return True; the embedded error estimate enables adaptivity."""

        return True

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""

        return 1

    @property
    def order(self) -> int:
        """Return the classical order of the Crank–Nicolson method."""

        return 2
