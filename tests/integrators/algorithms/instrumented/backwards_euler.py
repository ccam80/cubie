"""Backward Euler step with instrumented Newton–Krylov solver."""

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

    def build_implicit_helpers(self) -> Callable:
        """Return the instrumented nonlinear solver for backward Euler."""

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
            stage_rhs = cuda.local.array(n, numba_precision)

            solver_scratch = shared[:solver_shared_elements]
            observable_count = proposed_observables.shape[0]

            for idx in range(n):
                guess_value = solver_scratch[idx]
                proposed_state[idx] = guess_value
                residuals[0, idx] = typed_zero
                jacobian_updates[0, idx] = typed_zero
                stage_states[0, idx] = typed_zero
                stage_derivatives[0, idx] = typed_zero
                stage_increments[0, idx] = typed_zero

            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = typed_zero
            for driver_idx in range(stage_drivers.shape[1]):
                stage_drivers[0, driver_idx] = typed_zero

            fixed_dt = dt if dt is not None else dt_scalar
            next_time = time_scalar + fixed_dt

            if has_driver_function:
                driver_function(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )
            for driver_idx in range(stage_drivers.shape[1]):
                stage_drivers[0, driver_idx] = proposed_drivers[driver_idx]

            status = solver_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                next_time,
                fixed_dt,
                a_ij,
                state,
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
                solver_scratch[idx] = increment_value
                proposed_state[idx] = increment_value + state[idx]
                stage_increments[0, idx] = increment_value
                residuals[0, idx] = residual_value
                stage_states[0, idx] = proposed_state[idx]

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )

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
