"""Fully implicit Runge--Kutta integration step implementation."""

from typing import Callable, Optional

import attrs
import numpy as np
from numba import cuda, int16, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.generic_firk_tableaus import (
    DEFAULT_FIRK_TABLEAU,
    FIRKTableau,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ImplicitStepConfig,
    ODEImplicitStep,
)
from tests.integrators.algorithms.instrumented.matrix_free_solvers import (
    inst_linear_solver_factory,
    inst_newton_krylov_solver_factory,
)


FIRK_ADAPTIVE_DEFAULTS = StepControlDefaults(
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

FIRK_FIXED_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)


@attrs.define
class FIRKStepConfig(ImplicitStepConfig):
    """Configuration describing the FIRK integrator."""

    tableau: FIRKTableau = attrs.field(
        default=DEFAULT_FIRK_TABLEAU,
    )

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""

        return self.tableau.stage_count

    @property
    def all_stages_n(self) -> int:
        """Return the flattened dimension covering all stage increments."""

        return self.stage_count * self.n


class FIRKStep(ODEImplicitStep):
    """Fully implicit Runge--Kutta step with an embedded error estimate."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dt: Optional[float] = None,
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        preconditioner_order: int = 2,
        krylov_tolerance: float = 1e-6,
        max_linear_iters: int = 200,
        linear_correction_type: str = "minimal_residual",
        newton_tolerance: float = 1e-6,
        max_newton_iters: int = 100,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
        tableau: FIRKTableau = DEFAULT_FIRK_TABLEAU,
        n_drivers: int = 0,
    ) -> None:
        """Initialise the FIRK step configuration."""

        mass = np.eye(n, dtype=precision)
        config_kwargs = {
            "precision": precision,
            "n": n,
            "n_drivers": n_drivers,
            "dxdt_function": dxdt_function,
            "observables_function": observables_function,
            "driver_function": driver_function,
            "get_solver_helper_fn": get_solver_helper_fn,
            "preconditioner_order": preconditioner_order,
            "krylov_tolerance": krylov_tolerance,
            "max_linear_iters": max_linear_iters,
            "linear_correction_type": linear_correction_type,
            "newton_tolerance": newton_tolerance,
            "max_newton_iters": max_newton_iters,
            "newton_damping": newton_damping,
            "newton_max_backtracks": newton_max_backtracks,
            "tableau": tableau,
            "beta": 1.0,
            "gamma": 1.0,
            "M": mass,
        }
        if dt is not None:
            config_kwargs["dt"] = dt
        
        config = FIRKStepConfig(**config_kwargs)
        
        if tableau.has_error_estimate:
            defaults = FIRK_ADAPTIVE_DEFAULTS
        else:
            defaults = FIRK_FIXED_DEFAULTS
        
        super().__init__(config, defaults)

    def build_implicit_helpers(
        self,
    ) -> Callable:
        """Construct the nonlinear solver chain used by implicit methods."""

        precision = self.precision
        config = self.compile_settings
        tableau = config.tableau
        beta = config.beta
        gamma = config.gamma
        mass = config.M
        stage_count = config.stage_count
        all_stages_n = config.all_stages_n

        get_fn = config.get_solver_helper_fn

        stage_coefficients = [list(row) for row in tableau.a]
        stage_nodes = list(tableau.c)

        residual = get_fn(
            "n_stage_residual",
            beta=beta,
            gamma=gamma,
            mass=mass,
            stage_coefficients=stage_coefficients,
            stage_nodes=stage_nodes,
        )

        operator = get_fn(
            "n_stage_linear_operator",
            beta=beta,
            gamma=gamma,
            mass=mass,
            stage_coefficients=stage_coefficients,
            stage_nodes=stage_nodes,
        )

        preconditioner = get_fn(
            "n_stage_neumann_preconditioner",
            beta=beta,
            gamma=gamma,
            preconditioner_order=config.preconditioner_order,
            stage_coefficients=stage_coefficients,
            stage_nodes=stage_nodes,
        )

        krylov_tolerance = config.krylov_tolerance
        max_linear_iters = config.max_linear_iters
        correction_type = config.linear_correction_type

        linear_solver = inst_linear_solver_factory(
            operator,
            n=all_stages_n,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
        )

        newton_tolerance = config.newton_tolerance
        max_newton_iters = config.max_newton_iters
        newton_damping = config.newton_damping
        newton_max_backtracks = config.newton_max_backtracks

        nonlinear_solver = inst_newton_krylov_solver_factory(
            residual_function=residual,
            linear_solver=linear_solver,
            n=all_stages_n,
            tolerance=newton_tolerance,
            max_iters=max_newton_iters,
            damping=newton_damping,
            max_backtracks=newton_max_backtracks,
            precision=precision,
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
        n_drivers: int = 0,
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the FIRK device step."""

        config = self.compile_settings
        tableau = config.tableau
        nonlinear_solver = solver_fn
        stage_count = self.stage_count
        all_stages_n = config.all_stages_n
        

        has_driver_function = driver_function is not None
        has_error = self.is_adaptive
        stage_rhs_coeffs = tableau.typed_rows(tableau.a, numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        typed_zero = numba_precision(0.0)
        error_weights = tableau.error_weights(numba_precision)
        if error_weights is None or not has_error:
            error_weights = tuple(typed_zero for _ in range(stage_count))
        stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)

        # Last-step caching optimization (issue #163):
        # Replace streaming accumulation with direct assignment when
        # stage matches b or b_hat row in coupling matrix.
        # accumulates_output = 1
        accumulates_output = tableau.accumulates_output
        accumulates_error = tableau.accumulates_error
        b_row = tableau.b_matches_a_row
        b_hat_row = tableau.b_hat_matches_a_row

        ends_at_one = stage_time_fractions[-1] == numba_precision(1.0)

        solver_shared_elements = self.solver_shared_elements
        stage_driver_total = stage_count * n_drivers
        drivers_start = solver_shared_elements
        drivers_end = solver_shared_elements + stage_driver_total
        stages_start = drivers_end
        stages_end = stages_start + all_stages_n
        # no cover: start
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
                int32[:],
            ),
            device=True,
            inline=True,
        )
        def step(
            state,
            proposed_state,
            parameters,
            driver_coeffs,
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
            proposed_drivers_out,
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
            counters,
        ):
            stage_state = cuda.local.array(n, numba_precision)
            observable_count = proposed_observables.shape[0]

            current_time = time_scalar
            end_time = current_time + dt_scalar

            current_time = time_scalar
            end_time = time_scalar + dt_scalar

            stage_increment = shared[stages_start:stages_end]
            stage_driver_stack = shared[drivers_start:drivers_end]
            solver_scratch = shared[:solver_shared_elements]
            stage_rhs_flat = solver_scratch[:all_stages_n]
            status_code = int32(0)

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] = state[idx]
                if has_error and accumulates_error:
                    error[idx] = typed_zero

            if has_driver_function:
                for stage_idx in range(stage_count):
                    stage_time = time_scalar + (
                        stage_time_fractions[stage_idx] * dt_scalar
                    )
                    driver_offset = stage_idx * n_drivers
                    driver_slice = stage_driver_stack[
                        driver_offset : driver_offset + n_drivers
                    ]
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        driver_slice,
                    )

            solver_ret = nonlinear_solver(
                stage_increment,
                parameters,
                stage_driver_stack,
                current_time,
                dt_scalar,
                typed_zero,
                state,
                solver_scratch,
                counters,
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
            status_code |= solver_ret

            for stage_idx in range(stage_count):
                stage_time = (
                    current_time + dt_scalar * stage_time_fractions[stage_idx]
                )

                if has_driver_function:
                    stage_base = stage_idx * n_drivers
                    stage_slice = stage_driver_stack[
                        stage_base:stage_base + n_drivers
                    ]
                    for idx in range(n_drivers):
                        proposed_drivers[idx] = stage_slice[idx]

                for idx in range(n):
                    value = state[idx]
                    for contrib_idx in range(stage_count):
                        coeff = stage_rhs_coeffs[stage_idx][contrib_idx]
                        if coeff != typed_zero:
                            value += (
                                coeff * stage_increment[contrib_idx * n + idx]
                            )
                    stage_state[idx] = value

                for idx in range(n):
                    stage_states[stage_idx, idx] = stage_state[idx]
                    stage_increments[stage_idx, idx] = stage_increment[stage_idx * n + idx]
                    jacobian_updates[stage_idx, idx] = typed_zero


                # Capture precalculated outputs if tableau allows
                if not accumulates_output:
                    if b_row == stage_idx:
                        for idx in range(n):
                            proposed_state[idx] = stage_state[idx]
                if not accumulates_error:
                    if b_hat_row == stage_idx:
                        for idx in range(n):
                            error[idx] = stage_state[idx]

                # If error and output can be derived from stage_state,
                # don't bother evaluating f at each stage.
                do_more_work = (
                    has_error and accumulates_error
                ) or accumulates_output

                if do_more_work:
                    observables_function(
                        stage_state,
                        parameters,
                        proposed_drivers,
                        proposed_observables,
                        stage_time,
                    )

                    for obs_idx in range(observable_count):
                        stage_observables[stage_idx, obs_idx] = (
                            proposed_observables[obs_idx]
                        )

                    stage_rhs = stage_rhs_flat[stage_idx * n : (stage_idx + 1) * n]
                    dxdt_fn(
                            stage_state,
                            parameters,
                            proposed_drivers,
                            proposed_observables,
                            stage_rhs,
                            stage_time,
                    )

            #use a Kahan summation algorithm to reduce floating point errors
            #see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            if accumulates_output:
                for idx in range(n):
                    solution_acc = typed_zero
                    compensation = typed_zero
                    for stage_idx in range(stage_count):
                        rhs_value = stage_rhs_flat[stage_idx * n + idx]
                        term = (solution_weights[stage_idx] * rhs_value -
                                compensation)
                        temp = solution_acc + term
                        compensation = (temp - solution_acc) - term
                        solution_acc += solution_weights[stage_idx] * rhs_value
                    proposed_state[idx] = state[idx] + solution_acc * dt_scalar

            if has_error and accumulates_error:
                # Standard accumulation path for error
                for idx in range(n):
                    error_acc = typed_zero
                    for stage_idx in range(stage_count):
                        rhs_value = stage_rhs_flat[stage_idx * n + idx]
                        error_acc += error_weights[stage_idx] * rhs_value
                    error[idx] = dt_scalar * error_acc

            if not ends_at_one:
                if has_driver_function:
                    driver_function(
                        end_time,
                        driver_coeffs,
                        proposed_drivers,
                    )

                observables_function(
                    proposed_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    end_time,
                )

            if not accumulates_error:
                for idx in range(n):
                    error[idx] = proposed_state[idx] - error[idx]

            return status_code

        # no cover: end
        return StepCache(step=step, nonlinear_solver=nonlinear_solver)

    @property
    def is_multistage(self) -> bool:
        """Return ``True`` as the method has multiple stages."""

        return self.stage_count > 1

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` when the tableau supplies an error estimate."""

        return self.tableau.has_error_estimate

    @property
    def shared_memory_required(self) -> int:
        """Return the number of precision entries required in shared memory."""

        config = self.compile_settings
        stage_driver_total = config.stage_count * config.n_drivers
        return (
            self.solver_shared_elements
            + stage_driver_total
            + config.all_stages_n
        )

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""
        return self.compile_settings.n

    @property
    def persistent_local_required(self) -> int:
        """Return the number of persistent local entries required."""

        return 0

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""

        return self.compile_settings.stage_count

    @property
    def solver_shared_elements(self) -> int:
        """Return solver scratch elements accounting for flattened stages."""

        return 3 * self.compile_settings.all_stages_n

    @property
    def algorithm_shared_elements(self) -> int:
        """Return additional shared memory required by the algorithm."""

        return 0

    @property
    def algorithm_local_elements(self) -> int:
        """Return persistent local memory required by the algorithm."""

        return 0

    @property
    def is_implicit(self) -> bool:
        """Return ``True`` because the method solves nonlinear systems."""

        return True

    @property
    def order(self) -> int:
        """Return the classical order of accuracy."""

        return self.tableau.order

    @property
    def threads_per_step(self) -> int:
        """Return the number of CUDA threads that advance one state."""

        return 1
