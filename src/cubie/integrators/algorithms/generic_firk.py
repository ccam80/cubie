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
from cubie.integrators.matrix_free_solvers import (
    linear_solver_factory,
    newton_krylov_solver_factory,
)


FIRK_DEFAULTS = StepControlDefaults(
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
        dt: Optional[float],
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
    ) -> None:
        """Initialise the FIRK step configuration."""

        mass = np.eye(n, dtype=precision)
        config = FIRKStepConfig(
            precision=precision,
            n=n,
            dt=dt,
            dxdt_function=dxdt_function,
            observables_function=observables_function,
            driver_function=driver_function,
            get_solver_helper_fn=get_solver_helper_fn,
            preconditioner_order=preconditioner_order,
            krylov_tolerance=krylov_tolerance,
            max_linear_iters=max_linear_iters,
            linear_correction_type=linear_correction_type,
            newton_tolerance=newton_tolerance,
            max_newton_iters=max_newton_iters,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
            tableau=tableau,
            beta=1.0,
            gamma=1.0,
            M=mass,
        )
        self._cached_auxiliary_count = 0
        super().__init__(config, FIRK_DEFAULTS)

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

        linear_solver = linear_solver_factory(
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

        nonlinear_solver = newton_krylov_solver_factory(
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

        solver_shared_elements = self.solver_shared_elements

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
            driver_coeffs,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,
            dt_scalar,
            time_scalar,
            first_step_flag,
            accepted_flag,
            shared,
            persistent_local,
        ):
            stage_increment = cuda.local.array(all_stages_n, numba_precision)
            stage_state = cuda.local.array(n, numba_precision)

            dt_value = dt_scalar
            current_time = time_scalar
            end_time = current_time + dt_value

            solver_scratch = shared[:solver_shared_elements]
            stage_rhs_flat = solver_scratch[:all_stages_n]

            status_code = int32(0)

            for idx in range(all_stages_n):
                stage_increment[idx] = typed_zero
                stage_rhs_flat[idx] = typed_zero

            for idx in range(n):
                proposed_state[idx] = state[idx]
                if has_error:
                    error[idx] = typed_zero

            if has_driver_function:
                driver_function(
                    current_time,
                    driver_coeffs,
                    proposed_drivers,
                )
            else:
                for idx in range(drivers_buffer.shape[0]):
                    proposed_drivers[idx] = drivers_buffer[idx]

            status_code |= nonlinear_solver(
                stage_increment,
                parameters,
                proposed_drivers,
                current_time,
                dt_value,
                typed_zero,
                state,
                solver_scratch,
            )

            for stage_idx in range(stage_count):
                stage_time = (
                    current_time + dt_value * stage_time_fractions[stage_idx]
                )
                if has_driver_function:
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        proposed_drivers,
                    )

                for comp_idx in range(n):
                    value = state[comp_idx]
                    for contrib_idx in range(stage_count):
                        coeff = stage_rhs_coeffs[stage_idx][contrib_idx]
                        if coeff != typed_zero:
                            value += (
                                coeff
                                * stage_increment[
                                    contrib_idx * n + comp_idx
                                ]
                            )
                    stage_state[comp_idx] = value

                observables_function(
                    stage_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                stage_rhs = stage_rhs_flat[
                    stage_idx * n:(stage_idx + 1) * n
                ]
                dxdt_fn(
                    stage_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

            for comp_idx in range(n):
                solution_acc = typed_zero
                error_acc = typed_zero
                for stage_idx in range(stage_count):
                    rhs_value = stage_rhs_flat[stage_idx * n + comp_idx]
                    solution_acc += solution_weights[stage_idx] * rhs_value
                    if has_error:
                        error_acc += error_weights[stage_idx] * rhs_value
                proposed_state[comp_idx] = (
                    state[comp_idx] + dt_value * solution_acc
                )
                if has_error:
                    error[comp_idx] = dt_value * error_acc

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
    def cached_auxiliary_count(self) -> int:
        """Return the number of cached auxiliary entries for the JVP."""

        return self._cached_auxiliary_count

    @property
    def shared_memory_required(self) -> int:
        """Return the number of precision entries required in shared memory."""

        return self.solver_shared_elements + self.cached_auxiliary_count

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""

        state_dim = self.compile_settings.n
        return self.stage_count * state_dim + state_dim

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

