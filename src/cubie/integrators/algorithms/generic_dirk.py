"""Diagonally implicit Runge–Kutta integration step implementation."""

from typing import Callable, Optional

import attrs
import numpy as np
from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DEFAULT_DIRK_TABLEAU,
    DIRKTableau,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ImplicitStepConfig,
    ODEImplicitStep,
)
from cubie.integrators.matrix_free_solvers import (
    linear_solver_factory,
    newton_krylov_solver_factory,
)


DIRK_DEFAULTS = StepControlDefaults(
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
class DIRKStepConfig(ImplicitStepConfig):
    """Configuration describing the DIRK integrator."""

    tableau: DIRKTableau = attrs.field(
        default=DEFAULT_DIRK_TABLEAU,
    )


class DIRKStep(ODEImplicitStep):
    """Diagonally implicit Runge–Kutta step with an embedded error estimate."""

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
        tableau: DIRKTableau = DEFAULT_DIRK_TABLEAU,
    ) -> None:
        """Initialise the DIRK step configuration."""

        mass = np.eye(n, dtype=precision)
        config = DIRKStepConfig(
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
        super().__init__(config, DIRK_DEFAULTS)

    def build_implicit_helpers(
        self,
    ) -> Callable:
        """Construct the nonlinear solver chain used by implicit methods."""

        precision = self.precision
        config = self.compile_settings
        beta = config.beta
        gamma = config.gamma
        mass = config.M
        preconditioner_order = config.preconditioner_order
        n = config.n

        get_fn = config.get_solver_helper_fn

        preconditioner = get_fn(
            "neumann_preconditioner", # neumann preconditioner cached?
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
            "linear_operator", # linear operator cached?
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        # prepare_jacobian = get_fn(
        #     "prepare_jac",
        #     preconditioner_order=preconditioner_order,
        # )
        # self._cached_auxiliary_count = get_fn("cached_aux_count")
        #TODO: evaluate if this is necessary; if we must also get a cached
        # solver. WHEN CAN WE CACHE THE DANG JACOBIAN IN A SOLVER
        krylov_tolerance = config.krylov_tolerance
        max_linear_iters = config.max_linear_iters
        correction_type = config.linear_correction_type

        linear_solver = linear_solver_factory(
            operator,
            n=n,
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
            n=n,
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
        """Compile the DIRK device step."""

        config = self.compile_settings
        tableau = config.tableau
        nonlinear_solver = solver_fn
        stage_count = tableau.stage_count
        has_driver_function = driver_function is not None
        has_error = self.is_adaptive


        stage_rhs_coeffs = tableau.typed_rows(tableau.a, numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        typed_zero = numba_precision(0.0)
        error_weights = tableau.error_weights(numba_precision)
        if error_weights is None or not has_error:
            error_weights = tuple(typed_zero for _ in range(stage_count))
        stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)
        diagonal_coeffs = tableau.diagonal(numba_precision)
        accumulator_length = max(stage_count - 1, 0) * n
        solver_shared_elements = self.solver_shared_elements
        cached_auxiliary_count = self.cached_auxiliary_count

        # Shared memory indices
        acc_start = 0
        acc_end = accumulator_length
        solver_start = acc_end
        solver_end = acc_end + solver_shared_elements
        cached_start = solver_end
        cached_end = solver_end + cached_auxiliary_count

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
            shared,
            persistent_local,
        ):
            stage_base = cuda.local.array(n, numba_precision)

            dt_value = dt_scalar
            current_time = time_scalar
            end_time = current_time + dt_value

            stage_accumulator = shared[acc_start:acc_end]
            solver_scratch = shared[solver_start:solver_end]
            stage_rhs = solver_scratch[:n]



            stage_state = stage_accumulator[:n]

            for idx in range(n):
                error[idx] = typed_zero
                stage_base[idx] = state[idx]
                stage_state[idx] = state[idx]
                proposed_state[idx] = state[idx]

            for idx in range(accumulator_length - n):
                stage_accumulator[idx+n] = typed_zero

            status_code = int32(0)

            # --------------------------------------------------------------- #
            #            Stage 0: operates out of supplied buffers            #
            # --------------------------------------------------------------- #

            stage_time = current_time + dt_value * stage_time_fractions[0]
            stage_drivers = proposed_drivers

            # Not required if c[0] = 0.0
            if has_driver_function:
                driver_function(
                    stage_time,
                    driver_coeffs,
                    stage_drivers,
                )

            status_code |= nonlinear_solver(
                stage_state,
                parameters,
                stage_drivers,
                dt_value,
                diagonal_coeffs[0],
                stage_base,
                solver_scratch,
            )

            observables_function(
                stage_state,
                parameters,
                stage_drivers,
                proposed_observables,
                stage_time,
            )

            dxdt_fn(
                stage_state,
                parameters,
                stage_drivers,
                proposed_observables,
                stage_rhs,
                stage_time,
            )

            solution_weight = solution_weights[0]
            error_weight = error_weights[0]
            for idx in range(n):
                rhs_value = stage_rhs[idx]
                increment = dt_value * rhs_value
                proposed_state[idx] += solution_weight * increment
                error[idx] += error_weight * increment

            # --------------------------------------------------------------- #
            #            Stages 1-s: must refresh obs/drivers                 #
            # --------------------------------------------------------------- #

            for stage_idx in range(1, stage_count):

                # Fill accumulators with previous step's contributions
                prev_idx = stage_idx - 1
                successor_range = stage_count - stage_idx

                # This loop order might look backwards, but it allows the
                # aliasing of stage_increment and stage_accumulator. Shared
                # memory buffers won't suffer from bad locality - if local
                # buffers have spilled into global memory, then you're at
                # worst-case locality anyhow, so this has no penalty.
                for idx in range(n):
                    if prev_idx == 0:
                        stage_accumulator[idx] = typed_zero
                    for successor_offset in range(successor_range):
                        successor_idx = stage_idx + successor_offset
                        base = (successor_idx - 1) * n
                        state_coeff = stage_rhs_coeffs[successor_idx][prev_idx]
                        contribution = state_coeff * stage_rhs[idx]
                        stage_accumulator[base + idx] += contribution

                # Position in accumulator
                stage_offset = (stage_idx - 1) * n
                stage_time = (
                    current_time + dt_value * stage_time_fractions[stage_idx]
                )

                if has_driver_function:
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        stage_drivers,
                    )

                stage_state = stage_accumulator[stage_offset:stage_offset + n]
                for idx in range(n):
                    stage_state[idx] += state[idx]
                    stage_base[idx] = stage_state[idx]

                status_code |= nonlinear_solver(
                    stage_state,
                    parameters,
                    stage_drivers,
                    dt_value,
                    diagonal_coeffs[stage_idx],
                    stage_base,
                    solver_scratch,
                )

                observables_function(
                    stage_state,
                    parameters,
                    stage_drivers,
                    proposed_observables,
                    stage_time,
                )
                dxdt_fn(
                    stage_state,
                    parameters,
                    stage_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                solution_weight = solution_weights[stage_idx]
                error_weight = error_weights[stage_idx]
                for idx in range(n):
                    rhs_value = stage_rhs[idx]
                    increment = dt_value * rhs_value
                    proposed_state[idx] += solution_weight * increment
                    error[idx] += error_weight * increment

            # --------------------------------------------------------------- #
            final_time = end_time
            if has_driver_function:
                driver_function(
                    final_time,
                    driver_coeffs,
                    proposed_drivers,
                )

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                final_time,
            )
            return status_code
        # no cover: end
        return StepCache(step=step, nonlinear_solver=nonlinear_solver)

    @property
    def is_multistage(self) -> bool:
        """Return ``True`` as the method has multiple stages."""
        return self.tableau.stage_count > 1


    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` because an embedded error estimate is produced."""
        return self.tableau.has_error_estimate


    @property
    def cached_auxiliary_count(self) -> int:
        """Return the number of cached auxiliary entries for the JVP.

        Lazily builds implicit helpers so as not to return an errant 'None'."""
        if self._cached_auxiliary_count is None:
            self.build_implicit_helpers()
        return self._cached_auxiliary_count

    @property
    def shared_memory_required(self) -> int:
        """Return the number of precision entries required in shared memory."""

        tableau = self.tableau
        stage_count = tableau.stage_count
        accumulator_span = max(stage_count - 1, 0) * self.compile_settings.n
        return (accumulator_span
            + self.solver_shared_elements
            + self.cached_auxiliary_count
        )

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""
        return 2 * self.compile_settings.n

    @property
    def persistent_local_required(self) -> int:
        """Return the number of persistent local entries required."""
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
