"""Rosenbrock-W integration step using a streamed accumulator layout."""

from typing import Callable, Optional, Tuple

import attrs
import numpy as np
from numba import cuda, int16, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ImplicitStepConfig,
    ODEImplicitStep,
)
from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    DEFAULT_ROSENBROCK_TABLEAU,
    ROSENBROCK_W6S4OS_TABLEAU,
    RosenbrockTableau,
)
from .matrix_free_solvers import linear_solver_cached_factory


ROSENBROCK_DEFAULTS = StepControlDefaults(
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
class RosenbrockWStepConfig(ImplicitStepConfig):
    """Configuration describing the Rosenbrock-W integrator."""

    tableau: RosenbrockTableau = attrs.field(
        default=DEFAULT_ROSENBROCK_TABLEAU,
    )


class GenericRosenbrockWStep(ODEImplicitStep):
    """Rosenbrock-W step with an embedded error estimate."""

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
        tableau: RosenbrockTableau = DEFAULT_ROSENBROCK_TABLEAU,
    ) -> None:
        """Initialise the Rosenbrock-W step configuration."""

        mass = np.eye(n, dtype=precision)
        tableau_value = tableau
        config = RosenbrockWStepConfig(
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
            tableau=tableau_value,
            beta=1.0,
            gamma=tableau_value.gamma,
            M=mass,
        )
        self._cached_auxiliary_count = None
        super().__init__(config, ROSENBROCK_DEFAULTS)

    def build_implicit_helpers(
        self,
    ) -> Tuple[Callable, Callable, Callable]:
        """Construct the nonlinear solver chain used by implicit methods.

        Returns
        -------
        tuple of Callables
            Linear solver function and Jacobian helpers for the Rosenbrock-W
            step.
        """
        precision = self.precision
        config = self.compile_settings
        beta = config.beta
        gamma = config.tableau.gamma
        mass = config.M
        preconditioner_order = config.preconditioner_order
        n = config.n

        get_fn = config.get_solver_helper_fn

        preconditioner = get_fn(
            "neumann_preconditioner_cached",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        linear_operator = get_fn(
            "linear_operator_cached",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        prepare_jacobian = get_fn(
            "prepare_jac",
            preconditioner_order=preconditioner_order,
        )
        self._cached_auxiliary_count = get_fn("cached_aux_count")

        cached_jvp = get_fn(
            "calculate_cached_jvp",
            beta=precision(0.0),
            gamma=precision(-1.0),
            mass=mass,
            preconditioner_order=preconditioner_order,
        )


        krylov_tolerance = config.krylov_tolerance
        max_linear_iters = config.max_linear_iters
        correction_type = config.linear_correction_type

        linear_solver = linear_solver_cached_factory(
            linear_operator,
            n=n,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
            precision=config.precision,
        )

        return linear_solver, prepare_jacobian, cached_jvp

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
        """Compile the Rosenbrock-W device step."""

        config = self.compile_settings
        tableau = config.tableau
        (
            linear_solver,
            prepare_jacobian,
            cached_jvp,
        ) = solver_fn

        stage_count = tableau.stage_count
        multistage = stage_count > 1
        has_driver_function = driver_function is not None
        first_same_as_last = self.first_same_as_last
        can_reuse_accepted_start = self.can_reuse_accepted_start
        has_error = self.is_adaptive

        stage_rhs_coeffs = tableau.typed_rows(tableau.a, numba_precision)
        jacobian_update_coeffs = tableau.typed_rows(tableau.C, numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        typed_zero = numba_precision(0.0)
        error_weights = tableau.error_weights(numba_precision)
        if error_weights is None or not has_error:
            error_weights = tuple(typed_zero for _ in range(stage_count))
        stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)

        accumulator_length = max(stage_count - 1, 0) * n
        cached_auxiliary_count = self.cached_auxiliary_count
        acc_start = 0
        acc_end = accumulator_length
        jac_start = acc_end
        jac_end = jac_start + accumulator_length
        aux_start = jac_end
        aux_end = aux_start + cached_auxiliary_count

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
            stage_drivers_out,
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
            stage_rhs = cuda.local.array(n, numba_precision)
            jacobian_stage_product = stage_rhs # lifetimes are disjoint - reuse

            observable_count = proposed_observables.shape[0]

            dt_value = dt_scalar
            current_time = time_scalar
            end_time = current_time + dt_value

            stage_accumulator = shared[acc_start:acc_end]
            jacobian_product_accumulator = shared[jac_start:jac_end]
            cached_auxiliaries = shared[aux_start:aux_end]

            idt = numba_precision(1.0) / dt_value

            if multistage:
                # Alias "increment" array over first stage accumulator -
                # their lifetimes are disjoint.
                stage_increment = jacobian_product_accumulator[:n]
            else:
                stage_increment = cuda.local.array(n, numba_precision)

            prepare_jacobian(
                state,
                parameters,
                drivers_buffer,
                current_time,
                cached_auxiliaries,
            )

            for idx in range(n):
                if has_error:
                    error[idx] = typed_zero

            for idx in range(n):
                stage_states[0, idx] = state[idx]
            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = observables[obs_idx]

            status_code = int32(0)

            stage_time = current_time + dt_value * stage_time_fractions[0]

            # --------------------------------------------------------------- #
            #            Stage 0: may use cached values                       #
            # --------------------------------------------------------------- #
            if can_reuse_accepted_start:
                # Get dxdt at time=0 and use that to calculate gradient
                dxdt_fn(
                        state,
                        parameters,
                        drivers_buffer,
                        observables,
                        stage_rhs,
                        current_time,
                )

            else:
                # We're not at time = 0, recalculate rhs
                stage_time = (
                    current_time + dt_value * stage_time_fractions[0]
                )
                if has_driver_function:
                    driver_function(
                        current_time,
                        driver_coeffs,
                    )
                observables_function(
                        state,
                        parameters,
                        proposed_drivers,
                        proposed_observables,
                        stage_time,
                )

                dxdt_fn(
                        state,
                        parameters,
                        proposed_drivers,
                        proposed_observables,
                        stage_rhs,
                        stage_time,
                )

            for idx in range(n):
                stage_derivatives[0, idx] = stage_rhs[idx]
                stage_states[0, idx] = state[idx]
                stage_rhs[idx] = dt_value * stage_rhs[idx]
                proposed_state[idx] = typed_zero
                residuals[0, idx] = stage_rhs[idx]
            for driver_idx in range(stage_drivers_out.shape[1]):
                stage_drivers_out[0, driver_idx] = drivers_buffer[driver_idx]

            initial_linear_slot = int32(0)
            solver_ret = linear_solver(
                state,
                parameters,
                drivers_buffer,
                cached_auxiliaries,
                stage_time,
                dt_value,
                stage_rhs,
                stage_increment,
                initial_linear_slot,
                linear_initial_guesses,
                linear_iteration_guesses,
                linear_residuals,
                linear_squared_norms,
                linear_preconditioned_vectors,
            )
            status_code |= solver_ret

            for idx in range(n):
                stage_increments[0, idx] = stage_increment[idx]
                jacobian_updates[0, idx] = stage_increment[idx]

            for idx in range(n):
                increment = stage_increment[idx]
                # It's janky, but we're going to stash the increment
                # here until the stage accumulator frees up again.
                proposed_state[idx] = increment
                if has_error:
                    error[idx] += error_weights[0] * increment

            cached_jvp(
                state,
                parameters,
                drivers_buffer,
                cached_auxiliaries,
                stage_time,
                stage_increment,
                jacobian_stage_product,
            )

            for idx in range(n, accumulator_length):
                # zero non-cache-aliased accumulator entries
                stage_accumulator[idx] = typed_zero
                jacobian_product_accumulator[idx] = typed_zero

            # --------------------------------------------------------------- #
            #            Stages 1-s: must refresh all values                  #
            # --------------------------------------------------------------- #

            for stage_idx in range(1, stage_count):

                # Fill accumulators with previous step's contributions
                prev_idx = stage_idx - 1
                successor_range = stage_count - stage_idx

                for idx in range(n):
                    increment_n = stage_increment[idx]
                    jacobian_value = jacobian_stage_product[idx]
                    if prev_idx == 0:
                        # Zero first-stage cache portions of accumulators
                        stage_accumulator[idx] = typed_zero
                        jacobian_product_accumulator[idx] = typed_zero
                    for successor_offset in range(successor_range):
                        succsr_idx = stage_idx + successor_offset
                        state_coeff = stage_rhs_coeffs[succsr_idx][prev_idx]
                        jacobian_coeff = jacobian_update_coeffs[succsr_idx][
                            prev_idx
                        ]
                        base = (succsr_idx - 1) * n + idx
                        stage_accumulator[base] += state_coeff * increment_n
                        jacobian_product_accumulator[base] += (
                            jacobian_coeff * jacobian_value
                        )

                stage_offset = (stage_idx - 1) * n
                stage_time = (
                    current_time + dt_value * stage_time_fractions[stage_idx]
                )

                stage_state = stage_accumulator[stage_offset:stage_offset + n]
                for idx in range(n):
                    stage_state[idx] += state[idx]

                for idx in range(n):
                    stage_states[stage_idx, idx] = stage_state[idx]
                    stage_increments[stage_idx, idx] = typed_zero

                if has_driver_function:
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        proposed_drivers,
                    )
                for driver_idx in range(stage_drivers_out.shape[1]):
                    stage_drivers_out[stage_idx, driver_idx] = proposed_drivers[driver_idx]

                observables_function(
                    stage_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                for obs_idx in range(observable_count):
                    stage_observables[stage_idx, obs_idx] = proposed_observables[obs_idx]

                dxdt_fn(
                    stage_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                for idx in range(n):
                    stage_derivatives[stage_idx, idx] = stage_rhs[idx]

                for idx in range(n):
                    rhs_value = stage_rhs[idx]
                    jacobian_term = jacobian_product_accumulator[
                        stage_offset + idx
                    ]
                    stage_rhs[idx] = dt_value * (rhs_value + jacobian_term)

                    if prev_idx == 0:
                        # Reclaim stage increment as a solver guess.
                        # the lower n elements of stage accumulator are
                        # only used for stage increment after stage 1
                        stage_increment[idx] = proposed_state[idx]
                        # Belatedly adjust solution with weight from tableau,
                        # add state back in.
                        proposed_state[idx] *= solution_weights[0]
                        proposed_state[idx] += state[idx]

                for idx in range(n):
                    residuals[stage_idx, idx] = stage_rhs[idx]

                stage_linear_slot = int32(stage_idx)
                solver_ret = linear_solver(
                    state,
                    parameters,
                    drivers_buffer,
                    cached_auxiliaries,
                    stage_time,
                    dt_value,
                    stage_rhs,
                    stage_increment,
                    stage_linear_slot,
                    linear_initial_guesses,
                    linear_iteration_guesses,
                    linear_residuals,
                    linear_squared_norms,
                    linear_preconditioned_vectors,
                )
                status_code |= solver_ret

                for idx in range(n):
                    jacobian_updates[stage_idx, idx] = stage_increment[idx]
                    stage_increments[stage_idx, idx] = stage_increment[idx]

                solution_weight = solution_weights[stage_idx]
                error_weight = error_weights[stage_idx]
                for idx in range(n):
                    increment = stage_increment[idx]
                    proposed_state[idx] += solution_weight * increment
                    if has_error:
                        error[idx] += error_weight * increment

                if stage_idx < stage_count - 1:
                    cached_jvp(
                        state,
                        parameters,
                        drivers_buffer,
                        cached_auxiliaries,
                        stage_time,
                        stage_increment,
                        jacobian_stage_product,
                    )
            # ----------------------------------------------------------- #
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
        return StepCache(step=step)

    @property
    def is_multistage(self) -> bool:
        """Return ``True`` as the method has multiple stages."""
        return self.tableau.stage_count > 1

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` if algorithm calculates an error estimate."""
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
        cached_auxiliary_count = self.cached_auxiliary_count
        return 2 * accumulator_span + cached_auxiliary_count

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""

        return 4 * self.compile_settings.n

    @property
    def persistent_local_required(self) -> int:
        """Return the number of persistent local entries required."""
        return 0

    @property
    def is_implicit(self) -> bool:
        """Return ``True`` because the method solves linear systems."""
        return True

    @property
    def order(self) -> int:
        """Return the classical order of accuracy."""
        return self.tableau.order

    @property
    def threads_per_step(self) -> int:
        """Return the number of CUDA threads that advance one state."""
        return 1


__all__ = [
    "GenericRosenbrockWStep",
    "RosenbrockWStepConfig",
    "RosenbrockTableau",
    "ROSENBROCK_W6S4OS_TABLEAU",
]

