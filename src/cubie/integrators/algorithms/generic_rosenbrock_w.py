"""Rosenbrock-W integration step as described in (5.2) in Lang & Verwer (2001).

 Lang, J., Verwer, J. ROS3P—An Accurate Third-Order Rosenbrock Solver Designed
 for Parabolic Problems. BIT Numerical Mathematics 41, 731–738 (2001).
 https://doi.org/10.1023/A:1021900219772"""

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
from cubie.integrators.matrix_free_solvers import linear_solver_cached_factory


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

    tableau: RosenbrockTableau = attrs.field(default=DEFAULT_ROSENBROCK_TABLEAU)
    time_derivative_fn: Optional[Callable] = attrs.field(default=None)
    driver_del_t: Optional[Callable] = attrs.field(default=None)


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
        driver_del_t_function: Optional[Callable] = None,
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
            driver_del_t=driver_del_t_function,
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
    ) -> Tuple[Callable, Callable, Callable, Callable]:
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
        )

        time_derivative_rhs = get_fn("time_derivative_rhs")

        return (
            linear_solver,
            prepare_jacobian,
            cached_jvp,
            time_derivative_rhs,
        )

    def build(self) -> StepCache:
        """Create and cache the device helpers for the implicit algorithm.
        Rosenbrock gets its own override due to its use of time-derivative
        functions.

        Returns
        -------
        StepCache
            Container with the compiled step and nonlinear solver.
        """

        solver_fn = self.build_implicit_helpers()
        config = self.compile_settings
        dxdt_fn = config.dxdt_function
        driver_del_t = config.driver_del_t
        numba_precision = config.numba_precision
        n = config.n
        dt = config.dt
        observables_function = config.observables_function
        driver_function = config.driver_function

        return self.build_step(
            solver_fn,
            dxdt_fn,
            observables_function,
            driver_function,
            driver_del_t,
            numba_precision,
            n,
            dt
        )

    def build_step(
        self,
        solver_fn: Callable,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        driver_del_t: Optional[Callable],
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
            time_derivative_rhs,
        ) = solver_fn

        stage_count = tableau.stage_count
        multistage = stage_count > 1
        has_driver_function = driver_function is not None
        first_same_as_last = self.first_same_as_last
        can_reuse_accepted_start = self.can_reuse_accepted_start
        has_error = self.is_adaptive

        a_coeffs = tableau.typed_rows(tableau.a, numba_precision)
        c_coeffs = tableau.typed_rows(tableau.C, numba_precision)
        stage_gamma_shifts = tableau.typed_vector(
                tableau.gamma_i, numba_precision
        )
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        typed_zero = numba_precision(0.0)
        error_weights = tableau.error_weights(numba_precision)
        if error_weights is None or not has_error:
            error_weights = tuple(typed_zero for _ in range(stage_count))
        stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)

        stage_buffer_n = stage_count * n
        cached_auxiliary_count = self.cached_auxiliary_count
        del_t_start = 0
        del_t_end = n
        stage_state_start = del_t_end
        stage_state_end = stage_state_start + n
        stage_rhs_start = stage_state_end
        stage_rhs_end = stage_rhs_start + n
        acc_start = stage_rhs_end
        acc_end = stage_buffer_n
        aux_start = acc_start
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
            # ----------------------------------------------------------- #
            # Shared and local buffer guide:
            # stage_accumulator: size (stage_count-1) * n, shared memory.
            #   Default behaviour:
            #       - Holds each stage rhs * dt until future stages need it.
            #       - Slice k stores the contribution consumed by stage k+1.
            # jacobian_accumulator: size (stage_count-1) * n, shared memory.
            #   Default behaviour:
            #       - Stores saved J*v products for later stage corrections.
            #       - Each slice pairs with the future diagonal solve it feeds.
            #   Reuse:
            #       - stage_increment: first slice (size n)
            #           - Keeps the accepted increment for the current stage.
            #           - During stage-0 solve we park it in proposed_state
            #             while stage 1 reuses the accumulator slice.
            # cached_auxiliaries: size cached_auxiliary_count, shared memory.
            #   Default behaviour:
            #       - Provides Jacobian helper data prepared before the loop.
            #       - Remains read-only for the entire stage loop.
            # stage_rhs: size n, per-thread local memory.
            #   Default behaviour:
            #       - Starts as the stage rhs
            #   Reuse:
            #       - jacobian_stage_product:
            #           - Holds the J*v product. from evaluation to
            #           accumulation
            #           - Lifetimes are disjoint, so reuse stays safe.
            # ----------------------------------------------------------- #

            dt_value = dt_scalar
            current_time = time_scalar
            end_time = current_time + dt_value

            time_derivative = shared[del_t_start:del_t_end]
            stage_state = shared[stage_state_start:stage_state_end]
            stage_rhs = shared[stage_rhs_start:stage_rhs_end]
            f_stage_store = shared[acc_start:acc_end]
            cached_auxiliaries = shared[aux_start:aux_end]

            idt = numba_precision(1.0) / dt_value

            prepare_jacobian(
                state,
                parameters,
                drivers_buffer,
                current_time,
                cached_auxiliaries,
            )

            #Evaluate del_t term at t_n, y_n
            if has_driver_function:
                driver_del_t(
                        current_time,
                        driver_coeffs,
                        proposed_drivers,
                )

            time_derivative_rhs(
                    state,
                    parameters,
                    drivers_buffer,
                    proposed_drivers,
                    observables,
                    time_derivative,
                    current_time,
            )

            for idx in range(n):
                proposed_state[idx] = state[idx]
                if has_error:
                    error[idx] = typed_zero

            status_code = int32(0)
            stage_time = current_time + dt_value * stage_time_fractions[0]

            # --------------------------------------------------------------- #
            #            Stage 0: may use cached values                       #
            # --------------------------------------------------------------- #

            if can_reuse_accepted_start:
                dxdt_fn(
                    state,
                    parameters,
                    drivers_buffer,
                    observables,
                    stage_rhs,
                    current_time,
                )

            else:
                # We're not at time = 0, recalculate f(y_0, t_0)
                if has_driver_function:
                    driver_function(
                        current_time,
                        driver_coeffs,
                        proposed_drivers,
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
                # No accumulated contributions at stage 0.
                stage_rhs[idx] = (dt_value * stage_rhs[idx] +
                                  stage_gamma_shifts[0] * time_derivative[idx])

            # Use final stage solution from previous step as initial guess.
            # Zerod on loop start
            stage_increment = f_stage_store[:n]
            final_base = n * (stage_count - 1)
            for idx in range(n):
                stage_increment = f_stage_store[final_base + idx]

            status_code |= linear_solver(
                state,
                parameters,
                drivers_buffer,
                cached_auxiliaries,
                stage_time,
                dt_value,
                numba_precision(1.0),
                stage_rhs,
                stage_increment,
            )


            for idx in range(n):
                proposed_state[idx] += stage_increment[idx] * solution_weights[0]
                if has_error:
                    error[idx] += stage_increment[idx] * error_weights[0]
            # --------------------------------------------------------------- #
            #            Stages 1-s: must refresh all values                  #
            # --------------------------------------------------------------- #

            for stage_idx in range(1, stage_count):
                # Fill buffers with previous step's contributions
                prev_idx = stage_idx - 1
                stage_gamma = stage_gamma_shifts[stage_idx]

                # Get base state for F(t + c_i * dt, Y_n + sum(a_ij * Y_nj))
                for idx in range(n):
                    stage_state[idx] = state[idx]
                for predecessor_offset in range(stage_idx):
                    for idx in range(n):
                        base = predecessor_offset * n + idx
                        prior_val = f_stage_store[base + idx]
                        a_coeff = a_coeffs[predecessor_offset][prev_idx]
                        stage_state[idx] += a_coeff * prior_val

                stage_time = (
                    current_time + dt_value * stage_time_fractions[stage_idx]
                )

                # Get t + c_i * dt parts
                if has_driver_function:
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        proposed_drivers,
                    )

                observables_function(
                    stage_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                dxdt_fn(
                    stage_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                # Add C_ij*Y_j/dt + dt * gamma_i * d/dt terms to rhs before
                # solve
                for idx in range(n):
                    prior_contribution = numba_precision(0.0)

                    # f_stage_store access pattern n-strided - OK in shared mem
                    for predecessor_offset in range(stage_idx):
                        prior_idx = predecessor_offset * n + idx
                        prior_val = f_stage_store[prior_idx]
                        c_coeff = c_coeffs[predecessor_offset][prev_idx]
                        prior_contribution += c_coeff * prior_val

                    stage_inc = stage_rhs[idx]
                    deriv_val = stage_gamma * time_derivative[idx]
                    stage_rhs[idx] = (
                        prior_contribution * idt + stage_inc + deriv_val
                    )


                # Alias slice of stage storage for convenience/readability
                stage_increment = f_stage_store[
                    n * stage_idx : n * (stage_idx + 1)
                ]

                # Use previous stage's solution as a guess for this stage
                previous_base = n * (stage_idx - 1)
                for idx in range(n):
                    stage_increment[idx] = f_stage_store[previous_base + idx]

                status_code |= linear_solver(
                    state,
                    parameters,
                    drivers_buffer,
                    cached_auxiliaries,
                    stage_time,
                    dt_value,
                    numba_precision(1.0),
                    stage_rhs,
                    stage_increment,
                )

                solution_weight = solution_weights[stage_idx]
                error_weight = error_weights[stage_idx]
                for idx in range(n):
                    increment = stage_increment[idx]
                    proposed_state[idx] += solution_weight * increment
                    if has_error:
                        error[idx] += error_weight * increment

            # ----------------------------------------------------------- #
            # Apply final dt scaling at end to reduce round-off error
            for idx in range(n):
                proposed_state[idx] *= dt_value
                if has_error:
                    error[idx] *= dt_value

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
        accumulator_span = self.stage_count * self.n
        cached_auxiliary_count = self.cached_auxiliary_count
        shared_buffers = 3 * self.n# stage_rhs, time_derivative, stage_state
        return accumulator_span + cached_auxiliary_count + shared_buffers

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""

        return 0

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

