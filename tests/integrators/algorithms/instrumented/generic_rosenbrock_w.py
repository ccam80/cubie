"""Rosenbrock-W integration step using a streamed accumulator layout."""

from typing import Callable, Optional, Tuple

import attrs
import numpy as np
from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.buffer_registry import buffer_registry
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
    RosenbrockTableau,
)
from .matrix_free_solvers import inst_linear_solver_cached_factory


ROSENBROCK_ADAPTIVE_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "pid",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
        "kp": 0.7,
        "ki": -0.4,
        "deadband_min": 1.0,
        "deadband_max": 1.1,
        "min_gain": 0.5,
        "max_gain": 2.0,
    }
)

ROSENBROCK_FIXED_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)


@attrs.define
class RosenbrockWStepConfig(ImplicitStepConfig):
    """Configuration describing the Rosenbrock-W integrator."""

    tableau: RosenbrockTableau = attrs.field(
        default=DEFAULT_ROSENBROCK_TABLEAU,
    )
    time_derivative_fn: Optional[Callable] = attrs.field(default=None)
    driver_del_t: Optional[Callable] = attrs.field(default=None)
    stage_rhs_location: str = attrs.field(default='local')
    stage_store_location: str = attrs.field(default='local')
    cached_auxiliaries_location: str = attrs.field(default='local')


class GenericRosenbrockWStep(ODEImplicitStep):
    """Rosenbrock-W step with an embedded error estimate."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        preconditioner_order: int = 2,
        krylov_tolerance: float = 1e-6,
        max_linear_iters: int = 200,
        linear_correction_type: str = "minimal_residual",
        tableau: RosenbrockTableau = DEFAULT_ROSENBROCK_TABLEAU,
        time_derivative_function: Optional[Callable] = None,
        driver_del_t: Optional[Callable] = None,
        stage_rhs_location: Optional[str] = None,
        stage_store_location: Optional[str] = None,
        cached_auxiliaries_location: Optional[str] = None,
    ) -> None:
        """Initialise the Rosenbrock-W step configuration."""

        mass = np.eye(n, dtype=precision)
        tableau_value = tableau

        # Build config first so buffer registration can use config defaults
        config_kwargs = {
            "precision": precision,
            "n": n,
            "dxdt_function": dxdt_function,
            "observables_function": observables_function,
            "driver_function": driver_function,
            "driver_del_t": driver_del_t,
            "time_derivative_fn": time_derivative_function,
            "get_solver_helper_fn": get_solver_helper_fn,
            "preconditioner_order": preconditioner_order,
            "krylov_tolerance": krylov_tolerance,
            "max_linear_iters": max_linear_iters,
            "linear_correction_type": linear_correction_type,
            "tableau": tableau_value,
            "beta": 1.0,
            "gamma": tableau_value.gamma,
            "M": mass,
        }
        if stage_rhs_location is not None:
            config_kwargs["stage_rhs_location"] = stage_rhs_location
        if stage_store_location is not None:
            config_kwargs["stage_store_location"] = stage_store_location
        if cached_auxiliaries_location is not None:
            config_kwargs["cached_auxiliaries_location"] = cached_auxiliaries_location

        config = RosenbrockWStepConfig(**config_kwargs)
        self._cached_auxiliary_count = None

        # Clear any existing buffer registrations
        buffer_registry.clear_parent(self)

        # Calculate buffer sizes
        stage_store_elements = tableau.stage_count * n

        # Register algorithm buffers using config values
        buffer_registry.register(
            'rosenbrock_stage_rhs', self, n, config.stage_rhs_location,
            precision=precision
        )
        buffer_registry.register(
            'rosenbrock_stage_store', self, stage_store_elements,
            config.stage_store_location, precision=precision
        )
        # cached_auxiliaries registered with 0 size; updated in build_implicit_helpers
        buffer_registry.register(
            'rosenbrock_cached_auxiliaries', self, 0,
            config.cached_auxiliaries_location, precision=precision
        )

        # stage_cache: persistent when stage_store is local
        if config.stage_store_location == 'local':
            buffer_registry.register(
                'rosenbrock_stage_cache', self, n, 'local',
                persistent=True, precision=precision
            )
        else:
            # Aliases stage_store when shared
            buffer_registry.register(
                'rosenbrock_stage_cache', self, n, 'shared',
                aliases='rosenbrock_stage_store', precision=precision
            )

        if tableau.has_error_estimate:
            defaults = ROSENBROCK_ADAPTIVE_DEFAULTS
        else:
            defaults = ROSENBROCK_FIXED_DEFAULTS

        super().__init__(config, defaults)

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
        # Update buffer registry with the actual cached_auxiliary_count
        buffer_registry.update_buffer(
            'rosenbrock_cached_auxiliaries', self,
            size=self._cached_auxiliary_count
        )

        krylov_tolerance = config.krylov_tolerance
        max_linear_iters = config.max_linear_iters
        correction_type = config.linear_correction_type

        linear_solver = inst_linear_solver_cached_factory(
            linear_operator,
            n=n,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
            precision=config.precision,
        )

        time_derivative_rhs = get_fn("time_derivative_rhs")

        return linear_solver, prepare_jacobian, time_derivative_rhs

    def build(self) -> StepCache:
        """Create and cache the device helpers for the implicit algorithm."""

        solver_fn = self.build_implicit_helpers()
        config = self.compile_settings
        dxdt_fn = config.dxdt_function
        driver_del_t = config.driver_del_t
        numba_precision = config.numba_precision
        n = config.n
        n_drivers = config.n_drivers
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
            n_drivers,
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
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the Rosenbrock-W device step."""

        config = self.compile_settings
        precision = self.precision
        tableau = config.tableau
        (linear_solver, prepare_jacobian, time_derivative_rhs) = solver_fn

        n = int32(n)
        stage_count = int32(self.stage_count)
        stages_except_first = stage_count - int32(1)
        has_driver_function = driver_function is not None
        has_error = self.is_adaptive
        typed_zero = numba_precision(0.0)

        a_coeffs = tableau.typed_columns(tableau.a, numba_precision)
        C_coeffs = tableau.typed_columns(tableau.C, numba_precision)
        gamma_stages = tableau.typed_gamma_stages(numba_precision)
        gamma = numba_precision(tableau.gamma)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        error_weights = tableau.error_weights(numba_precision)
        if error_weights is None or not has_error:
            error_weights = tuple(typed_zero for _ in range(stage_count))
        stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)

        # Last-step caching optimization (issue #163):
        # Replace streaming accumulation with direct assignment when
        # stage matches b or b_hat row in coupling matrix.
        accumulates_output = tableau.accumulates_output
        accumulates_error = tableau.accumulates_error
        b_row = tableau.b_matches_a_row
        b_hat_row = tableau.b_hat_matches_a_row
        if b_row is not None:
            b_row = int32(b_row)
        if b_hat_row is not None:
            b_hat_row = int32(b_hat_row)

        # Get allocators from buffer registry
        alloc_stage_rhs = buffer_registry.get_allocator(
            'rosenbrock_stage_rhs', self
        )
        alloc_stage_store = buffer_registry.get_allocator(
            'rosenbrock_stage_store', self
        )
        alloc_cached_auxiliaries = buffer_registry.get_allocator(
            'rosenbrock_cached_auxiliaries', self
        )
        alloc_stage_cache = buffer_registry.get_allocator(
            'rosenbrock_stage_cache', self
        )

        # Stage store size for initialization
        stage_store_elements = tableau.stage_count * config.n

        # Compile-time flag for stage_store memory location
        stage_store_shared = config.stage_store_location == 'shared'

        # no cover: start
        @cuda.jit(
            # (
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision,
            #     numba_precision,
            #     int32,
            #     int32,
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     int32[::1],
            # ),
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
            counters,
        ):
            # ----------------------------------------------------------- #
            # Shared and local buffer guide:
            # stage_rhs: size n, shared or local memory.
            #   - Receives the stage right-hand side and doubles as a residual
            #     buffer before each linear solve.
            # stage_store: size stage_count * n, shared or local memory.
            #   - Slice i caches the accepted stage increment K_i.
            #   - Stage slices double as the initial guess for the following
            #     stage and provide the stage state when assembling rhs values.
            #   - The final slice stores the scaled d f / d t vector until the
            #     last stage recomputes it immediately before use.
            # cached_auxiliaries: size cached_auxiliary_count, shared or local.
            #   - Provides Jacobian helper data prepared before the loop.
            # ----------------------------------------------------------- #

            # ----------------------------------------------------------- #
            # Selective allocation from local or shared memory
            # ----------------------------------------------------------- #
            stage_rhs = alloc_stage_rhs(shared, persistent_local)
            stage_store = alloc_stage_store(shared, persistent_local)
            cached_auxiliaries = alloc_cached_auxiliaries(shared, persistent_local)

            # Initialize arrays
            for _i in range(n):
                stage_rhs[_i] = typed_zero
            for _i in range(stage_store_elements):
                stage_store[_i] = typed_zero
            # ----------------------------------------------------------- #

            observable_count = proposed_observables.shape[0]
            driver_count = proposed_drivers.shape[0]

            current_time = time_scalar
            end_time = current_time + dt_scalar
            final_stage_base = n * (stage_count - int32(1))
            time_derivative = stage_store[
                final_stage_base : final_stage_base + n
            ]

            inv_dt = numba_precision(1.0) / dt_scalar

            prepare_jacobian(
                state,
                parameters,
                drivers_buffer,
                current_time,
                cached_auxiliaries,
            )

            # Evaluate del_t term at t_n, y_n
            if has_driver_function:
                driver_del_t(
                    current_time,
                    driver_coeffs,
                    proposed_drivers,
                )
            else:
                proposed_drivers[:] = numba_precision(0.0)

            # Stage 0 uses cached value as initial guess.
            # When stage_store is shared, time_derivative persists between steps.
            # When stage_store is local, use persistent_local for caching.
            stage_increment = stage_store[:n]
            first_step = first_step_flag != int32(0)

            if stage_store_shared:
                # When shared, time_derivative persists automatically
                for idx in range(n):
                    stage_increment[idx] = time_derivative[idx]
            else:
                for idx in range(n):
                    stage_increment[idx] = persistent_local[idx]

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
                time_derivative[idx] *= dt_scalar # Prescale by dt_scalar once
                if has_error:
                    error[idx] = typed_zero

            # LOGGING
                stage_states[0, idx] = state[idx]
            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = observables[obs_idx]
            for driver_idx in range(driver_count):
                stage_drivers_out[0, driver_idx] = drivers_buffer[driver_idx]

            status_code = int32(0)
            stage_time = current_time + dt_scalar * stage_time_fractions[0]

            # Stage 0:
            dxdt_fn(
                state,
                parameters,
                drivers_buffer,
                observables,
                stage_rhs,
                current_time,
            )

            for idx in range(n):
                f_value = stage_rhs[idx]
                rhs_value = (
                        (f_value + gamma_stages[0] * time_derivative[idx])
                        * dt_scalar
                )
                stage_rhs[idx] = rhs_value * gamma

                # LOGGING
                stage_derivatives[0, idx] = f_value
                residuals[0, idx] = rhs_value

            stage_increment = stage_store[:n]
            final_base = n * (stage_count - 1)
            for idx in range(n):
                stage_increment[idx] = stage_store[final_base + idx]

            # Use stored copy as the initial guess for the first stage.

            base_state_placeholder = shared[int32(0):int32(0)]
            initial_linear_slot = int32(0)
            krylov_iters = cuda.local.array(1, dtype=int32)
            solver_ret = linear_solver(
                state,
                parameters,
                drivers_buffer,
                base_state_placeholder,
                cached_auxiliaries,
                stage_time,
                dt_scalar,
                numba_precision(1.0),
                stage_rhs,
                stage_increment,
                initial_linear_slot,
                linear_initial_guesses,
                linear_iteration_guesses,
                linear_residuals,
                linear_squared_norms,
                linear_preconditioned_vectors,
                krylov_iters,
            )
            status_code |= solver_ret

            for idx in range(n):
                increment = stage_increment[idx]
                if accumulates_output:
                    proposed_state[idx] += (solution_weights[int32(0)] *
                                            increment)
                if has_error and accumulates_error:
                    error[idx] += error_weights[int32(0)] * increment

                # LOGGING
                stage_increments[0, idx] = increment
                jacobian_updates[0, idx] = increment

            # Subsequent stages:
            for prev_idx in range(stages_except_first):
                stage_idx = prev_idx + int32(1)
                stage_offset = stage_idx * n
                stage_gamma = gamma_stages[stage_idx]
                stage_time = (
                    current_time + dt_scalar * stage_time_fractions[stage_idx]
                )

                # Get base state for F(t + c_i * dt, Y_n + sum(a_ij * K_j))
                stage_slice = stage_store[stage_offset:stage_offset + n]
                for idx in range(n):
                    stage_slice[idx] = state[idx]

                # Accumulate contributions from predecessor stages
                # Loop over all stages for static loop bounds (better unrolling)
                for predecessor_idx in range(stages_except_first):
                    a_col = a_coeffs[predecessor_idx]
                    a_coeff = a_col[stage_idx]
                    # Only accumulate valid predecessors
                    if predecessor_idx < stage_idx:
                        base_idx = predecessor_idx * n
                        for idx in range(n):
                            stage_slice[idx] += (
                                a_coeff * stage_store[base_idx + idx]
                            )

                if has_driver_function:
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        proposed_drivers,
                    )

                observables_function(
                    stage_slice,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                dxdt_fn(
                    stage_slice,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                # Capture precalculated outputs here, before overwrite
                if b_row == stage_idx:
                    for idx in range(n):
                        proposed_state[idx] = stage_slice[idx]
                if b_hat_row == stage_idx:
                    for idx in range(n):
                        error[idx] = stage_slice[idx]


                #LOGGING
                for idx in range(n):
                    stage_states[stage_idx, idx] = stage_slice[idx]

                # Overwrite the final accumulator slice with time-derivative
                if stage_idx == stage_count - int32(1):
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
                        time_derivative[idx] *= dt_scalar

                # LOGGING
                for driver_idx in range(driver_count):
                    stage_drivers_out[stage_idx, driver_idx] = (
                        proposed_drivers[driver_idx]
                    )
                for obs_idx in range(observable_count):
                    stage_observables[stage_idx, obs_idx] = (
                        proposed_observables[obs_idx]
                    )

                for idx in range(n):
                    correction = typed_zero
                    # Loop over all stages for static loop bounds
                    for predecessor_idx in range(stages_except_first):
                        c_col = C_coeffs[predecessor_idx]
                        c_coeff = c_col[stage_idx]
                        # Only accumulate valid predecessors
                        if predecessor_idx < stage_idx:
                            base = predecessor_idx * n
                            correction += (c_coeff * stage_store[base + idx])
                    f_stage_val = stage_rhs[idx]
                    deriv_val = stage_gamma * time_derivative[idx]
                    rhs_value = f_stage_val + correction * inv_dt + deriv_val
                    stage_rhs[idx] = rhs_value * dt_scalar * gamma

                    # LOGGING
                    stage_derivatives[stage_idx, idx] = f_stage_val
                    residuals[stage_idx, idx] = rhs_value * dt_scalar

                stage_increment = stage_slice

                previous_base = prev_idx * n
                for idx in range(n):
                    stage_increment[idx] = stage_store[previous_base + idx]

                stage_linear_slot = int32(stage_idx)
                solver_ret = linear_solver(
                    state,
                    parameters,
                    drivers_buffer,
                    base_state_placeholder,
                    cached_auxiliaries,
                    stage_time,
                    dt_scalar,
                    numba_precision(1.0),
                    stage_rhs,
                    stage_increment,
                    stage_linear_slot,
                    linear_initial_guesses,
                    linear_iteration_guesses,
                    linear_residuals,
                    linear_squared_norms,
                    linear_preconditioned_vectors,
                    krylov_iters,
                )
                status_code |= solver_ret

                if accumulates_output:
                    # Standard accumulation path for proposed_state
                    solution_weight = solution_weights[stage_idx]
                    for idx in range(n):
                        increment = stage_increment[idx]
                        proposed_state[idx] += solution_weight * increment

                if has_error:
                    if accumulates_error:
                        # Standard accumulation path for error
                        error_weight = error_weights[stage_idx]
                        for idx in range(n):
                            increment = stage_increment[idx]
                            error[idx] += error_weight * increment
                # LOGGING
                for idx in range(n):
                    increment = stage_increment[idx]
                    stage_increments[stage_idx, idx] = increment
                    jacobian_updates[stage_idx, idx] = increment
            # ----------------------------------------------------------- #

            if not accumulates_error:
                for idx in range(n):
                    error[idx] = proposed_state[idx] - error[idx]


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

            # Cache time_derivative for next step's initial guess.
            # When stage_store is shared, time_derivative persists automatically.
            # When local, save to persistent_local for next step.
            if not stage_store_shared:
                for idx in range(n):
                    persistent_local[idx] = time_derivative[idx]

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
        return buffer_registry.shared_buffer_size(self)

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""
        return buffer_registry.local_buffer_size(self)

    @property
    def persistent_local_required(self) -> int:
        """Return the number of persistent local entries required."""
        return buffer_registry.persistent_local_buffer_size(self)

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
]
