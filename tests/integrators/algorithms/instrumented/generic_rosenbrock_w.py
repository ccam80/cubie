from typing import Callable, Optional

import numpy as np
from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
)
from cubie.integrators.algorithms.generic_rosenbrock_w import (
    ROSENBROCK_FIXED_DEFAULTS,
    RosenbrockWStepConfig,
    ROSENBROCK_ADAPTIVE_DEFAULTS,
)
from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    DEFAULT_ROSENBROCK_TABLEAU,
    RosenbrockTableau,
)
from cubie.buffer_registry import buffer_registry
from tests.integrators.algorithms.instrumented.ode_implicitstep import (
    InstrumentedODEImplicitStep,
)


class InstrumentedRosenbrockWStep(InstrumentedODEImplicitStep):
    """Rosenbrock-W step with an embedded error estimate."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        evaluate_f: Optional[Callable] = None,
        evaluate_observables: Optional[Callable] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        driver_del_t: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        preconditioner_order: Optional[int] = None,
        krylov_atol: Optional[float] = None,
        krylov_rtol: Optional[float] = None,
        krylov_max_iters: Optional[int] = None,
        linear_correction_type: Optional[str] = None,
        tableau: RosenbrockTableau = DEFAULT_ROSENBROCK_TABLEAU,
        stage_rhs_location: Optional[str] = None,
        stage_store_location: Optional[str] = None,
        cached_auxiliaries_location: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialise the Rosenbrock-W step configuration.

        This constructor creates a Rosenbrock-W step object and automatically
        selects appropriate default step controller settings based on whether
        the tableau has an embedded error estimate. Tableaus with error
        estimates default to adaptive stepping (PI controller), while
        errorless tableaus default to fixed stepping.

        Parameters
        ----------
        precision
            Floating-point precision for CUDA computations.
        n
            Number of state variables in the ODE system.
        evaluate_f
            Compiled CUDA device function computing state derivatives.
        evaluate_observables
            Optional compiled CUDA device function computing observables.
        evaluate_driver_at_t
            Optional compiled CUDA device function computing time-varying
            drivers.
        driver_del_t
            Optional compiled CUDA device function computing time derivatives
            of drivers (required for some Rosenbrock formulations).
        get_solver_helper_fn
            Factory function returning solver helper for Jacobian operations.
        preconditioner_order
            Order of the finite-difference Jacobian approximation used in the
            preconditioner. If None, uses default value of 2.
        krylov_atol
            Absolute tolerance for the Krylov linear solver. If None, uses
            default from LinearSolverConfig.
        krylov_rtol
            Relative tolerance for the Krylov linear solver. If None, uses
            default from LinearSolverConfig.
        krylov_max_iters
            Maximum iterations allowed for the Krylov solver. If None, uses
            default from LinearSolverConfig.
        linear_correction_type
            Type of Krylov correction ("minimal_residual" or other). If None,
            uses default from LinearSolverConfig.
        tableau
            Rosenbrock tableau describing the coefficients and gamma values.
            Defaults to :data:`DEFAULT_ROSENBROCK_TABLEAU`.
        stage_rhs_location
            Memory location for stage RHS buffer: 'local' or 'shared'. If
            None, defaults to 'local'.
        stage_store_location
            Memory location for stage store buffer: 'local' or 'shared'. If
            None, defaults to 'local'.
        cached_auxiliaries_location
            Memory location for cached auxiliaries buffer: 'local' or 'shared'.
            If None, defaults to 'local'.

        Notes
        -----
        The step controller defaults are selected dynamically:

        - If ``tableau.has_error_estimate`` is ``True``:
          Uses :data:`ROSENBROCK_ADAPTIVE_DEFAULTS` (PI controller)
        - If ``tableau.has_error_estimate`` is ``False``:
          Uses :data:`ROSENBROCK_FIXED_DEFAULTS` (fixed-step controller)

        This automatic selection prevents incompatible configurations where
        an adaptive controller is paired with an errorless tableau.

        Rosenbrock methods linearize the ODE around the current state,
        avoiding the need for iterative Newton solves. This makes them
        efficient for moderately stiff problems. The gamma parameter from the
        tableau controls the implicit treatment of the linearized system.
        """

        mass = np.eye(n, dtype=precision)
        tableau_value = tableau

        # Build config first so buffer registration can use config defaults
        config_kwargs = {
            "precision": precision,
            "n": n,
            "evaluate_f": evaluate_f,
            "evaluate_observables": evaluate_observables,
            "evaluate_driver_at_t": evaluate_driver_at_t,
            "driver_del_t": driver_del_t,
            "get_solver_helper_fn": get_solver_helper_fn,
            "preconditioner_order": preconditioner_order,
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
            config_kwargs["cached_auxiliaries_location"] = (
                cached_auxiliaries_location
            )

        config = RosenbrockWStepConfig(**config_kwargs)
        self._cached_auxiliary_count = None

        # Select defaults based on error estimate
        if tableau_value.has_error_estimate:
            controller_defaults = ROSENBROCK_ADAPTIVE_DEFAULTS
        else:
            controller_defaults = ROSENBROCK_FIXED_DEFAULTS

        # Build kwargs dict conditionally (only linear solver kwargs for Rosenbrock)
        solver_kwargs = {}
        if krylov_atol is not None:
            solver_kwargs["krylov_atol"] = krylov_atol
        if krylov_rtol is not None:
            solver_kwargs["krylov_rtol"] = krylov_rtol
        if krylov_max_iters is not None:
            solver_kwargs["krylov_max_iters"] = krylov_max_iters
        if linear_correction_type is not None:
            solver_kwargs["linear_correction_type"] = linear_correction_type

        # Call parent __init__ to create solver instances
        super().__init__(
            config, controller_defaults, solver_type="linear", **solver_kwargs
        )

        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers according to locations in compile settings."""
        config = self.compile_settings
        precision = config.precision
        n = config.n
        tableau = config.tableau

        # Calculate buffer sizes
        stage_store_elements = tableau.stage_count * n

        # Register algorithm buffers using config values
        buffer_registry.register(
            "stage_rhs",
            self,
            n,
            config.stage_rhs_location,
            precision=precision,
        )
        buffer_registry.register(
            "stage_store",
            self,
            stage_store_elements,
            config.stage_store_location,
            precision=precision,
        )
        # cached_auxiliaries registered with 0 size; updated in build_implicit_helpers
        buffer_registry.register(
            "cached_auxiliaries",
            self,
            0,
            config.cached_auxiliaries_location,
            precision=precision,
        )

        # Stage increment should persist between steps for initial guess
        buffer_registry.register(
            "stage_increment",
            self,
            n,
            config.stage_store_location,
            aliases="stage_store",
            persistent=True,
            precision=precision,
        )

        buffer_registry.register(
            "base_state_placeholder",
            self,
            1,
            config.base_state_placeholder_location,
            precision=np.int32,
        )
        buffer_registry.register(
            "krylov_iters_out",
            self,
            1,
            config.krylov_iters_out_location,
            precision=np.int32,
        )

    def build_implicit_helpers(
        self,
    ) -> Callable:
        """Construct instrumented linear solver chain for Rosenbrock methods.

        Returns
        -------
        Callable
            Linear solver function compiled for the configured scheme.
        """
        config = self.compile_settings
        beta = config.beta
        gamma = config.gamma
        mass = config.M
        preconditioner_order = config.preconditioner_order

        get_fn = config.get_solver_helper_fn

        # Get device functions from ODE system (cached versions for Rosenbrock)
        preconditioner = get_fn(
            "neumann_preconditioner_cached",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )
        operator = get_fn(
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
            "cached_auxiliaries", self, size=self._cached_auxiliary_count
        )

        time_derivative_function = get_fn("time_derivative_rhs")

        # Update solver with operator and preconditioner
        self.solver.update(
            operator_apply=operator,
            preconditioner=preconditioner,
            use_cached_auxiliaries=True,
        )

        # Return linear solver device function
        self.update_compile_settings(
            {
                "solver_function": self.solver.device_function,
                "time_derivative_function": time_derivative_function,
                "prepare_jacobian_function": prepare_jacobian,
            }
        )

        self.register_buffers()

    def build_step(
        self,
        evaluate_f: Callable,
        evaluate_observables: Callable,
        evaluate_driver_at_t: Optional[Callable],
        solver_function: Callable,
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the Rosenbrock-W device step."""

        config = self.compile_settings
        tableau = config.tableau

        # Access solver from parameter
        linear_solver = solver_function
        prepare_jacobian = config.prepare_jacobian_function
        time_derivative_rhs = config.time_derivative_function
        driver_del_t = config.driver_del_t

        n = int32(n)
        stage_count = int32(self.stage_count)
        stages_except_first = stage_count - int32(1)
        has_evaluate_driver_at_t = evaluate_driver_at_t is not None
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
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(
                self, self.solver, name="solver"
            )
        )
        getalloc = buffer_registry.get_allocator
        alloc_stage_rhs = getalloc("stage_rhs", self)
        alloc_stage_store = getalloc("stage_store", self)
        alloc_cached_auxiliaries = getalloc("cached_auxiliaries", self)
        alloc_stage_increment = getalloc("stage_increment", self)
        alloc_base_state_placeholder = getalloc("base_state_placeholder", self)
        alloc_krylov_iters_out = getalloc("krylov_iters_out", self)

        # no cover: start
        @cuda.jit(
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
            # Allocate buffers
            stage_rhs = alloc_stage_rhs(shared, persistent_local)
            stage_store = alloc_stage_store(shared, persistent_local)
            cached_auxiliaries = alloc_cached_auxiliaries(
                shared, persistent_local
            )
            stage_increment = alloc_stage_increment(shared, persistent_local)
            base_state_placeholder = alloc_base_state_placeholder(
                shared, persistent_local
            )
            krylov_iters_out = alloc_krylov_iters_out(shared, persistent_local)
            solver_shared = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(
                shared, persistent_local
            )
            # ----------------------------------------------------------- #

            # LOGGING: Get array dimensions for logging loops
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
            if has_evaluate_driver_at_t:
                driver_del_t(
                    current_time,
                    driver_coeffs,
                    proposed_drivers,
                )
            else:
                for i in range(n_drivers):
                    proposed_drivers[i] = numba_precision(0.0)

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
                time_derivative[idx] *= dt_scalar
                if has_error:
                    error[idx] = typed_zero

                # LOGGING
                stage_states[0, idx] = state[idx]

            # LOGGING
            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = observables[obs_idx]
            for driver_idx in range(driver_count):
                stage_drivers_out[0, driver_idx] = drivers_buffer[driver_idx]

            status_code = int32(0)
            stage_time = current_time + dt_scalar * stage_time_fractions[0]

            # --------------------------------------------------------------- #
            #            Stage 0: uses starting values                        #
            # --------------------------------------------------------------- #

            evaluate_f(
                state,
                parameters,
                drivers_buffer,
                observables,
                stage_rhs,
                current_time,
            )

            for idx in range(n):
                # No accumulated contributions at stage 0.
                f_value = stage_rhs[idx]
                rhs_value = (
                    f_value + gamma_stages[0] * time_derivative[idx]
                ) * dt_scalar
                stage_rhs[idx] = rhs_value * gamma

                # LOGGING
                stage_derivatives[0, idx] = f_value
                residuals[0, idx] = rhs_value

            krylov_iters_out[0] = int32(0)

            # Use stored copy as the initial guess for the first stage.
            status_code |= linear_solver(
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
                solver_shared,
                solver_persistent,
                krylov_iters_out,
                # LOGGING parameters
                int32(0),
                linear_initial_guesses,
                linear_iteration_guesses,
                linear_residuals,
                linear_squared_norms,
                linear_preconditioned_vectors,
            )

            for idx in range(n):
                stage_store[idx] = stage_increment[idx]

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] += (
                        stage_increment[idx] * solution_weights[int32(0)]
                    )
                if has_error and accumulates_error:
                    error[idx] += (
                        stage_increment[idx] * error_weights[int32(0)]
                    )

                # LOGGING
                stage_increments[0, idx] = stage_increment[idx]
                jacobian_updates[0, idx] = stage_increment[idx]

            # --------------------------------------------------------------- #
            #            Stages 1-s: must refresh all values                  #
            # --------------------------------------------------------------- #
            for prev_idx in range(stages_except_first):
                stage_idx = prev_idx + int32(1)
                stage_offset = stage_idx * n
                stage_gamma = gamma_stages[stage_idx]
                stage_time = (
                    current_time + dt_scalar * stage_time_fractions[stage_idx]
                )

                # Get base state for F(t + c_i * dt, Y_n + sum(a_ij * K_j))
                for idx in range(n):
                    stage_store[stage_offset + idx] = state[idx]

                # Accumulate contributions from predecessor stages
                # Loop over all stages for static loop bounds (better unrolling)
                # Zero coefficients from strict lower triangular structure
                for predecessor_idx in range(stages_except_first):
                    a_col = a_coeffs[predecessor_idx]
                    a_coeff = a_col[stage_idx]
                    # Only accumulate valid predecessors (coefficient will be
                    # zero for predecessor_idx >= stage_idx due to strict
                    # lower triangular structure)
                    if predecessor_idx < stage_idx:
                        base_idx = predecessor_idx * n
                        for idx in range(n):
                            prior_val = stage_store[base_idx + idx]
                            stage_store[stage_offset + idx] += (
                                a_coeff * prior_val
                            )

                for idx in range(n):
                    stage_increment[idx] = stage_store[stage_offset + idx]

                # Get t + c_i * dt parts
                if has_evaluate_driver_at_t:
                    evaluate_driver_at_t(
                        stage_time,
                        driver_coeffs,
                        proposed_drivers,
                    )

                evaluate_observables(
                    stage_increment,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                evaluate_f(
                    stage_increment,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                # Capture precalculated outputs here, before overwrite
                if b_row == stage_idx:
                    for idx in range(n):
                        proposed_state[idx] = stage_increment[idx]
                if b_hat_row == stage_idx:
                    for idx in range(n):
                        error[idx] = stage_increment[idx]

                # LOGGING
                for idx in range(n):
                    stage_states[stage_idx, idx] = stage_store[
                        stage_offset + idx
                    ]

                # Overwrite the final accumulator slice with time-derivative
                if stage_idx == stage_count - int32(1):
                    if has_evaluate_driver_at_t:
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

                # Add C_ij*K_j/dt + dt * gamma_i * d/dt terms to rhs
                for idx in range(n):
                    correction = numba_precision(0.0)
                    # Loop over all stages for static loop bounds
                    for predecessor_idx in range(stages_except_first):
                        c_col = C_coeffs[predecessor_idx]
                        c_coeff = c_col[stage_idx]
                        # Only accumulate valid predecessors
                        if predecessor_idx < stage_idx:
                            prior_idx = predecessor_idx * n + idx
                            prior_val = stage_store[prior_idx]
                            correction += c_coeff * prior_val

                    f_stage_val = stage_rhs[idx]
                    deriv_val = stage_gamma * time_derivative[idx]
                    rhs_value = f_stage_val + correction * inv_dt + deriv_val
                    stage_rhs[idx] = rhs_value * dt_scalar * gamma

                    # LOGGING
                    stage_derivatives[stage_idx, idx] = f_stage_val
                    residuals[stage_idx, idx] = rhs_value * dt_scalar

                # Use previous stage's solution as a guess for this stage
                previous_base = prev_idx * n
                for idx in range(n):
                    stage_increment[idx] = stage_store[previous_base + idx]

                status_code |= linear_solver(
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
                    solver_shared,
                    solver_persistent,
                    krylov_iters_out,
                    # LOGGING parameters
                    stage_idx,
                    linear_initial_guesses,
                    linear_iteration_guesses,
                    linear_residuals,
                    linear_squared_norms,
                    linear_preconditioned_vectors,
                )

                for idx in range(n):
                    stage_store[stage_offset + idx] = stage_increment[idx]

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

            if has_evaluate_driver_at_t:
                evaluate_driver_at_t(
                    end_time,
                    driver_coeffs,
                    proposed_drivers,
                )

            evaluate_observables(
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
    "InstrumentedRosenbrockWStep",
    "RosenbrockWStepConfig",
    "RosenbrockTableau",
]
