"""Rosenbrock-W integration step as described in (5.2) in Lang & Verwer (2001).

This module provides the :class:`GenericRosenbrockWStep` class, which
implements Rosenbrock-W methods for stiff ODEs. Rosenbrock methods are
linearly implicit methods that avoid the need for iterative nonlinear
solvers by linearizing the problem around the current state.

Key Features
------------
- Configurable tableaus via :class:`RosenbrockTableau`
- Automatic controller defaults selection based on error estimate capability
- Matrix-free linear solvers with cached Jacobian approximation
- Efficient for moderately stiff systems without Newton iteration overhead

Notes
-----
The module defines two sets of default step controller settings:

- :data:`ROSENBROCK_ADAPTIVE_DEFAULTS`: Used when the tableau has an
  embedded error estimate. Defaults to PI controller with adaptive stepping.
- :data:`ROSENBROCK_FIXED_DEFAULTS`: Used when the tableau lacks an error
  estimate. Defaults to fixed-step controller.

This dynamic selection ensures that users cannot accidentally pair an
errorless tableau with an adaptive controller, which would fail at runtime.

References
----------
Lang, J., Verwer, J. ROS3P—An Accurate Third-Order Rosenbrock Solver Designed
for Parabolic Problems. BIT Numerical Mathematics 41, 731–738 (2001).
https://doi.org/10.1023/A:1021900219772
"""

from typing import Callable, Optional

import attrs
import numpy as np
from numba import cuda, int32

from cubie._utils import PrecisionDType, build_config, is_device_validator
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
from cubie.buffer_registry import buffer_registry





ROSENBROCK_ADAPTIVE_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "pid",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
        "kp": 0.6,
        "ki": -0.4,
        "deadband_min": 1.0,
        "deadband_max": 1.1,
        "min_gain": 0.5,
        "max_gain": 2.0,
    }
)
"""Default step controller settings for adaptive Rosenbrock tableaus.

This configuration is used when the Rosenbrock tableau has an embedded error
estimate (``tableau.has_error_estimate == True``).

The PI controller provides robust adaptive stepping with proportional and
derivative terms to smooth step size adjustments. The deadband prevents
unnecessary step size changes for small variations in the error estimate.

Notes
-----
These defaults are applied automatically when creating a
:class:`GenericRosenbrockWStep` with an adaptive tableau. Users can override
any of these settings by explicitly specifying step controller parameters.
"""

ROSENBROCK_FIXED_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)
"""Default step controller settings for errorless Rosenbrock tableaus.

This configuration is used when the Rosenbrock tableau lacks an embedded
error estimate (``tableau.has_error_estimate == False``).

Fixed-step controllers maintain a constant step size throughout the
integration. This is the only valid choice for errorless tableaus since
adaptive stepping requires an error estimate to adjust the step size.

Notes
-----
These defaults are applied automatically when creating a
:class:`GenericRosenbrockWStep` with an errorless tableau. Users can
override the step size ``dt`` by explicitly specifying it in the step
controller settings.
"""


@attrs.define
class RosenbrockWStepConfig(ImplicitStepConfig):
    """Configuration describing the Rosenbrock-W integrator."""

    tableau: RosenbrockTableau = attrs.field(default=DEFAULT_ROSENBROCK_TABLEAU)
    time_derivative_function: Optional[Callable] = attrs.field(
            default=None,
            validator=attrs.validators.optional(is_device_validator)
    )
    prepare_jacobian_function: Optional[Callable] = attrs.field(
            default=None,
            validator=attrs.validators.optional(is_device_validator)
    )
    driver_del_t: Optional[Callable] = attrs.field(
            default=None,
            validator=attrs.validators.optional(is_device_validator)
    )
    stage_rhs_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )
    stage_store_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )
    cached_auxiliaries_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )
    base_state_placeholder_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )
    krylov_iters_out_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )


class GenericRosenbrockWStep(ODEImplicitStep):
    """Rosenbrock-W step with an embedded error estimate."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        driver_del_t: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        tableau: RosenbrockTableau = DEFAULT_ROSENBROCK_TABLEAU,
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
        dxdt_function
            Compiled CUDA device function computing state derivatives.
        observables_function
            Optional compiled CUDA device function computing observables.
        driver_function
            Optional compiled CUDA device function computing time-varying
            drivers.
        driver_del_t
            Optional compiled CUDA device function computing time derivatives
            of drivers (required for some Rosenbrock formulations).
        get_solver_helper_fn
            Factory function returning solver helper for Jacobian operations.
        tableau
            Rosenbrock tableau describing the coefficients and gamma values.
            Defaults to :data:`DEFAULT_ROSENBROCK_TABLEAU`.
        **kwargs
            Optional parameters passed to config classes. See
            RosenbrockWStepConfig, ImplicitStepConfig, and solver config
            classes for available parameters. None values are ignored.

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

        config = build_config(
            RosenbrockWStepConfig,
            required={
                'precision': precision,
                'n': n,
                'dxdt_function': dxdt_function,
                'observables_function': observables_function,
                'driver_function': driver_function,
                'driver_del_t': driver_del_t,
                'get_solver_helper_fn': get_solver_helper_fn,
                'tableau': tableau_value,
                'beta': 1.0,
                'gamma': tableau_value.gamma,
                'M': mass,
            },
            **kwargs
        )
        self._cached_auxiliary_count = None

        # Select defaults based on error estimate
        if tableau_value.has_error_estimate:
            controller_defaults = ROSENBROCK_ADAPTIVE_DEFAULTS
        else:
            controller_defaults = ROSENBROCK_FIXED_DEFAULTS

        super().__init__(
            config, controller_defaults, solver_type='linear', **kwargs
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
            'stage_rhs', self, n, config.stage_rhs_location,
            precision=precision
        )
        buffer_registry.register(
            'stage_store', self, stage_store_elements,
            config.stage_store_location, precision=precision
        )
        # cached_auxiliaries registered with 0 size; updated in build_implicit_helpers
        buffer_registry.register(
            'cached_auxiliaries', self, 0,
            config.cached_auxiliaries_location, precision=precision
        )

        # Stage increment should persist between steps for initial guess
        buffer_registry.register(
            'stage_increment', self, n,
            config.stage_store_location,
            aliases='stage_store',
            persistent=True,
            precision=precision
        )

        buffer_registry.register(
            'base_state_placeholder', self, 1,
            config.base_state_placeholder_location,
            precision=np.int32
        )
        buffer_registry.register(
            'krylov_iters_out', self, 1,
            config.krylov_iters_out_location,
            precision=np.int32
        )

    def build_implicit_helpers(
        self,
    ) -> Callable:
        """Construct the linear solver used by Rosenbrock methods.

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

        # Get device functions from ODE system
        preconditioner = get_fn(
            'neumann_preconditioner_cached',
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order
        )
        operator = get_fn(
            'linear_operator_cached',
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order
        )

        prepare_jacobian = get_fn(
            "prepare_jac",
            preconditioner_order=preconditioner_order,
        )
        self._cached_auxiliary_count = get_fn("cached_aux_count")

        # Update buffer registry with the actual cached_auxiliary_count
        buffer_registry.update_buffer(
            'cached_auxiliaries', self,
            size=self._cached_auxiliary_count
        )

        time_derivative_function = get_fn('time_derivative_rhs')

        # Update linear solver with device functions
        self.solver.update(
            operator_apply=operator,
            preconditioner=preconditioner,
            use_cached_auxiliaries=True,
        )

        # Return linear solver device function
        self.update_compile_settings(
                {'solver_function': self.solver.device_function,
                 'time_derivative_function': time_derivative_function,
                 'prepare_jacobian_function': prepare_jacobian}
        )

    def build_step(
        self,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
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
                    self,
                    self.solver,
                    name="solver"
            )
        )
        getalloc = buffer_registry.get_allocator
        alloc_stage_rhs = getalloc('stage_rhs', self)
        alloc_stage_store = getalloc('stage_store', self)
        alloc_cached_auxiliaries = getalloc('cached_auxiliaries', self)
        alloc_stage_increment = getalloc('stage_increment', self)
        alloc_base_state_placeholder = getalloc('base_state_placeholder', self)
        alloc_krylov_iters_out = getalloc('krylov_iters_out', self)

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
            cached_auxiliaries = alloc_cached_auxiliaries(shared, persistent_local)
            stage_increment = alloc_stage_increment(shared, persistent_local)
            base_state_placeholder = alloc_base_state_placeholder(
                shared, persistent_local
            )
            krylov_iters_out = alloc_krylov_iters_out(shared, persistent_local)
            solver_shared = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(shared, persistent_local)
            # ----------------------------------------------------------- #

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

            status_code = int32(0)
            stage_time = current_time + dt_scalar * stage_time_fractions[0]

            # --------------------------------------------------------------- #
            #            Stage 0: uses starting values                        #
            # --------------------------------------------------------------- #

            dxdt_fn(
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
                        (f_value + gamma_stages[0] * time_derivative[idx])
                        * dt_scalar
                )
                stage_rhs[idx] = rhs_value * gamma

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
            )

            for idx in range(n):
                stage_store[idx] = stage_increment[idx]

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] += (stage_increment[idx] *
                                            solution_weights[int32(0)])
                if has_error and accumulates_error:
                    error[idx] += stage_increment[idx] * error_weights[
                        int32(0)]

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
                            stage_store[stage_offset + idx] += (a_coeff *
                                                              prior_val)

                for idx in range(n):
                    stage_increment[idx] = stage_store[stage_offset + idx]

                # Get t + c_i * dt parts
                if has_driver_function:
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        proposed_drivers,
                    )

                observables_function(
                    stage_increment,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                dxdt_fn(
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
    "GenericRosenbrockWStep",
    "RosenbrockWStepConfig",
    "RosenbrockTableau",
]
