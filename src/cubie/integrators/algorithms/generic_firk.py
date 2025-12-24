"""Fully implicit Runge--Kutta integration step implementation.

This module provides the :class:`FIRKStep` class, which implements fully
implicit Runge--Kutta (FIRK) methods using configurable Butcher tableaus.
Unlike DIRK methods, FIRK methods have a fully dense coefficient matrix,
requiring all stages to be solved simultaneously as a coupled system.

Key Features
------------
- Configurable tableaus via :class:`FIRKTableau`
- Automatic controller defaults selection based on error estimate capability
- Matrix-free Newton-Krylov solvers for coupled implicit stages
- Support for high-order implicit methods (e.g., Gauss-Legendre)

Notes
-----
The module defines two sets of default step controller settings:

- :data:`FIRK_ADAPTIVE_DEFAULTS`: Used when the tableau has an embedded
  error estimate. Defaults to PI controller with adaptive stepping.
- :data:`FIRK_FIXED_DEFAULTS`: Used when the tableau lacks an error
  estimate. Defaults to fixed-step controller.

This dynamic selection ensures that users cannot accidentally pair an
errorless tableau with an adaptive controller, which would fail at runtime.
"""

from typing import Callable, Optional

import attrs
import numpy as np
from numba import cuda, int32

from cubie._utils import PrecisionDType, build_config
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
from cubie.buffer_registry import buffer_registry





FIRK_ADAPTIVE_DEFAULTS = StepControlDefaults(
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
"""Default step controller settings for adaptive FIRK tableaus.

This configuration is used when the FIRK tableau has an embedded error
estimate (``tableau.has_error_estimate == True``).

The PI controller provides robust adaptive stepping with proportional and
derivative terms to smooth step size adjustments. The deadband prevents
unnecessary step size changes for small variations in the error estimate.

Notes
-----
These defaults are applied automatically when creating a :class:`FIRKStep`
with an adaptive tableau. Users can override any of these settings by
explicitly specifying step controller parameters.
"""

FIRK_FIXED_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)
"""Default step controller settings for errorless FIRK tableaus.

This configuration is used when the FIRK tableau lacks an embedded error
estimate (``tableau.has_error_estimate == False``).

Fixed-step controllers maintain a constant step size throughout the
integration. This is the only valid choice for errorless tableaus since
adaptive stepping requires an error estimate to adjust the step size.

Notes
-----
These defaults are applied automatically when creating a :class:`FIRKStep`
with an errorless tableau. Users can override the step size ``dt`` by
explicitly specifying it in the step controller settings.
"""


@attrs.define
class FIRKStepConfig(ImplicitStepConfig):
    """Configuration describing the FIRK integrator."""

    tableau: FIRKTableau = attrs.field(
        default=DEFAULT_FIRK_TABLEAU,
    )
    stage_increment_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )
    stage_driver_stack_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )
    stage_state_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
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
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        tableau: FIRKTableau = DEFAULT_FIRK_TABLEAU,
        n_drivers: int = 0,
        **kwargs,
    ) -> None:
        """Initialise the FIRK step configuration.

        This constructor creates a FIRK step object and automatically selects
        appropriate default step controller settings based on whether the
        tableau has an embedded error estimate. Tableaus with error estimates
        default to adaptive stepping (PI controller), while errorless tableaus
        default to fixed stepping.

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
        get_solver_helper_fn
            Factory function returning solver helper for Jacobian operations.
        tableau
            FIRK tableau describing the coefficients. Defaults to
            :data:`DEFAULT_FIRK_TABLEAU`.
        n_drivers
            Number of driver variables in the system.
        **kwargs
            Optional parameters passed to config classes. See
            FIRKStepConfig, ImplicitStepConfig, and solver config classes
            for available parameters. None values are ignored.

        Notes
        -----
        The step controller defaults are selected dynamically:

        - If ``tableau.has_error_estimate`` is ``True``:
          Uses :data:`FIRK_ADAPTIVE_DEFAULTS` (PI controller)
        - If ``tableau.has_error_estimate`` is ``False``:
          Uses :data:`FIRK_FIXED_DEFAULTS` (fixed-step controller)

        This automatic selection prevents incompatible configurations where
        an adaptive controller is paired with an errorless tableau.

        FIRK methods require solving a coupled system of all stages
        simultaneously, which is more computationally expensive than DIRK
        methods but can achieve higher orders of accuracy for stiff systems.
        """
        mass = np.eye(n, dtype=precision)

        config = build_config(
            FIRKStepConfig,
            required={
                'precision': precision,
                'n': n,
                'n_drivers': n_drivers,
                'dxdt_function': dxdt_function,
                'observables_function': observables_function,
                'driver_function': driver_function,
                'get_solver_helper_fn': get_solver_helper_fn,
                'tableau': tableau,
                'beta': 1.0,
                'gamma': 1.0,
                'M': mass,
            },
            **kwargs
        )

        # Select defaults based on error estimate
        if tableau.has_error_estimate:
            controller_defaults = FIRK_ADAPTIVE_DEFAULTS
        else:
            controller_defaults = FIRK_FIXED_DEFAULTS

        super().__init__(config, controller_defaults, **kwargs)

        self.solver.update(n=self.tableau.stage_count * n)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers according to locations in compile settings."""
        config = self.compile_settings
        precision = config.precision
        n = config.n
        tableau = config.tableau

        # Calculate buffer sizes
        all_stages_n = tableau.stage_count * n
        stage_driver_stack_elements = tableau.stage_count * config.n_drivers

        _,_ = buffer_registry.get_child_allocators(
                self,
                self.solver,
                name='solver'
        )
        buffer_registry.register(
            'stage_increment',
            self,
            all_stages_n,
            config.stage_increment_location,
            persistent=True,
            precision=precision
        )
        buffer_registry.register(
            'stage_driver_stack', self, stage_driver_stack_elements,
            config.stage_driver_stack_location, precision=precision
        )
        buffer_registry.register(
            'stage_state', self, n, config.stage_state_location,
            precision=precision
        )

    def build_implicit_helpers(
        self,
    ) -> Callable:
        """Construct the nonlinear solver chain used by implicit methods."""

        config = self.compile_settings
        tableau = config.tableau
        beta = config.beta
        gamma = config.gamma
        mass = config.M

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

        # Update solvers with device functions
        self.solver.update(
            operator_apply=operator,
            preconditioner=preconditioner,
            residual_function=residual,
            n=tableau.stage_count * config.n
        )

        self.update_compile_settings(
                {'solver_function':self.solver.device_function}
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
        """Compile the FIRK device step."""

        config = self.compile_settings
        tableau = config.tableau

        nonlinear_solver = solver_function

        n = int32(n)
        n_drivers = int32(n_drivers)
        stage_count = int32(self.stage_count)

        has_driver_function = driver_function is not None
        has_error = self.is_adaptive

        stage_rhs_coeffs = tableau.a_flat(numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        typed_zero = numba_precision(0.0)
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

        ends_at_one = stage_time_fractions[-1] == numba_precision(1.0)

        # Get allocators from buffer registry
        getalloc = buffer_registry.get_allocator
        alloc_stage_increment = getalloc('stage_increment', self)
        alloc_stage_driver_stack = getalloc('stage_driver_stack', self)
        alloc_stage_state = getalloc('stage_state', self)

        # Get child allocators for Newton solver
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(self, nonlinear_solver)
        )
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

            # ----------------------------------------------------------- #
            # Selective allocation from local or shared memory
            # ----------------------------------------------------------- #
            stage_state = alloc_stage_state(shared, persistent_local)
            solver_shared = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(shared, persistent_local)
            stage_increment = alloc_stage_increment(shared, persistent_local)
            stage_driver_stack = alloc_stage_driver_stack(shared, persistent_local)

            # ----------------------------------------------------------- #


            current_time = time_scalar
            end_time = current_time + dt_scalar
            status_code = int32(0)

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] = state[idx]
                if has_error and accumulates_error:
                    error[idx] = typed_zero

            # Fill stage_drivers_stack if driver arrays provided
            if has_driver_function:
                for stage_idx in range(stage_count):
                    stage_time = (
                        current_time
                        + dt_scalar * stage_time_fractions[stage_idx]
                    )
                    driver_offset = stage_idx * n_drivers
                    driver_slice = stage_driver_stack[
                        driver_offset:driver_offset + n_drivers
                    ]
                    driver_function(stage_time, driver_coeffs, driver_slice)

            # Solve n-stage nonlinear problem for all stages
            solver_status = nonlinear_solver(
                stage_increment,
                parameters,
                stage_driver_stack,
                current_time,
                dt_scalar,
                typed_zero,
                state,
                solver_shared,
                solver_persistent,
                counters,
            )
            status_code = int32(status_code | solver_status)

            for stage_idx in range(stage_count):
                if has_driver_function:
                    stage_base = stage_idx * n_drivers
                    for idx in range (n_drivers):
                        proposed_drivers[idx] = stage_driver_stack[stage_base + idx]

                for idx in range(n):
                    value = state[idx]
                    for contrib_idx in range(stage_count):
                        flat_idx = stage_idx * stage_count + contrib_idx
                        increment_idx = contrib_idx * n
                        coeff = stage_rhs_coeffs[flat_idx]
                        if coeff != typed_zero:
                            value += coeff * stage_increment[increment_idx + idx]
                    stage_state[idx] = value

                # Capture precalculated outputs if tableau allows
                if not accumulates_output:
                    if b_row == stage_idx:
                        for idx in range(n):
                            proposed_state[idx] = stage_state[idx]
                if not accumulates_error:
                    if b_hat_row == stage_idx:
                        for idx in range(n):
                            error[idx] = stage_state[idx]

            # Kahan summation to reduce floating point errors
            # see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            if accumulates_output:
                for idx in range(n):
                    solution_acc = typed_zero
                    compensation = typed_zero
                    for stage_idx in range(stage_count):
                        increment_value = stage_increment[stage_idx * n + idx]
                        weighted = (
                            solution_weights[stage_idx] * increment_value
                        )
                        term = weighted - compensation
                        temp = solution_acc + term
                        compensation = (temp - solution_acc) - term
                        solution_acc = temp
                    proposed_state[idx] = state[idx] + solution_acc

            if has_error and accumulates_error:
                # Standard accumulation path for error
                for idx in range(n):
                    error_acc = typed_zero
                    compensation = typed_zero
                    for stage_idx in range(stage_count):
                        increment_value = stage_increment[stage_idx * n + idx]
                        weighted = error_weights[stage_idx] * increment_value
                        term = weighted - compensation
                        temp = error_acc + term
                        compensation = (temp - error_acc) - term
                        error_acc = temp
                    error[idx] = error_acc

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
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""
        return self.compile_settings.stage_count

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
