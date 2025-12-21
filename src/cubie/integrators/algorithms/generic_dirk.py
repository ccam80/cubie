"""Diagonally implicit Runge–Kutta integration step implementation.

This module provides the :class:`DIRKStep` class, which implements
diagonally implicit Runge--Kutta (DIRK) methods using configurable Butcher
tableaus. DIRK methods are linearly implicit with a diagonal structure in
the coefficient matrix, allowing each implicit stage to be solved
independently.

Key Features
------------
- Configurable tableaus via :class:`DIRKTableau`
- Automatic controller defaults selection based on error estimate capability
- Matrix-free Newton-Krylov solvers for implicit stages
- Efficient diagonal structure reduces computational cost vs fully implicit

Notes
-----
The module defines two sets of default step controller settings:

- :data:`DIRK_ADAPTIVE_DEFAULTS`: Used when the tableau has an embedded
  error estimate. Defaults to PI controller with adaptive stepping.
- :data:`DIRK_FIXED_DEFAULTS`: Used when the tableau lacks an error
  estimate. Defaults to fixed-step controller.

This dynamic selection ensures that users cannot accidentally pair an
errorless tableau with an adaptive controller, which would fail at runtime.
"""

from typing import Callable, Optional

import attrs
import numpy as np
from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.cuda_simsafe import activemask, all_sync, syncwarp
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
from cubie.buffer_registry import buffer_registry





DIRK_ADAPTIVE_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "pid",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
        "kp": 0.7,
        "ki": -0.4,
        "deadband_min": 1.0,
        "deadband_max": 1.1,
        "min_gain": 0.1,
        "max_gain": 5.0,
    }
)
"""Default step controller settings for adaptive DIRK tableaus.

This configuration is used when the DIRK tableau has an embedded error
estimate (``tableau.has_error_estimate == True``).

The PI controller provides robust adaptive stepping with proportional and
derivative terms to smooth step size adjustments. The deadband prevents
unnecessary step size changes for small variations in the error estimate.

Notes
-----
These defaults are applied automatically when creating a :class:`DIRKStep`
with an adaptive tableau. Users can override any of these settings by
explicitly specifying step controller parameters.
"""

DIRK_FIXED_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)
"""Default step controller settings for errorless DIRK tableaus.

This configuration is used when the DIRK tableau lacks an embedded error
estimate (``tableau.has_error_estimate == False``).

Fixed-step controllers maintain a constant step size throughout the
integration. This is the only valid choice for errorless tableaus since
adaptive stepping requires an error estimate to adjust the step size.

Notes
-----
These defaults are applied automatically when creating a :class:`DIRKStep`
with an errorless tableau. Users can override the step size ``dt`` by
explicitly specifying it in the step controller settings.
"""
@attrs.define
class DIRKStepConfig(ImplicitStepConfig):
    """Configuration describing the DIRK integrator."""

    tableau: DIRKTableau = attrs.field(
        default=DEFAULT_DIRK_TABLEAU,
    )
    stage_increment_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )
    stage_base_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )
    accumulator_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )
    stage_rhs_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )

class DIRKStep(ODEImplicitStep):
    """Diagonally implicit Runge–Kutta step with an embedded error estimate."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        preconditioner_order: Optional[int] = None,
        krylov_tolerance: Optional[float] = None,
        max_linear_iters: Optional[int] = None,
        linear_correction_type: Optional[str] = None,
        newton_tolerance: Optional[float] = None,
        max_newton_iters: Optional[int] = None,
        newton_damping: Optional[float] = None,
        newton_max_backtracks: Optional[int] = None,
        tableau: DIRKTableau = DEFAULT_DIRK_TABLEAU,
        n_drivers: int = 0,
        stage_increment_location: Optional[str] = None,
        stage_base_location: Optional[str] = None,
        accumulator_location: Optional[str] = None,
        stage_rhs_location: Optional[str] = None,
    ) -> None:
        """Initialise the DIRK step configuration.
        
        This constructor creates a DIRK step object and automatically selects
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
        preconditioner_order
            Order of the truncated Neumann preconditioner. If None, uses
            default value of 2.
        krylov_tolerance
            Convergence tolerance for the Krylov linear solver. If None, uses
            default from LinearSolverConfig.
        max_linear_iters
            Maximum iterations allowed for the Krylov solver. If None, uses
            default from LinearSolverConfig.
        linear_correction_type
            Type of Krylov correction. If None, uses default from
            LinearSolverConfig.
        newton_tolerance
            Convergence tolerance for the Newton iteration. If None, uses
            default from NewtonKrylovConfig.
        max_newton_iters
            Maximum iterations permitted for the Newton solver. If None, uses
            default from NewtonKrylovConfig.
        newton_damping
            Damping factor applied within Newton updates. If None, uses
            default from NewtonKrylovConfig.
        newton_max_backtracks
            Maximum number of backtracking steps within the Newton solver. If
            None, uses default from NewtonKrylovConfig.
        tableau
            DIRK tableau describing the coefficients. Defaults to
            :data:`DEFAULT_DIRK_TABLEAU`.
        n_drivers
            Number of driver variables in the system.
        stage_increment_location
            Memory location for stage increment buffer: 'local' or 'shared'.
            If None, defaults to 'local'.
        stage_base_location
            Memory location for stage base buffer: 'local' or 'shared'. If
            None, defaults to 'local'.
        accumulator_location
            Memory location for accumulator buffer: 'local' or 'shared'. If
            None, defaults to 'local'.
        stage_rhs_location
            Memory location for stage RHS buffer: 'local' or 'shared'. If
            None, defaults to 'local'.
        
        Notes
        -----
        The step controller defaults are selected dynamically:
        
        - If ``tableau.has_error_estimate`` is ``True``:
          Uses :data:`DIRK_ADAPTIVE_DEFAULTS` (PI controller)
        - If ``tableau.has_error_estimate`` is ``False``:
          Uses :data:`DIRK_FIXED_DEFAULTS` (fixed-step controller)
        
        This automatic selection prevents incompatible configurations where
        an adaptive controller is paired with an errorless tableau.
        """

        mass = np.eye(n, dtype=precision)

        # Build config first so buffer registration can use config defaults
        config_kwargs = {
            "precision": precision,
            "n": n,
            "n_drivers": n_drivers,
            "dxdt_function": dxdt_function,
            "observables_function": observables_function,
            "driver_function": driver_function,
            "get_solver_helper_fn": get_solver_helper_fn,
            "preconditioner_order": preconditioner_order,
            "tableau": tableau,
            "beta": 1.0,
            "gamma": 1.0,
            "M": mass,
        }
        if stage_increment_location is not None:
            config_kwargs["stage_increment_location"] = stage_increment_location
        if stage_base_location is not None:
            config_kwargs["stage_base_location"] = stage_base_location
        if accumulator_location is not None:
            config_kwargs["accumulator_location"] = accumulator_location
        if stage_rhs_location is not None:
            config_kwargs["stage_rhs_location"] = stage_rhs_location

        config = DIRKStepConfig(**config_kwargs)

        # Select defaults based on error estimate
        if tableau.has_error_estimate:
            controller_defaults = DIRK_ADAPTIVE_DEFAULTS
        else:
            controller_defaults = DIRK_FIXED_DEFAULTS

        # Build kwargs dict conditionally
        solver_kwargs = {}
        if krylov_tolerance is not None:
            solver_kwargs['krylov_tolerance'] = krylov_tolerance
        if max_linear_iters is not None:
            solver_kwargs['max_linear_iters'] = max_linear_iters
        if linear_correction_type is not None:
            solver_kwargs['linear_correction_type'] = linear_correction_type
        if newton_tolerance is not None:
            solver_kwargs['newton_tolerance'] = newton_tolerance
        if max_newton_iters is not None:
            solver_kwargs['max_newton_iters'] = max_newton_iters
        if newton_damping is not None:
            solver_kwargs['newton_damping'] = newton_damping
        if newton_max_backtracks is not None:
            solver_kwargs['newton_max_backtracks'] = newton_max_backtracks

        # Call parent __init__ to create solver instances
        super().__init__(config, controller_defaults, **solver_kwargs)

        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers according to locations in compile settings."""
        config = self.compile_settings
        precision = config.precision
        n = config.n
        tableau = config.tableau

        # Clear any existing buffer registrations
        buffer_registry.clear_parent(self)

        # Calculate buffer sizes
        accumulator_length = max(tableau.stage_count - 1, 0) * n

        # Register solver scratch and solver persistent buffers so they can
        # be aliased
        _ = buffer_registry.get_child_allocators(
                self,
                self.solver,
                name='solver'
        )

        # Register buffers
        buffer_registry.register(
            'stage_increment',
            self,
            n,
            config.stage_increment_location,
            persistent=True,
            precision=precision
        )
        buffer_registry.register(
            'accumulator',
            self,
            accumulator_length,
            config.accumulator_location,
            precision=precision
        )


        buffer_registry.register(
            'stage_base',
            self,
            n,
            config.stage_base_location,
            aliases='accumulator',
            precision=precision
        )

        buffer_registry.register(
            'stage_rhs',
            self,
            n,
            config.stage_rhs_location,
            persistent=True,
            precision=precision
        )

    def build_implicit_helpers(
        self,
    ) -> Callable:
        """Construct the nonlinear solver chain used by implicit methods."""

        config = self.compile_settings
        beta = config.beta
        gamma = config.gamma
        mass = config.M
        preconditioner_order = config.preconditioner_order

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
            "linear_operator",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        # Update solvers with device functions
        self.solver.update(
            operator_apply=operator,
            preconditioner=preconditioner,
            residual_function=residual,
        )
        
        self.update_compile_settings(
                {'solver_function': self.solver.device_function}
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
        """Compile the DIRK device step."""

        config = self.compile_settings
        tableau = config.tableau
        nonlinear_solver = solver_function

        n = int32(n)
        stage_count = int32(tableau.stage_count)
        stages_except_first = stage_count - int32(1)

        # Compile-time toggles
        has_driver_function = driver_function is not None
        has_error = self.is_adaptive
        multistage = stage_count > 1
        first_same_as_last = self.first_same_as_last
        can_reuse_accepted_start = self.can_reuse_accepted_start

        explicit_a_coeffs = tableau.explicit_terms(numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        typed_zero = numba_precision(0.0)
        error_weights = tableau.error_weights(numba_precision)
        if error_weights is None or not has_error:
            error_weights = tuple(typed_zero for _ in range(stage_count))
        stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)
        diagonal_coeffs = tableau.diagonal(numba_precision)

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

        stage_implicit = tuple(coeff != numba_precision(0.0)
                          for coeff in diagonal_coeffs)
        accumulator_length = int32(max(stage_count - 1, 0) * n)

        # Get child allocators for Newton solver
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(self, self.solver,
                                                 name='solver')
        )

        # Get allocators from buffer registry
        getalloc = buffer_registry.get_allocator
        alloc_stage_increment = getalloc('stage_increment', self)
        alloc_accumulator = getalloc('accumulator', self)
        alloc_stage_base = getalloc('stage_base', self)
        alloc_stage_rhs = getalloc('stage_rhs', self)



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
            # Shared and local buffer guide:
            # stage_accumulator: size (stage_count-1) * n, shared memory.
            #   Default behaviour:
            #       - Stores accumulated explicit contributions for successors.
            #       - Slice k feeds the base state for stage k+1.
            #   Reuse:
            #       - stage_base: first slice (size n)
            #           - Holds the working state during the current stage.
            #           - New data lands only after the prior stage has finished.
            # solver_scratch: size solver_shared_elements, shared memory.
            #   Default behaviour:
            #       - Provides workspace for the Newton iteration helpers.
            #   Reuse:
            #       - stage_rhs: first slice (size n)
            #           - Carries the Newton residual and then the stage rhs.
            #           - Once a stage closes we reuse it for the next residual,
            #             so no live data remains.
            #       - increment_cache: second slice (size n)
            #           - Receives the accepted increment at step end for FSAL.
            #           - Solver stops touching it once convergence is reached.
            #   Note:
            #       - Evaluation state is computed inline by operators and
            #         residuals; no dedicated buffer required.
            # stage_increment: size n, shared or local memory.
            #   Default behaviour:
            #       - Starts as the Newton guess and finishes as the step.
            #       - Copied into increment_cache once the stage closes.
            # proposed_state: size n, global memory.
            #   Default behaviour:
            #       - Carries the running solution with each stage update.
            #       - Only updated after a stage converges, keeping data stable.
            # proposed_drivers / proposed_observables: size n each, global.
            #   Default behaviour:
            #       - Refresh to the stage time before rhs or residual work.
            #       - Later stages reuse only the newest values, so no clashes.
            # ----------------------------------------------------------- #

            # ----------------------------------------------------------- #
            # Selective allocation from local or shared memory
            # ----------------------------------------------------------- #
            stage_increment = alloc_stage_increment(shared, persistent_local)
            stage_accumulator = alloc_accumulator(shared, persistent_local)
            stage_base = alloc_stage_base(shared, persistent_local)
            solver_shared = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(shared, persistent_local)
            stage_rhs = alloc_stage_rhs(shared, persistent_local)

            # Initialize local arrays
            for _i in range(n):
                stage_increment[_i] = typed_zero
                stage_base[_i] = typed_zero
            for _i in range(accumulator_length):
                stage_accumulator[_i] = typed_zero
            # --------------------------------------------------------------- #

            current_time = time_scalar
            end_time = current_time + dt_scalar

            for idx in range(n):
                if has_error and accumulates_error:
                    error[idx] = typed_zero

            status_code = int32(0)
            # --------------------------------------------------------------- #
            #            Stage 0: may reuse cached values                     #
            # --------------------------------------------------------------- #

            first_step = first_step_flag != int32(0)

            # Only use cache if all threads in warp can - otherwise no gain
            use_cached_rhs = False
            # Compile-time branch: guarded by static configuration flags
            if first_same_as_last and multistage:
                # Runtime branch: depends on previous step acceptance
                if not first_step:
                    mask = activemask()
                    all_threads_accepted = all_sync(mask, accepted_flag != int32(0))
                    use_cached_rhs = all_threads_accepted
            else:
                use_cached_rhs = False

            stage_time = current_time + dt_scalar * stage_time_fractions[0]
            diagonal_coeff = diagonal_coeffs[0]

            for idx in range(n):
                stage_base[idx] = state[idx]
                if accumulates_output:
                    proposed_state[idx] = typed_zero

            # Recompute if not FSAL cached
            if not use_cached_rhs:
                if can_reuse_accepted_start:
                    for idx in range(int32(drivers_buffer.shape[0])):
                        # Use step-start driver values
                        proposed_drivers[idx] = drivers_buffer[idx]

                else:
                    if has_driver_function:
                        driver_function(
                            stage_time,
                            driver_coeffs,
                            proposed_drivers,
                        )

                if stage_implicit[0]:
                    solver_status = nonlinear_solver(
                        stage_increment,
                        parameters,
                        proposed_drivers,
                        stage_time,
                        dt_scalar,
                        diagonal_coeffs[0],
                        stage_base,
                        solver_shared,
                        solver_persistent,
                        counters,
                    )
                    status_code = int32(status_code | solver_status)

                    for idx in range(n):
                        stage_base[idx] += (
                            diagonal_coeff * stage_increment[idx]
                        )

                # Get obs->dxdt from stage_base
                observables_function(
                    stage_base,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                dxdt_fn(
                    stage_base,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

            solution_weight = solution_weights[0]
            error_weight = error_weights[0]
            for idx in range(n):
                rhs_value = stage_rhs[idx]
                # Accumulate if required; save directly if tableau allows
                if accumulates_output:
                    # Standard accumulation
                    proposed_state[idx] += solution_weight * rhs_value
                elif b_row == int32(0):
                    # Direct assignment when stage 0 matches b_row
                    proposed_state[idx] = stage_base[idx]
                if has_error:
                    if accumulates_error:
                        # Standard accumulation
                        error[idx] += error_weight * rhs_value
                    elif b_hat_row == int32(0):
                        # Direct assignment for error
                        error[idx] = stage_base[idx]
                        
            for idx in range(accumulator_length):
                stage_accumulator[idx] = typed_zero

            # --------------------------------------------------------------- #
            #            Stages 1-s: must refresh all qtys                    #
            # --------------------------------------------------------------- #
            mask = activemask()
            for prev_idx in range(stages_except_first):
                # DIRK is missing the instruction cache. The unrolled loop
                # is instruction dense, taking up most of the instruction space.
                # A block-wide sync hangs indefinitely, as some warps will
                # finish early and never reach it. We sync a warp to minimal
                # effect (it's a wash in the profiler) in case of divergence in
                # big systems.
                syncwarp(mask)
                stage_offset = prev_idx * n
                stage_idx = prev_idx + int32(1)
                matrix_col = explicit_a_coeffs[prev_idx]

                # Stream previous stage's RHS into accumulators for successors
                for successor_idx in range(stages_except_first):
                    coeff = matrix_col[successor_idx + int32(1)]
                    row_offset = successor_idx * n
                    for idx in range(n):
                        contribution = coeff * stage_rhs[idx]
                        stage_accumulator[row_offset + idx] += contribution

                stage_time = (
                    current_time + dt_scalar * stage_time_fractions[stage_idx]
                )

                if has_driver_function:
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        proposed_drivers,
                    )

                # Convert accumulator slice to state by adding y_n
                for idx in range(n):
                    stage_base[idx] = (stage_accumulator[stage_offset + idx]
                                       * dt_scalar + state[idx])

                diagonal_coeff = diagonal_coeffs[stage_idx]

                if stage_implicit[stage_idx]:
                    solver_status = nonlinear_solver(
                        stage_increment,
                        parameters,
                        proposed_drivers,
                        stage_time,
                        dt_scalar,
                        diagonal_coeffs[stage_idx],
                        stage_base,
                        solver_shared,
                        solver_persistent,
                        counters,
                    )
                    status_code = int32(status_code | solver_status)

                    for idx in range(n):
                        stage_base[idx] += diagonal_coeff * stage_increment[idx]

                observables_function(
                    stage_base,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                dxdt_fn(
                    stage_base,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                solution_weight = solution_weights[stage_idx]
                error_weight = error_weights[stage_idx]
                for idx in range(n):
                    increment = stage_rhs[idx]
                    if accumulates_output:
                        proposed_state[idx] += solution_weight * increment
                    elif b_row == stage_idx:
                        proposed_state[idx] = stage_base[idx]

                    if has_error:
                        if accumulates_error:
                            error[idx] += error_weight * increment
                        elif b_hat_row == stage_idx:
                            # Direct assignment for error
                            error[idx] = stage_base[idx]

            # --------------------------------------------------------------- #

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] *= dt_scalar
                    proposed_state[idx] += state[idx]
                if has_error:
                    if accumulates_error:
                        error[idx] *= dt_scalar
                    else:
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

            return int32(status_code)
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
