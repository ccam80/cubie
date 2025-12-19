"""Diagonally implicit Runge–Kutta integration step implementation."""

from typing import Callable, Optional

import attrs
import numpy as np
from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.buffer_registry import buffer_registry
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
from tests.integrators.algorithms.instrumented.matrix_free_solvers import (
    inst_linear_solver_factory,
    inst_newton_krylov_solver_factory,
)


DIRK_ADAPTIVE_DEFAULTS = StepControlDefaults(
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

DIRK_FIXED_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)

@attrs.define
class DIRKStepConfig(ImplicitStepConfig):
    """Configuration describing the DIRK integrator."""

    tableau: DIRKTableau = attrs.field(
        default=DEFAULT_DIRK_TABLEAU,
    )
    stage_increment_location: str = attrs.field(default='local')
    stage_base_location: str = attrs.field(default='local')
    accumulator_location: str = attrs.field(default='local')


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
        preconditioner_order: int = 2,
        krylov_tolerance: float = 1e-6,
        max_linear_iters: int = 200,
        linear_correction_type: str = "minimal_residual",
        newton_tolerance: float = 1e-6,
        max_newton_iters: int = 100,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
        tableau: DIRKTableau = DEFAULT_DIRK_TABLEAU,
        n_drivers: int = 0,
        stage_increment_location: Optional[str] = None,
        stage_base_location: Optional[str] = None,
        accumulator_location: Optional[str] = None,
    ) -> None:
        """Initialise the DIRK step configuration."""

        mass = np.eye(n, dtype=precision)

        # Clear any existing buffer registrations
        buffer_registry.clear_parent(self)

        # Determine locations (use defaults if not specified)
        inc_loc = stage_increment_location if stage_increment_location else 'local'
        base_loc = stage_base_location if stage_base_location else 'local'
        acc_loc = accumulator_location if accumulator_location else 'local'

        # Calculate buffer sizes
        accumulator_length = max(tableau.stage_count - 1, 0) * n
        multistage = tableau.stage_count > 1

        # Register algorithm buffers
        buffer_registry.register(
            'dirk_stage_increment', self, n, inc_loc, precision=precision
        )
        buffer_registry.register(
            'dirk_accumulator', self, accumulator_length, acc_loc,
            precision=precision
        )

        # stage_base aliasing: can alias accumulator when both are shared
        # and method has multiple stages
        stage_base_aliases_acc = (
            multistage and acc_loc == 'shared' and base_loc == 'shared'
        )
        if stage_base_aliases_acc:
            buffer_registry.register(
                'dirk_stage_base', self, n, 'shared',
                aliases='dirk_accumulator', precision=precision
            )
        else:
            buffer_registry.register(
                'dirk_stage_base', self, n, base_loc, precision=precision
            )

        # solver_scratch is always shared (Newton delta + residual)
        solver_shared_size = 2 * n
        buffer_registry.register(
            'dirk_solver_scratch', self, solver_shared_size, 'shared',
            precision=precision
        )

        # FSAL caches alias solver_scratch
        buffer_registry.register(
            'dirk_rhs_cache', self, n, 'shared',
            aliases='dirk_solver_scratch', precision=precision
        )
        buffer_registry.register(
            'dirk_increment_cache', self, n, 'shared',
            aliases='dirk_solver_scratch', precision=precision
        )

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
            "stage_increment_location": inc_loc,
            "stage_base_location": base_loc,
            "accumulator_location": acc_loc,
        }

        config = DIRKStepConfig(**config_kwargs)
        self._cached_auxiliary_count = 0
        
        if tableau.has_error_estimate:
            defaults = DIRK_ADAPTIVE_DEFAULTS
        else:
            defaults = DIRK_FIXED_DEFAULTS
        
        super().__init__(config, defaults)

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

        krylov_tolerance = config.krylov_tolerance
        max_linear_iters = config.max_linear_iters
        correction_type = config.linear_correction_type

        linear_solver = inst_linear_solver_factory(operator, n=n,
                                                   preconditioner=preconditioner,
                                                   correction_type=correction_type,
                                                   tolerance=krylov_tolerance,
                                                   max_iters=max_linear_iters)

        newton_tolerance = config.newton_tolerance
        max_newton_iters = config.max_newton_iters
        newton_damping = config.newton_damping
        newton_max_backtracks = config.newton_max_backtracks

        nonlinear_solver = inst_newton_krylov_solver_factory(
            residual_function=residual, linear_solver=linear_solver, n=n,
            tolerance=newton_tolerance, max_iters=max_newton_iters,
            damping=newton_damping, max_backtracks=newton_max_backtracks,
            precision=precision)

        return nonlinear_solver

    def build_step(
        self,
        solver_fn: Callable,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the DIRK device step."""

        config = self.compile_settings
        precision = self.precision
        tableau = config.tableau
        nonlinear_solver = solver_fn
        n_arraysize = n
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

        # Get allocators from buffer registry
        alloc_stage_increment = buffer_registry.get_allocator(
            'dirk_stage_increment', self
        )
        alloc_accumulator = buffer_registry.get_allocator(
            'dirk_accumulator', self
        )
        alloc_stage_base = buffer_registry.get_allocator(
            'dirk_stage_base', self
        )
        alloc_solver_scratch = buffer_registry.get_allocator(
            'dirk_solver_scratch', self
        )
        alloc_rhs_cache = buffer_registry.get_allocator(
            'dirk_rhs_cache', self
        )
        alloc_increment_cache = buffer_registry.get_allocator(
            'dirk_increment_cache', self
        )

        # FSAL scratch allocation flags (solver_scratch >= 2n always)
        has_rhs_in_scratch = True
        has_increment_in_scratch = True

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
            solver_scratch = alloc_solver_scratch(shared, persistent_local)

            # Initialize local arrays
            for _i in range(n):
                stage_increment[_i] = typed_zero
                stage_base[_i] = typed_zero
            for _i in range(accumulator_length):
                stage_accumulator[_i] = typed_zero

            # --------------------------------------------------------------- #
            # Instrumentation local buffers
            base_state_snapshot = cuda.local.array(n_arraysize, numba_precision)
            observable_count = proposed_observables.shape[0]

            current_time = time_scalar
            end_time = current_time + dt_scalar

            # stage_rhs is used during Newton iterations and overwritten.
            # slice from solver_scratch (size 2n always provides space)
            stage_rhs = solver_scratch[:n]

            # increment_cache and rhs_cache persist between steps for FSAL.
            # Both fit in solver_scratch (size 2n)
            increment_cache = solver_scratch[n:int32(2)*n]
            rhs_cache = solver_scratch[:n]  # Aliases stage_rhs

            for idx in range(n):
                if has_error and accumulates_error:
                    error[idx] = typed_zero
                stage_increment[idx] = increment_cache[idx]  # cache spent

            status_code = int32(0)
            # --------------------------------------------------------------- #
            #            Stage 0: may reuse cached values                     #
            # --------------------------------------------------------------- #

            first_step = first_step_flag != int32(0)

            # Only use cache if all threads in warp can - otherwise no gain
            use_cached_rhs = False
            if first_same_as_last and multistage:
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

            if use_cached_rhs:
                # Load cached RHS from persistent storage.
                # When rhs_cache aliases stage_rhs (has_rhs_in_scratch=True),
                # this is a no-op. Otherwise, copy from persistent_local.
                if not has_rhs_in_scratch:
                    for idx in range(n):
                        stage_rhs[idx] = rhs_cache[idx]

            else:
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
                    status_code |= nonlinear_solver(
                        stage_increment,
                        parameters,
                        proposed_drivers,
                        stage_time,
                        dt_scalar,
                        diagonal_coeffs[0],
                        stage_base,
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

            for idx in range(n):
                stage_derivatives[0, idx] = stage_rhs[idx]
                stage_states[0, idx] = stage_base[idx]
                residuals[0, idx] = solver_scratch[idx + n]
                jacobian_updates[0, idx] = typed_zero
                stage_increments[0, idx] = (stage_increment[idx] *
                                            diagonal_coeff)
            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = proposed_observables[obs_idx]
            for driver_idx in range(proposed_drivers_out.shape[1]):
                proposed_drivers_out[0, driver_idx] = proposed_drivers[
                    driver_idx
                ]

            solution_weight = solution_weights[0]
            error_weight = error_weights[0]
            for idx in range(n):
                rhs_value = stage_rhs[idx]
                # Accumulate if required; save directly if tableau allows
                if accumulates_output:
                    # Standard accumulation
                    proposed_state[idx] += solution_weight * rhs_value
                elif b_row == 0:
                    # Direct assignment when stage 0 matches b_row
                    proposed_state[idx] = stage_base[idx]
                if has_error:
                    if accumulates_error:
                        # Standard accumulation
                        error[idx] += error_weight * rhs_value
                    elif b_hat_row == 0:
                        # Direct assignment for error
                        error[idx] = stage_base[idx]

            for idx in range(accumulator_length):
                stage_accumulator[idx] = typed_zero

            # --------------------------------------------------------------- #
            #            Stages 1-s: must refresh obs/drivers                 #
            # --------------------------------------------------------------- #
            mask = activemask()
            for prev_idx in range(stages_except_first):

                #DIRK is missing the instruction cache. The unrolled stage loop
                # is instruction dense, taking up most of the instruction space.
                # Try syncing block-wide per-stage to see whether this will help
                # the whole block stay in one cache chunk. Play with block size
                # to enforce the number of blocks per SM.
                syncwarp(mask)
                stage_offset = int32(prev_idx * n)
                stage_idx = prev_idx + int32(1)
                matrix_col = explicit_a_coeffs[prev_idx]

                # Stream previous stage's RHS into accumulators for successors
                # Only stream to current stage and later (not already-processed)
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

                for driver_idx in range(proposed_drivers_out.shape[1]):
                    proposed_drivers_out[stage_idx, driver_idx] = proposed_drivers[driver_idx]

                # Convert accumulator slice to state by adding y_n
                for idx in range(n):
                    stage_base[idx] = (
                            stage_accumulator[stage_offset + idx]* dt_scalar
                            + state[idx]
                    )


                diagonal_coeff = diagonal_coeffs[stage_idx]

                for idx in range(n):
                    base_state_snapshot[idx] = stage_base[idx]

                if stage_implicit[stage_idx]:
                    solver_ret = nonlinear_solver(
                        stage_increment,
                        parameters,
                        proposed_drivers,
                        stage_time,
                        dt_scalar,
                        diagonal_coeffs[stage_idx],
                        stage_base,
                        solver_scratch,
                        counters,
                        int32(stage_idx),
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

                    for idx in range(n):
                        stage_base[idx] += diagonal_coeff * stage_increment[idx]

                for idx in range(n):
                    stage_states[stage_idx, idx] = stage_base[idx]
                    scaled_increment = diagonal_coeff * stage_increment[idx]
                    stage_increments[stage_idx, idx] = scaled_increment
                    jacobian_updates[stage_idx, idx] = typed_zero

                observables_function(
                    stage_base,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                for obs_idx in range(observable_count):
                    stage_observables[stage_idx, obs_idx] = proposed_observables[obs_idx]

                dxdt_fn(
                    stage_base,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                for idx in range(n):
                    stage_derivatives[stage_idx, idx] = stage_rhs[idx]
                    residuals[stage_idx, idx] = solver_scratch[idx + n]

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


            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] *= dt_scalar
                    proposed_state[idx] += state[idx]
                if has_error:
                    if accumulates_error:
                        error[idx] *= dt_scalar
                    else:
                        error[idx] = proposed_state[idx] - error[idx]

            # --------------------------------------------------------------- #
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

            #Cache end-step values as appropriate
            # Cache increment and RHS for FSAL optimization
            for idx in range(n):
                increment_cache[idx] = stage_increment[idx]
                # rhs_cache aliases stage_rhs when has_rhs_in_scratch=True,
                # so no explicit copy needed. Otherwise, copy to persistent_local.
                if not has_rhs_in_scratch:
                    rhs_cache[idx] = stage_rhs[idx]

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
