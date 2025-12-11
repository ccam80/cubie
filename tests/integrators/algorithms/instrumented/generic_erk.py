"""Generic explicit Runge--Kutta integration step with streamed accumulators."""

from typing import Callable, Optional

import attrs
from attrs import validators
from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.cuda_simsafe import activemask, all_sync
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.generic_erk import (
    ERKBufferSettings,
)
from cubie.integrators.algorithms.ode_explicitstep import (
    ExplicitStepConfig,
    ODEExplicitStep,
)
from cubie.integrators.algorithms.generic_erk_tableaus import (
    DEFAULT_ERK_TABLEAU,
    ERKTableau,
)


ERK_ADAPTIVE_DEFAULTS = StepControlDefaults(
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

ERK_FIXED_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)


@attrs.define
class ERKStepConfig(ExplicitStepConfig):
    """Configuration describing an explicit Runge--Kutta integrator."""

    tableau: ERKTableau = attrs.field(default=DEFAULT_ERK_TABLEAU)
    buffer_settings: Optional[ERKBufferSettings] = attrs.field(
        default=None,
        validator=validators.optional(
            validators.instance_of(ERKBufferSettings)
        ),
    )

    @property
    def first_same_as_last(self) -> bool:
        """Return ``True`` when the tableau shares the first and last stage."""

        return self.tableau.first_same_as_last

class ERKStep(ODEExplicitStep):
    """Generic explicit Runge--Kutta step with configurable tableaus."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        tableau: ERKTableau = DEFAULT_ERK_TABLEAU,
        n_drivers: int = 0,
    ) -> None:
        """Initialise the Runge--Kutta step configuration.

        Parameters
        ----------
        tableau
            Explicit Runge--Kutta tableau describing the coefficients used by
            the integrator. Defaults to :data:`DEFAULT_ERK_TABLEAU`.
        """

        # Create buffer_settings
        buffer_settings = ERKBufferSettings(
            n=n,
            stage_count=tableau.stage_count,
        )
        config_kwargs = {
            "precision": precision,
            "n": n,
            "n_drivers": n_drivers,
            "dxdt_function": dxdt_function,
            "observables_function": observables_function,
            "driver_function": driver_function,
            "get_solver_helper_fn": get_solver_helper_fn,
            "tableau": tableau,
            "buffer_settings": buffer_settings,
        }
        config = ERKStepConfig(**config_kwargs)
        
        if tableau.has_error_estimate:
            defaults = ERK_ADAPTIVE_DEFAULTS
        else:
            defaults = ERK_FIXED_DEFAULTS
        
        super().__init__(config, defaults)

    def build_step(
        self,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the explicit Runge--Kutta device step."""

        config = self.compile_settings
        precision = self.precision
        tableau = config.tableau

        typed_zero = numba_precision(0.0)
        n_arraysize = n
        n = int32(n)
        stage_count = int32(tableau.stage_count)
        stages_except_first = stage_count - int32(1)
        accumulator_length = (tableau.stage_count - 1) * n_arraysize

        has_driver_function = driver_function is not None
        first_same_as_last = self.first_same_as_last
        multistage = stage_count > 1
        has_error = self.is_adaptive

        stage_rhs_coeffs = tableau.typed_columns(tableau.a, numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        stage_nodes = tableau.typed_vector(tableau.c, numba_precision)

        if has_error:
            error_weights = tableau.error_weights(numba_precision)
        else:
            error_weights = tuple(typed_zero for _ in range(stage_count))

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

        # Buffer settings from compile_settings for selective shared/local
        buffer_settings = config.buffer_settings

        # Unpack boolean flags as compile-time constants
        stage_rhs_shared = buffer_settings.use_shared_stage_rhs
        stage_accumulator_shared = buffer_settings.use_shared_stage_accumulator
        stage_cache_shared = buffer_settings.use_shared_stage_cache

        # Unpack slice indices for shared memory layout
        shared_indices = buffer_settings.shared_indices
        stage_rhs_slice = shared_indices.stage_rhs
        stage_accumulator_slice = shared_indices.stage_accumulator
        stage_cache_slice = shared_indices.stage_cache

        # Unpack local sizes for local array allocation
        local_sizes = buffer_settings.local_sizes
        stage_rhs_local_size = local_sizes.nonzero('stage_rhs')
        stage_accumulator_local_size = local_sizes.nonzero('stage_accumulator')
        stage_cache_local_size = local_sizes.nonzero('stage_cache')

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
            # stage_accumulator: size (stage_count-1) * n, shared or local.
            #   Default behaviour:
            #       - Holds finished stage rhs * dt for later stage sums.
            #       - Slice k stores contributions streamed into stage k+1.
            #   Reuse:
            #       - stage_cache: first slice (size n)
            #           - Saves the FSAL rhs when the tableau supports it.
            #           - Cache survives after the loop so no live slice is hit.
            # proposed_state: size n, global memory.
            #   Default behaviour:
            #       - Starts as the accepted state and gathers stage updates.
            #       - Each stage applies its weighted increment before moving on.
            # proposed_drivers / proposed_observables: size n each, global.
            #   Default behaviour:
            #       - Refresh to the current stage time before rhs evaluation.
            #       - Later stages only read the newest values, so nothing lingers.
            # stage_rhs: size n, shared or local memory.
            #   Default behaviour:
            #       - Holds the current stage rhs before scaling by dt.
            #   Reuse:
            #       - When FSAL hits we copy cached rhs here before touching
            #         shared memory, keeping lifetimes separate.
            # error: size n, global memory (adaptive runs only).
            #   Default behaviour:
            #       - Accumulates error-weighted f(y_jn) during the loop.
            #       - Cleared at loop entry so prior steps cannot leak in.
            # ----------------------------------------------------------- #

            # ----------------------------------------------------------- #
            # Selective allocation from local or shared memory
            # ----------------------------------------------------------- #
            if stage_rhs_shared:
                stage_rhs = shared[stage_rhs_slice]
            else:
                stage_rhs = cuda.local.array(stage_rhs_local_size, precision)
                for _i in range(stage_rhs_local_size):
                    stage_rhs[_i] = typed_zero

            if stage_accumulator_shared:
                stage_accumulator = shared[stage_accumulator_slice]
            else:
                stage_accumulator = cuda.local.array(
                    stage_accumulator_local_size, precision
                )
                for _i in range(stage_accumulator_local_size):
                    stage_accumulator[_i] = typed_zero

            if multistage:
                # stage_cache persists between steps for FSAL optimization.
                # When shared, slice from shared memory; when local, use
                # persistent_local to maintain state between step invocations.
                if stage_cache_shared:
                    stage_cache = shared[stage_cache_slice]
                else:
                    stage_cache = persistent_local[:stage_cache_local_size]
            # ----------------------------------------------------------- #

            observable_count = proposed_observables.shape[0]

            current_time = time_scalar
            end_time = current_time + dt_scalar

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] = typed_zero
                if has_error and accumulates_error:
                    error[idx] = typed_zero

            for idx in range(n):
                stage_states[0, idx] = state[idx]
                residuals[0, idx] = typed_zero
                jacobian_updates[0, idx] = typed_zero
                stage_increments[0, idx] = typed_zero
            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = observables[obs_idx]
            for driver_idx in range(stage_drivers_out.shape[1]):
                stage_drivers_out[0, driver_idx] = drivers_buffer[driver_idx]

            status_code = int32(0)
            # ----------------------------------------------------------- #
            #            Stage 0: operates out of supplied buffers          #
            # ----------------------------------------------------------- #
            # Only use cache if all threads in warp can - otherwise no gain
            use_cached_rhs = False
            if first_same_as_last and multistage:
                if not first_step_flag:
                    mask = activemask()
                    all_threads_accepted = all_sync(mask, accepted_flag != int32(0))
                    use_cached_rhs = all_threads_accepted
            else:
                use_cached_rhs = False

            if multistage:
                if use_cached_rhs:
                    for idx in range(n):
                        stage_rhs[idx] = stage_cache[idx]
                else:
                    dxdt_fn(
                        state,
                        parameters,
                        drivers_buffer,
                        observables,
                        stage_rhs,
                        current_time,
                    )
            else:
                dxdt_fn(
                    state,
                    parameters,
                    drivers_buffer,
                    observables,
                    stage_rhs,
                    current_time,
                )

            for idx in range(n):
                stage_derivatives[0, idx] = stage_rhs[idx]
                stage_increments[0, idx] = dt_scalar * stage_rhs[idx]

            for idx in range(n):
                increment = stage_rhs[idx]
                if accumulates_output:
                    proposed_state[idx] += solution_weights[0] * increment
                if has_error:
                    if accumulates_error:
                        error[idx] += error_weights[0] * increment


            for idx in range(accumulator_length):
                stage_accumulator[idx] = typed_zero

            # ----------------------------------------------------------- #
            #            Stages 1-s: refresh observables and drivers       #
            # ----------------------------------------------------------- #

            for prev_idx in range(stages_except_first):
                stage_offset = prev_idx * n
                stage_idx = prev_idx + int32(1)
                matrix_col = stage_rhs_coeffs[prev_idx]

                # Stream last result into the accumulators
                for successor_idx in range(stages_except_first):
                    coeff = matrix_col[successor_idx + int32(1)]
                    row_offset = successor_idx * n
                    for idx in range(n):
                        increment = stage_rhs[idx]
                        stage_accumulator[row_offset + idx] += (
                            coeff * increment
                        )

                base = stage_offset
                dt_stage = dt_scalar * stage_nodes[stage_idx]
                stage_time = current_time + dt_stage

                # Convert accumulated gradients sum(f(y_nj) into a state y_j
                for idx in range(n):
                    stage_accumulator[base] = (
                        stage_accumulator[base] * dt_scalar + state[idx]
                    )
                    base += int32(1)

                stage_state = stage_accumulator[stage_offset:stage_offset + n]

                for idx in range(n):
                    stage_states[stage_idx, idx] = stage_state[idx]
                    residuals[stage_idx, idx] = typed_zero
                    jacobian_updates[stage_idx, idx] = typed_zero
                    stage_increments[stage_idx, idx] = typed_zero

                stage_driver_values = proposed_drivers
                if has_driver_function:
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        stage_driver_values,
                    )
                for driver_idx in range(stage_drivers_out.shape[1]):
                    stage_drivers_out[stage_idx, driver_idx] = stage_driver_values[driver_idx]

                observables_function(
                    stage_state,
                    parameters,
                    stage_driver_values,
                    proposed_observables,
                    stage_time,
                )

                for obs_idx in range(observable_count):
                    stage_observables[stage_idx, obs_idx] = proposed_observables[obs_idx]

                dxdt_fn(
                    stage_state,
                    parameters,
                    stage_driver_values,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                for idx in range(n):
                    stage_derivatives[stage_idx, idx] = stage_rhs[idx]
                    stage_increments[stage_idx, idx] = (
                        dt_scalar * stage_rhs[idx]
                    )

                # Accumulate f(y_jn) terms or capture direct stage state
                solution_weight = solution_weights[stage_idx]
                error_weight = error_weights[stage_idx]
                for idx in range(n):
                    if accumulates_output:
                        increment = stage_rhs[idx]
                        proposed_state[idx] += solution_weight * increment
                    elif b_row == stage_idx:
                        proposed_state[idx] = stage_state[idx]

                    if has_error:
                        if accumulates_error:
                            increment = stage_rhs[idx]
                            error[idx] += error_weight * increment
                        elif b_hat_row == stage_idx:
                            error[idx] = stage_state[idx]


            # ----------------------------------------------------------- #
            for idx in range(n):

                # Scale and shift f(Y_n) value if accumulated
                if accumulates_output:
                    proposed_state[idx] *= dt_scalar
                    proposed_state[idx] += state[idx]

                if has_error:
                    # Scale error if accumulated
                    if accumulates_error:
                        error[idx] *= dt_scalar

                    #Or form error from difference if captured from a-row
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
            if first_same_as_last:
                for idx in range(n):
                    stage_cache[idx] = stage_rhs[idx]
            return status_code

        return StepCache(step=step)

    @property
    def is_multistage(self) -> bool:
        """Return ``True`` when the method has multiple stages."""
        return self.tableau.stage_count > 1

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` if algorithm calculates an error estimate."""
        return self.tableau.has_error_estimate

    @property
    def shared_memory_required(self) -> int:
        """Return the number of precision entries required in shared memory."""
        return self.compile_settings.buffer_settings.shared_memory_elements

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""
        return self.compile_settings.n

    @property
    def persistent_local_required(self) -> int:
        """Return the number of persistent local entries required.

        Returns n for stage_cache when neither stage_rhs nor stage_accumulator
        uses shared memory. When either is shared, stage_cache aliases it.
        """
        buffer_settings = self.compile_settings.buffer_settings
        return buffer_settings.persistent_local_elements

    @property
    def order(self) -> int:
        """Return the classical order of accuracy."""

        return self.tableau.order

    @property
    def threads_per_step(self) -> int:
        """Return the number of CUDA threads that advance one state."""

        return 1
