"""Generic explicit Runge--Kutta integration step with streamed accumulators."""

from typing import Callable, Optional

import attrs
from numba import cuda, int16, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.ode_explicitstep import (
    ExplicitStepConfig,
    ODEExplicitStep,
)
from cubie.integrators.algorithms.generic_erk_tableaus import (
    DEFAULT_ERK_TABLEAU,
    ERKTableau,
)


ERK_DEFAULTS = StepControlDefaults(
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
class ERKStepConfig(ExplicitStepConfig):
    """Configuration describing an explicit Runge--Kutta integrator."""

    tableau: ERKTableau = attrs.field(default=DEFAULT_ERK_TABLEAU)

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
        dt: Optional[float],
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        tableau: ERKTableau = DEFAULT_ERK_TABLEAU,
    ) -> None:
        """Initialise the Runge--Kutta step configuration.

        Parameters
        ----------
        tableau
            Explicit Runge--Kutta tableau describing the coefficients used by
            the integrator. Defaults to :data:`DEFAULT_ERK_TABLEAU`.
        """

        config = ERKStepConfig(
            precision=precision,
            n=n,
            dt=dt,
            dxdt_function=dxdt_function,
            observables_function=observables_function,
            driver_function=driver_function,
            get_solver_helper_fn=get_solver_helper_fn,
            tableau=tableau,
        )
        super().__init__(config, ERK_DEFAULTS)

    def build_step(
        self,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
        dt: Optional[float],
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the explicit Runge--Kutta device step."""

        config = self.compile_settings
        tableau = config.tableau

        stage_count = tableau.stage_count
        accumulator_length = max(stage_count - 1, 0) * n
        typed_zero = numba_precision(0.0)

        has_driver_function = driver_function is not None
        multistage = stage_count > 1
        has_error = self.is_adaptive
        first_same_as_last = self.first_same_as_last

        stage_rhs_coeffs = tableau.typed_rows(tableau.a, numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        error_weights = tableau.error_weights(numba_precision)
        if error_weights is None or not has_error:
            error_weights = tuple(typed_zero for _ in range(stage_count))
        stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)

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

            observable_count = proposed_observables.shape[0]

            dt_value = dt_scalar
            current_time = time_scalar
            end_time = current_time + dt_value

            stage_accumulator = shared[:accumulator_length]
            if multistage:
                stage_cache = stage_accumulator[:n]

            for idx in range(n):
                proposed_state[idx] = typed_zero
                if has_error:
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
            use_cached_rhs = (
                (not first_step_flag) and accepted_flag and first_same_as_last
            )
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
                stage_increments[0, idx] = dt_value * stage_rhs[idx]

            for idx in range(n):
                increment = stage_rhs[idx]
                proposed_state[idx] += solution_weights[0] * increment
                if has_error:
                    error[idx] += error_weights[0] * increment

            for idx in range(accumulator_length):
                stage_accumulator[idx] = typed_zero

            # ----------------------------------------------------------- #
            #            Stages 1-s: refresh observables and drivers       #
            # ----------------------------------------------------------- #

            for stage_idx in range(1, stage_count):

                # Stream last result into the accumulators
                prev_idx = stage_idx - 1
                successor_range = stage_count - stage_idx

                for successor_offset in range(successor_range):
                    successor_idx = stage_idx + successor_offset
                    state_coeff = stage_rhs_coeffs[successor_idx][prev_idx]
                    base = (successor_idx - 1) * n
                    for idx in range(n):
                        # 1x duplicated FMUL to avoid a memory save/load - nice
                        increment = stage_rhs[idx]
                        contribution = state_coeff * increment
                        stage_accumulator[base + idx] += contribution

                stage_offset = (stage_idx - 1) * n
                stage_time = (
                    current_time + dt_value * stage_time_fractions[stage_idx]
                )

                for idx in range(n):
                    stage_accumulator[stage_offset + idx] = (
                        state[idx] + stage_accumulator[stage_offset + idx] *
                        dt_value
                    )

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
                    stage_increments[stage_idx, idx] = dt_value * stage_rhs[idx]

                for idx in range(n):
                    increment = stage_rhs[idx]
                    proposed_state[idx] += (
                        solution_weights[stage_idx] * increment
                    )
                    if has_error:
                        error[idx] += (
                            error_weights[stage_idx] * increment
                        )


            # ----------------------------------------------------------- #
            for idx in range(n):
                proposed_state[idx] *= dt_value
                proposed_state[idx] += state[idx]
                if has_error:
                    error[idx] *= dt_value

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
        stage_count = self.tableau.stage_count
        accumulator_span = max(stage_count - 1, 0) * self.compile_settings.n
        return accumulator_span

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""
        return self.compile_settings.n

    @property
    def persistent_local_required(self) -> int:
        """Return the number of persistent local entries required."""
        return 0

    @property
    def order(self) -> int:
        """Return the classical order of accuracy."""

        return self.tableau.order

    @property
    def threads_per_step(self) -> int:
        """Return the number of CUDA threads that advance one state."""

        return 1
