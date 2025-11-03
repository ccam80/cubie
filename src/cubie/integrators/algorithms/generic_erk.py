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
        n_drivers: int = 0,
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
            n_drivers=n_drivers,
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

        typed_zero = numba_precision(0.0)
        stage_count = tableau.stage_count
        accumulator_length = max(stage_count - 1, 0) * n

        has_driver_function = driver_function is not None
        first_same_as_last = self.first_same_as_last
        multistage = stage_count > 1
        has_error = self.is_adaptive

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
            # stage_rhs: size n, per-thread local memory.
            #   Default behaviour:
            #       - Holds the current stage rhs before scaling by dt.
            #   Reuse:
            #       - When FSAL hits we copy cached rhs here before touching
            #         shared memory, keeping lifetimes separate.
            # error: size n, global memory (adaptive runs only).
            #   Default behaviour:
            #       - Accumulates weighted stage increments during the loop.
            #       - Cleared at loop entry so prior steps cannot leak in.
            # ----------------------------------------------------------- #
            stage_rhs = cuda.local.array(n, numba_precision)

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

            status_code = int32(0)
            # ----------------------------------------------------------- #
            #            Stage 0: may use cached values                   #
            # ----------------------------------------------------------- #
            use_cached_rhs = ((not first_step_flag) and accepted_flag and
                              first_same_as_last)
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

                stage_drivers = proposed_drivers
                if has_driver_function:
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        stage_drivers,
                    )

                observables_function(
                    stage_state,
                    parameters,
                    stage_drivers,
                    proposed_observables,
                    stage_time,
                )

                dxdt_fn(
                    stage_state,
                    parameters,
                    stage_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

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
            return int32(0)

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
