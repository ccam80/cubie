"""Explicit Runge--Kutta integration step with streamed accumulators."""

from typing import Callable, Optional

import attrs
from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.ode_explicitstep import (
    ExplicitStepConfig,
    ODEExplicitStep,
)
from cubie.integrators.algorithms.ode_implicitstep import ButcherTableau



@attrs.define(frozen=True)
class ERKTableau(ButcherTableau):
    """Coefficient tableau describing an explicit Runge--Kutta scheme."""
    pass

DORMAND_PRINCE_54_TABLEAU = ERKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            44.0 / 45.0,
            -56.0 / 15.0,
            32.0 / 9.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
            0.0,
            0.0,
        ),
        (
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
            0.0,
        ),
    ),
    b=(
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ),
    d=(
        35.0 / 384.0 - 5179.0 / 57600.0,
        0.0,
        500.0 / 1113.0 - 7571.0 / 16695.0,
        125.0 / 192.0 - 393.0 / 640.0,
        -2187.0 / 6784.0 + 92097.0 / 339200.0,
        11.0 / 84.0 - 187.0 / 2100.0,
        0.0 - 1.0 / 40.0,
    ),
    c=(
        0.0,
        1.0 / 5.0,
        3.0 / 10.0,
        4.0 / 5.0,
        8.0 / 9.0,
        1.0,
        1.0,
    ),
    order=4
)


ERK_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "pi",
        "order": 5,
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

    tableau: ERKTableau = attrs.field(default=DORMAND_PRINCE_54_TABLEAU)

    @property
    def settings_dict(self) -> dict:
        """Return configuration values as a dictionary."""

        settings = super().settings_dict
        settings.update({"tableau": self.tableau})
        return settings


class ERKStep(ODEExplicitStep):
    """Dormand--Prince Runge--Kutta step with an embedded error estimate."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dt: Optional[float],
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        tableau: ERKTableau = DORMAND_PRINCE_54_TABLEAU,
    ) -> None:
        """Initialise the Runge--Kutta step configuration."""

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
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the explicit Runge--Kutta device step."""

        config = self.compile_settings
        tableau = config.tableau

        stage_count = tableau.stage_count
        has_driver_function = driver_function is not None

        stage_rhs_coeffs = self.tableau.typed_rows(tableau.a, numba_precision)
        solution_weights = tuple(
            numba_precision(value) for value in tableau.b
        )
        error_weights = tuple(numba_precision(value) for value in tableau.d)
        stage_time_fractions = tuple(
            numba_precision(value) for value in tableau.c
        )
        typed_zero = numba_precision(0.0)
        accumulator_length = max(stage_count - 1, 0) * n
        has_error = self.is_adaptive

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
            shared,
            persistent_local,
        ):
            stage_rhs = cuda.local.array(n, numba_precision)
            stage_state = cuda.local.array(n, numba_precision)
            stage_increment = cuda.local.array(n, numba_precision)

            dt_value = dt_scalar
            current_time = time_scalar
            end_time = current_time + dt_value

            stage_accumulator = shared[:accumulator_length]

            for idx in range(accumulator_length):
                stage_accumulator[idx] = typed_zero

            for idx in range(n):
                proposed_state[idx] = state[idx]
                error[idx*has_error] = typed_zero

            status_code = int32(0)
            # ----------------------------------------------------------- #
            #            Stage 0: operates out of supplied buffers          #
            # ----------------------------------------------------------- #

            dxdt_fn(
                state,
                parameters,
                drivers_buffer,
                observables,
                stage_rhs,
                current_time,
            )

            for idx in range(n):
                stage_increment[idx] = dt_value * stage_rhs[idx]
                proposed_state[idx] += (
                    solution_weights[0] * stage_increment[idx]
                )
                error[idx*has_error] += (
                    error_weights[0] * stage_increment[idx]
                )

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
                        contribution = state_coeff * stage_increment[idx]
                        stage_accumulator[base + idx] += contribution


                stage_offset = (stage_idx - 1) * n
                stage_time = (
                    current_time + dt_value * stage_time_fractions[stage_idx]
                )

                for idx in range(n):
                    stage_state[idx] = state[idx] + stage_accumulator[
                        stage_offset + idx
                    ]

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
                    stage_increment[idx] = dt_value * stage_rhs[idx]
                    proposed_state[idx] += (
                        solution_weights[stage_idx] * stage_increment[idx]
                    )
                    error[idx*has_error] += (
                        error_weights[stage_idx] * stage_increment[idx]
                    )
            # ----------------------------------------------------------- #

            final_time = end_time
            if has_driver_function:
                driver_function(
                    final_time,
                    driver_coeffs,
                    proposed_drivers,
                )

            for idx in range(n):
                proposed_state[idx] = state[idx] + proposed_state[idx]

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                final_time,
            )
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
        return 3 * self.compile_settings.n

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

    @property
    def tableau(self) -> ERKTableau:
        """Return the tableau used by the integrator."""

        return self.compile_settings.tableau