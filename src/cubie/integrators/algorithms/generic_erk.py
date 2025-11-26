"""Generic explicit Runge--Kutta integration step with streamed accumulators.

This module provides the :class:`ERKStep` class, which implements generic
explicit Runge--Kutta methods using configurable Butcher tableaus. The
implementation supports both adaptive and fixed-step variants, automatically
selecting appropriate default step controller settings based on whether the
tableau includes an embedded error estimate.

Key Features
------------
- Configurable tableaus via :class:`ERKTableau`
- Automatic controller defaults selection based on error estimate capability
- Efficient CUDA kernel generation with streamed accumulation
- Support for FSAL (First Same As Last) optimization

Notes
-----
The module defines two sets of default step controller settings:

- :data:`ERK_ADAPTIVE_DEFAULTS`: Used when the tableau has an embedded error
  estimate (e.g., Dormand-Prince). Defaults to PI controller with adaptive
  stepping.
- :data:`ERK_FIXED_DEFAULTS`: Used when the tableau lacks an error estimate
  (e.g., Classical RK4). Defaults to fixed-step controller.

This dynamic selection ensures that users cannot accidentally pair an
errorless tableau with an adaptive controller, which would fail at runtime.
"""

from typing import Callable, Optional

import attrs
from numba import cuda, int16, int32

from cubie._utils import PrecisionDType
from cubie.cuda_simsafe import all_sync, activemask
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


ERK_ADAPTIVE_DEFAULTS = StepControlDefaults(
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
"""Default step controller settings for adaptive ERK tableaus.

This configuration is used when the ERK tableau has an embedded error
estimate (``tableau.has_error_estimate == True``), such as Dormand-Prince
or other embedded RK methods.

The PI controller provides robust adaptive stepping with proportional and
derivative terms to smooth step size adjustments. The deadband prevents
unnecessary step size changes for small variations in the error estimate.

Notes
-----
These defaults are applied automatically when creating an :class:`ERKStep`
with an adaptive tableau. Users can override any of these settings by
explicitly specifying step controller parameters.
"""

ERK_FIXED_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)
"""Default step controller settings for errorless ERK tableaus.

This configuration is used when the ERK tableau lacks an embedded error
estimate (``tableau.has_error_estimate == False``), such as Classical RK4
or Heun's method.

Fixed-step controllers maintain a constant step size throughout the
integration. This is the only valid choice for errorless tableaus since
adaptive stepping requires an error estimate to adjust the step size.

Notes
-----
These defaults are applied automatically when creating an :class:`ERKStep`
with an errorless tableau. Users can override the step size ``dt`` by
explicitly specifying it in the step controller settings.
"""


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
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        tableau: ERKTableau = DEFAULT_ERK_TABLEAU,
        n_drivers: int = 0,
    ) -> None:
        """Initialise the Runge--Kutta step configuration.
        
        This constructor creates an ERK step object and automatically selects
        appropriate default step controller settings based on whether the
        tableau has an embedded error estimate. Tableaus with error estimates
        default to adaptive stepping (PI controller), while errorless tableaus
        default to fixed stepping.

        Parameters
        ----------
        precision
            Floating-point precision for CUDA computations (np.float32 or
            np.float64).
        n
            Number of state variables in the ODE system.
        dxdt_function
            Compiled CUDA device function computing state derivatives. Should
            match signature expected by the integration kernel.
        observables_function
            Optional compiled CUDA device function computing observable
            quantities from the state.
        driver_function
            Optional compiled CUDA device function computing time-varying
            driver inputs.
        get_solver_helper_fn
            Factory function returning solver helper for implicit stages (not
            used in explicit methods but included for interface consistency).
        tableau
            Explicit Runge--Kutta tableau describing the coefficients used by
            the integrator. Defaults to :data:`DEFAULT_ERK_TABLEAU`
            (Dormand-Prince 5(4)).
        n_drivers
            Number of driver variables in the system.
        
        Notes
        -----
        The step controller defaults are selected dynamically:
        
        - If ``tableau.has_error_estimate`` is ``True``:
          Uses :data:`ERK_ADAPTIVE_DEFAULTS` (PI controller)
        - If ``tableau.has_error_estimate`` is ``False``:
          Uses :data:`ERK_FIXED_DEFAULTS` (fixed-step controller)
        
        This automatic selection prevents incompatible configurations where
        an adaptive controller is paired with an errorless tableau. Users
        can still override these defaults by explicitly specifying step
        controller settings when creating a :class:`Solver` or calling
        :func:`solve_ivp`.
        
        Examples
        --------
        Create an ERK step with the default Dormand-Prince tableau:
        
        >>> from cubie.integrators.algorithms.generic_erk import ERKStep
        >>> import numpy as np
        >>> step = ERKStep(precision=np.float32,n=3)
        >>> step.controller_defaults.step_controller["step_controller"]
        'pi'
        
        Create an ERK step with Classical RK4 (errorless):
        
        >>> from cubie.integrators.algorithms.generic_erk_tableaus import (
        ...     CLASSICAL_RK4_TABLEAU
        ... )
        >>> step = ERKStep(precision=np.float32,n=3,tableau=CLASSICAL_RK4_TABLEAU)
        >>> step.controller_defaults.step_controller["step_controller"]
        'fixed'
        """

        config_kwargs = {
            "precision": precision,
            "n": n,
            "n_drivers": n_drivers,
            "dxdt_function": dxdt_function,
            "observables_function": observables_function,
            "driver_function": driver_function,
            "get_solver_helper_fn": get_solver_helper_fn,
            "tableau": tableau,
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
        tableau = config.tableau

        typed_zero = numba_precision(0.0)
        n_arraysize = n
        n = int32(n)
        stage_count = int32(tableau.stage_count)
        accumulator_length = int32(max(stage_count - 1, 0) * n)

        has_driver_function = driver_function is not None
        first_same_as_last = self.first_same_as_last
        multistage = stage_count > 1
        has_error = self.is_adaptive

        stage_rhs_coeffs = tableau.typed_rows(tableau.a, numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        stage_nodes = tableau.typed_vector(tableau.c, numba_precision)

        if has_error:
            embedded_weights = tableau.typed_vector(tableau.b_hat, numba_precision)
            error_weights = tableau.error_weights(numba_precision)
        else:
            embedded_weights = tuple(typed_zero for _ in range(stage_count))
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

        # no cover: start
        @cuda.jit(
            (
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[:, :, ::1],
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
                numba_precision,
                numba_precision,
                int16,
                int16,
                numba_precision[::1],
                numba_precision[::1],
                int32[::1],
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
            counters,
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
            #       - Accumulates error-weighted f(y_jn) during the
            #       loop.
            #       - Cleared at loop entry so prior steps cannot leak in.
            # ----------------------------------------------------------- #
            stage_rhs = cuda.local.array(n_arraysize, numba_precision)

            current_time = time_scalar
            end_time = current_time + dt_scalar

            stage_accumulator = shared[:accumulator_length]
            if multistage:
                stage_cache = stage_accumulator[:n]

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] = typed_zero
                if has_error and accumulates_error:
                    error[idx] = typed_zero

            # ----------------------------------------------------------- #
            #            Stage 0: may use cached values                   #
            # ----------------------------------------------------------- #
            # Only use cache if all threads in warp can - otherwise no gain
            use_cached_rhs = False
            if first_same_as_last and multistage:
                if not first_step_flag:
                    mask = activemask()
                    all_threads_accepted = all_sync(mask, accepted_flag != int16(0))
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

            # b weights can't match a rows for erk, as they would return 0
            # So we include ifs to skip accumulating but do no direct assign.
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
                dt_stage = dt_scalar * stage_nodes[stage_idx]
                stage_time = current_time + dt_stage

                # Convert accumulated gradients sum(f(y_nj) into a state y_j
                for idx in range(n):
                    stage_accumulator[stage_offset + idx] *= dt_scalar
                    stage_accumulator[stage_offset + idx] += state[idx]

                # Rename the slice for clarity
                stage_state = stage_accumulator[stage_offset:stage_offset + n]


                # get rhs for next stage
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
                    proposed_state[idx] = (
                            proposed_state[idx] * dt_scalar + state[idx]
                    )
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
