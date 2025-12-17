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
from attrs import validators
from numba import cuda, int32, int32

from cubie._utils import PrecisionDType, getype_validator
from cubie.BufferSettings import BufferSettings, LocalSizes, SliceIndices
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


@attrs.define
class ERKLocalSizes(LocalSizes):
    """Local array sizes for ERK buffers with nonzero guarantees.

    Attributes
    ----------
    stage_rhs : int
        Stage RHS buffer size.
    stage_accumulator : int
        Stage accumulator buffer size.
    stage_cache : int
        Stage cache buffer size (for FSAL optimization).
    """

    stage_rhs: int = attrs.field(validator=getype_validator(int, 0))
    stage_accumulator: int = attrs.field(validator=getype_validator(int, 0))
    stage_cache: int = attrs.field(validator=getype_validator(int, 0))


@attrs.define
class ERKSliceIndices(SliceIndices):
    """Slice container for ERK shared memory buffer layouts.

    Attributes
    ----------
    stage_rhs : slice
        Slice covering the stage RHS buffer (empty if local).
    stage_accumulator : slice
        Slice covering the stage accumulator buffer (empty if local).
    stage_cache : slice
        Slice covering the stage cache buffer (aliases rhs or accumulator).
    local_end : int
        Offset of the end of algorithm-managed shared memory.
    """

    stage_rhs: slice = attrs.field()
    stage_accumulator: slice = attrs.field()
    stage_cache: slice = attrs.field()
    local_end: int = attrs.field()


@attrs.define
class ERKBufferSettings(BufferSettings):
    """Configuration for ERK step buffer sizes and memory locations.

    Controls whether stage_rhs and stage_accumulator buffers use shared
    or local memory. Also manages stage_cache aliasing logic for FSAL
    optimization.

    Attributes
    ----------
    n : int
        Number of state variables.
    stage_count : int
        Number of RK stages.
    stage_rhs_location : str
        Memory location for stage RHS buffer: 'local' or 'shared'.
    stage_accumulator_location : str
        Memory location for stage accumulator buffer: 'local' or 'shared'.
    """

    n: int = attrs.field(validator=getype_validator(int, 1))
    stage_count: int = attrs.field(validator=getype_validator(int, 1))
    stage_rhs_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    stage_accumulator_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )

    @property
    def use_shared_stage_rhs(self) -> bool:
        """Return True if stage_rhs buffer uses shared memory."""
        return self.stage_rhs_location == 'shared'

    @property
    def use_shared_stage_accumulator(self) -> bool:
        """Return True if stage_accumulator buffer uses shared memory."""
        return self.stage_accumulator_location == 'shared'

    @property
    def use_shared_stage_cache(self) -> bool:
        """Return True if stage_cache should use shared memory.

        stage_cache is shared if either stage_rhs or stage_accumulator
        is shared (it aliases onto one of them).
        """
        return self.use_shared_stage_rhs or self.use_shared_stage_accumulator

    @property
    def stage_cache_aliases_rhs(self) -> bool:
        """Return True if stage_cache aliases stage_rhs.

        stage_cache aliases stage_rhs when stage_rhs is in shared memory.
        """
        return self.use_shared_stage_rhs

    @property
    def stage_cache_aliases_accumulator(self) -> bool:
        """Return True if stage_cache aliases stage_accumulator.

        stage_cache aliases accumulator when stage_rhs is local but
        accumulator is shared.
        """
        return (not self.use_shared_stage_rhs
                and self.use_shared_stage_accumulator)

    @property
    def accumulator_length(self) -> int:
        """Return the length of the stage accumulator buffer."""
        return max(self.stage_count - 1, 0) * self.n

    @property
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required.

        Includes stage_rhs (n) if shared, and accumulator if shared.
        """
        total = 0
        if self.use_shared_stage_rhs:
            total += self.n
        if self.use_shared_stage_accumulator:
            total += self.accumulator_length
        return total

    @property
    def local_memory_elements(self) -> int:
        """Return total local memory elements required.

        Includes stage_rhs (n) if local, accumulator if local,
        plus persistent_local for stage_cache if not aliased.
        """
        total = 0
        if not self.use_shared_stage_rhs:
            total += self.n
        if not self.use_shared_stage_accumulator:
            total += self.accumulator_length
        # stage_cache needs persistent local if neither is shared
        if not self.use_shared_stage_cache:
            total += self.n
        return total

    @property
    def persistent_local_elements(self) -> int:
        """Return persistent local elements for stage_cache.

        Returns n if stage_cache cannot alias onto shared buffers.
        """
        if self.use_shared_stage_cache:
            return 0
        return self.n

    @property
    def local_sizes(self) -> ERKLocalSizes:
        """Return ERKLocalSizes instance with buffer sizes.

        The returned object provides nonzero sizes suitable for
        cuda.local.array allocation.
        """
        return ERKLocalSizes(
            stage_rhs=self.n,
            stage_accumulator=self.accumulator_length,
            stage_cache=self.n if not self.use_shared_stage_cache else 0,
        )

    @property
    def shared_indices(self) -> ERKSliceIndices:
        """Return ERKSliceIndices instance with shared memory layout.

        The returned object contains slices for each buffer's region
        in shared memory. Local buffers receive empty slices.
        """
        ptr = 0

        if self.use_shared_stage_rhs:
            stage_rhs_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            stage_rhs_slice = slice(0, 0)

        if self.use_shared_stage_accumulator:
            stage_accumulator_slice = slice(ptr, ptr + self.accumulator_length)
            ptr += self.accumulator_length
        else:
            stage_accumulator_slice = slice(0, 0)

        # stage_cache aliases rhs or accumulator when either is in shared
        if self.stage_cache_aliases_rhs:
            stage_cache_slice = stage_rhs_slice
        elif self.stage_cache_aliases_accumulator:
            stage_cache_slice = slice(
                stage_accumulator_slice.start,
                stage_accumulator_slice.start + self.n
            )
        else:
            stage_cache_slice = slice(0, 0)

        return ERKSliceIndices(
            stage_rhs=stage_rhs_slice,
            stage_accumulator=stage_accumulator_slice,
            stage_cache=stage_cache_slice,
            local_end=ptr,
        )


# Buffer location parameters for ERK algorithms
ALL_ERK_BUFFER_LOCATION_PARAMETERS = {
    "stage_rhs_location",
    "stage_accumulator_location",
}


ERK_ADAPTIVE_DEFAULTS = StepControlDefaults(
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
        "safety": 0.9
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
        stage_rhs_location: Optional[str] = None,
        stage_accumulator_location: Optional[str] = None,
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

        # Create buffer_settings - only pass locations if explicitly provided
        buffer_kwargs = {
            'n': n,
            'stage_count': tableau.stage_count,
        }
        if stage_rhs_location is not None:
            buffer_kwargs['stage_rhs_location'] = stage_rhs_location
        if stage_accumulator_location is not None:
            buffer_kwargs['stage_accumulator_location'] = stage_accumulator_location
        buffer_settings = ERKBufferSettings(**buffer_kwargs)
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
            #       - Accumulates error-weighted f(y_jn) during the
            #       loop.
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

            current_time = time_scalar
            end_time = current_time + dt_scalar

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
                    all_threads_accepted = all_sync(
                            mask,
                            accepted_flag != int32(0))
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
                        error[idx] = error[idx] + error_weights[0] * increment

            for idx in range(accumulator_length):
                stage_accumulator[idx] = typed_zero

            # ----------------------------------------------------------- #
            #            Stages 1-s: refresh observables and drivers       #
            # ----------------------------------------------------------- #
            for prev_idx in range(stages_except_first):
                stage_offset = prev_idx * n
                stage_idx = prev_idx + int32(1)
                matrix_col = stage_rhs_coeffs[prev_idx]

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

                # get rhs for next stage
                stage_drivers = proposed_drivers
                if has_driver_function:
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        stage_drivers,
                    )

                observables_function(
                    stage_accumulator[stage_offset : stage_offset + n],
                    parameters,
                    stage_drivers,
                    proposed_observables,
                    stage_time,
                )

                dxdt_fn(
                    stage_accumulator[stage_offset : stage_offset + n],
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

                    if has_error:
                        if accumulates_error:
                            increment = stage_rhs[idx]
                            error[idx] += error_weight * increment

            if b_row is not None:
                for idx in range(n):
                    proposed_state[idx] = stage_accumulator[
                        (b_row - 1) * n + idx
                    ]
            if b_hat_row is not None:
                for idx in range(n):
                    error[idx] = stage_accumulator[(b_hat_row - 1) * n + idx]
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
                    # Or form error from difference if captured from a-row
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
