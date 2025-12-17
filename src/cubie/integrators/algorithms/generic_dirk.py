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
from attrs import validators
import numpy as np
from numba import cuda, int32

from cubie._utils import PrecisionDType, getype_validator
from cubie.BufferSettings import BufferSettings, LocalSizes, SliceIndices
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
from cubie.integrators.matrix_free_solvers.linear_solver import (
    LinearSolverBufferSettings,
    linear_solver_factory,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonBufferSettings,
    newton_krylov_solver_factory,
)


@attrs.define
class DIRKLocalSizes(LocalSizes):
    """Local array sizes for DIRK buffers with nonzero guarantees.

    Attributes
    ----------
    stage_increment : int
        Stage increment buffer size.
    stage_base : int
        Stage base buffer size.
    accumulator : int
        Stage accumulator buffer size.
    solver_scratch : int
        Solver scratch buffer size.
    increment_cache : int
        Increment cache buffer size (for FSAL when solver_scratch local).
    rhs_cache : int
        RHS cache buffer size (for FSAL when solver_scratch local).
    """

    stage_increment: int = attrs.field(validator=getype_validator(int, 0))
    stage_base: int = attrs.field(validator=getype_validator(int, 0))
    accumulator: int = attrs.field(validator=getype_validator(int, 0))
    solver_scratch: int = attrs.field(validator=getype_validator(int, 0))
    increment_cache: int = attrs.field(validator=getype_validator(int, 0))
    rhs_cache: int = attrs.field(validator=getype_validator(int, 0))


@attrs.define
class DIRKSliceIndices(SliceIndices):
    """Slice container for DIRK shared memory buffer layouts.

    Attributes
    ----------
    stage_increment : slice
        Slice covering the stage increment buffer (empty if local).
    stage_base : slice
        Slice covering the stage base buffer (may alias accumulator).
    accumulator : slice
        Slice covering the stage accumulator buffer.
    solver_scratch : slice
        Slice covering the solver scratch buffer (always shared).
    local_end : int
        Offset of the end of algorithm-managed shared memory.
    """

    stage_increment: slice = attrs.field()
    stage_base: slice = attrs.field()
    accumulator: slice = attrs.field()
    solver_scratch: slice = attrs.field()
    local_end: int = attrs.field()


@attrs.define
class DIRKBufferSettings(BufferSettings):
    """Configuration for DIRK step buffer sizes and memory locations.

    Controls memory locations for stage_increment, stage_base, and accumulator
    buffers used during DIRK integration steps. solver_scratch is always
    passed from shared memory.

    Attributes
    ----------
    n : int
        Number of state variables.
    stage_count : int
        Number of RK stages.
    stage_increment_location : str
        Memory location for stage increment buffer: 'local' or 'shared'.
    stage_base_location : str
        Memory location for stage base buffer: 'local' or 'shared'.
    accumulator_location : str
        Memory location for stage accumulator buffer: 'local' or 'shared'.
    newton_buffer_settings : NewtonBufferSettings
        Buffer settings for the Newton solver (for memory accounting).
    """

    n: int = attrs.field(validator=getype_validator(int, 1))
    stage_count: int = attrs.field(validator=getype_validator(int, 1))
    stage_increment_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    stage_base_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    accumulator_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    newton_buffer_settings: Optional[NewtonBufferSettings] = attrs.field(
        default=None,
    )

    def __attrs_post_init__(self):
        """Set default newton_buffer_settings if not provided."""
        if self.newton_buffer_settings is None:
            linear_settings = LinearSolverBufferSettings(n=self.n)
            newton_settings = NewtonBufferSettings(
                n=self.n,
                linear_solver_buffer_settings=linear_settings,
            )
            object.__setattr__(self, 'newton_buffer_settings', newton_settings)

    @property
    def use_shared_stage_increment(self) -> bool:
        """Return True if stage_increment buffer uses shared memory."""
        return self.stage_increment_location == 'shared'

    @property
    def use_shared_stage_base(self) -> bool:
        """Return True if stage_base buffer uses shared memory."""
        return self.stage_base_location == 'shared'

    @property
    def use_shared_accumulator(self) -> bool:
        """Return True if accumulator buffer uses shared memory."""
        return self.accumulator_location == 'shared'

    @property
    def accumulator_length(self) -> int:
        """Return the length of the stage accumulator buffer."""
        return max(self.stage_count - 1, 0) * self.n

    @property
    def solver_scratch_elements(self) -> int:
        """Return the number of solver scratch elements.

        Returns newton_buffer_settings.shared_memory_elements.
        """
        return self.newton_buffer_settings.shared_memory_elements

    @property
    def solver_scratch_has_rhs_space(self) -> bool:
        """Return True if solver_scratch has space for stage_rhs.

        stage_rhs requires n elements. When available in shared memory,
        rhs_cache can alias it for stateful FSAL caching.
        """
        return self.solver_scratch_elements >= self.n

    @property
    def solver_scratch_has_increment_space(self) -> bool:
        """Return True if solver_scratch has space for increment_cache.

        increment_cache requires n elements beyond stage_rhs.
        When solver_scratch >= 2n, increment_cache can be sliced from it.
        """
        return self.solver_scratch_elements >= 2 * self.n

    @property
    def persistent_local_elements(self) -> int:
        """Return persistent local elements for FSAL caches.

        - When solver_scratch >= 2n: no persistent local needed (both caches
          fit in solver_scratch).
        - When solver_scratch >= n: increment_cache needs n persistent local
          (rhs_cache aliases stage_rhs in shared solver_scratch).
        - When solver_scratch < n: both caches need persistent local (2n).
        """
        if self.solver_scratch_has_increment_space:
            return 0  # Both caches fit in solver_scratch
        elif self.solver_scratch_has_rhs_space:
            return self.n  # Only increment_cache needs persistent local
        else:
            return 2 * self.n  # Both caches need persistent local

    @property
    def multistage(self) -> bool:
        """Return True if method has multiple stages."""
        return self.stage_count > 1

    @property
    def stage_base_aliases_accumulator(self) -> bool:
        """Return True if stage_base can alias first slice of accumulator.

        Only valid when multistage, BOTH stage_base AND accumulator are
        in shared memory.
        """
        return (self.multistage
                and self.use_shared_accumulator
                and self.use_shared_stage_base)

    @property
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required.

        Includes accumulator, solver_scratch, and stage_increment if shared.
        stage_base is counted separately when it cannot alias accumulator.
        """
        total = 0
        if self.use_shared_accumulator:
            total += self.accumulator_length
        total += self.solver_scratch_elements  # Always included
        if self.use_shared_stage_increment:
            total += self.n
        # stage_base needs separate allocation when:
        # - shared AND (single-stage OR accumulator is local)
        if self.use_shared_stage_base and not self.stage_base_aliases_accumulator:
            total += self.n
        return total

    @property
    def local_memory_elements(self) -> int:
        """Return total local memory elements required.

        Includes buffers configured with location='local'.
        solver_scratch is not included as it is always shared from parent.
        """
        total = 0
        if not self.use_shared_accumulator:
            total += self.accumulator_length
        if not self.use_shared_stage_increment:
            total += self.n
        # stage_base needs local storage when local
        if not self.use_shared_stage_base:
            total += self.n
        return total

    @property
    def local_sizes(self) -> DIRKLocalSizes:
        """Return DIRKLocalSizes instance with buffer sizes.

        The returned object provides nonzero sizes suitable for
        cuda.local.array allocation.
        """
        # stage_base needs local allocation when not aliasing accumulator
        # and configured for local memory
        if self.use_shared_stage_base:
            stage_base_size = 0  # Will use shared memory
        elif self.multistage and not self.use_shared_accumulator:
            # Can alias local accumulator in device function
            stage_base_size = int(self.n)
        else:
            stage_base_size = int(self.n)
        return DIRKLocalSizes(
            stage_increment=int(self.n),
            stage_base=stage_base_size,
            accumulator=int(self.accumulator_length),
            solver_scratch=int(self.solver_scratch_elements),
            increment_cache=0,
            rhs_cache=0,
        )

    @property
    def shared_indices(self) -> DIRKSliceIndices:
        """Return DIRKSliceIndices instance with shared memory layout.

        The returned object contains slices for each buffer's region
        in shared memory. Local buffers receive empty slices.
        """
        ptr = 0

        if self.use_shared_accumulator:
            accumulator_slice = slice(ptr, ptr + self.accumulator_length)
            ptr += self.accumulator_length
        else:
            accumulator_slice = slice(0, 0)

        # solver_scratch always included in shared memory layout
        solver_scratch_slice = slice(ptr, ptr + self.solver_scratch_elements)
        ptr += self.solver_scratch_elements

        if self.use_shared_stage_increment:
            stage_increment_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            stage_increment_slice = slice(0, 0)

        # stage_base: alias accumulator, separate shared, or local
        if self.stage_base_aliases_accumulator:
            # Alias first n elements of accumulator
            stage_base_slice = slice(
                accumulator_slice.start,
                accumulator_slice.start + self.n
            )
        elif self.use_shared_stage_base:
            # Separate shared allocation
            stage_base_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            # Local allocation
            stage_base_slice = slice(0, 0)

        return DIRKSliceIndices(
            stage_increment=stage_increment_slice,
            stage_base=stage_base_slice,
            accumulator=accumulator_slice,
            solver_scratch=solver_scratch_slice,
            local_end=ptr,
        )


# Buffer location parameters for DIRK algorithms
ALL_DIRK_BUFFER_LOCATION_PARAMETERS = {
    "stage_increment_location",
    "stage_base_location",
    "accumulator_location",
}


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
    buffer_settings: Optional[DIRKBufferSettings] = attrs.field(
        default=None,
        validator=validators.optional(
            validators.instance_of(DIRKBufferSettings)
        ),
    )

    @property
    def newton_buffer_settings(self) -> NewtonBufferSettings:
        """Return newton_buffer_settings from buffer_settings."""
        return self.buffer_settings.newton_buffer_settings

    @property
    def linear_solver_buffer_settings(self) -> LinearSolverBufferSettings:
        """Return linear_solver_buffer_settings from newton_buffer_settings."""
        return self.buffer_settings.newton_buffer_settings.linear_solver_buffer_settings


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
        max_linear_iters: int = 10,
        linear_correction_type: str = "minimal_residual",
        newton_tolerance: float = 1e-6,
        max_newton_iters: int = 10,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
        tableau: DIRKTableau = DEFAULT_DIRK_TABLEAU,
        n_drivers: int = 0,
        stage_increment_location: Optional[str] = None,
        stage_base_location: Optional[str] = None,
        accumulator_location: Optional[str] = None,
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
            Order of the finite-difference Jacobian approximation used in the
            preconditioner.
        krylov_tolerance
            Convergence tolerance for the Krylov linear solver.
        max_linear_iters
            Maximum iterations allowed for the Krylov solver.
        linear_correction_type
            Type of Krylov correction ("minimal_residual" or other).
        newton_tolerance
            Convergence tolerance for Newton iterations.
        max_newton_iters
            Maximum Newton iterations per implicit stage.
        newton_damping
            Damping factor for Newton step size.
        newton_max_backtracks
            Maximum backtracking steps in Newton's method.
        tableau
            DIRK tableau describing the coefficients. Defaults to
            :data:`DEFAULT_DIRK_TABLEAU`.
        n_drivers
            Number of driver variables in the system.
        
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

        # Create solver buffer settings for accurate memory accounting.
        # DIRK solves each stage independently, so linear solver uses n.
        #TODO: Pass buffer locations through from top level
        linear_buffer_settings = LinearSolverBufferSettings(n=n)
        newton_buffer_settings = NewtonBufferSettings(
            n=n,
            linear_solver_buffer_settings=linear_buffer_settings,
        )

        # Create buffer_settings - only pass locations if explicitly provided
        buffer_kwargs = {
            'n': n,
            'stage_count': tableau.stage_count,
            'newton_buffer_settings': newton_buffer_settings,
        }
        if stage_increment_location is not None:
            buffer_kwargs['stage_increment_location'] = stage_increment_location
        if stage_base_location is not None:
            buffer_kwargs['stage_base_location'] = stage_base_location
        if accumulator_location is not None:
            buffer_kwargs['accumulator_location'] = accumulator_location
        buffer_settings = DIRKBufferSettings(**buffer_kwargs)
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
            "buffer_settings": buffer_settings,
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
            "linear_operator",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        krylov_tolerance = config.krylov_tolerance
        max_linear_iters = config.max_linear_iters
        correction_type = config.linear_correction_type

        newton_buffer_settings = config.newton_buffer_settings
        linear_buffer_settings = config.linear_solver_buffer_settings

        linear_solver = linear_solver_factory(
            operator,
            n=n,
            precision=precision,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
            buffer_settings=linear_buffer_settings,
        )

        newton_tolerance = config.newton_tolerance
        max_newton_iters = config.max_newton_iters
        newton_damping = config.newton_damping
        newton_max_backtracks = config.newton_max_backtracks

        nonlinear_solver = newton_krylov_solver_factory(
            residual_function=residual,
            linear_solver=linear_solver,
            n=n,
            tolerance=newton_tolerance,
            max_iters=max_newton_iters,
            damping=newton_damping,
            max_backtracks=newton_max_backtracks,
            precision=precision,
            buffer_settings=newton_buffer_settings,
        )

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
        n_arraysize = int(n)
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

        # Buffer settings from compile_settings for selective shared/local
        buffer_settings = config.buffer_settings

        # Unpack boolean flags as compile-time constants
        stage_increment_shared = buffer_settings.use_shared_stage_increment
        stage_base_shared = buffer_settings.use_shared_stage_base
        accumulator_shared = buffer_settings.use_shared_accumulator
        stage_base_aliases = buffer_settings.stage_base_aliases_accumulator
        has_rhs_in_scratch = buffer_settings.solver_scratch_has_rhs_space
        has_increment_in_scratch = buffer_settings.solver_scratch_has_increment_space

        # Unpack slice indices for shared memory layout
        shared_indices = buffer_settings.shared_indices
        stage_increment_slice = shared_indices.stage_increment
        stage_base_slice = shared_indices.stage_base
        accumulator_slice = shared_indices.accumulator
        solver_scratch_slice = shared_indices.solver_scratch

        # Unpack local sizes for local array allocation
        local_sizes = buffer_settings.local_sizes
        stage_increment_local_size = local_sizes.nonzero('stage_increment')
        stage_base_local_size = local_sizes.nonzero('stage_base')
        accumulator_local_size = local_sizes.nonzero('accumulator')

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
            if stage_increment_shared:
                stage_increment = shared[stage_increment_slice]
            else:
                stage_increment = cuda.local.array(stage_increment_local_size,
                                                   precision)
                for _i in range(int32(stage_increment_local_size)):
                    stage_increment[_i] = numba_precision(0.0)

            if accumulator_shared:
                stage_accumulator = shared[accumulator_slice]
            else:
                stage_accumulator = cuda.local.array(accumulator_local_size,
                                                     precision)
                for _i in range(int32(accumulator_local_size)):
                    stage_accumulator[_i] = numba_precision(0.0)

            # solver_scratch always from shared memory
            solver_scratch = shared[solver_scratch_slice]

            # Check aliasing eligibility based on BOTH parent and child locations
            if stage_base_aliases:
                # Both accumulator and stage_base are shared; alias first slice
                stage_base = stage_accumulator[:n]
            elif stage_base_shared:
                # Separate shared allocation (accumulator local or single-stage)
                stage_base = shared[stage_base_slice]
            else:
                # Separate local allocation
                stage_base = cuda.local.array(stage_base_local_size, precision)
                for _i in range(int32(stage_base_local_size)):
                    stage_base[_i] = numba_precision(0.0)

            # --------------------------------------------------------------- #

            current_time = time_scalar
            end_time = current_time + dt_scalar

            # stage_rhs is used during Newton iterations and overwritten.
            # When solver_scratch has at least n elements, slice from it.
            # This also enables rhs_cache to alias stage_rhs for FSAL.
            if has_rhs_in_scratch:
                stage_rhs = solver_scratch[:n]
            else:
                stage_rhs = cuda.local.array(n_arraysize, precision)

            # increment_cache and rhs_cache persist between steps for FSAL.
            # Allocation depends on solver_scratch size:
            # - >= 2n: both caches in solver_scratch, rhs_cache aliases stage_rhs
            # - >= n: increment_cache in persistent_local, rhs_cache aliases
            #         stage_rhs (which is in shared solver_scratch)
            # - < n: both caches in persistent_local
            if has_increment_in_scratch:
                increment_cache = solver_scratch[n:int32(2)*n]
                rhs_cache = solver_scratch[:n]  # Aliases stage_rhs
            elif has_rhs_in_scratch:
                increment_cache = persistent_local[:n]
                rhs_cache = solver_scratch[:n]  # Aliases stage_rhs
            else:
                increment_cache = persistent_local[:n]
                rhs_cache = persistent_local[n:int32(2)*n]

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
                    solver_status = nonlinear_solver(
                        stage_increment,
                        parameters,
                        proposed_drivers,
                        stage_time,
                        dt_scalar,
                        diagonal_coeffs[0],
                        stage_base,
                        solver_scratch,
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
                        solver_scratch,
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

            # Cache increment and RHS for FSAL optimization
            for idx in range(n):
                increment_cache[idx] = stage_increment[idx]
                # rhs_cache aliases stage_rhs when has_rhs_in_scratch=True,
                # so no explicit copy needed. Otherwise, copy to persistent_local.
                if not has_rhs_in_scratch:
                    rhs_cache[idx] = stage_rhs[idx]

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
        return self.compile_settings.buffer_settings.shared_memory_elements

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""
        return self.compile_settings.buffer_settings.local_memory_elements

    @property
    def persistent_local_required(self) -> int:
        """Return the number of persistent local entries required.

        Returns n for increment_cache when solver_scratch uses local memory.
        When solver_scratch is shared, increment_cache aliases it and no
        persistent local is needed.
        """
        buffer_settings = self.compile_settings.buffer_settings
        return buffer_settings.persistent_local_elements

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
