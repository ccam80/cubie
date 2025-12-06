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
from attrs import validators
import numpy as np
from numba import cuda, int16, int32

from cubie._utils import PrecisionDType, getype_validator
from cubie.BufferSettings import BufferSettings, LocalSizes, SliceIndices
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
from cubie.integrators.matrix_free_solvers import (
    linear_solver_factory,
    newton_krylov_solver_factory,
)


@attrs.define
class FIRKLocalSizes(LocalSizes):
    """Local array sizes for FIRK buffers with nonzero guarantees.

    Attributes
    ----------
    solver_scratch : int
        Solver scratch buffer size.
    stage_increment : int
        Stage increment buffer size.
    stage_driver_stack : int
        Stage driver stack buffer size.
    stage_state : int
        Stage state buffer size.
    """

    solver_scratch: int = attrs.field(validator=getype_validator(int, 0))
    stage_increment: int = attrs.field(validator=getype_validator(int, 0))
    stage_driver_stack: int = attrs.field(validator=getype_validator(int, 0))
    stage_state: int = attrs.field(validator=getype_validator(int, 0))


@attrs.define
class FIRKSliceIndices(SliceIndices):
    """Slice container for FIRK shared memory buffer layouts.

    Attributes
    ----------
    solver_scratch : slice
        Slice covering the solver scratch buffer (empty if local).
    stage_increment : slice
        Slice covering the stage increment buffer.
    stage_driver_stack : slice
        Slice covering the stage driver stack buffer.
    stage_state : slice
        Slice covering the stage state buffer.
    local_end : int
        Offset of the end of algorithm-managed shared memory.
    """

    solver_scratch: slice = attrs.field()
    stage_increment: slice = attrs.field()
    stage_driver_stack: slice = attrs.field()
    stage_state: slice = attrs.field()
    local_end: int = attrs.field()


@attrs.define
class FIRKBufferSettings(BufferSettings):
    """Configuration for FIRK step buffer sizes and memory locations.

    Controls memory locations for solver_scratch, stage_increment,
    stage_driver_stack, and stage_state buffers used during FIRK
    integration steps.

    Attributes
    ----------
    n : int
        Number of state variables.
    stage_count : int
        Number of RK stages.
    n_drivers : int
        Number of driver variables.
    solver_scratch_location : str
        Memory location for Newton solver scratch: 'local' or 'shared'.
    stage_increment_location : str
        Memory location for stage increment buffer: 'local' or 'shared'.
    stage_driver_stack_location : str
        Memory location for stage driver stack: 'local' or 'shared'.
    stage_state_location : str
        Memory location for stage state buffer: 'local' or 'shared'.
    """

    n: int = attrs.field(validator=getype_validator(int, 1))
    stage_count: int = attrs.field(validator=getype_validator(int, 1))
    n_drivers: int = attrs.field(default=0, validator=getype_validator(int, 0))
    solver_scratch_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    stage_increment_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    stage_driver_stack_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    stage_state_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )

    @property
    def use_shared_solver_scratch(self) -> bool:
        """Return True if solver_scratch buffer uses shared memory."""
        return self.solver_scratch_location == 'shared'

    @property
    def use_shared_stage_increment(self) -> bool:
        """Return True if stage_increment buffer uses shared memory."""
        return self.stage_increment_location == 'shared'

    @property
    def use_shared_stage_driver_stack(self) -> bool:
        """Return True if stage_driver_stack buffer uses shared memory."""
        return self.stage_driver_stack_location == 'shared'

    @property
    def use_shared_stage_state(self) -> bool:
        """Return True if stage_state buffer uses shared memory."""
        return self.stage_state_location == 'shared'

    @property
    def all_stages_n(self) -> int:
        """Return the flattened dimension covering all stage increments."""
        return self.stage_count * self.n

    @property
    def solver_scratch_elements(self) -> int:
        """Return solver scratch elements (2 * all_stages_n)."""
        return 2 * self.all_stages_n

    @property
    def stage_driver_stack_elements(self) -> int:
        """Return stage driver stack elements (stage_count * n_drivers)."""
        return self.stage_count * self.n_drivers

    @property
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required.

        Includes solver_scratch, stage_increment, and stage_driver_stack
        if configured for shared memory.
        """
        total = 0
        if self.use_shared_solver_scratch:
            total += self.solver_scratch_elements
        if self.use_shared_stage_increment:
            total += self.all_stages_n
        if self.use_shared_stage_driver_stack:
            total += self.stage_driver_stack_elements
        if self.use_shared_stage_state:
            total += self.n
        return total

    @property
    def local_memory_elements(self) -> int:
        """Return total local memory elements required.

        Includes buffers configured with location='local'.
        """
        total = 0
        if not self.use_shared_solver_scratch:
            total += self.solver_scratch_elements
        if not self.use_shared_stage_increment:
            total += self.all_stages_n
        if not self.use_shared_stage_driver_stack:
            total += self.stage_driver_stack_elements
        if not self.use_shared_stage_state:
            total += self.n
        return total

    @property
    def local_sizes(self) -> FIRKLocalSizes:
        """Return FIRKLocalSizes instance with buffer sizes.

        The returned object provides nonzero sizes suitable for
        cuda.local.array allocation.
        """
        return FIRKLocalSizes(
            solver_scratch=self.solver_scratch_elements,
            stage_increment=self.all_stages_n,
            stage_driver_stack=self.stage_driver_stack_elements,
            stage_state=self.n,
        )

    @property
    def shared_indices(self) -> FIRKSliceIndices:
        """Return FIRKSliceIndices instance with shared memory layout.

        The returned object contains slices for each buffer's region
        in shared memory. Local buffers receive empty slices.
        """
        ptr = 0

        if self.use_shared_solver_scratch:
            solver_scratch_slice = slice(ptr, ptr + self.solver_scratch_elements)
            ptr += self.solver_scratch_elements
        else:
            solver_scratch_slice = slice(0, 0)

        if self.use_shared_stage_increment:
            stage_increment_slice = slice(ptr, ptr + self.all_stages_n)
            ptr += self.all_stages_n
        else:
            stage_increment_slice = slice(0, 0)

        if self.use_shared_stage_driver_stack:
            stage_driver_stack_slice = slice(
                ptr, ptr + self.stage_driver_stack_elements
            )
            ptr += self.stage_driver_stack_elements
        else:
            stage_driver_stack_slice = slice(0, 0)

        if self.use_shared_stage_state:
            stage_state_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            stage_state_slice = slice(0, 0)

        return FIRKSliceIndices(
            solver_scratch=solver_scratch_slice,
            stage_increment=stage_increment_slice,
            stage_driver_stack=stage_driver_stack_slice,
            stage_state=stage_state_slice,
            local_end=ptr,
        )


# Buffer location parameters for FIRK algorithms
ALL_FIRK_BUFFER_LOCATION_PARAMETERS = {
    "solver_scratch_location",
    "stage_increment_location",
    "stage_driver_stack_location",
    "stage_state_location",
}


FIRK_ADAPTIVE_DEFAULTS = StepControlDefaults(
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
    buffer_settings: Optional[FIRKBufferSettings] = attrs.field(
        default=None,
        validator=validators.optional(
            validators.instance_of(FIRKBufferSettings)
        ),
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
        preconditioner_order: int = 2,
        krylov_tolerance: float = 1e-6,
        max_linear_iters: int = 200,
        linear_correction_type: str = "minimal_residual",
        newton_tolerance: float = 1e-6,
        max_newton_iters: int = 100,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
        tableau: FIRKTableau = DEFAULT_FIRK_TABLEAU,
        n_drivers: int = 0,
        solver_scratch_location: Optional[str] = None,
        stage_increment_location: Optional[str] = None,
        stage_driver_stack_location: Optional[str] = None,
        stage_state_location: Optional[str] = None,
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
            Maximum Newton iterations per implicit stage solve.
        newton_damping
            Damping factor for Newton step size.
        newton_max_backtracks
            Maximum backtracking steps in Newton's method.
        tableau
            FIRK tableau describing the coefficients. Defaults to
            :data:`DEFAULT_FIRK_TABLEAU`.
        n_drivers
            Number of driver variables in the system.
        
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
        # Create buffer_settings - only pass locations if explicitly provided
        buffer_kwargs = {
            'n': n,
            'stage_count': tableau.stage_count,
            'n_drivers': n_drivers,
        }
        if solver_scratch_location is not None:
            buffer_kwargs['solver_scratch_location'] = solver_scratch_location
        if stage_increment_location is not None:
            buffer_kwargs['stage_increment_location'] = stage_increment_location
        if stage_driver_stack_location is not None:
            buffer_kwargs['stage_driver_stack_location'] = stage_driver_stack_location
        if stage_state_location is not None:
            buffer_kwargs['stage_state_location'] = stage_state_location
        buffer_settings = FIRKBufferSettings(**buffer_kwargs)
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
        
        config = FIRKStepConfig(**config_kwargs)

        if tableau.has_error_estimate:
            defaults = FIRK_ADAPTIVE_DEFAULTS
        else:
            defaults = FIRK_FIXED_DEFAULTS

        super().__init__(config, defaults)

    def build_implicit_helpers(
        self,
    ) -> Callable:
        """Construct the nonlinear solver chain used by implicit methods."""

        precision = self.precision
        config = self.compile_settings
        tableau = config.tableau
        beta = config.beta
        gamma = config.gamma
        mass = config.M
        stage_count = config.stage_count
        all_stages_n = config.all_stages_n

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

        krylov_tolerance = config.krylov_tolerance
        max_linear_iters = config.max_linear_iters
        correction_type = config.linear_correction_type

        linear_solver = linear_solver_factory(
            operator,
            n=all_stages_n,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
        )

        newton_tolerance = config.newton_tolerance
        max_newton_iters = config.max_newton_iters
        newton_damping = config.newton_damping
        newton_max_backtracks = config.newton_max_backtracks

        nonlinear_solver = newton_krylov_solver_factory(
            residual_function=residual,
            linear_solver=linear_solver,
            n=all_stages_n,
            tolerance=newton_tolerance,
            max_iters=max_newton_iters,
            damping=newton_damping,
            max_backtracks=newton_max_backtracks,
            precision=precision,
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
        """Compile the FIRK device step."""

        config = self.compile_settings
        precision = self.precision
        tableau = config.tableau
        nonlinear_solver = solver_fn
        n = int32(n)
        n_drivers = int32(n_drivers)
        stage_count = int32(self.stage_count)
        all_stages_n = int32(config.all_stages_n)

        has_driver_function = driver_function is not None
        has_error = self.is_adaptive

        stage_rhs_coeffs = tableau.a_flat(numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        typed_zero = numba_precision(0.0)
        error_weights = tableau.error_weights(numba_precision)
        if error_weights is None or not has_error:
            error_weights = tuple(typed_zero for _ in range(stage_count))
        stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)

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

        ends_at_one = stage_time_fractions[-1] == numba_precision(1.0)

        # Buffer settings from compile_settings for selective shared/local
        buffer_settings = config.buffer_settings

        # Unpack boolean flags as compile-time constants
        solver_scratch_shared = buffer_settings.use_shared_solver_scratch
        stage_increment_shared = buffer_settings.use_shared_stage_increment
        stage_driver_stack_shared = buffer_settings.use_shared_stage_driver_stack
        stage_state_shared = buffer_settings.use_shared_stage_state

        # Unpack slice indices for shared memory layout
        shared_indices = buffer_settings.shared_indices
        solver_scratch_slice = shared_indices.solver_scratch
        stage_increment_slice = shared_indices.stage_increment
        stage_driver_stack_slice = shared_indices.stage_driver_stack
        stage_state_slice = shared_indices.stage_state

        # Unpack local sizes for local array allocation
        local_sizes = buffer_settings.local_sizes
        solver_scratch_local_size = local_sizes.nonzero('solver_scratch')
        stage_increment_local_size = local_sizes.nonzero('stage_increment')
        stage_driver_stack_local_size = local_sizes.nonzero('stage_driver_stack')
        stage_state_local_size = local_sizes.nonzero('stage_state')
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
            # Selective allocation from local or shared memory
            # ----------------------------------------------------------- #
            if stage_state_shared:
                stage_state = shared[stage_state_slice]
            else:
                stage_state = cuda.local.array(stage_state_local_size,
                                               precision)
                for _i in range(stage_state_local_size):
                    stage_state[_i] = numba_precision(0.0)

            if solver_scratch_shared:
                solver_scratch = shared[solver_scratch_slice]
            else:
                solver_scratch = cuda.local.array(
                    solver_scratch_local_size, precision
                )
                for _i in range(solver_scratch_local_size):
                    solver_scratch[_i] = numba_precision(0.0)

            if stage_increment_shared:
                stage_increment = shared[stage_increment_slice]
            else:
                stage_increment = cuda.local.array(stage_increment_local_size,
                                                   precision)
                for _i in range(stage_increment_local_size):
                    stage_increment[_i] = numba_precision(0.0)

            if stage_driver_stack_shared:
                stage_driver_stack = shared[stage_driver_stack_slice]
            else:
                stage_driver_stack = cuda.local.array(
                    stage_driver_stack_local_size, precision
                )
                for _i in range(stage_driver_stack_local_size):
                    stage_driver_stack[_i] = numba_precision(0.0)
            # ----------------------------------------------------------- #


            current_time = time_scalar
            end_time = current_time + dt_scalar
            status_code = int32(0)
            stage_rhs_flat = solver_scratch[:all_stages_n]

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
                    driver_function(
                            stage_time,
                            driver_coeffs,
                            driver_slice
                    )


            status_code |= nonlinear_solver(
                stage_increment,
                parameters,
                stage_driver_stack,
                current_time,
                dt_scalar,
                typed_zero,
                state,
                solver_scratch,
                counters,
            )

            for stage_idx in range(stage_count):
                stage_time = (
                    current_time + dt_scalar * stage_time_fractions[stage_idx]
                )

                if has_driver_function:
                    stage_base = stage_idx * n_drivers
                    stage_slice = stage_driver_stack[
                        stage_base:stage_base + n_drivers
                    ]
                    for idx in range (n_drivers):
                        proposed_drivers[idx] = stage_slice[idx]

                for idx in range(n):
                    value = state[idx]
                    for contrib_idx in range(stage_count):
                        flat_idx = stage_idx * stage_count + contrib_idx
                        coeff = stage_rhs_coeffs[flat_idx]
                        if coeff != typed_zero:
                            value += (
                                coeff * stage_increment[contrib_idx * n + idx]
                            )
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

                # If error and output can be derived from stage_state,
                # don't bother evaluating f at each stage.
                do_more_work = ((has_error and accumulates_error) or
                                accumulates_output)

                if do_more_work:
                    observables_function(
                        stage_state,
                        parameters,
                        proposed_drivers,
                        proposed_observables,
                        stage_time,
                    )

                    stage_rhs = stage_rhs_flat[stage_idx * n:(stage_idx +
                                                              int32(1)) * n]
                    dxdt_fn(
                        stage_state,
                        parameters,
                        proposed_drivers,
                        proposed_observables,
                        stage_rhs,
                        stage_time,
                    )


            #use a Kahan summation algorithm to reduce floating point errors
            #see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            if accumulates_output:
                for idx in range(n):
                    solution_acc = typed_zero
                    compensation = typed_zero
                    for stage_idx in range(stage_count):
                        rhs_value = stage_rhs_flat[stage_idx * n + idx]
                        term = (solution_weights[stage_idx] * rhs_value -
                                compensation)
                        temp = solution_acc + term
                        compensation = (temp - solution_acc) - term
                        solution_acc += solution_weights[stage_idx] * rhs_value
                    proposed_state[idx] = state[idx] + solution_acc * dt_scalar

            if has_error and accumulates_error:
                # Standard accumulation path for error
                for idx in range(n):
                    error_acc = typed_zero
                    for stage_idx in range(stage_count):
                        rhs_value = stage_rhs_flat[stage_idx * n + idx]
                        error_acc += error_weights[stage_idx] * rhs_value
                    error[idx] = dt_scalar * error_acc   

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
    def shared_memory_required(self) -> int:
        """Return the number of precision entries required in shared memory."""

        config = self.compile_settings
        stage_driver_total = self.stage_count * config.n_drivers
        return (
            self.solver_shared_elements
            + stage_driver_total
            + config.all_stages_n
        )

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""

        state_dim = self.compile_settings.n
        return state_dim

    @property
    def persistent_local_required(self) -> int:
        """Return the number of persistent local entries required."""

        return 0

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""

        return self.compile_settings.stage_count

    @property
    def solver_shared_elements(self) -> int:
        """Return solver scratch elements accounting for flattened stages."""

        return 2 * self.compile_settings.all_stages_n

    @property
    def algorithm_shared_elements(self) -> int:
        """Return additional shared memory required by the algorithm."""

        return 0

    @property
    def algorithm_local_elements(self) -> int:
        """Return persistent local memory required by the algorithm."""

        return 0

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

