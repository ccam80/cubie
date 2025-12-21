"""Backward Euler step implementation using Newton–Krylov."""

from typing import Callable, Optional

from numba import cuda, int32
import numpy as np

from cubie._utils import PrecisionDType
from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms import ImplicitStepConfig
from cubie.integrators.algorithms.base_algorithm_step import StepCache, \
    StepControlDefaults
from cubie.integrators.algorithms.ode_implicitstep import ODEImplicitStep
from tests.integrators.algorithms.instrumented.matrix_free_solvers import (
    InstrumentedLinearSolver,
    InstrumentedNewtonKrylov,
)

ALGO_CONSTANTS = {'beta': 1.0,
                  'gamma': 1.0,
                  'M': np.eye}

BE_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)

class BackwardsEulerStep(ODEImplicitStep):
    """Backward Euler step solved with matrix-free Newton–Krylov."""

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
    ) -> None:
        """Initialise the backward Euler step configuration.

        Parameters
        ----------
        precision
            Precision applied to device buffers.
        n
            Number of state entries advanced per step.
        dxdt_function
            Device derivative function evaluating ``dx/dt``.
        observables_function
            Device function computing system observables.
        driver_function
            Optional device function evaluating drivers at arbitrary times.
        get_solver_helper_fn
            Callable returning device helpers used by the nonlinear solver.
        preconditioner_order
            Order of the truncated Neumann preconditioner. If None, uses
            default from ImplicitStepConfig.
        krylov_tolerance
            Tolerance used by the linear solver. If None, uses default from
            LinearSolverConfig.
        max_linear_iters
            Maximum iterations permitted for the linear solver. If None, uses
            default from LinearSolverConfig.
        linear_correction_type
            Identifier for the linear correction strategy. If None, uses
            default from LinearSolverConfig.
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
        """
        beta = ALGO_CONSTANTS['beta']
        gamma = ALGO_CONSTANTS['gamma']
        M = ALGO_CONSTANTS['M'](n, dtype=precision)
        config = ImplicitStepConfig(
            get_solver_helper_fn=get_solver_helper_fn,
            beta=beta,
            gamma=gamma,
            M=M,
            n=n,
            preconditioner_order=preconditioner_order,
            dxdt_function=dxdt_function,
            observables_function=observables_function,
            driver_function=driver_function,
            precision=precision,
        )
        
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
            solver_kwargs["newton_max_backtracks"] = newton_max_backtracks

        super().__init__(config, BE_DEFAULTS.copy(), **solver_kwargs)

    def build_implicit_helpers(self) -> None:
        """Construct the nonlinear solver chain used by implicit methods.

        Overrides the parent method to use instrumented solvers that record
        logging data for each Newton and linear solver iteration.
        """
        config = self.compile_settings
        precision = config.precision
        n = config.n
        beta = config.beta
        gamma = config.gamma
        mass = config.M
        preconditioner_order = config.preconditioner_order

        get_fn = config.get_solver_helper_fn

        preconditioner = get_fn(
            "neumann_preconditioner",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        residual_fn = get_fn(
            "stage_residual",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        linear_operator = get_fn(
            "linear_operator",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        # Create instrumented linear solver
        linear_solver = InstrumentedLinearSolver(
            precision=precision,
            n=n,
            correction_type=config.linear_correction_type,
            krylov_tolerance=config.krylov_tolerance,
            max_linear_iters=config.max_linear_iters,
        )
        linear_solver.update(
            operator_apply=linear_operator,
            preconditioner=preconditioner,
        )

        # Create instrumented Newton-Krylov solver
        self.solver = InstrumentedNewtonKrylov(
            precision=precision,
            n=n,
            linear_solver=linear_solver,
            newton_tolerance=config.newton_tolerance,
            max_newton_iters=config.max_newton_iters,
            newton_damping=config.newton_damping,
            newton_max_backtracks=config.newton_max_backtracks,
        )
        self.solver.update(residual_function=residual_fn)

        self.update_compile_settings(
            solver_function=self.solver.device_function
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
    ) -> StepCache:  # pragma: no cover - cuda code
        """Build the device function for a backward Euler step.

        Parameters
        ----------
        dxdt_fn
            Device derivative function for the ODE system.
        observables_function
            Device observable computation helper.
        driver_function
            Optional device function evaluating drivers at arbitrary times.
        numba_precision
            Numba precision corresponding to the configured precision.
        n
            Dimension of the state vector.
        n_drivers
            Number of driver signals provided to the system.

        Returns
        -------
        StepCache
            Container holding the compiled step function and solver.
        """
        a_ij = numba_precision(1.0)
        has_driver_function = driver_function is not None
        driver_function = driver_function
        n = int32(n)
        
        # Get child allocators for Newton solver
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(self, self.solver,
                                                 name='solver')
        )
        #ANTI-PATTERN registration, move to register_buffers and add a config
        # subclass with this location
        buffer_registry.register('increment_cache', n,
                                        precision=self.precision,
                                        location='local',
                                        aliases='solver_shared',
                                        persistent=True)
        
        solver_fn = solver_function

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
            driver_coefficients,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,  # Non-adaptive algorithms receive a zero-length slice.
            residuals,
            jacobian_updates,
            stage_states,
            stage_derivatives,
            stage_observables,
            stage_drivers,
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
            """Perform one backward Euler update.

            Parameters
            ----------
            state
                Device array storing the current state.
            proposed_state
                Device array receiving the updated state.
            parameters
                Device array of static model parameters.
            driver_coefficients
                Device array containing spline driver coefficients.
            drivers_buffer
                Device array of time-dependent drivers.
            proposed_drivers
                Device array receiving proposed driver samples.
            observables
                Device array storing accepted observable outputs.
            proposed_observables
                Device array receiving proposed observable outputs.
            error
                Device array capturing solver diagnostics. Fixed-step
                algorithms receive a zero-length slice that can be repurposed
                as scratch when available.
            dt_scalar
                Scalar containing the proposed step size.
            time_scalar
                Scalar containing the current simulation time.
            shared
                Device array providing shared scratch buffers.
            persistent_local
                Device array for persistent local storage (unused here).

            Returns
            -------
            int
                Status code returned by the nonlinear solver.
            """
            solver_scratch = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(shared, persistent_local)
            typed_zero = numba_precision(0.0)
            stage_rhs = cuda.local.array(n, numba_precision)

            observable_count = proposed_observables.shape[0]

            # LOGGING: Initialize instrumentation arrays
            for i in range(n):
                proposed_state[i] = solver_scratch[i]
                residuals[0, i] = typed_zero
                jacobian_updates[0, i] = typed_zero
                stage_states[0, i] = typed_zero
                stage_derivatives[0, i] = typed_zero
                stage_increments[0, i] = typed_zero

            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = typed_zero
            for driver_idx in range(stage_drivers.shape[1]):
                stage_drivers[0, driver_idx] = typed_zero

            next_time = time_scalar + dt_scalar
            if has_driver_function:
                driver_function(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            # LOGGING: Record stage drivers
            for driver_idx in range(stage_drivers.shape[1]):
                stage_drivers[0, driver_idx] = proposed_drivers[driver_idx]

            status = solver_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                next_time,
                dt_scalar,
                a_ij,
                state,
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

            # LOGGING: Record increment, residual, and state values
            for i in range(n):
                solver_scratch[i] = proposed_state[i]
                proposed_state[i] += state[i]
                stage_increments[0, i] = solver_scratch[i]
                residuals[0, i] = solver_scratch[i + n]
                stage_states[0, i] = proposed_state[i]

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )

            # LOGGING: Record observables
            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = proposed_observables[obs_idx]

            dxdt_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                stage_rhs,
                next_time,
            )

            # LOGGING: Record derivatives
            for i in range(n):
                stage_derivatives[0, i] = stage_rhs[i]
                jacobian_updates[0, i] = typed_zero

            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because backward Euler is a single-stage method."""

        return False

    @property
    def is_adaptive(self) -> bool:
        """Return ``False`` because backward Euler is fixed step."""

        return False

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""

        return 1

    @property
    def settings_dict(self) -> dict:
        """Return the configuration dictionary for the step."""

        return self.compile_settings.settings_dict

    @property
    def order(self) -> int:
        """Return the classical order of the backward Euler method."""
        return 1

    @property
    def dxdt_function(self) -> Optional[Callable]:
        """Return the derivative device function."""

        return self.compile_settings.dxdt_function

    @property
    def identifier(self) -> str:
        return "backwards_euler"
