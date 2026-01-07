"""Backward Euler step implementation using Newton–Krylov."""

from typing import Callable, Optional

from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.backwards_euler import \
    BackwardsEulerStepConfig, ALGO_CONSTANTS, BE_DEFAULTS
from cubie.integrators.algorithms.base_algorithm_step import StepCache
from tests.integrators.algorithms.instrumented.ode_implicitstep import \
    InstrumentedODEImplicitStep

class InstrumentedBackwardsEulerStep(InstrumentedODEImplicitStep):
    """Backward Euler integration step for implicit ODE updates."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        evaluate_f: Optional[Callable] = None,
        evaluate_observables: Optional[Callable] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        preconditioner_order: Optional[int] = None,
        krylov_tolerance: Optional[float] = None,
        max_linear_iters: Optional[int] = None,
        linear_correction_type: Optional[str] = None,
        newton_tolerance: Optional[float] = None,
        max_newton_iters: Optional[int] = None,
        newton_damping: Optional[float] = None,
        newton_max_backtracks: Optional[int] = None,
        increment_cache_location: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialise the backward Euler step configuration.

        Parameters
        ----------
        precision
            Precision applied to device buffers.
        n
            Number of state entries advanced per step.
        evaluate_f
            Device function for evaluating f(t, y) right-hand side.
        evaluate_observables
            Device function computing system observables.
        evaluate_driver_at_t
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
        increment_cache_location
            Memory location for increment cache buffer: 'local' or 'shared'.
            If None, defaults to 'local'.
        """
        beta = ALGO_CONSTANTS["beta"]
        gamma = ALGO_CONSTANTS["gamma"]
        M = ALGO_CONSTANTS["M"](n, dtype=precision)

        config_kwargs = {
            "get_solver_helper_fn": get_solver_helper_fn,
            "beta": beta,
            "gamma": gamma,
            "M": M,
            "n": n,
            "preconditioner_order": preconditioner_order,
            "evaluate_f": evaluate_f,
            "evaluate_observables": evaluate_observables,
            "evaluate_driver_at_t": evaluate_driver_at_t,
            "precision": precision,
        }
        if increment_cache_location is not None:
            config_kwargs["increment_cache_location"] = (
                increment_cache_location
            )

        config = BackwardsEulerStepConfig(**config_kwargs)

        solver_kwargs = {}
        if krylov_tolerance is not None:
            solver_kwargs["krylov_tolerance"] = krylov_tolerance
        if max_linear_iters is not None:
            solver_kwargs["max_linear_iters"] = max_linear_iters
        if linear_correction_type is not None:
            solver_kwargs["linear_correction_type"] = linear_correction_type
        if newton_tolerance is not None:
            solver_kwargs["newton_tolerance"] = newton_tolerance
        if max_newton_iters is not None:
            solver_kwargs["max_newton_iters"] = max_newton_iters
        if newton_damping is not None:
            solver_kwargs["newton_damping"] = newton_damping
        if newton_max_backtracks is not None:
            solver_kwargs["newton_max_backtracks"] = newton_max_backtracks

        super().__init__(config, BE_DEFAULTS.copy(), **solver_kwargs)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers with buffer_registry."""
        config = self.compile_settings

        # Register solver child buffers
        _ = buffer_registry.get_child_allocators(
            self, self.solver, name='solver_scratch'
        )

        # Register increment cache buffer
        buffer_registry.register(
            'increment_cache',
            self,
            config.n,
            config.increment_cache_location,
            persistent=True,
            precision=config.precision
        )

    def build_step(
        self,
        evaluate_f: Callable,
        evaluate_observables: Callable,
        evaluate_driver_at_t: Optional[Callable],
        solver_function: Callable,
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - cuda code
        """Build the device function for a backward Euler step.

        Parameters
        ----------
        evaluate_f
            Device function for evaluating f(t, y).
        evaluate_observables
            Device function for computing observables.
        evaluate_driver_at_t
            Optional device function for evaluating drivers at time t.
        solver_function
            Device function for the Newton-Krylov nonlinear solver.
        numba_precision
            Numba type for device buffers.
        n
            State vector dimension.
        n_drivers
            Number of driver signals.

        Returns
        -------
        StepCache
            Compiled step function and solver.
        """
        a_ij = numba_precision(1.0)
        has_driver_function = evaluate_driver_at_t is not None
        driver_function = evaluate_driver_at_t
        n = int32(n)
        
        # Get child allocators for Newton solver (already registered in register_buffers)
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(self, self.solver,
                                                 name='solver')
        )
        
        # Get increment cache allocator (registered in register_buffers)
        alloc_increment_cache = buffer_registry.get_allocator(
            'increment_cache', self
        )
        
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
            solver_shared = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(shared, persistent_local)
            typed_zero = numba_precision(0.0)
            increment_cache = alloc_increment_cache(shared, persistent_local)

            for i in range(n):
                proposed_state[i] = increment_cache[i]

            observable_count = proposed_observables.shape[0]
            stage_rhs = cuda.local.array(n, numba_precision) # logging
            # LOGGING: Initialize instrumentation arrays
            for i in range(n):
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
                solver_shared,
                solver_persistent,
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
                increment_cache[i] = proposed_state[i]
                proposed_state[i] += state[i]
                stage_increments[0, i] = proposed_state[i]
                stage_states[0, i] = proposed_state[i]

            evaluate_observables(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )

            # LOGGING: Record observables
            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = proposed_observables[obs_idx]

            evaluate_f(
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
