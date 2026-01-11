"""Crank–Nicolson step with embedded backward Euler error estimation."""

from typing import Callable, Optional

from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.base_algorithm_step import StepCache
from tests.integrators.algorithms.instrumented.ode_implicitstep import (
    InstrumentedODEImplicitStep,
)
from cubie.integrators.algorithms.crank_nicolson import (
    ALGO_CONSTANTS,
    CN_DEFAULTS,
    CrankNicolsonStepConfig,
)


class InstrumentedCrankNicolsonStep(InstrumentedODEImplicitStep):
    """Crank–Nicolson step with embedded backward Euler error estimation."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        evaluate_f: Optional[Callable] = None,
        evaluate_observables: Optional[Callable] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        preconditioner_order: Optional[int] = None,
        krylov_atol: Optional[float] = None,
        krylov_rtol: Optional[float] = None,
        krylov_max_iters: Optional[int] = None,
        linear_correction_type: Optional[str] = None,
        newton_atol: Optional[float] = None,
        newton_rtol: Optional[float] = None,
        newton_max_iters: Optional[int] = None,
        newton_damping: Optional[float] = None,
        newton_max_backtracks: Optional[int] = None,
        dxdt_location: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialise the Crank–Nicolson step configuration.

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
        krylov_atol
            Absolute tolerance used by the linear solver. If None, uses default
            from LinearSolverConfig.
        krylov_rtol
            Relative tolerance used by the linear solver. If None, uses default
            from LinearSolverConfig.
        krylov_max_iters
            Maximum iterations permitted for the linear solver. If None, uses
            default from LinearSolverConfig.
        linear_correction_type
            Identifier for the linear correction strategy. If None, uses
            default from LinearSolverConfig.
        newton_atol
            Absolute tolerance for the Newton iteration. If None, uses
            default from NewtonKrylovConfig.
        newton_rtol
            Relative tolerance for the Newton iteration. If None, uses
            default from NewtonKrylovConfig.
        newton_max_iters
            Maximum iterations permitted for the Newton solver. If None, uses
            default from NewtonKrylovConfig.
        newton_damping
            Damping factor applied within Newton updates. If None, uses
            default from NewtonKrylovConfig.
        newton_max_backtracks
            Maximum number of backtracking steps within the Newton solver. If
            None, uses default from NewtonKrylovConfig.
        dxdt_location
            Memory location for dxdt buffer: 'local' or 'shared'. If None,
            defaults to 'local'.

        Returns
        -------
        None
            This constructor updates internal configuration state.
        """

        beta = ALGO_CONSTANTS["beta"]
        gamma = ALGO_CONSTANTS["gamma"]
        M = ALGO_CONSTANTS["M"](n, dtype=precision)

        # Build config kwargs conditionally
        config_kwargs = {
            "precision": precision,
            "n": n,
            "get_solver_helper_fn": get_solver_helper_fn,
            "beta": beta,
            "gamma": gamma,
            "M": M,
            "evaluate_f": evaluate_f,
            "evaluate_observables": evaluate_observables,
            "evaluate_driver_at_t": evaluate_driver_at_t,
        }
        if preconditioner_order is not None:
            config_kwargs["preconditioner_order"] = preconditioner_order
        if dxdt_location is not None:
            config_kwargs["dxdt_location"] = dxdt_location

        config = CrankNicolsonStepConfig(**config_kwargs)

        # Build solver kwargs dict conditionally
        solver_kwargs = {}
        if krylov_atol is not None:
            solver_kwargs["krylov_atol"] = krylov_atol
        if krylov_rtol is not None:
            solver_kwargs["krylov_rtol"] = krylov_rtol
        if krylov_max_iters is not None:
            solver_kwargs["krylov_max_iters"] = krylov_max_iters
        if linear_correction_type is not None:
            solver_kwargs["linear_correction_type"] = linear_correction_type
        if newton_atol is not None:
            solver_kwargs["newton_atol"] = newton_atol
        if newton_rtol is not None:
            solver_kwargs["newton_rtol"] = newton_rtol
        if newton_max_iters is not None:
            solver_kwargs["newton_max_iters"] = newton_max_iters
        if newton_damping is not None:
            solver_kwargs["newton_damping"] = newton_damping
        if newton_max_backtracks is not None:
            solver_kwargs["newton_max_backtracks"] = newton_max_backtracks

        super().__init__(config, CN_DEFAULTS.copy(), **solver_kwargs)

        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers with buffer_registry."""
        config = self.compile_settings

        # Register solver child buffers
        _ = buffer_registry.get_child_allocators(
            self, self.solver, name="solver"
        )

        # Register cn_dxdt buffer
        buffer_registry.register(
            "cn_dxdt",
            self,
            config.n,
            config.dxdt_location,
            precision=config.precision,
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
        """Build the device function for the Crank–Nicolson step.

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
            Container holding the compiled step function and solver.
        """

        stage_coefficient = numba_precision(0.5)
        be_coefficient = numba_precision(1.0)
        has_evaluate_driver_at_t = evaluate_driver_at_t is not None
        n = int32(n)
        typed_zero = numba_precision(0.0)

        # Get child allocators for Newton solver
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(
                self, self.solver, name="solver"
            )
        )
        alloc_dxdt = buffer_registry.get_allocator("cn_dxdt", self)

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
            error,
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
            solver_shared = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(
                shared, persistent_local
            )
            dxdt = alloc_dxdt(shared, persistent_local)

            stage_rhs = cuda.local.array(n, numba_precision)
            cn_increment = cuda.local.array(n, numba_precision)
            observable_count = proposed_observables.shape[0]

            for idx in range(n):
                cn_increment[idx] = typed_zero
                residuals[0, idx] = typed_zero
                jacobian_updates[0, idx] = typed_zero
                stage_states[0, idx] = typed_zero
                stage_derivatives[0, idx] = typed_zero
                stage_increments[0, idx] = typed_zero

            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = typed_zero
            for driver_idx in range(stage_drivers.shape[1]):
                stage_drivers[0, driver_idx] = typed_zero

            # base_state aliases error as their lifetimes are disjoint
            base_state = error

            # Evaluate f(state)
            evaluate_f(
                state,
                parameters,
                drivers_buffer,
                observables,
                dxdt,
                time_scalar,
            )

            half_dt = dt_scalar * numba_precision(0.5)
            end_time = time_scalar + dt_scalar

            # Form the Crank-Nicolson stage base
            for idx in range(n):
                base_state[idx] = state[idx] + half_dt * dxdt[idx]

            # Solve Crank-Nicolson step (main solution)
            if has_evaluate_driver_at_t:
                evaluate_driver_at_t(
                    end_time,
                    driver_coefficients,
                    proposed_drivers,
                )
            for driver_idx in range(stage_drivers.shape[1]):
                stage_drivers[0, driver_idx] = proposed_drivers[driver_idx]

            status = solver_function(
                proposed_state,
                parameters,
                proposed_drivers,
                end_time,
                dt_scalar,
                stage_coefficient,
                base_state,
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

            # LOGGING: capture final residual from solver scratch
            for idx in range(n):
                increment_value = proposed_state[idx]
                final_state = base_state[idx] + (
                    stage_coefficient * increment_value
                )
                trapezoidal_increment = final_state - state[idx]
                proposed_state[idx] = final_state
                base_state[idx] = increment_value
                cn_increment[idx] = trapezoidal_increment
                stage_increments[0, idx] = trapezoidal_increment
                stage_states[0, idx] = final_state

            be_status = solver_function(
                base_state,
                parameters,
                proposed_drivers,
                end_time,
                dt_scalar,
                be_coefficient,
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
            status |= be_status

            # Compute error as difference between Crank-Nicolson and Backward Euler
            for idx in range(n):
                error[idx] = proposed_state[idx] - (
                    state[idx] + base_state[idx]
                )

            evaluate_observables(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                end_time,
            )

            # LOGGING: capture stage observables
            for obs_idx in range(observable_count):
                stage_observables[0, obs_idx] = proposed_observables[obs_idx]

            # LOGGING: capture stage derivatives
            evaluate_f(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                stage_rhs,
                end_time,
            )

            for idx in range(n):
                rhs_value = stage_rhs[idx]
                stage_derivatives[0, idx] = rhs_value
                jacobian_updates[0, idx] = typed_zero

            return status

        return StepCache(step=step, nonlinear_solver=solver_function)

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because Crank–Nicolson is a single-stage method."""

        return False

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` because the embedded error estimate enables adaptivity."""

        return True

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""

        return 1

    @property
    def order(self) -> int:
        """Return the classical order of the Crank–Nicolson method."""

        return 2
