"""Crank–Nicolson step with embedded backward Euler error estimation."""

from typing import Callable, Optional

from numba import cuda, int16, int32
import numpy as np

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms import ImplicitStepConfig
from cubie.integrators.algorithms.base_algorithm_step import StepCache, \
    StepControlDefaults
from cubie.integrators.algorithms.ode_implicitstep import ODEImplicitStep

ALGO_CONSTANTS = {'beta': 1.0,
                  'gamma': 1.0,
                  'M': np.eye}

CN_DEFAULTS = StepControlDefaults(
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
class CrankNicolsonStep(ODEImplicitStep):
    """Crank–Nicolson step with embedded backward Euler error estimation."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dt: Optional[float],
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        preconditioner_order: int = 1,
        krylov_tolerance: float = 1e-6,
        max_linear_iters: int = 100,
        linear_correction_type: str = "minimal_residual",
        newton_tolerance: float = 1e-6,
        max_newton_iters: int = 1000,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 10,
    ) -> None:
        """Initialise the Crank–Nicolson step configuration.

        Parameters
        ----------
        precision
            Precision applied to device buffers.
        n
            Number of state entries advanced per step.
        dt
            Optional fixed step size for fixed-step algorithms. When ``None``
            the controller default is used.
        dxdt_function
            Device derivative function evaluating ``dx/dt``.
        observables_function
            Device function computing system observables.
        driver_function
            Optional device function evaluating drivers at arbitrary times.
        get_solver_helper_fn
            Callable returning device helpers used by the nonlinear solver.
        preconditioner_order
            Order of the truncated Neumann preconditioner.
        krylov_tolerance
            Tolerance used by the linear solver.
        max_linear_iters
            Maximum iterations permitted for the linear solver.
        linear_correction_type
            Identifier for the linear correction strategy.
        newton_tolerance
            Convergence tolerance for the Newton iteration.
        max_newton_iters
            Maximum iterations permitted for the Newton solver.
        newton_damping
            Damping factor applied within Newton updates.
        newton_max_backtracks
            Maximum number of backtracking steps within the Newton solver.

        Returns
        -------
        None
            This constructor updates internal configuration state.
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
            krylov_tolerance=krylov_tolerance,
            max_linear_iters=max_linear_iters,
            linear_correction_type=linear_correction_type,
            newton_tolerance=newton_tolerance,
            max_newton_iters=max_newton_iters,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
            dxdt_function=dxdt_function,
            observables_function=observables_function,
            driver_function=driver_function,
            precision=precision,
        )
        super().__init__(config, CN_DEFAULTS)

    def build_step(
        self,
        solver_fn: Callable,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
        dt: Optional[float],
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - cuda code
        """Build the device function for the Crank–Nicolson step.

        Parameters
        ----------
        solver_fn
            Device nonlinear solver produced by the implicit helper chain.
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
        dt
            Fixed step size supplied for fixed-step execution.

        Returns
        -------
        StepCache
            Container holding the compiled step function and solver.
        """

        stage_coefficient = numba_precision(0.5)
        be_coefficient = numba_precision(1.0)
        has_driver_function = driver_function is not None
        driver_function = driver_function

        solver_shared_elements = self.solver_shared_elements

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
            driver_coefficients,
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
            """Advance the state using Crank–Nicolson with embedded error check.

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
                Device array capturing embedded error estimates.
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
            typed_zero = numba_precision(0.0)

            # Initialize increment buffer
            for i in range(n):
                proposed_state[i] = typed_zero

            solver_scratch = shared[:solver_shared_elements]
            # Reuse solver scratch for the dx/dt evaluation buffer.
            dxdt = solver_scratch[:n]
            # error buffer tracks the stage base during setup.
            base_state = error

            # Evaluate f(state)
            dxdt_fn(
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
            for i in range(n):
                base_state[i] = state[i] + half_dt * dxdt[i]


            # Solve Crank-Nicolson step (main solution)
            if has_driver_function:
                driver_function(
                    end_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            status = solver_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                end_time,
                dt_scalar,
                stage_coefficient,
                base_state,
                solver_scratch,
            )

            for i in range(n):
                increment = proposed_state[i]
                proposed_state[i] = base_state[i] + stage_coefficient * increment
                base_state[i] = increment

            status |= solver_fn(
                base_state,
                parameters,
                proposed_drivers,
                end_time,
                dt_scalar,
                be_coefficient,
                state,
                solver_scratch,
            ) & int32(0xFFFF)  # don't record Newton iterations for error check

            # Compute error as difference between Crank-Nicolson and Backward Euler
            for i in range(n):
                error[i] = proposed_state[i] - (state[i] + base_state[i])

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                end_time,
            )

            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because Crank–Nicolson is a single-stage method."""

        return False

    @property
    def shared_memory_required(self) -> int:
        """Shared memory usage expressed in precision-sized entries."""

        return super().shared_memory_required

    @property
    def local_scratch_required(self) -> int:
        """Local scratch usage expressed in precision-sized entries."""

        return 0

    @property
    def algorithm_shared_elements(self) -> int:
        """Crank–Nicolson does not reserve extra shared scratch."""

        return 0

    @property
    def algorithm_local_elements(self) -> int:
        """Crank–Nicolson does not require persistent local storage."""

        return 0

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
