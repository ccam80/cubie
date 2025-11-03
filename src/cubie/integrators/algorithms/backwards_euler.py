"""Backward Euler step implementation using Newton–Krylov."""

from typing import Callable, Optional

from numba import cuda, int16
import numpy as np

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms import ImplicitStepConfig
from cubie.integrators.algorithms.base_algorithm_step import StepCache, \
    StepControlDefaults
from cubie.integrators.algorithms.ode_implicitstep import ODEImplicitStep

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
        dt: float,
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        preconditioner_order: int = 1,
        krylov_tolerance: float = 1e-5,
        max_linear_iters: int = 100,
        linear_correction_type: str = "minimal_residual",
        newton_tolerance: float = 1e-5,
        max_newton_iters: int = 100,
        newton_damping: float = 0.85,
        newton_max_backtracks: int = 25,
    ) -> None:
        """Initialise the backward Euler step configuration.

        Parameters
        ----------
        precision
            Precision applied to device buffers.
        n
            Number of state entries advanced per step.
        dt
            Fixed step size for fixed-step algorithms. When ``None`` the
            controller default is used.
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
        """
        if dt is None:
            dt = BE_DEFAULTS.step_controller['dt']

        beta = ALGO_CONSTANTS['beta']
        gamma = ALGO_CONSTANTS['gamma']
        M = ALGO_CONSTANTS['M'](n, dtype=precision)
        config = ImplicitStepConfig(
            get_solver_helper_fn=get_solver_helper_fn,
            beta=beta,
            gamma=gamma,
            M=M,
            n=n,
            dt=dt,
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
        super().__init__(config, BE_DEFAULTS.copy())

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
        """Build the device function for a backward Euler step.

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

        a_ij = numba_precision(1.0)
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
            error,  # Non-adaptive algorithms receive a zero-length slice.
            dt_scalar,
            time_scalar,
            first_step_flag,
            accepted_flag,
            shared,
            persistent_local,
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
            solver_scratch = shared[: solver_shared_elements]

            for i in range(n):
                proposed_state[i] = solver_scratch[i]

            next_time = time_scalar + dt
            if has_driver_function:
                driver_function(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            status = solver_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                next_time,
                dt,
                a_ij,
                state,
                solver_scratch,
            )

            for i in range(n):
                solver_scratch[i] = proposed_state[i]
                proposed_state[i] += state[i]

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )

            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because backward Euler is a single-stage method."""

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
        """Backward Euler has no additional shared-memory requirements."""

        return 0

    @property
    def algorithm_local_elements(self) -> int:
        """Backward Euler does not reserve persistent local storage."""

        return 0

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