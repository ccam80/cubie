"""Backward Euler step implementation using Newton–Krylov."""

from typing import Callable, Optional

from numba import cuda
import numpy as np

from cubie._utils import PrecisionDtype
from cubie.integrators.algorithms import ImplicitStepConfig
from cubie.integrators.algorithms.base_algorithm_step import StepCache, \
    StepControlDefaults
from cubie.integrators.algorithms.ode_implicitstep import ODEImplicitStep

ALGO_CONSTANTS = {'beta': 1.0,
                  'gamma': 1.0,
                  'M': np.eye}

BE_DEFAULTS = StepControlDefaults(
        step_controller='fixed',
        step_controller_kwargs={
            'dt': 1e-3,
        })

class BackwardsEulerStep(ODEImplicitStep):
    """Backward Euler step solved with matrix-free Newton–Krylov."""

    def __init__(
        self,
        precision: PrecisionDtype,
        n: int,
        dt: float,
        dxdt_function: Callable,
        observables_function: Callable,
        get_solver_helper_fn: Callable,
        driver_function: Optional[Callable] = None,
        preconditioner_order: int = 1,
        linsolve_tolerance: float = 1e-5,
        max_linear_iters: int = 100,
        linear_correction_type: str = "minimal_residual",
        nonlinear_tolerance: float = 1e-5,
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
        linsolve_tolerance
            Tolerance used by the linear solver.
        max_linear_iters
            Maximum iterations permitted for the linear solver.
        linear_correction_type
            Identifier for the linear correction strategy.
        nonlinear_tolerance
            Convergence tolerance for the Newton iteration.
        max_newton_iters
            Maximum iterations permitted for the Newton solver.
        newton_damping
            Damping factor applied within Newton updates.
        newton_max_backtracks
            Maximum number of backtracking steps within the Newton solver.
        """
        if dt is None:
            dt = BE_DEFAULTS.step_controller_kwargs['dt']

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
            linsolve_tolerance=linsolve_tolerance,
            max_linear_iters=max_linear_iters,
            linear_correction_type=linear_correction_type,
            nonlinear_tolerance=nonlinear_tolerance,
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

        Returns
        -------
        StepCache
            Container holding the compiled step function and solver.
        """

        a_ij = numba_precision(1.0)
        has_driver_function = driver_function is not None
        driver_function = driver_function
        dt = self.dt
        @cuda.jit(
            (
                numba_precision[:],
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
            work_buffer,
            parameters,
            driver_coefficients,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,  # unused here
            error,
            dt_scalar,
            time_scalar,
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
            work_buffer
                Device array used as temporary storage.
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
                Device array capturing solver diagnostics.
            dt_scalar
                Scalar containing the proposed step size.
            time_scalar
                Scalar containing the current simulation time.
            shared
                Device array used for shared memory (unused here).
            persistent_local
                Device array for persistent local storage (unused here).

            Returns
            -------
            int
                Status code returned by the nonlinear solver.
            """

            for i in range(n):
                proposed_state[i] = state[i]

            next_time = time_scalar + dt
            if has_driver_function:
                driver_function(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            resid = cuda.local.array(n, numba_precision)
            z = cuda.local.array(n, numba_precision)

            status = solver_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                dt,
                a_ij,
                state,
                work_buffer,
                resid,
                z,
                error,
            )

            # calculate and save observables (wastes some compute)
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

        return 0

    @property
    def local_scratch_required(self) -> int:
        """Local scratch usage expressed in precision-sized entries."""

        return 4 * self.compile_settings.n

    @property
    def persistent_local_required(self) -> int:
        """Persistent local storage expressed in precision-sized entries."""

        return 0

    @property
    def is_adaptive(self) -> bool:
        """Return ``False`` because backward Euler is fixed step here."""

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
