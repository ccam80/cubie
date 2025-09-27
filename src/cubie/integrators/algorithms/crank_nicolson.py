"""Crank–Nicolson step with embedded backward Euler error estimation."""

from typing import Callable

from numba import cuda, int32
import numpy as np

from cubie._utils import PrecisionDtype
from cubie.integrators.algorithms import ImplicitStepConfig
from cubie.integrators.algorithms.base_algorithm_step import StepCache
from cubie.integrators.algorithms.ode_implicitstep import ODEImplicitStep

ALGO_CONSTANTS = {'beta': 1.0,
                  'gamma': 1.0,
                  'M': np.eye}

class CrankNicolsonStep(ODEImplicitStep):
    """Crank–Nicolson step with embedded backward Euler error estimation."""

    def __init__(
        self,
        precision: PrecisionDtype,
        n: int,
        dxdt_function: Callable,
        get_solver_helper_fn: Callable,
        preconditioner_order: int = 1,
        linsolve_tolerance: float = 1e-6,
        max_linear_iters: int = 100,
        linear_correction_type: str = "minimal_residual",
        nonlinear_tolerance: float = 1e-6,
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
        dxdt_function
            Device derivative function evaluating ``dx/dt``.
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
            linsolve_tolerance=linsolve_tolerance,
            max_linear_iters=max_linear_iters,
            linear_correction_type=linear_correction_type,
            nonlinear_tolerance=nonlinear_tolerance,
            max_newton_iters=max_newton_iters,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
            dxdt_function=dxdt_function,
            precision=precision,
        )
        super().__init__(config)

    def build_step(
        self,
        solver_fn: Callable,
        dxdt_fn: Callable,
        obs_fn: Callable,
        numba_precision: type,
        n: int,
    ) -> StepCache:  # pragma: no cover - cuda code
        """Build the device function for the Crank–Nicolson step.

        Parameters
        ----------
        solver_fn
            Device nonlinear solver produced by the implicit helper chain.
        dxdt_fn
            Device derivative function for the ODE system.
        obs_fn
            Device observable computation helper.
        numba_precision
            Numba precision corresponding to the configured precision.
        n
            Dimension of the state vector.

        Returns
        -------
        StepCache
            Container holding the compiled step function and solver.
        """

        half = numba_precision(0.5)
        a_ij = numba_precision(1.0)
        # Backward Euler coefficient for error estimation

        @cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
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
            drivers,
            observables,
            error,
            dt_scalar,
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
            work_buffer
                Device array used as temporary storage.
            parameters
                Device array of static model parameters.
            drivers
                Device array of time-dependent drivers.
            observables
                Device array receiving observable outputs.
            error
                Device array capturing embedded error estimates.
            dt_scalar
                Scalar containing the proposed step size.
            shared
                Device array used for shared memory (unused here).
            persistent_local
                Device array for persistent local storage (unused here).

            Returns
            -------
            int
                Status code returned by the nonlinear solver.
            """


            # Initialize proposed state
            for i in range(n):
                proposed_state[i] = state[i]

            # Work arrays (reused for both CN and BE computations)
            resid = cuda.local.array(n, numba_precision)
            z = cuda.local.array(n, numba_precision)
            temp = cuda.local.array(n, numba_precision)

            # Additional array for error estimation
            be_state = cuda.local.array(n, numba_precision)
            base_adjusted = cuda.local.array(n, numba_precision)

            # Evaluate f(state) to enforce Crank-Nicolson averaging
            dxdt_fn(state, parameters, drivers, observables, resid)

            cn_dt = dt_scalar * half
            for i in range(n):
                base_adjusted[i] = state[i] + cn_dt * resid[i]

            # Solve Crank-Nicolson step (main solution)
            status = solver_fn(
                proposed_state,
                parameters,
                drivers,
                cn_dt,
                a_ij,
                base_adjusted,
                work_buffer,
                resid,
                z,
                temp,
            )

            # Solve Backward Euler step for error estimation (start from CN solution)
            for i in range(n):
                be_state[i] = proposed_state[i]

            # calculate and save observables (wastes some compute)
            obs_fn(proposed_state, parameters, drivers, observables)

            status |= solver_fn(
                be_state,
                parameters,
                drivers,
                dt_scalar,
                a_ij,
                state,
                work_buffer,
                resid,
                z,
                temp,
            ) & int32(0xFFFF)  # don't record Newton iterations for error check

            # Compute error as difference between Crank-Nicolson and Backward Euler
            for i in range(n):
                error[i] = proposed_state[i] - be_state[i]

            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because Crank–Nicolson is a single-stage method."""

        return False

    @property
    def shared_memory_required(self) -> int:
        """Shared memory usage expressed in precision-sized entries."""

        return 0

    @property
    def local_scratch_required(self) -> int:
        """Local scratch usage expressed in precision-sized entries."""

        return 5 * self.compile_settings.n

    @property
    def persistent_local_required(self) -> int:
        """Persistent local storage expressed in precision-sized entries."""

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
