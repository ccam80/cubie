"""Backward Euler step implementation using Newton–Krylov."""

from numba import cuda
import numpy as np

from cubie.integrators.algorithms_ import ImplicitStepConfig
from cubie.integrators.algorithms_.base_algorithm_step import StepCache
from cubie.integrators.algorithms_.ode_implicitstep import ODEImplicitStep

ALGO_CONSTANTS = {'beta': 1.0,
                  'gamma': 1.0,
                  'M': np.eye}

class BackwardsEulerStep(ODEImplicitStep):
    """Backward Euler step solved with matrix-free Newton–Krylov."""
    def __init__(self,
                 precision,
                 n,
                 dxdt_function,
                 get_solver_helper_fn,
                 preconditioner_order = 1,
                 linsolve_tolerance = 1e-6,
                 max_linear_iters = 100,
                 linear_correction_type = "minimal_residual",
                 nonlinear_tolerance = 1e-6,
                 max_newton_iters = 1000,
                 newton_damping = 0.5,
                 newton_max_backtracks = 10):

        beta = ALGO_CONSTANTS['beta']
        gamma = ALGO_CONSTANTS['gamma']
        M = ALGO_CONSTANTS['M'](n, dtype=precision)
        config = ImplicitStepConfig(
            get_solver_helper_fn=get_solver_helper_fn,
            beta=beta,
            gamma=gamma,
            M=M,
            n=n,
            norm_type="hairer",
            preconditioner_order=preconditioner_order,
            linsolve_tolerance=linsolve_tolerance,
            max_linear_iters=max_linear_iters,
            linear_correction_type=linear_correction_type,
            nonlinear_tolerance=nonlinear_tolerance,
            max_newton_iters=max_newton_iters,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
            dxdt_function=dxdt_function,
        )
        super().__init__(config)

    def build_step(self,
                   solver_fn,
                   dxdt_fn,
                   obs_fn,
                   numba_precision,
                   n):  # pragma: no cover - complex
        """Build the device function for a backward Euler step."""

        a_ij = numba_precision(1.0)

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
            observables, # unused here
            error,
            dt_scalar,
            shared,
            persistent_local,
        ):
            #calculate and save observables (wastes some compute)
            obs_fn(state, parameters, drivers, observables)

            for i in range(n):
                proposed_state[i] = state[i]

            resid = cuda.local.array(n, numba_precision)
            z = cuda.local.array(n, numba_precision)

            status = solver_fn(
                proposed_state,
                parameters,
                drivers,
                dt_scalar,
                a_ij,
                state,
                work_buffer,
                resid,
                z,
                error,
            )
            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)

    @property
    def is_multistage(self) -> bool:  # pragma: no cover - simple
        return False

    @property
    def shared_memory_required(self) -> int:  # pragma: no cover - simple
        # 4 work vectors = 4*n. All reusable; no kept state.
        return 4 * self.compile_settings.n

    @property
    def local_scratch_required(self) -> int:  # pragma: no cover - simple
        return 4 * self.compile_settings.n

    @property
    def persistent_local_required(self) -> int:
        return 0

    @property
    def is_adaptive(self) -> bool:
        return False

    @property
    def threads_per_step(self) -> int:
        return 1

    @property
    def settings_dict(self) -> dict:
        return self.compile_settings.settings_dict

    @property
    def order(self) -> int:
        """Order of the algorithm."""
        return 1