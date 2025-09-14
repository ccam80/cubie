"""Backward Euler step implementation using Newton–Krylov."""

from numba import cuda
import numpy as np

from cubie.integrators.algorithms_ import ImplicitStepConfig
from cubie.integrators.algorithms_.ode_implicitstep import (
    ODEImplicitStep,
)

ALGO_CONSTANTS = {'beta': 1.0,
                  'gamma': 1.0,
                  'M': np.eye}

class BackwardsEulerStep(ODEImplicitStep):
    """Backward Euler step solved with matrix-free Newton–Krylov."""
    def __init__(self,
                 precision,
                 n,
                 preconditioner_order = 1,
                 linsolve_tolerance = 1e-6,
                 max_linear_iters = 100,
                 linear_correction_type = "minimal_residual",
                 nonlinear_tolerance = 1e-6,
                 max_newton_iters = 100,
                 newton_damping = 0.5,
                 newton_max_backtracks = 10):

        beta = ALGO_CONSTANTS['beta']
        gamma = ALGO_CONSTANTS['gamma']
        M = ALGO_CONSTANTS['M'](n, dtype=precision)
        config = ImplicitStepConfig(
                beta=beta,
                gamma=gamma,
                M=M,
                preconditioner_order=preconditioner_order,
                linsolve_tolerance=linsolve_tolerance,
                max_linear_iters=max_linear_iters,
                linear_correction_type=linear_correction_type,
                nonlinear_tolerance=nonlinear_tolerance,
                max_newton_iters=max_newton_iters,
                newton_damping=newton_damping,
                newton_max_backtracks=newton_max_backtracks)
        super().__init__(config)

    def build_step(self,
                   solver_fn,
                   dxdt_fn,
                   numba_precision,
                   n):  # pragma: no cover - complex
        # Build any implicit helpers before creating the step function
        a_ij = numba_precision(1.0)

        # no cover: start
        @cuda.jit(
            (
                numba_precision[:],  # state (in/out)
                numba_precision[:],  # parameters
                numba_precision[:],  # drivers
                numba_precision[:],  # observables (unused here)
                numba_precision[:],  # proposal state (used as current guess)
                numba_precision[:],  # error
                numba_precision[:],  # timestep (size 1 array)
                numba_precision[:],  # shared (unused)
                numba_precision[:],  # local workspace (size >= 5*n)
            ),
            device=True,
            inline=True,
        )
        def step(state,
                 parameters,
                 drivers,
                 observables,
                 proposed_state,
                 error,
                 dt,
                 shared, persistent_local):

            dt = dt[0]

            # Initial guess is current state
            for i in range(n):
                proposed_state[i] = state[i]

            # Instantiate 4n local arrays for duration of solve.
            delta = cuda.local.array(n, numba_precision)
            resid = cuda.local.array(n, numba_precision)
            z = cuda.local.array(n, numba_precision)
            temp = cuda.local.array(n, numba_precision)

            # Call damped Newton–Krylov. Returns status code
            status = solver_fn(
                proposed_state,           # state (in: guess, out: solution)
                parameters,               # System parameters
                drivers,                  # Forcing vectors
                dt,                       # timestep
                a_ij,                     # stage weight (BE: 1.0)
                state,                    # base_state (y_n)
                delta,                    # work: Newton direction
                resid,                    # work: residual/rhs buffer
                z,                        # work: preconditioned vector
                temp,                     # work: extra workspace
            )

            # On success or failure, write latest iterate back to state
            for i in range(n):
                state[i] = proposed_state[i]
            return status
        # no cover: end
        return step

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

