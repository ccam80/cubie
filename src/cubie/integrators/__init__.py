"""Numerical integration algorithms and settings for ODE solving.

``SingleIntegratorRun`` is the primary entry point. It composes a device
loop callable from controller, algorithm, and loop factories based on the
provided
:class:`cubie.integrators.IntegratorRunSettings.IntegratorRunSettings`.

This package collects CUDA-oriented integration components, including
algorithm factories, solver helpers, loop builders, and controller
utilities that orchestrate initial value problem (IVP) integrations.

Subpackages
-----------
algorithms
    Explicit and implicit step factories that share configuration helpers
    and provide Euler, backward Euler, predictor-corrector, and
    Crank--Nicolson implementations.
loops
    CUDA loop factories that assemble device functions and manage shared
    and local memory layouts for IVP execution.
matrix_free_solvers
    Matrix-free Newton--Krylov and linear solver factories consumed by the
    implicit algorithms.
step_control
    Adaptive and fixed-step controller factories used to update step sizes
    on device.

Notes
-----
``IntegratorReturnCodes`` encodes algorithm-level statuses. Matrix-free
solver status codes remain available from
``cubie.integrators.matrix_free_solvers`` and embed the Newton iteration
count in the upper 16 bits when returned from implicit algorithms.
"""

from enum import IntEnum

from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from cubie.integrators.algorithms import (
    BackwardsEulerPCStep,
    BackwardsEulerStep,
    CrankNicolsonStep,
    ExplicitEulerStep,
    ExplicitStepConfig,
    ImplicitStepConfig,
    get_algorithm_step,
)
from cubie.integrators.loops import IVPLoop
from cubie.integrators.matrix_free_solvers import (
    linear_solver_factory,
    newton_krylov_solver_factory,
)
from cubie.integrators.step_control import (
    AdaptiveIController,
    AdaptivePIController,
    AdaptivePIDController,
    FixedStepController,
    GustafssonController,
    get_controller,
)


class IntegratorReturnCodes(IntEnum):
    """Enumerate outcomes returned by integrator kernels.

    Notes
    -----
    Integer codes mirror the solver codes for compatibility, but
    ``SUCCESS`` values differ between integrator and solver enumerations.
    """

    SUCCESS = 0
    NEWTON_BACKTRACKING_NO_SUITABLE_STEP = 1
    MAX_NEWTON_ITERATIONS_EXCEEDED = 2
    MAX_LINEAR_ITERATIONS_EXCEEDED = 4
    STEP_TOO_SMALL = 8
    DT_EFF_EFFECTIVELY_ZERO = 16
    MAX_LOOP_ITERS_EXCEEDED = 32


__all__ = [
    "SingleIntegratorRun",
    "IntegratorReturnCodes",
    "get_algorithm_step",
    "ExplicitStepConfig",
    "ImplicitStepConfig",
    "ExplicitEulerStep",
    "BackwardsEulerStep",
    "BackwardsEulerPCStep",
    "CrankNicolsonStep",
    "IVPLoop",
    "linear_solver_factory",
    "newton_krylov_solver_factory",
    "AdaptiveIController",
    "AdaptivePIController",
    "AdaptivePIDController",
    "FixedStepController",
    "GustafssonController",
    "get_controller",
]
