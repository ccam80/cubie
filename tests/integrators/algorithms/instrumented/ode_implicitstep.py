"""Infrastructure for implicit integration step implementations."""

from typing import Optional

from tests.integrators.algorithms.instrumented.matrix_free_solvers import(
    InstrumentedNewtonKrylov,
    InstrumentedLinearSolver,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ODEImplicitStep,
    ImplicitStepConfig,
    StepControlDefaults
)


class InstrumentedODEImplicitStep(ODEImplicitStep):
    """Base helper for implicit integration algorithms."""

    def __init__(
        self,
        config: ImplicitStepConfig,
        _controller_defaults: StepControlDefaults,
        solver_type: str = "newton",
        krylov_atol: Optional[float] = None,
        krylov_rtol: Optional[float] = None,
        krylov_max_iters: Optional[int] = None,
        linear_correction_type: Optional[str] = None,
        newton_atol: Optional[float] = None,
        newton_rtol: Optional[float] = None,
        newton_max_iters: Optional[int] = None,
        newton_damping: Optional[float] = None,
        newton_max_backtracks: Optional[int] = None,
        **kwargs,
    ):
            super().__init__(
                config=config,
                _controller_defaults=_controller_defaults,
                solver_type=solver_type,
                krylov_atol=krylov_atol,
                krylov_rtol=krylov_rtol,
                krylov_max_iters=krylov_max_iters,
                linear_correction_type=linear_correction_type,
                newton_atol=newton_atol,
                newton_rtol=newton_rtol,
                newton_max_iters=newton_max_iters,
                newton_damping=newton_damping,
                newton_max_backtracks=newton_max_backtracks,
                **kwargs,
            )

            # Build instrumented solvers for use in place of production ones
            linear_solver = InstrumentedLinearSolver(
                precision=config.precision,
                n=config.n,
                correction_type=linear_correction_type,
                krylov_atol=krylov_atol,
                krylov_rtol=krylov_rtol,
                krylov_max_iters=krylov_max_iters,
            )

            if solver_type == "newton":
                self.solver = InstrumentedNewtonKrylov(
                    precision=config.precision,
                    n=config.n,
                    linear_solver=linear_solver,
                    newton_atol=newton_atol,
                    newton_rtol=newton_rtol,
                    newton_max_iters=newton_max_iters,
                    newton_damping=newton_damping,
                    newton_max_backtracks=newton_max_backtracks,
                )
            else:  # solver_type == 'linear'
                self.solver = linear_solver

            self.register_buffers()
