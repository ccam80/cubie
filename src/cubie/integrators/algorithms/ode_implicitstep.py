from abc import abstractmethod
from typing import Union, Callable

import attrs
import numpy as np
import sympy as sp

from cubie._utils import inrangetype_validator
from cubie.integrators.matrix_free_solvers import (
    linear_solver_factory,
    newton_krylov_solver_factory
)
from cubie.integrators.algorithms.base_algorithm_step import (
    BaseAlgorithmStep,
    BaseStepConfig,
    StepCache,
)

@attrs.define
class ImplicitStepConfig(BaseStepConfig):
    """Configuration settings for implicit integration steps."""

    _beta: float = attrs.field(default=1.0)
    _gamma: float = attrs.field(default=1.0)
    M: Union[np.ndarray, sp.Matrix] = attrs.field(default=sp.eye(1))
    preconditioner_order: int = attrs.field(default=1)
    _linsolve_tolerance: float = attrs.field(default=1e-3)
    max_linear_iters: int = attrs.field(
            default=100,
            validator=inrangetype_validator(int, 1, 32767)
    )
    linear_correction_type: str = attrs.field(default="minimal_residual")

    _nonlinear_tolerance: float = attrs.field(default=1e-3)
    max_newton_iters: int = attrs.field(
            default=100,
            validator=inrangetype_validator(int, 1, 32767)
    )
    _newton_damping: float = attrs.field(default=0.5)
    newton_max_backtracks: int = attrs.field(default=10)

    @property
    def beta(self) -> float:
        """returns beta"""
        return self.precision(self._beta)

    @property
    def gamma(self) -> float:
        """returns gamma"""
        return self.precision(self._gamma)

    @property
    def linsolve_tolerance(self) -> float:
        """returns linear solve tolerance"""
        return self.precision(self._linsolve_tolerance)

    @property
    def nonlinear_tolerance(self) -> float:
        """returns nonlinear tolerance"""
        return self.precision(self._nonlinear_tolerance)

    @property
    def newton_damping(self) -> float:
        """returns newton damping"""
        return self.precision(self._newton_damping)


    @property
    def settings_dict(self) -> dict:
        """Returns settings as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update({'beta': self.beta,
                              'gamma': self.gamma,
                              'M': self.M,
                              'preconditioner_order': self.preconditioner_order,
                              'linsolve_tolerance': self.linsolve_tolerance,
                              'max_linear_iters': self.max_linear_iters,
                              'linear_correction_type': self.linear_correction_type,
                              'nonlinear_tolerance': self.nonlinear_tolerance,
                              'max_newton_iters': self.max_newton_iters,
                              'newton_damping': self.newton_damping,
                              'newton_max_backtracks': self.newton_max_backtracks,
                              'get_solver_helper_fn': self.get_solver_helper_fn
                              })
        return settings_dict


class ODEImplicitStep(BaseAlgorithmStep):
    def __init__(self,
                 config: ImplicitStepConfig):
        super().__init__(config)

    def build(self):
        # Build the nonlinear solver chain and pass into concrete step builder
        solver_fn, obs_fn = self.build_implicit_helpers()
        config = self.compile_settings
        dxdt_fn = config.dxdt_function
        numba_precision = config.numba_precision
        n = config.n

        return self.build_step(
            solver_fn,
            dxdt_fn,
            obs_fn,
            numba_precision,
            n,
        )


    @abstractmethod
    def build_step(self,
                   solver_fn: Callable,
                   dxdt_fn: Callable,
                   obs_fn: Callable,
                   numba_precision:  type,
                   n: int) -> StepCache:
        raise NotImplementedError

    def build_implicit_helpers(self) -> Callable:
        """Construct the matrix-free solver for implicit methods.

        Constructs a chain of device functions that pieces together the
        matrix-free solvers for implicit methods.

        Returns
        -------
        callable
            Device function that performs the matrix-free solve operation.
        """
        config = self.compile_settings
        beta = config.beta
        gamma = config.gamma
        mass = config.M
        preconditioner_order = config.preconditioner_order
        n = config.n

        get_fn = config.get_solver_helper_fn
    
        preconditioner = get_fn(
                    'neumann_preconditioner',
                    beta=beta,
                    gamma=gamma,
                    mass=mass,
                    preconditioner_order=preconditioner_order
            )
        if self.is_multistage:
            residual = get_fn(
                    'stage_residual',
                    beta=beta,
                    gamma=gamma,
                    mass=mass,
                    preconditioner_order=preconditioner_order
            )
        else:
            residual = get_fn(
                    'end_residual',
                    beta=beta,
                    gamma=gamma,
                    mass=mass,
                    preconditioner_order=preconditioner_order,
            )
        operator = get_fn(
                'linear_operator',
                beta=beta,
                gamma=gamma,
                mass=mass,
                preconditioner_order=preconditioner_order)

        obs_fn = get_fn('observables')

        linsolve_tolerance = config.linsolve_tolerance
        max_linear_iters = config.max_linear_iters
        correction_type = config.linear_correction_type

        linear_solver = linear_solver_factory(operator,
                                              n=n,
                                              preconditioner=preconditioner,
                                              correction_type=correction_type,
                                              tolerance=linsolve_tolerance,
                                              max_iters=max_linear_iters)

        nonlinear_tolerance = config.nonlinear_tolerance
        max_newton_iters = config.max_newton_iters
        newton_damping = config.newton_damping
        newton_max_backtracks = config.newton_max_backtracks

        nonlinear_solver = newton_krylov_solver_factory(
                residual_function=residual,
                linear_solver=linear_solver,
                n=n,
                tolerance=nonlinear_tolerance,
                max_iters=max_newton_iters,
                damping=newton_damping,
                max_backtracks=newton_max_backtracks)
        return nonlinear_solver, obs_fn
    
    @property
    def is_implicit(self) -> bool:
        return True
