from abc import abstractmethod
from typing import Union, Optional, Callable

import attrs
import numpy as np
import sympy as sp
from attrs import validators
from numba import from_dtype

from cubie._utils import is_device_validator
from cubie.integrators.matrix_free_solvers import (
    linear_solver_factory,
    newton_krylov_solver_factory
)
from cubie.integrators.algorithms_.base_algorithm_step import (
    BaseAlgorithmStep,
    BaseStepConfig,
    StepCache,
)
from cubie.outputhandling.output_sizes import LoopBufferSizes

@attrs.define
class ImplicitStepConfig(BaseStepConfig):
    """Configuration settings for a implicit integration steps."""

    atol: float = attrs.field(
            default=1e-6,
            validator=attrs.validators.instance_of(float)
    )
    rtol: float = attrs.field(
            default=1e-6,
            validator=attrs.validators.instance_of(float)
    )
    buffer_sizes: LoopBufferSizes = attrs.field(
        factory=LoopBufferSizes,
        validator=attrs.validators.instance_of(LoopBufferSizes)
    )
    
    get_solver_helper: Optional[Callable] = attrs.field(
            default=None,
            validator=validators.optional(is_device_validator)
    )


    beta: float = attrs.field(default=1.0)
    gamma: float = attrs.field(default=1.0)
    M: Union[np.ndarray, sp.Matrix] = attrs.field(default=sp.eye(1))
    preconditioner_order: int = attrs.field(default=1)
    linsolve_tolerance: float = attrs.field(default=1e-6)
    max_linear_iters: int = attrs.field(default=100)
    linear_correction_type: str = attrs.field(default="minimal_residual")

    nonlinear_tolerance: float = attrs.field(default=1e-6)
    max_newton_iters: int = attrs.field(default=100)
    newton_damping: float = attrs.field(default=0.5)
    newton_max_backtracks: int = attrs.field(default=10)

    @property
    def is_implicit(self):
        return True


class ODEImplicitStep(BaseAlgorithmStep):
    def __init__(self,
                 config: ImplicitStepConfig):
        super().__init__(config)

    def build(self):
        # Build the nonlinear solver chain and pass into concrete step builder
        solver_fn = self.build_implicit_helpers()
        config = self.compile_settings
        dxdt_fn = config.dxdt_function
        numba_precision = from_dtype(config.precision)
        n = config.n

        return self.build_step(solver_fn,
                               dxdt_fn,
                               numba_precision,
                               n)


    @abstractmethod
    def build_step(self,
                   solver_fn: Callable,
                   dxdt_fn: Callable,
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

        get_fn = config.get_solver_helper
    
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
        return nonlinear_solver
    
