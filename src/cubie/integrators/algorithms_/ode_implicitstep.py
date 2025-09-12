from abc import abstractmethod
from typing import Union

import attrs
import numpy as np
import sympy as sp

from cubie.integrators.matrix_free_solvers import (linear_solver_factory,
    newton_krylov_solver_factory)
from cubie.integrators.algorithms_.base_algorithm_step import BaseAlgorithmStep
from cubie.integrators.algorithms_.base_step_config import BaseStepConfig
from cubie.outputhandling import LoopBufferSizes

@attrs.define
class ImplicitStepConfig(BaseStepConfig):
    """Configuration settings for a single integration step.

    Explicit algorithms do not access the full range of fields.
    """
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

    _beta: float = attrs.field(default=1.0)
    _gamma: float = attrs.field(default=1.0)
    _M: Union[np.ndarray, sp.Matrix] = attrs.field(default=sp.eye(1))
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

    @property
    def beta(self):
        if self.style == 'explicit':
            raise NotImplementedError("beta not supported for explicit "
                                      "methods")
        else:
            return self._beta

    @property
    def gamma(self):
        if self.style == 'explicit':
            raise NotImplementedError("gamma not supported for explicit "
                                      "methods")
        else:
            return self._gamma

    @property
    def mass_matrix(self):
        if self.style == 'explicit':
            raise NotImplementedError("mass matrix not supported for explicit "
                                      "methods")
        else:
            return self.operator_M


    def set_operator_fields(self,
                            beta: float,
                            gamma: float,
                            M: Union[np.ndarray, sp.Matrix]) -> None:
        """Set the beta, gamma, and M fields for the linear solver operator.

        The linear operator is of the form (beta * M + a_ij * h * gamma *
        J)(v). This method sets the values of beta, gamma, and M. h, a_ij.
        The remaining parameters are set at runtime and vary between calls.

        Parameters
        ----------
        beta : float
            "Shift" parameter for the linear operator.
        gamma : float


        """
        self._beta = beta
        self._gamma = gamma
        self._M = M


class ODEImplicitStep(BaseAlgorithmStep):
    def __init__(self,
                 config: ImplicitStepConfig):
        super().__init__(config)

    @abstractmethod
    def build_step(self):
        return NotImplementedError

    def build_implicit_helpers(self):
        """Construct the matrix-free solver for implicit methods.

        Constructs a chain of device functions that pieces together the
        matrix-free solvers for implicit methods.

        Returns
        -------
        callable
            Device function that performs the matrix-free solve operation.
        """
        beta = self.compile_settings.beta
        gamma = self.compile_settings.gamma
        mass = self.compile_settings.M
        preconditioner_order = self.compile_settings.preconditioner_order
        multistage = self.compile_settings.multistage_residual_fn
        system = self.system # This coupling isn't great, can we build and
        # pass in?
        n = system.sizes.states

        preconditioner = system.get_solver_helper(
                    'neumann_preconditioner',
                    beta=beta,
                    gamma=gamma,
                    mass=mass,
                    preconditioner_order=preconditioner_order
            )
        if multistage:
            residual = system.get_solver_helper(
                    'stage_residual',
                    beta=beta,
                    gamma=gamma,
                    mass=mass,
                    preconditioner_order=preconditioner_order
            )
        else:
            residual = system.get_solver_helper(
                    'end_residual',
                    beta=beta,
                    gamma=gamma,
                    mass=mass,
                    preconditioner_order=preconditioner_order,
            )
        operator = system.get_solver_helper(
                'linear_operator',
                beta=beta,
                gamma=gamma,
                mass=mass,
                preconditioner_order=preconditioner_order)

        linsolve_tolerance = self.compile_settings.linsolve_tolerance
        max_linear_iters = self.compile_settings.max_linear_iters
        correction_type = self.compile_settings.linear_correction_type

        linear_solver = linear_solver_factory(operator,
                                              n=n,
                                              preconditioner=preconditioner,
                                              correction_type=correction_type,
                                              tolerance=linsolve_tolerance,
                                              max_iters=max_linear_iters)

        nonlinear_tolerance = self.compile_settings.nonlinear_tolerance
        max_newton_iters = self.compile_settings.max_newton_iters
        newton_damping = self.compile_settings.newton_damping
        newton_max_backtracks = self.compile_settings.newton_max_backtracks

        nonlinear_solver = newton_krylov_solver_factory(
                residual_function=residual,
                linear_solver=linear_solver,
                n=n,
                tolerance=nonlinear_tolerance,
                max_iters=max_newton_iters,
                damping=newton_damping,
                max_backtracks=newton_max_backtracks)
        return nonlinear_solver