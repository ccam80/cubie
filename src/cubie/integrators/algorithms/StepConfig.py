from typing import Union

import attrs
import numpy as np
import sympy as sp

from cubie.outputhandling import LoopBufferSizes



@attrs.define
class StepConfig:
    """Configuration settings for a single integration step.
    """
    style: str = attrs.field(
            default='explicit',
            validator=attrs.validators.in_(['explicit', 'implicit']))
    buffer_sizes: LoopBufferSizes = attrs.field()
    precision = attrs.field(default=np.float32)
    threads_per_step: int = attrs.field(default=1)
    operator_beta: float = attrs.field(default=1.0)
    operator_gamma: float = attrs.field(default=1.0)
    operator_M: Union[np.ndarray, sp.Matrix] = attrs.field(default=sp.eye(1))
    preconditioner_order: int = attrs.field(default=1)
    linsolve_tolerance: float = attrs.field(default=1e-6)
    max_linear_iters: int = attrs.field(default=100)
    linear_correction_type: str = attrs.field(default="minimal_residual")

    nonlinear_tolerance: float = attrs.field(default=1e-6)
    max_newton_iters: int = attrs.field(default=100)
    newton_damping: float = attrs.field(default=0.5)
    newton_max_backtracks: int = attrs.field(default=10)

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
        self.operator_beta = beta
        self.operator_gamma = gamma
        self.operator_M = M