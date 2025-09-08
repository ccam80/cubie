from typing import Union

import attrs
import numpy as np
import sympy as sp

from cubie.outputhandling import LoopBufferSizes



@attrs.define
class StepConfig:
    """Configuration settings for a single integration step.

    Explicit algorithms do not access the full range of fields.
    """
    n: int = attrs.field(
            default=1,
            validator=attrs.validators.instance_of(int)
    )
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

    style: str = attrs.field(
            default='explicit',
            validator=attrs.validators.in_(['explicit', 'implicit']))
    precision = attrs.field(default=np.float32)
    threads_per_step: int = attrs.field(default=1)

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

    #And so on
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
        self._mass = M