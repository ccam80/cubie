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


    precision = attrs.field(default=np.float32)
    threads_per_step: int = attrs.field(default=1)


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