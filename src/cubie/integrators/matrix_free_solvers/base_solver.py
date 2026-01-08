"""Base configuration for matrix-free solver factories.

This module provides shared configuration infrastructure for the
Newton and Krylov solvers in :mod:`cubie.integrators.matrix_free_solvers`.
"""

from numpy import dtype as np_dtype
from numba import from_dtype
from attrs import define, field

from cubie._utils import (
    PrecisionDType,
    getype_validator,
    precision_converter,
    precision_validator,
)
from cubie.cuda_simsafe import from_dtype as simsafe_dtype


@define
class MatrixFreeSolverConfig:
    """Base configuration for matrix-free solver factories.

    Provides common attributes shared by LinearSolverConfig and
    NewtonKrylovConfig including precision, vector size, and
    Numba/CUDA type accessors.

    Attributes
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    n : int
        Size of state vectors (must be >= 1).
    """

    precision: PrecisionDType = field(
        converter=precision_converter,
        validator=precision_validator
    )
    n: int = field(validator=getype_validator(int, 1))

    @property
    def numba_precision(self) -> type:
        """Return Numba type for precision."""
        return from_dtype(np_dtype(self.precision))

    @property
    def simsafe_precision(self) -> type:
        """Return CUDA-sim-safe type for precision."""
        return simsafe_dtype(np_dtype(self.precision))
