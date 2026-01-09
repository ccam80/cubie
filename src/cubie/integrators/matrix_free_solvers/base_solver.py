"""Base configuration for matrix-free solver factories.

This module provides shared configuration infrastructure for the
Newton and Krylov solvers in :mod:`cubie.integrators.matrix_free_solvers`.
"""

from typing import Any, Callable, Dict, Optional, Set

from numpy import dtype as np_dtype
from numba import from_dtype
from attrs import define, field

from cubie._utils import (
    PrecisionDType,
    getype_validator,
    inrangetype_validator,
    precision_converter,
    precision_validator,
)
from cubie.cuda_simsafe import from_dtype as simsafe_dtype
from cubie.CUDAFactory import CUDAFactory
from cubie.integrators.norms import ScaledNorm


@define
class MatrixFreeSolverConfig:
    """Base configuration for matrix-free solver factories.

    Provides common attributes shared by LinearSolverConfig and
    NewtonKrylovConfig including precision, vector size, iteration
    limits, and Numba/CUDA type accessors.

    Attributes
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    n : int
        Size of state vectors (must be >= 1).
    max_iters : int
        Maximum solver iterations permitted (1 to 32767).
    norm_device_function : Optional[Callable]
        Compiled norm function for convergence checks. Updated when
        norm factory rebuilds; changes invalidate solver cache.
    """

    precision: PrecisionDType = field(
        converter=precision_converter,
        validator=precision_validator
    )
    n: int = field(validator=getype_validator(int, 1))
    max_iters: int = field(
        default=100,
        validator=inrangetype_validator(int, 1, 32767)
    )
    norm_device_function: Optional[Callable] = field(
        default=None,
        eq=False
    )

    @property
    def numba_precision(self) -> type:
        """Return Numba type for precision."""
        return from_dtype(np_dtype(self.precision))

    @property
    def simsafe_precision(self) -> type:
        """Return CUDA-sim-safe type for precision."""
        return simsafe_dtype(np_dtype(self.precision))


class MatrixFreeSolver(CUDAFactory):
    """Base factory for matrix-free solver device functions.

    Provides shared infrastructure for tolerance parameter mapping
    and norm factory management. Subclasses set `settings_prefix`
    to enable automatic mapping of prefixed parameters (e.g.,
    "krylov_atol" -> "atol" for norm updates).

    Attributes
    ----------
    settings_prefix : str
        Prefix for tolerance parameters (e.g., "krylov_" or "newton_").
        Set by subclasses.
    norm : ScaledNorm
        Factory for scaled norm device function used in convergence checks.
    """

    settings_prefix: str = ""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        atol: Optional[Any] = None,
        rtol: Optional[Any] = None,
    ) -> None:
        """Initialize base solver with norm factory.

        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for computations.
        n : int
            Size of state vectors.
        atol : array-like, optional
            Absolute tolerance for scaled norm.
        rtol : array-like, optional
            Relative tolerance for scaled norm.
        """
        super().__init__()

        # Build norm kwargs, filtering None values
        norm_kwargs = {}
        if atol is not None:
            norm_kwargs['atol'] = atol
        if rtol is not None:
            norm_kwargs['rtol'] = rtol

        self.norm = ScaledNorm(
            precision=precision,
            n=n,
            **norm_kwargs,
        )

    def _extract_prefixed_tolerance(
        self,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract and map prefixed tolerance parameters.

        Looks for `{prefix}atol` and `{prefix}rtol` in updates dict,
        removes them, and returns dict with unprefixed keys for norm.

        Parameters
        ----------
        updates : dict
            Updates dictionary (modified in place).

        Returns
        -------
        dict
            Norm updates with unprefixed tolerance keys.
        """
        prefix = self.settings_prefix
        norm_updates = {}

        prefixed_atol = f"{prefix}atol"
        prefixed_rtol = f"{prefix}rtol"

        if prefixed_atol in updates:
            norm_updates['atol'] = updates.pop(prefixed_atol)
        if prefixed_rtol in updates:
            norm_updates['rtol'] = updates.pop(prefixed_rtol)

        return norm_updates

    def _update_norm_and_config(
        self,
        norm_updates: Dict[str, Any],
    ) -> None:
        """Update norm factory and propagate device function to config.

        Parameters
        ----------
        norm_updates : dict
            Tolerance updates for norm factory.
        """
        if norm_updates:
            self.norm.update(norm_updates, silent=True)

        # Update config with current norm device function
        # This triggers cache invalidation if the function changed
        self.update_compile_settings(
            norm_device_function=self.norm.device_function,
            silent=True,
        )
