"""Base configuration for matrix-free solver factories.

This module provides shared configuration infrastructure for the
Newton and Krylov solvers in :mod:`cubie.integrators.matrix_free_solvers`.
"""

from typing import Any, Callable, Dict, Optional, Set

from attrs import define, field

from cubie._utils import (
    PrecisionDType,
    getype_validator,
    inrangetype_validator,
)
from cubie.CUDAFactory import (
    MultipleInstanceCUDAFactory,
    CUDAFactoryConfig,
)
from cubie.integrators.norms import ScaledNorm


@define
class MatrixFreeSolverConfig(CUDAFactoryConfig):
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

    n: int = field(validator=getype_validator(int, 1))
    max_iters: int = field(
        default=100, validator=inrangetype_validator(int, 1, 32767)
    )
    norm_device_function: Optional[Callable] = field(default=None, eq=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()


class MatrixFreeSolver(MultipleInstanceCUDAFactory):
    """Base factory for matrix-free solver device functions.

    Provides shared infrastructure for tolerance parameter mapping
    and norm factory management. Subclasses set `solver_type`
    to enable automatic mapping of prefixed parameters (e.g.,
    "krylov_atol" -> "atol" for norm updates).

    Attributes
    ----------
    solver_type : str
        Prefix for tolerance parameters (e.g., "krylov_" or "newton_").
        Set by subclasses.
    norm : ScaledNorm
        Factory for scaled norm device function used in convergence checks.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        solver_type: str,
        n: int,
        atol: Optional[Any] = None,
        rtol: Optional[Any] = None,
        max_iters: int = 100,
        **kwargs,
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
        self.solver_type = solver_type
        super().__init__(instance_label=solver_type)

        # Build norm kwargs, filtering None values
        norm_kwargs = {}
        if atol is not None:
            norm_kwargs["atol"] = atol
        if rtol is not None:
            norm_kwargs["rtol"] = rtol

        self.norm = ScaledNorm(
            precision=precision,
            n=n,
            **norm_kwargs,
        )

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs,
    ) -> Set[str]:
        """Update compile settings with tolerance extraction.

        Handles common parameter processing for all matrix-free solvers:
        1. Transforms prefixed keys using inherited transform_prefixed_keys
        2. Extracts atol/rtol from transformed dict for norm factory
        3. Updates norm and propagates device function to config
        4. Forwards remaining parameters to update_compile_settings

        Parameters
        ----------
        updates_dict : dict, optional
            Dictionary of settings to update.
        silent : bool, default False
            If True, suppress warnings about unrecognized keys.
        **kwargs
            Additional settings as keyword arguments.

        Returns
        -------
        set
            Set of recognized parameter names (original prefixed forms).
        """
        # Merge updates into a copy
        all_updates = {}
        if updates_dict:
            all_updates.update(updates_dict)
        all_updates.update(kwargs)

        if not all_updates:
            return set()

        recognized = set()

        # Transform prefixed keys and get mapping
        transformed, key_mapping = self.transform_prefixed_keys(all_updates)

        # Extract tolerance parameters from transformed dict
        norm_updates = {}
        if "atol" in transformed:
            norm_updates["atol"] = transformed.pop("atol")
        if "rtol" in transformed:
            norm_updates["rtol"] = transformed.pop("rtol")

        # Update norm factory and track recognized keys
        if norm_updates:
            self.norm.update(norm_updates, silent=True)
            # Map tolerance keys back to original prefixed forms
            for key in norm_updates:
                if key in key_mapping:
                    recognized.add(key_mapping[key])
                else:
                    recognized.add(key)

        # Propagate norm device function to config
        self._update_norm_and_config({})

        # Forward remaining parameters to compile settings
        if transformed:
            recognized_from_settings = super().update_compile_settings(
                updates_dict=transformed, silent=True
            )
            # Map recognized keys back to original prefixed forms
            for key in recognized_from_settings:
                if key in key_mapping:
                    recognized.add(key_mapping[key])
                else:
                    recognized.add(key)

        return recognized

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
