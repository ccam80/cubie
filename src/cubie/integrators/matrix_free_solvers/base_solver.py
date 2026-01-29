"""Base configuration for matrix-free solver factories.

This module provides shared configuration infrastructure for the
Newton and Krylov solvers in
:mod:`cubie.integrators.matrix_free_solvers`.

Published Classes
-----------------
:class:`MatrixFreeSolverConfig`
    Attrs configuration base for solver factories, extending
    :class:`~cubie.CUDAFactory.MultipleInstanceCUDAFactoryConfig`
    with vector size, iteration limit, and norm device function
    fields.

:class:`MatrixFreeSolver`
    Factory base class managing a :class:`~cubie.integrators.norms.ScaledNorm`
    instance and prefixed tolerance parameter extraction.

See Also
--------
:class:`~cubie.integrators.matrix_free_solvers.linear_solver.LinearSolver`
    Concrete linear solver subclass.
:class:`~cubie.integrators.matrix_free_solvers.newton_krylov.NewtonKrylov`
    Concrete Newton--Krylov solver subclass.
:class:`~cubie.CUDAFactory.MultipleInstanceCUDAFactory`
    Parent factory providing prefixed parameter support.
"""

from typing import Any, Callable, Dict, Optional, Set

from attrs import define, field
from numpy import ndarray

from cubie._utils import (
    PrecisionDType,
    getype_validator,
    inrangetype_validator,
)
from cubie.CUDAFactory import (
    MultipleInstanceCUDAFactory,
    MultipleInstanceCUDAFactoryConfig,
)
from cubie.integrators.norms import ScaledNorm


@define
class MatrixFreeSolverConfig(MultipleInstanceCUDAFactoryConfig):
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

    n: int = field(default=0, validator=getype_validator(int, 1))
    max_iters: int = field(
        default=100,
        validator=inrangetype_validator(int, 1, 32767),
        metadata={"prefixed": True},
    )
    norm_device_function: Optional[Callable] = field(
        default=None,
        eq=False,
    )

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
        **kwargs,
    ) -> None:
        """Initialize base solver with norm factory.

        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for computations.
        solver_type : str
            Prefix for tolerance parameters (e.g., "krylov" or "newton").
        n : int
            Size of state vectors.
        **kwargs
            Forwarded to
            :class:`~cubie.integrators.norms.ScaledNorm`. Includes
            prefixed tolerance parameters (e.g. ``krylov_atol``,
            ``newton_rtol``).
        """
        self.solver_type = solver_type
        super().__init__(instance_label=solver_type)
        self.norm = ScaledNorm(
            precision=precision,
            n=n,
            instance_label=solver_type,
            **kwargs,
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

        recognized |= self.norm.update(all_updates, silent=True)
        all_updates.update({"norm_device_function": self.norm.device_function})
        recognized |= self.update_compile_settings(all_updates, silent=True)

        return recognized

    @property
    def atol(self) -> ndarray:
        """Absolute tolerance for the solver."""
        return self.norm.atol

    @property
    def rtol(self) -> ndarray:
        """Relative tolerance for the solver."""
        return self.norm.rtol

    @property
    def max_iters(self) -> int:
        """Maximum iterations allowed for the solver."""
        return self.compile_settings.max_iters

    @property
    def n(self) -> int:
        """Size of state vectors for the solver."""
        return self.compile_settings.n
