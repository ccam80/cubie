"""Norm computation factories for tolerance-scaled convergence checks.

This module provides CUDAFactory subclasses that compile device functions
for computing scaled norms used in convergence testing across integrators
and matrix-free solvers.
"""

from typing import Callable

from numpy import asarray, ndarray
from numba import cuda
from attrs import Converter, define, field

from cubie._utils import (
    PrecisionDType,
    build_config,
    float_array_validator,
    getype_validator,
    is_device_validator,
    tol_converter,
)
from cubie.CUDAFactory import (
    CUDADispatcherCache,
    MultipleInstanceCUDAFactoryConfig,
    MultipleInstanceCUDAFactory,
)
from cubie.cuda_simsafe import compile_kwargs


@define
class ScaledNormConfig(MultipleInstanceCUDAFactoryConfig):
    """Configuration for ScaledNorm factory compilation.

    Attributes
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    n : int
        Size of vectors to compute norm over.
    atol : ndarray
        Absolute tolerance array of shape (n,).
    rtol : ndarray
        Relative tolerance array of shape (n,).
    """

    n: int = field(default=0, validator=getype_validator(int, 1))
    atol: ndarray = field(
        default=asarray([1e-6]),
        validator=float_array_validator,
        converter=Converter(tol_converter, takes_self=True),
        metadata={"prefixed": True},
    )
    rtol: ndarray = field(
        default=asarray([1e-6]),
        validator=float_array_validator,
        converter=Converter(tol_converter, takes_self=True),
        metadata={"prefixed": True},
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def inv_n(self) -> float:
        """Return precomputed 1/n in configured precision."""
        return self.precision(1.0 / self.n)

    @property
    def tol_floor(self) -> float:
        """Return minimum tolerance floor to avoid division by zero."""
        return self.precision(1e-16)


@define
class ScaledNormCache(CUDADispatcherCache):
    """Cache container for ScaledNorm outputs.

    Attributes
    ----------
    scaled_norm : Callable
        Compiled CUDA device function computing scaled norm squared.
    """

    scaled_norm: Callable = field(validator=is_device_validator)


class ScaledNorm(MultipleInstanceCUDAFactory):
    """Factory for scaled norm device functions.

    Compiles a CUDA device function that computes the mean squared
    scaled error norm, where each element's contribution is weighted
    by a tolerance computed from absolute and relative tolerance
    arrays.

    The returned norm value is the mean of squared ratios:
        sum((|values[i]| / tol_i)^2) / n
    where tol_i = max(atol[i] + rtol[i] * |reference[i]|, floor).

    Convergence is achieved when the norm <= 1.0.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        instance_type: str = "",
        **kwargs,
    ) -> None:
        """Initialize ScaledNorm factory.

        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for computations.
        n : int
            Size of vectors to compute norm over.
        **kwargs
            Optional parameters passed to ScaledNormConfig including
            atol and rtol. None values are ignored.
        """
        super().__init__(instance_type=instance_type)

        config = build_config(  # Need to get init_from_prefixed into here
            # somehow.
            ScaledNormConfig,
            required={
                "precision": precision,
                "n": n,
                "instance_type": instance_type,
            },
            **kwargs,
        )

        self.setup_compile_settings(config)

    def build(self) -> ScaledNormCache:
        """Compile scaled norm device function.

        Returns
        -------
        ScaledNormCache
            Container with compiled scaled_norm device function.
        """
        config = self.compile_settings

        n = config.n
        atol = config.atol
        rtol = config.rtol
        numba_precision = config.numba_precision
        inv_n = config.inv_n
        tol_floor = config.tol_floor

        typed_zero = numba_precision(0.0)
        n_val = n

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **compile_kwargs,
        )
        def scaled_norm(values, reference):
            """Compute mean squared scaled error norm.

            Parameters
            ----------
            values : array
                Error or residual values to measure.
            reference : array
                Reference state for relative tolerance scaling.

            Returns
            -------
            float
                Mean squared scaled norm. Converged when <= 1.0.
            """
            nrm2 = typed_zero
            for i in range(n_val):
                value_i = values[i]
                ref_i = reference[i]
                abs_ref = ref_i if ref_i >= typed_zero else -ref_i
                tol_i = atol[i] + rtol[i] * abs_ref
                tol_i = tol_i if tol_i > tol_floor else tol_floor
                abs_val = value_i if value_i >= typed_zero else -value_i
                ratio = abs_val / tol_i
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(scaled_norm=scaled_norm)

    def update(self, updates_dict=None, silent=False, **kwargs):
        """Update compile settings and invalidate cache if changed.

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
            Set of recognized parameter names that were updated.
        """
        all_updates = {}
        if updates_dict:
            all_updates.update(updates_dict)
        all_updates.update(kwargs)

        if not all_updates:
            return set()

        return self.update_compile_settings(
            updates_dict=all_updates, silent=silent
        )

    @property
    def device_function(self) -> Callable:
        """Return cached scaled norm device function."""
        return self.get_cached_output("scaled_norm")

    @property
    def precision(self) -> PrecisionDType:
        """Return configured precision."""
        return self.compile_settings.precision

    @property
    def n(self) -> int:
        """Return vector size."""
        return self.compile_settings.n

    @property
    def atol(self) -> ndarray:
        """Return absolute tolerance array."""
        return self.compile_settings.atol

    @property
    def rtol(self) -> ndarray:
        """Return relative tolerance array."""
        return self.compile_settings.rtol
