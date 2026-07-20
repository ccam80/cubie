"""CUDA factories for scaled norms."""

from typing import Callable

from numpy import asarray, ndarray, all, full
from cubie.cuda_simsafe import cuda
from attrs import define, field, Converter

from cubie._utils import (
    PrecisionDType,
    build_config,
    getype_validator,
    nonnegative_float_array_validator,
    is_device_validator,
    tol_converter,
)
from cubie.CUDAFactory import (
    CUDADispatcherCache,
    MultipleInstanceCUDAFactoryConfig,
    MultipleInstanceCUDAFactory,
)


def resize_tolerances(instance, attribute, value):
    """Resize tolerance arrays to match configured vector size.

    Parameters
    ----------
    instance : ScaledNormConfig
        Instance of ScaledNormConfig being modified.
    attribute : attrs.Attribute
        Attribute being set (``n``).
    value : int
        New vector size.

    Notes
    -----
    This is only useful (and valid) when the tolerance arrays were set from
    a scalar value. That is the only case where it's safe to assume that the
    user wants the same tolerance applied to all elements. If tolerance is a
    non-equal array then we leave it unchanged, presuming an update to
    tolerance is incoming shortly. If it isn't the consumer will fail,
    as expected.
    """
    n = value
    tols = ("atol", "rtol")
    instance._n_changing = True
    for tol in tols:
        tolarray = getattr(instance, tol)
        if tolarray.shape[0] == n:
            continue
        # If all values are the same, then expand to new size
        if all(tolarray == tolarray[0]):
            setattr(
                instance,
                tol,
                full(n, tolarray[0], dtype=instance.precision),
            )
    instance._n_changing = False
    return value


@define
class ScaledNormConfig(MultipleInstanceCUDAFactoryConfig):
    """Configure a scaled norm.

    Attributes
    ----------
    n : int
        Size of vectors to compute norm over.
    atol : ndarray
        Absolute tolerance array of shape (n,).
    rtol : ndarray
        Relative tolerance array of shape (n,).
    """

    n: int = field(
        default=1,
        validator=getype_validator(int, 1),
        on_setattr=resize_tolerances,
    )
    atol: ndarray = field(
        default=asarray([1e-6]),
        validator=nonnegative_float_array_validator,
        converter=Converter(tol_converter, takes_self=True),
        metadata={"prefixed": True},
    )
    rtol: ndarray = field(
        default=asarray([1e-6]),
        validator=nonnegative_float_array_validator,
        converter=Converter(tol_converter, takes_self=True),
        metadata={"prefixed": True},
    )

    _n_changing: bool = field(default=False, init=False, repr=False, eq=False)

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
class FIRKCorrectionNormConfig(ScaledNormConfig):
    """Configure a coupled FIRK correction norm.

    Attributes
    ----------
    state_n : int
        Number of physical states per stage.
    stage_coefficients : tuple
        Row-major flattened Butcher ``a`` matrix, as produced by
        ``tableau.a_flat``.
    """

    state_n: int = field(default=1, validator=getype_validator(int, 1))
    stage_coefficients: tuple = field(default=(1.0,))

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.n % self.state_n != 0:
            raise ValueError("n must be a multiple of state_n")
        stage_count = self.n // self.state_n
        if len(self.stage_coefficients) != stage_count * stage_count:
            raise ValueError(
                "stage_coefficients must hold stage_count**2 values"
            )

    @property
    def stage_count(self) -> int:
        """Return the number of coupled stages."""
        return self.n // self.state_n


@define
class ScaledNormCache(CUDADispatcherCache):
    """Hold a scaled norm device function."""

    scaled_norm: Callable = field(validator=is_device_validator)


class ScaledNorm(MultipleInstanceCUDAFactory):
    """Compile a mean squared scaled norm."""

    config_type = ScaledNormConfig

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        instance_label: str = "",
        **kwargs,
    ) -> None:
        """Initialize ScaledNorm factory.

        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for computations.
        n : int
            Size of vectors to compute norm over.
        instance_label : str, optional
            Prefix label for parameter names when used as a nested factory.
        **kwargs
            Optional parameters passed to ScaledNormConfig including
            atol and rtol. None values are ignored.
        """
        super().__init__(instance_label=instance_label)

        config = build_config(
            self.config_type,
            required={
                "precision": precision,
                "n": n,
            },
            instance_label=instance_label,
            **kwargs,
        )

        self.setup_compile_settings(config)

    def build(self) -> ScaledNormCache:
        """Compile the whole-vector norm."""
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
            **self.jit_kwargs,
        )
        def scaled_norm(values, reference):
            """Return the mean squared scaled norm."""
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


@define
class TiledScaledNormConfig(ScaledNormConfig):
    """Configure a scaled norm with a stage-tiled reference.

    Attributes
    ----------
    state_n : int
        Number of physical states per stage. The reference vector
        holds one entry per physical state and is reused for every
        stage block of the ``n``-element value vector.
    """

    state_n: int = field(default=1, validator=getype_validator(int, 1))

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.n % self.state_n != 0:
            raise ValueError("n must be a multiple of state_n")


class TiledScaledNorm(ScaledNorm):
    """Compile a scaled norm whose reference tiles across stages.

    Coupled FIRK solves stack ``n = stage_count * state_n`` values,
    but the physical reference vector holds only ``state_n`` entries.
    The compiled function reads the reference entry for value ``i``
    at ``i mod state_n`` so callers pass the single-stage reference
    directly.
    """

    config_type = TiledScaledNormConfig

    def build(self) -> ScaledNormCache:
        """Compile the stage-tiled norm."""
        config = self.compile_settings

        atol = config.atol
        rtol = config.rtol
        numba_precision = config.numba_precision
        inv_n = config.inv_n
        tol_floor = config.tol_floor
        n_val = config.n
        state_n = config.state_n

        typed_zero = numba_precision(0.0)

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def scaled_norm(values, reference):
            """Return the mean squared scaled norm."""
            nrm2 = typed_zero
            for index in range(n_val):
                stage_index = index // state_n
                state_index = index - stage_index * state_n
                ref_i = reference[state_index]
                abs_ref = ref_i if ref_i >= typed_zero else -ref_i
                tol_i = atol[index] + rtol[index] * abs_ref
                tol_i = tol_i if tol_i > tol_floor else tol_floor
                value_i = values[index]
                abs_val = value_i if value_i >= typed_zero else -value_i
                ratio = abs_val / tol_i
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(scaled_norm=scaled_norm)


class CorrectionNorm(ScaledNorm):
    """Base factory for Newton correction norms.

    Correction norms scale the Newton update against the physical
    stage state and the step-start state, matching the reference
    scaling ``atol + rtol * max(|stage_value|, |step_start|)``. The
    compiled function takes ``(values, stage_increment, stage_base,
    step_start, a_ij)`` in place of the two-argument scaled norm.
    """


class DIRKCorrectionNorm(CorrectionNorm):
    """Compile a DIRK correction norm."""

    def build(self) -> ScaledNormCache:
        """Compile the correction norm function."""
        config = self.compile_settings
        atol = config.atol
        rtol = config.rtol
        inv_n = config.inv_n
        tol_floor = config.tol_floor
        numba_precision = config.numba_precision
        n_val = config.n
        typed_zero = numba_precision(0.0)

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def correction_norm(
            values,
            stage_increment,
            stage_base,
            step_start,
            a_ij,
        ):
            """Return the mean squared scaled correction norm."""
            nrm2 = typed_zero
            for i in range(n_val):
                stage_value = (
                    stage_base[i] + a_ij * stage_increment[i]
                )
                reference = max(abs(stage_value), abs(step_start[i]))
                tolerance = atol[i] + rtol[i] * reference
                tolerance = max(tolerance, tol_floor)
                ratio = values[i] / tolerance
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(scaled_norm=correction_norm)


class FIRKCorrectionNorm(CorrectionNorm):
    """Compile a coupled FIRK correction norm."""

    config_type = FIRKCorrectionNormConfig

    def build(self) -> ScaledNormCache:
        """Compile the correction norm function."""
        config = self.compile_settings
        atol = config.atol
        rtol = config.rtol
        inv_n = config.inv_n
        tol_floor = config.tol_floor
        numba_precision = config.numba_precision
        n_val = config.n
        state_n = config.state_n
        stage_count = config.stage_count
        stage_coefficients = tuple(
            numba_precision(value) for value in config.stage_coefficients
        )
        typed_zero = numba_precision(0.0)

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def correction_norm(
            values,
            stage_increment,
            stage_base,
            step_start,
            a_ij,
        ):
            """Return the mean squared scaled correction norm."""
            nrm2 = typed_zero
            for index in range(n_val):
                stage_index = index // state_n
                state_index = index - stage_index * state_n
                stage_value = stage_base[state_index]
                for contribution_index in range(stage_count):
                    coefficient_index = (
                        stage_index * stage_count + contribution_index
                    )
                    increment_index = (
                        contribution_index * state_n + state_index
                    )
                    stage_value += (
                        stage_coefficients[coefficient_index]
                        * stage_increment[increment_index]
                    )

                reference = max(
                    abs(stage_value), abs(step_start[state_index])
                )
                tolerance = atol[index] + rtol[index] * reference
                tolerance = max(tolerance, tol_floor)
                ratio = values[index] / tolerance
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(scaled_norm=correction_norm)
