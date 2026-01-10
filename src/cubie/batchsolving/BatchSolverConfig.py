"""Compile-time configuration for batch solver kernels."""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import attrs
from attrs import validators as val

from cubie._utils import (
    getype_validator,
    is_device_validator,
)
from cubie.CUDAFactory import CUDAFactoryConfig, _CubieConfigBase
from cubie.outputhandling.output_config import OutputCompileFlags


@attrs.define
class ActiveOutputs(_CubieConfigBase):
    """
    Track which output arrays are configured to be produced.

    This class provides boolean flags indicating which output types are
    enabled according to compile-time configuration flags, for example
    values derived from ``OutputCompileFlags``.

    Parameters
    ----------
    state
        Whether state output is active.
    observables
        Whether observables output is active.
    state_summaries
        Whether state summaries output is active.
    observable_summaries
        Whether observable summaries output is active.
    status_codes
        Whether status code output is active.
    iteration_counters
        Whether iteration counter output is active.
    """

    state: bool = attrs.field(default=False, validator=val.instance_of(bool))
    observables: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    state_summaries: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    observable_summaries: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    status_codes: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    iteration_counters: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )

    @classmethod
    def from_compile_flags(cls, flags: OutputCompileFlags) -> "ActiveOutputs":
        """
        Create ActiveOutputs from compile flags.

        Parameters
        ----------
        flags
            The compile flags determining which outputs are active.

        Returns
        -------
        ActiveOutputs
            Instance with flags derived from compile flags.

        Notes
        -----
        Maps OutputCompileFlags to ActiveOutputs:
        - save_state → state
        - save_observables → observables
        - summarise_state → state_summaries
        - summarise_observables → observable_summaries
        - save_counters → iteration_counters
        - status_codes is always True (always written during execution)
        """
        return cls(
            state=flags.save_state,
            observables=flags.save_observables,
            state_summaries=flags.summarise_state,
            observable_summaries=flags.summarise_observables,
            status_codes=True,
            iteration_counters=flags.save_counters,
        )


@attrs.define
class CacheConfig(_CubieConfigBase):
    """Configuration for disk-based kernel caching.

    This class holds cache-related settings that do NOT affect kernel
    compilation. Changes to these settings should not trigger kernel
    rebuild.

    Parameters
    ----------
    enabled
        Whether disk caching is enabled.
    cache_path
        Directory path for cache files. None uses default location.
    source_stamp
        Tuple of (mtime, size) for cache validation. None disables
        source stamp checking.

    Notes
    -----
    All cache operations (hashing, path generation, file I/O) use pure
    Python and work without CUDA intrinsics. This enables cache testing
    with NUMBA_ENABLE_CUDASIM=1.
    """

    enabled: bool = attrs.field(
        default=False,
        validator=val.instance_of(bool)
    )
    _cache_path: Optional[Path] = attrs.field(
        default=None,
        alias="cache_path",
        validator=attrs.validators.optional(
            attrs.validators.instance_of(Path)
        ),
        converter=attrs.converters.optional(Path),
    )
    source_stamp: Optional[Tuple[float, int]] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(tuple)
        ),
    )

    @property
    def cache_path(self) -> Optional[Path]:
        """Resolved cache directory path."""
        return self._cache_path

    @property
    def cache_directory(self) -> Optional[Path]:
        """Return resolved cache directory or None if disabled."""
        if not self.enabled:
            return None
        return self._cache_path

    @classmethod
    def from_cache_param(
        cls,
        cache: Union[bool, str, Path, None]
    ) -> "CacheConfig":
        """Parse cache parameter into CacheConfig.

        Parameters
        ----------
        cache
            Cache configuration:
            - True: Enable caching with default path
            - False or None: Disable caching
            - str or Path: Enable caching at specified path

        Returns
        -------
        CacheConfig
            Configured cache settings.
        """
        if cache is None or cache is False:
            return cls(enabled=False, cache_path=None)

        if cache is True:
            return cls(enabled=True, cache_path=None)

        # str or Path provided
        cache_path = Path(cache) if isinstance(cache, str) else cache
        return cls(enabled=True, cache_path=cache_path)


@attrs.define
class BatchSolverConfig(CUDAFactoryConfig):
    """Compile-critical settings for the batch solver kernel.

    Attributes
    ----------
    precision
        NumPy floating-point data type used for host and device arrays.
    loop_fn
        CUDA device loop function generated by :class:`SingleIntegratorRun`.
    local_memory_elements
        Number of precision elements required in local memory per run.
    shared_memory_elements
        Number of precision elements required in shared memory per run.
    compile_flags
        Boolean compile-time controls for output features.
    """

    loop_fn: Optional[Callable] = attrs.field(
        default=None,
        validator=attrs.validators.optional(is_device_validator),
        eq=False,
    )
    local_memory_elements: int = attrs.field(
        default=0,
        validator=getype_validator(int, 0),
    )
    shared_memory_elements: int = attrs.field(
        default=0,
        validator=getype_validator(int, 0),
    )
    compile_flags: Optional[OutputCompileFlags] = attrs.field(
        factory=OutputCompileFlags,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(OutputCompileFlags)
        ),
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def active_outputs(self) -> ActiveOutputs:
        """Derive ActiveOutputs from compile_flags."""
        return ActiveOutputs.from_compile_flags(self.compile_flags)
