"""Configuration helpers for CUDA-based integration loops.

The objects defined here capture shared and local buffer layouts alongside
compile-critical metadata such as precision, save cadence, and device
callbacks. They centralise validation so that loop factories receive
consistent, ready-to-compile settings.
"""
from typing import Callable, MutableMapping, Optional, Union, TYPE_CHECKING

from attrs import define, field, validators
import numba
from numpy import float32

from cubie._utils import (
    PrecisionDType,
    gttype_validator,
    is_device_validator,
    precision_converter,
    precision_validator,
    opt_getype_validator,
    opt_gttype_validator,
)
from cubie.cuda_simsafe import from_dtype as simsafe_dtype
from cubie.outputhandling.output_config import OutputCompileFlags

if TYPE_CHECKING:
    from cubie.integrators.loops.ode_loop import LoopBufferSettings

valid_opt_slice = validators.optional(validators.instance_of(slice))

@define
class LoopLocalIndices:
    """Index layout for persistent local memory buffers.

    Attributes
    ----------
    dt
        Slice pointing to the timestep storage element.
    accept
        Slice pointing to the acceptance flag storage element.
    controller
        Slice covering scratch space reserved for the controller state.
    algorithm
        Slice covering scratch space reserved for the algorithm state.
    loop_end
        Offset of the end of loop-managed storage.
    total_end
        Offset of the end of the persistent local buffer.
    all
        Slice that spans the entire persistent local buffer.
    """

    dt: Optional[slice] = field(default=None, validator=valid_opt_slice)
    accept: Optional[slice] = field(default=None, validator=valid_opt_slice)
    controller: Optional[slice] = field(
        default=None, validator=valid_opt_slice
    )
    algorithm: Optional[slice] = field(default=None, validator=valid_opt_slice)
    loop_end: Optional[int] = field(
        default=None, validator=opt_getype_validator(int, 0)
    )
    total_end: Optional[int] = field(
        default=None, validator=opt_getype_validator(int, 0)
    )
    all: Optional[slice] = field(default=None, validator=valid_opt_slice)

    @classmethod
    def empty(cls) -> "LoopLocalIndices":
        """Build an empty local-memory layout.

        Returns
        -------
        LoopLocalIndices
            Layout with zero-length slices for all buffers.
        """

        zero = slice(0, 0)
        return cls(
            dt=zero,
            accept=zero,
            controller=zero,
            algorithm=zero,
            loop_end=0,
            total_end=0,
            all=slice(None),
        )

    @classmethod
    def from_sizes(
        cls, controller_len: int, algorithm_len: int
    ) -> "LoopLocalIndices":
        """Build index slices from component memory requirements.

        Parameters
        ----------
        controller_len
            Number of persistent elements reserved for the controller.
        algorithm_len
            Number of persistent elements reserved for the algorithm.

        Returns
        -------
        LoopLocalIndices
            Layout sized to cover the requested buffer lengths.
        """

        controller_len = max(int(controller_len), 0)
        algorithm_len = max(int(algorithm_len), 0)

        dt_slice = slice(0, 1)
        accept_slice = slice(1, 2)
        controller_start = accept_slice.stop
        controller_stop = controller_start + controller_len
        controller_slice = slice(controller_start, controller_stop)

        algorithm_start = controller_stop
        algorithm_stop = algorithm_start + algorithm_len
        algorithm_slice = slice(algorithm_start, algorithm_stop)

        return cls(
            dt=dt_slice,
            accept=accept_slice,
            controller=controller_slice,
            algorithm=algorithm_slice,
            loop_end=accept_slice.stop,
            total_end=algorithm_slice.stop,
            all=slice(None),
        )

    @property
    def loop_elements(self) -> int:
        """Return the loop's intrinsic persistent local requirement."""
        return int(self.loop_end or 0)

@define
class LoopSharedIndices:
    """Slice container describing shared-memory buffer layouts.

    This class holds slices for partitioning shared memory among loop
    buffers. Size information should come from BufferSettings, not from
    the slices themselves.

    Attributes
    ----------
    state
        Slice covering the primary state buffer.
    proposed_state
        Slice covering the proposed state buffer.
    observables
        Slice covering observable work buffers.
    proposed_observables
        Slice covering the proposed observable buffer.
    parameters
        Slice covering parameter storage.
    drivers
        Slice covering driver storage.
    proposed_drivers
        Slice covering the proposed driver storage.
    state_summaries
        Slice covering aggregated state summaries.
    observable_summaries
        Slice covering aggregated observable summaries.
    error
        Slice covering the shared error buffer reused by adaptive algorithms.
    counters
        Slice covering the iteration counters buffer.
    proposed_counters
        Slice covering the proposed iteration counters buffer.
    local_end
        Offset of the end of loop-managed shared memory.
    scratch
        Slice covering any remaining shared-memory scratch space.
    all
        Slice that spans the full shared-memory buffer.
    """

    state:  Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    proposed_state: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    observables: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    proposed_observables: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    parameters: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    drivers: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    proposed_drivers: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    state_summaries: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    observable_summaries: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    error: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    counters: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    proposed_counters: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    local_end: Optional[int] = field(
            default=None,
            validator=opt_getype_validator(int, 0)
    )
    scratch: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    all: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )

    def from_dict(
        self,
        indices_dict: MutableMapping[str, Union[slice, tuple[int, ...]]],
    ) -> None:
        """Load indices from a mapping.

        Parameters
        ----------
        indices_dict
            Mapping of attribute names to slices or ``slice`` constructor
            arguments.
        """
        for key, value in indices_dict.items():
            if isinstance(value, slice):
                setattr(self, key, value)
            else:
                setattr(self, key, slice(*value))

        self.local_end = self.scratch.stop

    @property
    def loop_shared_elements(self) -> int:
        """Return the number of shared memory elements."""
        return int(self.local_end or 0)


@define
class ODELoopConfig:
    """Compile-critical settings for an integrator loop.

    Attributes
    ----------
    shared_buffer_indices
        Shared-memory layout describing scratch buffers and outputs.
    local_indices
        Persistent local-memory layout describing private buffers.
    precision
        Precision used for all loop-managed computations.
    compile_flags
        Output configuration governing save and summary cadence.
    _dt_save
        Interval between accepted saves.
    _dt_summarise
        Interval between summary accumulations.
    save_state_fn
        Device function that records state and observable snapshots.
    update_summaries_fn
        Device function that accumulates summary statistics.
    save_summaries_fn
        Device function that writes summary statistics to output buffers.
    step_controller_fn
        Device function that updates the timestep and acceptance flag.
    step_function
        Device function that advances the solution by one tentative step.
    driver_function
        Device function that evaluates driver signals for a given time.
    observables_fn
        Device function that evaluates observables for the current state.
    _dt0
        Initial timestep prior to controller feedback.
    _dt_min
        Minimum allowable timestep.
    _dt_max
        Maximum allowable timestep.
    is_adaptive
        Whether the loop operates with an adaptive controller.
    """

    shared_buffer_indices: LoopSharedIndices = field(
        validator=validators.instance_of(LoopSharedIndices)
    )
    local_indices: LoopLocalIndices = field(
        validator=validators.instance_of(LoopLocalIndices)
    )

    precision: PrecisionDType = field(
        default=float32,
        converter=precision_converter,
        validator=precision_validator,
    )
    compile_flags: OutputCompileFlags = field(
        default=OutputCompileFlags(),
        validator=validators.instance_of(OutputCompileFlags),
    )
    _dt_save: float = field(
        default=0.1,
        validator=gttype_validator(float, 0)
    )
    _dt_summarise: float = field(
        default=1.0,
        validator=gttype_validator(float, 0)
    )
    save_state_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    update_summaries_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    save_summaries_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    step_controller_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    step_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    driver_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    observables_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    _dt0: Optional[float] = field(
        default=0.01,
        validator=opt_gttype_validator(float, 0),
    )
    _dt_min: Optional[float] = field(
        default=0.01,
        validator=opt_gttype_validator(float, 0),
    )
    _dt_max: Optional[float] = field(
        default=0.1,
        validator=opt_gttype_validator(float, 0),
    )
    is_adaptive: Optional[bool] = field(
            default=False,
            validator=validators.optional(validators.instance_of(bool)))
    buffer_settings: Optional["LoopBufferSettings"] = field(
        default=None,
        validator=validators.optional(validators.instance_of(object)),
    )

    @property
    def saves_per_summary(self) -> int:
        """Return the number of saves between summary outputs."""
        return int(self.dt_summarise // self.dt_save)

    @property
    def numba_precision(self) -> type:
        """Return the Numba precision type."""
        return numba.from_dtype(self.precision)

    @property
    def simsafe_precision(self) -> type:
        """Return the simulator safe precision."""
        return simsafe_dtype(self.precision)

    @property
    def dt_save(self) -> float:
        """Return the output save interval."""
        return self.precision(self._dt_save)

    @property
    def dt_summarise(self) -> float:
        """Return the summary interval."""
        return self.precision(self._dt_summarise)

    @property
    def dt0(self) -> float:
        """Return the initial timestep."""
        return self.precision(self._dt0)

    @property
    def dt_min(self) -> float:
        """Return the minimum allowable timestep."""
        return self.precision(self._dt_min)

    @property
    def dt_max(self) -> float:
        """Return the maximum allowable timestep."""
        return self.precision(self._dt_max)

    @property
    def loop_shared_elements(self) -> int:
        """Return the loop's shared-memory contribution."""

        local_end = getattr(self.shared_buffer_indices, "local_end", None)
        return int(local_end or 0)

    @property
    def loop_local_elements(self) -> int:
        """Return the loop's persistent local-memory contribution."""

        return self.local_indices.loop_elements

