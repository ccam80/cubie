"""Configuration helpers for CUDA-based integration loops.

The objects defined here capture shared and local buffer layouts alongside
compile-critical metadata such as precision, save cadence, and device
callbacks. They centralise validation so that loop factories receive
consistent, ready-to-compile settings.
"""
from typing import Callable, Optional
from warnings import warn

from attrs import define, field, validators
from numba import from_dtype as numba_from_dtype
from numpy import float32

from cubie._utils import (
    PrecisionDType,
    getype_validator,
    gttype_validator,
    is_device_validator,
    precision_converter,
    precision_validator,
    opt_gttype_validator,
)
from cubie.cuda_simsafe import from_dtype as simsafe_dtype
from cubie.outputhandling.output_config import OutputCompileFlags

valid_opt_slice = validators.optional(validators.instance_of(slice))

@define
class ODELoopConfig:
    """Compile-critical settings for an integrator loop.

    Attributes
    ----------
    n_states
        Number of state variables.
    n_parameters
        Number of parameters.
    n_drivers
        Number of driver variables.
    n_observables
        Number of observable variables.
    n_error
        Number of error elements (typically equals n_states for adaptive).
    n_counters
        Number of counter elements.
    state_summaries_buffer_height
        Height of state summary buffer.
    observable_summaries_buffer_height
        Height of observable summary buffer.
    controller_local_len
        Number of persistent local memory elements for the controller.
    algorithm_local_len
        Number of persistent local memory elements for the algorithm.
    precision
        Precision used for all loop-managed computations.
    compile_flags
        Output configuration governing save and summary cadence.
    _save_every
        Interval between accepted saves.
    _summarise_every
        Interval between summary accumulations.
    _sample_summaries_every
        Interval between summary metric updates.
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

    # System size parameters
    n_states: int = field(default=0, validator=getype_validator(int, 0))
    n_parameters: int = field(default=0, validator=getype_validator(int, 0))
    n_drivers: int = field(default=0, validator=getype_validator(int, 0))
    n_observables: int = field(default=0, validator=getype_validator(int, 0))
    n_error: int = field(default=0, validator=getype_validator(int, 0))
    n_counters: int = field(default=0, validator=getype_validator(int, 0))

    # Array sizes
    state_summaries_buffer_height: int = field(
        default=0, validator=getype_validator(int, 0)
    )
    observable_summaries_buffer_height: int = field(
        default=0, validator=getype_validator(int, 0)
    )
    controller_local_len: int = field(
        default=0,
        validator=getype_validator(int, 0)
    )
    algorithm_local_len: int = field(
        default=0,
        validator=getype_validator(int, 0)
    )

    # Buffer location settings
    state_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    proposed_state_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    parameters_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    drivers_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    proposed_drivers_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    observables_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    proposed_observables_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    error_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    counters_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    state_summary_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    observable_summary_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    dt_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    accept_step_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    proposed_counters_location: str = field(
        default='local',
        validator=validators.in_(['shared', 'local'])
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

    # Loop timing parameters
    _save_every: Optional[float] = field(
        default=None,
        validator=opt_gttype_validator(float, 0)
    )
    _summarise_every: Optional[float] = field(
        default=None,
        validator=opt_gttype_validator(float, 0)
    )
    _sample_summaries_every: Optional[float] = field(
        default=None,
        validator=opt_gttype_validator(float, 0)
    )
    
    # Deprecated timing parameters (backward compatibility)
    _dt_save: Optional[float] = field(
        default=None,
        validator=opt_gttype_validator(float, 0)
    )
    _dt_summarise: Optional[float] = field(
        default=None,
        validator=opt_gttype_validator(float, 0)
    )
    _dt_update_summaries: Optional[float] = field(
        default=None,
        validator=opt_gttype_validator(float, 0)
    )

    save_state_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
    )
    update_summaries_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
    )
    save_summaries_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
    )
    step_controller_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
    )
    step_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
    )
    driver_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
    )
    observables_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False
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

    def __attrs_post_init__(self):
        """Validate timing parameters and apply backward compatibility.
        
        Handles deprecated parameter names (dt_save, dt_summarise,
        dt_update_summaries) with deprecation warnings. Applies None-handling
        logic for all timing parameters based on configuration context.
        """
        # Backward compatibility: handle deprecated parameter names
        deprecated_used = False
        if self._dt_save is not None:
            if self._save_every is not None:
                raise ValueError(
                    "Cannot specify both 'dt_save' (deprecated) and "
                    "'save_every'. Use 'save_every' only."
                )
            warn(
                "Parameter 'dt_save' is deprecated. Use 'save_every' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            object.__setattr__(self, '_save_every', self._dt_save)
            deprecated_used = True
        
        if self._dt_summarise is not None:
            if self._summarise_every is not None:
                raise ValueError(
                    "Cannot specify both 'dt_summarise' (deprecated) and "
                    "'summarise_every'. Use 'summarise_every' instead."
                )
            warn(
                "Parameter 'dt_summarise' is deprecated. Use 'summarise_every' "
                "instead.",
                DeprecationWarning,
                stacklevel=2
            )
            object.__setattr__(self, '_summarise_every', self._dt_summarise)
            deprecated_used = True
        
        if self._dt_update_summaries is not None:
            if self._sample_summaries_every is not None:
                raise ValueError(
                    "Cannot specify both 'dt_update_summaries' (deprecated) "
                    "and 'sample_summaries_every'. Use 'sample_summaries_every' "
                    "only."
                )
            warn(
                "Parameter 'dt_update_summaries' is deprecated. Use "
                "'sample_summaries_every' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            object.__setattr__(
                self, '_sample_summaries_every', self._dt_update_summaries
            )
            deprecated_used = True
        
        # Apply None-handling logic based on what's configured
        # Case 1: All None (use defaults: save=0.1, summarise=1.0, sample=save)
        if (self._save_every is None and self._summarise_every is None and 
                self._sample_summaries_every is None):
            object.__setattr__(self, '_save_every', 0.1)
            object.__setattr__(self, '_summarise_every', 1.0)
            object.__setattr__(self, '_sample_summaries_every', 0.1)
        
        # Case 2: Only save_every specified
        elif (self._save_every is not None and self._summarise_every is None and
                self._sample_summaries_every is None):
            # summarise_every = 10 * save_every, sample = save
            object.__setattr__(self, '_summarise_every', 10.0 * self._save_every)
            object.__setattr__(self, '_sample_summaries_every', self._save_every)
        
        # Case 3: Only summarise_every specified
        elif (self._save_every is None and self._summarise_every is not None and
                self._sample_summaries_every is None):
            # save_every = summarise_every / 10, sample = save
            object.__setattr__(self, '_save_every', self._summarise_every / 10.0)
            object.__setattr__(
                self, '_sample_summaries_every', self._summarise_every / 10.0
            )
        
        # Case 4: save_every and summarise_every specified
        elif (self._save_every is not None and self._summarise_every is not None and
                self._sample_summaries_every is None):
            # sample = save
            object.__setattr__(self, '_sample_summaries_every', self._save_every)
        
        # Case 5: save_every and sample_summaries_every specified
        elif (self._save_every is not None and self._summarise_every is None and
                self._sample_summaries_every is not None):
            # summarise_every = 10 * save_every
            object.__setattr__(self, '_summarise_every', 10.0 * self._save_every)
        
        # Case 6: summarise_every and sample_summaries_every specified
        elif (self._save_every is None and self._summarise_every is not None and
                self._sample_summaries_every is not None):
            # save_every = sample_summaries_every
            object.__setattr__(self, '_save_every', self._sample_summaries_every)
        
        # Case 7: All three specified - no defaults needed
        # (values already set)
        
        # Validate that sample_summaries_every divides summarise_every evenly
        # Use precision-aware tolerance for float32 compatibility
        tolerance = 1e-6 if self.precision == float32 else 1e-9
        ratio = self._summarise_every / self._sample_summaries_every
        if abs(ratio - round(ratio)) > tolerance:
            raise ValueError(
                f"sample_summaries_every ({self._sample_summaries_every}) must "
                f"be an integer divisor of summarise_every "
                f"({self._summarise_every}). Ratio: {ratio}"
            )

    @property
    def saves_per_summary(self) -> int:
        """Return the number of saves between summary outputs."""
        return round(self.summarise_every / self.save_every)

    @property
    def updates_per_summary(self) -> int:
        """Return the number of updates between summary outputs."""
        return round(self.summarise_every / self.sample_summaries_every)

    @property
    def numba_precision(self) -> type:
        """Return the Numba precision type."""
        return numba_from_dtype(self.precision)

    @property
    def simsafe_precision(self) -> type:
        """Return the simulator safe precision."""
        return simsafe_dtype(self.precision)

    @property
    def save_every(self) -> float:
        """Return the output save interval."""
        return self.precision(self._save_every)
    
    @property
    def summarise_every(self) -> float:
        """Return the summary interval."""
        return self.precision(self._summarise_every)
    
    @property
    def sample_summaries_every(self) -> float:
        """Return the summary sampling interval."""
        return self.precision(self._sample_summaries_every)
    
    # Backward compatibility properties (deprecated)
    @property
    def dt_save(self) -> float:
        """Return the output save interval (deprecated, use save_every)."""
        return self.save_every

    @property
    def dt_summarise(self) -> float:
        """Return the summary interval (deprecated, use summarise_every)."""
        return self.summarise_every

    @property
    def dt_update_summaries(self) -> float:
        """Return the summary update interval (deprecated, use sample_summaries_every)."""
        return self.sample_summaries_every

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



