"""Configuration helpers for CUDA-based integration loops.

The objects defined here capture shared and local buffer layouts alongside
compile-critical metadata such as precision, save cadence, and device
callbacks. They centralise validation so that loop factories receive
consistent, ready-to-compile settings.
"""
from typing import Callable, Optional

from attrs import define, field, validators
import numba
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

    # Size parameters (previously from buffer_settings)
    n_states: int = field(default=0, validator=getype_validator(int, 0))
    n_parameters: int = field(default=0, validator=getype_validator(int, 0))
    n_drivers: int = field(default=0, validator=getype_validator(int, 0))
    n_observables: int = field(default=0, validator=getype_validator(int, 0))
    n_error: int = field(default=0, validator=getype_validator(int, 0))
    n_counters: int = field(default=0, validator=getype_validator(int, 0))
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



