import attrs
import numpy as np
from typing import Optional, Tuple, Union
from CuMC.ForwardSim.OutputHandling.output_functions import OutputFunctions
from CuMC.SystemModels.Systems.GenericODE import GenericODE
from CuMC.ForwardSim.integrators.IntegratorRunSettings import IntegatorRunSettings
from CuMC.ForwardSim.integrators.SingleIntegratorRun import SingleIntegratorRun
from numba.cuda import is_cuda_array, mapped_array
from numba.types import Float
from numba import float32
from CuMC.SystemModels.Systems.ODEData import SystemSizes
from numba.np.numpy_support import as_dtype as to_np_dtype


def _ensure_nonzero(value: Union[int, Tuple], nozeros: bool) -> Union[int, Tuple]:
    """Helper function to replace zeros with ones if nozeros is True"""
    if not nozeros:
        return value

    if isinstance(value, int):
        return max(1, value)
    elif isinstance(value, tuple):
        return tuple(max(1, v) for v in value)
    else:
        return value


@attrs.define
class SummariesBufferSizes:
    """Given an OutputFunctions object, return the heights of the 1d arrays used to store summary data"""
    _output_functions: OutputFunctions = attrs.field(validator=attrs.validators.instance_of(OutputFunctions))
    _nozeros: bool = attrs.field(default=False, validator=attrs.validators.instance_of(bool))
    state: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observables: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    
    def __attrs_post_init__(self):
        self.state = _ensure_nonzero(self._output_functions.state_summaries_output_height, self._nozeros)
        self.observables = _ensure_nonzero(self._output_functions.observable_summaries_output_height, self._nozeros)


@attrs.define
class InnerLoopBufferSizes:
    """Given an ODE system, return the heights of the 1d arrays used to store non-summary data"""
    _system_sizes: SystemSizes = attrs.field(validator=attrs.validators.instance_of(SystemSizes))
    _nozeros: bool = attrs.field(default=False, validator=attrs.validators.instance_of(bool))
    state: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observables: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    dxdt: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    parameters: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    drivers: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    
    def __attrs_post_init__(self):
        self.state = _ensure_nonzero(self._system_sizes.states, self._nozeros)
        self.observables = _ensure_nonzero(self._system_sizes.observables, self._nozeros)
        self.dxdt = _ensure_nonzero(self._system_sizes.states, self._nozeros)
        self.parameters = _ensure_nonzero(self._system_sizes.parameters, self._nozeros)
        self.drivers = _ensure_nonzero(self._system_sizes.drivers, self._nozeros)


@attrs.define
class LoopBufferSizes:
    """ Given a system and an output functions object, return the heights of the 1d arrays used to store loop
    and summary data inside an integration loop."""
    _system_sizes: SystemSizes = attrs.field(validator=attrs.validators.instance_of(SystemSizes))
    _output_functions: OutputFunctions = attrs.field(validator=attrs.validators.instance_of(OutputFunctions))
    _nozeros: bool = attrs.field(default=False, validator=attrs.validators.instance_of(bool))
    state_summaries: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observable_summaries: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    state: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observables: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    dxdt: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    parameters: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    drivers: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)

    def __attrs_post_init__(self):
        summary_sizes = SummariesBufferSizes(self._output_functions, nozeros=self._nozeros)
        innerloop_sizes = InnerLoopBufferSizes(self._system_sizes, nozeros=self._nozeros)
        self.state_summaries = summary_sizes.state
        self.observable_summaries = summary_sizes.observables
        self.state = innerloop_sizes.state
        self.observables = innerloop_sizes.observables
        self.dxdt = innerloop_sizes.dxdt
        self.parameters = innerloop_sizes.parameters
        self.drivers = innerloop_sizes.drivers


@attrs.define
class OutputArrayHeights:
    _output_functions: OutputFunctions = attrs.field(validator=attrs.validators.instance_of(OutputFunctions))
    _nozeros: bool = attrs.field(default=False, validator=attrs.validators.instance_of(bool))
    state: int = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observables: int = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    state_summaries: int = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observable_summaries: int = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)

    def __attrs_post_init__(self):
        self.state = _ensure_nonzero(self._output_functions.n_saved_states + 1 * self._output_functions.save_time, self._nozeros)
        self.observables = _ensure_nonzero(self._output_functions.n_saved_observables, self._nozeros)
        self.state_summaries = _ensure_nonzero(self._output_functions.state_summaries_output_height, self._nozeros)
        self.observable_summaries = _ensure_nonzero(self._output_functions.observable_summaries_output_height, self._nozeros)

@attrs.define
class SingleRunOutputSizes:
    """ Returns 2d single-slice output array sizes for a single integration run, given output functions object and a run
    settings class which has  output_samples and summarise_samples attributes."""
    _output_functions: OutputFunctions = attrs.field(validator=attrs.validators.instance_of(OutputFunctions))
    _run_settings: IntegatorRunSettings = attrs.field(validator=attrs.validators.instance_of(IntegatorRunSettings))
    _nozeros: bool = attrs.field(default=False, validator=attrs.validators.instance_of(bool))
    state: Tuple[int, int] = attrs.field(default=(0,0), validator=attrs.validators.instance_of(Tuple), init=False)
    observables: Tuple[int, int] = attrs.field(default=(0,0), validator=attrs.validators.instance_of(Tuple), init=False)
    state_summaries: Tuple[int, int] = attrs.field(default=(0,0), validator=attrs.validators.instance_of(Tuple),
                                                init=False)
    observable_summaries: Tuple[int, int] = attrs.field(default=(0,0), validator=attrs.validators.instance_of(Tuple),
                                                    init=False)

    def __attrs_post_init__(self):
        heights = OutputArrayHeights(output_functions=self._output_functions, nozeros=self._nozeros)

        output_samples: int = _ensure_nonzero(self._run_settings.output_samples, self._nozeros)
        summarise_samples: int = _ensure_nonzero(self._run_settings.summarise_samples, self._nozeros)
        self.state = (heights.state, output_samples)
        self.observables = (heights.observables, output_samples)
        self.state_summaries = (heights.state_summaries, summarise_samples)
        self.observable_summaries = (heights.observable_summaries, summarise_samples)

@attrs.define
class BatchOutputSizes:
    """ Returns 3d output array sizes for a batch of integration runs, given a singleintegrator sizes object and
    num_runs"""
    _single_run_sizes: SingleRunOutputSizes = attrs.field(validator=attrs.validators.instance_of(SingleRunOutputSizes))
    numruns: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
    _nozeros: bool = attrs.field(default=False, validator=attrs.validators.instance_of(bool))
    state: Tuple[int, int, int] = attrs.field(default=(0,0,0), validator=attrs.validators.instance_of(Tuple), init=False)
    observables: Tuple[int, int, int] = attrs.field(default=(0,0,0), validator=attrs.validators.instance_of(Tuple), init=False)
    state_summaries: Tuple[int, int, int] = attrs.field(default=(0,0,0), validator=attrs.validators.instance_of(Tuple),
                                                    init=False)
    observable_summaries: Tuple[int, int, int] = attrs.field(default=(0,0,0), validator=attrs.validators.instance_of(Tuple),
                                                    init=False)

    def __attrs_post_init__(self):
        numruns = _ensure_nonzero(self.numruns, self._nozeros)
        self.state = _ensure_nonzero((self._single_run_sizes.state[0], numruns, self._single_run_sizes.state[1]), self._nozeros)
        self.observables = _ensure_nonzero((self._single_run_sizes.observables[0], numruns, self._single_run_sizes.observables[1]), self._nozeros)
        self.state_summaries = _ensure_nonzero((self._single_run_sizes.state_summaries[0], numruns, self._single_run_sizes.state_summaries[1]), self._nozeros)
        self.observable_summaries = _ensure_nonzero((self._single_run_sizes.observable_summaries[0], numruns,
                                                    self._single_run_sizes.observable_summaries[1]), self._nozeros)


def cuda_3d_array_validator(instance, attribute, value):
    return is_cuda_array(value) and len(value.shape) == 3

@attrs.define
class BatchArrays:
    """ Allocates pinned and mapped output arrays for a batch of integration runs, caching them in case of a
    consecutive run with the same sizes"""
    _sizes: BatchOutputSizes = attrs.field(validator=attrs.validators.instance_of(BatchOutputSizes))
    _precision: Float = attrs.field(default=float32, validator=attrs.validators.instance_of(Float))
    state = attrs.field(default=None, validator=attrs.validators.optional(cuda_3d_array_validator))
    observables = attrs.field(default=None, validator=attrs.validators.optional(cuda_3d_array_validator))
    state_summaries = attrs.field(default=None, validator=attrs.validators.optional(cuda_3d_array_validator))
    observable_summaries = attrs.field(default=None, validator=attrs.validators.optional(cuda_3d_array_validator))

    @classmethod
    def from_output_functions_and_run_settings(cls,
                                               output_functions: OutputFunctions,
                                               run_settings: IntegatorRunSettings,
                                               numruns: int=1,
                                               nozeros: bool=False):
        """
        Create a BatchArrays instance from an OutputFunctions object and an IntegatorRunSettings object.
        This is useful for creating the batch arrays for a batch of runs with the same output functions and run settings.
        """
        single_run_sizes = SingleRunOutputSizes(output_functions=output_functions, run_settings=run_settings, nozeros=nozeros)
        batch_sizes = BatchOutputSizes(single_run_sizes=single_run_sizes, numruns=numruns, nozeros=nozeros)
        return cls(sizes=batch_sizes, precision=run_settings.precision)

    #TODO: Add adapters to array size classes from single run and solver classes.
    #  The single integrator run sizes and up have been implemented before the classes were refactored to separate
    #  data and build, and so their interface is not yet fixed. It makes more sense to accept whatever interface they
    #  find and add an adapter in these modules, rather than try to guess how it will look, because my crystal ball
    #  often fails me.

    def _allocate_new(self):
        np_precision = to_np_dtype(self._precision)
        self.state = mapped_array(self._sizes.state, np_precision)
        self.observables = mapped_array(self._sizes.observables, np_precision)
        self.state_summaries = mapped_array(self._sizes.state_summaries, np_precision)
        self.observable_summaries = mapped_array(self._sizes.observable_summaries, np_precision)

    def _clear_cache(self):
        if self.state:
            del self.state
        if self.observables:
            del self.observables
        if self.state_summaries:
            del self.state_summaries
        if self.observable_summaries:
            del self.observable_summaries

    def cache_valid(self, sizes: BatchOutputSizes, precision: Optional[Float] = None):
        """Check if we have cached arrays that are still a match for the given sizes and precision."""
        valid = True
        if self.state is None:
            valid = False
        if precision is not None and precision != self._precision:
            self._precision = precision
            valid = False
        if self.state.shape != sizes.state or \
           self.observables.shape != sizes.observables or \
           self.state_summaries.shape != sizes.state_summaries or \
           self.observable_summaries.shape != sizes.observable_summaries:
            self._sizes = sizes
            valid = False

        return valid

    def allocate(self, sizes, precision=None):
        """
        Allocate the arrays for the batch of runs, using the sizes provided in the BatchOutputSizes object.
        """
        if not self.cache_valid(sizes, precision):
            self._clear_cache()
            self._allocate_new()
