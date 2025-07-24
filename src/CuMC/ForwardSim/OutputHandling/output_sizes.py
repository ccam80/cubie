import attrs
import numpy as np
from typing import Optional, Tuple
from CuMC.ForwardSim.OutputHandling.output_functions import OutputFunctions
from CuMC.SystemModels.Systems.GenericODE import GenericODE
from CuMC.ForwardSim.integrators.IntegratorRunSettings import IntegatorRunSettings
from CuMC.ForwardSim.integrators.SingleIntegratorRun import SingleIntegratorRun
from numba.cuda import is_cuda_array
from numba.types import Float
from numba import float32
from CuMC.SystemModels.Systems.ODEData import SystemSizes


#TODO: Add a no_zeros toggle to arraysize classes
# Add a _nozeros toggle to all of the dataclasses in output_sizes to give the user the ability to get a minimum-1
# size for use in local or device memory allocation.
# ref: #39

@attrs.define
class SummariesBufferSizes:
    """Given an OutputFunctions object, return the heights of the 1d arrays used to store summary data"""
    _output_functions: OutputFunctions = attrs.field(validator=attrs.validators.instance_of(OutputFunctions))
    state: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observables: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    
    def __attrs_post_init__(self):
        self.state = self._output_functions.state_summaries_output_height
        self.observables = self._output_functions.observable_summaries_output_height


@attrs.define
class InnerLoopBufferSizes:
    """Given an ODE system, return the heights of the 1d arrays used to store non-summary data"""
    _system_sizes: SystemSizes = attrs.field(validator=attrs.validators.instance_of(SystemSizes))
    state: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observables: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    dxdt: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    parameters: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    drivers: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    
    def __attrs_post_init__(self):
        self.state = self._system_sizes.state
        self.observables = self._system_sizes.observables
        self.dxdt = self._system_sizes.states
        self.parameters = self._system_sizes.parameters
        self.drivers = self._system_sizes.drivers


@attrs.define
class LoopBufferSizes:
    """ Given a system and an output functions object, return the heights of the 1d arrays used to store loop
    and summary data inside an integration loop."""
    _system_sizes: SystemSizes = attrs.field(validator=attrs.validators.instance_of(SystemSizes))
    _output_functions: OutputFunctions = attrs.field(validator=attrs.validators.instance_of(OutputFunctions))
    state_summaries: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observable_summaries: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    state: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observables: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    dxdt: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    parameters: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    drivers: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)

    def __attrs_post_init__(self):
        summary_sizes = SummariesBufferSizes(self._output_functions)
        innerloop_sizes = InnerLoopBufferSizes(self._system_sizes)
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
    state: int = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observables: int = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    state_summaries: int = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observable_summaries: int = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)

    def __attrs_post_init__(self):
        self.state = self._output_functions.n_saved_states + 1 * self._output_functions.save_time
        self.observables = self._output_functions.n_saved_observables
        self.state_summaries = self._output_functions.state_summaries_output_height
        self.observable_summaries = self._output_functions.observable_summaries_output_height

@attrs.define
class SingleRunOutputSizes:
    """ Returns 2d single-slice output array sizes for a single integration run, given output functions object and a run
    settings class which has  output_samples and summarise_samples attributes."""
    _output_functions: OutputFunctions = attrs.field(validator=attrs.validators.instance_of(OutputFunctions))
    _run_settings: IntegatorRunSettings = attrs.field(validator=attrs.validators.instance_of(IntegatorRunSettings))
    state: Tuple[int, int] = attrs.field(default=(0,0), validator=attrs.validators.instance_of(Tuple), init=False)
    observables: Tuple[int, int] = attrs.field(default=(0,0), validator=attrs.validators.instance_of(Tuple), init=False)
    state_summaries: Tuple[int, int] = attrs.field(default=(0,0), validator=attrs.validators.instance_of(Tuple),
                                                init=False)
    observable_summaries: Tuple[int, int] = attrs.field(default=(0,0), validator=attrs.validators.instance_of(Tuple),
                                                    init=False)

    def __attrs_post_init__(self):
        heights = OutputArrayHeights(output_functions=self._output_functions)

        output_samples: int = self._run_settings.output_samples
        summarise_samples: int = self._run_settings.summarise_samples
        self.state = (heights.state, output_samples)
        self.observables = (heights.observables, output_samples)
        self.state_summaries = (heights.state_summaries, summarise_samples)
        self.observable_summaries = (heights.observable_summaries, summarise_samples)

class BatchOutputSizes:
    """ Returns 3d output array sizes for a batch of integration runs, given a singleintegrator sizes object and
    num_runs"""
    _single_run_sizes: SingleRunOutputSizes = attrs.field(validator=attrs.validators.instance_of(SingleRunOutputSizes))
    numruns: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
    state: Tuple[int, int, int] = attrs.field(default=(0,0,0), validator=attrs.validators.instance_of(Tuple), init=False)
    observables: Tuple[int, int, int] = attrs.field(default=(0,0,0), validator=attrs.validators.instance_of(Tuple), init=False)
    state_summaries: Tuple[int, int, int] = attrs.field(default=(0,0,0), validator=attrs.validators.instance_of(Tuple),
                                                    init=False)
    observable_summaries: Tuple[int, int, int] = attrs.field(default=(0,0,0), validator=attrs.validators.instance_of(Tuple),
                                                    init=False)

    def __attrs_post_init__(self):
        self.state = (self._single_run_sizes.state[0], self.numruns, self._single_run_sizes.state[1])
        self.observables = (self._single_run_sizes.observables[0], self.numruns, self._single_run_sizes.observables[1])
        self.state_summaries = (self._single_run_sizes.state_summaries[0], self.numruns, self._single_run_sizes.state_summaries[1])
        self.observable_summaries = (self._single_run_sizes.observable_summaries[0], self.numruns,
                                                    self._single_run_sizes.observable_summaries[1])


def cuda_3d_array_validator(instance, attribute, value):
    return is_cuda_array(value) and len(value.shape) == 3

class BatchArrays:
    """ Allocates pinned and mapped output arrays for a batch of integration runs, caching them in case of a
    consecutive run with the same sizes"""
    _sizes: BatchOutputSizes = attrs.field(validator=attrs.validators.instance_of(BatchOutputSizes))
    _precision: Float = attrs.field(default=float32, validator=attrs.validators.instance_of(Float))
    state = None
    observables = None
    state_summaries = None
    observable_summaries = None

    @classmethod
    def from_output_functions_and_run_settings(cls,
                                               output_functions: OutputFunctions,
                                               run_settings: IntegatorRunSettings
                                               ,numruns: int=1):
        """
        Create a BatchArrays instance from an OutputFunctions object and an IntegatorRunSettings object.
        This is useful for creating the batch arrays for a batch of runs with the same output functions and run settings.
        """
        single_run_sizes = SingleRunOutputSizes(output_functions=output_functions, run_settings=run_settings)
        batch_sizes = BatchOutputSizes(single_run_sizes=single_run_sizes, numruns=numruns)
        return cls(sizes=batch_sizes, precision=run_settings.precision)

    #TODO: Add adapters to array size classes from single run and solver classes.
    #  The single integrator run sizes and up have been implemented before the classes were refactored to separate
    #  data and build, and so their interface is not yet fixed. It makes more sense to accept whatever interface they
    #  find and add an adapter in these modules, rather than try to guess how it will look, because my crystal ball
    #  often fails me.

    def _allocate_new(self):
        self.state = np.zeros(self._sizes.state, dtype=self._precision)
        self.observables = np.zeros(self._sizes.observables, dtype=self._precision)
        self.state_summaries = np.zeros(self._sizes.state_summaries, dtype=self._precision)
        self.observable_summaries = np.zeros(self._sizes.observable_summaries, dtype=self._precision)

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
            self._allocate_new()
