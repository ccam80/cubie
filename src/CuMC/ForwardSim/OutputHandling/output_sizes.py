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
        state = self._output_functions.state_summaries_output_height
        observables = self._output_functions.observable_summaries_output_height


@attrs.define
class InnerLoopBufferSizes:
    """Given an ODE system, return the heights of the 1d arrays used to store non-summary data"""
    _system: GenericODE = attrs.field(validator=attrs.validators.instance_of(GenericODE))
    state: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    observables: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    dxdt: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    parameters: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    drivers: Optional[int] = attrs.field(default=0, validator=attrs.validators.instance_of(int), init=False)
    
    def __attrs_post_init__(self):
        self.state = self._system.sizes.state
        self.observables = self._system.sizes.observables
        self.dxdt = self._system.sizes.states
        self.parameters = self._system.sizes.parameters
        self.drivers = self._system.sizes.drivers


@attrs.define
class LoopBufferSizes:
    """ Given a system and an output functions object, return the heights of the 1d arrays used to store loop
    and summary data inside an integration loop."""
    _system: GenericODE = attrs.field(validator=attrs.validators.instance_of(GenericODE))
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
        innerloop_sizes = InnerLoopBufferSizes(self._system)
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
    precision: Float = attrs.field(default=float32, validator=attrs.validators.instance_of(Float))
    state = None
    observables = None
    state_summaries = None
    observable_summaries = None

    def allocate(self):
        """
        Allocate the arrays for the batch of runs, using the sizes provided in the BatchOutputSizes object.
        """
        if self.state is None:
            self.state = np.zeros(self._sizes.state, dtype=self.precision)
            self.observables = np.zeros(self._sizes.observables, dtype=self.precision)
            self.state_summaries = np.zeros(self._sizes.state_summaries, dtype=self.precision)
            self.observable_summaries = np.zeros(self._sizes.observable_summaries, dtype=self.precision)

        return self



#
#
# @attrs.define
# class OutputArrayDimensions:
#     """
#     Manages 2D and 3D array dimensions with time and run information.
#     """
#     array_sizes: OutputArraySizes = attrs.field(validator=attrs.validators.instance_of(OutputArraySizes))
#     n_samples: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
#     n_summaries_samples: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
#     numruns: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
#
#     def get_shapes(self, for_allocation=True):
#
#
#         """
#         Get output array sizes for the current configuration. Call with no arguments for the heights of arrays (
#         number of elements per sample), call with n_samples and n_summaries_samples to get 2d "slice" shapes,
#         and call with numruns as well to get the 3d full run array shapes.
#
#         Args:
#             n_samples: int
#                 Number of time-domain samples. Sets the first dimension of 2d and 3d time-domain arrays
#             n_summaries_samples: int
#                 Number of summaries samples. Sets the first dimension of 2d and 3d summaries arrays
#             numruns: int
#                 Number of runs, used to set the "middle" dimension of 3d arrays
#             for_allocation: If you're using this to allocate Memory, return minimum size 1 arrays to avoid breaking
#                 the memory allocator in numba.cuda.
#
#         Returns:
#             Dictionary with array names and their (samples, variables) shapes
#
#         Example:
#             '''
#             >>> output_sizes = output_functions.get_output_sizes()
#             >>> print(output_sizes)
#             {
#                 'state': 5,
#                 'observables': 3,
#                 'state_summaries': 4,
#                 'observable_summaries': 2
#             }
#             >>> output_sizes = output_functions.get_output_sizes(n_samples=100, n_summaries_samples=10)
#             >>> print(output_sizes)
#             {
#                 'state': (100, 5),
#                 'observables': (100, 3),
#                 'state_summaries': (10, 4),
#                 'observable_summaries': (10, 2)
#             }
#             >>> output_sizes = output_functions.get_output_sizes(n_samples=100, n_summaries_samples=10, numruns=32)
#             >>> print(output_sizes)
#             {
#                 'state': (32, 100, 5),
#                 'observables': (32, 100, 3),
#                 'state_summaries': (32, 10, 4),
#                 'observable_summaries': (32, 10, 2)
#             }
#
#             '''
#             if for_allocation is true, any shapes featuring a zero will be replaced with a tuple full of ones.
#
#             #TODO: Move this into a kernel-level allocator function.
#         """
#         sizes = self.array_sizes()
#         if n_samples == 0 and n_summaries_samples == 0:
#             state_shape = sizes.state.output
#             observable_shape = sizes.observables.output
#             state_summaries_shape = sizes.state_summaries.output
#             observable_summaries_shape = sizes.observable_summaries.output
#             one_element = 1
#         elif numruns == 0:
#             state_shape = (n_samples, sizes.state.output)
#             observable_shape = (n_samples, sizes.observables.output)
#             state_summaries_shape = (n_summaries_samples, sizes.state_summaries.output)
#             observable_summaries_shape = (n_summaries_samples, sizes.observable_summaries.output)
#             one_element = (1, 1)
#         else:
#             state_shape = (n_samples, numruns, sizes.state.output)
#             observable_shape = (n_samples, numruns, sizes.observables.output)
#             state_summaries_shape = (n_summaries_samples, numruns, sizes.state_summaries.output)
#             observable_summaries_shape = (n_summaries_samples, numruns, sizes.observable_summaries.output)
#             one_element = (1, 1, 1)
#
#         array_size_dict = {'state':                state_shape,
#                            'observables':          observable_shape,
#                            'state_summaries':      state_summaries_shape,
#                            'observable_summaries': observable_summaries_shape
#                            }
#
#         if for_allocation:
#             # Replace any zero dimensions with ones to avoid breaking the memory allocator
#             for key, value in array_size_dict.items():
#                 if 0 in value:
#                     array_size_dict[key] = one_element
#
#         return array_size_dict
#
#
#     @property
#     def array_sizes(self):
#         return self.compile_settings.get_array_sizes()
#
#     @property
#     def nonzero_array_sizes(self):
#         return self.compile_settings.get_array_sizes(CUDA_allocation_safe=True)
#
#     def get_array_sizes(self, CUDA_allocation_safe=False) -> OutputArraySizes:
#         """Calculate the number of entries required in each array for the requested outputs. Optionally,
#         pass CUDA_allocation_safe=True and the function will provide a minimum size of 1, avoiding zero-sized arrays"""
#
#         return OutputArraySizes.from_sizes(self.summaries_temp_memory_per_var,
#                                            self.summaries_output_memory_per_var,
#                                            n_saved_states=self.n_saved_states,
#                                            n_saved_observables=self.n_saved_observables,
#                                            n_summarised_states=self.n_summarised_states,
#                                            n_summarised_observables=self.n_summarised_observables,
#                                            max_states=self.max_states,
#                                            max_observables=self.max_observables,
#                                            save_time=self.save_time,
#                                            CUDA_allocation_safe=CUDA_allocation_safe,
#                                            )