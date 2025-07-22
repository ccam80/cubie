import attrs
import numpy as np
from typing import Optional

@attrs.define
class ArrayHeights:
    """
    Shape information for arrays - array.heights_temp is the number of samples required for the working array,
    and ArrayHeights.output is the number of samples required for the output array. If you provide a boolean
    argument CUDA_allocation_safe = True, then heights return a minimum value of 1, to avoid allocating a size-0
    array."""
    _CUDA_allocation_safe: Optional[bool] = attrs.field(default=False, validator=attrs.validators.instance_of(bool))
    temp: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
    output: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))

    def __attrs_post_init__(self):
        """If these are being generated for the purpose of filling a CUDA allocation slot, set any zero values to 1"""
        if self._CUDA_allocation_safe:
            self.temp = max(1, self.temp)
            self.output = max(1, self.output)



@attrs.define
class OutputArraySizes:
    """
    Class to hold the sizes of output arrays. This is used to ensure that the arrays are allocated with the correct
    sizes for the requested outputs.
    """
    state: ArrayHeights = attrs.field(default=ArrayHeights(temp=0, output=0),
                                      validator=attrs.validators.instance_of(ArrayHeights),
                                      )
    observables: ArrayHeights = attrs.field(default=ArrayHeights(temp=0, output=0),
                                            validator=attrs.validators.instance_of(ArrayHeights),
                                            )
    state_summaries: ArrayHeights = attrs.field(default=ArrayHeights(temp=0, output=0),
                                                validator=attrs.validators.instance_of(ArrayHeights),
                                                )
    observable_summaries: ArrayHeights = attrs.field(default=ArrayHeights(temp=0, output=0),
                                                     validator=attrs.validators.instance_of(ArrayHeights),
                                                     )
    _CUDA_allocation_safe: bool = attrs.field(default=False, validator=attrs.validators.instance_of(bool))

    @classmethod
    def from_sizes(cls,
                   summary_temp_per_var: int,
                   summary_output_per_var: int,
                   n_saved_states: int,
                   n_saved_observables: int,
                   n_summarised_states: int,
                   n_summarised_observables: int,
                   max_states: int,
                   max_observables: int,
                   save_time,
                   CUDA_allocation_safe=False,
                   ):
        return cls(state=ArrayHeights(temp=max_states,
                                      output=(n_saved_states + 1 * save_time),
                                      CUDA_allocation_safe=CUDA_allocation_safe,
                                      ),
                   observables=ArrayHeights(temp=max_observables,
                                            output=n_saved_observables,
                                            CUDA_allocation_safe=CUDA_allocation_safe,
                                            ),
                   state_summaries=ArrayHeights(temp=summary_temp_per_var * n_summarised_states,
                                                output=summary_output_per_var * n_summarised_states,
                                                CUDA_allocation_safe=CUDA_allocation_safe,
                                                ),
                   observable_summaries=ArrayHeights(temp=summary_temp_per_var * n_summarised_observables,
                                                     output=summary_output_per_var * n_summarised_observables,
                                                     CUDA_allocation_safe=CUDA_allocation_safe,
                                                     ),
                   )


@attrs.define
class OutputArrayDimensions:
    """
    Manages 2D and 3D array dimensions with time and run information.
    """
    array_sizes: OutputArraySizes = attrs.field(validator=attrs.validators.instance_of(OutputArraySizes))
    n_samples: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
    n_summary_samples: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
    numruns: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))

    def get_shapes(self, for_allocation=True):


        """
        Get output array sizes for the current configuration. Call with no arguments for the heights of arrays (
        number of elements per sample), call with n_samples and n_summary_samples to get 2d "slice" shapes,
        and call with numruns as well to get the 3d full run array shapes.

        Args:
            n_samples: int
                Number of time-domain samples. Sets the first dimension of 2d and 3d time-domain arrays
            n_summary_samples: int
                Number of summary samples. Sets the first dimension of 2d and 3d summary arrays
            numruns: int
                Number of runs, used to set the "middle" dimension of 3d arrays
            for_allocation: If you're using this to allocate Memory, return minimum size 1 arrays to avoid breaking
                the memory allocator in numba.cuda.

        Returns:
            Dictionary with array names and their (samples, variables) shapes

        Example:
            '''
            >>> output_sizes = output_functions.get_output_sizes()
            >>> print(output_sizes)
            {
                'state': 5,
                'observables': 3,
                'state_summaries': 4,
                'observable_summaries': 2
            }
            >>> output_sizes = output_functions.get_output_sizes(n_samples=100, n_summary_samples=10)
            >>> print(output_sizes)
            {
                'state': (100, 5),
                'observables': (100, 3),
                'state_summaries': (10, 4),
                'observable_summaries': (10, 2)
            }
            >>> output_sizes = output_functions.get_output_sizes(n_samples=100, n_summary_samples=10, numruns=32)
            >>> print(output_sizes)
            {
                'state': (32, 100, 5),
                'observables': (32, 100, 3),
                'state_summaries': (32, 10, 4),
                'observable_summaries': (32, 10, 2)
            }

            '''
            if for_allocation is true, any shapes featuring a zero will be replaced with a tuple full of ones.

            #TODO: Move this into a kernel-level allocator function.
        """
        sizes = self.array_sizes()
        if n_samples == 0 and n_summary_samples == 0:
            state_shape = sizes.state.output
            observable_shape = sizes.observables.output
            state_summaries_shape = sizes.state_summaries.output
            observable_summaries_shape = sizes.observable_summaries.output
            one_element = 1
        elif numruns == 0:
            state_shape = (n_samples, sizes.state.output)
            observable_shape = (n_samples, sizes.observables.output)
            state_summaries_shape = (n_summary_samples, sizes.state_summaries.output)
            observable_summaries_shape = (n_summary_samples, sizes.observable_summaries.output)
            one_element = (1, 1)
        else:
            state_shape = (n_samples, numruns, sizes.state.output)
            observable_shape = (n_samples, numruns, sizes.observables.output)
            state_summaries_shape = (n_summary_samples, numruns, sizes.state_summaries.output)
            observable_summaries_shape = (n_summary_samples, numruns, sizes.observable_summaries.output)
            one_element = (1, 1, 1)

        array_size_dict = {'state':                state_shape,
                           'observables':          observable_shape,
                           'state_summaries':      state_summaries_shape,
                           'observable_summaries': observable_summaries_shape
                           }

        if for_allocation:
            # Replace any zero dimensions with ones to avoid breaking the memory allocator
            for key, value in array_size_dict.items():
                if 0 in value:
                    array_size_dict[key] = one_element

        return array_size_dict


    @property
    def array_sizes(self):
        return self.compile_settings.get_array_sizes()

    @property
    def nonzero_array_sizes(self):
        return self.compile_settings.get_array_sizes(CUDA_allocation_safe=True)

    def get_array_sizes(self, CUDA_allocation_safe=False) -> OutputArraySizes:
        """Calculate the number of entries required in each array for the requested outputs. Optionally,
        pass CUDA_allocation_safe=True and the function will provide a minimum size of 1, avoiding zero-sized arrays"""

        return OutputArraySizes.from_sizes(self.summary_temp_memory_per_var,
                                           self.summary_output_memory_per_var,
                                           n_saved_states=self.n_saved_states,
                                           n_saved_observables=self.n_saved_observables,
                                           n_summarised_states=self.n_summarised_states,
                                           n_summarised_observables=self.n_summarised_observables,
                                           max_states=self.max_states,
                                           max_observables=self.max_observables,
                                           save_time=self.save_time,
                                           CUDA_allocation_safe=CUDA_allocation_safe,
                                           )