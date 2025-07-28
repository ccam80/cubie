import attrs
from typing import Optional, Tuple, Union

from numba.cuda import is_cuda_array, mapped_array
from numba.types import Float
from numba import float32
from numba.np.numpy_support import as_dtype as to_np_dtype


@attrs.define
class ArraySizingClass:
    """Base class for all array sizing classes. Provides a nonzero method which returns a copy of the object where
    all sizes have a minimum of one element, useful for allocating memory.."""

    @property
    def nonzero(self) -> "ArraySizingClass":
        new_obj = attrs.evolve(self)
        for field in attrs.fields(self.__class__):
            value = getattr(new_obj, field.name)
            if isinstance(value, (int, tuple)):
                setattr(new_obj, field.name, self._ensure_nonzero_size(value))
        return new_obj

    def _ensure_nonzero_size(self, value: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
        """Helper function to replace zeros with ones"""
        if isinstance(value, int):
            return max(1, value)
        elif isinstance(value, tuple):
            if any(v == 0 for v in value):
                return tuple(1 for v in value)
            else:
                return value
        else:
            return value


@attrs.define
class SummariesBufferSizes(ArraySizingClass):
    """Given heights of buffers, return them directly under state and observable aliases. Most useful when called
    with an adapter factory - for example, give it an output_functions object, and it returns sizes without awkward 
    property names from a more cluttered namespace"""
    state: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    observables: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    per_variable: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))

    @classmethod
    def from_output_fns(cls, output_fns: "OutputFunctions") -> "SummariesBufferSizes":
        return cls(output_fns.state_summaries_buffer_height,
                   output_fns.observable_summaries_buffer_height,
                   output_fns.summaries_buffer_height_per_var,
                   )


@attrs.define
class LoopBufferSizes(ArraySizingClass):
    """Dataclass which presents the sizes of all buffers used in the inner loop of an integrator - system-size based
    buffers like state, dxdt and summary buffers derived from output functions information."""
    state_summaries: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    observable_summaries: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    state: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    observables: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    dxdt: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    parameters: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    drivers: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))

    @classmethod
    def from_system_and_output_fns(cls, system: "GenericODE", output_fns: "OutputFunctions") -> "LoopBufferSizes":
        summary_sizes = SummariesBufferSizes.from_output_fns(output_fns)
        system_sizes = system.sizes
        obj = cls(summary_sizes.state,
                  summary_sizes.observables,
                  system_sizes.states,
                  system_sizes.observables,
                  system_sizes.states,
                  system_sizes.parameters,
                  system_sizes.drivers,
                  )
        return obj


@attrs.define
class OutputArrayHeights(ArraySizingClass):
    state: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    observables: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    state_summaries: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    observable_summaries: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    per_variable: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))

    @classmethod
    def from_output_fns(cls, output_fns: "OutputFunctions") -> "OutputArrayHeights":
        state = output_fns.n_saved_states + 1 * output_fns.save_time
        observables = output_fns.n_saved_observables
        state_summaries = output_fns.state_summaries_output_height
        observable_summaries = output_fns.observable_summaries_output_height
        per_variable = output_fns.summaries_output_height_per_var
        obj = cls(state,
                  observables,
                  state_summaries,
                  observable_summaries,
                  per_variable,
                  )
        return obj


@attrs.define
class SingleRunOutputSizes(ArraySizingClass):
    """ Returns 2d single-slice output array sizes for a single integration run."""
    state: Tuple[int, int] = attrs.field(default=(1, 1), validator=attrs.validators.instance_of(Tuple))
    observables: Tuple[int, int] = attrs.field(default=(1, 1), validator=attrs.validators.instance_of(Tuple))
    state_summaries: Tuple[int, int] = attrs.field(default=(1, 1), validator=attrs.validators.instance_of(Tuple))
    observable_summaries: Tuple[int, int] = attrs.field(default=(1, 1), validator=attrs.validators.instance_of(Tuple))

    @classmethod
    def from_output_fns_and_run_settings(cls, output_fns, run_settings):
        heights = OutputArrayHeights.from_output_fns(output_fns)

        state = (run_settings.output_samples, heights.state)
        observables = (run_settings.output_samples, heights.observables)
        state_summaries = (run_settings.summarise_samples, heights.state_summaries)
        observable_summaries = (run_settings.summarise_samples, heights.observable_summaries)
        obj = cls(state,
                  observables,
                  state_summaries,
                  observable_summaries,
                  )

        return obj


@attrs.define
class BatchOutputSizes(ArraySizingClass):
    """ Returns 3d output array sizes for a batch of integration runs, given a singleintegrator sizes object and
    num_runs"""
    state: Tuple[int, int, int] = attrs.field(default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple))
    observables: Tuple[int, int, int] = attrs.field(default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple))
    state_summaries: Tuple[int, int, int] = attrs.field(default=(1, 1, 1),
                                                        validator=attrs.validators.instance_of(Tuple),
                                                        )
    observable_summaries: Tuple[int, int, int] = attrs.field(default=(1, 1, 1),
                                                             validator=attrs.validators.instance_of(Tuple),
                                                             )

    @classmethod
    def from_output_fns_and_run_settings(cls,
                                         output_fns: "OutputFunctions",
                                         run_settings: "IntegratorRunSettings",
                                         numruns: int,
                                         ) -> "BatchOutputSizes":
        """
        Create a BatchOutputSizes instance from a SingleRunOutputSizes object and the number of runs.
        """
        single_run_sizes = SingleRunOutputSizes.from_output_fns_and_run_settings(output_fns, run_settings)
        state = (single_run_sizes.state[0], numruns, single_run_sizes.state[1])
        observables = (single_run_sizes.observables[0], numruns, single_run_sizes.observables[1])
        state_summaries = (single_run_sizes.state_summaries[0], numruns, single_run_sizes.state_summaries[1])
        observable_summaries = (single_run_sizes.observable_summaries[0],
                                numruns,
                                single_run_sizes.observable_summaries[1]
                                )
        obj = cls(state,
                  observables,
                  state_summaries,
                  observable_summaries,
                  )
        return obj


def cuda_3d_array_validator(instance, attribute, value):
    return is_cuda_array(value) and len(value.shape) == 3


@attrs.define
class BatchArrays:
    """ Allocates pinned and mapped output arrays for a batch of integration runs, caching them in case of a
    consecutive run with the same sizes"""
    sizes: BatchOutputSizes = attrs.field(validator=attrs.validators.instance_of(BatchOutputSizes))
    _precision: Float = attrs.field(default=float32, validator=attrs.validators.instance_of(Float))
    state = attrs.field(default=None, validator=attrs.validators.optional(cuda_3d_array_validator))
    observables = attrs.field(default=None, validator=attrs.validators.optional(cuda_3d_array_validator))
    state_summaries = attrs.field(default=None, validator=attrs.validators.optional(cuda_3d_array_validator))
    observable_summaries = attrs.field(default=None, validator=attrs.validators.optional(cuda_3d_array_validator))

    @classmethod
    def from_output_fns_and_run_settings(cls,
                                         output_fns: "OutputFunctions",
                                         run_settings: "IntegratorRunSettings",
                                         numruns: int,
                                         precision: Optional[Float] = float32,
                                         ) -> "BatchArrays":
        """
        Create a BatchArrays instance from a output functions and run settings. Does not allocate, just sets up sizes
        """
        sizes = BatchOutputSizes.from_output_fns_and_run_settings(output_fns, run_settings, numruns)

        return cls(sizes, precision=precision)

    # TODO: Add adapters to array size classes from single run and solver classes.
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

    def initialize_zeros(self, sizes: BatchOutputSizes, precision: Optional[Float] = None):
        """
        Initialize the arrays for the batch of runs, using the sizes provided in the BatchOutputSizes object.
        If the arrays are already allocated and valid, this does nothing.
        """
        if not self.cache_valid(sizes, precision):
            self.allocate(sizes, precision)

        self.state[:, :, :] = precision(0.0)
        self.observables[:, :, :] = precision(0.0)
        self.state_summaries[:, :, :] = precision(0.0)
        self.observable_summaries[:, :, :] = precision(0.0)