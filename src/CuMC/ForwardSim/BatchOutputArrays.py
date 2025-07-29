from numba import from_dtype
from numba.cuda import mapped_array
from numba.np.numpy_support import as_dtype as to_np_dtype
import attrs
import attrs.validators as val
from numpy import float32
from CuMC.ForwardSim._utils import optional_cuda_array_validator, optional_cuda_array_validator_3d
from CuMC.ForwardSim.OutputHandling.output_sizes import BatchOutputSizes
from warnings import warn


@attrs.define
class OutputArrays:
    """ Manages batch integration output arrays between the host and device. This class is initialised with a
    BatchOutputSizes instance (which is drawn from a solver instance using the from_solver factory method),
    which sets the allowable 3d array sizes from the ODE system's data and run settings. Once initialised,
    the object can be called with a solver instance to update the expected sizes, check the cache, and allocate if
    required.
    """
    _sizes: BatchOutputSizes = attrs.field(validator=val.instance_of(BatchOutputSizes))
    _precision: type = attrs.field(default=float32, validator=val.instance_of(type))
    state = attrs.field(default=None, validator=val.optional(optional_cuda_array_validator_3d))
    observables = attrs.field(default=None, validator=val.optional(optional_cuda_array_validator_3d))
    state_summaries = attrs.field(default=None, validator=val.optional(optional_cuda_array_validator_3d))
    observable_summaries = attrs.field(default=None, validator=val.optional(optional_cuda_array_validator_3d))

    def __call__(self, solver_instance):
        self.update_from_solver(solver_instance)
        self.allocate()

    def update_from_solver(self, solver_instance: "BatchSolverKernel"):
        """
        Update the sizes and precision of the OutputArrays instance from a solver instance.
        This is useful if the solver instance has changed and we need to update the output arrays accordingly.
        """
        self._sizes = BatchOutputSizes.from_solver(solver_instance)
        self._precision = solver_instance.precision
        self._clear_cache()
        self._allocate_new()

    def _allocate_new(self):
        self.state = mapped_array(self._sizes.state, self._precision)
        self.observables = mapped_array(self._sizes.observables, self._precision)
        self.state_summaries = mapped_array(self._sizes.state_summaries, self._precision)
        self.observable_summaries = mapped_array(self._sizes.observable_summaries, self._precision)

    def _clear_cache(self):
        if self.state is not None:
            del self.state
        if self.observables is not None:
            del self.observables
        if self.state_summaries is not None:
            del self.state_summaries
        if self.observable_summaries is not None:
            del self.observable_summaries

    def _check_dims(self,
                    state,
                    observables,
                    state_summaries,
                    observable_summaries,
                    sizes: BatchOutputSizes
                    ):
        """
        Check dimensions of provided arrays match the expected sizes. Return True if sizes match.
        """
        if any(array is None for array in (state, observables, state_summaries, observable_summaries)):
            return False

        match = True
        if state.shape != sizes.state:
            match = False
        if observables.shape != sizes.observables:
            match = False
        if state_summaries.shape != sizes.state_summaries:
            match = False
        if observable_summaries.shape != sizes.observable_summaries:
            match = False
        return match

    def _check_type(self,
                    state,
                    observables,
                    state_summaries,
                    observable_summaries,
                    precision
                    ):
        """
        Check types of provided arrays match the expected precision. Return True if types match.
        """
        if any(array is None for array in (state, observables, state_summaries, observable_summaries)):
            return False

        match = True

        if precision is not None:
            if state.dtype != precision:
                match = False
            if observables.dtype != precision:
                match = False
            if state_summaries.dtype != precision:
                match = False
            if observable_summaries.dtype != precision:
                match = False
        return match

    def cache_valid(self):
        """
        Check dimensions of cached arrays match the expected sizes.
        Raises ValueError if any of the arrays are not allocated or have incorrect dimensions.
        """
        size_match = self._check_dims(self.state,
                                      self.observables,
                                      self.state_summaries,
                                      self.observable_summaries,
                                      self._sizes
                                      )
        type_match = self._check_type(self.state,
                                      self.observables,
                                      self.state_summaries,
                                      self.observable_summaries,
                                      self._precision
                                      )

        return size_match and type_match

    def check_external_arrays(self, state, observables, state_summaries, observable_summaries):
        """
        Check dimensions and dtype of provided arrays match the expected sizes. Returns True if they all match.
        """
        dims_ok = self._check_dims(state, observables, state_summaries, observable_summaries, self._sizes)
        type_ok = self._check_type(state, observables, state_summaries, observable_summaries, self._precision)
        return dims_ok and type_ok

    def attach(self, state, observables, state_summaries, observable_summaries):
        """
        Attach existing arrays to the BatchArrays instance. This is useful for reusing already allocated arrays.
        """
        if self.check_external_arrays(state, observables, state_summaries, observable_summaries):
            self.state = state
            self.observables = observables
            self.state_summaries = state_summaries
            self.observable_summaries = observable_summaries

        else:
            warn("Provided arrays do not match the expected sizes or types, allocating new ones instead.",
                 UserWarning
                 )
            self._allocate_new()

    def allocate(self):
        """
        Allocate the arrays for the batch of runs, using the sizes provided in the BatchOutputSizes object.
        """
        if not self.cache_valid():
            self._clear_cache()
            self._allocate_new()
            self.initialize_zeros()

    def initialize_zeros(self):
        """
        Initialize the arrays for the batch of runs, using the sizes provided in the BatchOutputSizes object.
        If the arrays are already allocated and valid, this does nothing.
        """
        self.state[:, :, :] = self._precision(0.0)
        self.observables[:, :, :] = self._precision(0.0)
        self.state_summaries[:, :, :] = self._precision(0.0)
        self.observable_summaries[:, :, :] = self._precision(0.0)

    def summary_views(self, solver_instance: "BatchSolverKernel"):  # noqa: F821
        """
        Return views of the summaries arrays for the given solver instance.
        This is useful for passing the summaries to the solver without copying the data.
        """
        # break up into per-summary views by digging out summary indices from the solver instance. Might require
        # getting a list of indices from the summary_metrics or output functions module.

        return (self.state_summaries, self.observable_summaries)

    @classmethod
    def from_solver(cls, solver_instance: "BatchSolverKernel") -> "OutputArrays":  # noqa: F821
        """
        Create a OutputArrays instance from a solver instance. Does not allocate, just sets up sizes
        """
        sizes = BatchOutputSizes.from_solver(solver_instance)
        return cls(sizes, precision=solver_instance.precision)
