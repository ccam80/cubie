from warnings import warn

import attrs
import attrs.validators as val
from numba.cuda import mapped_array
from numpy import float32
import numpy as np
from CuMC import summary_metrics
from CuMC.ForwardSim.OutputHandling.output_sizes import BatchOutputSizes
from CuMC.ForwardSim._utils import optional_cuda_array_validator_3d

@attrs.define
class ActiveOutputs:
    state: bool = attrs.field(default=False, validator=val.instance_of(bool))
    observables: bool = attrs.field(default=False, validator=val.instance_of(bool))
    state_summaries: bool = attrs.field(default=False, validator=val.instance_of(bool))
    observable_summaries: bool = attrs.field(default=False, validator=val.instance_of(bool))

    def update_from_outputarrays(self, output_arrays: "OutputArrays"):
        """Update the active outputs based on the provided OutputArrays instance."""
        self.state = output_arrays.state is not None and output_arrays.state.size > 1
        self.observables = output_arrays.observables is not None and output_arrays.observables.size > 1
        self.state_summaries = output_arrays.state_summaries is not None and output_arrays.state_summaries.size > 1
        self.observable_summaries = output_arrays.observable_summaries is not None and output_arrays.observable_summaries.size > 1

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
    _active_outputs: ActiveOutputs = attrs.field(default=ActiveOutputs(), validator=val.instance_of(ActiveOutputs))

    def __call__(self, solver_instance):
        self.update_from_solver(solver_instance)
        self._active_outputs.update_from_outputarrays(self)
        self.allocate()

    def update_from_solver(self, solver_instance: "BatchSolverKernel"):  # noqa: F821
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

    @property
    def active_outputs(self) -> ActiveOutputs:
        """ Check which outputs are requested, treating size-1 arrays as an artefact of the default allocation."""
        self._active_outputs.update_from_outputarrays(self)
        return self._active_outputs

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
                    sizes: BatchOutputSizes,
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
                    precision,
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
                                      self._sizes,
                                      )
        type_match = self._check_type(self.state,
                                      self.observables,
                                      self.state_summaries,
                                      self.observable_summaries,
                                      self._precision,
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
                 UserWarning,
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

    def output_arrays(self) -> dict[str, np.ndarray]:
        """
        Return a dictionary of host-device output arrays

        Returns
        -------
        array_dict: dict[str, np.ndarray]
            A dictionary with keys 'state', 'observables', 'state_summaries', 'observable_summaries',
        """
        return {
            'state': self.state,
            'observables': self.observables,
            'state_summaries': self.state_summaries,
            'observable_summaries': self.observable_summaries
        }
        
    def output_arrays_with_legend(self, solver_instance: "BatchSolverKernel") -> dict:
        """
        Return a dictionary of output arrays with legends attached as SolverResult objects
        
        Args:
            solver_instance: The solver instance containing system and integrator information
            
        Returns:
            A dictionary with keys 'state', 'observables', 'state_summaries', 'observable_summaries',
            'time_domain', and 'summaries', each containing a SolverResult object with the
            corresponding array data and legend attached.
        """
        # Get the legend
        legend_dict = self.legend(solver_instance)
        
        # Create SolverResult objects with legends attached
        result = {}
        
        # Handle state array
        if self.state is not None and self.state.size > 1:
            result['state'] = SolverResult(self.state, legend_dict['state'])
            
        # Handle observables array
        if self.observables is not None and self.observables.size > 1:
            result['observables'] = SolverResult(self.observables, legend_dict['observables'])
            
        # Handle state_summaries array
        if self.state_summaries is not None and self.state_summaries.size > 1:
            result['state_summaries'] = SolverResult(self.state_summaries, legend_dict['state_summaries'])
            
        # Handle observable_summaries array
        if self.observable_summaries is not None and self.observable_summaries.size > 1:
            result['observable_summaries'] = SolverResult(self.observable_summaries, legend_dict['observable_summaries'])
            
        # Handle time_domain array
        time_domain = self.time_domain_array()
        if time_domain.size > 0:
            # Combine state and observables legends
            time_domain_legend = {}
            state_legend = legend_dict['state']
            obs_legend = legend_dict['observables']
            
            # Add state legend entries
            for idx, entry in state_legend.items():
                time_domain_legend[idx] = entry
                
            # Add observables legend entries with offset
            state_size = len(state_legend)
            for idx, entry in obs_legend.items():
                time_domain_legend[idx + state_size] = entry
                
            result['time_domain'] = SolverResult(time_domain, time_domain_legend)
            
        # Handle summaries array
        summaries = self.summaries_array()
        if summaries.size > 0:
            # Combine state_summaries and observable_summaries legends
            summaries_legend = {}
            state_summaries_legend = legend_dict['state_summaries']
            obs_summaries_legend = legend_dict['observable_summaries']

            # Add state_summaries legend entries
            for idx, entry in state_summaries_legend.items():
                summaries_legend[idx] = entry

            # Add observable_summaries legend entries with offset
            state_summaries_size = len(state_summaries_legend)
            for idx, entry in obs_summaries_legend.items():
                summaries_legend[idx + state_summaries_size] = entry

            result['summaries'] = SolverResult(summaries, summaries_legend)

        return result


    @classmethod
    def from_solver(cls, solver_instance: "BatchSolverKernel") -> "OutputArrays":  # noqa: F821
        """
        Create a OutputArrays instance from a solver instance. Does not allocate, just sets up sizes
        """
        sizes = BatchOutputSizes.from_solver(solver_instance)
        return cls(sizes, precision=solver_instance.precision)