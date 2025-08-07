# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""

from typing import Optional

from numpy.typing import ArrayLike

from cubie.outputhandling.output_functions import OutputFunctions
from cubie.outputhandling.output_sizes import LoopBufferSizes
from cubie.integrators.IntegratorRunSettings import \
    IntegratorRunSettings
from cubie.integrators.algorithms import ImplementedAlgorithms
from cubie.systemmodels.systems.ODEData import SystemSizes
from cubie._utils import in_attr


class SingleIntegratorRun:
    """ Coordinates the low-level CUDA machinery to create a device function
    that runs a single run of an ODE
    integration. Doesn't compile its own device function, but instead
    performs dependency injection to the integrator
    loop algorithm. Contains light-weight cache management to ensure that a
    change in one subfunction is communicated
    to the others, but does not inherit from CUDAFactory as it performs a
    different role than the others.

    This class presents the interface to lower-level CUDA code.
    Modifications that invalidate the currently compiled
    loop are passed to this class. Namely, those are:

    - Changes to the system constants (the compiled-in parameters, not the
    "parameters", "initial_values",
    or "drivers" which are passed as inputs to the loop function)
    - Changes to the outputs of the loop - specifically adding or removing
    an output type, such as a summary (max,
    min), whether we save time, state, or observables, or which states we
    should save (if we're only saving a subset).
    - Changes to algorithm parameters - things like step size, tolerances,
    or the algorithm itself.

    This class also maintains a list of currently implemented algorithms.
    Select an algorithm by passing a string
    which specifies which algorithm to use.
    Additional algorithms can be added by adding an object that builds the
    loop function given a set of common
    parameters (subclassed from GenericIntegratorAlgorithm, which contains
    instructions, see euler.py for an example).

    This class is not typically exposed to the user directly, and so does
    not have a lot in the way of input
    sanitisation. The user-facing API is the above the Solver class,
    which handles the batching up of runs and
    management of input/output memory.

    All device functions maintain a local cache of their output functions
    and compile-sensitive attributes,
    and will invalidate and rebuild if any of these are updated.
    """

    def __init__(self, system, algorithm: str = 'euler', dt_min: float = 0.01,
                 dt_max: float = 0.1, dt_save: float = 0.1,
                 dt_summarise: float = 1.0, atol: float = 1e-6,
                 rtol: float = 1e-6,
                 saved_state_indices: Optional[ArrayLike] = None,
                 saved_observable_indices: Optional[ArrayLike] = None,
                 summarised_state_indices: Optional[ArrayLike] = None,
                 summarised_observable_indices: Optional[ArrayLike] = None,
                 output_types: list[str] = None, ):

        # Store the system
        self._system = system
        system_sizes = system.sizes

        # Initialize output functions with shapes from system
        self._output_functions = OutputFunctions(
                max_states=system_sizes.states,
                max_observables=system_sizes.observables,
                output_types=output_types,
                saved_state_indices=saved_state_indices,
                saved_observable_indices=saved_observable_indices,
                summarised_state_indices=summarised_state_indices,
                summarised_observable_indices=summarised_observable_indices, )

        compile_settings = IntegratorRunSettings(dt_min=dt_min, dt_max=dt_max,
                                                 dt_save=dt_save,
                                                 dt_summarise=dt_summarise,
                                                 atol=atol, rtol=rtol,
                                                 output_types=output_types,
                                                 # saved_state_indices=saved_state_indices,
                                                 # saved_observable_indices=saved_observable_indices,
                                                 # summarised_state_indices=summarised_state_indices,
                                                 # summarised_observable_indices=summarised_observable_indices,
                                                 )

        self.config = compile_settings

        # Instantiate algorithm with info from system and output functions
        self.algorithm_key = algorithm.lower()
        algorithm = ImplementedAlgorithms[self.algorithm_key]
        self._integrator_instance = algorithm.from_single_integrator_run(self)

        self._compiled_loop = None
        self._loop_cache_valid = False

    @property
    def loop_buffer_sizes(self):
        return LoopBufferSizes.from_system_and_output_fns(self._system,
                                                          self._output_functions)

    @property
    def output_array_heights(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctions.output_array_heights` from the child
        OutputFunctions object."""
        return self._output_functions.output_array_heights

    @property
    def summaries_buffer_sizes(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctions.summaries_buffer_sizes` from the child
        OutputFunctions object."""
        return self._output_functions.summaries_buffer_sizes

    def update(self, updates_dict=None, silent=False, **kwargs):
        """
        Update parameters across all components..

        This method sends all parameters to all child components with
        silent=True
        to avoid spurious warnings, then checks if any parameters were not
        recognized by any component.

        Args:
            update_dict (dict): Dictionary of parameters to update
            silent (bool): If True, suppress warnings about unrecognized
            parameters
            **kwargs: Parameter updates to apply

        Raises:
            ValueError: If no parameters are recognized by any component
        """
        if updates_dict == None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        all_unrecognized = set(updates_dict.keys())
        recognized = set()

        # Update anything held in the config object (step sizes, etc)
        for key, value in updates_dict.items():
            if in_attr(key, self.config):
                setattr(self.config, key, value)
                recognized.add(key)

        if 'algorithm' in updates_dict.keys():
            # If the algorithm is being updated, we need to reset the
            # integrator instance
            self.algorithm_key = updates_dict['algorithm'].lower()
            algorithm = ImplementedAlgorithms[self.algorithm_key]
            self._integrator_instance = algorithm.from_single_integrator_run(
                    self)
            recognized.add('algorithm')

        recognized |= self._system.update(updates_dict, silent=True)
        recognized |= self._output_functions.update(updates_dict, silent=True)

        cached_loop_updates = {'dxdt_function': self.dxdt_function,
            'save_state_func'                 : self.save_state_func,
            'update_summaries_func'           : self.update_summaries_func,
            'save_summaries_func'             : self.save_summaries_func,
            'buffer_sizes'                    : self.loop_buffer_sizes,
            'loop_step_config'                : self.loop_step_config,
            'precision'                       : self.precision,
            'compile_flags'                   : self.compile_flags}

        recognized |= self._integrator_instance.update(cached_loop_updates,
                                                       silent=True)
        all_unrecognized -= recognized

        if all_unrecognized:
            if not silent:
                raise KeyError(
                        f"The following updates were not recognized by any "
                        f"component:"
                        f" {all_unrecognized}", )

        self.config.validate_settings()
        self._invalidate_cache()
        return recognized

    def _invalidate_cache(self):
        """Invalidate the compiled loop cache."""
        self._loop_cache_valid = False
        self._compiled_loop = None

    def build(self):
        """Build the complete integrator loop."""

        # Update with latest function references

        self._compiled_loop = self._integrator_instance.device_function
        self._loop_cache_valid = True

        return self._compiled_loop

    @property
    def device_function(self):
        """Get the compiled loop function, building if necessary."""
        if not self._loop_cache_valid or self._compiled_loop is None:
            self.build()
        return self._compiled_loop

    @property
    def cache_valid(self):
        """Check if the compiled loop is current."""
        return self._loop_cache_valid

    @property
    def shared_memory_elements(self):
        """Get the number of elements of shared memory required for a single
        run of the integrator."""
        if not self.cache_valid:
            self.build()
        loop_memory = self._integrator_instance.shared_memory_required
        summary_buffers = self._output_functions.total_summary_buffer_size
        total_elements = loop_memory + summary_buffers
        if total_elements % 2 == 0:
            total_elements += 1  # Make it odd to reduce bank conflicts
        return (self._integrator_instance.shared_memory_required +
                self._output_functions.total_summary_buffer_size)

    @property
    def shared_memory_bytes(self):
        """Returns the number of bytes of dynamic shared memory required for
        a single run of the integrator"""
        if not self.cache_valid:
            self.build()
        datasize = self.precision(0.0).nbytes
        return self.shared_memory_elements * datasize

    # Reach through this interface class to get lower level features:
    @property
    def precision(self):
        """Exposes :attr:`~cubie.systemmodels.systems.GenericODE.GenericODE
        .precision` from the child GenericODE object."""
        return self._system.precision

    @property
    def threads_per_loop(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.algorithms
        .genericIntegratorAlgorithm.GenericIntegratorAlgorithm
        .threads_per_loop` from the child GenericIntegratorAlgorithm object."""
        return self._integrator_instance.threads_per_loop

    @property
    def dxdt_function(self):
        """Exposes :attr:`~cubie.systemmodels.systems.GenericODE.GenericODE
        .dxdt_function` from the child GenericODE object."""
        return self._system.dxdt_function

    @property
    def save_state_func(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctionCache.save_state_function` from the child
        OutputFunctionCache object."""
        return self._output_functions.save_state_func

    @property
    def update_summaries_func(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctionCache.update_summaries_function` from the child
        OutputFunctionCache object."""
        return self._output_functions.update_summaries_func

    @property
    def save_summaries_func(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctionCache.save_summaries_function` from the child
        OutputFunctionCache object."""
        return self._output_functions.save_summary_metrics_func

    @property
    def loop_step_config(self):
        """Get the loop step configuration."""
        return self.config.loop_step_config

    @property
    def fixed_step_size(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.algorithms
        .genericIntegratorAlgorithm.GenericIntegratorAlgorithm.step_size`
        from the child GenericIntegratorAlgorithm object."""
        return self._integrator_instance.fixed_step_size

    @property
    def dt_save(self):
        """Get the time step size for saving states and observables."""
        return self.config.dt_save

    @property
    def dt_summarise(self):
        """Get the time step size for summarising states and observables."""
        return self.config.dt_summarise

    @property
    def system_sizes(self) -> SystemSizes:
        """Exposes :attr:`~cubie.systemmodels.systems.GenericODE.GenericODE
        .sizes` from the child GenericODE object."""
        return self._system.sizes

    @property
    def compile_flags(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctions.compile_flags` from the child OutputFunctions
        object."""
        return self._output_functions.compile_flags

    @property
    def output_types(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctions.output_types` from the child OutputFunctions
        object."""
        return self._output_functions.output_types

    @property
    def summary_legend_per_variable(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctions.summary_legend_per_variable` from the child
        OutputFunctions object."""
        return self._output_functions.summary_legend_per_variable

    @property
    def saved_state_indices(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctions.saved_state_indices` from the child OutputFunctions
        object."""
        return self._output_functions.saved_state_indices

    @property
    def saved_observable_indices(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctions.saved_observable_indices` from the child
        OutputFunctions object."""
        return self._output_functions.saved_observable_indices

    @property
    def summarised_state_indices(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctions.summarised_state_indices` from the child
        OutputFunctions object."""
        return self._output_functions.summarised_state_indices

    @property
    def summarised_observable_indices(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctions.summarised_observable_indices` from the child
        OutputFunctionCache object."""
        return self._output_functions.summarised_observable_indices

    @property
    def save_time(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions
        .OutputFunctions.save_time` from the child OutputFunctions object."""
        return self._output_functions.save_time

    @property
    def system(self):
        """Exposes child :attr:`~cubie.systemmodels.systems.GenericODE
        .GenericODE` object
        object."""
        return self._system
