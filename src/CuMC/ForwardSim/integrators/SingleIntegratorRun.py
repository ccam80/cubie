# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""

from typing import Optional

from numpy.typing import ArrayLike

from CuMC.ForwardSim.OutputHandling.output_functions import OutputFunctions
from CuMC.ForwardSim.OutputHandling.output_sizes import LoopBufferSizes
from CuMC.ForwardSim.integrators.IntegratorRunSettings import IntegratorRunSettings
from CuMC.ForwardSim.integrators.algorithms import ImplementedAlgorithms
from CuMC.SystemModels.Systems.ODEData import SystemSizes


class SingleIntegratorRun:
    """ Coordinates the low-level CUDA machinery to create a device function that runs a single run of an ODE
    integration. Doesn't compile its own device function, but instead performs dependency injection to the integrator
    loop algorithm. Contains light-weight cache management to ensure that a change in one subfunction is communicated
    to the others, but does not inherit from CUDAFactory as it performs a different role than the others.

    This class presents the interface to lower-level CUDA code. Modifications that invalidate the currently compiled
    loop are passed to this class. Namely, those are:

    - Changes to the system constants (the compiled-in parameters, not the "parameters", "initial_values",
    or "drivers" which are passed as inputs to the loop function)
    - Changes to the outputs of the loop - specifically adding or removing an output type, such as a summary (max,
    min), whether we save time, state, or observables, or which states we should save (if we're only saving a subset).
    - Changes to algorithm parameters - things like step size, tolerances, or the algorithm itself.

    This class also maintains a list of currently implemented algorithms. Select an algorithm by passing a string
    which specifies which algorithm to use.
    Additional algorithms can be added by adding an object that builds the loop function given a set of common
    parameters (subclassed from GenericIntegratorAlgorithm, which contains instructions, see euler.py for an example).

    This class is not typically exposed to the user directly, and so does not have a lot in the way of input
    sanitisation. The user-facing API is the above the Solver class, which handles the batching up of runs and
    management of input/output memory.

    All device functions maintain a local cache of their output functions and compile-sensitive attributes,
    and will invalidate and rebuild if any of these are updated.
    """

    def __init__(self,
                 system,
                 algorithm: str = 'euler',
                 dt_min: float = 0.01,
                 dt_max: float = 0.1,
                 dt_save: float = 0.1,
                 dt_summarise: float = 1.0,
                 atol: float = 1e-6,
                 rtol: float = 1e-6,
                 saved_states: Optional[ArrayLike] = None,
                 saved_observables: Optional[ArrayLike] = None,
                 summarised_states: Optional[ArrayLike] = None,
                 summarised_observables: Optional[ArrayLike] = None,
                 output_types: list[str] = None,
                 ):

        # Store the system
        self._system = system
        system_sizes = system.sizes

        # Initialize output functions with shapes from system
        self._output_functions = OutputFunctions(
                max_states=system_sizes.states,
                max_observables=system_sizes.observables,
                output_types=output_types,
                saved_states=saved_states,
                saved_observables=saved_observables,
                summarised_states=summarised_states,
                summarised_observables=summarised_observables,
                )

        compile_settings = IntegratorRunSettings(dt_min=dt_min,
                                                 dt_max=dt_max,
                                                 dt_save=dt_save,
                                                 dt_summarise=dt_summarise,
                                                 atol=atol,
                                                 rtol=rtol,
                                                 requested_outputs=output_types,
                                                 saved_states=saved_states,
                                                 saved_observables=saved_observables,
                                                 summarised_states=summarised_states,
                                                 summarised_observables=summarised_observables,
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
        return LoopBufferSizes.from_system_and_output_fns(self._system, self._output_functions)

    @property
    def output_array_heights(self):
        """Get the heights of the output arrays used for saving states and observables."""
        return self._output_functions.output_array_heights

    @property
    def summaries_buffer_sizes(self):
        """Get the sizes of the buffers used for summaries."""
        return self._output_functions.summaries_buffer_sizes

    def update(self, **kwargs):
        """
        Update parameters across all components..

        This method sends all parameters to all child components with silent=True
        to avoid spurious warnings, then checks if any parameters were not
        recognized by any component.

        Args:
            **kwargs: Parameter updates to apply

        Raises:
            ValueError: If no parameters are recognized by any component
        """
        if not kwargs:
            return

        all_unrecognized = set(kwargs.keys())
        # Update anything held in the config object (step sizes, etc)
        recognized = []
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                recognized.append(key)
            if hasattr(self.config.loop_step_config, key):
                setattr(self.config.loop_step_config, key, value)
                recognized.append(key)

        all_unrecognized -= set(recognized)

        unrecognized = self._system.update(silent=True, **kwargs)
        all_unrecognized -= set(kwargs.keys()) - set(unrecognized)

        unrecognized = self._output_functions.update(silent=True, **kwargs)
        all_unrecognized -= set(kwargs.keys()) - set(unrecognized)

        if 'algorithm' in kwargs.keys():
            # If the algorithm is being updated, we need to reset the integrator instance
            self.algorithm_key = kwargs['algorithm'].lower()
            algorithm = ImplementedAlgorithms[self.algorithm_key]
            self._integrator_instance = algorithm.from_single_integrator_run(self)
            all_unrecognized.discard('algorithm')

        # Check if any parameters were unrecognized (indicating an entry error)
        if all_unrecognized:
            unrecognized_list = sorted(all_unrecognized)
            raise KeyError(f"The following updates were not recognized by any component. Was this a typo?:"
                           f" {unrecognized_list}",
                           )

        self._invalidate_cache()

    def _invalidate_cache(self):
        """Invalidate the compiled loop cache."""
        self._loop_cache_valid = False
        self._compiled_loop = None

    def build(self):
        """Build the complete integrator loop."""

        # Update with latest function references
        updates = {
            'dxdt_function':         self.dxdt_function,
            'save_state_func':       self.save_state_func,
            'update_summaries_func': self.update_summaries_func,
            'save_summaries_func':   self.save_summaries_func,
            'buffer_sizes':          self.loop_buffer_sizes,
            'loop_step_config':      self.loop_step_config,
            'precision':             self.precision,
            'compile_flags':         self.compile_flags
            }

        self.config.validate_settings()
        self._integrator_instance.update(**updates)

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

    def _get_dynamic_memory_required(self):
        """Returns the number of bytes of dynamic shared memory required for a single run of the integrator"""
        # Ensure everything is built
        if not self.cache_valid:
            self.build()

        datasize = self.precision(0.0).nbytes
        summary_items = self._output_functions.total_summary_buffer_size
        loop_items = self._integrator_instance.shared_memory_required
        dynamic_sharedmem = (loop_items + summary_items) * datasize
        return dynamic_sharedmem

    @property
    def shared_memory_bytes(self):
        """Get the number of bytes of shared memory required for a single run of the integrator."""
        return self._get_dynamic_memory_required()

    # Reach through this interface class to get lower level features:
    @property
    def precision(self):
        "Numpy-format floating-point datatype from the system model."
        return self._system.precision

    @property
    def threads_per_loop(self):
        """Number of threads per loop iteration."""
        return self._integrator_instance.threads_per_loop

    @property
    def dxdt_function(self):
        """Get the dxdt function used by the integrator."""
        return self._system.dxdt_function

    @property
    def save_state_func(self):
        """Get the save_state function used by the integrator."""
        return self._output_functions.save_state_func

    @property
    def update_summaries_func(self):
        """Get the update_summary_metrics function used by the integrator."""
        return self._output_functions.update_summaries_func

    @property
    def save_summaries_func(self):
        """Get the save_summary_metrics function used by the integrator."""
        return self._output_functions.save_summary_metrics_func

    @property
    def loop_step_config(self):
        """Get the loop step configuration."""
        return self.config.loop_step_config

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
        """Get the sizes of the system."""
        return self._system.sizes

    @property
    def compile_flags(self):
        """Get the compile flags for the output functions."""
        return self._output_functions.compile_flags