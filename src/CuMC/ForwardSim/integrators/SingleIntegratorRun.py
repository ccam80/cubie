# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""

import numpy as np
from CuMC.ForwardSim.OutputFunctions.output_functions import OutputFunctions
from CuMC.ForwardSim.integrators.algorithms import Euler
from numpy.typing import NDArray

_INTEGRATION_ALGORITHMS = {"euler": Euler}


class SingleIntegratorRun:
    """ Coordinates the low-low-level CUDA machinery to create a device function that runs a single thread of an ODE
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
                 saved_states: NDArray[np.int_] = None,
                 saved_observables: NDArray[np.int_] = None,
                 output_types: list[str] = None,
                 n_peaks: int = 0,
                 ):
        if output_types is None:
            output_types = ["state"]

        # Store the system (already a CUDAFactory)
        self._system = system

        # Store output function parameters for lazy initialization
        self._output_params = {
            'outputs_list':      output_types,
            'saved_states':      saved_states,
            'saved_observables': saved_observables,
            'n_peaks':           n_peaks,
            }
        self._output_functions = None

        # Store algorithm parameters for lazy initialization
        self.algorithm_key = algorithm.lower()
        self._algorithm_params = self._build_algorithm_params(
                system, dt_min, dt_max, dt_save, dt_summarise, atol, rtol,
                saved_states, saved_observables, output_types,
                )
        self._integrator_instance = None

        self._compiled_loop = None
        self._loop_cache_valid = False

    def _build_algorithm_params(self, system, dt_min, dt_max, dt_save, dt_summarise,
                                atol, rtol, saved_states, saved_observables, output_types,
                                ):
        """Build parameters dict for algorithm initialization."""
        precision = system.get_precision()
        system_sizes = system.get_sizes()

        return {
            'precision':           precision,
            'n_states':            system_sizes['n_states'],
            'n_obs':               system_sizes['n_observables'],
            'n_par':               system_sizes['n_parameters'],
            'n_drivers':           system_sizes['n_drivers'],
            'dt_min':              dt_min,
            'dt_max':              dt_max,
            'dt_save':             dt_save,
            'dt_summarise':        dt_summarise,
            'atol':                atol,
            'rtol':                rtol,
            'save_time':           "time" in output_types,
            'n_saved_states':      len(saved_states) if saved_states is not None else 0,
            'n_saved_observables': len(saved_observables) if saved_observables is not None else 0,
            }

    def _ensure_output_functions(self):
        """Instantiate output_functions if not yet instantiated."""
        if self._output_functions is None:
            self._output_functions = OutputFunctions(**self._output_params)

    def _ensure_integrator_instance(self):
        """Instantiate loop algorithm object if not yet instantiated."""
        if self._integrator_instance is None:
            # Ensure output functions are built first
            self._ensure_output_functions()

            self._algorithm_params.update({
                'dxdt_func':           self._system.device_function,
                'save_state_func':     self._output_functions.save_state_func,
                'update_summary_func': self._output_functions.update_summary_metrics_func,
                'save_summary_func':   self._output_functions.save_summary_metrics_func,
                'summary_temp_memory': self._output_functions.memory_per_summarised_variable['temporary'],
                },
                    )

            self._integrator_instance = _INTEGRATION_ALGORITHMS[self.algorithm_key](
                    **self._algorithm_params,
                    )

    def update_parameters(self, **kwargs):
        """
        General parameter update method that routes parameters to the appropriate components.

        This method accepts any parameters and distributes them to the right components,
        marking those components as needing rebuilding.
        """
        # Track what needs updating
        system_updates = {}
        output_updates = {}
        algorithm_updates = {}

        # Route parameters to appropriate components
        for key, value in kwargs.items():
            if key in self._system.compile_settings['constants'].values_dict:
                system_updates[key] = value
            elif key in self._output_params:
                output_updates[key] = value
            elif key in self._algorithm_params or key == 'algorithm':
                if key == 'algorithm':
                    self.algorithm_key = value.lower()
                    self._integrator_instance = None  # Algorithm change requires full rebuild
                else:
                    algorithm_updates[key] = value

        # Apply updates to components
        if system_updates:
            self._system.update_compile_settings(**system_updates)

        if output_updates:
            self._output_params.update(output_updates)
            if self._output_functions is not None:
                self._output_functions.update(**output_updates)

        if algorithm_updates:
            self._algorithm_params.update(algorithm_updates)
            if self._integrator_instance is not None:
                self._integrator_instance.update(**algorithm_updates)

        # Invalidate our cache if anything changed
        if system_updates or output_updates or algorithm_updates:
            self._invalidate_cache()

    def _invalidate_cache(self):
        """Invalidate the compiled loop cache."""
        self._loop_cache_valid = False
        self._compiled_loop = None

    def build(self):
        """Build the complete integrator loop."""
        # Ensure all components are built and current (calls _ensure_output_functions internally)
        self._ensure_integrator_instance()
        lazy_updates = {'dxdt_func':           self._system.device_function,
                        'save_state_func':     self._output_functions.save_state_func,
                        'update_summary_func': self._output_functions.update_summary_metrics_func,
                        'save_summary_func':   self._output_functions.save_summary_metrics_func,
                        'summary_temp_memory': self._output_functions.memory_per_summarised_variable['temporary'],
                        }

        self._algorithm_params.update(lazy_updates)
        self._integrator_instance.update(**lazy_updates)  # Only update the function parameters

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

    def get_dynamic_memory_required(self):
        """Returns the number of bytes of dynamic shared memory required for a single run of the integrator"""
        # Ensure everything is built
        if not self.cache_valid:
            self.build()

        datasize = int(np.ceil(self._algorithm_params['precision'].bitwidth / 8))

        n_summary_variables = (self._algorithm_params['n_saved_states'] +
                               self._algorithm_params['n_saved_observables'])
        summary_memory_per_variable = self._output_functions.memory_per_summarised_variable['temporary']
        summary_items_total = n_summary_variables * summary_memory_per_variable

        loop_items = self._integrator_instance.get_cached_output('loop_shared_memory')

        dynamic_sharedmem = (loop_items + summary_items_total) * datasize
        return dynamic_sharedmem

    @property
    def shared_memory_bytes(self):
        """Get the number of bytes of shared memory required for a single run of the integrator."""
        return self.get_dynamic_memory_required()

    @property
    def shared_memory_elements(self):
        """Get the number of elements (i.e. values) of shared memory required for a single run of the integrator."""
        return self.get_dynamic_memory_required() // (self.precision.bitwidth // 8)

    #Reach through this interface class to get lower level features:
    @property
    def precision(self):
        "Numba-format floating-point datatype from the system model."
        return self._system.get_precision()

    @property
    def threads_per_loop(self):
        """Number of threads per loop iteration."""
        return self._integrator_instance.threads_per_loop

    @property
    def dt_save(self):
        """Time between saving output samples."""
        return self._integrator_instance.dt_save

    @property
    def dt_summarise(self):
        """Time between summarising output samples."""
        return self._integrator_instance.dt_summarise

    def output_sizes(self, duration):
        """Get the number of samples in the output arrays for a given simulation duration."""
        if not self.cache_valid:
            self.build()

        summaries_per_var = self._output_functions.memory_per_summarised_variable['output']
        n_saved_states = self._integrator_instance.n_saved_states
        n_saved_observables = self._integrator_instance.n_saved_observables
        n_state_summaries = summaries_per_var * n_saved_states
        n_obs_summaries = summaries_per_var * n_saved_observables
        n_output_samples = int(np.ceil(duration / self._integrator_instance.dt_save))
        n_summaries_samples = int(np.ceil(duration / self._integrator_instance.dt_summarise))

        sizes = {'state':                (n_output_samples, n_saved_states + self._output_functions.save_time * 1),
                 'observables':          (n_output_samples, n_saved_observables),
                 'state_summaries':      (n_summaries_samples, n_state_summaries),
                 'observable_summaries': (n_summaries_samples, n_obs_summaries),
                 }

        return sizes