# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""

import numpy as np
# from numba import float32, float64, int32, int16, literally
from numba import cuda
# from warnings import warn
from CuMC.ForwardSim.OutputFunctions.output_functions import OutputFunctions
from CuMC.ForwardSim.integrators.algorithms import Euler
from numpy.typing import NDArray

_INTEGRATION_ALGORITHMS = {"euler": Euler}


class SingleIntegratorRun:
    """ Creates and builds the loop and output functions for a single thread of a CUDA integrator. This is a
    glue class that performs dependency injection for the loop class. The singleIntegratorRun ferries changes to
    compile-sensitive parameters to the lower level machinery, delivering a loop function that can be compiled into a
    kernel.

    This class presents the interface to lower-level CUDA code. Modifications that invalidate the currently compiled
    loop are passed to this class. Namely, those are:

    - Changes to the system constants (the compiled-in parameters, not the "parameters", "initial_values",
    or "drivers" which are passed as inputs to the loop function)
    - Changes to the outputs of the loop - specifically adding or removing an output type, such as a summary (max,
    min), whether we save time, state, or observables, or which states we should save (if we're only saving a subset).

    This class also maintains a list of currently implemented algorithms. Select an algorithm by passing a string
    which specifies which algorithm to use.
    Additional algorithms can be added by adding an object that builds the loop function given a set of common
    parameters (subclassed from GenericIntegratorAlgorithm, which contains instructions, see euler.py for an example).

    This class is not typically exposed to the user directly, and so does not have a lot in the way of input
    sanitisation. The user-facing API is the above the  Solver class, which handles the batching up of runs and
    management of input/output memory. This class contains core functionality for running a single thread of an
    integrator for a
    given (individual) set of initial values, parameters, and drivers. It is separate from the
    genericIntegratorAlgorithm class so that we can insert any algorithm we choose without duplicating
    the surrounding code.

    The device functions organised by this class needs to be rebuilt when there are changes to:

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

        self._system = system

        precision = self._system.precision
        system_sizes = system.get_sizes()

        self.output_settings = {
            'saved_states':      saved_states,
            'saved_observables': saved_observables,
            'output_functions':  output_types,
            'n_peaks':           n_peaks,
            }

        self.algo_compile_settings = {
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
            'summary_temp_memory': 0  # updated once outputs are built
            }

        self.thread_settings = {
            'total_shared_memory':   0,
            'loop_shared_memory':    0,
            'output_summary_length': 0
            }

        self.algorithm_key = algorithm.lower()
        self.integrator_instance = None
        self._single_run_function = None

        self._output_functions = OutputFunctions(**self.output_settings)

    def update_parameters(self, **kwargs):
        """
        General parameter update method that routes parameters to the appropriate components.

        This method accepts any parameters and distributes them to the right components,
        marking those components as needing rebuilding.
        """
        # Track which components need rebuilding
        rebuild_needed = {
            'system':           False,
            'output_functions': False,
            'integrator':       False
            }

        # Process system constants updates
        system_constants = {}
        for key, value in kwargs.items():
            if key in self._system.constants.keys():
                system_constants[key] = value
                rebuild_needed['system'] = True

        if system_constants:
            self._system.set_constants(system_constants)

        # Process output settings updates
        output_updates = {}
        for key, value in kwargs.items():
            if key in self.output_settings:
                output_updates[key] = value
                rebuild_needed['output_functions'] = True

        if output_updates:
            self.output_settings.update(output_updates)

        # Process algorithm settings updates
        algo_updates = {}
        for key, value in kwargs.items():
            if key in self.algo_compile_settings:
                algo_updates[key] = value
                rebuild_needed['integrator'] = True

        if algo_updates:
            self.algo_compile_settings.update(algo_updates)
            self._component_is_current['integrator'] = False

        # Special case for algorithm change
        if 'algorithm' in kwargs:
            self.algorithm_key = kwargs['algorithm'].lower()
            self.integrator_instance = None
            rebuild_needed['integrator'] = True
            self._component_is_current['integrator'] = False

        # If any component needs rebuilding, the loop is no longer current
        if any(rebuild_needed.values()):
            self._component_is_current['loop'] = False

    def _build_output_functions(self):
        """Update the output functions and associated memory parameters"""

        if self._component_is_current.get('output_functions', False):
            pass
        else:
            if self._output_functions is None:
                self._output_functions = OutputFunctions(**self.output_settings)

            self.algo_compile_settings = {'save_state_func':     self.output_functions.save_state_func,
                                          'save_summary_func':   self.output_functions.save_summary_metrics_func,
                                          'update_summary_func': self.output_functions.update_summary_metrics_func,
                                          'save_time':           self.output_functions.save_time,
                                          'summary_temp_memory': self.output_functions.memory_requirements_per_variable[
                                                                     'temporary']
                                          }

            self.thread_settings['output_summary_length'] = self.output_functions.memory_requirements_per_variable['output']
            self._component_is_current['output_functions'] = True



    def get_device_function(self):
        """Get the compiled loop function.

        If the function is not current, rebuild it.

        Returns:
            The compiled loop function
        """
        if not self.is_current:
            self.build()
        return self.loop_function

    def update_and_rebuild_loop(self,
                                dxdt_function: cuda.jit(device=True, inline=True),
                                ):
        #TOD: dxdt as an argument seems wrong. If a dxdt function rebuild was required, then we should do it here (or
        # in the calling function). This is due to the no-one-remembers-their-own-device-function
        #Collect all current settings and functions for delivery to the loop object.
        update_dict = self.algo_compile_settings.copy()
        update_dict['dxdt_function'] = dxdt_function
        update_dict.update({'save_state_func':     self.output_functions.save_state_func,
                            'update_summary_func': self.output_functions.update_summary_metrics_func,
                            'save_summary_func':   self.output_functions.save_summary_metrics_func,
                            },
                           )

        self.is_current = False
        self.integrator_instance.update_settings(update_dict)
        return self.integrator_instance.get_device_function()

    def _instantiate_loop(self, dxdt_function: cuda.jit(device=True, inline=True)):
        """Instantiate the loop function if it's not yet created. If build dxdt and output functions are
        out of date, this loop will go together with old parts, and will need updated before building."""

        #Abort if the integrator is already instantiated and current.
        if self.integrator_instance is not None and self._component_is_current.get('integrator', False):
            return

        self.integrator_instance = _INTEGRATION_ALGORITHMS[self.algorithm_key](
                self.algo_compile_settings['precision'],
                dxdt_function,
                self.algo_compile_settings['n_states'],
                self.algo_compile_settings['n_obs'],
                self.algo_compile_settings['n_par'],
                self.algo_compile_settings['n_drivers'],
                self.algo_compile_settings['dt_min'],
                self.algo_compile_settings['dt_max'],
                self.algo_compile_settings['dt_save'],
                self.algo_compile_settings['dt_summarise'],
                self.algo_compile_settings['atol'],
                self.algo_compile_settings['rtol'],
                self.algo_compile_settings['save_time'],
                self.output_functions.save_state_func,
                self.output_functions.update_summary_metrics_func,
                self.output_functions.save_summary_metrics_func,
                self.algo_compile_settings['n_saved_states'],
                self.algo_compile_settings['n_saved_observables'],
                self.algo_compile_settings['summary_temp_memory'],
                )

        self.thread_settings['loop_shared_memory'] = self.integrator_instance.get_loop_internal_shared_memory()

    def _update_integrator(self):
        """Update the integrator with current settings and functions."""

        #Check if
        if self._component_is_current.get('integrator', False) and self._component_is_current.get('loop', False):
            return

        # Ensure the integrator instance exists
        if self.integrator_instance is None:
            self._instantiate_loop()
            return

        # Update the integrator with current settings
        update_dict = self.algo_compile_settings.copy()
        update_dict['dxdt_function'] = self.dxdt_function
        update_dict.update({
            'save_state_func':     self.output_functions.save_state_func,
            'update_summary_func': self.output_functions.update_summary_metrics_func,
            'save_summary_func':   self.output_functions.save_summary_metrics_func,
            }
                )

        self.integrator_instance.update_settings(update_dict)
        self._component_is_current['integrator'] = True
        self._component_is_current['loop'] = False

    def get_dynamic_memory_required(self):
        """Returns the number of bytes of dynamic shared memory required for a single run of the integrator"""
        self.build()

        datasize = int(np.ceil(self.algo_compile_settings['precision'].bitwidth / 8))

        n_summary_variables = self.algo_compile_settings['n_saved_states'] + \
                              self.algo_compile_settings['n_saved_observables']
        summary_memory_per_variable = self.algo_compile_settings['summary_temp_memory']
        summary_items_total = n_summary_variables * summary_memory_per_variable

        loop_items = self.thread_settings['loop_shared_memory']

        dynamic_sharedmem = (loop_items + summary_items_total) * datasize
        return dynamic_sharedmem

    def build(self):
        """Build the inner loop kernel for the integrator.

        Returns:
            Built loop function: Cuda device function to be called by the kernel for a single
            "run" (set of inputs) of the integrator.
            Dynamic shared memory required: The number of bytes of dynamic shared memory required for a single run.
        """
        # If everything is current, return the existing function
        if self._component_is_current.get('loop', False) and self._loop_function is not None:
            return self._loop_function

        # Ensure output functions are built
        self._build_output_functions()

        # Ensure the integrator is instantiated and updated
        if self.integrator_instance is None:
            self._instantiate_loop()
        else:
            self._update_integrator()

        # Build the loop function
        self._loop_function = self.integrator_instance.build()
        self._component_is_current['loop'] = True

        return self._loop_function

    @property
    def output_functions(self):
        """Get the output functions for the current run."""
        return self._build_output_functions()

    @property
    def dxdt_function(self):
        """Get the dxdt function for the current system."""
        self._component_is_current['system'] = True
        return self._system.get_device_function()



    @property
    def loop_function(self):
        """Get the loop function for the current algorithm."""
        return self.build()

