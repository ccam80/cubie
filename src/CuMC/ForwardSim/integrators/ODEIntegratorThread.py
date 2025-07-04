# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""

if __name__ == "__main__":
    import os

    os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
    os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"
    os.environ["NUMBA_OPT"] = "0"

import numpy as np
# from numba import float32, float64, int32, int16, literally
from numba import cuda, from_dtype
# from warnings import warn
from CuMC.ForwardSim.integrators.output_functions import build_output_functions
from CuMC.ForwardSim.integrators.algorithms import Euler
from numpy.typing import ArrayLike, NDArray
_INTEGRATION_ALGORITHMS = {"euler": Euler}


class ODEIntegratorLoop:
    """ Creates and builds a device function for a CUDA ODE integration. Interprets user-space compile_settings in terms
    of times, system parameters, and desired outputs, and translates them into CUDA-space parameters for configuring and
    compiling the per-thread device function. Does not handle the dividing of batch runs into threads; this is handled
    by the next layer up, the Solver class.

    Accepts a string which specifies which algorithm to use. Additional algorithms can be added by adding an object that
    builds the loop function given a set of common parameters (subclassed from GenericIntegratorAlgorithm, see euler.py
    for an example). The loop algorithm must implement a build_loop() factory method that returns a device function
    which executes the whole integrator loop from start to finish, using the device functions from output_functions for
    output. The subclass must also have a _calculate_loop_internal_shared_memory() method which returns the number of
    items of shared memory required for a given system (not bytes, but slots of the size of whatever precision we use).

    This class is not typically exposed to the user directly. The user-facing API is the Solver class, which handles
    the batching up of runs and management of input/output memory. This class contains core functionality for running
    a single thread of an integrator for a given (individual) set of initial values, parameters, and drivers. It is
    separate from the genericIntegratorAlgorithm class so that we can insert any algorithm we choose without duplicating
    the surrounding code.

    The device function in this class needs to be rebuilt when there are changes to:

     Lower-level:
     - The system/problem (number of states, observables, parameters, drivers)

     This-level:
     - Solver parameters such as step size, output size.
     - The desired outputs are changed (e.g. save signals, running mean, running max).

     Higher-level:
     - The saved states and observables (indices of states and observables to save).
     - xblocksize for the integrator kernel (as this sets the grid dimensions)."""

    def __init__(self,
                 system,
                 algorithm='euler',
                 xblocksize=128,
                 saved_states: NDArray[np.int_]=None,
                 saved_observables: NDArray[np.int_]=None,
                 dt_min=0.01,
                 dt_max=0.1,
                 atol=1e-6,
                 rtol=1e-6,
                 dt_save=0.1,
                 dt_summarise=1.0,
                 output_functions=None):

        # Keep parameters that specifically set the compile state of the loop functions in  a dict, so that a user can
        # rebuild without having to pass all parameters again.
        self.algo_compile_settings = {'dt_min': dt_min,
                                      'dt_max': dt_max,
                                     'atol': atol,
                                     'rtol': rtol,
                                     'dt_save': dt_save,
                                     'dt_summarise': dt_summarise,
                                     'saved_states': saved_states,
                                     'saved_observables': saved_observables,
                                     'output_functions': output_functions,
                                     'n_peaks': 0
                                        }

        self.thread_compile_settings = {'xblocksize': xblocksize,
                                        'algorithm': algorithm,
                                        'shared_memory_per_thread':0}

        self.built_algorithm_function = None
        self.integrator_algorithm = None
        self.shared_memory_per_thread = 0
        self.build(system)




    def build_outputs(self, saved_states, saved_observables):

        outputfunctions = build_output_functions(self.compile_settings['output_functions'],
                                                 saved_states,
                                                 saved_observables,
                                                 self.compile_settings['n_peaks'])

        save_summaries = outputfunctions.save_summary_metrics_func
        update_summaries = outputfunctions.update_summary_metrics_func
        save_state = outputfunctions.save_state_func
        summary_sharedmem = outputfunctions.temp_memory_requirements
        summary_outputmem = outputfunctions.summary_output_length
        save_time = outputfunctions.save_time  # Get the save_time flag

        return save_state, update_summaries, save_summaries, summary_sharedmem, summary_outputmem, save_time

    def _clarify_None_output_functions(self, n_saved_observables):
        """Clarify the output functions to be used in the loop, if None is specified, set to default values."""
        # TODO: add empty list check
        if self.compile_settings['output_functions'] is None:
            self.compile_settings['output_functions'] = ['state']
            if n_saved_observables > 0:
                self.compile_settings['output_functions'].append('observables')

    def build(self,
              system):
        """Implementation notes:

        - Future work: When creating a flexible-step integrator, we will have to allocate some compromise length of output
        function, as using dt_min will over-allocate, and dt_max will under-allocate. This is only relevant if dt_save is not set.
        """
        #NOTE: All of this preamble must be in the same function as the loop definition, to get the values into the
        #global scope from the loop definition's perspective. As such, it will be repeated in different integrator
        #loop subclasses. This is a necessary evil.
        precision = system.precision

        #Parameters to size the arrays in the integrator loop
        n_states = system.num_states  # total number of states (complete, for inner loop)
        n_obs = system.num_observables
        n_par = system.num_parameters
        n_drivers = system.num_drivers

        saved_states, saved_observables, n_saved_states, n_saved_observables = self._get_saved_values(n_states)
        self._clarify_None_output_functions(n_saved_observables)
        save_state, update_summaries, save_summaries, summary_temp_memory, summary_output_memory, save_time = self.build_outputs(
            saved_states, saved_observables)
        self.summary_shared_memory = summary_temp_memory * (
                    n_saved_states + n_saved_observables)  #Total memory required for summaries, in items.
        self.summary_output_memory = summary_output_memory

        dxdt_func = system.dxdtfunc

        self.integrator_algorithm = self.build_integrator_algorithm(precision,
                                                                    dxdt_func,
                                                                    n_states,
                                                                    n_obs,
                                                                    n_par,
                                                                    n_drivers,
                                                                    self.compile_settings['dt_min'],
                                                                    self.compile_settings['dt_max'],
                                                                    self.compile_settings['dt_save'],
                                                                    self.compile_settings['dt_summarise'],
                                                                    self.compile_settings['atol'],
                                                                    self.compile_settings['rtol'],
                                                                    save_state,
                                                                    update_summaries,
                                                                    save_summaries,
                                                                    n_saved_states,
                                                                    n_saved_observables,
                                                                    summary_temp_memory,
                                                                    save_time)  # Pass save_time flag
        self.integrator_algorithm.build()

        self.update_dynamic_shared_memory(system)

    def build_integrator_algorithm(self,
                                   precision,
                                   dxdt_func,
                                   n_states,
                                   n_obs,
                                   n_par,
                                   n_drivers,
                                   dt_min,
                                   dt_max,
                                   dt_save,
                                   dt_summarise,
                                   atol,
                                   rtol,
                                   save_time,
                                   save_state_func,
                                   update_summary_func,
                                   save_summary_func,
                                   n_saved_states,
                                   n_saved_observables,
                                   summary_temp_memory):  # Add save_time parameter
        """Build the inner loop kernel for the integrator."""
        return _INTEGRATION_ALGORITHMS[self.algorithm](precision,
                                                       dxdt_func,
                                                       n_states,
                                                       n_obs,
                                                       n_par,
                                                       n_drivers,
                                                       dt_min,
                                                       dt_max,
                                                       dt_save,
                                                       dt_summarise,
                                                       atol,
                                                       rtol,
                                                       save_time,
                                                       save_state_func,
                                                       update_summary_func,
                                                       save_summary_func,
                                                       n_saved_states,
                                                       n_saved_observables,
                                                       summary_temp_memory,
                                                       save_time)  # Pass save_time flag

    def update_dynamic_shared_memory(self, system):
        """Overload this function with the number of bytes of shared memory required for a single run of the integrator"""
        datasize = np.ceil(system.precision.bitwidth / 8)
        loop_items = self.integrator_algorithm._calculate_loop_internal_shared_memory()
        summary_items = self.summary_shared_memory
        self.dynamic_sharedmem = int(np.ceil((loop_items + summary_items) * datasize))

        return self.dynamic_sharedmem

    def get_dynamic_shared_memory_per_thread(self):
        """Returns the number of bytes of shared memory required for a single thread."""
        return self.dynamic_sharedmem


if __name__ == "__main__":
    from CuMC.SystemModels.Systems.threeCM import ThreeChamberModel

    precision = np.float32
    numba_precision = from_dtype(precision)
    sys = ThreeChamberModel(precision=precision)
    sys.build()

    internal_step = 0.001
    save_step = 0.001
    summarise_step = 0.1
    duration = 0.1
    warmup = 0

    output_samples = int(duration / save_step)
    warmup_samples = int(warmup / save_step)

    saved_states = None  # Default to all states
    saved_observables = [0, 1, 2, 3, 4, 5]

    integrator = ODEIntegratorLoop(sys, saved_states=saved_states, saved_observables=saved_observables,
                                   dt_min=internal_step, dt_max=internal_step, dt_save=save_step,
                                   dt_summarise=summarise_step, output_functions=["state", "observables", "max"])

    integrator.build(sys)
    intfunc = integrator.integrator_algorithm.loop_function


    @cuda.jit()
    def loop_test_kernel(inits,
                         params,
                         forcing_vector,
                         output,
                         observables,
                         summary_outputs,
                         summary_observables):
        c_forcing_vector = cuda.const.array_like(forcing_vector)

        shared_memory = cuda.shared.array(0, dtype=numba_precision)

        intfunc(
            inits,
            params,
            c_forcing_vector,
            shared_memory,
            output,
            observables,
            summary_outputs,
            summary_observables,
            output_samples,
            warmup_samples
        )


    output = cuda.pinned_array((output_samples, sys.num_states), dtype=precision).T
    observables = cuda.pinned_array((output_samples, sys.num_observables), dtype=precision).T

    summary_samples = int(np.ceil(output_samples * save_step / summarise_step))
    num_state_summaries = integrator.summary_output_memory * sys.num_states
    num_observable_summaries = integrator.summary_output_memory * len(saved_observables)

    summary_outputs = cuda.pinned_array((summary_samples, num_state_summaries), dtype=precision).T
    summary_observables = cuda.pinned_array((summary_samples, num_observable_summaries), dtype=precision).T

    forcing_vector = cuda.pinned_array((output_samples + warmup_samples, sys.num_drivers), dtype=precision)

    output[:, :] = precision(0.0)
    observables[:, :] = precision(0.0)
    forcing_vector[:, :] = precision(0.0)
    forcing_vector[0::25, :] = precision(1.0)

    d_forcing = cuda.to_device(forcing_vector)
    d_inits = cuda.to_device(sys.init_values.values_array)
    d_params = cuda.to_device(sys.parameters.values_array)
    d_output = cuda.to_device(output)
    d_observables = cuda.to_device(observables)
    d_summary_state = cuda.to_device(summary_outputs)
    d_summary_observables = cuda.to_device(summary_observables)

    # global sharedmem
    sharedmem = integrator.dynamic_sharedmem
    loop_test_kernel[1, 1, 0, sharedmem](d_inits,
                                         d_params,
                                         d_forcing,
                                         d_output,
                                         d_observables,
                                         d_summary_state,
                                         d_summary_observables
                                         )

    cuda.synchronize()
    output = d_output.copy_to_host()
    obs = d_observables.copy_to_host()
    summary_states = d_summary_state.copy_to_host()
    summary_observables = d_summary_observables.copy_to_host()

    print(output)
    print(obs)
    print(summary_states)
    print(summary_observables)
