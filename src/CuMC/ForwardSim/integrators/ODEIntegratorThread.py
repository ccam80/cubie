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

_INTEGRATION_ALGORITHMS = {"euler": Euler}


class ODEIntegratorLoop:
    """ Creates and builds a device function for a CUDA ODE integration. Interprets user-space compile_settings in terms of
    times, system parameters, and desired outputs, and translates them into CUDA-space parameters for configuring and
    compiling the per-thread device function. Does not handle the dividing of batch runs into threads; this is handled
    by the next layer up, the Solver class.

    Accepts a string which specifies the algorithm to use. Additional algorithms can be added by adding an object that
    builds the loop function given a set of common paramters, and also returns how much shared memory is needed inside
    the loop given the number of states/observables in the system.

     Specifically, this kernel needs to be rebuilt when there are changes to:

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
                 saved_states=None,
                 saved_observables=None,
                 dtmin=0.01,
                 dtmax=0.1,
                 atol=1e-6,
                 rtol=1e-6,
                 dt_save=0.1,
                 dt_summarise=1.0,
                 output_functions=None):

        # Keep parameters that specifically set the compile state of the loop functions in  a dict, so that a user can
        # rebuild without having to pass all parameters again.
        self.compile_settings = {'dtmin': dtmin,
                                 'dtmax': dtmax,
                                 'atol': atol,
                                 'rtol': rtol,
                                 'dt_save': dt_save,
                                 'dt_summarise': dt_summarise,
                                 'saved_states': saved_states,
                                 'saved_observables': saved_observables,
                                 'output_functions': output_functions,
                                 'n_peaks': 0
                                 }

        self.xblocksize = xblocksize
        self.integrator_algorithm = None
        self.summary_shared_memory = 0
        self.dynamic_sharedmem = 0
        self.algorithm = algorithm
        self.build(system)
        #TODO: add a routine to handle saved_state or saved_observables being given as strings - figure out at which level this should
        # happen and whether it can just call one of them fancy systemvalues functions.

    def _get_saved_values(self, n_states):
        """Sanitise empty lists and None values - statse default to all, observables default to none."""

        #TODO: Use systemvalues' get_indices to make this simpler and convert to a list of int16s at this level every time.

        saved_states = self.compile_settings['saved_states']
        saved_observables = self.compile_settings['saved_observables']

        # If no saved states specified, assume all states are saved.
        if saved_states is None:
            saved_states = np.arange(n_states, dtype=np.int16)
        n_saved_states = len(saved_states)

        # On the other hand, if no observables are specified, assume no observables are saved.
        if saved_observables is None:
            saved_observables = []
        n_saved_observables = len(saved_observables)

        return saved_states, saved_observables, n_saved_states, n_saved_observables

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

        return save_state, update_summaries, save_summaries, summary_sharedmem, summary_outputmem

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
        function, as using dtmin will over-allocate, and dtmax will under-allocate. This is only relevant if dt_save is not set.
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
        save_state, update_summaries, save_summaries, summary_temp_memory, summary_output_memory = self.build_outputs(
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
                                                                    self.compile_settings['dtmin'],
                                                                    self.compile_settings['dtmax'],
                                                                    self.compile_settings['dt_save'],
                                                                    self.compile_settings['dt_summarise'],
                                                                    self.compile_settings['atol'],
                                                                    self.compile_settings['rtol'],
                                                                    save_state,
                                                                    update_summaries,
                                                                    save_summaries,
                                                                    n_saved_states,
                                                                    n_saved_observables,
                                                                    summary_temp_memory)
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
                                   save_state_func,
                                   update_summary_func,
                                   save_summary_func,
                                   n_saved_states,
                                   n_saved_observables,
                                   summary_temp_memory):
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
                                                       save_state_func,
                                                       update_summary_func,
                                                       save_summary_func,
                                                       n_saved_states,
                                                       n_saved_observables,
                                                       summary_temp_memory)

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

    integrator = ODEIntegratorLoop(sys,
                                   saved_states=saved_states,
                                   saved_observables=saved_observables,
                                   dtmin=internal_step,
                                   dtmax=internal_step,
                                   dt_save=save_step,
                                   dt_summarise=summarise_step,
                                   output_functions=["state", "observables", "max"])

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
