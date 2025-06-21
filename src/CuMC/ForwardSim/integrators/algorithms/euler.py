# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""

if __name__ == "__main__":
    import os
    os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
    os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"
    os.environ["NUMBA_OPT"] = "0"

import numpy as np
from numba import float32, float64, int32, int16, literally
from numba import cuda, from_dtype
from warnings import warn
from CuMC.ForwardSim.integrators.output_functions import build_output_functions


class genericODEIntegratorLoop():
    """ A parent class for ODE integrator loops. It comprises a factory function for building the inner integrator loop,
     a function to calculate shared memory requirements, and a function to rebuild when the system changes.

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
                 xblocksize=128,
                 saved_states=None,
                 saved_observables=None,
                 dtmin=0.01,
                 dtmax=0.1,
                 save_every=0.1,
                 summarise_every=1.0,
                 output_functions=None):


        if output_functions is None:
            output_functions = ['save_state']

        # Keep parameters not derived from "system" class as a dict, so that a user can rebuild without having to
        # pass all parameters again.
        self.settings = {'dtmin': dtmin,
                         'dtmax': dtmax,
                         'save_every': save_every,
                         'summarise_every': summarise_every,
                         'output_functions': output_functions,
                         'saved_states': saved_states,
                         'saved_observables': saved_observables}

        self.xblocksize = xblocksize
        self.shared_memory_required = self.get_shared_memory_required(system)

        self.build(system,
                   saved_states,
                   saved_observables,
                   )

    def build_loop(self,
                  system):
        """Implementation notes:

        TODO: Consider adaptive-step-size requirements - keeping an unspecified number of time steps, modifying the step size if
            an output sample is requested at that time step. Do we need to allow for full output, or only downsampled/regular?
            """
        # NOTE: All of this preamble must be in the same function as the loop definition, to get the values into the
        # global scope from the loop definition's perspective. As such, it will be repeated in different integrator
        # loop subclasses. This is a necessary evil.

        #Parameters to size the arrays in the integrator loop
        nstates = system.num_states
        nobs = system.num_observables
        npar = system.num_parameters
        ndrivers = system.num_drivers

        #Update the loop's memory requirement if the system has changed

        funcreturn = build_output_functions(self.settings)
        save_summary_func = funcreturn.save_summary_metrics_func
        update_summary_func = funcreturn.update_summary_metrics_func
        save_state_func = funcreturn.save_state_func
        summary_memory_required = funcreturn.summary_memory_required

        self.shared_memory_required = self.get_shared_memory_required(system) + summary_memory_required

        saved_states = self.settings['saved_states']
        saved_observables = self.settings['saved_observables']

        if saved_states is None:
            saved_states = np.arange(nstates, dtype=np.int16)
        if saved_observables is None:
            saved_observables = np.arange(nobs, dtype=np.int16)

        n_saved_states = len(saved_states)
        n_saved_observables = len(saved_observables)

        precision = system.precision

        # This will be implementation specific - variable-step algorithms will need to handle it differently.
        save_every_samples = int32(round(self.settings['save_every'] / self.settings['dtmin']))
        summarise_every_samples = int32(round(self.settings['summarise_every'] / self.settings['save_every']))

        #samples < 1 won't make much sense.
        if summarise_every_samples <= 1:
            raise ValueError("summarise_every must be greater than save_every, as it sets the number of saved samples between summaries,"
                             "which must be >1")
        if self.settings['save_every'] > self.settings['dtmin']:
            raise ValueError("save_every must be less than dtmin, as it is the number of loop-steps between saves, which must be >1. ")

        # Update and log the actual save and summary intervals, which will differ from what was ordered if they are not
        # a multiple of the loop step size.
        new_save_every = save_every_samples * self.settings['dtmin']
        new_summarise_every = summarise_every_samples

        if new_save_every != self.settings['save_every']:
            self.settings['save_every'] = new_save_every
            warn("save_every was set to {new_save_every}s, because it is not a multiple of dtmin ({self.settings['dtmin']}s)."
                 "save_every can only save a value after an integer number of steps in a fixed-step integrator")
        if new_summarise_every != self.settings['summarise_every']:
            self.settings['summarise_every'] = new_summarise_every
            warn("summarise_every was set to {new_summarise_every}s, because it is not a multiple of save_every ({self.settings['save_every']}s)."
                 "summarise_every can only save a value after an integer number of steps in a fixed-step integrator")




        #Pick up here - add temporary summary array (can be length 0) then we can save summaries and outputs in this loop

        dxdt_func = system.dxdtfunc

        @cuda.jit(
                     (precision[:],
                      precision[:],
                      precision[:,:],
                      precision[:],
                      precision[:,:],
                      precision[:,:],
                      int32,
                      int32,
                      ),
                     device=True,
                     inline=True)
        def euler_loop(inits,
                      parameters,
                      forcing_vec,
                      shared_memory,
                      state_output,
                      observables_output,
                      output_length,
                      warmup_samples=0):
            """

            """

            #Allocate state and dxdt a slice of the shared memory slice passed to this thread
            state = shared_memory[:running_index]
            running_index = int16(nstates)
            dxdt = shared_memory[running_index:running_index + nstates]
            running_index += nstates
            observables = shared_memory[running_index:running_index + nobs]
            running_index += nobs
            drivers = shared_memory[running_index: running_index + ndrivers]
            running_index += ndrivers
            state_summaries = shared_memory[running_index:running_index + n_saved_states]
            running_index += n_saved_states
            observables_summaries = shared_memory[running_index: running_index + n_saved_observables]

            driver_length = forcing_vec.shape[0]
            #Initialise/Assign values to allocated memory
            for i in range(nstates):
                state[:] = inits[i]
            dxdt[:] = precision(0.0)
            l_parameters = cuda.local.array((npar),
                                          dtype=precision)

            for i in range(npar):
                l_parameters[i] = parameters[i]

            # Loop through output samples, one iteration per output sample
            for i in range(warmup_samples + output_length):

                # Euler loop - internal step size <= outout step size
                for j in range(save_every_samples):
                    for k in range(ndrivers):
                        drivers[k] = forcing_vec[(i*save_every_samples + j) % driver_length,k]

                    # Calculate derivative at sample
                    dxdt_func(state,
                             parameters,
                             drivers,
                             observables,
                             dxdt)

                    #Forward-step state using euler
                    for k in range(nstates):
                        state[k] += dxdt[k] * internal_step_size


                #Start saving only after warmup period (to get past transient behaviour)
                if i > (warmup_samples - 1):

                    save_state_func(state_output, observables_output, state, observables, i - warmup_samples)
                    update_summary_func(state_summaries, observables_summaries, state, observables,
                                                i - warmup_samples)

                    if i % summarise_every_samples == 0:
                        save_summary_func(state_summaries, observables_summaries,
                                                  state_summaries_output, observables_summaries_output, i)
        self.integratorLoop = euler_loop

    def get_shared_memory_required(self, system):
        """Overload this function with the number of bytes of shared memory required for a single run of the integrator"""
        num_states = system.num_states
        num_obs = system.num_observables
        num_drivers = system.num_drivers
        total_items = 2*num_states + num_obs + num_drivers
        return total_items * system.precision().itemsize

if __name__ == "__main__":
    from CuMC.SystemModels.Systems.threeCM import ThreeChamberModel
    precision = np.float32
    numba_precision = from_dtype(precision)
    sys = ThreeChamberModel(precision=precision)
    sys.build()

    internal_step = 0.001
    save_step = 0.01
    duration = 0.1
    warmup = 0.1

    output_samples= int(duration / save_step)
    warmup_samples = int(warmup/save_step)

    integrator = genericODEIntegrator(precision=precision)
    integrator.build_loop(sys,
                          internal_step,
                          save_step)
    intfunc = integrator.integratorLoop

    @cuda.jit()
    def loop_test_kernel(inits,
                         params,
                         forcing_vector,
                         output,
                         observables):

        c_forcing_vector = cuda.const.array_like(forcing_vector)

        shared_memory = cuda.shared.array(0, dtype=numba_precision)

        intfunc(
            inits,
            params,
            c_forcing_vector,
            shared_memory,
            output,
            observables,
            output_samples,
            warmup_samples
        )

    output = cuda.pinned_array((output_samples,sys.num_states), dtype=precision)
    observables = cuda.pinned_array((output_samples,sys.num_observables), dtype=precision)
    forcing_vector = cuda.pinned_array((output_samples + warmup_samples, sys.num_drivers), dtype=precision)


    output[:,:] = precision(0.0)
    observables[:,:] = precision(0.0)
    forcing_vector[:,:] = precision(0.0)
    forcing_vector[0,:] = precision(1.0)

    d_forcing = cuda.to_device(forcing_vector)
    d_inits = cuda.to_device(sys.init_values.values_array)
    d_params = cuda.to_device(sys.parameters.values_array)
    d_output = cuda.to_device(output)
    d_observables = cuda.to_device(observables)


    # global sharedmem
    sharedmem=integrator.get_shared_memory_requirements(sys)

    loop_test_kernel[1, 1, 0, sharedmem](d_inits,
                          d_params,
                          d_forcing,
                          d_output,
                          d_observables,
                          )

    cuda.synchronize()
    output = d_output.copy_to_host()
    obs = d_observables.copy_to_host()
    print(output)
    print(obs)
