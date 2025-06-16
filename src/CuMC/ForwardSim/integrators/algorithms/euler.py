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
from CuMC._utils import _convert_to_indices


# noinspection PyRedundantParentheses
class genericODEIntegrator():
    """ Contains generic implementation of an ODE integrator, including allocation and the inner loop.
     Allocation might """
    def __init__(self,
                 precision,
                 saved_states=None,
                 saved_observables=None):

        self.precision = precision
        self.saved_states = saved_states
        self.saved_observables = saved_observables

    #TODO: force rebuild if system or saved states change

    def get_shared_memory_requirements(self, system):
        """Overload this function with the number of bytes of shared memory required for a single run of the integrator"""
        num_states = system.num_states
        num_obs = system.num_observables
        num_drivers = system.num_drivers
        total_numbers = 2*num_states + num_obs + num_drivers
        return total_numbers*self.precision().itemsize

    def solve_single(self,
                     system):
        sharedbytes = self.get_shared_memory_requirements(self,system)

        output_samples = int(duration / save_step)
        warmup_samples = int(warmup / save_step)
        pass

    def solve_batch(self):
        pass


    def build_kernel(self,
                    system,
                    internal_step_size,
                    output_step_size,
                    saved_states=None,
                    saved_observables=None,
                    xblocksize=128,
                    warmup_time=0.0):

        shared_bytes = self.get_shared_memory_requirements(self,system)
        numba_precision = from_dtype(self.precision)
        system.build() #TODO: Check where the system is built - do we need to rebuild here?

        #Convert indices from user-provided form to a CUDA-friendly array
        if saved_states is not None:
            saved_states = _convert_to_indices(saved_states, system.states)



        if saved_observables is not None:
            saved_observables = _convert_to_indices(saved_states, system.states)

        self.build_loop(system,
                        internal_step_size,
                        output_step_size,
                        saved_states=saved_states,
                        saved_observables=saved_observables)

        intfunc = self.integratorLoop


        @cuda.jit((numba_precision[:],
                   numba_precision[:],
                   numba_precision[:,:],
                   numba_precision[:,:],
                   numba_precision[:,:])
                  )
        def integration_kernel(inits,
                            params,
                            forcing_vector,
                            output,
                            observables,
                            nruns=1,
                            ):
            """Master integration kernel - calls integratorLoop and dxdt device functions.
            Accepts all sets of inits, params, and runs one in each thread.

            TODO: Currently saves forcing vector as a constant array. Consider moving this to a device function, which takes
            time and thread index as arguments, returning a forcing value.
            """
            tx = int16(cuda.threadIdx.x)
            block_index = int32(cuda.blockIdx.x)
            run_index = int32(xblocksize * block_index + tx)

            if run_index >= nruns:
                return None

            # Allocate shared memory for this thread
            c_forcing_vector = cuda.const.array_like(forcing_vector)
            shared_memory = cuda.shared.array(0,
                                              dtype=precision)

            #Get this thread's portions of shared memory, init values, parameters, outputs.
            tx_shared_memory = shared_memory[tx*shared_bytes:(tx+1)*shared_bytes]
            rx_inits = inits[run_index,:]
            rx_params = params[run_index,:]
            rx_output = output[run_index,:,:] #TODO: Check slice order learnings from last CUDA-fest
            rx_observables = observables[run_index,:,:]

            intfunc(
                rx_inits,
                rx_params,
                c_forcing_vector,
                tx_shared_memory,
                rx_output,
                rx_observables,
                output_samples,
                warmup_samples
            )

            return None




# @cuda.jit(
#     # (threeCM_precision[:],
#     #           threeCM_precision[:],
#     #           threeCM_precision[:],
#     #           threeCM_precision,
#     #           threeCM_precision,
#     #           int32,
#     #           )#, keep it lazy for now - pre-compiling caused issues with literals in previous iteration of code
#              device=True,
#              inline=True)
# # TODO: Reorder arguments once list finalised
# def single_integrator(dxdt_func,
#                       inits,                     # pass as local array, thread-specific
#                       parameters,                # pass as local array from batch-organising function. thread-specific
#                       forcing_vec,               # Forcing vector (repeating) for non-autonomous SystemModels
#                       duration,                  # pass as device constant?
#                       step_size,                 # pass as device constant?
#                       output_fs,                 # pass as device constant?
#                       saved_state_indices,       # Pass as constant device array
#                       saved_observable_indices,  # pass as constant device array
#                       output_array,              # Reference to device array view
#                       observables_array,         # Reference to device array view
#                       integrator=euler_run,      # Device function
#                       save_every = 1,
#                       warmup_time=0.0):
#     """
#

#     """
#     nsavedstates = len(saved_state_indices)
#     nsavedobs = len(saved_observable_indices)
#     nstates = len(inits)
#
#     output_samples = int32(round((duration * output_fs)))
#     warmup_samples = int32(warmup_time * output_fs)
#     save_every
#
#
#     #Experiment with local vs shared here - reducing local can reduce register dependence,
#     # (I think), shared is fast but limited. allocation may require globalisation (to allow literalisation)
#     # of dimensions
#
#     dstates_temp = cuda.local.array(
#         shape=(nstates),
#         dtype=precision)


#
# @cuda.jit(opt=True, lineinfo=True)  # Lazy compilation allows for literalisation of shared mem params.
# def single_integrator_kernel(xblocksize,
#                              system,
#                              output,
#                              observables,
#                              parameters,
#                              inits,
#                              step_size,
#                              duration,
#                              output_fs,
#                              filter_coefficients,
#                              RNG,
#                              noise_sigmas,
#                              integrator=euler_run,
#                              warmup_time = 0):
#     # All variables prefixed by memory location - l_ local, c_ constant, d_ device, s_ shared.
#     l_tx = int16(cuda.threadIdx.x)
#     l_block_index = int32(cuda.blockIdx.x)
#     l_run_index = int32(xblocksize * l_block_index + l_tx)
#
#     # Don't try and do a run that hasn't been requested.
#     if l_run_index >= 1:
#         return None
#
#     l_step_size = precision(step_size)
#     l_save_every = int32(round(1 / (output_fs * l_step_size)))
#     l_output_samples = int32(round((duration / l_step_size) / l_ds_rate))  # samples per output value
#     l_warmup_samples = int32(warmup_time * output_fs)
#     l_t = precision(0.0)
#
#     #literalise globals used to allocate shared memory
#     #TODO: Test in which scenarios this is required
#     litzero = literally(zero)
#     litstates = literally(nstates)
#     litparamsslength = literally(nparams)
#
#    
#

#     obs_start = xblocksize * 2 * nstates

#
#     c_params = cuda.const.array_like(parameters)
#     c_filtercoefficients = cuda.const.array_like(filter_coefficients)
#
#     # Initialise w starting states
#     for i in range(nstates):
#         s_state_temp[l_tx * nstates + i] = inits[i]
#         s_saved_state[l_tx * nstates + i] = precision(0.0)
#         s_obs_temp[l_tx * nobs + i] = precision(0.0)
#         s_obs_saved[l_tx * nobs + i] = precision(0.0)
#
#     dxdt_func = None
#     forcing_vec = None
#     output_samples = None
#     filtercoeffs =
#
#     #Run the integrator
#     integrator(dxdt_func,
#                inits,
#                c_params,
#                forcing_vec,
#                step_size,
#                output_samples,
#                c_filtercoefficients,
#                l_dstates_temp,
#                s_state_temp,
#                saved_state_indices,
#                saved_states_temp,
#                output_array,
#                saved_observable_indices,
#                obs_temp,
#                saved_obs_temp,
#                observables_array,
#                precision=precision,
#                save_every=save_every,
#                warmup_samples=warmup_samples


    def build_loop(self,
                  system,
                  internal_step_size,
                  output_step_size,
                  saved_states=None,
                  saved_observables=None):
        """Implementation notes:
        -
        - To reduce memory requirements, saved states and observables should be
        passed as function-scope variables (considered global by the device function),
        so that they're "baked in" to the loop. This might cause issues with repeated
        compilations when the saved states change, but I think that will happen outside
        of the time-intensive algorithm, a once-per-run change that won't add significant
        overhead to the whole process.

        - How far can I push the hard-coding and globalisation of constants?
            - can I hard-code the whole forcing vector? It's pre-calculated... Fine if forcing is
            same between threads, but not if it's different for each thread. Perhaps that's acceptable, and we accomodate
            variable forcing by modifying the system such that it's a state.

        - The memory needs are still high - an inner-loop temp and output array to keep current and running sums,
        and a bulk output array to write to in the outer loop. Each algorithm function should allocate its own temporary arrays,
        only taking input and output arrays. HOWEVER this does not allow for access to dynamically allocated shared memory.
        UNLESS we give information about thread indices etc. so that the function can find the correct slice of shared memory.
        Should we just pass it a pointer to allocated shared memory, and let it pick it's own slice, or should we pass it a slice
        address, so that it can ignore whether it's shared or local? Total shared memory allocation must be done by the kernel,
        we can only pass a reference to this function.

        - Will the storage arrays differ between algorithms? I think an algorithm should work on a dxdt function, and produce
        output arrays for states and observables only. A Euler integrator will just need one temp dxdt array. An RK4
        algorithm, depending on whether it's one-step-per-thread or all operations in one thread, will have different
        memory requirements.

        - Some user control over storage of temp values should be available, as this offers a performance boost for
        some system sizes.

        -The device function MUST accept varying initial conditions and parameters.


        Consider adaptive-step-size requirements - keeping an unspecified number of time steps, modifying the step size if
        an output sample is requested at that time step. Do we need to allow for full output, or only downsampled/regular?

        When is this compiled? Do we compile the solver once for each run, or in a full monte-carlo simulation, once per monte-carlo?
        What changes?
            - Changes in step size require changes in downsampling if filter is used
            - Changes in selection of saved states and obeservables require reallocation
            - Changes in system (reallocation of storage arrays)
            - Changes in sampling freq (simulated or saving) but not duration. this might allow the compiler
            to optimise e.g. save_every=1
            """

        #Parameters to size the arrays in the integrator loop
        nstates = system.num_states
        nobs = system.num_observables
        npar = system.num_parameters
        ndrivers = system.num_drivers

        if saved_states is None:
            saved_states = np.arange(nstates, dtype=np.int16)
        if saved_observables is None:
            saved_observables = np.arange(nobs, dtype=np.int16)

        n_saved_states = len(saved_states)
        n_saved_observables = len(saved_observables)

        precision = system.precision
        save_every = int32(round(output_step_size / internal_step_size))

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
                      forcing_vec, # Device array (floats) - constant memory if all-threads, local if per-thread
                      shared_memory,
                      state_output,
                      observables_output,
                      output_length,
                      warmup_samples=0):
            """
            Note to self: This is an attempt to isolate the integrator algorithm from
            the surrounding solver support, with a view to make it easier to implement
            other algorithms. Inputs to this function should be problem-independent, and
            cover as much of the possible algorithms as possible. This might not work.

            CUDA memory management notes: This function does not create any new device
                arrays. Placement of arrays into memory locations should be handled
                in a higher-level function. This is why there's about a million arguments.

            Arguments:
                dxdt_func (device function):
                    A function that modifies an observables and
                    dxdt array based on a state, parameters, and forcing term. One
                    step only - this is the gradient function, or transition kernel(?).

                parameters (1d device array (floats)):
                    Parameters for the dxdt_func. Must correspond element-wise to the
                    expected values in dxdt_func

                inits (1d device array, (floats)):
                    initial values for each state variable

                forcing_vec: (1d device array, (floats)):
                    A forcing function, interpolated to have a step_size (internal,
                    integrator loop step) resolution. This will loop indefinitely if shorter
                    than the number of samples requested.

                step_size (float):
                    The step size for each iteration of the internal integrator loop

                output_length (int):
                    The number of output samples to run for, duration * output_fs

                filter_coefficients (1d device array, (floats)):
                    Coefficients for a downsampling filter, for simulations with noise.
                    Not required for noiseless simulations (just give a vector of 1/save_every),
                    as no aliasing will occur unless you're sampling well below the speed
                    of the system dynamics. Should be length save_every.

                state_output (2d device array, (floats)):
                    Bulk storage for integrator output - should be shape (num_states, output_length)

                output_temp (1d device array, (floats)):
                    An array to hold running sums of the output states between iterations;
                    should be stored somewhere fast. Length: nsavedstates.

                saved_states (1d device array, (int)):
                    Indices of states to save.

                saved_observables (1d device array, (int)):
                    Indices of obervables to save.

                obervables_temp (1d device array, (floats)):
                    An array to hold individual samples of observable (auxiliary) variables
                    between iterations; should be stored somewhere fast.

                observables_output_temp (1d device arrat (float)):
                    An array to hold running sums of the saved observables between iterations;
                    should be stored somewhere fast. Length: nsavedobs

                observables_output (2d device array, (floats)):
                    Bulk storage for observable (auxiliary) variables for output -
                    should be shape (num_states, output_length)

                save_every (int):
                    Downsampling rate - number of step_size increments to take before
                    saving a value.

                warmup_samples (int):
                    How many output samples to wait (to allow system to settle) before
                    recording output in precious, precious memory.

            returns:
                None, modifications are made in-place.
            """

            #Allocate state and dxdt a slice of the shared memory slice passed to this thread
            state = shared_memory[:nstates]
            dxdt = shared_memory[nstates:2*nstates]
            observables = shared_memory[2*nstates:2*nstates + nobs]
            drivers = shared_memory[2*nstates + nobs: 2*nstates + nobs + ndrivers]
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

                # Euler loop
                for j in range(save_every):
                    for k in range(ndrivers):
                        drivers[k] = forcing_vec[(i*save_every + j) % driver_length,k]

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

                    for n in range(n_saved_states):
                        state_output[i-warmup_samples, n] = state[saved_states[n]]
                    for n in range(n_saved_observables):
                        observables_output[i-warmup_samples, n] = observables[saved_observables[n]]

        self.integratorLoop = euler_loop


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
