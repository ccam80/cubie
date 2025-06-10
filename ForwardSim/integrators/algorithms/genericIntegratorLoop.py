# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""
#Clumsy way to force precision


import numpy as np
from numba import float32, float64, int32, int16, literally
from numba import cuda


### WORKAROUND This section is a potentially cludgey way to manage setting the
### precision (data type) for all CUDA device functions at module input,
### allowing them to compile for 32 or 64 bit floats, configurable in the top
### module. There's got to be a better way, I just don't know it yet.
if __name__ == '__main__':
    import os
    os.environ["cuda_precision"] = "float64"

import os
precision = os.environ.get("cuda_precision")

if precision == "float32":
    precision = float32
elif precision == "float64":
    precision = float64
elif precision is None:
    precision = float64
### END WORKAROUND


#ERROR These will need to be made in a factory function that has the dxdt device function ,or else they'll be lacking
#dxdt when compiling. Maybe.
@cuda.jit(
    #LOOKHEREFORTYPEERRORS: go back to lazy compilation (no signature) if some weird type errors occur.
             (precision[:],
              precision[:],
              precision[:],
              precision,
              precision,
              int32,
              )#, keep it lazy for now - pre-compiling caused issues with literals in previous iteration of code
             device=True,
             inline=True)
# TODO: Reorder arguments once list finalised
def euler_run(dxdt_func,
              inits,
              parameters,
              forcing_vec,
              step_size,
              output_length,
              filter_coefficients,
              dxdt_temp,
              state_temp,
              saved_states,
              output_temp,
              state_output,
              saved_observables,
              observables_temp,
              observables_output_temp,
              observables_output,
              precision=precision,
              save_every = 1,
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


        dxdt_temp (1d device array, floats):
            An array to hold individual samples of the state inside an iteration;
            should be stored somewhere fast.

        state_temp (1d device array, (floats)):
            An array to hold individual samples of the state between iterations;
            should be stored somewhere fast.

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

    #Note: We will only need to keep track of time for non-autonomous systems,
    # or if we choose to provide a parameterised driver function, but we can
    # do this at a higher level instead and pass a vector unless very memory-bound.
    # l_t = 0
    nsavedstates = len(saved_states)
    nsavedobs = len(saved_observables)
    nstates = len(inits)

    # Loop through output samples, one iteration per output
    for i in range(warmup_samples + output_length):

        #Loop through euler solver steps - smaller step than output for numerical accuracy reasons
        for j in range(save_every):

            # l_t += step_size
            driver = forcing_vec[(i*save_every + j) % len(forcing_vec)]


            #Get current filter coefficient for the downsampling filter
            filter_coefficient = filter_coefficients[j]

            # Calculate derivative at sample
            dxdt_func(state_temp,
                     parameters,
                     driver,
                     observables_temp,
                     dxdt_temp)

            #Forward-step state using euler
            #Add sum*filter coefficient to a running sum for downsampler
            for k in range(nstates):
                state_temp[k] += dxdt_temp[k] * step_size
            for k in range(nsavedstates):
                output_temp[k] += state_temp[saved_states[k]] * filter_coefficient
            for k in range(nsavedobs):
                observables_output_temp[k] += observables_temp[saved_observables[k]] * filter_coefficient

        #Start saving only after warmup period (to get past transient behaviour)
        if i > (warmup_samples - 1):

            for n in range(nsavedstates):
                state_output[i-warmup_samples, n] = output_temp[n]
            for n in range(nsavedobs):
                observables_output[i-warmup_samples, n] = observables_temp[n]

        #Reset filters to zero for another run
        output_temp[:] = precision(0.0)


@cuda.jit(
    # (threeCM_precision[:],
    #           threeCM_precision[:],
    #           threeCM_precision[:],
    #           threeCM_precision,
    #           threeCM_precision,
    #           int32,
    #           )#, keep it lazy for now - pre-compiling caused issues with literals in previous iteration of code
             device=True,
             inline=True)
# TODO: Reorder arguments once list finalised
def single_integrator(dxdt_func,
                      inits,                     # pass as local array, thread-specific
                      parameters,                # pass as local array from batch-organising function. thread-specific
                      forcing_vec,               # Forcing vector (repeating) for non-autonomous systems
                      duration,                  # pass as device constant?
                      step_size,                 # pass as device constant?
                      output_fs,                 # pass as device constant?
                      saved_state_indices,       # Pass as constant device array
                      saved_observable_indices,  # pass as constant device array
                      filtercoeffs,              # pass as constant device array
                      output_array,              # Reference to device array view
                      observables_array,         # Reference to device array view
                      integrator=euler_run,      # Device function
                      save_every = 1,
                      warmup_time=0.0):
    """
    Note: This function should handle the allocation and distribution of device
    memory for temporary and working data inside an integration. The caller will
    provide a dxdt function, parameter set, forcing vector, initial conditions, and
    parameters for the solver, including allocated storage for the output.

    A higher-level function will generate parameter sets for each instance of this
    device function, and split up the output memory views accordingly. This
    function will be thread-agnostic - any indexing and location is handled in
    the caller.

    Problem with this approach - shared memory for temp sets needs to be allocated
    for the whole batch at once, so needs to be created and split up at the batch-allocator level.

    Parameters
    ----------
    tx : TYPE
        DESCRIPTION.
    dxdt_func : TYPE
        DESCRIPTION.
    nstates : TYPE
        DESCRIPTION.
    inits : TYPE
        DESCRIPTION.
    duration : TYPE
        DESCRIPTION.
    step_size : TYPE
        DESCRIPTION.
    output_fs : TYPE
        DESCRIPTION.
    filtercoeffs : TYPE
        DESCRIPTION.
    saved_states : TYPE
        DESCRIPTION.
    saved_observables : TYPE
        DESCRIPTION.
    output_array : TYPE
        DESCRIPTION.
    observables_array : TYPE
        DESCRIPTION.
    warmup_time : TYPE, optional
        DESCRIPTION. The default is 0.0.

    Returns
    -------
    None.

    """
    nsavedstates = len(saved_state_indices)
    nsavedobs = len(saved_observable_indices)
    nstates = len(inits)

    output_samples = int32(round((duration * output_fs)))
    warmup_samples = int32(warmup_time * output_fs)
    save_every


    #Experiment with local vs shared here - reducing local can reduce register dependence,
    # (I think), shared is fast but limited. allocation may require globalisation (to allow literalisation)
    # of dimensions

    dstates_temp = cuda.local.array(
        shape=(nstates),
        dtype=precision)




    #Run integrator loop


#This stuff is in global scope but should sit in a kernel-builder function
global zero
zero = 0
global nstates
nstates = 3
global nparams
nparams = 7
global nobs
nobs = 6

@cuda.jit(opt=True, lineinfo=True)  # Lazy compilation allows for literalisation of shared mem params.
def single_integrator_kernel(xblocksize,
                             system,
                             output,
                             observables,
                             parameters,
                             inits,
                             step_size,
                             duration,
                             output_fs,
                             filter_coefficients,
                             RNG,
                             noise_sigmas,
                             integrator=euler_run,
                             warmup_time = 0):
    # All variables prefixed by memory location - l_ local, c_ constant, d_ device, s_ shared.
    l_tx = int16(cuda.threadIdx.x)
    l_block_index = int32(cuda.blockIdx.x)
    l_run_index = int32(xblocksize * l_block_index + l_tx)

    # Don't try and do a run that hasn't been requested.
    if l_run_index >= 1:
        return None

    l_step_size = precision(step_size)
    l_save_every = int32(round(1 / (output_fs * l_step_size)))
    l_output_samples = int32(round((duration / l_step_size) / l_ds_rate))  # samples per output value
    l_warmup_samples = int32(warmup_time * output_fs)
    l_t = precision(0.0)

    #literalise globals used to allocate shared memory
    #TODO: Test in which scenarios this is required
    litzero = literally(zero)
    litstates = literally(nstates)
    litparamsslength = literally(nparams)

    l_dstates_temp = cuda.local.array(
        shape=(litstates),
        dtype=precision)
    l_dstates_temp[:] = precision(0.0)
    # Declare temp arrays to be kept in shared memory - very quick access.
    #One thread's portion of the shared memory contains:
    # - 2 * chunks of nstates, for internal loop state and  running sums
    # - 2* chinks of nobs, for internal obs state and running subs
    # The whole block has (2*nstates + 2*nobs) * xblocksize allocated

    #TODO: Test assigning only the thread's individual portion here. Unsure if this breaks
    # the dynamic shared memory stuff in Numba
    obs_start = xblocksize * 2 * nstates
    dynamic_mem = cuda.shared.array(litzero, dtype=precision)
    s_saved_state = dynamic_mem[:xblocksize * nstates]
    s_state_temp = dynamic_mem[xblocksize * obs_start]
    s_obs_temp = dynamic_mem[obs_start: obs_start + xblocksize * nobs]
    s_obs_saved = dynamic_mem[obs_start + nobs: obs_start + 2 * xblocksize * nobs]



    c_params = cuda.const.array_like(parameters)
    c_filtercoefficients = cuda.const.array_like(filter_coefficients)

    # Initialise w starting states
    for i in range(nstates):
        s_state_temp[l_tx * nstates + i] = inits[i]
        s_saved_state[l_tx * nstates + i] = precision(0.0)
        s_obs_temp[l_tx * nobs + i] = precision(0.0)
        s_obs_saved[l_tx * nobs + i] = precision(0.0)

    dxdt_func = None
    forcing_vec = None
    output_samples = None
    filtercoeffs =

    #Run the integrator
    integrator(dxdt_func,
               inits,
               c_params,
               forcing_vec,
               step_size,
               output_samples,
               c_filtercoefficients,
               l_dstates_temp,
               s_state_temp,
               saved_state_indices,
               saved_states_temp,
               output_array,
               saved_observable_indices,
               obs_temp,
               saved_obs_temp,
               observables_array,
               precision=precision,
               save_every=save_every,
               warmup_samples=warmup_samples
               )

def batch_integrator(nsavedobs, nsavedstates):
    saved_obs_indices = cuda.constant.array(
        shape=(nsavedobs),
        dtype=int16)

    saved_state_indices = cuda.constant.array(
        shape=(nsavedstates),
        dtype=int16)


if __name__ == "__main__":

    dxdt = three_chamber_model_dV
    output = cuda.device_array((3, 1000), dtype=precision)
    observables = cuda.device_array((6, 1000), dtype=precision)

    nsavedobs = observables.shape[0]
    nsavedstates = output.shape[0]
    saved_obs = np.arange(observables.shape[0])
    saved_states = np.arange(nsavedstates)

    #Setup for inside batch or single integrator
    saved_obs_indices = cuda.constant.array(
        shape=(nsavedobs),
        dtype=int16)

    for i in range(nsavedobs):
        saved_obs_indices[i] = saved_obs[i]

    saved_state_indices = cuda.constant.array(
        shape=(nsavedstates),
        dtype=int16)