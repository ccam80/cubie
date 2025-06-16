# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""

import numpy as np
from numba import float32, float64, int32, int16, literally
from numba import cuda


# noinspection PyRedundantParentheses
class genericODEIntegrator():
    """ Contains generic implementation of an ODE integrator, including allocation and the inner loop.
     Allocation might """
    def __init__(self,
                 precision,
                 saved_states,
                 saved_observables):

        self.precision = precision
        self.saved_states = saved_states
        self.saved_observables = saved_observables

    #TODO: force rebuild if system or saved states change

    def get_shared_memory_requirements(self):
        """Overload this function with the number of bytes of shared memory required for a single run of the integrator"""
        return 2*nstates*self.precision().itemsize

    def solve_single(self):

    def solve_batch(self):

    def build_single_kernel(self):

        sharedbytes = self.get_shared_memory_requirements(self)
        @cuda.jit((self.precision[:,:],
                   self.precision[:])
                  )
        def single_kernel(inits,
                          parameters,
                          output,
                          observables
                          ):
            """A single run of the integrator, with a single set of initial conditions and parameters."""
            tx = int16(cuda.threadIdx.x)

            if (tx==1):
                # Allocate shared memory for this thread
                shared_memory = cuda.shared.array(self.get_shared_memory_requirements(),
                                                  dtype=self.precision)

    def build_batch_kernel(self,
                           inits,
                           parameters,
                           output,
                           observables
                           ):
        """A single run of the integrator, with a single set of initial conditions and parameters."""
        tx = int16(cuda.threadIdx.x)
        block_index = int32(cuda.blockIdx.x)
        l_param_set = int32(xblocksize * block_index + tx)
        ):


    def build_loop(self,
                  system,
                  step_size,
                  output_step_size,
                  warmup_time=0,
                  saved_states=None,
                  saved_observables=None):
        """Implementation notes:
        -
        - The integrator loop is compiled once for all threads, even though its contents vary between threads.
        between-thread-variable elements should be passed in device arrays.
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

        - Some user control over storage of temp values should be available, as this offers a performance booth for
        some system sizes.

        -The device function MUST accept varying initial conditions and parameters.
        - We could possibly have a separate inner and outer loop function - the outer loop function can handle saving and
        downsampling. This will be a pain if the function isn't inlined properly by the compiler.

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
        n_saved_states = len(saved_states) if saved_states is not None else nstates
        n_saved_observables = len(saved_observables) if saved_observables is not None else nobs

        precision = system.precision


        dxdt_func = system.dxdt

        @cuda.jit(
                     (self.precision[:],
                      self.precision[:],
                      self.precision[:],
                      self.precision,
                      self.precision,
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


            #Initialise/Assign values to allocated memory
            for i in range(nstates):
                state[:] = inits[i]
            dxdt[:] = precision[0.0]
            l_parameters = cuda.local.array((npar),
                                          dtype=precision)

            for i in range(npar):
                l_parameters[i] = parameters[i]

            # Loop through output samples, one iteration per output sample
            for i in range(warmup_samples + output_length):

                # Euler loop
                for j in range(save_every):
                    driver = forcing_vec[(i*save_every + j) % len(forcing_vec)]

                    # Calculate derivative at sample
                    dxdt_func(state,
                             parameters,
                             driver,
                             observables,
                             dxdt)

                    #Forward-step state using euler
                    for k in range(nstates):
                        state[k] += dxdt[k] * step_size


                #Start saving only after warmup period (to get past transient behaviour)
                if i > (warmup_samples - 1):

                    for n in range(n_saved_states):
                        state_output[i-warmup_samples, n] = output[n]
                    for n in range(n_saved_observables):
                        observables_output[i-warmup_samples, n] = observables[n]

        self.integratorLoop = euler_loop


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
                      forcing_vec,               # Forcing vector (repeating) for non-autonomous SystemModels
                      duration,                  # pass as device constant?
                      step_size,                 # pass as device constant?
                      output_fs,                 # pass as device constant?
                      saved_state_indices,       # Pass as constant device array
                      saved_observable_indices,  # pass as constant device array
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