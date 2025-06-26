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


class SolverKernel():
    """Class which builds and uses the outer kernel for solving ODEs. This kernel handles interfacing with the
    whole-batch (if used) parameter sets and input/output arrays, assigning one run to an ODEIntegrator device function
    for each run, and copying the results back to host.

    methods batch_solve and single_solve are the main entry points for running the solver, depending on whether
    you want to do a single run or a batch (no surprises there)."""
    def __init__(self,
                 precision,
                 profileFlags=False):

        self.kernel = None
        self.precision = precision
        self.numba_precision = from_dtype(precision)

    #TODO: force rebuild if system or saved states change


    def build_all(self,
                system,
                integrator_algorithm,
                saved_states=None,
                saved_observables=None,
                xblocksize=128,
                **integrator_kwargs):

        self.xblocksize=xblocksize
        if saved_states is not None:
            self.saved_states = system.states.get_indices(saved_states)
        else:
            saved_states = self.saved_states

        if saved_observables is not None:
            self.saved_observables = system.observables.get_indices(saved_observables)
        else:
            saved_observables = self.saved_observables


        system.build()
        self.build_loop(system,
                        integrator_kwargs["internal_step_size"],
                        integrator_kwargs["output_step_size"],
                        saved_states,
                        saved_observables)

        self.build_kernel(system,
                          self.integratorLoop,
                          xblocksize)


    def get_shared_memory_requirements(self):
        """Returns the number of bytes of shared memory required for the system."""
        return self.integratorLoop.get_dynamic_shared_memory_per_thread()

    def solve(self,
              duration,
              warmup,
              params_2d,
              inits_2d,
              forcing_vector):

        numruns = params_2d.shape[0] * inits_2d.shape[0]
        #TODO: check indexing orders for learnings from last CUDA-fest
        state_output = cuda.pinned_array((numruns, self.output_samples, self.system.num_states), dtype=self.precision)
        observables_output = cuda.pinned_array((numruns, self.output_samples, self.system.num_observables), dtype=self.precision)
        state_summaries_output = cuda.pinned_array((numruns, self.output_samples, self.system.num_states), dtype=self.precision)
        observables_summaries_output = cuda.pinned_array((numruns, self.output_samples, self.system.num_observables),
                                        dtype=self.precision)
        state_output[:, :, :] = 0
        observables_output[:, :, :] = 0
        state_summaries_output[:, :, :] = 0
        observables_summaries_output[:, :, :] = 0

        params = cuda.device_array_like(params_2d)
        inits = cuda.device_array_like(inits_2d)

        d_state_output = cuda.to_device(state_output)
        d_observables_output = cuda.to_device(observables_output)
        d_state_summaries_output = cuda.to_device(state_summaries_output)
        d_observables_summaries_output = cuda.to_device(observables_summaries_output)
        d_params = cuda.to_device(params)
        d_inits = cuda.to_device(inits)
        d_forcing = cuda.to_device(forcing_vector)

        BLOCKSPERGRID = int(max(1, np.ceil(numruns / self.xblocksize)))
        dynamic_sharedmem = self.get_shared_memory_requirements() * numruns

        output_samples = duration / self.integratorLoop.dt_save
        warmup_samples = warmup / self.integratorLoop.dt_save


        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1" and self.profileFlags:
            cuda.profile_start()

        self.kernel[BLOCKSPERGRID, self.xblocksize, 0, dynamic_sharedmem](d_inits,
                                                                          d_params,
                                                                          d_forcing,
                                                                          d_state_output,
                                                                          d_observables_output,
                                                                          d_state_summaries_output,
                                                                          d_observables_summaries_output,
                                                                          self.output_samples,
                                                                          warmup_samples=0,
                                                                          n_runs=numruns)

        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1" and self.profileFlags:
            cuda.profile_stop()

        observables = d_observables_output.copy_to_host()
        output = d_state_output.copy_to_host()
        state_summaries_output = d_state_summaries_output.copy_to_host()
        observables_summaries_output = d_observables_summaries_output.copy_to_host()

        return output, observables, state_summaries_output, observables_summaries_output

    def get_set_at_output_index(self, inits_3d, params_3d, idx):
        """ Returns the initial values and parameters for a specific output index."""
        num_inits = inits_3d.shape[0]
        init_index = idx % num_inits
        param_index = idx // num_inits
        return inits_3d[init_index, :], params_3d[param_index, :]

    def build_kernel(self,
                    integrator_algorithm,
                    xblocksize=128):
        """Builds the kernel that will run the integrator algorithm. Accepts a pre-built integrator algorithm function,
        which will need to be built with the same parameters as this kernel (namely: number of states,"""

        shared_bytes = self.get_shared_memory_requirements()
        numba_precision = self.numba_precision



        @cuda.jit((numba_precision[:,:],
                   numba_precision[:,:],
                   numba_precision[:,:],
                   numba_precision[:,:,:],
                   numba_precision[:,:,:],
                   numba_precision[:, :, :],
                   numba_precision[:, :, :],
                   int32,
                   int32,
                   int32
                  ))
        def integration_kernel(inits,
                            params,
                            forcing_vector,
                            state_output,
                            observables_output,
                            state_summaries_output,
                            observables_summaries_output,
                            duration_samples,
                            warmup_samples=0,
                            n_runs=1,
                            ):
            """Master integration kernel - calls integratorLoop and dxdt device functions.
            Accepts all sets of inits, params, and runs one in each thread.

            TODO: Currently saves forcing vector as a constant array. Consider moving this to a device function, which takes
                time and thread index as arguments, returning a forcing value.
            """
            tx = int16(cuda.threadIdx.x)
            block_index = int32(cuda.blockIdx.x)
            run_index = int32(xblocksize * block_index + tx)

            if run_index >= n_runs:
                return None

            n_init_sets = inits.shape[0]
            init_index = run_index % n_init_sets
            param_index = run_index // n_init_sets

            # Allocate shared memory for this thread
            c_forcing_vector = cuda.const.array_like(forcing_vector)
            shared_memory = cuda.shared.array(0,
                                              dtype=precision)

            # Get this thread's portions of shared memory, init values, parameters, outputs.
            # tx_ indicates a thread-specific allocation, while rx_ indicates a run-specific allocation. These may be different
            # if we implement a solver which uses multiple threads per run.
            #TODO: Include a threads-per-run parameter to the kernel, which will allow us to use multiple threads per run.
            # Think at least a little bit harder about how that would work.
            tx_shared_memory = shared_memory[tx*shared_bytes:(tx+1)*shared_bytes]
            rx_inits = inits[init_index,:]
            rx_params = params[param_index,:]
            rx_state = state_output[run_index,:,:] #TODO: Check slice order learnings from last CUDA-fest
            rx_observables = observables_output[run_index,:,:]
            rx_state_summaries = state_summaries_output[run_index,:,:]
            rx_observables_summaries = observables_summaries_output[run_index,:,:]

            integrator_algorithm(
                rx_inits,
                rx_params,
                c_forcing_vector,
                tx_shared_memory,
                rx_state,
                rx_observables,
                rx_state_summaries,
                rx_observables_summaries,
                duration_samples,
                warmup_samples
                )

            return Nonee

        self.kernel = integration_kernel

    #This is essentially a static method except for the return to an attribute - consider moving loops into a function
    # instead to remove the clutter. These would be defined per-algorithm in their own file, so we can potentially avoid
    # subclassing and just have a single function for each algorithm.

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
