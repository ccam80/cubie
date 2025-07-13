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
from numba import int32, int16
from numba import cuda, from_dtype
from CuMC.ForwardSim.integrators.SingleIntegratorRun import SingleIntegratorRun
from CuMC._utils import pinned_zeros
from CuMC.CUDAFactory import CUDAFactory
from numpy.typing import NDArray


class SolverKernel(CUDAFactory):
    """Class which builds and holds the integrating kernel and interfaces with lower-level modules: loop
    algorithms, ODE systems, and output functions
    The kernel function accepts single or batched sets of inputs, and distributes those amongst the threads on the
    GPU. It runs the loop device function on a given slice of it's allocated memory, and serves as the distributor
    of work amongst the individual runs of the integrators.
    This class is one level down from the user, managing sanitised inputs and handling the machinery of batching and
    running integrators. It does not handle:
     - Integration logic/algorithms - these are handled in SingleIntegratorRun, and below
     - Input sanitisation / batch construction - this is handled in the solver api.
     - System equations - these are handled in the system model classes.
    """

    def __init__(self,
                 system,
                 precision: np.dtype = np.float32,  #Is this the point where we accept the numpy dtype, since we're
                 # almost user-facing?
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
                 profileCUDA: bool = False,
                 ):
        super().__init__()
        self._profileCUDA = profileCUDA
        self.kernel = None
        self.precision = precision
        self.numba_precision = from_dtype(precision)
        self.sizes = system.get_sizes()

        self.compile_settings = {'n_saved_states': len(saved_states),
                                 'n_saved_observables': len(saved_observables),
                                 'n_state_summaries': system.states.n_summaries,
                                    'n_observables_summaries': system.observables.n_summaries,

                                 }


        #All run settings might be more appropriate in the higher-level interface model?
        self.run_settings = self.initialize_run_settings()

        self.singleIntegrator = SingleIntegratorRun(system,
                                                    algorithm=algorithm,
                                                    dt_min=dt_min,
                                                    dt_max=dt_max,
                                                    dt_save=dt_save,
                                                    dt_summarise=dt_summarise,
                                                    atol=atol,
                                                    rtol=rtol,
                                                    saved_states=saved_states,
                                                    saved_observables=saved_observables,
                                                    output_types=output_types,
                                                    n_peaks=n_peaks,
                                                    )

    def build(self,
              system,
              integrator_algorithm,
              saved_states=None,
              saved_observables=None,
              xblocksize=128,
              **integrator_kwargs,
              ):

        self.xblocksize = xblocksize
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
                        saved_observables,
                        )

        self.build_kernel(system,
                          self.integratorLoop,
                          xblocksize,
                          )

    def get_shared_memory_requirements(self):
        """Returns the number of bytes of shared memory required for the system."""
        return self.integratorLoop.get_dynamic_shared_memory_per_thread()

    def allocate_device_arrays(self,
                               output_length,
                               summary_length,
                               n_saved_states,
                               n_saved_observables,
                               n_state_summaries,
                               n_observables_summaries,
                               paramsets,
                               initsets,
                               forcing_vector=None,
                               _dtype=np.float32,
                               ):

        numruns = paramsets.shape[0] * initsets.shape[0]

        #Optimise: CuNODE had arrays strided in [time, run, state] order, which we'll proceed with until there's time
        # or need to examine it. Each run will then load state into L1 cache, and the chunk of the 3d array will be
        # contain all runs' information for a time step (or several, or part of one). Can check here optimality by
        # changing stride order for a few different sized solves.

        #Note: These arrays might be quite large, so pinning and mapping may prove to be performance-negative. Not sure
        # until this rears its head in later optimisation. It was faster on 4GB arrays in testing. For now,
        # we pin them to cut down on memory copies.
        state_output = cuda.mapped_array((numruns,
                                          output_length,
                                          n_saved_states
                                          ),
                                         dtype=_dtype,
                                         )
        observables_output = cuda.mapped_array((numruns,
                                                output_length,
                                                n_saved_observables
                                                ),
                                               dtype=_dtype,
                                               )
        state_summaries_output = cuda.mapped_array((numruns,
                                                    summary_length,
                                                    n_state_summaries
                                                    ),
                                                   dtype=_dtype,
                                                   )
        observables_summaries_output = cuda.mapped_array((numruns,
                                                          summary_length,
                                                          n_observables_summaries
                                                          ),
                                                         dtype=_dtype,
                                                         )

        #Optimise: Compare to _utils.pinned_zeros to determine if this is faster. This is a once-per-long-kernel
        # allocation so not performance-critical.
        state_output[:, :, :] = 0
        observables_output[:, :, :] = 0
        state_summaries_output[:, :, :] = 0
        observables_summaries_output[:, :, :] = 0

        # For read-only arrays, the memory needs to be allocated regardless, just send them in to be device arrays.
        params = cuda.device_array_like(paramsets)
        inits = cuda.device_array_like(initsets)

        if forcing_vector is not None:
            forcing_vector = cuda.device_array_like(forcing_vector)

        device_arrays = {
            "state_output":                 state_output,
            "observables_output":           observables_output,
            "state_summaries_output":       state_summaries_output,
            "observables_summaries_output": observables_summaries_output,
            "params":                       params,
            "inits":                        inits,
            "forcing_vector":               forcing_vector
            }

        return device_arrays

    def fetch_output_arrays(self, device_arrays):
        """Returns host-accessible arrays from by the solver kernel. No transfers are performed here due to the use
        of mapped arrays - if these are changed to device or pinned arrays, array.copy_to_host() will need to be
        called on each of these arrays to transfer them to the host."""
        host_arrays = {'state':                 device_arrays['state_output'],
                       'observables':           device_arrays['observables_output'],
                       'state_summaries':       device_arrays['state_summaries_output'],
                       'observables_summaries': device_arrays['observables_summaries_output']
                       }

        return host_arrays

    def run(self,
            param_sets,
            inits_sets,
            forcing_vector,
            stream=0,
            ):
        # TODO [$6873084f7cdbf00008a72cfd]: Check the ramifications of returning this function as the .device_function of the CUDA factory. It
        #  contains a CUDA kernel, but this supporting method that handles the setup is also CUDA-side rather than
        #  user-side, so it may be appropriate to return the whole run function.

        output_length = self.run_settings['output_length']
        warmup_samples = self.run_settings['warmup_samples']
        summary_samples = self.run_settings['summary_length']
        n_saved_states = self.compile_settings['n_saved_states']
        n_saved_observables = self.compile_settings['n_saved_observables']
        n_state_summaries = self.compile_settings['n_state_summaries']
        n_observables_summaries = self.compile_settings['n_observables_summaries']
        numruns = param_sets.shape[0] * inits_sets.shape[0]

        device_arrays = self.allocate_device_arrays(output_length,
                                                    summary_samples,
                                                    n_saved_states,
                                                    n_saved_observables,
                                                    n_state_summaries,
                                                    n_observables_summaries,
                                                    param_sets,
                                                    inits_sets,
                                                    forcing_vector,
                                                    _dtype=self.precision,
                                                    )

        d_state_output = device_arrays['state_output']
        d_observables_output = device_arrays['observables_output']
        d_state_summaries_output = device_arrays['state_summaries_output']
        d_observables_summaries_output = device_arrays['observables_summaries_output']
        d_params = device_arrays['params']
        d_inits = device_arrays['inits']
        d_forcing = device_arrays['forcing_vector']

        BLOCKSPERGRID = int(max(1, np.ceil(numruns / self.xblocksize)))
        dynamic_sharedmem = self.get_shared_memory_requirements() * numruns

        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1" and self._profileCUDA:
            cuda.profile_start()

        self.kernel[BLOCKSPERGRID, self.xblocksize, stream, dynamic_sharedmem](d_inits,
                                                                               d_params,
                                                                               d_forcing,
                                                                               d_state_output,
                                                                               d_observables_output,
                                                                               d_state_summaries_output,
                                                                               d_observables_summaries_output,
                                                                               self.output_samples,
                                                                               warmup_samples=warmup_samples,
                                                                               n_runs=numruns,
                                                                               )

        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1" and self._profileCUDA:
            cuda.profile_stop()

        observables = d_observables_output.copy_to_host()
        output = d_state_output.copy_to_host()
        state_summaries_output = d_state_summaries_output.copy_to_host()
        observables_summaries_output = d_observables_summaries_output.copy_to_host()

        return output, observables, state_summaries_output, observables_summaries_output

    def build(self):
        """

        """
        precision = self.numba_precision
        threadsperloop = self.compile_settings['threadsperloop']
        loopfunction = self.singleIntegrator.device_function
        shared_bytes_per_thread = self.get_shared_memory_requirements()
        numba_precision = self.numba_precision

        @cuda.jit((numba_precision[:, :],
                   numba_precision[:, :],
                   numba_precision[:, :],
                   numba_precision[:, :, :],
                   numba_precision[:, :, :],
                   numba_precision[:, :, :],
                   numba_precision[:, :, :],
                   int32,
                   int32,
                   int32
                   ),
                  )
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

            which takes
                time and thread index as arguments, returning a forcing value.
            """

            tx = int16(cuda.threadIdx.x)

            block_index = int32(cuda.blockIdx.x)
            block_width = cuda.blockDim.x
            run_index = int32(block_width * block_index + tx)

            if run_index >= n_runs:
                return None

            n_init_sets = inits.shape[0]
            init_index = run_index % n_init_sets
            param_index = run_index // n_init_sets

            # Allocate shared memory for this thread
            c_forcing_vector = cuda.const.array_like(forcing_vector)
            shared_memory = cuda.shared.array(0,
                                              dtype=precision,
                                              )

            # Get this thread's portions of shared memory, init values, parameters, outputs.
            # tx_ indicates a thread-specific allocation, while rx_ indicates a run-specific allocation. These may be different
            # if we implement a solver which uses multiple threads per run.
            #TODO: Figure out allocation in multi-thread runs - each should operate out of a shared pointer,
            # but shared_bytes_per_thread may need to be renamed or complemented by a shared_bytes_per_run to allow
            # this.
            # Working concept: We use a 2d block for a multi-thread run. This will require a thread-aware loop
            # function, and a kernel built with a different blocksize config (I think). To keep it in a single kernel,
            # we could use a compile-time constant toggle to set multi- or single-threaded run, and set second dim to
            # 0 for single-threaded runs.
            # Confirmed that we can request the y index of a 1d grid - it just returns zero.
            tx_shared_memory = shared_memory[tx * shared_bytes_per_thread:(tx + 1) * shared_bytes_per_thread]

            rx_inits = inits[init_index, :]
            rx_params = params[param_index, :]
            rx_state = state_output[:, run_index, :]
            rx_observables = observables_output[:, run_index, :]
            rx_state_summaries = state_summaries_output[:, run_index, :]
            rx_observables_summaries = observables_summaries_output[:, run_index, :]

            loopfunction(
                    rx_inits,
                    rx_params,
                    c_forcing_vector,
                    tx_shared_memory,
                    rx_state,
                    rx_observables,
                    rx_state_summaries,
                    rx_observables_summaries,
                    duration_samples,
                    warmup_samples,
                    )

        return integration_kernel

    #This is essentially a static method except for the return to an attribute - consider moving loops into a function
    # instead to remove the clutter. These would be defined per-algorithm in their own file, so we can potentially avoid
    # subclassing and just have a single function for each algorithm.

#
# if __name__ == "__main__":
#     from CuMC.SystemModels.Systems.threeCM import ThreeChamberModel
#
#     precision = np.float32
#     numba_precision = from_dtype(precision)
#     sys = ThreeChamberModel(precision=precision)
#     sys.build()
#
#     internal_step = 0.001
#     save_step = 0.01
#     duration = 0.1
#     warmup = 0.1
#
#     output_samples = int(duration / save_step)
#     warmup_samples = int(warmup / save_step)
#
#     integrator = genericODEIntegrator(precision=precision)
#     integrator.build_loop(sys,
#                           internal_step,
#                           save_step,
#                           )
#     intfunc = integrator.integratorLoop
#
#
#     @cuda.jit()
#     def loop_test_kernel(inits,
#                          params,
#                          forcing_vector,
#                          output,
#                          observables,
#                          ):
#         c_forcing_vector = cuda.const.array_like(forcing_vector)
#
#         shared_memory = cuda.shared.array(0, dtype=numba_precision)
#
#         intfunc(
#                 inits,
#                 params,
#                 c_forcing_vector,
#                 shared_memory,
#                 output,
#                 observables,
#                 output_samples,
#                 warmup_samples,
#                 )
#
#
#     output = cuda.pinned_array((output_samples, sys.num_states), dtype=precision)
#     observables = cuda.pinned_array((output_samples, sys.num_observables), dtype=precision)
#     forcing_vector = cuda.pinned_array((output_samples + warmup_samples, sys.num_drivers), dtype=precision)
#
#     output[:, :] = precision(0.0)
#     observables[:, :] = precision(0.0)
#     forcing_vector[:, :] = precision(0.0)
#     forcing_vector[0, :] = precision(1.0)
#
#     d_forcing = cuda.to_device(forcing_vector)
#     d_inits = cuda.to_device(sys.init_values.values_array)
#     d_params = cuda.to_device(sys.parameters.values_array)
#     d_output = cuda.to_device(output)
#     d_observables = cuda.to_device(observables)
#
#     # global sharedmem
#     sharedmem = integrator.get_shared_memory_requirements(sys)
#
#     loop_test_kernel[1, 1, 0, sharedmem](d_inits,
#                                          d_params,
#                                          d_forcing,
#                                          d_output,
#                                          d_observables,
#                                          )
#
#     cuda.synchronize()
#     output = d_output.copy_to_host()
#     obs = d_observables.copy_to_host()
#     print(output)
#     print(obs)