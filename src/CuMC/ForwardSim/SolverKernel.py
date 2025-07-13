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
from numba.np.numpy_support import as_dtype as to_np_dtype


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
        self.sizes = system.get_sizes()

        self.compile_settings = {'n_saved_states':          len(saved_states),
                                 'n_saved_observables':     len(saved_observables),
                                 'n_state_summaries':       system.states.n_summaries,
                                 'n_observables_summaries': system.observables.n_summaries,
                                 }

        #TODO: Allocate compile settings in a sensible way - singleIntegratorRun should have all of it's own settings
        # saved, and this module just passes them through.

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

    def check_array_shapes(self,
                           device_arrays: dict):
        """ Check shapes of arrays if user has provided them."""


    def allocate_device_arrays(self,
                               duration,
                               params,
                               inits,
                               forcing_vector,
                               _dtype=np.float32,
                               ):

        numruns = params.shape[0] * inits.shape[0]

        output_sizes = self.singleIntegrator.output_sizes(duration)

        #Optimise: CuNODE had arrays strided in [time, run, state] order, which we'll proceed with until there's time
        # or need to examine it. Each run will then load state into L1 cache, and the chunk of the 3d array will be
        # contain all runs' information for a time step (or several, or part of one). Can check here optimality by
        # changing stride order for a few different sized solves.

        #Note: These arrays might be quite large, so pinning and mapping may prove to be performance-negative. Not sure
        # until this rears its head in later optimisation. It was faster on 4GB arrays in testing. For now,
        # we pin them to cut down on memory copies.

        state_output_shape = (output_sizes['state'][0], numruns, output_sizes['state'][1])
        state_output = cuda.mapped_array(state_output_shape,
                                         dtype=_dtype,
                                         )

        observable_output_shape = (output_sizes['observables'][0], numruns, output_sizes['observables'][1])
        observables_output = cuda.mapped_array(observable_output_shape,
                                               dtype=_dtype,
                                               )

        state_summary_shape = (output_sizes['state_summaries'][0], numruns, output_sizes['state_summaries'][1])
        state_summaries_output = cuda.mapped_array(state_summary_shape,
                                                   dtype=_dtype,
                                                   )

        observable_summary_shape = (output_sizes['observables_summaries'][0], numruns, output_sizes['observables_summaries'][1])
        observables_summaries_output = cuda.mapped_array(observable_summary_shape,
                                                         dtype=_dtype,
                                                         )

        #Optimise: Compare to _utils.pinned_zeros to determine if this is faster. This is a once-per-long-kernel
        # allocation so not performance-critical.
        state_output[:, :, :] = 0
        observables_output[:, :, :] = 0
        state_summaries_output[:, :, :] = 0
        observables_summaries_output[:, :, :] = 0

        # For read-only arrays, the memory needs to be allocated regardless, just send them in to be device arrays.
        params = cuda.device_array_like(params)
        inits = cuda.device_array_like(inits)

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

    def kernel(self):
        self.device_function()

    def run(self,
            output_samples,
            device_arrays, #Controversial: it feels wrong to get a higher-level module to call this class' allocation
            # method. However, a higher-level module might reasonably want to reuse existing arrays, or override some
            # other memory allocation idea.
            numruns,
            runs_per_block=32,
            stream=0,
            warmup_samples=0
            ):

        BLOCKSPERGRID = int(max(1, np.ceil(numruns / self.xblocksize)))
        dynamic_sharedmem = self.shared_memory_bytes_per_run * numruns

        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1" and self._profileCUDA:
            cuda.profile_start()

        self.kernel[BLOCKSPERGRID, (self.xblocksize, runs_per_block), stream, dynamic_sharedmem](
                device_arrays['inits'],
                device_arrays['params'],
                device_arrays['forcing_vector'],
                device_arrays['state_output'],
                device_arrays['observables_output'],
                device_arrays['state_summaries_output'],
                device_arrays['observables_summaries_output'],
                output_samples,
                warmup_samples=warmup_samples,
                n_runs=numruns,
                )
        cuda.synchronize()
        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1" and self._profileCUDA:
            cuda.profile_stop()

        outputarrays = self.fetch_output_arrays(device_arrays)

        return outputarrays

    def build_kernel(self):
        """

        """
        precision = self.precision
        loopfunction = self.singleIntegrator.device_function
        shared_elements_per_run = self.shared_memory_elements_per_run

        @cuda.jit((precision[:, :],
                   precision[:, :],
                   precision[:, :],
                   precision[:, :, :],
                   precision[:, :, :],
                   precision[:, :, :],
                   precision[:, :, :],
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

            tx = int16(cuda.threadIdx.x)  # Intra-loop index (always 0 for single-threaded loops)
            ty = int16(cuda.threadIdx.y)  # per-run index
            threads_per_loop = int16(cuda.blockDim.x)

            block_index = int32(cuda.blockIdx.x)
            runs_per_block = cuda.blockDim.y
            run_index = int32(runs_per_block * block_index + tx)

            if run_index >= n_runs:
                return None

            n_init_sets = inits.shape[0]
            init_index = run_index % n_init_sets
            param_index = run_index // n_init_sets

            shared_memory = cuda.shared.array(0,
                                              dtype=precision,
                                              )

            #Put forcing vector in constant memory as it will be broadcast-only
            c_forcing_vector = cuda.const.array_like(forcing_vector)

            #Run-indexed (rx) slices and allocations of shared and output memory. Allow the loop to handle distrubution
            # amongst threads if it's multi-threaded using the x block index.
            rx_shared_memory = shared_memory[ty * shared_elements_per_run:(ty + 1) * shared_elements_per_run]
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
                    rx_shared_memory,
                    rx_state,
                    rx_observables,
                    rx_state_summaries,
                    rx_observables_summaries,
                    duration_samples,
                    warmup_samples,
                    )

            return None

        return integration_kernel

    @property
    def shared_memory_bytes_per_run(self):
        """Returns the number of bytes of shared memory required for each run."""
        return self.singleIntegrator.shared_memory_bytes

    @property
    def shared_memory_elements_per_run(self):
        """Returns the number of elements (values) in shared memory required for each run."""
        return self.singleIntegrator.shared_memory_elements

    @property
    def precision(self):
        """Returns the precision (numba data type) used by the solver kernel."""
        return self.singleIntegrator.precision

    @property
    def xblocksize(self):
        """Returns the number of threads in the x dimension of the block."""
        return self.singleIntegrator.threads_per_loop
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