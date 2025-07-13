# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""

import os
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
        self.sizes = system.get_sizes()

        self.setup_compile_settings({'n_saved_states':          len(saved_states),
                                     'n_saved_observables':     len(saved_observables),
                                     },
                                    )

        #TODO: Allocate compile settings in a sensible way - singleIntegratorRun should have all of it's own settings
        # saved, and this module just passes them through.

        #All run settings might be more appropriate in the higher-level interface model?

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

    def _check_input_array_shapes(self,
                                  input_device_arrays,
                                  numruns,
                                  ):
        """Check shapes of input arrays match the ordered run if user has provided them. If shapes do not match, raise
        an error to alert the user that their convenience method of feeding back preallocated arrays is not working."""
        correct_sizes = {'forcing_vector': self.sizes['n_drivers'],
                         'params':          self.sizes['n_parameters'],
                         'inits':           self.sizes['n_states']
                         }

        for key, array in input_device_arrays.items():
            if key not in ['params', 'inits', 'forcing_vector']:
                raise ValueError(f"Input device arrays do not contain expected key '{key}'. "
                                 f"Available keys: {list(input_device_arrays.keys())}",
                                 )

            if array.shape[1] != correct_sizes[key]:
                raise ValueError(f"Supplied input device array for '{key}' does not match expected shape. ")

        array_suggested_numruns = input_device_arrays['params'].shape[0] * input_device_arrays['inits'].shape[0]
        if array_suggested_numruns != numruns:
            raise ValueError(f"Supplied input device arrays suggest {array_suggested_numruns} runs, "
                             f"but numruns is {numruns}. Please check your input arrays and numruns value.",
                             )

    def _check_output_array_shapes(self,
                                   output_device_arrays: dict,
                                   duration: float,
                                   numruns,
                                   ):
        """Check shapes of output arrays match the ordered run if user has provided them. If shapes do not match,
        raise an error to alert the user that their convenience method of feeding back preallocated arrays is not
        working"""
        output_sizes = self.singleIntegrator.output_sizes(duration)

        # HACK: Set zero-sized arrays to 1 to avoid an invalid memory allocation in CUDA.
        for key, size in output_sizes.items():
            if any(dim <= 0 for dim in size):
                output_sizes[key] = (1, 1)

        for key, shape in output_sizes.items():
            if key not in output_device_arrays:
                raise ValueError(f"Output device arrays do not contain expected key '{key}'. "
                                 f"Available keys: {list(output_device_arrays.keys())}",
                                 )

            supplied_shape = output_device_arrays[key].shape
            correct_shape = (output_sizes[key][0], numruns, output_sizes[key][1])
            if not np.all(supplied_shape == correct_shape):
                raise ValueError(f"Supplied output device array for '{key}' does not match expected shape. "
                                 f"Expected shape: {correct_shape}, got: {supplied_shape}. ",
                                 )

    def allocate_output_arrays(self,
                               duration,
                               numruns,
                               _dtype,
                               ):

        #Optimise: CuNODE had arrays strided in [time, run, state] order, which we'll proceed with until there's time
        # or need to examine it. Intended behaviour is for each run to load state into L1 cache, and the chunk of
        # the 3d array will contain all runs' information for a time step (or several, or part of one). Can check for
        # optimality by changing stride order for a few different sized solves.

        #Note: These arrays might be quite large, so pinning and mapping may prove to be performance-negative. Not sure
        # until this rears its head in later optimisation. It was faster on 4GB arrays in testing. For now,
        # we pin them to cut down on memory copies.
        output_arrays = {}
        output_sizes = self.singleIntegrator.output_sizes(duration)

        # For each of state, observables, state summaries, and observables summaries:
        for key, size in output_sizes.items():
            # HACK: Set zero-sized arrays to 1 to avoid an invalid memory allocation in CUDA.
            if any(dim <= 0 for dim in size):
                output_sizes[key] = (1, 1)
            shape = (output_sizes[key][0], numruns, output_sizes[key][1])
            output_arrays[key] = cuda.mapped_array(shape, dtype=_dtype)
            output_arrays[key][:, :, :] = 0.0  # Initialise to zero

        return output_arrays

    def fetch_output_arrays(self, device_arrays):
        """Returns host-accessible arrays from by the solver kernel. No transfers are performed here due to the use
        of mapped arrays - if these are changed to device or pinned arrays, array.copy_to_host() will need to be
        called on each of these arrays to transfer them to the host."""
        host_arrays = {'state':                 device_arrays['state'],
                       'observables':           device_arrays['observables'],
                       'state_summaries':       device_arrays['state_summaries'],
                       'observables_summaries': device_arrays['observable_summaries']
                       }

        return host_arrays

    def check_or_allocate_input_arrays(self,
                                       input_arrays,
                                       numruns,
                                       ):
        for label, array in input_arrays.items():
            if isinstance(array, np.ndarray):
                input_arrays[label] = cuda.to_device(array)
            elif isinstance(array, cuda.devicearray.DeviceNDArray):
                # Already on device, do nothing
                pass
            else:
                raise TypeError(f"Input array '{label}' must be a numpy array or a Numba device array, "
                                f"got {type(array)} instead.",
                                )

        self._check_input_array_shapes(input_device_arrays=input_arrays,
                                       numruns=numruns,
                                       )
        return input_arrays

    def check_or_allocate_output_arrays(self,
                                        output_arrays,
                                        duration,
                                        numruns,
                                        ):
        if output_arrays is None:
            output_arrays = self.allocate_output_arrays(duration=duration,
                                                        numruns=numruns,
                                                        _dtype=to_np_dtype(self.precision),
                                                        )
        #TODO: Add test for giving an incorrect dtype output array, then make it pass.
        for label, array in output_arrays.items():
            if isinstance(array, np.ndarray):
                # There's not much benefit to providing preallocated arrays if they need transferred anyway.
                output_arrays[label] = cuda.to_device(array)
            elif isinstance(array, cuda.devicearray.DeviceNDArray):
                # Already on device, do nothing
                pass
            else:
                raise TypeError(f"Output array '{label}' must be a numpy array or a Numba device array, "
                                f"got {type(array)} instead.",
                                )

        self._check_output_array_shapes(output_arrays,
                                               duration,
                                               numruns=numruns,
                                               )

        return output_arrays

    @property
    def kernel(self):
        return self.device_function

    def build(self):
        return self.build_kernel()

    def run(self,
            duration,
            numruns,
            params,
            inits,
            forcing_vectors,
            output_arrays=None,
            runs_per_block=32,
            stream=0,
            warmup=0.0,
            ):

        output_arrays = self.check_or_allocate_output_arrays(output_arrays,
                                                             duration=duration,
                                                             numruns=numruns,
                                                             )
        input_arrays = {'params':         params,
                        'inits':          inits,
                        'forcing_vector': forcing_vectors,
                        }
        input_arrays = self.check_or_allocate_input_arrays(input_arrays,
                                                           numruns=numruns,
                                                           )

        BLOCKSPERGRID = int(max(1, np.ceil(numruns / self.xblocksize)))
        dynamic_sharedmem = self.shared_memory_bytes_per_run * numruns

        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1" and self._profileCUDA:
            cuda.profile_start()

        self.device_function[BLOCKSPERGRID, (self.xblocksize, runs_per_block), stream, dynamic_sharedmem](
                input_arrays['inits'],
                input_arrays['params'],
                input_arrays['forcing_vector'],
                output_arrays['state'],
                output_arrays['observables'],
                output_arrays['state_summaries'],
                output_arrays['observable_summaries'],
                self.output_length(duration),
                self.output_length(warmup),
                numruns,
                )
        cuda.synchronize()

        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1" and self._profileCUDA:
            cuda.profile_stop()

        outputarrays = self.fetch_output_arrays(output_arrays)

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

    def output_length(self, duration):
        """Returns the number of output samples per run."""
        return int(np.ceil(duration / self.singleIntegrator.dt_save))

    def summaries_length(self, duration):
        """Returns the number of summary samples per run."""
        return int(np.ceil(duration / self.singleIntegrator.dt_summarise))
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