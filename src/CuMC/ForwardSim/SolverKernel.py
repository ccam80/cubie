# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""

import os
import numpy as np
from numba import int32, int16
from numba import cuda
from numba.cuda.cudadrv.devicearray import is_cuda_ndarray
from CuMC.ForwardSim.integrators.SingleIntegratorRun import SingleIntegratorRun
from CuMC.CUDAFactory import CUDAFactory
from numpy.typing import NDArray
from numba.np.numpy_support import as_dtype as to_np_dtype
import attrs


@attrs.define
class SolverKernelConfig:
    """Configuration for the solver kernel."""
    n_saved_states: int = attrs.field(validator=attrs.validators.instance_of(int))
    n_saved_observables: int = attrs.field(validator=attrs.validators.instance_of(int))


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

        self.sizes = system.sizes()

        # Setup compile settings for the kernel
        self.setup_compile_settings({'n_saved_states':      len(saved_states),
                                     'n_saved_observables': len(saved_observables),
                                     },
                                    )

        # TODO: Allocate compile settings in a sensible way - singleIntegratorRun should have all of it's own settings
        # saved, and this module just passes them through.
        # Initialize the single integrator run
        self.singleIntegrator = SingleIntegratorRun(
                system,
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
                         'params':         self.sizes['n_parameters'],
                         'inits':          self.sizes['n_states']
                         }
        for key, array in input_device_arrays.items():
            if key not in correct_sizes:
                raise ValueError(f"Input device arrays contain unexpected key '{key}'. "
                                 f"Expected keys: {list(correct_sizes.keys())}",
                                 )

            if array.shape[1] != correct_sizes[key]:
                raise ValueError(f"Input array '{key}' has incorrect shape. "
                                 f"Expected {correct_sizes[key]} elements, got {array.shape[1]}",
                                 )

        array_suggested_numruns = input_device_arrays['params'].shape[0] * input_device_arrays['inits'].shape[0]
        if array_suggested_numruns != numruns:
            raise ValueError(f"Input arrays suggest {array_suggested_numruns} runs, "
                             f"but numruns is {numruns}",
                             )

    def _check_output_array_shapes(self, output_device_arrays: dict, duration: float, numruns):
        """Check shapes of output arrays match the expected dimensions."""
        output_sizes = self.singleIntegrator.output_sizes(duration)

        # Set zero-sized arrays to 1 to avoid invalid memory allocation in CUDA
        for key, size in output_sizes.items():
            if any(dim <= 0 for dim in size):
                output_sizes[key] = (1, 1)

        for key, shape in output_sizes.items():
            if key not in output_device_arrays:
                raise ValueError(f"Output arrays missing expected key '{key}'. "
                                 f"Available keys: {list(output_device_arrays.keys())}",
                                 )

            supplied_shape = output_device_arrays[key].shape
            correct_shape = (output_sizes[key][0], numruns, output_sizes[key][1])
            if not np.all(supplied_shape == correct_shape):
                raise ValueError(f"Output array '{key}' has incorrect shape. "
                                 f"Expected {correct_shape}, got {supplied_shape}",
                                 )

    def allocate_output_arrays(self, duration, numruns, _dtype):
        # Optimise: CuNODE had arrays strided in [time, run, state] order, which we'll proceed with until there's time
        # or need to examine it. Intended behaviour is for each run to load state into L1 cache, and the chunk of
        # the 3d array will contain all runs' information for a time step (or several, or part of one). Can check for
        # optimality by changing stride order for a few different sized solves.

        # Note: These arrays might be quite large, so pinning and mapping may prove to be performance-negative. Not sure
        # until this rears its head in later optimisation. It was faster on 4GB arrays in testing. For now,
        # we pin them to cut down on memory copies.
        output_arrays = {}
        output_sizes = self.singleIntegrator.output_sizes(duration)

        for key, size in output_sizes.items():
            # Set zero-sized arrays to 1 to avoid invalid memory allocation in CUDA
            if any(dim <= 0 for dim in size):
                output_sizes[key] = (1, 1)

            shape = (output_sizes[key][0], numruns, output_sizes[key][1])
            output_arrays[key] = cuda.mapped_array(shape, dtype=_dtype)
            output_arrays[key][:, :, :] = 0.0

        return output_arrays

    def fetch_output_arrays(self, device_arrays):
        """Returns host-accessible arrays from the solver kernel."""
        return {
            'state':                 device_arrays['state'],
            'observables':           device_arrays['observables'],
            'state_summaries':       device_arrays['state_summaries'],
            'observables_summaries': device_arrays['observable_summaries']
            }

    def check_or_allocate_input_arrays(self, input_arrays, numruns):
        """Check or convert input arrays to device arrays."""
        for label, array in input_arrays.items():
            if isinstance(array, np.ndarray):
                input_arrays[label] = cuda.to_device(array)
            elif not is_cuda_ndarray(array):
                raise TypeError(f"Input array '{label}' must be a numpy array or Numba device array, "
                                f"got {type(array)}",
                                )

        self._check_input_array_shapes(input_arrays, numruns)
        return input_arrays

    def check_or_allocate_output_arrays(self, output_arrays, duration, numruns):
        """Check or allocate output arrays."""
        # TODO: Add test for a provided array of the wrong type
        if output_arrays is None:
            output_arrays = self.allocate_output_arrays(
                    duration=duration,
                    numruns=numruns,
                    _dtype=to_np_dtype(self.precision),
                    )

        for label, array in output_arrays.items():
            if isinstance(array, np.ndarray):
                output_arrays[label] = cuda.to_device(array)
            elif not is_cuda_ndarray(array):
                raise TypeError(f"Output array '{label}' must be a numpy array or Numba device array, "
                                f"got {type(array)}",
                                )

        self._check_output_array_shapes(output_arrays, duration, numruns)
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
        """Run the solver kernel."""
        output_arrays = self.check_or_allocate_output_arrays(
                output_arrays, duration=duration, numruns=numruns,
                )

        input_arrays = {
            'params':         params,
            'inits':          inits,
            'forcing_vector': forcing_vectors,
            }
        input_arrays = self.check_or_allocate_input_arrays(input_arrays, numruns)

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

        return self.fetch_output_arrays(output_arrays)

    def build_kernel(self):
        """Build the integration kernel."""
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
            """Master integration kernel - calls integratorLoop and dxdt device functions."""
            tx = int16(cuda.threadIdx.x)
            ty = int16(cuda.threadIdx.y)
            threads_per_loop = int16(cuda.blockDim.x)

            block_index = int32(cuda.blockIdx.x)
            runs_per_block = cuda.blockDim.y
            run_index = int32(runs_per_block * block_index + tx)

            if run_index >= n_runs:
                return None

            n_init_sets = inits.shape[0]
            init_index = run_index % n_init_sets
            param_index = run_index // n_init_sets

            shared_memory = cuda.shared.array(0, dtype=precision)
            c_forcing_vector = cuda.const.array_like(forcing_vector)

            # Run-indexed slices of shared and output memory
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
        """Returns the number of elements in shared memory required for each run."""
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