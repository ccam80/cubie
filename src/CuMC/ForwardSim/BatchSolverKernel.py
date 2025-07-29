# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""

import os
import numpy as np
from numba import int32, int16
from numba import cuda
from CuMC.ForwardSim.integrators.SingleIntegratorRun import SingleIntegratorRun
from CuMC.ForwardSim.BatchInputArrays import InputArrays
from CuMC.ForwardSim.BatchOutputArrays import OutputArrays
from CuMC.CUDAFactory import CUDAFactory
from numpy.typing import NDArray, ArrayLike
from typing import Optional
from numba.np.numpy_support import as_dtype as to_np_dtype
from CuMC.ForwardSim.BatchSolverConfig import BatchSolverConfig


class BatchSolverKernel(CUDAFactory):
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
                 duration: float = 1.0,
                 warmup: float = 0.0,
                 dt_min: float = 0.01,
                 dt_max: float = 0.1,
                 dt_save: float = 0.1,
                 dt_summarise: float = 1.0,
                 atol: float = 1e-6,
                 rtol: float = 1e-6,
                 saved_states: NDArray[np.int_] = None,
                 saved_observables: NDArray[np.int_] = None,
                 summarised_states: Optional[ArrayLike] = None,
                 summarised_observables: Optional[ArrayLike] = None,
                 output_types: list[str] = None,
                 precision: type = np.float64,
                 profileCUDA: bool = False,
                 ):
        super().__init__()

        self.sizes = system.sizes()

        config = BatchSolverConfig(precision = precision,
                                   algorithm=algorithm,
                                   duration=duration,
                                   warmup=warmup,
                                   profileCUDA=profileCUDA)

        # Setup compile settings for the kernel
        self.setup_compile_settings(config)

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
                summarised_states=summarised_states,
                summarised_observables=summarised_observables,
                output_types=output_types,
                )

        self.output_arrays = OutputArrays.from_solver(self)
        self.input_arrays = InputArrays.from_solver(self)

    @property
    def output_heights(self):
        """Returns the heights of the output arrays."""
        return self.singleIntegrator.output_array_heights

    @property
    def kernel(self):
        return self.device_function

    def build(self):
        return self.build_kernel()

    def run(self,
            duration,
            params,
            inits,
            forcing_vectors,
            output_arrays=None,
            runs_per_block=32,
            stream=0,
            warmup=0.0,
            ):
        """Run the solver kernel."""
        self.input_arrays(inits, params, forcing_vectors)
        self.output_arrays(self)
        numruns = len(inits)
        # input_arrays = self.check_or_allocate_input_arrays(input_arrays, numruns)

        BLOCKSPERGRID = int(max(1, np.ceil(numruns / self.xblocksize)))
        dynamic_sharedmem = self.shared_memory_bytes_per_run * numruns
        threads_per_loop = self.threads_per_loop

        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1" and self._profileCUDA:
            cuda.profile_start()

        self.device_function[BLOCKSPERGRID, (self.xblocksize, runs_per_block), stream, dynamic_sharedmem](
                input_arrays.device_inits,
                input_arrays.device_parameters,
                input_arrays.device_forcing_vectors,
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
        precision = from_dtype(self.precision)
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