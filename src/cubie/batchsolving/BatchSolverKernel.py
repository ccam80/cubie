# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""

import os
from typing import Optional
from warnings import warn

import numpy as np
from numba import cuda
from numba import int32, int16, from_dtype
from numpy.typing import NDArray, ArrayLike

from cubie.memory import default_memmgr
from cubie.CUDAFactory import CUDAFactory
from cubie.batchsolving.arrays.BatchInputArrays import InputArrays
from cubie.batchsolving.arrays.BatchOutputArrays import (OutputArrays,
                                                         ActiveOutputs)
from cubie.batchsolving.BatchSolverConfig import BatchSolverConfig
from cubie.outputhandling.output_sizes import BatchOutputSizes, \
    SingleRunOutputSizes
from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun


class BatchSolverKernel(CUDAFactory):
    """Class which builds and holds the integrating kernel and interfaces
    with lower-level modules: loop
    algorithms, ODE systems, and output functions
    The kernel function accepts single or batched sets of inputs,
    and distributes those amongst the threads on the
    GPU. It runs the loop device function on a given slice of its allocated
    memory, and serves as the distributor
    of work amongst the individual runs of the integrators.
    This class is one level down from the user, managing sanitised inputs
    and handling the machinery of batching and
    running integrators. It does not handle:
     - Integration logic/algorithms - these are handled in
     SingleIntegratorRun, and below
     - Input sanitisation / batch construction - this is handled in the
     solver api.
     - System equations - these are handled in the system model classes.
    """

    def __init__(self, system,
                 algorithm: str = 'euler',
                 duration: float = 1.0,
                 warmup: float = 0.0,
                 dt_min: float = 0.01,
                 dt_max: float = 0.1,
                 dt_save: float = 0.1,
                 dt_summarise: float = 1.0,
                 atol: float = 1e-6,
                 rtol: float = 1e-6,
                 saved_state_indices: NDArray[np.int_] = None,
                 saved_observable_indices: NDArray[np.int_] = None,
                 summarised_state_indices: Optional[ArrayLike] = None,
                 summarised_observable_indices: Optional[ArrayLike] = None,
                 output_types: list[str] = None, precision: type = np.float64,
                 profileCUDA: bool = False,
                 memory_manager=default_memmgr,
                 stream_group='solver',
                 mem_proportion=None,
                 ):
        super().__init__()
        self.chunks = None
        self.chunk_axis = 'run'
        self.num_runs = 1
        self._memory_manager = memory_manager
        self._memory_manager.register(
                self,
                stream_group=stream_group,
                proportion=mem_proportion,
                allocation_ready_hook=self._on_allocation)

        config = BatchSolverConfig(precision=precision, algorithm=algorithm,
                                   duration=duration, warmup=warmup,
                                   profileCUDA=profileCUDA, )

        # Setup compile settings for the kernel
        self.setup_compile_settings(config)

        if output_types is None:
            output_types = ["state"]

        self.single_integrator = SingleIntegratorRun(system,
                algorithm=algorithm,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_save=dt_save,
                dt_summarise=dt_summarise,
                atol=atol,
                rtol=rtol,
                saved_state_indices=saved_state_indices,
                saved_observable_indices=saved_observable_indices,
                summarised_state_indices=summarised_state_indices,
                summarised_observable_indices=summarised_observable_indices,
                output_types=output_types, )

        # input/output arrays supressed while refactoring
        self.input_arrays = InputArrays.from_solver(
                self)
        self.output_arrays = OutputArrays.from_solver(self)

    def _on_allocation(self, response):
        self.chunks = response.chunks

    @property
    def output_heights(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .output_array_heights` from the child SingleIntegratorRun object."""
        return self.single_integrator.output_array_heights

    @property
    def kernel(self):
        return self.device_function

    def build(self):
        return self.build_kernel()

    @property
    def memory_manager(self):
        """Returns the memory manager the solver is registered with."""
        return self._memory_manager

    @property
    def stream_group(self):
        """Returns the stream_group the solver is in."""
        return self.memory_manager.get_stream_group(self)

    @property
    def stream(self):
        """Return the stream assigned to the solver."""
        return self.memory_manager.get_stream(self)


    @property
    def mem_proportion(self):
        """Returns the memory proportion the solver is assigned."""
        return self.memory_manager.proportion(self)

    def run(self,
            inits,
            params,
            forcing_vectors,
            duration,
            blocksize=256,
            stream=None,
            warmup=0.0,
            chunk_axis='run'):
        """Run the solver kernel."""
        if stream is None:
            stream = self.stream

        self.duration = duration
        self.warmup = warmup
        numruns = inits.shape[0]
        self.num_runs = numruns

        self.input_arrays.update(self, inits, params, forcing_vectors)
        self.output_arrays.update(self)

        self.memory_manager.allocate_queue(self, chunk_axis=chunk_axis)
        chunks = self.chunks

        if chunk_axis == 'run':
            chunkruns = int(np.ceil(numruns / chunks))
            chunklength = self.output_length
            chunksize = chunkruns
        elif chunk_axis == 'time':
            chunklength = int(np.ceil(self.output_length / chunks))
            chunkruns = numruns
            chunksize = chunklength
        else:
            chunklength = self.output_length
            chunkruns = numruns
            chunksize = None
            chunks = 1

        #------------ from here on dimensions are "chunked" -----------------
        self.chunk_axis = chunk_axis
        self.chunks = chunks
        numruns = chunkruns
        output_length = chunklength
        warmup_length = self.warmup_length

        dynamic_sharedmem = int(
                self.shared_memory_bytes_per_run * min(numruns,
                                                       blocksize))
        while dynamic_sharedmem > 32768:
            if blocksize < 32:
                warn("Block size has been reduced to less than 32 threads, "
                     "which means your code will suffer a "
                     "performance hit. This is due to your problem requiring "
                     "too much shared memory - try casting "
                     "some parameters to constants, or trying a different "
                     "solving algorithm.")
            blocksize = blocksize / 2
            dynamic_sharedmem = int(
                    self.shared_memory_bytes_per_run * min(numruns, blocksize))

        threads_per_loop = self.single_integrator.threads_per_loop
        runsperblock = int(blocksize / self.single_integrator.threads_per_loop)
        BLOCKSPERGRID = int(max(1, np.ceil(numruns / blocksize)))  #
        # selectively chunk by chunk_size - depends on chunk_axis
        if (os.environ.get(
                "NUMBA_ENABLE_CUDASIM") != "1" and
                self.compile_settings.profileCUDA):
            cuda.profile_start()

        for i in range(chunks):
            indices = slice(i * chunksize, (i + 1) * chunksize)
            self.input_arrays.initialise(indices)
            self.output_arrays.initialise(indices)

            self.device_function[BLOCKSPERGRID,
                                (threads_per_loop, runsperblock),
                                stream,
                                dynamic_sharedmem](
                    self.input_arrays.device_initial_values,
                    self.input_arrays.device_parameters,
                    self.input_arrays.device_forcing_vectors,
                    self.output_arrays.device_state,
                    self.output_arrays.device_observables,
                    self.output_arrays.device_state_summaries,
                    self.output_arrays.device_observable_summaries, output_length,
                    warmup_length, numruns, )
            self.memory_manager.sync_stream(self)

            self.input_arrays.finalise(indices)
            self.output_arrays.finalise(indices)

        if (os.environ.get(
                "NUMBA_ENABLE_CUDASIM") != "1" and
                self.compile_settings.profileCUDA):
            cuda.profile_stop()

    def build_kernel(self):
        """Build the integration kernel."""
        precision = from_dtype(self.precision)
        loopfunction = self.single_integrator.device_function
        shared_elements_per_run = self.shared_memory_elements_per_run

        output_flags = self.active_output_arrays
        save_state = output_flags.state
        save_observables = output_flags.observables
        save_state_summaries = output_flags.state_summaries
        save_observable_summaries = output_flags.observable_summaries

        @cuda.jit((precision[:, :], precision[:, :], precision[:, :],
                   precision[:, :, :], precision[:, :, :], precision[:, :, :],
                   precision[:, :, :], int32, int32, int32), )
        def integration_kernel(inits, params, forcing_vector, state_output,
                               observables_output, state_summaries_output,
                               observables_summaries_output, duration_samples,
                               warmup_samples=0, n_runs=1, ):
            """Master integration kernel - calls integratorLoop and dxdt
            device functions."""
            tx = int16(cuda.threadIdx.x)
            ty = int16(cuda.threadIdx.y)

            block_index = int32(cuda.blockIdx.x)
            runs_per_block = cuda.blockDim.y
            run_index = int32(runs_per_block * block_index + ty)

            if run_index >= n_runs:
                return None

            shared_memory = cuda.shared.array(0, dtype=precision)
            c_forcing_vector = cuda.const.array_like(forcing_vector)

            # Run-indexed slices of shared and output memory
            rx_shared_memory = shared_memory[ty * shared_elements_per_run:(
                                                                                  ty + 1) * shared_elements_per_run]
            rx_inits = inits[run_index, :]
            rx_params = params[run_index, :]
            rx_state = state_output[:, run_index * save_state, :]
            rx_observables = observables_output[:,
                             run_index * save_observables, :]
            rx_state_summaries = state_summaries_output[:,
                                 run_index * save_state_summaries, :]
            rx_observables_summaries = observables_summaries_output[:,
                                       run_index * save_observable_summaries,
                                       :]

            loopfunction(rx_inits, rx_params, c_forcing_vector,
                    rx_shared_memory, rx_state, rx_observables,
                    rx_state_summaries, rx_observables_summaries,
                    duration_samples, warmup_samples, )

            return None

        return integration_kernel

    def update(self, updates_dict=None, silent=False, **kwargs):
        if updates_dict is None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        all_unrecognized = set(updates_dict.keys())
        all_unrecognized -= self.update_compile_settings(updates_dict,
                                                         silent=True)
        all_unrecognized -= self.single_integrator.update(updates_dict,
                                                          silent=True)
        recognised = set(updates_dict.keys()) - all_unrecognized

        if all_unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {all_unrecognized}")
        return recognised

    @property
    def shared_memory_bytes_per_run(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .shared_memory_bytes` from the child SingleIntegratorRun object."""
        return self.single_integrator.shared_memory_bytes

    @property
    def shared_memory_elements_per_run(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .shared_memory_elements` from the child SingleIntegratorRun object."""
        return self.single_integrator.shared_memory_elements

    @property
    def precision(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .precision` from the child SingleIntegratorRun object."""
        return self.single_integrator.precision

    @property
    def threads_per_loop(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .threads_per_loop` from the child SingleIntegratorRun object."""
        return self.single_integrator.threads_per_loop

    @property
    def duration(self):
        """Returns the duration of the simulation."""
        return self.compile_settings.duration

    @duration.setter
    def duration(self, value):
        """Sets the duration of the simulation."""
        self.compile_settings.duration = value

    @property
    def warmup(self):
        """Returns the warmup time of the simulation."""
        return self.compile_settings.warmup

    @warmup.setter
    def warmup(self, value):
        """Sets the warmup time of the simulation."""
        self.compile_settings.warmup = value

    @property
    def output_length(self):
        """Returns the number of output samples per run."""
        return int(
                np.ceil(self.compile_settings.duration /
                        self.single_integrator.dt_save))

    @property
    def summaries_length(self):
        """Returns the number of summary samples per run."""
        return int(
                np.ceil(self.compile_settings.duration /
                        self.single_integrator.dt_summarise))

    @property
    def warmup_length(self):
        return int(
                np.ceil(self.compile_settings.warmup /
                        self.single_integrator.dt_save))

    @property
    def system(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .system` from the SingleIntegratorRun
        instance."""
        return self.single_integrator.system

    @property
    def algorithm(self):
        """Returns the integration algorithm being used."""
        return self.single_integrator.algorithm_key

    @property
    def fixed_step_size(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .step_size` from the child SingleIntegratorRun object."""
        return self.single_integrator.fixed_step_size

    @property
    def dt_min(self):
        """Minimum step size allowed for the solver."""
        return self.single_integrator.config.dt_min

    @property
    def dt_max(self):
        """Maximum step size allowed for the solver."""
        return self.single_integrator.config.dt_max

    @property
    def atol(self):
        """Absolute tolerance for the solver."""
        return self.single_integrator.config.atol

    @property
    def rtol(self):
        """Relative tolerance for the solver."""
        return self.single_integrator.config.rtol

    @property
    def dt_save(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .dt_save` from the child SingleIntegratorRun object."""
        return self.single_integrator.dt_save

    @property
    def dt_summarise(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .dt_summarise` from the child SingleIntegratorRun object."""
        return self.single_integrator.dt_summarise

    @property
    def system_sizes(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .system_sizes` from the child SingleIntegratorRun object."""
        return self.single_integrator.system_sizes

    @property
    def output_array_heights(self):
        """Exposes :attr:`~cubie.batchsolving.integrators
        .SingleIntegratorRun.output_array_heights` from the child
        SingleIntegratorRun object.
        """
        return self.single_integrator.output_array_heights

    @property
    def ouput_array_sizes_2d(self):
        """Returns the 2D output array sizes for a single run."""
        return SingleRunOutputSizes.from_solver(self)

    @property
    def output_array_sizes_3d(self):
        """Returns the 3D output array sizes for a batch of runs."""
        return BatchOutputSizes.from_solver(self)

    @property
    def summaries_buffer_sizes(self):
        """Exposes :attr:`~cubie.batchsolving.integrators
        .SingleIntegratorRun.summaries_buffer_sizes` from the child"""
        return self.single_integrator.summaries_buffer_sizes

    @property
    def summary_legend_per_variable(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .summary_legend_per_variable` from the child SingleIntegratorRun
        object."""
        return self.single_integrator.summary_legend_per_variable

    @property
    def saved_state_indices(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .saved_state_indices` from the child SingleIntegratorRun object."""
        return self.single_integrator.saved_state_indices

    @property
    def saved_observable_indices(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .saved_observable_indices` from the child SingleIntegratorRun
        object."""
        return self.single_integrator.saved_observable_indices

    @property
    def summarised_state_indices(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .summarised_state_indices` from the child SingleIntegratorRun
        object."""
        return self.single_integrator.summarised_state_indices

    @property
    def summarised_observable_indices(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .summarised_observable_indices` from the child SingleIntegratorRun
        object."""
        return self.single_integrator.summarised_observable_indices

    @property
    def active_output_arrays(self) -> "ActiveOutputs":
        """Exposes :attr:`~cubie.batchsolving.BatchOutputArrays.OutputArrays
        ._active_outputs` from the child OutputArrays object."""
        self.output_arrays.allocate()
        return self.output_arrays.active_outputs

    @property
    def device_state_array(self):
        """Exposes :attr:`~cubie.batchsolving.BatchOutputArrays.OutputArrays
        .state` from the child OutputArrays object."""
        return self.output_arrays.device_state

    @property
    def device_observables_array(self):
        """Exposes :attr:`~cubie.batchsolving.BatchOutputArrays.OutputArrays
        .observables` from the child OutputArrays object."""
        return self.output_arrays.device_observables

    @property
    def device_state_summaries_array(self):
        """Exposes :attr:`~cubie.batchsolving.BatchOutputArrays.OutputArrays
        .state_summaries` from the child OutputArrays object."""
        return self.output_arrays.device_state_summaries

    @property
    def device_observable_summaries_array(self):
        """Exposes :attr:`~cubie.batchsolving.BatchOutputArrays.OutputArrays
        .observable_summaries` from the child OutputArrays object."""
        return self.output_arrays.device_observable_summaries

    @property
    def state(self):
        """Returns the state array."""
        return self.output_arrays.state

    @property
    def observables(self):
        """Returns the observables array."""
        return self.output_arrays.observables

    @property
    def state_summaries(self):
        """Returns the state summaries_array array."""
        return self.output_arrays.state_summaries

    @property
    def observable_summaries(self):
        """Returns the observable summaries_array array."""
        return self.output_arrays.observable_summaries

    @property
    def initial_values(self):
        """Returns the initial values array."""
        return self.input_arrays.initial_values

    @property
    def parameters(self):
        """Returns the parameters array."""
        return self.input_arrays.parameters

    @property
    def forcing_vectors(self):
        """Returns the forcing vectors array."""
        return self.input_arrays.forcing_vectors

    @property
    def output_stride_order(self):
        """Returns the axis order of the output arrays."""
        return self.output_arrays.host.stride_order

    @property
    def save_time(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .save_time` from the child SingleIntegratorRun object."""
        return self.single_integrator.save_time

    def enable_profiling(self):
        """
        Enable CUDA profiling for the solver. This will allow you to profile
        the performance of the solver on the
        GPU, but will slow things down.
        """
        # Consider disabling optimisation and enabling debug and line info
        # for profiling
        self.compile_settings.profileCUDA = True

    def disable_profiling(self):
        """
        Disable CUDA profiling for the solver. This will stop profiling the
        performance of the solver on the GPU,
        but will speed things up.
        """
        self.compile_settings.profileCUDA = False

    @property
    def output_types(self):
        """Exposes :attr:`~cubie.batchsolving.integrators.SingleIntegratorRun
        .output_types` from the child SingleIntegratorRun object."""
        return self.single_integrator.output_types
