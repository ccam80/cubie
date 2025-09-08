# -*- coding: utf-8 -*-
"""
Batch Solver Kernel Module.

This module provides the BatchSolverKernel class, which manages GPU-based
batch integration of ODE systems using CUDA. The kernel handles the distribution
of work across GPU threads and manages memory allocation for batched integrations.

Created on Tue May 27 17:45:03 2025

@author: cca79
"""

import os
from typing import Optional
from warnings import warn

import numpy as np
from numba import cuda, float64, float32
from numba import int32, int16, from_dtype
from cubie.cudasim_utils import from_dtype as simsafe_dtype
from numpy.typing import NDArray, ArrayLike

from cubie.memory import default_memmgr
from cubie.CUDAFactory import CUDAFactory
from cubie.batchsolving.arrays.BatchInputArrays import InputArrays
from cubie.batchsolving.arrays.BatchOutputArrays import (
    OutputArrays,
    ActiveOutputs,
)
from cubie.batchsolving.BatchSolverConfig import BatchSolverConfig
from cubie.outputhandling.output_sizes import (
    BatchOutputSizes,
    SingleRunOutputSizes,
)
from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun


class BatchSolverKernel(CUDAFactory):
    """
    CUDA-based batch solver kernel for ODE integration.

    This class builds and holds the integrating kernel and interfaces with
    lower-level modules including loop algorithms, ODE systems, and output
    functions. The kernel function accepts single or batched sets of inputs
    and distributes those amongst the threads on the GPU.

    Parameters
    ----------
    system : object
        The ODE system to be integrated.
    algorithm : str, default='euler'
        Integration algorithm to use.
    duration : float, default=1.0
        Duration of the simulation.
    warmup : float, default=0.0
        Warmup time before the main simulation.
    dt_min : float, default=0.01
        Minimum allowed time step.
    dt_max : float, default=0.1
        Maximum allowed time step.
    dt_save : float, default=0.1
        Time step for saving output.
    dt_summarise : float, default=1.0
        Time step for saving summaries.
    atol : float, default=1e-6
        Absolute tolerance for adaptive stepping.
    rtol : float, default=1e-6
        Relative tolerance for adaptive stepping.
    saved_state_indices : NDArray[np.int_], optional
        Indices of state variables to save.
    saved_observable_indices : NDArray[np.int_], optional
        Indices of observable variables to save.
    summarised_state_indices : ArrayLike, optional
        Indices of state variables to summarise.
    summarised_observable_indices : ArrayLike, optional
        Indices of observable variables to summarise.
    output_types : list[str], optional
        Types of outputs to generate. Default is ["state"].
    precision : type, default=np.float64
        Numerical precision to use.
    profileCUDA : bool, default=False
        Whether to enable CUDA profiling.
    memory_manager : MemoryManager, default=default_memmgr
        Memory manager instance to use.
    stream_group : str, default='solver'
        CUDA stream group identifier.
    mem_proportion : float, optional
        Proportion of GPU memory to allocate.

    Notes
    -----
    This class is one level down from the user, managing sanitised inputs
    and handling the machinery of batching and running integrators. It does
    not handle:

    - Integration logic/algorithms - these are handled in SingleIntegratorRun
      and below
    - Input sanitisation / batch construction - this is handled in the solver api
    - System equations - these are handled in the system model classes

    The class runs the loop device function on a given slice of its allocated
    memory and serves as the distributor of work amongst the individual runs
    of the integrators.
    """

    def __init__(
        self,
        system,
        algorithm: str = "euler",
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
        output_types: list[str] = None,
        precision: type = np.float64,
        profileCUDA: bool = False,
        memory_manager=default_memmgr,
        stream_group="solver",
        mem_proportion=None,
    ):
        super().__init__()
        self.chunks = None
        self.chunk_axis = "run"
        self.num_runs = 1
        self._memory_manager = memory_manager
        self._memory_manager.register(
            self,
            stream_group=stream_group,
            proportion=mem_proportion,
            allocation_ready_hook=self._on_allocation,
        )

        config = BatchSolverConfig(
            precision=precision,
            algorithm=algorithm,
            duration=duration,
            warmup=warmup,
            profileCUDA=profileCUDA,
        )

        # Setup compile settings for the kernel
        self.setup_compile_settings(config)

        if output_types is None:
            output_types = ["state"]

        self.single_integrator = SingleIntegratorRun(
            system,
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
            output_types=output_types,
        )

        # input/output arrays supressed while refactoring
        self.input_arrays = InputArrays.from_solver(self)
        self.output_arrays = OutputArrays.from_solver(self)

    def _on_allocation(self, response):
        """
        Handle memory allocation response.

        Parameters
        ----------
        response : ArrayResponse
            Memory allocation response containing chunk information.
        """
        self.chunks = response.chunks

    @property
    def output_heights(self):
        """
        Get output array heights.

        Returns
        -------
        OutputArrayHeights
            Output array heights from the child SingleIntegratorRun object.

        Notes
        -----
        Exposes the output_array_heights attribute from the child
        SingleIntegratorRun object.
        """
        return self.single_integrator.output_array_heights

    @property
    def kernel(self):
        """
        Get the device function kernel.

        Returns
        -------
        object
            The compiled CUDA device function.
        """
        return self.device_function

    def build(self):
        """
        Build the integration kernel.

        Returns
        -------
        CUDA device function
            The built integration kernel.
        """
        return self.build_kernel()

    @property
    def memory_manager(self):
        """
        Get the memory manager.

        Returns
        -------
        MemoryManager
            The memory manager the solver is registered with.
        """
        return self._memory_manager

    @property
    def stream_group(self):
        """
        Get the stream group.

        Returns
        -------
        str
            The stream group the solver is in.
        """
        return self.memory_manager.get_stream_group(self)

    @property
    def stream(self):
        """
        Get the assigned CUDA stream.

        Returns
        -------
        Stream
            The CUDA stream assigned to the solver.
        """
        return self.memory_manager.get_stream(self)

    @property
    def mem_proportion(self):
        """
        Get the memory proportion.

        Returns
        -------
        float
            The memory proportion the solver is assigned.
        """
        return self.memory_manager.proportion(self)

    def run(
        self,
        inits,
        params,
        forcing_vectors,
        duration,
        blocksize=256,
        stream=None,
        warmup=0.0,
        chunk_axis="run",
    ):
        """
        Execute the solver kernel for batch integration.

        Parameters
        ----------
        inits : array_like
            Initial conditions for each run. Shape should be (n_runs, n_states).
        params : array_like
            Parameters for each run. Shape should be (n_runs, n_params).
        forcing_vectors : array_like
            Forcing vectors for each run.
        duration : float
            Duration of the simulation.
        blocksize : int, default=256
            CUDA block size for kernel execution.
        stream : object, optional
            CUDA stream to use. If None, uses the solver's assigned stream.
        warmup : float, default=0.0
            Warmup time before the main simulation.
        chunk_axis : str, default='run'
            Axis along which to chunk the computation ('run' or 'time').

        Notes
        -----
        This method performs the main batch integration by:
        1. Setting up input and output arrays
        2. Allocating GPU memory in chunks
        3. Executing the CUDA kernel for each chunk
        4. Handling memory management and synchronization

        The method automatically adjusts block size if shared memory requirements
        exceed GPU limits, warning the user about potential performance impacts.
        """
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

        if chunk_axis == "run":
            chunkruns = int(np.ceil(numruns / chunks))
            chunklength = self.output_length
            chunksize = chunkruns
        elif chunk_axis == "time":
            chunklength = int(np.ceil(self.output_length / chunks))
            chunkruns = numruns
            chunksize = chunklength
        else:
            chunklength = self.output_length
            chunkruns = numruns
            chunksize = None
            chunks = 1

        # ------------ from here on dimensions are "chunked" -----------------
        self.chunk_axis = chunk_axis
        self.chunks = chunks
        numruns = chunkruns
        output_length = chunklength
        warmup_length = self.warmup_length
        pad_perrun = 4 if self.shared_memory_needs_padding else 0
        padded_bytes_perrun = self.shared_memory_bytes_per_run + pad_perrun
        dynamic_sharedmem = int(
            padded_bytes_perrun * min(numruns, blocksize)
        )
        while dynamic_sharedmem > 32768:
            if blocksize < 32:
                warn(
                    "Block size has been reduced to less than 32 threads, "
                    "which means your code will suffer a "
                    "performance hit. This is due to your problem requiring "
                    "too much shared memory - try casting "
                    "some parameters to constants, or trying a different "
                    "solving algorithm."
                )
            blocksize = blocksize / 2
            dynamic_sharedmem = int(
                padded_bytes_perrun * min(numruns, blocksize)
            )

        threads_per_loop = self.single_integrator.threads_per_loop
        runsperblock = int(blocksize / self.single_integrator.threads_per_loop)
        BLOCKSPERGRID = int(max(1, np.ceil(numruns / blocksize)))  #
        # selectively chunk by chunk_size - depends on chunk_axis
        if (
            os.environ.get("NUMBA_ENABLE_CUDASIM") != "1"
            and self.compile_settings.profileCUDA
        ): # pragma: no cover
            cuda.profile_start()

        for i in range(chunks):
            indices = slice(i * chunksize, (i + 1) * chunksize)
            self.input_arrays.initialise(indices)
            self.output_arrays.initialise(indices)

            self.device_function[
                BLOCKSPERGRID,
                (threads_per_loop, runsperblock),
                stream,
                dynamic_sharedmem,
            ](
                self.input_arrays.device_initial_values,
                self.input_arrays.device_parameters,
                self.input_arrays.device_forcing_vectors,
                self.output_arrays.device_state,
                self.output_arrays.device_observables,
                self.output_arrays.device_state_summaries,
                self.output_arrays.device_observable_summaries,
                output_length,
                warmup_length,
                numruns,
            )
            self.memory_manager.sync_stream(self)

            self.input_arrays.finalise(indices)
            self.output_arrays.finalise(indices)

        if (
            os.environ.get("NUMBA_ENABLE_CUDASIM") != "1"
            and self.compile_settings.profileCUDA
        ): # pragma: no cover
            cuda.profile_stop()

    @property
    def shared_memory_needs_padding(self):
        """True if we need to pad shared memory to avoid bank conflicts"""
        if self.precision == np.float64:
            return True
        elif self.shared_memory_elements_per_run % 2 == 0:
            return True
        else:
            return False

    def build_kernel(self):
        """
        Build and compile the CUDA integration kernel.

        Returns
        -------
        function
            Compiled CUDA kernel function for integration.

        Notes
        -----
        This method creates a CUDA kernel that:
        1. Distributes work across GPU threads
        2. Manages shared memory allocation
        3. Calls the underlying integration loop function
        4. Handles output array indexing and slicing

        The kernel uses a 2D thread block structure where:
        - x-dimension handles intra-run parallelism
        - y-dimension handles different runs
        """
        # Internal casting should use this simulator-safe dtype
        simsafe_precision = simsafe_dtype(self.precision)
        precision = from_dtype(self.precision)

        loopfunction = self.single_integrator.device_function

        output_flags = self.active_output_arrays
        save_state = output_flags.state
        save_observables = output_flags.observables
        save_state_summaries = output_flags.state_summaries
        save_observable_summaries = output_flags.observable_summaries
        needs_padding = self.shared_memory_needs_padding

        # Shared memory strides are set to minimise bank conflicts - each run
        # must span an odd number of 32b words.
        shared_elements_per_run = self.shared_memory_elements_per_run
        f32_per_element = 2 if (precision is float64) else 1
        f32_pad_perrun = 1 if needs_padding else 0
        run_stride_f32 = int((f32_per_element * shared_elements_per_run +
                          f32_pad_perrun))

        # no cover: start
        @cuda.jit(
            (
                precision[:, :],
                precision[:, :],
                precision[:, :],
                precision[:, :, :],
                precision[:, :, :],
                precision[:, :, :],
                precision[:, :, :],
                int32,
                int32,
                int32,
            ),
        )
        def integration_kernel(
            inits,
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
            tx = int16(cuda.threadIdx.x)
            ty = int16(cuda.threadIdx.y)

            block_index = int32(cuda.blockIdx.x)
            runs_per_block = cuda.blockDim.y
            run_index = int32(runs_per_block * block_index + ty)

            if run_index >= n_runs:
                return None

            #Declare shared memory in 32b units to allow for skewing/padding
            shared_memory = cuda.shared.array(0, dtype=float32)
            c_forcing_vector = cuda.const.array_like(forcing_vector)

            # Run-indexed slices of shared and output memory

            run_idx_low = ty * run_stride_f32
            run_idx_high = (run_idx_low + f32_per_element *
                            shared_elements_per_run)

            rx_shared_memory = shared_memory[run_idx_low:run_idx_high].view(
                    simsafe_precision)

            rx_inits = inits[run_index, :]
            rx_params = params[run_index, :]
            rx_state = state_output[:, run_index * save_state, :]
            rx_observables = observables_output[
                :, run_index * save_observables, :
            ]
            rx_state_summaries = state_summaries_output[
                :, run_index * save_state_summaries, :
            ]
            rx_observables_summaries = observables_summaries_output[
                :, run_index * save_observable_summaries, :
            ]

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

        # no cover: end
        return integration_kernel

    def update(self, updates_dict=None, silent=False, **kwargs):
        """
        Update solver configuration parameters.

        Parameters
        ----------
        updates_dict : dict, optional
            Dictionary of parameter updates.
        silent : bool, default=False
            If True, suppresses error messages for unrecognized parameters.
        **kwargs
            Additional parameter updates passed as keyword arguments.

        Returns
        -------
        set
            Set of recognized parameter names that were updated.

        Raises
        ------
        KeyError
            If unrecognized parameters are provided and silent=False.

        Notes
        -----
        This method attempts to update parameters in both the compile settings
        and the single integrator instance. Unrecognized parameters are
        collected and reported as an error unless silent mode is enabled.
        """
        if updates_dict is None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        all_unrecognized = set(updates_dict.keys())
        all_unrecognized -= self.update_compile_settings(
            updates_dict, silent=True
        )
        all_unrecognized -= self.single_integrator.update(
            updates_dict, silent=True
        )
        recognised = set(updates_dict.keys()) - all_unrecognized

        if all_unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {all_unrecognized}")
        return recognised

    @property
    def shared_memory_bytes_per_run(self):
        """
        Get shared memory bytes per run.

        Returns
        -------
        int
            Shared memory bytes required per run.

        Notes
        -----
        Exposes the shared_memory_bytes attribute from the child
        SingleIntegratorRun object.
        """
        return self.single_integrator.shared_memory_bytes

    @property
    def shared_memory_elements_per_run(self):
        """
        Get shared memory elements per run.

        Returns
        -------
        int
            Number of shared memory elements required per run.

        Notes
        -----
        Exposes the shared_memory_elements attribute from the child
        SingleIntegratorRun object.
        """
        return self.single_integrator.shared_memory_elements

    @property
    def precision(self):
        """
        Get numerical precision type.

        Returns
        -------
        type
            Numerical precision type (e.g., np.float64).

        Notes
        -----
        Exposes the precision attribute from the child SingleIntegratorRun object.
        """
        return self.single_integrator.precision

    @property
    def threads_per_loop(self):
        """
        Get threads per loop.

        Returns
        -------
        int
            Number of threads per integration loop.

        Notes
        -----
        Exposes the threads_per_loop attribute from the child
        SingleIntegratorRun object.
        """
        return self.single_integrator.threads_per_loop

    @property
    def duration(self):
        """
        Get simulation duration.

        Returns
        -------
        float
            Duration of the simulation.
        """
        return self.compile_settings.duration

    @duration.setter
    def duration(self, value):
        """
        Set simulation duration.

        Parameters
        ----------
        value : float
            Duration of the simulation.
        """
        self.compile_settings.duration = value

    @property
    def warmup(self):
        """
        Get warmup time.

        Returns
        -------
        float
            Warmup time of the simulation.
        """
        return self.compile_settings.warmup

    @warmup.setter
    def warmup(self, value):
        """
        Set warmup time.

        Parameters
        ----------
        value : float
            Warmup time of the simulation.
        """
        self.compile_settings.warmup = value

    @property
    def output_length(self):
        """
        Get number of output samples per run.

        Returns
        -------
        int
            Number of output samples per run.
        """
        return int(
            np.ceil(
                self.compile_settings.duration / self.single_integrator.dt_save
            )
        )

    @property
    def summaries_length(self):
        """
        Get number of summary samples per run.

        Returns
        -------
        int
            Number of summary samples per run.
        """
        return int(
            np.ceil(
                self.compile_settings.duration
                / self.single_integrator.dt_summarise
            )
        )

    @property
    def warmup_length(self):
        """
        Get number of warmup samples.

        Returns
        -------
        int
            Number of warmup samples.
        """
        return int(
            np.ceil(
                self.compile_settings.warmup / self.single_integrator.dt_save
            )
        )

    @property
    def system(self):
        """
        Get the ODE system.

        Returns
        -------
        object
            The ODE system being integrated.

        Notes
        -----
        Exposes the system attribute from the SingleIntegratorRun instance.
        """
        return self.single_integrator.system

    @property
    def algorithm(self):
        """
        Get the integration algorithm.

        Returns
        -------
        str
            The integration algorithm being used.
        """
        return self.single_integrator.algorithm_key

    @property
    def fixed_step_size(self):
        """
        Get the fixed step size.

        Returns
        -------
        float
            Fixed step size for the solver.

        Notes
        -----
        Exposes the step_size attribute from the child SingleIntegratorRun object.
        """
        return self.single_integrator.fixed_step_size

    @property
    def dt_min(self):
        """
        Get minimum step size.

        Returns
        -------
        float
            Minimum step size allowed for the solver.
        """
        return self.single_integrator.config.dt_min

    @property
    def dt_max(self):
        """
        Get maximum step size.

        Returns
        -------
        float
            Maximum step size allowed for the solver.
        """
        return self.single_integrator.config.dt_max

    @property
    def atol(self):
        """
        Get absolute tolerance.

        Returns
        -------
        float
            Absolute tolerance for the solver.
        """
        return self.single_integrator.config.atol

    @property
    def rtol(self):
        """
        Get relative tolerance.

        Returns
        -------
        float
            Relative tolerance for the solver.
        """
        return self.single_integrator.config.rtol

    @property
    def dt_save(self):
        """
        Get save time step.

        Returns
        -------
        float
            Time step for saving output.

        Notes
        -----
        Exposes the dt_save attribute from the child SingleIntegratorRun object.
        """
        return self.single_integrator.dt_save

    @property
    def dt_summarise(self):
        """
        Get summary time step.

        Returns
        -------
        float
            Time step for saving summaries.

        Notes
        -----
        Exposes the dt_summarise attribute from the child SingleIntegratorRun object.
        """
        return self.single_integrator.dt_summarise

    @property
    def system_sizes(self):
        """
        Get system sizes.

        Returns
        -------
        object
            System sizes information.

        Notes
        -----
        Exposes the system_sizes attribute from the child SingleIntegratorRun object.
        """
        return self.single_integrator.system_sizes

    @property
    def output_array_heights(self):
        """
        Get output array heights.

        Returns
        -------
        object
            Output array heights information.

        Notes
        -----
        Exposes the output_array_heights attribute from the child
        SingleIntegratorRun object.
        """
        return self.single_integrator.output_array_heights

    @property
    def ouput_array_sizes_2d(self):
        """
        Get 2D output array sizes.

        Returns
        -------
        object
            The 2D output array sizes for a single run.
        """
        return SingleRunOutputSizes.from_solver(self)

    @property
    def output_array_sizes_3d(self):
        """
        Get 3D output array sizes.

        Returns
        -------
        object
            The 3D output array sizes for a batch of runs.
        """
        return BatchOutputSizes.from_solver(self)

    @property
    def summaries_buffer_sizes(self):
        """
        Get summaries buffer sizes.

        Returns
        -------
        object
            Summaries buffer sizes information.

        Notes
        -----
        Exposes the summaries_buffer_sizes attribute from the child
        SingleIntegratorRun object.
        """
        return self.single_integrator.summaries_buffer_sizes

    @property
    def summary_legend_per_variable(self):
        """
        Get summary legend per variable.

        Returns
        -------
        object
            Summary legend per variable information.

        Notes
        -----
        Exposes the summary_legend_per_variable attribute from the child
        SingleIntegratorRun object.
        """
        return self.single_integrator.summary_legend_per_variable

    @property
    def saved_state_indices(self):
        """
        Get saved state indices.

        Returns
        -------
        NDArray[np.int_]
            Indices of state variables to save.

        Notes
        -----
        Exposes the saved_state_indices attribute from the child
        SingleIntegratorRun object.
        """
        return self.single_integrator.saved_state_indices

    @property
    def saved_observable_indices(self):
        """
        Get saved observable indices.

        Returns
        -------
        NDArray[np.int_]
            Indices of observable variables to save.

        Notes
        -----
        Exposes the saved_observable_indices attribute from the child
        SingleIntegratorRun object.
        """
        return self.single_integrator.saved_observable_indices

    @property
    def summarised_state_indices(self):
        """
        Get summarised state indices.

        Returns
        -------
        ArrayLike
            Indices of state variables to summarise.

        Notes
        -----
        Exposes the summarised_state_indices attribute from the child
        SingleIntegratorRun object.
        """
        return self.single_integrator.summarised_state_indices

    @property
    def summarised_observable_indices(self):
        """
        Get summarised observable indices.

        Returns
        -------
        ArrayLike
            Indices of observable variables to summarise.

        Notes
        -----
        Exposes the summarised_observable_indices attribute from the child
        SingleIntegratorRun object.
        """
        return self.single_integrator.summarised_observable_indices

    @property
    def active_output_arrays(self) -> "ActiveOutputs":
        """
        Get active output arrays.

        Returns
        -------
        ActiveOutputs
            Active output arrays configuration.

        Notes
        -----
        Exposes the _active_outputs attribute from the child OutputArrays object.
        """
        self.output_arrays.allocate()
        return self.output_arrays.active_outputs

    @property
    def device_state_array(self):
        """
        Get device state array.

        Returns
        -------
        object
            Device state array.

        Notes
        -----
        Exposes the state attribute from the child OutputArrays object.
        """
        return self.output_arrays.device_state

    @property
    def device_observables_array(self):
        """
        Get device observables array.

        Returns
        -------
        object
            Device observables array.

        Notes
        -----
        Exposes the observables attribute from the child OutputArrays object.
        """
        return self.output_arrays.device_observables

    @property
    def device_state_summaries_array(self):
        """
        Get device state summaries array.

        Returns
        -------
        object
            Device state summaries array.

        Notes
        -----
        Exposes the state_summaries attribute from the child OutputArrays object.
        """
        return self.output_arrays.device_state_summaries

    @property
    def device_observable_summaries_array(self):
        """
        Get device observable summaries array.

        Returns
        -------
        object
            Device observable summaries array.

        Notes
        -----
        Exposes the observable_summaries attribute from the child OutputArrays object.
        """
        return self.output_arrays.device_observable_summaries

    @property
    def state(self):
        """
        Get state array.

        Returns
        -------
        array_like
            The state array.
        """
        return self.output_arrays.state

    @property
    def observables(self):
        """
        Get observables array.

        Returns
        -------
        array_like
            The observables array.
        """
        return self.output_arrays.observables

    @property
    def state_summaries(self):
        """
        Get state summaries array.

        Returns
        -------
        array_like
            The state summaries array.
        """
        return self.output_arrays.state_summaries

    @property
    def observable_summaries(self):
        """
        Get observable summaries array.

        Returns
        -------
        array_like
            The observable summaries array.
        """
        return self.output_arrays.observable_summaries

    @property
    def initial_values(self):
        """
        Get initial values array.

        Returns
        -------
        array_like
            The initial values array.
        """
        return self.input_arrays.initial_values

    @property
    def parameters(self):
        """
        Get parameters array.

        Returns
        -------
        array_like
            The parameters array.
        """
        return self.input_arrays.parameters

    @property
    def forcing_vectors(self):
        """
        Get forcing vectors array.

        Returns
        -------
        array_like
            The forcing vectors array.
        """
        return self.input_arrays.forcing_vectors

    @property
    def output_stride_order(self):
        """
        Get output stride order.

        Returns
        -------
        str
            The axis order of the output arrays.
        """
        return self.output_arrays.host.stride_order

    @property
    def save_time(self):
        """
        Get save time array.

        Returns
        -------
        array_like
            Time points for saved output.

        Notes
        -----
        Exposes the save_time attribute from the child SingleIntegratorRun object.
        """
        return self.single_integrator.save_time

    def enable_profiling(self):
        """
        Enable CUDA profiling for the solver.

        Notes
        -----
        This will allow you to profile the performance of the solver on the
        GPU, but will slow things down. Consider disabling optimisation and
        enabling debug and line info for profiling.
        """
        # Consider disabling optimisation and enabling debug and line info
        # for profiling
        self.compile_settings.profileCUDA = True

    def disable_profiling(self):
        """
        Disable CUDA profiling for the solver.

        Notes
        -----
        This will stop profiling the performance of the solver on the GPU,
        but will speed things up.
        """
        self.compile_settings.profileCUDA = False

    @property
    def output_types(self):
        """
        Get output types.

        Returns
        -------
        list[str]
            Types of outputs generated.

        Notes
        -----
        Exposes the output_types attribute from the child SingleIntegratorRun object.
        """
        return self.single_integrator.output_types
