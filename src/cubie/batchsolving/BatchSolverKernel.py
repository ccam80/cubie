# -*- coding: utf-8 -*-
"""CUDA batch solver kernel utilities."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple
from warnings import warn

import numpy as np
from numba import cuda, float64, float32
from numba import int32, int16

import attrs

from cubie.cuda_simsafe import is_cudasim_enabled
from numpy.typing import NDArray

from cubie.memory import default_memmgr
from cubie.CUDAFactory import CUDAFactory
from cubie.batchsolving.arrays.BatchInputArrays import InputArrays
from cubie.batchsolving.arrays.BatchOutputArrays import (
    OutputArrays,
    ActiveOutputs,
)
from cubie.batchsolving.BatchSolverConfig import BatchSolverConfig
from cubie.odesystems.baseODE import BaseODE
from cubie.outputhandling.output_sizes import (
    BatchOutputSizes,
    SingleRunOutputSizes,
)
from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from cubie._utils import PrecisionDType

if TYPE_CHECKING:
    from cubie.memory import MemoryManager

DEFAULT_MEMORY_SETTINGS = {
    "memory_manager": default_memmgr,
    "stream_group": "solver",
    "mem_proportion": None,
}


@attrs.define(frozen=True)
class ChunkParams:
    """Chunked execution parameters calculated for a batch run.

    Attributes
    ----------
    duration
        Duration assigned to each chunk.
    warmup
        Warmup duration applied to the current chunk.
    t0
        Start time of the chunk.
    size
        Number of indices processed per chunk.
    runs
        Number of runs scheduled within a chunk.
    """

    duration: float
    warmup: float
    t0: float
    size: int
    runs: int


class BatchSolverKernel(CUDAFactory):
    """Factory for CUDA kernel which coordinates a batch integration.

    Parameters
    ----------
    system
        ODE system describing the problem to integrate.
    loop_settings
        Mapping of loop configuration forwarded to
        :class:`cubie.integrators.SingleIntegratorRun`. Recognised keys include
        ``"dt_save"`` and ``"dt_summarise"``.
    driver_function
        Optional evaluation function for an interpolated forcing term.
    profileCUDA
        Flag enabling CUDA profiling hooks.
    step_control_settings
        Mapping of overrides forwarded to
        :class:`cubie.integrators.SingleIntegratorRun` for controller
        configuration.
    algorithm_settings
        Mapping of overrides forwarded to
        :class:`cubie.integrators.SingleIntegratorRun` for algorithm
        configuration.
    output_settings
        Mapping of output configuration forwarded to the integrator. See
        :class:`cubie.outputhandling.OutputFunctions` for recognised keys.
    memory_settings
        Mapping of memory configuration forwarded to the memory manager,
        typically via :mod:`cubie.memory`.

    Notes
    -----
    The kernel delegates integration logic to :class:`SingleIntegratorRun`
    instances and expects upstream APIs to perform batch construction. It
    executes the compiled loop function against kernel-managed memory slices
    and distributes work across GPU threads for each input batch.
    """

    def __init__(
        self,
        system: "BaseODE",
        loop_settings: Optional[Dict[str, Any]] = None,
        driver_function: Optional[Callable] = None,
        driver_del_t: Optional[Callable] = None,
        profileCUDA: bool = False,
        step_control_settings: Optional[Dict[str, Any]] = None,
        algorithm_settings: Optional[Dict[str, Any]] = None,
        output_settings: Optional[Dict[str, Any]] = None,
        memory_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if memory_settings is None:
            memory_settings = {}
        if output_settings is None:
            output_settings = {}
        if loop_settings is None:
            loop_settings = {}

        # Store non compile-critical run parameters locally
        self._profileCUDA = profileCUDA

        precision = system.precision
        self._duration = precision(0.0)
        self._warmup = precision(0.0)
        self._t0 = precision(0.0)
        self.chunks = None
        self.chunk_axis = "run"
        self.num_runs = 1

        self._memory_manager = self._setup_memory_manager(memory_settings)

        # Build the single integrator to derive compile-critical metadata
        self.single_integrator = SingleIntegratorRun(
            system,
            loop_settings=loop_settings,
            driver_function=driver_function,
            driver_del_t=driver_del_t,
            step_control_settings=step_control_settings,
            algorithm_settings=algorithm_settings,
            output_settings=output_settings,
        )

        initial_config = BatchSolverConfig(
            precision=precision,
            loop_fn=None,
            local_memory_elements=(
                self.single_integrator.local_memory_elements
            ),
            shared_memory_elements=(
                self.single_integrator.shared_memory_elements
            ),
            ActiveOutputs=ActiveOutputs(),
            # placeholder, updated after arrays allocate
        )
        self.setup_compile_settings(initial_config)

        self.input_arrays = InputArrays.from_solver(self)
        self.output_arrays = OutputArrays.from_solver(self)

        # Allocate/update to set active outputs then refresh compile settings
        self.output_arrays.update(self)
        self.update_compile_settings(
            {
                "ActiveOutputs": self.output_arrays.active_outputs,
                "local_memory_elements": (
                    self.single_integrator.local_memory_elements
                ),
                "shared_memory_elements": (
                    self.single_integrator.shared_memory_elements
                ),
                "precision": self.single_integrator.precision,
            }
        )

    def _setup_memory_manager(
        self, settings: Dict[str, Any]
    ) -> "MemoryManager":
        """Register the kernel with a memory manager instance.

        Parameters
        ----------
        settings
            Mapping of memory configuration options recognised by the memory
            manager.

        Returns
        -------
        MemoryManager
            Memory manager configured for solver allocations.
        """

        merged_settings = DEFAULT_MEMORY_SETTINGS.copy()
        merged_settings.update(settings)
        memory_manager = merged_settings["memory_manager"]
        stream_group = merged_settings["stream_group"]
        mem_proportion = merged_settings["mem_proportion"]
        memory_manager.register(
            self,
            stream_group=stream_group,
            proportion=mem_proportion,
            allocation_ready_hook=self._on_allocation,
        )
        return memory_manager

    def run(
        self,
        inits: NDArray[np.floating],
        params: NDArray[np.floating],
        driver_coefficients: Optional[NDArray[np.floating]],
        duration: float,
        blocksize: int = 256,
        stream: Optional[Any] = None,
        warmup: float = 0.0,
        t0: float = 0.0,
        chunk_axis: str = "run",
    ) -> None:
        """Execute the solver kernel for batch integration.

        Parameters
        ----------
        inits
            Initial conditions with shape ``(n_runs, n_states)``.
        params
            Parameter table with shape ``(n_runs, n_params)``.
        driver_coefficients
            Optional Horner-ordered driver interpolation coefficients with
            shape ``(num_segments, num_drivers, order + 1)``.
        duration
            Duration of the simulation window.
        blocksize
            CUDA block size for kernel execution.
        stream
            CUDA stream assigned to the batch launch.
        warmup
            Warmup time before the main simulation.
        t0
            Initial integration time.
        chunk_axis
            Axis used to partition the workload, either ``"run"`` or
            ``"time"``.

        Returns
        -------
        None
            This method performs the integration for its side effects.

        Notes
        -----
        The kernel prepares array views, queues allocations, and executes the
        device loop on each chunked workload. Shared-memory demand may reduce
        the block size automatically, emitting a warning when the limit drops
        below a warp.
        """
        if stream is None:
            stream = self.stream

        precision = self.precision
        duration = precision(duration)
        warmup = precision(warmup)
        t0 = precision(t0)

        self._duration = duration
        self._warmup = warmup
        self._t0 = t0

        numruns = inits.shape[0]
        self.num_runs = numruns  # Don't delete - generates batchoutputsizes

        # Queue allocations
        self.input_arrays.update(self, inits, params, driver_coefficients)
        self.output_arrays.update(self)

        # Refresh compile-critical settings (may trigger rebuild)
        self.update_compile_settings(
            {
                "loop_fn": self.single_integrator.compiled_loop_function,
                "precision": self.single_integrator.precision,
                "local_memory_elements": (
                    self.single_integrator.local_memory_elements
                ),
                "shared_memory_elements": (
                    self.single_integrator.shared_memory_elements
                ),
                "ActiveOutputs": self.output_arrays.active_outputs,
            }
        )

        # Process allocations into chunks
        self.memory_manager.allocate_queue(self, chunk_axis=chunk_axis)

        # ------------ from here on dimensions are "chunked" -----------------
        chunk_params = self.chunk_run(
            chunk_axis,
            duration,
            warmup,
            t0,
            numruns,
            self.chunks,
        )
        chunk_warmup = chunk_params.warmup
        chunk_t0 = chunk_params.t0

        pad = 4 if self.shared_memory_needs_padding else 0
        padded_bytes = self.shared_memory_bytes + pad
        dynamic_sharedmem = int(
            padded_bytes * min(chunk_params.runs, blocksize)
        )

        blocksize, dynamic_sharedmem = self.limit_blocksize(
            blocksize,
            dynamic_sharedmem,
            padded_bytes,
            numruns,
        )
        threads_per_loop = self.single_integrator.threads_per_loop
        runsperblock = int(blocksize / self.single_integrator.threads_per_loop)
        BLOCKSPERGRID = int(max(1, np.ceil(numruns / blocksize)))

        if self.profileCUDA:  # pragma: no cover
            cuda.profile_start()

        for i in range(self.chunks):
            indices = slice(i * chunk_params.size, (i + 1) * chunk_params.size)
            self.input_arrays.initialise(indices)
            self.output_arrays.initialise(indices)

            # Don't use warmup in runs starting after t=t0
            if (chunk_axis == "time") and (i != 0):
                chunk_warmup = precision(0.0)
                chunk_t0 = t0 + precision(i) * chunk_params.duration
            
            self.device_function[
                BLOCKSPERGRID,
                (threads_per_loop, runsperblock),
                stream,
                dynamic_sharedmem,
            ](
                self.input_arrays.device_initial_values,
                self.input_arrays.device_parameters,
                self.input_arrays.device_driver_coefficients,
                self.output_arrays.device_state,
                self.output_arrays.device_observables,
                self.output_arrays.device_state_summaries,
                self.output_arrays.device_observable_summaries,
                self.output_arrays.device_status_codes,
                chunk_params.duration,
                chunk_warmup,
                chunk_t0,
                numruns,
            )
            self.memory_manager.sync_stream(self)

            self.input_arrays.finalise(indices)
            self.output_arrays.finalise(indices)

        if self.profileCUDA:  # pragma: no cover
            cuda.profile_stop()

    def limit_blocksize(
        self,
        blocksize: int,
        dynamic_sharedmem: int,
        bytes_per_run: int,
        numruns: int,
    ) -> tuple[int, int]:
        """Reduce block size until dynamic shared memory fits within limits.

        Parameters
        ----------
        blocksize
            Requested CUDA block size.
        dynamic_sharedmem
            Shared-memory footprint per block at the current block size.
        bytes_per_run
            Shared-memory requirement per run.
        numruns
            Total number of runs queued for the launch.

        Returns
        -------
        tuple[int, int]
            Adjusted block size and shared-memory footprint per block.

        Notes
        -----
        The shared-memory ceiling uses 32 kiB so three blocks can reside per SM
        on CC7* hardware. Larger requests reduce per-thread L1 availability.
        """
        while dynamic_sharedmem > 32768:
            if blocksize < 32:
                warn(
                    "Block size has been reduced to less than 32 threads, "
                    "which means your code will suffer a "
                    "performance hit. This is due to your problem requiring "
                    "too much shared memory - try changing "
                    "some parameters to constants, or trying a different "
                    "solving algorithm."
                )
            blocksize = int(blocksize // 2)
            dynamic_sharedmem = int(
                bytes_per_run * min(numruns, blocksize)
            )
        return blocksize, dynamic_sharedmem

    def chunk_run(
        self,
        chunk_axis: str,
        duration: float,
        warmup: float,
        t0: float,
        numruns: int,
        chunks: int,
    ) -> ChunkParams:
        """Split the workload into chunks along the selected axis.

        Parameters
        ----------
        chunk_axis
            Axis along which to partition the workload, either ``"run"`` or
            ``"time"``.
        duration
            Duration of the simulation window.
        warmup
            Warmup time before the main simulation.
        t0
            Initial integration time.
        numruns
            Total number of runs in the batch.
        chunks
            Number of partitions requested by the memory manager.

        Returns
        -------
        ChunkParams
            Chunked execution parameters describing the per-chunk workload.
        """
        chunkruns = numruns
        chunk_warmup = warmup
        chunk_duration = duration
        chunk_t0 = t0
        if chunk_axis == "run":
            chunkruns = int(np.ceil(numruns / chunks))
            chunksize = chunkruns
        elif chunk_axis == "time":
            chunk_duration = duration / chunks
            chunksize = int(np.ceil(self.output_length / chunks))
        else:
            raise ValueError("chunk_axis must be either 'run' or 'time'")

        return ChunkParams(
            duration=chunk_duration,
            warmup=chunk_warmup,
            t0=chunk_t0,
            size=chunksize,
            runs=chunkruns,
        )

    def build_kernel(self) -> None:
        """Build and compile the CUDA integration kernel."""
        config = self.compile_settings
        simsafe_precision = config.simsafe_precision
        precision = config.numba_precision

        loopfunction = self.single_integrator.device_function

        output_flags = config.ActiveOutputs
        save_state = output_flags.state
        save_observables = output_flags.observables
        save_state_summaries = output_flags.state_summaries
        save_observable_summaries = output_flags.observable_summaries
        needs_padding = self.shared_memory_needs_padding

        local_elements_per_run = config.local_memory_elements
        shared_elems_per_run = config.shared_memory_elements
        f32_per_element = 2 if (precision is float64) else 1
        f32_pad_perrun = 1 if needs_padding else 0
        run_stride_f32 = int(
            (f32_per_element * shared_elems_per_run + f32_pad_perrun)
        )
        # no cover: start
        @cuda.jit(
            (
                precision[:, :],
                precision[:, :],
                precision[:, :, :],
                precision[:, :, :],
                precision[:, :, :],
                precision[:, :, :],
                precision[:, :, :],
                int32[:],
                precision,
                precision,
                precision,
                int32,
            ),
        )
        def integration_kernel(
            inits,
            params,
            d_coefficients,
            state_output,
            observables_output,
            state_summaries_output,
            observables_summaries_output,
            status_codes_output,
            duration,
            warmup=precision(0.0),
            t0=precision(0.0),
            n_runs=1,
        ):
            """Execute the compiled single-run loop for each batch chunk.

            Parameters
            ----------
            inits
                Device array containing initial values for each run.
            params
                Device array containing parameter values for each run.
            d_coefficients
                Device array of driver interpolation coefficients.
            state_output
                Device array where state trajectories are written.
            observables_output
                Device array where observable trajectories are written.
            state_summaries_output
                Device array containing state summary reductions.
            observables_summaries_output
                Device array containing observable summary reductions.
            status_codes_output
                Device array storing per-run solver status codes.
            duration
                Duration assigned to the current chunk integration.
            warmup
                Warmup duration applied before the chunk starts.
            t0
                Start time of the chunk integration window.
            n_runs
                Number of runs scheduled for the kernel launch.

            Returns
            -------
            None
                The device kernel performs integration for its side effects.
            """
            tx = int16(cuda.threadIdx.x)
            ty = int16(cuda.threadIdx.y)
            block_index = int32(cuda.blockIdx.x)
            runs_per_block = cuda.blockDim.y
            run_index = int32(runs_per_block * block_index + ty)
            if run_index >= n_runs:
                return None
            shared_memory = cuda.shared.array(0, dtype=float32)

            # Declare shared memory in 32b units to allow for skewing/padding
            local_scratch = cuda.local.array(
                local_elements_per_run, dtype=float32
            )
            c_coefficients = cuda.const.array_like(d_coefficients)
            run_idx_low = ty * run_stride_f32
            run_idx_high = (
                run_idx_low + f32_per_element * shared_elems_per_run
            )
            rx_shared_memory = shared_memory[run_idx_low:run_idx_high].view(
                simsafe_precision
            )
            rx_inits = inits[run_index, :]
            rx_params = params[run_index, :]
            rx_state = state_output[:, run_index * save_state, :]
            rx_observables = observables_output[
                :, run_index * save_observables, :
            ]
            rx_state_summaries = state_summaries_output[
                :, run_index * save_observables, :
            ]
            rx_observables_summaries = observables_summaries_output[
                :, run_index * save_state_summaries, :
            ]
            status = loopfunction(
                rx_inits,
                rx_params,
                c_coefficients,
                rx_shared_memory,
                local_scratch,
                rx_state,
                rx_observables,
                rx_state_summaries,
                rx_observables_summaries,
                duration,
                warmup,
                t0,
            )
            if tx == 0:
                status_codes_output[run_index] = status
            return None
        # no cover: end
        return integration_kernel

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> set[str]:
        """Update solver configuration parameters.

        Parameters
        ----------
        updates_dict
            Mapping of parameter updates forwarded to the single integrator and
            compile settings.
        silent
            Flag suppressing errors when unrecognised parameters remain.
        **kwargs
            Additional parameter overrides merged into ``updates_dict``.

        Returns
        -------
        set[str]
            Names of parameters successfully applied.

        Raises
        ------
        KeyError
            Raised when unknown parameters persist and ``silent`` is ``False``.

        Notes
        -----
        The method applies updates to the single integrator before refreshing
        compile-critical settings so the kernel rebuild picks up new metadata.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        all_unrecognized = set(updates_dict.keys())
        all_unrecognized -= self.single_integrator.update(
                updates_dict, silent=True
        )
        updates_dict.update({
            "loop_function": self.single_integrator.device_function,
            "local_memory_elements": (
                self.single_integrator.local_memory_elements
            ),
            "shared_memory_elements": (
                self.single_integrator.shared_memory_elements
            ),
             "ActiveOutputs": self.output_arrays.active_outputs,
        })

        all_unrecognized -= self.update_compile_settings(
                updates_dict, silent=True
        )

        recognised = set(updates_dict.keys()) - all_unrecognized

        if all_unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {all_unrecognized}")

        return recognised

    @property
    def precision(self) -> PrecisionDType:
        """Precision dtype used in computations."""

        return self.compile_settings.precision

    @property
    def local_memory_elements(self) -> int:
        """Number of precision elements required in local memory per run."""

        return self.compile_settings.local_memory_elements

    @property
    def shared_memory_elements(self) -> int:
        """Number of precision elements required in shared memory per run."""

        return self.compile_settings.shared_memory_elements

    @property
    def ActiveOutputs(self) -> ActiveOutputs:
        """Active output array flags."""

        return self.compile_settings.ActiveOutputs

    @property
    def shared_memory_needs_padding(self) -> bool:
        """Indicate whether shared-memory padding is required.

        Returns
        -------
        bool
            ``True`` when a four-byte skew reduces bank conflicts for single
            precision.

        Notes
        -----
        Shared memory load instructions for ``float64`` require eight-byte
        alignment. Padding in that scenario would misalign alternate runs and
        trigger misaligned-access faults, so padding only applies to single
        precision workloads where the skew preserves alignment.
        """
        if self.precision == np.float64:
            return False
        elif self.shared_memory_elements % 2 == 0:
            return True
        else:
            return False

    def _on_allocation(self, response: Any) -> None:
        """Record the number of chunks required by the memory manager."""
        self.chunks = response.chunks

    @property
    def output_heights(self) -> Any:
        """Height metadata for each host output array."""

        return self.single_integrator.output_array_heights

    @property
    def kernel(self) -> Callable:
        """Compiled integration kernel callable."""

        return self.device_function

    def build(self) -> Callable:
        """Compile the integration kernel and return it."""

        return self.build_kernel()

    @property
    def profileCUDA(self) -> bool:
        """Indicate whether CUDA profiling hooks are enabled."""

        return self._profileCUDA and not is_cudasim_enabled()

    @property
    def memory_manager(self) -> "MemoryManager":
        """Registered memory manager for this kernel."""

        return self._memory_manager

    @property
    def stream_group(self) -> str:
        """Stream group label assigned by the memory manager."""

        return self.memory_manager.get_stream_group(self)

    @property
    def stream(self) -> Any:
        """CUDA stream used for kernel launches."""

        return self.memory_manager.get_stream(self)

    @property
    def mem_proportion(self) -> Optional[float]:
        """Fraction of managed memory reserved for this kernel."""

        return self.memory_manager.proportion(self)

    @property
    def shared_memory_bytes(self) -> int:
        """Shared-memory footprint per run for the compiled kernel."""

        return self.single_integrator.shared_memory_bytes


    @property
    def threads_per_loop(self) -> int:
        """CUDA threads consumed by each run in the loop."""

        return self.single_integrator.threads_per_loop

    @property
    def duration(self) -> float:
        """Requested integration duration."""

        return self.precision(self._duration)

    @duration.setter
    def duration(self, value: float) -> None:
        self._duration = self.precision(value)

    @property
    def dt(self) -> Optional[float]:
        """Current integrator step size when available."""

        return self.single_integrator.dt or None

    @property
    def warmup(self) -> float:
        """Configured warmup duration."""

        return self.precision(self._warmup)

    @warmup.setter
    def warmup(self, value: float) -> None:
        self._warmup = self.precision(value)

    @property
    def t0(self) -> float:
        """Configured initial integration time."""

        return self.precision(self._t0)

    @t0.setter
    def t0(self, value: float) -> None:
        self._t0 = self.precision(value)

    @property
    def output_length(self) -> int:
        """Number of saved trajectory samples in the main run."""

        return int(np.ceil(self.duration / self.single_integrator.dt_save))

    @property
    def summaries_length(self) -> int:
        """Number of summary intervals across the integration window."""

        return int(
            np.ceil(self._duration / self.single_integrator.dt_summarise)
        )

    @property
    def warmup_length(self) -> int:
        """Number of warmup save intervals completed before capturing output."""

        return int(np.ceil(self._warmup / self.single_integrator.dt_save))

    @property
    def system(self) -> "BaseODE":
        """Underlying ODE system handled by the kernel."""

        return self.single_integrator.system

    @property
    def algorithm(self) -> str:
        """Identifier of the selected integration algorithm."""

        return self.single_integrator.algorithm_key

    @property
    def dt_min(self) -> float:
        """Minimum allowable step size from the controller."""

        return self.single_integrator.dt_min

    @property
    def dt_max(self) -> float:
        """Maximum allowable step size from the controller."""

        return self.single_integrator.dt_max

    @property
    def atol(self) -> float:
        """Absolute error tolerance applied during adaptive stepping."""

        return self.single_integrator.atol

    @property
    def rtol(self) -> float:
        """Relative error tolerance applied during adaptive stepping."""

        return self.single_integrator.rtol

    @property
    def dt_save(self) -> float:
        """Interval between saved samples from the loop."""

        return self.single_integrator.dt_save

    @property
    def dt_summarise(self) -> float:
        """Interval between summary reductions from the loop."""

        return self.single_integrator.dt_summarise

    @property
    def system_sizes(self) -> Any:
        """Structured size metadata for the system."""

        return self.single_integrator.system_sizes

    @property
    def output_array_heights(self) -> Any:
        """Height metadata for the batched output arrays."""

        return self.single_integrator.output_array_heights

    @property
    def ouput_array_sizes_2d(self) -> SingleRunOutputSizes:
        """Two-dimensional output sizes for individual runs."""

        return SingleRunOutputSizes.from_solver(self)

    @property
    def output_array_sizes_3d(self) -> BatchOutputSizes:
        """Three-dimensional output sizes for batched runs."""

        return BatchOutputSizes.from_solver(self)

    @property
    def summaries_buffer_sizes(self) -> Any:
        """Device buffer sizes required for summary reductions."""

        return self.single_integrator.summaries_buffer_sizes

    @property
    def summary_legend_per_variable(self) -> Any:
        """Legend entries describing each summarised variable."""

        return self.single_integrator.summary_legend_per_variable

    @property
    def saved_state_indices(self) -> Any:
        """Indices of saved state variables."""

        return self.single_integrator.saved_state_indices

    @property
    def saved_observable_indices(self) -> Any:
        """Indices of saved observable variables."""

        return self.single_integrator.saved_observable_indices

    @property
    def summarised_state_indices(self) -> Any:
        """Indices of summarised state variables."""

        return self.single_integrator.summarised_state_indices

    @property
    def summarised_observable_indices(self) -> Any:
        """Indices of summarised observable variables."""

        return self.single_integrator.summarised_observable_indices

    @property
    def active_output_arrays(self) -> "ActiveOutputs":
        """Active output flags after ensuring arrays are allocated."""

        self.output_arrays.allocate()
        return self.output_arrays.active_outputs

    @property
    def device_state_array(self) -> Any:
        """Device buffer storing saved state trajectories."""

        return self.output_arrays.device_state

    @property
    def device_observables_array(self) -> Any:
        """Device buffer storing saved observable trajectories."""

        return self.output_arrays.device_observables

    @property
    def device_state_summaries_array(self) -> Any:
        """Device buffer storing state summary reductions."""

        return self.output_arrays.device_state_summaries

    @property
    def device_observable_summaries_array(self) -> Any:
        """Device buffer storing observable summary reductions."""

        return self.output_arrays.device_observable_summaries

    @property
    def d_statuscodes(self) -> Any:
        """Device buffer storing integration status codes."""

        return self.output_arrays.device_status_codes

    @property
    def state(self) -> Any:
        """Host view of saved state trajectories."""

        return self.output_arrays.state

    @property
    def observables(self) -> Any:
        """Host view of saved observable trajectories."""

        return self.output_arrays.observables

    @property
    def state_summaries(self) -> Any:
        """Host view of state summary reductions."""

        return self.output_arrays.state_summaries

    @property
    def status_codes(self) -> Any:
        """Host view of integration status codes."""

        return self.output_arrays.status_codes

    @property
    def observable_summaries(self) -> Any:
        """Host view of observable summary reductions."""

        return self.output_arrays.observable_summaries

    @property
    def initial_values(self) -> Any:
        """Host view of initial state values."""

        return self.input_arrays.initial_values

    @property
    def parameters(self) -> Any:
        """Host view of parameter tables."""

        return self.input_arrays.parameters

    @property
    def driver_coefficients(self) -> Optional[NDArray[np.floating]]:
        """Horner-ordered driver coefficients on the host."""

        return self.input_arrays.driver_coefficients

    @property
    def device_driver_coefficients(self) -> Optional[NDArray[np.floating]]:
        """Device-resident driver coefficients."""

        return self.input_arrays.device_driver_coefficients

    @property
    def state_stride_order(self) -> Tuple[str, ...]:
        """Stride order for state arrays on the host."""

        return self.output_arrays.host.state.stride_order

    @property
    def save_time(self) -> float:
        """Elapsed time spent saving outputs during integration."""

        return self.single_integrator.save_time

    def enable_profiling(self) -> None:
        """Enable CUDA profiling hooks for subsequent launches."""

        self._profileCUDA = True

    def disable_profiling(self) -> None:
        """Disable CUDA profiling hooks for subsequent launches."""

        self._profileCUDA = False

    @property
    def output_types(self) -> Any:
        """Active output type identifiers configured for the run."""

        return self.single_integrator.output_types
