# -*- coding: utf-8 -*-
"""CUDA batch solver kernel utilities."""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from warnings import warn
from pathlib import Path

from numpy import ceil as np_ceil, float64 as np_float64, floating
from numba import cuda, float64
from numba import int32

from attrs import define, field

from cubie.odesystems import SymbolicODE
from cubie.cuda_simsafe import is_cudasim_enabled, compile_kwargs
from cubie.cubie_cache import (
    CacheConfig,
    CubieCacheHandler,
)

from cubie.time_logger import CUDAEvent
from numpy.typing import NDArray

from cubie.memory import default_memmgr
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import CUDAFactory, CUDADispatcherCache
from cubie.batchsolving.arrays.BatchInputArrays import InputArrays
from cubie.batchsolving.arrays.BatchOutputArrays import (
    OutputArrays,
)
from cubie.batchsolving.BatchSolverConfig import ActiveOutputs
from cubie.batchsolving.BatchSolverConfig import BatchSolverConfig
from cubie.odesystems.baseODE import BaseODE
from cubie.outputhandling.output_sizes import (
    BatchOutputSizes,
    SingleRunOutputSizes,
)
from cubie.outputhandling.output_config import OutputCompileFlags
from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from cubie._utils import unpack_dict_values

if TYPE_CHECKING:
    from cubie.memory import MemoryManager

DEFAULT_MEMORY_SETTINGS = {
    "memory_manager": default_memmgr,
    "stream_group": "solver",
    "mem_proportion": None,
}


@define(frozen=True)
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


@define()
class BatchSolverCache(CUDADispatcherCache):
    solver_kernel: Union[int, Callable] = field(default=-1)


class BatchSolverKernel(CUDAFactory):
    """Factory for CUDA kernel which coordinates a batch integration.

    Parameters
    ----------
    system
        ODE system describing the problem to integrate.
    loop_settings
        Mapping of loop configuration forwarded to
        :class:`cubie.integrators.SingleIntegratorRun`. Recognised keys include
        ``"save_every"`` and ``"summarise_every"``.
    evaluate_driver_at_t
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
        system: "SymbolicODE",
        loop_settings: Optional[Dict[str, Any]] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        driver_del_t: Optional[Callable] = None,
        profileCUDA: bool = False,
        step_control_settings: Optional[Dict[str, Any]] = None,
        algorithm_settings: Optional[Dict[str, Any]] = None,
        output_settings: Optional[Dict[str, Any]] = None,
        memory_settings: Optional[Dict[str, Any]] = None,
        cache_settings: Optional[Dict[str, Any]] = None,
        cache: Union[bool, str, Path] = True,
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

        # CUDA event tracking for timing
        self._cuda_events: List = []
        self._gpu_workload_event: Optional[CUDAEvent] = None

        self._memory_manager = self._setup_memory_manager(memory_settings)

        # Build the single integrator to derive compile-critical metadata
        self.single_integrator = SingleIntegratorRun(
            system,
            loop_settings=loop_settings,
            evaluate_driver_at_t=evaluate_driver_at_t,
            driver_del_t=driver_del_t,
            step_control_settings=step_control_settings,
            algorithm_settings=algorithm_settings,
            output_settings=output_settings,
        )

        # Extract system identification for cache
        system_name = system.name
        system_hash = system.fn_hash
        if system_name == system_hash:
            system_name = f"unnamed_{system_hash[:8]}"

        # Build cache settings dict from cache_settings
        if cache_settings is None:
            cache_settings = {}

        # Initialize cache_handler BEFORE setup_compile_settings since
        # _invalidate_cache is called during setup and requires cache_handler
        self.cache_handler = CubieCacheHandler(
            cache_arg=cache,
            system_name=system_name,
            system_hash=system_hash,
            **cache_settings,
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
            compile_flags=self.single_integrator.output_compile_flags,
        )
        self.setup_compile_settings(initial_config)

        self.input_arrays = InputArrays.from_solver(self)
        self.output_arrays = OutputArrays.from_solver(self)

        self.output_arrays.update(self)
        self.update_compile_settings(
            {
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

    def _setup_cuda_events(self, chunks: int) -> None:
        """Create CUDA events for timing instrumentation.

        Parameters
        ----------
        chunks : int
            Number of chunks to process

        Notes
        -----
        Creates one GPU workload event and 3 events per chunk
        (h2d_transfer, kernel, d2h_transfer).
        Events are created regardless of verbosity - they become no-ops
        internally when verbosity is None.
        """
        # Create overall GPU workload event
        self._gpu_workload_event = CUDAEvent("gpu_workload")

        # Create per-chunk events (3 events per chunk: h2d, kernel, d2h)
        self._cuda_events = []
        for i in range(chunks):
            h2d_event = CUDAEvent(f"h2d_transfer_chunk_{i}")
            kernel_event = CUDAEvent(f"kernel_chunk_{i}")
            d2h_event = CUDAEvent(f"d2h_transfer_chunk_{i}")
            self._cuda_events.extend([h2d_event, kernel_event, d2h_event])

    def _get_chunk_events(self, chunk_idx: int) -> Tuple:
        """Get the three CUDA events for a specific chunk.

        Parameters
        ----------
        chunk_idx : int
            Chunk index (0-based)

        Returns
        -------
        tuple
            (h2d_event, kernel_event, d2h_event) for the chunk
        """
        base_idx = chunk_idx * 3
        return (
            self._cuda_events[base_idx],
            self._cuda_events[base_idx + 1],
            self._cuda_events[base_idx + 2],
        )

    def _validate_timing_parameters(self, duration: float) -> None:
        """Validate timing parameters to prevent invalid array accesses.

        Parameters
        ----------
        duration
            Integration duration in time units.

        Raises
        ------
        ValueError
            When timing parameters would result in no outputs or invalid
            sampling.

        Notes
        -----
        Uses dt_min as an absolute tolerance when comparing floating
        point timing parameters by adding dt_min to the requested
        duration. Small in-loop timing oversteps smaller than dt_min
        are treated as valid and do not trigger validation errors.
        """
        integrator = self.single_integrator
        end_time = self.precision(duration) + self.dt_min

        # Validate time-domain output timing parameters
        if integrator.has_time_domain_outputs:
            save_every = integrator.save_every
            save_last = integrator.save_last
            if (
                save_every is not None
                and save_every > end_time
                and not save_last
            ):
                raise ValueError(
                    f"save_every ({save_every}) > duration ({duration}) "
                    f"so this loop will produce no outputs"
                )

        # Validate summary timing parameters
        if integrator.has_summary_outputs:
            sample_summaries_every = integrator.sample_summaries_every
            summarise_every = integrator.summarise_every

            if sample_summaries_every is None:
                raise ValueError(
                    "Summary outputs are enabled but sample_summaries_every "
                    "is None"
                )
            if summarise_every is None:
                raise ValueError(
                    "Summary outputs are enabled but summarise_every is None"
                )

            if sample_summaries_every >= summarise_every:
                raise ValueError(
                    f"sample_summaries_every ({sample_summaries_every}) "
                    f">= summarise_every ({summarise_every}); "
                    f"The saved summary will be based on 0 samples, so will "
                    f"result in 0/inf/NaN values."
                )

            if summarise_every > end_time:
                raise ValueError(
                    f"summarise_every ({summarise_every}) > duration "
                    f"({duration}), so this loop will produce no summary "
                    f"outputs"
                )

    def run(
        self,
        inits: NDArray[floating],
        params: NDArray[floating],
        driver_coefficients: Optional[NDArray[floating]],
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

        # Time parameters always use float64 for accumulation accuracy
        duration = np_float64(duration)
        warmup = np_float64(warmup)
        t0 = np_float64(t0)

        self._duration = duration
        self._warmup = warmup
        self._t0 = t0

        # inits is in (variable, run) format - run count is in shape[1]
        numruns = inits.shape[1]
        self.num_runs = numruns  # Don't delete - generates batchoutputsizes

        # Update the single integrator with requested duration if required
        self.single_integrator.set_summary_timing_from_duration(duration)

        # Validate timing parameters to prevent array index errors
        self._validate_timing_parameters(duration)

        # Refresh compile-critical settings before array updates
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
            }
        )

        # Queue allocations
        self.input_arrays.update(self, inits, params, driver_coefficients)
        self.output_arrays.update(self)

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

        # Update internal time parameters if duration has been chunked
        if self.chunks != 1 and chunk_axis == "time":
            self.single_integrator.set_summary_timing_from_duration(
                chunk_params.duration
            )
            try:
                self._validate_timing_parameters(chunk_params.duration)
            except ValueError as e:
                raise ValueError(
                    f"Your timing parameters were OK for the full duration, "
                    f"but the run was divided into multiple time-chunks due "
                    f"to GPU memory constraints so they're now invalid. "
                    f"Adjust timing parameters OR set chunk_axis='run' to "
                    f"avoid this. Time-check exception: {e}"
                ) from e

        # Use the chunk-local run count for run-chunking, and the full run
        # count for time-chunking.
        if chunk_axis == "run":
            kernel_runs = int(chunk_params.runs)
        else:
            kernel_runs = int(numruns)

        pad = 4 if self.shared_memory_needs_padding else 0
        padded_bytes = self.shared_memory_bytes + pad
        dynamic_sharedmem = int(padded_bytes * min(kernel_runs, blocksize))

        blocksize, dynamic_sharedmem = self.limit_blocksize(
            blocksize,
            dynamic_sharedmem,
            padded_bytes,
            kernel_runs,
        )

        # We need a nonzero number to tell the compiler we're using dynamic
        # memory. If zero, then the cuda.shared.array(0) call fails as we
        # can't declare a size-0 static shared memory array.
        dynamic_sharedmem = max(4, dynamic_sharedmem)
        threads_per_loop = self.single_integrator.threads_per_loop
        runsperblock = int(blocksize / self.single_integrator.threads_per_loop)
        BLOCKSPERGRID = int(max(1, np_ceil(kernel_runs / blocksize)))

        # Update cache for this configuration and attach
        cfg_hash = self.config_hash
        self.kernel._cache = self.cache_handler.configured_cache(cfg_hash)

        if self.profileCUDA:  # pragma: no cover
            cuda.profile_start()

        # Setup CUDA events for timing (no-op when verbosity is None)
        self._setup_cuda_events(self.chunks)

        # Record start of overall GPU workload
        self._gpu_workload_event.record_start(stream)

        for i in range(self.chunks):
            indices = slice(i * chunk_params.size, (i + 1) * chunk_params.size)

            # Get events for this chunk
            h2d_event, kernel_event, d2h_event = self._get_chunk_events(i)

            # h2d transfer timing
            h2d_event.record_start(stream)
            self.input_arrays.initialise(indices)
            self.output_arrays.initialise(indices)
            h2d_event.record_end(stream)

            # Don't use warmup in runs starting after t=t0
            if (chunk_axis == "time") and (i != 0):
                chunk_warmup = np_float64(0.0)
                chunk_t0 = t0 + np_float64(i) * chunk_params.duration

            # Kernel execution timing
            kernel_event.record_start(stream)
            self.kernel[
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
                self.output_arrays.device_iteration_counters,
                self.output_arrays.device_status_codes,
                chunk_params.duration,
                chunk_warmup,
                chunk_t0,
                kernel_runs,
            )
            kernel_event.record_end(stream)

            # We don't want to sync between chunks, we should queue runs and
            # transfers in the stream and sync before final result fetch.
            # self.memory_manager.sync_stream(self)

            # d2h transfer timing
            d2h_event.record_start(stream)
            self.input_arrays.finalise(indices)
            self.output_arrays.finalise(indices)
            d2h_event.record_end(stream)

        # Finalize GPU workload timing
        self._gpu_workload_event.record_end(stream)

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
        while dynamic_sharedmem >= 32768:
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
            dynamic_sharedmem = int(bytes_per_run * min(numruns, blocksize))
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
            chunkruns = int(np_ceil(numruns / chunks))
            chunksize = chunkruns
        elif chunk_axis == "time":
            chunk_duration = duration / chunks
            chunksize = int(np_ceil(self.output_length / chunks))
            chunkruns = numruns

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

        if "lineinfo" in compile_kwargs:
            compile_kwargs["lineinfo"] = self.profileCUDA

        loopfunction = self.single_integrator.device_function

        output_flags = self.active_outputs
        save_state = output_flags.state
        save_observables = output_flags.observables
        save_state_summaries = output_flags.state_summaries
        save_observable_summaries = output_flags.observable_summaries
        needs_padding = self.shared_memory_needs_padding

        shared_elems_per_run = config.shared_memory_elements
        f32_per_element = 2 if (precision is float64) else 1
        f32_pad_perrun = 1 if needs_padding else 0
        run_stride_f32 = int(
            (f32_per_element * shared_elems_per_run + f32_pad_perrun)
        )

        # Get memory allocators from buffer registry
        alloc_shared, alloc_persistent = (
            buffer_registry.get_toplevel_allocators(self)
        )

        # no cover: start
        @cuda.jit(
            **compile_kwargs,
        )
        def integration_kernel(
            inits,
            params,
            d_coefficients,
            state_output,
            observables_output,
            state_summaries_output,
            observables_summaries_output,
            iteration_counters_output,
            status_codes_output,
            duration,
            warmup,
            t0,
            n_runs,
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
            iteration_counters_output
                Device array storing iteration counter values at each save point.
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
            tx = int32(cuda.threadIdx.x)
            ty = int32(cuda.threadIdx.y)
            block_index = int32(cuda.blockIdx.x)
            runs_per_block = int32(cuda.blockDim.y)
            run_index = int32(runs_per_block * block_index + ty)
            if run_index >= n_runs:
                return None
            shared_memory = alloc_shared()
            persistent_local = alloc_persistent()
            c_coefficients = cuda.const.array_like(d_coefficients)
            run_idx_low = int32(ty * run_stride_f32)
            run_idx_high = int32(
                run_idx_low + f32_per_element * shared_elems_per_run
            )
            rx_shared_memory = shared_memory[run_idx_low:run_idx_high].view(
                simsafe_precision
            )
            rx_inits = inits[:, run_index]
            rx_params = params[:, run_index]
            rx_state = state_output[:, :, run_index * save_state]
            rx_observables = observables_output[
                :, :, run_index * save_observables
            ]
            rx_state_summaries = state_summaries_output[
                :, :, run_index * save_state_summaries
            ]
            rx_observables_summaries = observables_summaries_output[
                :, :, run_index * save_observable_summaries
            ]
            rx_iteration_counters = iteration_counters_output[:, :, run_index]
            status = loopfunction(
                rx_inits,
                rx_params,
                c_coefficients,
                rx_shared_memory,
                persistent_local,
                rx_state,
                rx_observables,
                rx_state_summaries,
                rx_observables_summaries,
                rx_iteration_counters,
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

        # Flatten nested dict values so that grouped settings can be passed
        # naturally. For example, step_controller_settings={'dt_min': 0.01}
        # becomes dt_min=0.01, allowing sub-components to recognize and
        # apply parameters correctly.
        updates_dict, unpacked_keys = unpack_dict_values(updates_dict)

        all_unrecognized = set(updates_dict.keys())
        all_unrecognized -= self.single_integrator.update(
            updates_dict, silent=True
        )

        updates_dict.update(
            {
                "loop_fn": self.single_integrator.device_function,
                "local_memory_elements": (
                    self.single_integrator.local_memory_elements
                ),
                "shared_memory_elements": (
                    self.single_integrator.shared_memory_elements
                ),
                "compile_flags": self.single_integrator.output_compile_flags,
            }
        )

        all_unrecognized -= self.update_compile_settings(
            updates_dict, silent=True
        )

        all_unrecognized -= self.cache_handler.update(
            updates_dict, silent=True
        )

        recognised = set(updates_dict.keys()) - all_unrecognized

        if all_unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {all_unrecognized}")

        # Include unpacked dict keys in recognized set
        return recognised | unpacked_keys

    @property
    def local_memory_elements(self) -> int:
        """Number of precision elements required in local memory per run."""

        return self.compile_settings.local_memory_elements

    @property
    def shared_memory_elements(self) -> int:
        """Number of precision elements required in shared memory per run."""

        return self.compile_settings.shared_memory_elements

    @property
    def compile_flags(self) -> OutputCompileFlags:
        """Boolean compile-time controls for which output features are enabled."""

        return self.compile_settings.compile_flags

    @property
    def active_outputs(self) -> ActiveOutputs:
        """Active output array flags derived from compile_flags."""

        return self.compile_settings.active_outputs

    @property
    def cache_config(self) -> "CacheConfig":
        """Cache configuration for the kernel, parsed on demand."""
        return self.cache_handler.config

    def set_cache_dir(self, path: Union[str, Path]) -> None:
        """Set a custom cache directory for compiled kernels.

        Parameters
        ----------
        path
            New cache directory path. Can be absolute or relative.
        """
        self.cache_handler.update(cache_dir=Path(path))

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
        if self.precision == np_float64:
            return False
        elif self.shared_memory_elements == 0:
            return False
        elif self.shared_memory_elements % 2 == 0:
            return True
        else:
            return False

    def _on_allocation(self, response: Any) -> None:
        """Record the number of chunks required by the memory manager."""
        self.chunks = response.chunks

    def _invalidate_cache(self) -> None:
        """Mark cached outputs as invalid, flushing cache if cache_handler
        in "flush on change" mode."""
        super()._invalidate_cache()
        self.cache_handler.invalidate()

    @property
    def output_heights(self) -> Any:
        """Height metadata for each host output array."""

        return self.single_integrator.output_array_heights

    @property
    def kernel(self) -> Callable:
        """Compiled integration kernel callable."""
        return self.device_function

    @property
    def device_function(self):
        return self.get_cached_output("solver_kernel")

    def build(self) -> BatchSolverCache:
        """Compile the integration kernel and return it."""
        return BatchSolverCache(solver_kernel=self.build_kernel())

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

        return np_float64(self._duration)

    @duration.setter
    def duration(self, value: float) -> None:
        self._duration = np_float64(value)

    @property
    def dt(self) -> Optional[float]:
        """Current integrator step size when available."""

        return self.single_integrator.dt or None

    @property
    def warmup(self) -> float:
        """Configured warmup duration."""

        return np_float64(self._warmup)

    @warmup.setter
    def warmup(self, value: float) -> None:
        self._warmup = np_float64(value)

    @property
    def t0(self) -> float:
        """Configured initial integration time."""

        return np_float64(self._t0)

    @t0.setter
    def t0(self, value: float) -> None:
        self._t0 = np_float64(value)

    @property
    def output_length(self) -> int:
        """Number of saved trajectory samples in the main run.

        Delegates to SingleIntegratorRun.output_length() with the current
        duration.
        """
        return self.single_integrator.output_length(self._duration)

    @property
    def summaries_length(self) -> int:
        """Number of complete summary intervals across the integration window.

        Delegates to SingleIntegratorRun.summaries_length() with the current
        duration.
        """
        return self.single_integrator.summaries_length(self._duration)

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
    def save_every(self) -> Optional[float]:
        """Interval between saved samples from the loop, or None if save_last only."""

        return self.single_integrator.save_every

    @property
    def summarise_every(self) -> Optional[float]:
        """Interval between summary reductions from the loop"""

        return self.single_integrator.summarise_every

    @property
    def sample_summaries_every(self) -> float:
        """Interval between summary metric samples from the loop."""

        return self.single_integrator.sample_summaries_every

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
    def summary_legend_per_variable(self) -> Any:
        """Legend entries describing each summarised variable."""

        return self.single_integrator.summary_legend_per_variable

    @property
    def summary_unit_modifications(self) -> Any:
        """Unit modifications for each summarised variable."""

        return self.single_integrator.summary_unit_modifications

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
    def iteration_counters(self) -> Any:
        """Host view of iteration counters at each save point."""

        return self.output_arrays.iteration_counters

    @property
    def initial_values(self) -> Any:
        """Host view of initial state values."""

        return self.input_arrays.initial_values

    @property
    def parameters(self) -> Any:
        """Host view of parameter tables."""

        return self.input_arrays.parameters

    @property
    def driver_coefficients(self) -> Optional[NDArray[floating]]:
        """Horner-ordered driver coefficients on the host."""

        return self.input_arrays.driver_coefficients

    @property
    def device_driver_coefficients(self) -> Optional[NDArray[floating]]:
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

    def set_stride_order(self, order: Tuple[str]) -> None:
        """Set the stride order for device arrays.

        Parameters
        ----------
        order
            Tuple of labels in ["time", "run", "variable"]. The last string in
            this order is the contiguous dimension on chip.
        """
        self.memory_manager.set_global_stride_ordering(order)

    @property
    def output_types(self) -> Any:
        """Active output type identifiers configured for the run."""

        return self.single_integrator.output_types
