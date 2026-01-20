# -*- coding: utf-8 -*-
"""CUDA batch solver kernel utilities.

Notes
-----
Chunking is performed along the run axis when memory constraints require
splitting the batch. This chunking is automatic and transparent to users.
"""

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

from attrs import define, field, evolve

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
from cubie._utils import unpack_dict_values, getype_validator

if TYPE_CHECKING:
    from cubie.memory import MemoryManager
    from cubie.memory.array_requests import ArrayResponse


DEFAULT_MEMORY_SETTINGS = {
    "memory_manager": default_memmgr,
    "stream_group": "solver",
    "mem_proportion": None,
}


@define(frozen=True)
class RunParams:
    """Run parameters with optional chunking metadata.

    Chunking always occurs along the run axis.

    Parameters
    ----------
    duration : float
        Full duration of the simulation window.
    warmup : float
        Full warmup time before the main simulation.
    t0 : float
        Initial integration time.
    runs : int
        Total number of runs in the batch.
    num_chunks : int, default=1
        Number of chunks the batch is divided into.
    chunk_length : int, default=0
        Number of runs per chunk (except possibly the last).

    Notes
    -----
    When num_chunks=1, no chunking has occurred.
    When num_chunks>1, chunk_length represents the standard chunk size.
    """

    duration: float = field(validator=getype_validator(float, 0.0))
    warmup: float = field(validator=getype_validator(float, 0.0))
    t0: float = field(validator=getype_validator(float, 0.0))
    runs: int = field(validator=getype_validator(int, 1))
    num_chunks: int = field(
        default=1, repr=False, validator=getype_validator(int, 1)
    )
    chunk_length: int = field(
        default=0, repr=False, validator=getype_validator(int, 0)
    )

    def __getitem__(self, index: int) -> "RunParams":
        """Return RunParams for a specific chunk.

        Parameters
        ----------
        index : int
            Chunk index (0-based).

        Returns
        -------
        RunParams
            New RunParams instance with runs set to chunk size.

        Raises
        ------
        IndexError
            If index is out of range [0, num_chunks).

        Notes
        -----
        For the last chunk (index == num_chunks - 1), the number of runs
        is calculated as runs - (num_chunks - 1) * chunk_length to handle
        the "dangling" chunk case.
        """
        # Validation
        if index < 0 or index >= self.num_chunks:
            raise IndexError(
                f"Chunk index {index} out of range "
                f"(valid range: 0 to {self.num_chunks - 1})"
            )

        if index == self.num_chunks - 1:
            # Last chunk: calculate remaining runs
            chunk_runs = self.runs - (self.num_chunks - 1) * self.chunk_length
        else:
            chunk_runs = self.chunk_length

        return evolve(self, runs=chunk_runs)

    def update_from_allocation(self, response: "ArrayResponse") -> "RunParams":
        """Update with chunking metadata from allocation response.

        Parameters
        ----------
        response : ArrayResponse
            Allocation response containing chunking information.

        Returns
        -------
        RunParams
            New RunParams instance with updated chunking metadata.

        Notes
        -----
        Extracts num_chunks and chunk_length from the response. When
        num_chunks=1, chunk_length is set equal to runs (no chunking).
        """

        return evolve(
            self,
            num_chunks=response.chunks,
            chunk_length=response.chunk_length,
        )


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

        # Initialize run parameters with defaults
        self.run_params = RunParams(
            duration=precision(0.0),
            warmup=precision(0.0),
            t0=precision(0.0),
            runs=1,
        )

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
            compile_flags=self.single_integrator.output_compile_flags,
        )
        self.setup_compile_settings(initial_config)

        self.input_arrays = InputArrays.from_solver(self)
        self.output_arrays = OutputArrays.from_solver(self)

        self.output_arrays.update(self)

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
    ) -> None:
        """Execute the solver kernel for batch integration.

        Chunking is performed along the run axis when memory constraints
        require splitting the batch.

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

        # Update run params with actual values before allocation
        self.run_params = RunParams(
            duration=duration,
            warmup=np_float64(warmup),
            t0=np_float64(t0),
            runs=inits.shape[1],
        )

        # Update the single integrator with requested duration if required
        self.single_integrator.set_summary_timing_from_duration(duration)

        # Validate timing parameters to prevent array index errors
        self._validate_timing_parameters(duration)

        # Refresh compile-critical settings before array updates
        self.update_compile_settings(
            {
                "loop_fn": self.single_integrator.compiled_loop_function,
                "precision": self.single_integrator.precision,
            }
        )

        # Queue allocations
        self.input_arrays.update(self, inits, params, driver_coefficients)
        self.output_arrays.update(self)

        # Process allocations into chunks
        self.memory_manager.allocate_queue(self)

        # ------------ from here on dimensions are "chunked" -----------------
        # self.run_params is updated in the on_allocation callback.
        chunks = self.run_params.num_chunks

        # Get first chunk runs for initial block size calculation
        first_chunk_params = self.run_params[0]
        runs = first_chunk_params.runs

        # Add 4-byte padding when required by GPU architecture to ensure
        # proper alignment of shared memory allocations per thread block
        pad = 4 if self.shared_memory_needs_padding else 0
        padded_bytes = self.shared_memory_bytes + pad
        dynamic_sharedmem = int(padded_bytes * min(runs, blocksize))

        blocksize, dynamic_sharedmem = self.limit_blocksize(
            blocksize,
            dynamic_sharedmem,
            padded_bytes,
            runs,
        )

        # We need a nonzero number to tell the compiler we're using dynamic
        # memory. If zero, then the cuda.shared.array(0) call fails as we
        # can't declare a size-0 static shared memory array.
        dynamic_sharedmem = max(4, dynamic_sharedmem)
        threads_per_loop = self.single_integrator.threads_per_loop
        runsperblock = int(blocksize / self.single_integrator.threads_per_loop)

        if self.profileCUDA:  # pragma: no cover
            cuda.profile_start()

        # Setup CUDA events for timing (no-op when verbosity is None)
        self._setup_cuda_events(chunks)

        # Record start of overall GPU workload
        self._gpu_workload_event.record_start(stream)
        precision = self.precision
        for i in range(chunks):
            # Get parameters for this specific chunk
            chunk_run_params = self.run_params[i]
            duration = precision(chunk_run_params.duration)
            warmup = precision(chunk_run_params.warmup)
            t0 = precision(chunk_run_params.t0)

            # Use the chunk-local run count
            runs = chunk_run_params.runs

            # Recompute blocks needed for this chunk's actual run count
            chunk_blocks = int(max(1, np_ceil(runs / blocksize)))

            # Get events for this chunk
            h2d_event, kernel_event, d2h_event = self._get_chunk_events(i)

            # h2d transfer timing
            h2d_event.record_start(stream)
            self.input_arrays.initialise(i)
            self.output_arrays.initialise(i)
            h2d_event.record_end(stream)

            # Kernel execution timing
            kernel_event.record_start(stream)
            self.kernel[
                chunk_blocks,
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
                duration,
                warmup,
                t0,
                runs,
            )
            kernel_event.record_end(stream)

            # d2h transfer timing
            d2h_event.record_start(stream)
            self.input_arrays.finalise(i)
            self.output_arrays.finalise(i)
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
                    "performance hit."
                )
            blocksize = int(blocksize // 2)
            dynamic_sharedmem = int(bytes_per_run * min(numruns, blocksize))
        return blocksize, dynamic_sharedmem

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

        # Query buffer_registry for current shared memory size
        # This ensures build_kernel uses the actual registered buffer size,
        # which may have been updated via buffer_registry.update()
        shared_elems_per_run = self.shared_memory_elements
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

        # Update cache for this configuration and attach
        cfg_hash = self.config_hash
        integration_kernel._cache = self.cache_handler.configured_cache(
            self.system.fn_hash, cfg_hash
        )
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

        # Allow buffer_registry to recognize and update buffer location parameters
        # (e.g., 'state_location', 'proposed_state_location'). This delegates
        # location management to buffer_registry, following the same pattern as
        # IVPLoop.update().
        all_unrecognized -= buffer_registry.update(
            self.single_integrator._loop, updates_dict, silent=True
        )

        updates_dict.update(
            {
                "loop_fn": self.single_integrator.device_function,
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

    def wait_for_writeback(self):
        """Wait for async writebacks into host arrays after chunked runs"""
        self.output_arrays.wait_pending()

    @property
    def local_memory_elements(self) -> int:
        """Number of precision elements required in local memory per run."""
        return self.single_integrator.local_memory_elements

    @property
    def shared_memory_elements(self) -> int:
        """Number of precision elements required in shared memory per run."""
        return self.single_integrator.shared_memory_elements

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

    def _on_allocation(self, response: "ArrayResponse") -> None:
        """Update run parameters with chunking metadata from allocation."""
        self.run_params = self.run_params.update_from_allocation(response)

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
        return np_float64(self.run_params.duration)

    @duration.setter
    def duration(self, value: float) -> None:
        oldparams = self.run_params
        self.run_params = evolve(oldparams, duration=np_float64(value))

    @property
    def dt(self) -> Optional[float]:
        """Current integrator step size when available."""
        return self.single_integrator.dt or None

    @property
    def warmup(self) -> float:
        """Configured warmup duration."""
        return np_float64(self.run_params.warmup)

    @warmup.setter
    def warmup(self, value: float) -> None:
        oldparams = self.run_params
        self.run_params = evolve(oldparams, warmup=np_float64(value))

    @property
    def t0(self) -> float:
        """Configured initial integration time."""
        return np_float64(self.run_params.t0)

    @t0.setter
    def t0(self, value: float) -> None:
        oldparams = self.run_params
        self.run_params = evolve(oldparams, t0=np_float64(value))

    @property
    def num_runs(self) -> int:
        """Number of runs scheduled for the batch integration."""
        return self.run_params.runs

    @num_runs.setter
    def num_runs(self, value: int) -> None:
        oldparams = self.run_params
        self.run_params = evolve(oldparams, runs=value)

    @property
    def chunks(self):
        """Number of chunks in the most recent run."""
        return self.run_params.num_chunks

    @property
    def total_runs(self) -> int:
        """Total number of runs in the full batch."""
        return self.run_params.runs

    @property
    def output_length(self) -> int:
        """Number of saved trajectory samples in the main run.

        Delegates to SingleIntegratorRun.output_length() with the current
        duration.
        """
        return self.single_integrator.output_length(self.duration)

    @property
    def summaries_length(self) -> int:
        """Number of complete summary intervals across the integration window.

        Delegates to SingleIntegratorRun.summaries_length() with the current
        duration.
        """
        return self.single_integrator.summaries_length(self.duration)

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
