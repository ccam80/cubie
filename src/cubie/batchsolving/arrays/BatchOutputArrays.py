"""Manage output array lifecycles for batch solver executions."""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

from attrs import define, field
from attrs.validators import instance_of as attrsval_instance_of
from numba import cuda
from numpy import (
    float32 as np_float32,
    floating as np_floating,
    int32 as np_int32,
    integer as np_integer,
    issubdtype as np_issubdtype,
)
from numpy.typing import NDArray

from cubie.outputhandling.output_sizes import BatchOutputSizes
from cubie.memory.mem_manager import ArrayResponse
from cubie.batchsolving.arrays.BaseArrayManager import (
    ArrayContainer,
    BaseArrayManager,
    ManagedArray,
)
from cubie.batchsolving import ArrayTypes
from cubie.memory.chunk_buffer_pool import ChunkBufferPool
from cubie.batchsolving.writeback_watcher import (
    WritebackWatcher,
    PendingBuffer,
)
from cubie.cuda_simsafe import CUDA_SIMULATION

ChunkIndices = Union[slice, NDArray[np_integer]]


@define(slots=False)
class OutputArrayContainer(ArrayContainer):
    """Container for batch output arrays."""

    state: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("time", "variable", "run"),
            default_shape=(1, 1, 1),
        )
    )
    observables: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("time", "variable", "run"),
            default_shape=(1, 1, 1),
        )
    )
    state_summaries: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("time", "variable", "run"),
            default_shape=(1, 1, 1),
        )
    )
    observable_summaries: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("time", "variable", "run"),
            default_shape=(1, 1, 1),
        )
    )
    status_codes: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_int32,
            stride_order=("run",),
            default_shape=(1,),
        )
    )
    iteration_counters: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_int32,
            stride_order=("time", "variable", "run"),
            default_shape=(1, 4, 1),
        )
    )

    @classmethod
    def host_factory(
        cls, memory_type: str = "pinned"
    ) -> "OutputArrayContainer":
        """
        Create a new host memory container.

        Parameters
        ----------
        memory_type
            Memory type for host arrays: "pinned" or "host".
            Default is "pinned" for non-chunked operation.

        Returns
        -------
        OutputArrayContainer
            A new container configured for the specified memory type.

        Notes
        -----
        Uses pinned (page-locked) memory to enable asynchronous
        device-to-host transfers with CUDA streams. Using ``"host"``
        memory type instead would result in pageable memory that blocks
        async transfers due to required intermediate buffering.
        """
        container = cls()
        container.set_memory_type(memory_type)
        return container

    @classmethod
    def device_factory(cls) -> "OutputArrayContainer":
        """
        Create a new device memory container.

        Returns
        -------
        OutputArrayContainer
            A new container configured for device memory.
        """
        container = cls()
        container.set_memory_type("device")
        return container


@define
class OutputArrays(BaseArrayManager):
    """
    Manage batch integration output arrays between host and device.

    This class manages the allocation, transfer, and synchronization of output
    arrays generated during batch integration operations. It handles state
    trajectories, observables, summary statistics, and per-run status codes.

    Parameters
    ----------
    _sizes
        Size specifications for the output arrays.
    host
        Container for host-side arrays.
    device
        Container for device-side arrays.

    Notes
    -----
    This class is initialized with a BatchOutputSizes instance (which is drawn
    from a solver instance using the from_solver factory method), which sets
    the allowable 3D array sizes from the ODE system's data and run settings.
    Once initialized, the object can be updated with a solver instance to
    update the expected sizes, check the cache, and allocate if required.
    """

    _sizes: BatchOutputSizes = field(
        factory=BatchOutputSizes,
        validator=attrsval_instance_of(BatchOutputSizes),
    )
    host: OutputArrayContainer = field(
        factory=OutputArrayContainer.host_factory,
        validator=attrsval_instance_of(OutputArrayContainer),
        init=True,
    )
    device: OutputArrayContainer = field(
        factory=OutputArrayContainer.device_factory,
        validator=attrsval_instance_of(OutputArrayContainer),
        init=False,
    )
    _buffer_pool: ChunkBufferPool = field(factory=ChunkBufferPool, init=False)
    _watcher: WritebackWatcher = field(factory=WritebackWatcher, init=False)
    _pending_buffers: List[PendingBuffer] = field(factory=list, init=False)

    def __attrs_post_init__(self) -> None:
        """
        Configure default memory types after initialization.

        Returns
        -------
        None
            This method updates the host and device container metadata.

        Notes
        -----
        Host containers use pinned memory to enable asynchronous
        device-to-host transfers with CUDA streams.
        """
        super().__attrs_post_init__()
        self.host.set_memory_type("pinned")
        self.device.set_memory_type("device")

    def _on_allocation_complete(self, response: ArrayResponse) -> None:
        """
        Callback for when the allocation response is received.

        Parameters
        ----------
        response
            Response object containing allocated arrays and metadata.

        Returns
        -------
        None
            Nothing is returned.

        Notes
        -----
        After setting chunk count from parent implementation, converts
        pinned host arrays to regular numpy when chunking is active.
        Chunked arrays use per-chunk pinned buffers instead.
        """
        super()._on_allocation_complete(response)
        if self.is_chunked:
            self._convert_host_to_numpy()
        else:
            self.host.set_memory_type("pinned")

    def _convert_host_to_pinned(self) -> None:
        """Convert regular numpy host arrays to pinned for non-chunked mode.

        When a run is not chunked, the host arrays should be pinned to
        enable asynchronous transfers.
        """
        for name, slot in self.host.iter_managed_arrays():
            old_array = slot.array
            if old_array is not None:
                if slot.memory_type == "host":
                    new_array = self._memory_manager.create_host_array(
                        old_array.shape, old_array.dtype, "pinned"
                    )
                    slot.array = new_array
        self.host.set_memory_type("pinned")

    def _convert_host_to_numpy(self) -> None:
        """Convert pinned host arrays to regular numpy for chunked mode.

        When chunking is active, host arrays should be regular numpy
        to limit pinned memory usage. Per-chunk pinned buffers are
        used for staging during transfers.
        """
        for name, slot in self.host.iter_managed_arrays():
            # Convert to regular numpy only for arrays with chunked transfers
            if slot.memory_type == "pinned" and slot.needs_chunked_transfer:
                old_array = slot.array
                if old_array is not None:
                    new_array = self._memory_manager.create_host_array(
                        old_array.shape, old_array.dtype, "host"
                    )
                    slot.array = new_array
                    slot.memory_type = "host"

    def update(self, solver_instance: "BatchSolverKernel") -> None:
        """
        Update output arrays from solver instance.

        Parameters
        ----------
        solver_instance
            The solver instance providing configuration and sizing information.

        Returns
        -------
        None
            This method updates cached arrays in place.
        """
        new_arrays = self.update_from_solver(solver_instance)
        self.update_host_arrays(new_arrays, shape_only=True)
        self.allocate()

    @property
    def state(self) -> ArrayTypes:
        """Host state output array."""
        return self.host.state.array

    @property
    def observables(self) -> ArrayTypes:
        """Host observables output array."""
        return self.host.observables.array

    @property
    def state_summaries(self) -> ArrayTypes:
        """Host state summary output array."""
        return self.host.state_summaries.array

    @property
    def observable_summaries(self) -> ArrayTypes:
        """Host observable summary output array."""
        return self.host.observable_summaries.array

    @property
    def device_state(self) -> ArrayTypes:
        """Device state output array."""
        return self.device.state.array

    @property
    def device_observables(self) -> ArrayTypes:
        """Device observables output array."""
        return self.device.observables.array

    @property
    def device_state_summaries(self) -> ArrayTypes:
        """Device state summary output array."""
        return self.device.state_summaries.array

    @property
    def device_observable_summaries(self) -> ArrayTypes:
        """Device observable summary output array."""
        return self.device.observable_summaries.array

    @property
    def status_codes(self) -> ArrayTypes:
        """Host status code output array."""
        return self.host.status_codes.array

    @property
    def device_status_codes(self) -> ArrayTypes:
        """Device status code output array."""
        return self.device.status_codes.array

    @property
    def iteration_counters(self) -> ArrayTypes:
        """Host iteration counters output array."""
        return self.host.iteration_counters.array

    @property
    def device_iteration_counters(self) -> ArrayTypes:
        """Device iteration counters output array."""
        return self.device.iteration_counters.array

    @classmethod
    def from_solver(
        cls, solver_instance: "BatchSolverKernel"
    ) -> "OutputArrays":
        """
        Create an OutputArrays instance from a solver.

        Does not allocate arrays, just sets up size specifications.

        Parameters
        ----------
        solver_instance
            The solver instance to extract configuration from.

        Returns
        -------
        OutputArrays
            A new OutputArrays instance configured for the solver.
        """
        sizes = BatchOutputSizes.from_solver(solver_instance).nonzero
        return cls(
            sizes=sizes,
            precision=solver_instance.precision,
            memory_manager=solver_instance.memory_manager,
            stream_group=solver_instance.stream_group,
        )

    def update_from_solver(
        self, solver_instance: "BatchSolverKernel"
    ) -> Dict[str, NDArray[np_floating]]:
        """
        Update sizes and precision from solver, returning new host arrays.

        Only creates new pinned arrays when existing arrays do not match
        the expected shape and dtype. This avoids expensive pinned memory
        allocation on repeated solver runs with identical configurations.

        Parameters
        ----------
        solver_instance
            The solver instance to update from.

        Returns
        -------
        dict[str, numpy.ndarray]
            Host arrays with updated shapes for ``update_host_arrays``.
            Arrays that already match are still included for consistency.
        """
        self._sizes = BatchOutputSizes.from_solver(solver_instance).nonzero
        self._precision = solver_instance.precision
        new_arrays = {}
        for name, slot in self.host.iter_managed_arrays():
            newshape = getattr(self._sizes, name)
            dtype = slot.dtype
            if np_issubdtype(dtype, np_floating):
                slot.dtype = self._precision
                dtype = slot.dtype
            # Fast path: skip allocation if existing array matches
            current = slot.array
            if (
                current is not None
                and current.shape == newshape
                and current.dtype == dtype
            ):
                new_arrays[name] = current
            else:
                new_arrays[name] = self._memory_manager.create_host_array(
                    newshape, dtype, slot.memory_type
                )
        for name, slot in self.device.iter_managed_arrays():
            dtype = slot.dtype
            if np_issubdtype(dtype, np_floating):
                slot.dtype = self._precision
        return new_arrays

    def finalise(self, chunk_index: int) -> None:
        """Queue device-to-host transfers for a chunk.

        Parameters
        ----------
        chunk_index
            Indices for the chunk being finalized.

        Returns
        -------
        None
            Queues async transfers. For chunked mode, submits writeback
            tasks to the watcher thread for non-blocking completion.

        Notes
        -----
        Host slices are made contiguous before transfer to ensure
        compatible strides with device arrays. For chunked mode, data
        is transferred to pooled pinned buffers and submitted to the
        watcher thread for async writeback. For non-chunked mode,
        the writeback call is made immediately (but will happen
        asynchronously).
        """
        from_ = []
        to_ = []
        stream = self._memory_manager.get_stream(self)

        for array_name, slot in self.host.iter_managed_arrays():
            device_array = self.device.get_array(array_name)
            device_slot = self.device.get_managed_array(array_name)
            host_array = slot.array
            stride_order = slot.stride_order

            to_target = host_array
            from_target = device_array
            if slot.needs_chunked_transfer:
                slice_tuple = slot.chunked_slice_fn(chunk_index)
                host_slice = host_array[slice_tuple]
                # Chunked mode: use buffer pool and watcher
                # Buffer must match device array shape for D2H copy
                buffer = self._buffer_pool.acquire(
                    array_name, device_array.shape, host_slice.dtype
                )
                # Set pinned buffer as target and register for writeback
                to_target = buffer.array
                self._pending_buffers.append(
                    PendingBuffer(
                        buffer=buffer,
                        target_array=host_array,
                        slice_tuple=slice_tuple,
                        array_name=array_name,
                        data_shape=host_slice.shape,
                        buffer_pool=self._buffer_pool,
                    )
                )

            to_.append(to_target)
            from_.append(from_target)

        self.from_device(from_, to_)

        # Record events and submit to watcher for chunked mode
        if self._pending_buffers:
            if not CUDA_SIMULATION:
                event = cuda.event()
                event.record(stream)
            else:
                event = None

            for buffer in self._pending_buffers:
                self._watcher.submit_from_pending_buffer(
                    event=event,
                    pending_buffer=buffer,
                )
            self._pending_buffers.clear()

    def wait_pending(self, timeout: Optional[float] = None) -> None:
        """Wait for all pending async writebacks to complete.

        Parameters
        ----------
        timeout
            Maximum seconds to wait. None waits indefinitely.

        Returns
        -------
        None
            Blocks until all pending operations complete.

        Notes
        -----
        Only applies to chunked mode with watcher-based writebacks.
        """
        self._watcher.wait_all(timeout=timeout)

    def initialise(self, chunk_index: int) -> None:
        """
        Initialize device arrays before kernel execution.

        Parameters
        ----------
        chunk_index
            Indices for the chunk being initialized.

        Returns
        -------
        None
            This method performs no operations by default.

        Notes
        -----
        No initialization to zeros is needed unless chunk calculations in time
        leave a dangling sample at the end, which is possible but not expected.
        """
        pass

    def reset(self) -> None:
        """Clear all cached arrays and reset allocation tracking.

        Extends the base reset to also clear the buffer pool, shut down
        the watcher thread, and clear any pending buffers.

        Returns
        -------
        None
            Nothing is returned.
        """
        super().reset()
        self._buffer_pool.clear()
        self._watcher.shutdown()
        self._pending_buffers.clear()
