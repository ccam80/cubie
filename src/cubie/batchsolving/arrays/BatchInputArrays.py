"""Manage host and device input arrays for batch integrations."""

from attrs import define, field
from attrs.validators import (
    instance_of as attrsval_instance_of,
    optional as attrsval_optional,
)
from numpy import (
    dtype as np_dtype,
    float32 as np_float32,
    floating as np_floating,
    issubdtype as np_issubdtype,
)

from numpy.typing import NDArray
from typing import List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

from cubie.memory.chunk_buffer_pool import ChunkBufferPool, PinnedBuffer
from cubie.outputhandling.output_sizes import BatchInputSizes
from cubie.batchsolving.arrays.BaseArrayManager import (
    ArrayContainer,
    BaseArrayManager,
    ManagedArray,
)
from cubie.batchsolving import ArrayTypes


@define(slots=False)
class InputArrayContainer(ArrayContainer):
    """Container for batch input arrays used by solver kernels."""

    initial_values: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("variable", "run"),
            shape=(1, 1),
        )
    )
    parameters: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("variable", "run"),
            shape=(1, 1),
        )
    )
    driver_coefficients: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("time", "variable", "run"),
            shape=(1, 1, 1),
            is_chunked=False,
        )
    )

    @classmethod
    def host_factory(
        cls, memory_type: str = "pinned"
    ) -> "InputArrayContainer":
        """Create a container configured for host memory transfers.

        Parameters
        ----------
        memory_type
            Memory type for host arrays: "pinned" or "host".
            Default is "pinned" for non-chunked operation.

        Returns
        -------
        InputArrayContainer
            Host-side container instance.

        Notes
        -----
        Uses pinned (page-locked) memory to enable asynchronous
        host-to-device transfers with CUDA streams. Using ``"host"``
        memory type instead would result in pageable memory that blocks
        async transfers due to required intermediate buffering.
        """
        container = cls()
        container.set_memory_type(memory_type)
        return container

    @classmethod
    def device_factory(cls) -> "InputArrayContainer":
        """Create a container configured for device memory transfers.

        Returns
        -------
        InputArrayContainer
            Device-side container instance.
        """
        container = cls()
        container.set_memory_type("device")
        return container


@define
class InputArrays(BaseArrayManager):
    """Manage allocation and transfer of batch input arrays.

    Parameters
    ----------
    _sizes
        Size specifications for the input arrays.
    host
        Container for host-side arrays.
    device
        Container for device-side arrays.

    Notes
    -----
    Instances are configured from :class:`~cubie.batchsolving.BatchSolverKernel`
    metadata. Updates request memory through the shared manager, ensure array
    heights match solver expectations, and attach received buffers prior to
    device transfers.
    """

    _sizes: Optional[BatchInputSizes] = field(
        factory=BatchInputSizes,
        validator=attrsval_optional(attrsval_instance_of(BatchInputSizes)),
    )
    host: InputArrayContainer = field(
        factory=InputArrayContainer.host_factory,
        validator=attrsval_instance_of(InputArrayContainer),
        init=True,
    )
    device: InputArrayContainer = field(
        factory=InputArrayContainer.device_factory,
        validator=attrsval_instance_of(InputArrayContainer),
        init=False,
    )
    _buffer_pool: ChunkBufferPool = field(factory=ChunkBufferPool, init=False)
    _active_buffers: List[PinnedBuffer] = field(factory=list, init=False)

    def __attrs_post_init__(self) -> None:
        """Ensure host and device containers use explicit memory types.

        Returns
        -------
        None
            This method mutates container configuration in place.

        Notes
        -----
        Host containers use pinned memory to enable asynchronous
        host-to-device transfers with CUDA streams.
        """
        super().__attrs_post_init__()
        self.host.set_memory_type("pinned")
        self.device.set_memory_type("device")

    def update(
        self,
        solver_instance: "BatchSolverKernel",
        initial_values: NDArray,
        parameters: NDArray,
        driver_coefficients: Optional[NDArray],
    ) -> None:
        """Set host arrays and request device allocations.

        Parameters
        ----------
        solver_instance
            The solver instance providing configuration and sizing information.
        initial_values
            Initial state values for each integration run.
        parameters
            Parameter values for each integration run.
        driver_coefficients
            Horner-ordered driver interpolation coefficients.

        Returns
        -------
        None
            This method updates internal references and enqueues allocations.
        """
        updates_dict = {
            "initial_values": initial_values,
            "parameters": parameters,
        }
        if driver_coefficients is not None:
            updates_dict["driver_coefficients"] = driver_coefficients
        self.update_from_solver(solver_instance)
        self.update_host_arrays(updates_dict)
        self.allocate()  # Will queue request if in a stream group

    @property
    def initial_values(self) -> ArrayTypes:
        """Host initial values array."""
        return self.host.initial_values.array

    @property
    def parameters(self) -> ArrayTypes:
        """Host parameters array."""
        return self.host.parameters.array

    @property
    def driver_coefficients(self) -> ArrayTypes:
        """Host driver coefficients array."""

        return self.host.driver_coefficients.array

    @property
    def device_initial_values(self) -> ArrayTypes:
        """Device initial values array."""
        return self.device.initial_values.array

    @property
    def device_parameters(self) -> ArrayTypes:
        """Device parameters array."""
        return self.device.parameters.array

    @property
    def device_driver_coefficients(self) -> ArrayTypes:
        """Device driver coefficients array."""

        return self.device.driver_coefficients.array

    @classmethod
    def from_solver(
        cls, solver_instance: "BatchSolverKernel"
    ) -> "InputArrays":
        """
        Create an InputArrays instance from a solver.

        Creates an empty instance from a solver instance, importing the heights
        of the parameters, initial values, and driver arrays from the ODE system
        for checking inputs against. Does not allocate host or device arrays.

        Parameters
        ----------
        solver_instance
            The solver instance to extract configuration from.

        Returns
        -------
        InputArrays
            A new InputArrays instance configured for the solver.
        """
        sizes = BatchInputSizes.from_solver(solver_instance)
        return cls(
            sizes=sizes,
            precision=solver_instance.precision,
            memory_manager=solver_instance.memory_manager,
            stream_group=solver_instance.stream_group,
        )

    def update_from_solver(self, solver_instance: "BatchSolverKernel") -> None:
        """Refresh size, precision, and chunk axis from the solver.

        Parameters
        ----------
        solver_instance
            The solver instance to update from.

        Returns
        -------
        None
            This method mutates cached solver metadata in place.
        """
        self._sizes = BatchInputSizes.from_solver(solver_instance).nonzero
        self._precision = solver_instance.precision
        for name, arr_obj in self.host.iter_managed_arrays():
            arr_obj.shape = getattr(self._sizes, name)
            if np_issubdtype(np_dtype(arr_obj.dtype), np_floating):
                arr_obj.dtype = self._precision
        for name, arr_obj in self.device.iter_managed_arrays():
            arr_obj.shape = getattr(self._sizes, name)
            if np_issubdtype(np_dtype(arr_obj.dtype), np_floating):
                arr_obj.dtype = self._precision

    def finalise(self, host_indices: Union[slice, NDArray]) -> None:
        """Release buffers back to host."""
        self.release_buffers()

    def initialise(self, host_indices: Union[slice, NDArray]) -> None:
        """Copy a batch chunk of host data to device buffers.

        Parameters
        ----------
        host_indices
            Indices for the chunk being initialized.

        Returns
        -------
        None
            Host slices are staged into device arrays in place.

        Notes
        -----
        For chunked mode, pinned buffers are acquired from the pool for
        staging data before H2D transfer. Buffers are stored in
        _active_buffers and released after the H2D transfer completes.
        For non-chunked mode, pinned buffers are allocated directly.
        """
        from_ = []
        to_ = []

        if self._chunks <= 1:
            arrays_to_copy = [array for array in self._needs_overwrite]
            self._needs_overwrite = []
        else:
            arrays_to_copy = list(self.device.array_names())

        for array_name in arrays_to_copy:
            device_obj = self.device.get_managed_array(array_name)
            to_.append(device_obj.array)
            host_obj = self.host.get_managed_array(array_name)

            # Use needs_chunked_transfer for simple branching
            if not device_obj.needs_chunked_transfer:
                from_.append(host_obj.array)
            else:
                stride_order = host_obj.stride_order
                if self._chunk_axis not in stride_order:
                    from_.append(host_obj.array)
                    continue
                chunk_index = stride_order.index(self._chunk_axis)
                slice_tuple = [slice(None)] * len(stride_order)
                slice_tuple[chunk_index] = host_indices
                host_slice = host_obj.array[tuple(slice_tuple)]

                # Chunked mode: use buffer pool for pinned staging
                # Buffer must match device array shape for H2D copy
                device_shape = device_obj.array.shape
                buffer = self._buffer_pool.acquire(
                    array_name, device_shape, host_slice.dtype
                )
                # Copy host slice into smallest indices of buffer,
                # as the final host slice may be smaller than the buffer.
                data_slice = tuple(slice(0, s) for s in host_slice.shape)
                buffer.array[data_slice] = host_slice
                from_.append(buffer.array)
                # Record that we're using this buffer for later release.
                self._active_buffers.append(buffer)

        self.to_device(from_, to_)

    def release_buffers(self) -> None:
        """Release all active buffers back to the pool.

        Should be called after H2D transfer completes to return pooled
        pinned buffers for reuse by subsequent chunks.
        """
        for buffer in self._active_buffers:
            self._buffer_pool.release(buffer)
        self._active_buffers.clear()

    def reset(self) -> None:
        """Clear all cached arrays and reset allocation tracking."""
        super().reset()
        self._buffer_pool.clear()
        self._active_buffers.clear()
