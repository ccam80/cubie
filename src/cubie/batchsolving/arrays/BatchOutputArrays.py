"""Manage output array lifecycles for batch solver executions."""

from typing import TYPE_CHECKING, Dict, Union

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

import attrs
import attrs.validators as val
import numpy as np
from numpy.typing import NDArray

from cubie.outputhandling.output_sizes import BatchOutputSizes
from cubie.batchsolving.arrays.BaseArrayManager import (
    ArrayContainer,
    BaseArrayManager,
    ManagedArray,
)
from cubie.batchsolving import ArrayTypes
from cubie._utils import slice_variable_dimension

ChunkIndices = Union[slice, NDArray[np.integer]]


@attrs.define(slots=False)
class OutputArrayContainer(ArrayContainer):
    """Container for batch output arrays."""

    state: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            stride_order=("time", "variable", "run"),
            shape=(1, 1, 1),
        )
    )
    observables: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            stride_order=("time", "variable", "run"),
            shape=(1, 1, 1),
        )
    )
    state_summaries: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            stride_order=("time", "variable", "run"),
            shape=(1, 1, 1),
        )
    )
    observable_summaries: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            stride_order=("time", "variable", "run"),
            shape=(1, 1, 1),
        )
    )
    status_codes: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.int32,
            stride_order=("run",),
            shape=(1,),
            is_chunked=False,
        )
    )
    iteration_counters: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.int32,
            stride_order=("time", "variable", "run"),
            shape=(1, 4, 1),
        )
    )

    @classmethod
    def host_factory(cls) -> "OutputArrayContainer":
        """
        Create a new pinned host memory container.

        Returns
        -------
        OutputArrayContainer
            A new container configured for pinned host memory.

        Notes
        -----
        Uses pinned (page-locked) memory to enable asynchronous
        device-to-host transfers with CUDA streams. Using ``"host"``
        memory type instead would result in pageable memory that blocks
        async transfers due to required intermediate buffering.
        """
        container = cls()
        container.set_memory_type("pinned")
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


@attrs.define
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

    _sizes: BatchOutputSizes = attrs.field(
        factory=BatchOutputSizes, validator=val.instance_of(BatchOutputSizes)
    )
    host: OutputArrayContainer = attrs.field(
        factory=OutputArrayContainer.host_factory,
        validator=val.instance_of(OutputArrayContainer),
        init=True,
    )
    device: OutputArrayContainer = attrs.field(
        factory=OutputArrayContainer.device_factory,
        validator=val.instance_of(OutputArrayContainer),
        init=False,
    )

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
    ) -> Dict[str, NDArray[np.floating]]:
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
            slot.shape = newshape
            dtype = slot.dtype
            if np.issubdtype(dtype, np.floating):
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
                    newshape, dtype, slot.stride_order, slot.memory_type
                )
        for name, slot in self.device.iter_managed_arrays():
            slot.shape = getattr(self._sizes, name)
            dtype = slot.dtype
            if np.issubdtype(dtype, np.floating):
                slot.dtype = self._precision
        return new_arrays

    def finalise(self, host_indices: ChunkIndices) -> None:
        """
        Copy device arrays to host array slices.

        Parameters
        ----------
        host_indices
            Indices for the chunk being finalized.

        Returns
        -------
        None
            This method mutates host buffers in place.

        Notes
        -----
        This method queues async transfers from device arrays to host
        arrays using the solver's registered stream.
        """
        from_ = []
        to_ = []

        for array_name, slot in self.host.iter_managed_arrays():
            device_array = self.device.get_array(array_name)
            host_array = slot.array
            stride_order = slot.stride_order

            if self._chunk_axis in stride_order:
                chunk_index = stride_order.index(self._chunk_axis)
                slice_tuple = slice_variable_dimension(
                    host_indices, chunk_index, len(stride_order)
                )
                to_.append(host_array[slice_tuple])
            else:
                to_.append(host_array)
            from_.append(device_array)

        self.from_device(from_, to_)

    def initialise(self, host_indices: ChunkIndices) -> None:
        """
        Initialize device arrays before kernel execution.

        Parameters
        ----------
        host_indices
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
