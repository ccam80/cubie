"""Manage output array lifecycles for batch solver executions."""

from typing import TYPE_CHECKING, Dict, Union

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

import attrs
import attrs.validators as val
import numpy as np
from numpy.typing import NDArray

ChunkIndices = Union[slice, NDArray[np.integer]]

from cubie.outputhandling.output_sizes import BatchOutputSizes
from cubie.batchsolving.arrays.BaseArrayManager import (
    ArrayContainer,
    BaseArrayManager,
    ManagedArray,
)
from cubie.batchsolving import ArrayTypes
from cubie._utils import slice_variable_dimension


@attrs.define(slots=False)
class OutputArrayContainer(ArrayContainer):
    """Container for batch output arrays."""

    state: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            stride_order=("time", "run", "variable"),
            shape=(1, 1, 1),
        )
    )
    observables: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            stride_order=("time", "run", "variable"),
            shape=(1, 1, 1),
        )
    )
    state_summaries: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            stride_order=("time", "run", "variable"),
            shape=(1, 1, 1),
        )
    )
    observable_summaries: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            stride_order=("time", "run", "variable"),
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

    @classmethod
    def host_factory(cls) -> "OutputArrayContainer":
        """
        Create a new host memory container.

        Returns
        -------
        OutputArrayContainer
            A new container configured for host memory.
        """
        container = cls()
        container.set_memory_type("host")
        return container

    @classmethod
    def device_factory(cls) -> "OutputArrayContainer":
        """
        Create a new device memory container.

        Returns
        -------
        OutputArrayContainer
            A new container configured for mapped memory.
        """
        container = cls()
        container.set_memory_type("mapped")
        return container


@attrs.define
class ActiveOutputs:
    """
    Track which output arrays are actively being used.

    This class provides boolean flags indicating which output types are
    currently active based on array sizes and allocation status.

    Parameters
    ----------
    state
        Whether state output is active.
    observables
        Whether observables output is active.
    state_summaries
        Whether state summaries output is active.
    observable_summaries
        Whether observable summaries output is active.
    status_codes
        Whether status code output is active.
    """

    state: bool = attrs.field(default=False, validator=val.instance_of(bool))
    observables: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    state_summaries: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    observable_summaries: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    status_codes: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )

    def update_from_outputarrays(self, output_arrays: "OutputArrays") -> None:
        """
        Update active outputs based on OutputArrays instance.

        Parameters
        ----------
        output_arrays
            The OutputArrays instance to check for active outputs.

        Returns
        -------
        None
            Flags are updated in place.

        Notes
        -----
        An output is considered active if the corresponding array exists
        and has more than one element (size > 1).
        """
        self.state = (
            output_arrays.host.state.array is not None
            and output_arrays.host.state.array.size > 1
        )
        self.observables = (
            output_arrays.host.observables.array is not None
            and output_arrays.host.observables.array.size > 1
        )
        self.state_summaries = (
            output_arrays.host.state_summaries.array is not None
            and output_arrays.host.state_summaries.array.size > 1
        )
        self.observable_summaries = (
            output_arrays.host.observable_summaries.array is not None
            and output_arrays.host.observable_summaries.array.size > 1
        )
        self.status_codes = (
            output_arrays.host.status_codes.array is not None
            and output_arrays.host.status_codes.array.size > 1
        )


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
    _active_outputs
        Tracker for which outputs are currently active.

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
    _active_outputs: ActiveOutputs = attrs.field(
        default=ActiveOutputs(),
        validator=val.instance_of(ActiveOutputs),
        init=False,
    )

    def __attrs_post_init__(self) -> None:
        """
        Configure default memory types after initialization.

        Returns
        -------
        None
            This method updates the host and device container metadata.
        """
        super().__attrs_post_init__()
        self.host.set_memory_type("host")
        self.device.set_memory_type("mapped")

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
        self.update_host_arrays(new_arrays)
        self.allocate()

    @property
    def active_outputs(self) -> ActiveOutputs:
        """Active output configuration derived from host allocations."""
        self._active_outputs.update_from_outputarrays(self)
        return self._active_outputs

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

        Parameters
        ----------
        solver_instance
            The solver instance to update from.

        Returns
        -------
        dict[str, numpy.ndarray]
            Host arrays with updated shapes for ``update_host_arrays``.
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
            new_arrays[name] = np.zeros(newshape, dtype=dtype)
        for name, slot in self.device.iter_managed_arrays():
            slot.shape = getattr(self._sizes, name)
            dtype = slot.dtype
            if np.issubdtype(dtype, np.floating):
                slot.dtype = self._precision
        return new_arrays

    def finalise(self, host_indices: ChunkIndices) -> None:
        """
        Copy mapped arrays to host array slices.

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
        This method copies mapped device arrays over the specified slice
        of host arrays. The copy operation may trigger CUDA runtime
        synchronization.
        """
        for array_name, slot in self.host.iter_managed_arrays():
            array = slot.array
            device_array = self.device.get_array(array_name)
            if getattr(self.active_outputs, array_name):
                stride_order = slot.stride_order
                if self._chunk_axis in stride_order:
                    chunk_index = stride_order.index(self._chunk_axis)
                    slice_tuple = slice_variable_dimension(
                            host_indices, chunk_index, len(stride_order)
                    )
                    target_slice = slice_tuple
                else:
                    target_slice = Ellipsis
                array[target_slice] = device_array.copy()
                # I'm not sure that we can stream a Mapped transfer,
                # as transfer is managed by the CUDA runtime. If we just
                # overwrite, that might jog the cuda runtime to synchronize.

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
