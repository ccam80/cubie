"""
Batch Input Arrays Module.

This module provides classes for managing input arrays in batch integration
operations, including containers for storing arrays and managers for handling
memory allocation and data transfer between host and device.
"""

from os import environ

import attrs
import attrs.validators as val

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    pass
else:
    pass

from numpy.typing import NDArray
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

from cubie.outputhandling.output_sizes import BatchInputSizes
from cubie.batchsolving.arrays.BaseArrayManager import (
    BaseArrayManager,
    ArrayContainer,
)
from cubie.batchsolving import ArrayTypes


@attrs.define(slots=False)
class InputArrayContainer(ArrayContainer):
    """
    Container for batch input arrays.

    This container holds the input arrays needed for batch integration,
    including initial values, parameters, and forcing vectors.

    Parameters
    ----------
    initial_values : ArrayTypes, optional
        Initial state values for the integration.
    parameters : ArrayTypes, optional
        Parameter values for the integration.
    forcing_vectors : ArrayTypes, optional
        Forcing function vectors for the integration.
    stride_order : tuple[str, ...], default=("run", "variable")
        Order of array dimensions.
    _memory_type : str, default="device"
        Type of memory allocation.
    _unchunkable : tuple[str, ...], default=('forcing_vectors',)
        Array names that cannot be chunked.

    Notes
    -----
    This class uses attrs for automatic initialization and validation.
    The _unchunkable attribute specifies which arrays should not be
    divided into chunks during memory management.
    """

    initial_values: ArrayTypes = attrs.field(default=None)
    parameters: ArrayTypes = attrs.field(default=None)
    forcing_vectors: ArrayTypes = attrs.field(default=None)
    stride_order: tuple[str, ...] = attrs.field(
        default=("run", "variable"), init=False
    )
    _memory_type: str = attrs.field(
        default="device",
        validator=val.in_(["device", "mapped", "pinned", "managed", "host"]),
    )
    _unchunkable = attrs.field(default=("forcing_vectors",), init=False)

    @classmethod
    def host_factory(cls):
        """
        Create a new host memory container.

        Returns
        -------
        InputArrayContainer
            A new container configured for host memory.
        """
        return cls(memory_type="host")

    @classmethod
    def device_factory(cls):
        """
        Create a new device memory container.

        Returns
        -------
        InputArrayContainer
            A new container configured for device memory.
        """
        return cls(memory_type="device")


@attrs.define
class InputArrays(BaseArrayManager):
    """
    Manage batch integration input arrays between host and device.

    This class manages the allocation, transfer, and synchronization of input
    arrays needed for batch integration operations. It handles initial values,
    parameters, and forcing vectors.

    Parameters
    ----------
    _sizes : BatchInputSizes, optional
        Size specifications for the input arrays.
    host : InputArrayContainer
        Container for host-side arrays.
    device : InputArrayContainer
        Container for device-side arrays.

    Notes
    -----
    This class is initialized with a BatchInputSizes instance (which is drawn
    from a solver instance using the from_solver factory method), which sets
    the allowable array heights from the ODE system's data. Once initialized,
    the object can be updated with arguments (initial_values, parameters,
    forcing_vectors). Each call to the update method will:

    - Check if the input array has changed in shape or content since the last update
    - Queue allocation requests with the MemoryManager
    - Attach allocated arrays once received from MemoryManager
    """

    _sizes: Optional[BatchInputSizes] = attrs.field(
        factory=BatchInputSizes,
        validator=val.optional(val.instance_of(BatchInputSizes)),
    )
    host: InputArrayContainer = attrs.field(
        factory=InputArrayContainer.host_factory,
        validator=val.instance_of(InputArrayContainer),
        init=True,
    )
    device: InputArrayContainer = attrs.field(
        factory=InputArrayContainer.device_factory,
        validator=val.instance_of(InputArrayContainer),
        init=False,
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.host._memory_type = "host"
        self.device._memory_type = "device"

    def update(
        self,
        solver_instance,
        initial_values: NDArray,
        parameters: NDArray,
        forcing_vectors: NDArray,
    ) -> None:
        """
        Set the initial values, parameters, and forcing vectors.

        Queues allocation requests with the MemoryManager for batch processing.

        Parameters
        ----------
        solver_instance : BatchSolverKernel
            The solver instance providing configuration and sizing information.
        initial_values : NDArray
            Initial state values for each integration run.
        parameters : NDArray
            Parameter values for each integration run.
        forcing_vectors : NDArray
            Forcing vectors for each integration run.
        """
        updates_dict = {
            "initial_values": initial_values,
            "parameters": parameters,
            "forcing_vectors": forcing_vectors,
        }
        self.update_from_solver(solver_instance)
        self.update_host_arrays(updates_dict)
        self.allocate()  # Will queue request if in a stream group

    @property
    def initial_values(self):
        """
        Get host initial values array.

        Returns
        -------
        ArrayTypes
            The initial values array from the host container.
        """
        return self.host.initial_values

    @property
    def parameters(self):
        """
        Get host parameters array.

        Returns
        -------
        ArrayTypes
            The parameters array from the host container.
        """
        return self.host.parameters

    @property
    def forcing_vectors(self):
        """
        Get host forcing vectors array.

        Returns
        -------
        ArrayTypes
            The forcing vectors array from the host container.
        """
        return self.host.forcing_vectors

    @property
    def device_initial_values(self):
        """
        Get device initial values array.

        Returns
        -------
        ArrayTypes
            The initial values array from the device container.
        """
        return self.device.initial_values

    @property
    def device_parameters(self):
        """
        Get device parameters array.

        Returns
        -------
        ArrayTypes
            The parameters array from the device container.
        """
        return self.device.parameters

    @property
    def device_forcing_vectors(self):
        """
        Get device forcing vectors array.

        Returns
        -------
        ArrayTypes
            The forcing vectors array from the device container.
        """
        return self.device.forcing_vectors

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
        solver_instance : BatchSolverKernel
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

    def update_from_solver(self, solver_instance: "BatchSolverKernel"):
        """
        Update size and precision from solver instance.

        Parameters
        ----------
        solver_instance : BatchSolverKernel
            The solver instance to update from.
        """
        self._sizes = BatchInputSizes.from_solver(solver_instance).nonzero
        self._precision = solver_instance.precision
        self._chunk_axis = solver_instance.chunk_axis

    def finalise(self, host_indices):
        """
        Copy out final states if they're required.

        Parameters
        ----------
        host_indices : slice or array-like
            Indices for the chunk being finalized.

        Notes
        -----
        This method copies data from device back to host for the specified
        chunk indices.
        """
        chunk_index = self.host.stride_order.index(self._chunk_axis)
        slice_tuple = [slice(None)] * 2
        slice_tuple[chunk_index] = host_indices
        slice_tuple = tuple(slice_tuple)

        to_ = [self.host.initial_values[slice_tuple]]
        from_ = [self.device.initial_values]

        self.from_device(self, from_, to_)

    def initialise(self, host_indices):
        """
        Copy chunk of data to device.

        Parameters
        ----------
        host_indices : slice or array-like
            Indices for the chunk being initialized.

        Notes
        -----
        This method copies the appropriate chunk of data from host to device
        arrays before kernel execution.
        """
        from_ = []
        to_ = []

        if self._chunks <= 1:
            arrays_to_copy = [array for array in self._needs_overwrite]
            self._needs_overwrite = []
            slice_tuple = tuple([slice(None)] * len(self.host.stride_order))
        else:
            arrays_to_copy = [
                array
                for array in self.device.__dict__
                if not array.startswith("_")
            ]
            chunk_index = self.host.stride_order.index(self._chunk_axis)
            slice_tuple = [slice(None)] * len(self.host.stride_order)
            slice_tuple[chunk_index] = host_indices
            slice_tuple = tuple(slice_tuple)

        for array_name in arrays_to_copy:
            if not array_name.startswith("_"):
                to_.append(getattr(self.device, array_name))
                if array_name in self.host._unchunkable:
                    from_.append(getattr(self.host, array_name))
                else:
                    from_.append(getattr(self.host, array_name)[slice_tuple])

        self.to_device(from_, to_)
