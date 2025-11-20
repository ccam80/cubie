"""Manage host and device input arrays for batch integrations."""

import attrs
import attrs.validators as val
import numpy as np

from numpy.typing import NDArray
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

from cubie.outputhandling.output_sizes import BatchInputSizes
from cubie.batchsolving.arrays.BaseArrayManager import (
    ArrayContainer,
    BaseArrayManager,
    ManagedArray,
)
from cubie.batchsolving import ArrayTypes


@attrs.define(slots=False)
class InputArrayContainer(ArrayContainer):
    """Container for batch input arrays used by solver kernels."""

    initial_values: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            stride_order=("run", "variable"),
            shape=(1, 1),
        )
    )
    parameters: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            stride_order=("run", "variable"),
            shape=(1, 1),
        )
    )
    driver_coefficients: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            stride_order=("time", "run", "variable"),
            shape=(1, 1, 1),
            is_chunked=False,
        )
    )

    @classmethod
    def host_factory(cls) -> "InputArrayContainer":
        """Create a container configured for host memory transfers.

        Returns
        -------
        InputArrayContainer
            Host-side container instance.
        """
        container = cls()
        container.set_memory_type("host")
        return container

    @classmethod
    def device_factory(cls) -> "InputArrayContainer":
        """Create a container configured for mapped memory transfers.

        Returns
        -------
        InputArrayContainer
            Device-side container instance.
        """
        container = cls()
        container.set_memory_type("mapped")
        return container

    # @property
    # def initial_values(self) -> ArrayTypes:
    #     """Return the stored initial value array."""
    #
    #     return self.get_array("initial_values")
    #
    # @initial_values.setter
    # def initial_values(self, value: ArrayTypes) -> None:
    #     """Set the initial value array."""
    #
    #     self.set_array("initial_values", value)
    #
    # @property
    # def parameters(self) -> ArrayTypes:
    #     """Return the stored parameter array."""
    #
    #     return self.get_array("parameters")
    #
    # @parameters.setter
    # def parameters(self, value: ArrayTypes) -> None:
    #     """Set the parameter array."""
    #
    #     self.set_array("parameters", value)
    #
    # @property
    # def driver_coefficients(self) -> ArrayTypes:
    #     """Return the stored driver coefficients."""
    #
    #     return self.get_array("driver_coefficients")
    #
    # @driver_coefficients.setter
    # def driver_coefficients(self, value: ArrayTypes) -> None:
    #     """Set the driver coefficient array."""
    #
    #     self.set_array("driver_coefficients", value)


@attrs.define
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

    def __attrs_post_init__(self) -> None:
        """Ensure host and device containers use explicit memory types.

        Returns
        -------
        None
            This method mutates container configuration in place.
        """
        super().__attrs_post_init__()
        self.host.set_memory_type("host")
        self.device.set_memory_type("mapped")

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
        self._chunk_axis = solver_instance.chunk_axis
        for name, arr_obj in self.host.iter_managed_arrays():
            arr_obj.shape = getattr(self._sizes, name)
            if np.issubdtype(np.dtype(arr_obj.dtype), np.floating):
                arr_obj.dtype = self._precision
        for name, arr_obj in self.device.iter_managed_arrays():
            arr_obj.shape = getattr(self._sizes, name)
            if np.issubdtype(np.dtype(arr_obj.dtype), np.floating):
                arr_obj.dtype = self._precision

    def finalise(self, host_indices: Union[slice, NDArray]) -> None:
        """Copy final state slices back to host arrays when requested.

        Parameters
        ----------
        host_indices
            Indices for the chunk being finalized.

        Returns
        -------
        None
            Device buffers are read into host arrays in place.

        Notes
        -----
        This method copies data from device back to host for the specified
        chunk indices.
        """
        # This functionality was added without the device-code support to make
        # it do anything, so it just wastes time. To restore it, if useful,
        # The singleintegratorrun function needs a toggle and to overwrite
        # the initial states vecotr with it's own final state on exit.
        # This is requested in #76 https://github.com/ccam80/cubie/issues/76

        # stride_order = self.host.get_managed_array("initial_values").stride_order
        # slice_tuple = [slice(None)] * len(stride_order)
        # if self._chunk_axis in stride_order:
        #     chunk_index = stride_order.index(self._chunk_axis)
        #     slice_tuple[chunk_index] = host_indices
        #     slice_tuple = tuple(slice_tuple)
        #
        # to_ = [self.host.initial_values.array[slice_tuple]]
        # from_ = [self.device.initial_values.array]
        #
        # self.from_device(from_, to_)
        pass

    def initialise(
        self, host_indices: Union[slice, NDArray], chunk_index: int = 0
    ) -> None:
        """Copy a batch chunk of host data to device buffers.

        Parameters
        ----------
        host_indices
            Indices for the chunk being initialized.
        chunk_index
            The index of the current chunk being processed.

        Returns
        -------
        None
            Host slices are staged into device arrays in place.

        Notes
        -----
        Mapped device arrays are host-accessible pinned memory. Direct
        assignment triggers implicit synchronization by the CUDA runtime,
        eliminating the need for explicit copy operations. All chunked
        arrays are copied to ensure lingering data from previous chunks
        is overwritten.
        """
        for array_name, slot in self.host.iter_managed_arrays():
            array = slot.array
            device_array = self.device.get_array(array_name)
            device_obj = self.device.get_managed_array(array_name)
            
            # Copy all arrays if no chunking, or if this is unchunkable
            # For chunked arrays, always copy to overwrite lingering data
            if self._chunks <= 1 or not device_obj.is_chunked:
                # Direct assignment for mapped memory (no explicit copy needed)
                if device_obj.memory_type == "mapped":
                    device_array[:] = array
                else:
                    self.to_device([array], [device_array])
            else:
                # Copy the appropriate chunk slice
                stride_order = slot.stride_order
                if self._chunk_axis in stride_order:
                    chunk_idx = stride_order.index(self._chunk_axis)
                    slice_tuple = [slice(None)] * len(stride_order)
                    slice_tuple[chunk_idx] = host_indices
                    slice_tuple = tuple(slice_tuple)
                    
                    # Direct assignment for mapped memory
                    if device_obj.memory_type == "mapped":
                        device_array[:] = array[slice_tuple]
                    else:
                        self.to_device([array[slice_tuple]], [device_array])
