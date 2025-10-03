"""Manage host and device input arrays for batch integrations."""

import attrs
import attrs.validators as val

from numpy.typing import NDArray
from typing import Optional, TYPE_CHECKING, Union

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
    """Container for batch input arrays used by solver kernels.

    Parameters
    ----------
    initial_values
        Initial state values for the integration.
    parameters
        Parameter values for the integration.
    driver_coefficients
        Interpolant coefficients describing external drivers.
    stride_order
        Mapping of array labels to their stride orders.
    _memory_type
        Type of memory allocation.
    _unchunkable
        Array names that cannot be chunked.

    Notes
    -----
    This container keeps attrs-managed metadata describing which arrays are
    chunkable and how dimensions map to batching axes so the array manager can
    transfer slices correctly.
    """

    initial_values: ArrayTypes = attrs.field(default=None)
    parameters: ArrayTypes = attrs.field(default=None)
    driver_coefficients: ArrayTypes = attrs.field(default=None)
    stride_order: dict[str, tuple[str, ...]] = attrs.field(
        factory=lambda: {
            "initial_values": ("run", "variable"),
            "parameters": ("run", "variable"),
            "driver_coefficients": ("time", "run", "variable"),
        },
        init=False,
    )
    _memory_type: str = attrs.field(
        default="device",
        validator=val.in_(["device", "mapped", "pinned", "managed", "host"]),
    )
    _unchunkable = attrs.field(default=("driver_coefficients",), init=False)

    @classmethod
    def host_factory(cls) -> "InputArrayContainer":
        """Create a container configured for host memory transfers.

        Returns
        -------
        InputArrayContainer
            Host-side container instance.
        """
        return cls(memory_type="host")

    @classmethod
    def device_factory(cls) -> "InputArrayContainer":
        """Create a container configured for device memory transfers.

        Returns
        -------
        InputArrayContainer
            Device-side container instance.
        """
        return cls(memory_type="device")


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
        self.host._memory_type = "host"
        self.device._memory_type = "device"

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
        return self.host.initial_values

    @property
    def parameters(self) -> ArrayTypes:
        """Host parameters array."""
        return self.host.parameters

    @property
    def driver_coefficients(self) -> ArrayTypes:
        """Host driver coefficients array."""

        return self.host.driver_coefficients

    @property
    def device_initial_values(self) -> ArrayTypes:
        """Device initial values array."""
        return self.device.initial_values

    @property
    def device_parameters(self) -> ArrayTypes:
        """Device parameters array."""
        return self.device.parameters

    @property
    def device_driver_coefficients(self) -> ArrayTypes:
        """Device driver coefficients array."""

        return self.device.driver_coefficients

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
        stride_order = self.host.stride_order["initial_values"]
        slice_tuple = [slice(None)] * len(stride_order)
        if self._chunk_axis in stride_order:
            chunk_index = stride_order.index(self._chunk_axis)
            slice_tuple[chunk_index] = host_indices
            slice_tuple = tuple(slice_tuple)

        to_ = [self.host.initial_values[slice_tuple]]
        from_ = [self.device.initial_values]

        self.from_device(self, from_, to_)

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
        This method copies the appropriate chunk of data from host to device
        arrays before kernel execution.
        """
        from_ = []
        to_ = []

        if self._chunks <= 1:
            arrays_to_copy = [array for array in self._needs_overwrite]
            self._needs_overwrite = []
        else:
            arrays_to_copy = [
                array
                for array in self.device.__dict__
                if not array.startswith("_")
            ]

        for array_name in arrays_to_copy:
            if not array_name.startswith("_"):
                to_.append(getattr(self.device, array_name))
                host_array = getattr(self.host, array_name)
                if self._chunks <= 1 or array_name in self.host._unchunkable:
                    from_.append(host_array)
                else:
                    stride_order = self.host.stride_order[array_name]
                    chunk_index = stride_order.index(self._chunk_axis)
                    slice_tuple = [slice(None)] * len(stride_order)
                    slice_tuple[chunk_index] = host_indices
                    from_.append(host_array[tuple(slice_tuple)])

        self.to_device(from_, to_)
