"""Base utilities for managing batch arrays on host and device.

Notes
-----
Defines :class:`ArrayContainer` and :class:`BaseArrayManager`, which surface
stride metadata, register with :mod:`cubie.memory`, and orchestrate queued CUDA
allocations for batch solver workflows.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from warnings import warn

import attrs
import attrs.validators as val
import numpy as np
from numpy import float32
from numpy.typing import NDArray

from cubie.memory import default_memmgr
from cubie.memory.mem_manager import MemoryManager
from cubie.memory.mem_manager import ArrayRequest, ArrayResponse
from cubie.outputhandling.output_sizes import ArraySizingClass


@attrs.define(slots=False)
class ArrayContainer(ABC):
    """Store stride metadata and array references for a CUDA manager.

    Parameters
    ----------
    _stride_order
        Mapping of array labels to stride orders such as
        {"state": ("time", "run", "variable")}.
    _memory_type
        Memory allocation type. Must be one of "device", "mapped",
        "pinned", "managed", or "host".
    _unchunkable
        Names of arrays that cannot be chunked during memory management.

    Notes
    -----
    Underscored attributes are filtered when scanning ``__dict__`` so that
    helper methods can focus on user-facing arrays.
    """

    _stride_order: dict[str, tuple[str, ...]] = attrs.field(
        factory=dict,
        validator=val.deep_mapping(
            val.instance_of(str),
            val.deep_iterable(
                member_validator=val.instance_of(str),
                iterable_validator=val.instance_of(tuple),
            ),
        ),
    )
    _memory_type: str = attrs.field(
        default="device",
        validator=val.in_(["device", "mapped", "pinned", "managed", "host"]),
    )
    _unchunkable: tuple[str] = attrs.field(
        factory=tuple, validator=val.instance_of(tuple)
    )

    @property
    def stride_order(self) -> dict[str, tuple[str, ...]]:
        """Stride order mapping for managed arrays."""
        return self._stride_order

    @stride_order.setter
    def stride_order(self, value: dict[str, tuple[str, ...]]) -> None:
        """
        Set the stride order.

        Parameters
        ----------
        value
            Mapping of array labels to stride orders.
        """
        self._stride_order = value

    @property
    def memory_type(self) -> str:
        """Memory allocation type for attached arrays."""
        return self._memory_type

    @memory_type.setter
    def memory_type(self, value: str) -> None:
        """
        Set the memory type.

        Parameters
        ----------
        value
            The type of memory allocation.
        """
        self._memory_type = value

    def delete_all(self) -> None:
        """
        Delete all array references.

        Notes
        -----
        This method removes all non-private, non-callable attributes,
        effectively clearing all stored arrays.

        Returns
        -------
        None
            Nothing is returned.
        """
        for attr_name in list(self.__dict__.keys()):
            if not attr_name.startswith("_") and not callable(
                getattr(self, attr_name)
            ):
                setattr(self, attr_name, None)

    def attach(self, label: str, array: NDArray) -> None:
        """
        Attach an array to this container.

        Parameters
        ----------
        label
            The name or label for the array.
        array
            The array to attach.

        Warns
        -----
        UserWarning
            If the specified label does not exist as an attribute.

        Returns
        -------
        None
            Nothing is returned.
        """
        if hasattr(self, label):
            setattr(self, label, array)
        else:
            warn(
                f"Device array with label '{label}' does not exist. ignoring",
                UserWarning,
            )

    def delete(self, label: str) -> None:
        """
        Delete reference to an array.

        Parameters
        ----------
        label
            The name or label of the array to delete.

        Warns
        -----
        UserWarning
            If the specified label does not exist as an attribute.

        Returns
        -------
        None
            Nothing is returned.
        """
        if hasattr(self, label):
            setattr(self, label, None)
        else:
            warn(
                f"Host array with label '{label}' does not exist.", UserWarning
            )


@attrs.define
class BaseArrayManager(ABC):
    """Coordinate allocation and transfer for batch host and device arrays.

    Parameters
    ----------
    _precision
        Precision factory used to create new arrays.
    _sizes
        Size specifications for arrays managed by this instance.
    device
        Container for device-side arrays.
    host
        Container for host-side arrays.
    _chunks
        Number of chunks for memory management.
    _chunk_axis
        Axis along which to perform chunking. Must be one of "run",
        "variable", or "time".
    _stream_group
        Stream group identifier for CUDA operations.
    _memory_proportion
        Proportion of available memory to use.
    _needs_reallocation
        Array names that require device reallocation.
    _needs_overwrite
        Array names that require host overwrite.
    _memory_manager
        Memory manager instance for handling GPU memory.

    Notes
    -----
    Subclasses must implement :meth:`update`, :meth:`finalise`, and
    :meth:`initialise` to wire batching behaviour into host and device
    execution paths.
    """

    _precision: type = attrs.field(
        default=float32, validator=val.instance_of(type)
    )
    _sizes: Optional[ArraySizingClass] = attrs.field(
        default=None, validator=val.optional(val.instance_of(ArraySizingClass))
    )
    device: ArrayContainer = attrs.field(
        factory=ArrayContainer, validator=val.instance_of(ArrayContainer)
    )
    host: ArrayContainer = attrs.field(
        factory=ArrayContainer, validator=val.instance_of(ArrayContainer)
    )
    _chunks: int = attrs.field(default=0, validator=val.instance_of(int))
    _chunk_axis: str = attrs.field(
        default="run", validator=val.in_(["run", "variable", "time"])
    )
    _stream_group: str = attrs.field(
        default="default", validator=val.instance_of(str)
    )
    _memory_proportion: Optional[float] = attrs.field(
        default=None, validator=val.optional(val.instance_of(float))
    )
    _needs_reallocation: list[str] = attrs.field(factory=list, init=False)
    _needs_overwrite: list[str] = attrs.field(factory=list, init=False)
    _memory_manager: MemoryManager = attrs.field(default=default_memmgr)

    def __attrs_post_init__(self) -> None:
        """
        Initialize the array manager after attrs initialization.

        Notes
        -----
        This method registers with the memory manager, initializes default
        host arrays, and sets up invalidation hooks.

        Returns
        -------
        None
            Nothing is returned.
        """
        self.register_with_memory_manager()
        stride_orders = self.device.stride_order
        for name, arr in self.host.__dict__.items():
            if not name.startswith("_") and arr is None:
                shape = (1,) * len(stride_orders[name])
                setattr(self.host, name, np.zeros(shape, dtype=self._precision))
        self._invalidate_hook()

    @abstractmethod
    def update(self, *args: object, **kwargs: object) -> None:
        """
        Update arrays from external data.

        This method should handle updating the manager's arrays based on
        provided input data and trigger reallocation/allocation as needed.

        Parameters
        ----------
        *args
            Positional arguments passed by subclasses.
        **kwargs
            Keyword arguments passed by subclasses.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses
        with the desired behavior for updating arrays from external data.

        Returns
        -------
        None
            Nothing is returned.
        """

    def _on_allocation_complete(self, response: ArrayResponse) -> None:
        """
        Callback for when the allocation response is received.

        Parameters
        ----------
        response
            Response object containing allocated arrays and metadata.

        Warns
        -----
        UserWarning
            If a device array is not found in the allocation response.

        Notes
        -----
        WARNING - HERE BE DRAGONS
        This try/except is to catch case where tests were calling this method
        with an empty _needs_reallocation list. When the same tests were
        run one at a time, the error disappeared. I couldn't trace it to a
        module-scope fixture or anything obvious. Adding the try/except
        seems to have suppressed even the warning, and the problem has
        stopped.
        If you get this warning, check for the possibility of two
        different classes calling allocate_queue in between "init" and
        "initialise".

        Returns
        -------
        None
            Nothing is returned.
        """

        for array_label in self._needs_reallocation:
            try:
                self.device.attach(array_label, response.arr[array_label])
            except KeyError:
                warn(
                    f"Device array {array_label} not found in allocation "
                    f"response. See "
                    f"BaseArrayManager._on_allocation_complete docstring "
                    f"for more info.",
                    UserWarning,
                )
        self._chunks = response.chunks
        self._chunk_axis = response.chunk_axis
        self._needs_reallocation.clear()

    def register_with_memory_manager(self) -> None:
        """
        Register this instance with the MemoryManager.

        Notes
        -----
        This method sets up the necessary hooks and callbacks for memory
        management integration.

        Returns
        -------
        None
            Nothing is returned.
        """
        self._memory_manager.register(
            self,
            proportion=self._memory_proportion,
            invalidate_cache_hook=self._invalidate_hook,
            allocation_ready_hook=self._on_allocation_complete,
            stream_group=self._stream_group,
        )

    def request_allocation(
        self,
        request: dict[str, ArrayRequest],
        force_type: Optional[str] = None,
    ) -> None:
        """
        Send a request for allocation of device arrays.

        Parameters
        ----------
        request
            Dictionary mapping array names to allocation requests.
        force_type
            Force request type to "single" or "group". If ``None``, the type
            is determined automatically based on stream group membership.

        Notes
        -----
        If the object is the only instance in its stream group, or is on
        the default group, then the request will be sent as a "single"
        request and be allocated immediately. If the object shares a stream
        group, then the response will be queued, and the allocation will be
        grouped with other requests in the same group, until one of the
        instances calls "process_queue" to process the queue. This behaviour
        can be overridden by setting force_type to "single" or "group".

        Returns
        -------
        None
            Nothing is returned.
        """
        request_type = force_type
        if request_type is None:
            if self._memory_manager.is_grouped(self):
                request_type = "group"
            else:
                request_type = "single"
        if request_type == "single":
            self._memory_manager.single_request(self, request)
        else:
            self._memory_manager.queue_request(self, request)

    def _invalidate_hook(self) -> None:
        """
        Drop all references and assign all arrays for reallocation.

        Notes
        -----
        This method is called when the memory cache needs to be invalidated.
        It clears all device array references and marks them for reallocation.

        Returns
        -------
        None
            Nothing is returned.
        """
        self._needs_reallocation.clear()
        self._needs_overwrite.clear()
        self.device.delete_all()
        self._needs_reallocation.extend(
            [
                array
                for array in self.device.__dict__.keys()
                if not array.startswith("_")
            ]
        )

    def _arrays_equal(
        self, arr1: Optional[NDArray], arr2: Optional[NDArray]
    ) -> bool:
        """
        Check if two arrays are equal in shape and content.

        Parameters
        ----------
        arr1
            First array or ``None``.
        arr2
            Second array or ``None``.

        Returns
        -------
        bool
            ``True`` if arrays are equal, ``False`` otherwise.
        """
        if arr1 is None or arr2 is None:
            return arr1 is arr2
        return np.array_equal(arr1, arr2)

    def update_sizes(self, sizes: ArraySizingClass) -> None:
        """
        Update the expected sizes for arrays in this manager.

        Parameters
        ----------
        sizes
            Array sizing configuration with new dimensions.

        Raises
        ------
        TypeError
            If the new sizes object is not the same size as the existing one.

        Returns
        -------
        None
            Nothing is returned.
        """
        if not isinstance(sizes, type(self._sizes)):
            raise TypeError(
                "Expected the new sizes object to be the "
                f"same size as the previous one "
                f"({type(self._sizes)}), got {type(sizes)}"
            )
        self._sizes = sizes

    def check_type(self, arrays: Dict[str, NDArray]) -> Dict[str, bool]:
        """
        Check if the precision of arrays matches the system precision.

        Parameters
        ----------
        arrays
            Dictionary mapping array names to arrays.

        Returns
        -------
        Dict[str, bool]
            Dictionary indicating whether each array matches the expected
            precision.
        """
        matches = {}
        for array_name, array in arrays.items():
            if array is not None and array.dtype != self._precision:
                matches[array_name] = False
            else:
                matches[array_name] = True
        return matches

    def check_sizes(
        self, new_arrays: Dict[str, NDArray], location: str = "host"
    ) -> Dict[str, bool]:
        """
        Check whether arrays match configured sizes and stride order.

        Parameters
        ----------
        new_arrays
            Dictionary mapping array names to arrays.
        location
            ``"host"`` or ``"device"`` indicating which container to inspect.

        Returns
        -------
        Dict[str, bool]
            Dictionary indicating whether each array matches its expected
            shape.

        Raises
        ------
        AttributeError
            If the location is neither ``"host"`` nor ``"device"``.
        """
        try:
            container = getattr(self, location)
        except AttributeError:
            raise AttributeError(
                f"Invalid location: {location} - must be 'host' or 'device'"
            )
        expected_sizes = self._sizes
        source_stride_order = getattr(expected_sizes, "_stride_order", None)
        target_stride_orders = container._stride_order
        chunk_axis_name = self._chunk_axis
        matches = {}

        for array_name, array in new_arrays.items():
            if array_name not in container.__dict__.keys():
                matches[array_name] = False
                continue
            else:
                array_shape = array.shape
                expected_size_tuple = getattr(expected_sizes, array_name)
                if expected_size_tuple is None:
                    continue  # No size information for this array
                expected_shape = list(expected_size_tuple)

                target_stride_order = target_stride_orders[array_name]

                # Reorder expected_shape to match the container's stride order
                if (
                    source_stride_order
                    and target_stride_order
                    and source_stride_order != target_stride_order
                ):
                    size_map = {
                        axis: size
                        for axis, size in zip(
                            source_stride_order, expected_shape
                        )
                    }
                    expected_shape = [
                        size_map[axis]
                        for axis in target_stride_order
                        if axis in size_map
                    ]

                # Chunk if needed and arrays are device arrays, unless unchunkable
                if (
                    location == "device"
                    and self._chunks > 0
                    and array_name not in container._unchunkable
                ):
                    if chunk_axis_name in target_stride_order:
                        chunk_axis_index = target_stride_order.index(
                            chunk_axis_name
                        )
                        if expected_shape[chunk_axis_index] is not None:
                            expected_shape[chunk_axis_index] = int(
                                np.ceil(
                                    expected_shape[chunk_axis_index]
                                    / self._chunks
                                )
                            )

                if len(array_shape) != len(expected_shape):
                    matches[array_name] = False
                else:
                    shape_matches = True
                    for actual_dim, expected_dim in zip(
                        array_shape, expected_shape
                    ):
                        if (
                            expected_dim is not None
                            and actual_dim != expected_dim
                        ):
                            shape_matches = False
                            break
                    matches[array_name] = shape_matches
        return matches

    @abstractmethod
    def finalise(self, indices: List[int]) -> None:
        """
        Execute post-chunk behaviour for device outputs.

        Parameters
        ----------
        indices
            Chunk indices processed by the device execution path.

        Returns
        -------
        None
            Nothing is returned.
        """

    @abstractmethod
    def initialise(self, indices: List[int]) -> None:
        """
        Execute pre-chunk behaviour for device inputs.

        Parameters
        ----------
        indices
            Chunk indices about to run on the device.

        Returns
        -------
        None
            Nothing is returned.
        """

    def check_incoming_arrays(
        self, arrays: Dict[str, NDArray], location: str = "host"
    ) -> Dict[str, bool]:
        """
        Validate shape and precision for incoming arrays.

        Parameters
        ----------
        arrays
            Dictionary mapping array names to arrays.
        location
            ``"host"`` or ``"device"`` indicating the target container.

        Returns
        -------
        Dict[str, bool]
            Dictionary indicating whether each array is ready for attachment.
        """
        dims_ok = self.check_sizes(arrays, location=location)
        types_ok = self.check_type(arrays)
        all_ok = {}
        for array_name in arrays:
            all_ok[array_name] = dims_ok[array_name] and types_ok[array_name]
        return all_ok

    def attach_external_arrays(
        self, arrays: Dict[str, NDArray], location: str = "host"
    ) -> bool:
        """
        Attach existing arrays to a host or device container.

        Parameters
        ----------
        arrays
            Dictionary mapping array names to arrays.
        location
            ``"host"`` or ``"device"`` indicating the target container.

        Returns
        -------
        bool
            ``True`` if arrays pass validation, ``False`` otherwise.
        """
        matches = self.check_incoming_arrays(arrays, location=location)
        container = getattr(self, location)
        not_attached = []
        for array_name, array in arrays.items():
            if matches[array_name]:
                container.attach(array_name, array)
            else:
                not_attached.append(array_name)
        if not_attached:
            warn(
                f"The following arrays did not match the expected precision "
                f"and size, and so were not used"
                f" {', '.join(not_attached)}",
                UserWarning,
            )
        return True

    def _update_host_array(
        self, new_array: NDArray, current_array: Optional[NDArray], label: str
    ) -> None:
        """
        Mark host arrays for overwrite or reallocation based on updates.

        Parameters
        ----------
        new_array
            Updated array that should replace the stored host array.
        current_array
            Previously stored host array or ``None``.
        label
            Array name used to index tracking lists.

        Raises
        ------
        ValueError
            If ``new_array`` is ``None``.

        Returns
        -------
        None
            Nothing is returned.
        """
        if new_array is None:
            raise ValueError("New array is None")
        elif current_array is None:
            self._needs_reallocation.append(label)
            self._needs_overwrite.append(label)
            self.host.attach(label, new_array)
        elif not self._arrays_equal(new_array, current_array):
            if current_array.shape != new_array.shape:
                if label not in self._needs_reallocation:
                    self._needs_reallocation.append(label)
                if label not in self._needs_overwrite:
                    self._needs_overwrite.append(label)
                if 0 in new_array.shape:
                    new_array = np.zeros((1, 1, 1), dtype=self._precision)
            else:
                self._needs_overwrite.append(label)
            self.host.attach(label, new_array)
        return None

    def update_host_arrays(self, new_arrays: Dict[str, NDArray]) -> None:
        """
        Update host arrays and record allocation requirements.

        Parameters
        ----------
        new_arrays
            Dictionary mapping array names to new host arrays.

        Returns
        -------
        None
            Nothing is returned.
        """
        badnames = [
            array_name
            for array_name in new_arrays
            if array_name not in self.host.__dict__.keys()
        ]
        new_arrays = {
            k: v
            for k, v in new_arrays.items()
            if k in self.host.__dict__.keys()
        }
        if any(badnames):
            warn(
                f"Host arrays '{badnames}' does not exist, ignoring update",
                UserWarning,
            )
        if not any([check for check in self.check_sizes(new_arrays).values()]):
            warn(
                "Provided arrays do not match the expected system "
                "sizes, ignoring update",
                UserWarning,
            )
        for array_name in new_arrays:
            current_array = getattr(self.host, array_name)
            self._update_host_array(
                new_arrays[array_name], current_array, array_name
            )

    def allocate(self) -> None:
        """
        Queue allocation requests for arrays that need reallocation.

        Notes
        -----
        Builds :class:`ArrayRequest` objects for arrays marked for
        reallocation and sets the ``unchunkable`` hint based on host metadata.

        Returns
        -------
        None
            Nothing is returned.
        """
        requests = {}
        for array_label in list(set(self._needs_reallocation)):
            host_array = getattr(self.host, array_label, None)
            if host_array is None:
                continue
            request = ArrayRequest(
                shape=host_array.shape,
                dtype=self._precision,
                memory=self.device.memory_type,
                stride_order=self.device.stride_order[array_label],
                unchunkable=(
                    array_label in getattr(self.host, "_unchunkable", tuple())
                ),
            )
            requests[array_label] = request
        if requests:
            self.request_allocation(requests)

    def initialize_device_zeros(self) -> None:
        """
        Initialize device arrays to zero values.

        Returns
        -------
        None
            Nothing is returned.
        """
        for name, array in self.device.__dict__.items():
            if not name.startswith("_") and array is not None:
                if len(array.shape) >= 3:
                    array[:, :, :] = self._precision(0.0)
                elif len(array.shape) >= 2:
                    array[:, :] = self._precision(0.0)

    def reset(self) -> None:
        """
        Clear all cached arrays and reset allocation tracking.

        Returns
        -------
        None
            Nothing is returned.
        """
        self.host.delete_all()
        self.device.delete_all()
        self._needs_reallocation.clear()
        self._needs_overwrite.clear()

    def to_device(self, from_arrays: List[str], to_arrays: List[str]) -> None:
        """
        Copy host arrays to the device using the memory manager.

        Parameters
        ----------
        from_arrays
            Names of host arrays to copy.
        to_arrays
            Names of destination device arrays.

        Returns
        -------
        None
            Nothing is returned.
        """
        self._memory_manager.to_device(self, from_arrays, to_arrays)

    def from_device(
        self, instance: object, from_arrays: List[str], to_arrays: List[str]
    ) -> None:
        """
        Copy device arrays back to the host using the memory manager.

        Parameters
        ----------
        instance
            Object requesting the transfer.
        from_arrays
            Names of device arrays to copy.
        to_arrays
            Names of destination host arrays.

        Returns
        -------
        None
            Nothing is returned.
        """
        self._memory_manager.from_device(self, from_arrays, to_arrays)
