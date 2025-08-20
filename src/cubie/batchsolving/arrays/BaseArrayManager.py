"""
Base Array Manager Module.

This module provides abstract base classes for managing arrays between host and
device memory in batch operations. It includes container classes for storing
arrays and manager classes for handling memory allocation, transfer, and
synchronization.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
from warnings import warn

import attrs
import attrs.validators as val
import numpy as np
from numba.core.types import NoneType
from numpy import float32, ceil
from numpy.typing import NDArray
from cubie.memory import default_memmgr
from cubie.memory.mem_manager import MemoryManager
from cubie.memory.mem_manager import ArrayRequest, ArrayResponse
from cubie.outputhandling import ArraySizingClass


@attrs.define(slots=False)
class ArrayContainer(ABC):
    """
    Base class for storing arrays in CUDA array managers.

    Any CUDA array manager should have one subclass of this for both host
    and device arrays.

    Parameters
    ----------
    _stride_order : tuple[str, ...], optional
        Order of array dimensions (e.g., ("time", "run", "variable")).
    _memory_type : str, default="device"
        Type of memory allocation. Must be one of "device", "mapped",
        "pinned", "managed", or "host".
    _unchunkable : tuple[str], default=()
        Names of arrays that cannot be chunked during memory management.

    Notes
    -----
    Underscored attributes are not really private, but we want to be able
    to filter them in methods which use __dict__ to get attributes,
    so we prefix them and add getters/setters.
    """

    _stride_order: Optional[tuple[str, ...]] = attrs.field(
        default=None, validator=val.optional(val.instance_of(tuple))
    )
    _memory_type: str = attrs.field(
        default="device",
        validator=val.in_(["device", "mapped", "pinned", "managed", "host"]),
    )
    _unchunkable: tuple[str] = attrs.field(
        factory=tuple, validator=val.instance_of(tuple)
    )

    @property
    def stride_order(self):
        """
        Get the stride order.

        Returns
        -------
        tuple[str, ...] or None
            The order of array dimensions.
        """
        return self._stride_order

    @stride_order.setter
    def stride_order(self, value):
        """
        Set the stride order.

        Parameters
        ----------
        value : tuple[str, ...]
            The order of array dimensions.
        """
        self._stride_order = value

    @property
    def memory_type(self):
        """
        Get the memory type.

        Returns
        -------
        str
            The type of memory allocation.
        """
        return self._memory_type

    @memory_type.setter
    def memory_type(self, value):
        """
        Set the memory type.

        Parameters
        ----------
        value : str
            The type of memory allocation.
        """
        self._memory_type = value

    def delete_all(self):
        """
        Delete all array references.

        Notes
        -----
        This method removes all non-private, non-callable attributes,
        effectively clearing all stored arrays.
        """
        for attr_name in list(self.__dict__.keys()):
            if not attr_name.startswith("_") and not callable(
                getattr(self, attr_name)
            ):
                setattr(self, attr_name, None)

    def attach(self, label, array):
        """
        Attach an array to this container.

        Parameters
        ----------
        label : str
            The name/label for the array.
        array : array_like
            The array to attach.

        Warns
        -----
        UserWarning
            If the specified label does not exist as an attribute.
        """
        if hasattr(self, label):
            setattr(self, label, array)
        else:
            warn(
                f"Device array with label '{label}' does not exist. ignoring",
                UserWarning,
            )

    def delete(self, label):
        """
        Delete reference to an array.

        Parameters
        ----------
        label : str
            The name/label of the array to delete.

        Warns
        -----
        UserWarning
            If the specified label does not exist as an attribute.
        """
        if hasattr(self, label):
            setattr(self, label, None)
        else:
            warn(
                f"Host array with label '{label}' does not exist.", UserWarning
            )


@attrs.define
class BaseArrayManager(ABC):
    """
    Common base class for managing arrays between host and device.

    This class provides a unified interface for MemoryManager integration,
    allocation/deallocation patterns, stream management, change detection
    and caching, and queued allocation support.

    Parameters
    ----------
    _precision : type, default=float32
        Numerical precision type for arrays.
    _sizes : ArraySizingClass, optional
        Size specifications for arrays managed by this instance.
    device : ArrayContainer
        Container for device-side arrays.
    host : ArrayContainer
        Container for host-side arrays.
    _chunks : int, default=0
        Number of chunks for memory management.
    _chunk_axis : str, default="run"
        Axis along which to perform chunking. Must be one of "run",
        "variable", or "time".
    _stream_group : str, default="default"
        Stream group identifier for CUDA operations.
    _memory_proportion : float, optional
        Proportion of available memory to use.
    _needs_reallocation : list[str]
        List of array names that need reallocation.
    _needs_overwrite : list[str]
        List of array names that need data overwriting.
    _memory_manager : MemoryManager
        Memory manager instance for handling GPU memory.

    Notes
    -----
    This is an abstract base class that provides common functionality for
    managing arrays between host and device memory. Subclasses must implement
    the abstract methods: update, finalise, and initialise.
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

    def __attrs_post_init__(self):
        """
        Initialize the array manager after attrs initialization.

        Notes
        -----
        This method registers with the memory manager, initializes default
        host arrays, and sets up invalidation hooks.
        """
        self.register_with_memory_manager()
        for name, arr in self.host.__dict__.items():
            if not name.startswith("_") and arr is None:
                shape = (1,) * len(self.device.stride_order)
                setattr(
                    self.host, name, np.zeros(shape, dtype=self._precision)
                )
        self._invalidate_hook()

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update arrays from external data.

        This method should handle updating the manager's arrays based on
        provided input data and trigger reallocation/allocation as needed.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses
        with the desired behavior for updating arrays from external data.
        """

    def _on_allocation_complete(self, response: ArrayResponse):
        """
        Callback for when the allocation response is received.

        Parameters
        ----------
        response : ArrayResponse
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

    def register_with_memory_manager(self):
        """
        Register this instance with the MemoryManager.

        Notes
        -----
        This method sets up the necessary hooks and callbacks for memory
        management integration.
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
    ):
        """
        Send a request for allocation of device arrays.

        Parameters
        ----------
        request : dict[str, ArrayRequest]
            Dictionary mapping array names to allocation requests.
        force_type : str, optional
            Force request type to "single" or "group". If None, the type
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

    def _invalidate_hook(self):
        """
        Drop all references and assign all arrays for reallocation.

        Notes
        -----
        This method is called when the memory cache needs to be invalidated.
        It clears all device array references and marks them for reallocation.
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
        arr1 : NDArray or None
            First array to compare.
        arr2 : NDArray or None
            Second array to compare.

        Returns
        -------
        bool
            True if arrays are equal, False otherwise.
        """
        if arr1 is None or arr2 is None:
            return arr1 is arr2
        return np.array_equal(arr1, arr2)

    def update_sizes(self, sizes: ArraySizingClass):
        """
        Update the expected sizes for arrays in this manager.

        Parameters
        ----------
        sizes : ArraySizingClass
            An ArraySizingClass instance with new sizes.

        Raises
        ------
        TypeError
            If the new sizes object is not the same size as the existing one.
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
        arrays : Dict[str, NDArray]
            Dictionary of array_name -> array.

        Returns
        -------
        Dict[str, bool]
            Dictionary mapping array names to boolean values indicating
            whether each array matches the expected precision.
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
        """Check if provided arrays match the system along the "variable" axis.

        Args:
            new_arrays: Dictionary of array_name -> array
            location: 'host' or 'device'

        Returns:
            True if all arrays match their expected sizes, False otherwise
        """
        try:
            container = getattr(self, location)
        except AttributeError:
            raise AttributeError(
                f"Invalid location: {location} - must be 'host' or 'device'"
            )
        expected_sizes = self._sizes
        source_stride_order = getattr(expected_sizes, "_stride_order", None)
        target_stride_order = container._stride_order
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

                # Chunk if needed and arrays are device arrays
                if location == "device" and self._chunks > 0:
                    if chunk_axis_name in container._stride_order:
                        chunk_axis_index = container._stride_order.index(
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
    def finalise(self, indices):
        """Override with the desired behaviour after a chunk is executed.

        For most output arrays, this is a copy back to the host,
        and potentially a remap if mapped.
        For input arrays, this is a typically no-op."""

    @abstractmethod
    def initialise(self, indices):
        """Override with the desired behaviour before a chunk is executed.

        For most input arrays, this is a copy to device.
        For output arrays, this is typically a no-op."""

    def check_incoming_arrays(
        self, arrays: Dict[str, NDArray], location: str = "host"
    ) -> Dict[str, bool]:
        """Check dimensions and dtype of provided arrays match expected sizes and precision.

        Args:
            arrays: Dictionary of array_name -> array to check
            location: "host" or "device" - which container to check against
        Returns:
            True if all arrays match expected sizes and precision, False otherwise
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
        """Attach existing arrays to the specified container (host or device).

        Args:
            arrays: Dictionary of array_name -> array to attach
            location: "host" or "device" - which container to attach to

        Returns:
            True if arrays were successfully attached, False if validation failed
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
                f"The following arrays did not match the expected data "
                f"type and size, and so were not used"
                f" {', '.join(not_attached)}",
                UserWarning,
            )
        return True

    def _update_host_array(
        self, new_array: NDArray, current_array: NDArray, label: str
    ) -> NoneType:
        """Assign for reallocation or overwriting by shape/value change.

        Check for equality and shape equality, append to reallocation or
        overwrite lists accordingly. Attaches changed array to host array
        container."""
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

    def update_host_arrays(self, new_arrays: Dict[str, NDArray]):
        """Updates host arrays with new data, assigns for realloc or overwrite.

        Args:
            new_arrays: Dictionary of array_name -> new_array

        Returns:
            None
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

    def allocate(self):
        """Queue allocation requests for arrays that need reallocation."""
        requests = {}

        for array_label in set(self._needs_reallocation):
            host_array = getattr(self.host, array_label, None)
            if host_array is not None:
                request = ArrayRequest(
                    shape=host_array.shape,
                    dtype=self._precision,
                    memory=self.device.memory_type,
                    stride_order=self.device.stride_order,
                )
                requests[array_label] = request

        if requests:
            self.request_allocation(requests)

    def initialize_device_zeros(self):
        """Initialize device arrays to zero values."""
        for name, array in self.device.__dict__.items():
            if not name.startswith("_") and array is not None:
                if len(array.shape) >= 3:
                    array[:, :, :] = self._precision(0.0)
                elif len(array.shape) >= 2:
                    array[:, :] = self._precision(0.0)

    def reset(self):
        """Clear all cached arrays and reset allocation tracking."""
        self.host.delete_all()
        self.device.delete_all()
        self._needs_reallocation.clear()
        self._needs_overwrite.clear()

    def to_device(self, from_arrays: list, to_arrays: list):
        self._memory_manager.to_device(self, from_arrays, to_arrays)

    def from_device(
        self, instance: object, from_arrays: list, to_arrays: list
    ):
        self._memory_manager.from_device(self, from_arrays, to_arrays)
