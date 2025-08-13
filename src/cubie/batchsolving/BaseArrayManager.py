from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, TYPE_CHECKING
from warnings import warn

import attrs
import attrs.validators as val
import numpy as np
from numpy import float32, ndarray
from numpy.typing import NDArray
from cubie import default_memmgr
from cubie.memory.mem_manager import MemoryManager
from cubie.memory.mem_manager import ArrayRequest, ArrayResponse


@attrs.define(slots=False)
class ArrayContainer(ABC):
    """Base class for storing arrays - any CUDA array manager should have
    one subclass of each for host, device arrays."""

    def delete_all(self):
        for attr_name in list(self.__dict__.keys()):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                setattr(self, attr_name, None)

    def attach(self, label, array):
        """Attach a host array to a device array"""
        if hasattr(self, label):
            setattr(self, label, array)
        else:
            warn(
                f"Device array with label '{label}' does not exist. ignoring",
                UserWarning,
            )

    def delete(self, label):
        """Drop reference to a host array"""
        if hasattr(self, label):
            setattr(self, label, None)
        else:
            warn(f"Host array with label '{label}' does not exist.", UserWarning)


@attrs.define
class BaseArrayManager(ABC):
    """Common base class for managing arrays between host and device.
    This class provides:
    - Common interface for MemoryManager integration
    - Unified allocation/deallocation patterns
    - Stream management
    - Change detection and caching
    - Queued allocation support
    """

    _precision: type = attrs.field(
        default=float32,
        validator=val.instance_of(type))
    device: ArrayContainer = attrs.field(
        factory=ArrayContainer,
        validator=val.instance_of(ArrayContainer))
    host: ArrayContainer = attrs.field(
        factory=ArrayContainer,
        validator=val.instance_of(ArrayContainer))
    _stream_group: str = attrs.field(
        default="default",
        validator=val.instance_of(str))
    _memory_proportion: Optional[float] = attrs.field(
        default=None,
        validator=val.optional(val.instance_of(float))    )
    _needs_reallocation: list[str] = attrs.field(
        factory=list,
        init=False)
    _needs_overwrite: list[str] = attrs.field(
        factory=list,
        init=False)
    _memory_manager: MemoryManager = attrs.field(default=default_memmgr)

    def __attrs_post_init__(self):
        self.register_with_memory_manager()

    def _on_allocation_complete(self, response: ArrayResponse):
        for array_label in self._needs_reallocation:
            self.device.attach(array_label, response.arr[array_label])
        self._needs_reallocation.clear()
        self._needs_overwrite.clear()

    def register_with_memory_manager(self):
        """Register this instance with the MemoryManager"""
        self._memory_manager.register(
                self,
                proportion=self._memory_proportion,
                invalidate_cache_hook=self._invalidate_hook,
                allocation_ready_hook=self._on_allocation_complete,
                stream_group=self._stream_group,
                )

    def request_allocation(self,
                           request: dict[str, ArrayRequest],
                           force_type: Optional[str] = None):
        """Send a request for allocation of device arrays.

        If the object is the only instance in it's stream group, or is on
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
        if request_type == ("single"):
            self._memory_manager.single_request(self, request)
        else:
            self._memory_manager.queue_request(self, request)

    def _invalidate_hook(self):
        """Drop all references and assign all for reallocation"""
        self._needs_reallocation.clear()
        self._needs_overwrite.clear()
        self.device.delete_all()
        self._needs_reallocation.extend([
            array for array in self.device.__dict__.keys()])

    def _arrays_equal(self,
                      arr1: Optional[NDArray],
                      arr2: Optional[NDArray]) -> bool:
        """Check if two arrays are equal in shape and content."""
        if arr1 is None or arr2 is None:
            return arr1 is arr2
        return np.array_equal(arr1, arr2)

    def _update_host_array(self,
                           new_array: NDArray,
                           current_array: NDArray,
                           label: str) -> NDArray:
        """Assign for reallocation or overwriting by shape/value change.

        Check for equality and shape equality, append to reallocation or
        overwrite lists accordingly. Returns the new array if changed,
        otherwise returns current_array unchanged."""

        if not self._arrays_equal(new_array, current_array):
            if current_array.shape != new_array.shape:
                self._needs_reallocation.append(label)
                if 0 in new_array.shape:
                    return np.zeros((1, 1, 1), dtype=self._precision)
            else:
                self._needs_overwrite.append(label)
            return new_array
        return current_array

    @abstractmethod
    def finalise_chunk(self, indices, axis):
        """Override with the desired behaviour after a chunk is executed.

        For most output arrays, this is a copy back to the host,
        and potentially a remap if mapped.
        For input arrays, this is a typically no-op."""

    @abstractmethod
    def initialise_chunk(self, indices, axis):
        """Override with the desired behaviour before a chunk is executed.

        For most input arrays, this is a copy to device.
        For output arrays, this is typically a no-op."""

