"""GPU memory management utilities for coordinating cubie allocations."""

from typing import Any, Optional, Callable, Union, Dict
from warnings import warn
import contextlib
from copy import deepcopy

from numba import cuda
import attrs
import attrs.validators as val
from attrs import Factory
import numpy as np
from math import prod

from cubie.cuda_simsafe import (
    BaseCUDAMemoryManager,
    NumbaCUDAMemoryManager,
    current_mem_info,
    set_cuda_memory_manager,
)
from cubie.memory.cupy_emm import current_cupy_stream
from cubie.memory.stream_groups import StreamGroups
from cubie.memory.array_requests import ArrayRequest, ArrayResponse
from cubie.memory.cupy_emm import CuPyAsyncNumbaManager, CuPySyncNumbaManager


# Recognised configuration parameters for memory manager settings.
# These keys mirror the solver API so helpers can filter keyword
# arguments consistently.
ALL_MEMORY_MANAGER_PARAMETERS = {
    "memory_manager",
    "stream_group",
    "mem_proportion",
    "allocator",
}


MIN_AUTOPOOL_SIZE = 0.05


def placeholder_invalidate() -> None:
    """
    Default invalidate hook placeholder that performs no operations.

    Returns
    -------
    None
    """
    pass


def placeholder_dataready(response: ArrayResponse) -> None:
    """
    Default placeholder data ready hook that performs no operations.

    Parameters
    ----------
    response
        Array response object (unused).

    Returns
    -------
    None
    """


# These will be keys to a dict, so must be hashable: eq=False
@attrs.define(eq=False)
class InstanceMemorySettings:
    """
    Memory registry information for a registered instance.

    Parameters
    ----------
    proportion
        Proportion of total VRAM assigned to this instance.
    allocations
        Dictionary of current allocations keyed by label.
    invalidate_hook
        Function to call when CUDA memory system changes occur.
    allocation_ready_hook
        Function to call when allocations are ready.
    cap
        Maximum allocatable bytes for this instance.

    Attributes
    ----------
    proportion : float
        Proportion of total VRAM assigned to this instance.
    allocations : dict
        Dictionary of current allocations keyed by array label.
    invalidate_hook : callable
        Function to call when CUDA memory system changes.
    allocation_ready_hook : callable
        Function to call when allocations are ready.
    cap : int or None
        Maximum allocatable bytes for this instance.

    Properties
    ----------
    allocated_bytes : int
        Total number of bytes across all allocated arrays for the instance.

    Notes
    -----
    The allocations dictionary serves both as a "keepalive" reference and a way
    to calculate total allocated memory. The invalidate_hook is called when the
    allocator/memory manager changes, requiring arrays and kernels to be
    re-allocated or redefined.
    """

    proportion: float = attrs.field(
        default=1.0, validator=val.instance_of(float)
    )
    allocations: dict = attrs.field(
        default=Factory(dict), validator=val.instance_of(dict)
    )
    invalidate_hook: Callable[[], None] = attrs.field(
        default=placeholder_invalidate, validator=val.instance_of(Callable)
    )
    allocation_ready_hook: Callable[[ArrayResponse], None] = attrs.field(
        default=placeholder_dataready
    )
    cap: Optional[int] = attrs.field(
        default=None, validator=val.optional(val.instance_of(int))
    )

    def add_allocation(self, key: str, arr: Any) -> None:
        """
        Add an allocation to the instance's allocations list.

        Parameters
        ----------
        key
            Label for the allocation.
        arr
            Allocated array object.

        Notes
        -----
        This will overwrite any previous allocation with the same key, which
        should function as intended but suggests the previous batch has not
        been properly deallocated. A warning is emitted in this case.

        Returns
        -------
        None
        """

        if key in self.allocations:
            warn(
                f"Overwriting previous allocation for {key} at a "
                f"settings level - this suggests that the previous "
                f"array wasn't deallocated properly using the "
                f"memory manager."
            )
        self.allocations[key] = arr

    def free(self, key: str) -> None:
        """
        Free an allocation by key.

        Parameters
        ----------
        key
            Label of the allocation to free.

        Notes
        -----
        Emits a warning if the key is not found in allocations.

        Returns
        -------
        None
        """
        if key in self.allocations:
            newalloc = {k: v for k, v in self.allocations.items() if k != key}
        else:
            warn(
                f"Attempted to free allocation for {key}, but "
                f"it was not found in the allocations list."
            )
            newalloc = self.allocations
        self.allocations = newalloc

    def free_all(self) -> None:
        """
        Drop all references to allocated arrays.

        Returns
        -------
        None
        """
        for key in self.allocations:
            self.free(key)

    @property
    def allocated_bytes(self) -> int:
        """Total bytes allocated across tracked arrays."""
        total = 0
        for arr in self.allocations.values():
            total += arr.nbytes
        return total


@attrs.define
class MemoryManager:
    """
    Singleton interface coordinating GPU memory allocation and stream usage.

    Parameters
    ----------
    totalmem
        Total GPU memory in bytes. Determined automatically when omitted.
    registry
        Registry mapping instance identifiers to their memory settings.
    stream_groups
        Manager for organizing instances into stream groups.
    _mode
        Memory management mode, either ``"passive"`` or ``"active"``.
    _allocator
        Memory allocator class registered with Numba.
    _auto_pool
        List of instance identifiers using automatic memory allocation.
    _manual_pool
        List of instance identifiers using manual memory allocation.
    _stride_order
        Default stride ordering for three-dimensional arrays.
    _queued_allocations
        Queued allocation requests organized by stream group.

    Notes
    -----
    The manager accepts :class:`ArrayRequest` objects and returns
    :class:`ArrayResponse` instances that reference allocated arrays and
    chunking information. Active mode enforces per-instance VRAM proportions
    while passive mode mirrors standard allocation behaviour using chunking
    only when necessary.
    """

    totalmem: int = attrs.field(
        default=None, validator=val.optional(val.instance_of(int))
    )
    registry: dict[int, InstanceMemorySettings] = attrs.field(
        default=Factory(dict), validator=val.optional(val.instance_of(dict))
    )
    stream_groups: StreamGroups = attrs.field(default=Factory(StreamGroups))
    _mode: str = attrs.field(
        default="passive", validator=val.in_(["passive", "active"])
    )
    _allocator: BaseCUDAMemoryManager = attrs.field(
        default=NumbaCUDAMemoryManager,
        validator=val.optional(val.instance_of(object)),
    )
    _auto_pool: list[int] = attrs.field(
        default=Factory(list), validator=val.instance_of(list)
    )
    _manual_pool: list[int] = attrs.field(
        default=Factory(list), validator=val.instance_of(list)
    )
    _stride_order: tuple[str, str, str] = attrs.field(
        default=("time", "run", "variable"), validator=val.instance_of(tuple)
    )
    _queued_allocations: Dict[str, Dict] = attrs.field(
        default=Factory(dict), validator=val.instance_of(dict)
    )

    def __attrs_post_init__(self) -> None:
        """Initialise the manager with current GPU memory information."""
        try:
            free, total = self.get_memory_info()
        except ValueError as e:
            if e.args[0].startswith("not enough values to unpack"):
                warn(
                    "memory manager was initialised in a cuda-less "
                    "environment - memory manager will allow import but not"
                    "provide any memory (1 byte)"
                )
                total = 1
        except Exception as e:
            warn(
                f"Unexpected exception {e} encountered while instantiating "
                "memory manager"
            )
            total = 1
        self.totalmem = total
        self.registry = {}

    def register(
        self,
        instance: object,
        proportion: Optional[float] = None,
        invalidate_cache_hook: Callable = placeholder_invalidate,
        allocation_ready_hook: Callable = placeholder_dataready,
        stream_group: str = "default",
    ) -> None:
        """
        Register an instance and configure its memory allocation settings.

        Parameters
        ----------
        instance
            Instance to register for memory management.
        proportion
            Proportion of VRAM to allocate (0.0 to 1.0). When omitted, the
            instance joins the automatic allocation pool.
        invalidate_cache_hook
            Function to call when CUDA memory system changes occur.
        allocation_ready_hook
            Function to call when allocations are ready.
        stream_group
            Name of the stream group to assign the instance to.

        Raises
        ------
        ValueError
            If instance is already registered or proportion is not between 0
            and 1.

        Returns
        -------
        None
        """
        instance_id = id(instance)
        if instance_id in self.registry:
            raise ValueError("Instance already registered")

        self.stream_groups.add_instance(instance, stream_group)

        settings = InstanceMemorySettings(
            invalidate_hook=invalidate_cache_hook,
            allocation_ready_hook=allocation_ready_hook,
        )

        self.registry[instance_id] = settings

        if proportion:
            if not 0 <= proportion <= 1:
                raise ValueError("Proportion must be between 0 and 1")
            self._add_manual_proportion(instance, proportion)
        else:
            self._add_auto_proportion(instance)

    def set_allocator(self, name: str) -> None:
        """
        Set the external memory allocator in Numba.

        Parameters
        ----------
        name
            Memory allocator type. Accepted values are ``"cupy_async"`` to use
            CuPy's :class:`~cupy.cuda.memory.AsyncMemoryPool`, ``"cupy"`` to
            use :class:`~cupy.cuda.memory.MemoryPool`, and ``"default"`` for
            Numba's default manager.

        Raises
        ------
        ValueError
            If allocator name is not recognized.

        Warnings
        --------
        UserWarning
            A change to the memory manager requires the CUDA context to be
            closed and reopened. This invalidates all previously compiled
            kernels and allocated arrays, requiring a full rebuild.

        Returns
        -------
        None
        """
        if name == "cupy_async":
            # use CuPy async memory pool
            self._allocator = CuPyAsyncNumbaManager
        elif name == "cupy":
            self._allocator = CuPySyncNumbaManager
        elif name == "default":
            # use numba's default allocator
            self._allocator = NumbaCUDAMemoryManager
        else:
            raise ValueError(f"Unknown allocator: {name}")
        set_cuda_memory_manager(self._allocator)

        # Reset the context:
        # https://nvidia.github.io/numba-cuda/user/
        # external-memory.html#setting-emm-plugin
        # WARNING - this will invalidate all prior streams, arrays, and funcs!
        # CUDA_ERROR_INVALID_CONTEXT or CUDA_ERROR_CONTEXT_IS_DESTROYED
        # suggests you're using an old reference.
        cuda.close()
        self.invalidate_all()
        self.reinit_streams()

    def set_limit_mode(self, mode: str) -> None:
        """
        Set the memory allocation limiting mode.

        Parameters
        ----------
        mode
            Either ``"passive"`` or ``"active"`` memory management mode.

        Raises
        ------
        ValueError
            If mode is not "passive" or "active".

        Returns
        -------
        None
        """
        if mode not in ["passive", "active"]:
            raise ValueError(f"Unknown mode: {mode}")
        self._mode = mode

    def get_stream(self, instance: object) -> object:
        """
        Get the CUDA stream associated with an instance.

        Parameters
        ----------
        instance
            Instance to retrieve the stream for.

        Returns
        -------
        object
            CUDA stream associated with the instance.
        """
        return self.stream_groups.get_stream(instance)

    def change_stream_group(self, instance: object, new_group: str) -> None:
        """
        Move instance to another stream group.

        Parameters
        ----------
        instance
            Instance to move.
        new_group
            Name of the new stream group.

        Returns
        -------
        None
        """
        self.stream_groups.change_group(instance, new_group)

    def reinit_streams(self) -> None:
        """
        Reinitialise all streams after a CUDA context reset.

        Returns
        -------
        None
        """
        self.stream_groups.reinit_streams()

    def invalidate_all(self) -> None:
        """
        Call each invalidate hook and release all allocations.

        Returns
        -------
        None
        """
        self.free_all()
        for registered_instance in self.registry.values():
            registered_instance.invalidate_hook()

    def set_manual_proportion(
        self, instance: object, proportion: float
    ) -> None:
        """
        Set manual allocation proportion for an instance.

        If instance is currently in the auto-allocation pool, shift it to
        manual.

        Parameters
        ----------
        instance
            Instance to update proportion for.
        proportion
            New proportion between 0 and 1.

        Raises
        ------
        ValueError
            If proportion is not between 0 and 1.

        Returns
        -------
        None
        """
        instance_id = id(instance)
        if proportion < 0 or proportion > 1:
            raise ValueError("Proportion must be between 0 and 1")
        if instance_id in self._auto_pool:
            self._add_manual_proportion(instance, proportion)
        else:
            self._manual_pool.remove(instance_id)
            self._add_manual_proportion(instance, proportion)
            self.registry[instance_id].proportion = proportion

    def set_manual_limit_mode(
        self, instance: object, proportion: float
    ) -> None:
        """
        Convert an auto-limited instance to manual allocation mode.

        Parameters
        ----------
        instance
            Instance to convert to manual mode.
        proportion
            Memory proportion to assign (0.0 to 1.0).

        Raises
        ------
        ValueError
            If instance is already in manual allocation pool.

        Returns
        -------
        None
        """
        instance_id = id(instance)
        settings = self.registry[instance_id]
        if instance_id in self._manual_pool:
            raise ValueError("Instance is already in manual allocation pool")
        self._auto_pool.remove(instance_id)
        self._add_manual_proportion(instance, proportion)
        settings.proportion = proportion

    def set_auto_limit_mode(self, instance: object) -> None:
        """
        Convert a manual-limited instance to auto allocation mode.

        Parameters
        ----------
        instance
            Instance to convert to auto mode.

        Raises
        ------
        ValueError
            If instance is already in auto allocation pool.

        Returns
        -------
        None
        """
        instance_id = id(instance)
        settings = self.registry[instance_id]
        if instance_id in self._auto_pool:
            raise ValueError("Instance is already in auto allocation pool")
        self._manual_pool.remove(instance_id)
        settings.proportion = self._add_auto_proportion(instance)

    def proportion(self, instance: object) -> float:
        """
        Get the maximum proportion of VRAM allocated to an instance.

        Parameters
        ----------
        instance
            Instance to query.

        Returns
        -------
        float
            Proportion of VRAM allocated to this instance.
        """
        instance_id = id(instance)
        return self.registry[instance_id].proportion

    def cap(self, instance: object) -> Optional[int]:
        """
        Get the maximum allocatable bytes for an instance.

        Parameters
        ----------
        instance
            Instance to query.

        Returns
        -------
        int or None
            Maximum allocatable bytes for this instance.
        """
        instance_id = id(instance)
        settings = self.registry.get(instance_id)
        return settings.cap

    @property
    def manual_pool_proportion(self):
        """Total proportion of VRAM currently assigned manually."""
        manual_settings = [
            self.registry[instance_id] for instance_id in self._manual_pool
        ]
        pool_proportion = sum(
            [settings.proportion for settings in manual_settings]
        )
        return pool_proportion

    @property
    def auto_pool_proportion(self):
        """Total proportion of VRAM currently distributed automatically."""
        auto_settings = [
            self.registry[instance_id] for instance_id in self._auto_pool
        ]
        pool_proportion = sum(
            [settings.proportion for settings in auto_settings]
        )
        return pool_proportion

    def _add_manual_proportion(self, instance: object, proportion: float) -> None:
        """
        Add an instance to the manual allocation pool with the specified proportion.

        Parameters
        ----------
        instance
            Instance to add to manual allocation pool.
        proportion
            Memory proportion to assign (0.0 to 1.0).

        Raises
        ------
        ValueError
            If manual proportion would exceed total available memory or leave
            insufficient memory for auto-allocated processes.

        Warnings
        --------
        UserWarning
            If manual proportion leaves less than 5% of memory for auto allocation.

        Notes
        -----
        Updates the instance's proportion and cap, then rebalances the auto pool.
        Enforces minimum auto pool size constraints.

        Returns
        -------
        None
        """
        instance_id = id(instance)
        new_manual_pool_size = self.manual_pool_proportion + proportion
        if new_manual_pool_size > 1.0:
            raise ValueError(
                "Manual proportion would exceed total available memory"
            )
        elif new_manual_pool_size > 1.0 - MIN_AUTOPOOL_SIZE:
            if len(self._auto_pool) > 0:
                raise ValueError(
                    "Manual proportion would leave less than 5% "
                    "of memory for auto-allocated processes. If "
                    "this is desired, adjust MIN_AUTOPOOL_SIZE in "
                    "mem_manager.py."
                )
            else:
                warn(
                    "Manual proportion leaves less than 5% of memory for "
                    "auto allocation if management mode == 'active'."
                )
        self._manual_pool.append(instance_id)
        self.registry[instance_id].proportion = proportion
        self.registry[instance_id].cap = int(proportion * self.totalmem)

        self._rebalance_auto_pool()

    def _add_auto_proportion(self, instance: object) -> float:
        """
        Add an instance to the auto allocation pool with equal share.

        Parameters
        ----------
        instance
            Instance to add to auto allocation pool.

        Returns
        -------
        float
            Proportion assigned to this instance.

        Raises
        ------
        ValueError
            If available auto-allocation pool is less than minimum required size.

        Notes
        -----
        Splits the non-manually-allocated portion of VRAM equally among all
        auto-allocated instances. Triggers rebalancing of the auto pool.
        """
        instance_id = id(instance)
        autopool_available = 1.0 - self.manual_pool_proportion
        if autopool_available <= MIN_AUTOPOOL_SIZE:
            raise ValueError(
                "Available auto-allocation pool is less than "
                "5% of total due to manual allocations. If "
                "this is desired, adjust MIN_AUTOPOOL_SIZE in "
                "mem_manager.py."
            )
        self._auto_pool.append(instance_id)
        self._rebalance_auto_pool()
        return self.registry[instance_id].proportion

    def _rebalance_auto_pool(self) -> None:
        """
        Redistribute available memory equally among auto-allocated instances.

        Notes
        -----
        Calculates the available proportion after manual allocations and
        divides it equally among all instances in the auto pool. Updates
        both proportion and cap for each auto-allocated instance.

        Returns
        -------
        None
        """
        available_proportion = 1.0 - self.manual_pool_proportion
        if len(self._auto_pool) == 0:
            return
        each_proportion = available_proportion / len(self._auto_pool)
        cap = int(each_proportion * self.totalmem)
        for instance_id in self._auto_pool:
            self.registry[instance_id].proportion = each_proportion
            self.registry[instance_id].cap = cap

    def set_global_stride_ordering(
        self, ordering: tuple[str, str, str]
    ) -> None:
        """
        Set the global memory stride ordering for arrays.

        Parameters
        ----------
        ordering
            Tuple containing ``"time"``, ``"run"``, and ``"variable"`` in desired
            order.

        Raises
        ------
        ValueError
            If ordering doesn't contain exactly 'time', 'run', and 'variable'.

        Notes
        -----
        This invalidates all current allocations as arrays need to be
        reallocated with new stride patterns.

        Returns
        -------
        None
        """
        if not all(elem in ("time", "run", "variable") for elem in ordering):
            raise ValueError(
                "Invalid stride ordering - must containt 'time', "
                f"'run', 'variable' but got {ordering}"
            )
        self._stride_order = ordering
        # This will also override 2D arrays, which are unaffected, but the
        # overhead is not significant compared to the 3D arrays.
        self.invalidate_all()

    def free(self, array_label: str) -> None:
        """
        Free an allocation by label across all instances.

        Parameters
        ----------
        array_label
            Label of the allocation to free.

        Returns
        -------
        None
        """
        for settings in self.registry.values():
            if array_label in settings.allocations:
                settings.free(array_label)

    def free_all(self) -> None:
        """
        Free all allocations across all registered instances.

        Returns
        -------
        None
        """
        for settings in self.registry.values():
            settings.free_all()

    def _check_requests(self, requests: dict[str, ArrayRequest]) -> None:
        """
        Validate that all requests are properly formatted.

        Parameters
        ----------
        requests
            Dictionary of requests to validate.

        Raises
        ------
        TypeError
            If requests is not a dict or contains invalid ArrayRequest objects.

        Returns
        -------
        None
        """
        if not isinstance(requests, dict):
            raise TypeError(
                f"Expected dict for requests, got {type(requests)}"
            )
        for key, request in requests.items():
            if not isinstance(request, ArrayRequest):
                raise TypeError(
                    f"Expected ArrayRequest for {key}, got {type(request)}"
                )

    def get_strides(self, request: ArrayRequest) -> Optional[tuple[int, ...]]:
        """
        Calculate memory strides for a given access pattern (stride order).

        Parameters
        ----------
        request
            Array request to calculate strides for.

        Returns
        -------
        tuple of int or None
            Stride tuple for the array, or None if no custom strides needed.

        Notes
        -----
        Only 3D arrays get custom stride optimization. 2D arrays use
        default strides as they are not performance-critical.
        """
        # 2D arrays (in the cubie sytem) are not hammered like the 3d ones,
        # so they're not worth optimising.
        if len(request.shape) != 3:
            strides = None
        else:
            array_native_order = request.stride_order
            desired_order = self._stride_order
            shape = request.shape
            itemsize = request.dtype().itemsize

            if array_native_order == desired_order:
                strides = None
            else:
                dims = {
                    name: size for name, size in zip(array_native_order, shape)
                }
                strides = {}
                current_stride = itemsize

                # Iterate over the desired order reversed; the last dimension
                # in the order changes fastest so it gets the smallest stride.
                for name in reversed(desired_order):
                    strides[name] = current_stride
                    current_stride *= dims[name]
                strides = tuple(strides[dim] for dim in array_native_order)

        return strides

    def get_available_single(self, instance_id: int) -> int:
        """
        Get available memory for a single instance.

        Parameters
        ----------
        instance_id
            ID of the instance to check.

        Returns
        -------
        int
            Available memory in bytes for this instance.

        Warnings
        --------
        UserWarning
            If instance has used more than 95% of allocated memory.
        """
        free, total = self.get_memory_info()
        if self._mode == "passive":
            return free
        else:
            settings = self.registry[instance_id]
            cap = settings.cap
            allocated = settings.allocated_bytes
            headroom = cap - allocated
            if headroom / cap < 0.05:
                warn(
                    f"Instance {instance_id} has used more than 95% of it's "
                    "allotted memory already, and future requests will run "
                    "slowly/in many chunks"
                )
            return min(headroom, free)

    def get_available_group(self, group: str) -> int:
        """
        Get available memory for an entire stream group.

        Parameters
        ----------
        group
            Name of the stream group.

        Returns
        -------
        int
            Available memory in bytes for the group.

        Warnings
        --------
        UserWarning
            If group has used more than 95% of allocated memory.
        """
        free, total = self.get_memory_info()
        instances = self.stream_groups.get_instances_in_group(group)
        if self._mode == "passive":
            return free
        else:
            allocated = 0
            cap = 0
            for instance_id in instances:
                allocated += self.registry[instance_id].allocated_bytes
                cap += self.registry[instance_id].cap
            headroom = cap - allocated
            if headroom / cap < 0.05:
                warn(
                    f"Stream group {group} has used more than 95% of it's "
                    "allotted memory already, and future requests will run "
                    "slowly/in many chunks"
                )
            return min(headroom, free)

    def get_chunks(self, request_size: int, available: int = 0) -> int:
        """
        Calculate number of chunks needed for a memory request.

        Parameters
        ----------
        request_size
            Total size of the request in bytes.
        available
            Available memory in bytes. Defaults to 0.

        Returns
        -------
        int
            Number of chunks needed to fit the request.

        Warnings
        --------
        UserWarning
            If request exceeds available VRAM by more than 20x.
        """
        free, total = self.get_memory_info()
        if request_size / free > 20:
            warn(
                "This request exceeds available VRAM by more than 20x. "
                f"Available VRAM = {free}, request size = {request_size}.",
                UserWarning,
            )
        return int(np.ceil(request_size / available))

    def get_memory_info(self) -> tuple[int, int]:
        """
        Get free and total GPU memory information.

        Returns
        -------
        tuple of int
            (free_memory, total_memory) in bytes.
        """
        return current_mem_info()

    def get_stream_group(self, instance: object) -> str:
        """
        Get the name of the stream group for an instance.

        Parameters
        ----------
        instance
            Instance to query.

        Returns
        -------
        str
            Name of the stream group.
        """
        return self.stream_groups.get_group(instance)

    def is_grouped(self, instance: object) -> bool:
        """
        Check if instance is grouped with others in a named stream.

        Parameters
        ----------
        instance
            Instance to check.

        Returns
        -------
        bool
            True if instance shares a stream group with other instances.
        """
        group = self.get_stream_group(instance)
        if group == "default":
            return False
        peers = self.stream_groups.get_instances_in_group(group)
        if len(peers) == 1:
            return False
        return True

    def allocate_all(
        self,
        requests: dict[str, ArrayRequest],
        instance_id: int,
        stream: "cuda.cudadrv.driver.Stream",
    ) -> dict[str, object]:
        """
        Allocate multiple arrays based on a dictionary of requests.

        Parameters
        ----------
        requests
            Dictionary mapping labels to array requests.
        instance_id
            ID of the requesting instance.
        stream
            CUDA stream for the allocations.

        Returns
        -------
        dict of str to object
            Dictionary mapping labels to allocated arrays.
        """
        responses = {}
        instance_settings = self.registry[instance_id]
        for key, request in requests.items():
            strides = self.get_strides(request)
            arr = self.allocate(
                shape=request.shape,
                dtype=request.dtype,
                memory_type=request.memory,
                stream=stream,
                strides=strides,
            )
            instance_settings.add_allocation(key, arr)
            responses[key] = arr
        return responses

    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: Callable,
        memory_type: str,
        stream: "cuda.cudadrv.driver.Stream" = 0,
        strides: Optional[tuple[int, ...]] = None,
    ) -> object:
        """
        Allocate a single array with specified parameters.

        Parameters
        ----------
        shape
            Shape of the array to allocate.
        dtype
            Constructor returning the precision object for the array elements.
        memory_type
            Type of memory: "device", "mapped", "pinned", or "managed".
        stream
            CUDA stream for the allocation. Defaults to 0.
        strides
            Custom strides for the array. Defaults to None.

        Returns
        -------
        object
            Allocated GPU array.

        Raises
        ------
        ValueError
            If memory_type is not recognized.
        NotImplementedError
            If memory_type is "managed" (not yet supported).
        """
        cp_ = self._allocator == CuPyAsyncNumbaManager
        with current_cupy_stream(stream) if cp_ else contextlib.nullcontext():
            if memory_type == "device":
                return cuda.device_array(shape, dtype, strides=strides)
            elif memory_type == "mapped":
                return cuda.mapped_array(shape, dtype, strides=strides)
            elif memory_type == "pinned":
                return cuda.pinned_array(shape, dtype, strides=strides)
            elif memory_type == "managed":
                raise NotImplementedError("Managed memory not implemented")
            else:
                raise ValueError(f"Invalid memory type: {memory_type}")

    def queue_request(
        self, instance: object, requests: dict[str, ArrayRequest]
    ) -> None:
        """
        Queue allocation requests for batched stream group processing.

        Parameters
        ----------
        instance
            The instance making the request.
        requests
            Dictionary mapping labels to array requests.

        Notes
        -----
        Requests are queued per stream group, allowing multiple components
        to contribute to a single coordinated allocation that can be
        optimally chunked together.

        Returns
        -------
        None
        """
        self._check_requests(requests)
        stream_group = self.get_stream_group(instance)
        if self._queued_allocations.get(stream_group) is None:
            self._queued_allocations[stream_group] = {}
        instance_id = id(instance)
        self._queued_allocations[stream_group].update({instance_id: requests})

    def chunk_arrays(
        self,
        requests: dict[str, ArrayRequest],
        numchunks: int,
        axis: str = "run",
    ) -> dict[str, ArrayRequest]:
        """
        Divide array requests into smaller chunks along a specified axis.

        Parameters
        ----------
        requests
            Dictionary mapping labels to array requests.
        numchunks
            Number of chunks to divide arrays into.
        axis
            Axis name along which to chunk the arrays. Defaults to "run".

        Returns
        -------
        dict of str to ArrayRequest
            New dictionary with modified array shapes for chunking.

        Notes
        -----
        The axis must match a label in the stride ordering. Chunking is
        done conservatively with ceiling division to ensure no data is lost.

        Unchunkable requests (request.unchunkable == True) are left at full size.
        """
        chunked_requests = deepcopy(requests)
        for key, request in chunked_requests.items():
            # Skip chunking for explicitly unchunkable requests
            if getattr(request, "unchunkable", False):
                continue
            # Divide all indices along selected axis by chunks
            run_index = request.stride_order.index(axis)
            newshape = tuple(
                int(np.ceil(value / numchunks)) if i == run_index else value
                for i, value in enumerate(request.shape)
            )
            request.shape = newshape
            chunked_requests[key] = request
        return chunked_requests

    def single_request(
        self,
        instance: Union[object, int],
        requests: dict[str, ArrayRequest],
        chunk_axis: str = "run",
    ) -> None:
        """
        Process a single allocation request with automatic chunking.

        Parameters
        ----------
        instance
            The requesting instance or its ID.
        requests
            Dictionary mapping labels to array requests.
        chunk_axis
            Axis along which to chunk if memory is insufficient. Defaults to "run".

        Raises
        ------
        TypeError
            If requests is not a dict or contains invalid ArrayRequest objects.

        Notes
        -----
        This method calculates available memory, determines chunking needs,
        allocates arrays with optimal strides, and calls the instance's
        allocation_ready_hook with the results.

        Returns
        -------
        None
        """
        self._check_requests(requests)
        if isinstance(instance, int):
            instance_id = instance
        else:
            instance_id = id(instance)

        request_size = get_total_request_size(requests)
        available_memory = self.get_available_single(id(instance))
        numchunks = self.get_chunks(request_size, available_memory)
        chunked_requests = self.chunk_arrays(
            requests, numchunks, axis=chunk_axis
        )

        arrays = self.allocate_all(
            chunked_requests, instance_id, self.get_stream(instance)
        )
        self.registry[instance_id].allocation_ready_hook(
            ArrayResponse(arr=arrays, chunks=numchunks, chunk_axis=chunk_axis)
        )

    def allocate_queue(
        self,
        triggering_instance: object,
        limit_type: str = "group",
        chunk_axis: str = "run",
    ) -> None:
        """
        Process all queued requests for a stream group with coordinated chunking.

        Parameters
        ----------
        triggering_instance
            The instance that triggered queue processing.
        limit_type
            Limiting strategy: "group" for aggregate limits or "instance" for
            individual instance limits. Defaults to "group".
        chunk_axis
            Axis along which to chunk arrays if needed. Defaults to "run".

        Notes
        -----
        Processes all pending requests in the same stream group, applying
        coordinated chunking based on the specified limit type. Calls
        allocation_ready_hook for each instance with their results.

        Returns
        -------
        None
        """
        stream_group = self.get_stream_group(triggering_instance)
        peers = self.stream_groups.get_instances_in_group(stream_group)
        stream = self.get_stream(triggering_instance)
        queued_requests = self._queued_allocations.get(stream_group, {})
        n_queued = len(queued_requests)
        if not queued_requests:
            return None
        elif n_queued == 1:
            for instance_id, requests_dict in queued_requests.items():
                self.single_request(
                    instance=instance_id,
                    requests=requests_dict,
                    chunk_axis=chunk_axis,
                )
        else:
            numchunks = 1  # safe default
            if limit_type == "group":
                available_memory = self.get_available_group(stream_group)
                request_size = sum(
                    [
                        get_total_request_size(request)
                        for request in queued_requests.values()
                    ]
                )
                numchunks = self.get_chunks(request_size, available_memory)

            elif limit_type == "instance":
                numchunks = 0
                for instance_id, requests_dict in queued_requests.items():
                    available_memory = self.get_available_single(instance_id)
                    request_size = get_total_request_size(requests_dict)
                    chunks = self.get_chunks(request_size, available_memory)
                    # Take the runnning maximum per-instance chunk size
                    numchunks = chunks if chunks > numchunks else numchunks

            notaries = set(peers) - set(queued_requests.keys())
            for instance_id, requests_dict in queued_requests.items():
                chunked_request = self.chunk_arrays(
                    requests_dict, numchunks, chunk_axis
                )
                arrays = self.allocate_all(
                    chunked_request, instance_id, stream=stream
                )
                response = ArrayResponse(
                    arr=arrays, chunks=numchunks, chunk_axis=chunk_axis
                )
                self.registry[instance_id].allocation_ready_hook(response)

            for peer in notaries:
                self.registry[peer].allocation_ready_hook(
                    ArrayResponse(
                        arr={}, chunks=numchunks, chunk_axis=chunk_axis
                    )
                )
        return None

    def to_device(
        self,
        instance: object,
        from_arrays: list[object],
        to_arrays: list[object],
    ) -> None:
        """
        Copy data to device arrays using the instance's stream.

        Parameters
        ----------
        instance
            Instance whose stream to use for copying.
        from_arrays
            Source arrays to copy from.
        to_arrays
            Destination device arrays to copy to.

        Returns
        -------
        None
        """
        stream = self.get_stream(instance)
        cp_ = self._allocator == CuPyAsyncNumbaManager
        with current_cupy_stream(stream) if cp_ else contextlib.nullcontext():
            for i, from_array in enumerate(from_arrays):
                cuda.to_device(from_array, stream=stream, to=to_arrays[i])

    def from_device(
        self,
        instance: object,
        from_arrays: list[object],
        to_arrays: list[object],
    ) -> None:
        """
        Copy data from device arrays using the instance's stream.

        Parameters
        ----------
        instance
            Instance whose stream to use for copying.
        from_arrays
            Source device arrays to copy from.
        to_arrays
            Destination arrays to copy to.

        Returns
        -------
        None
        """
        stream = self.get_stream(instance)
        cp_ = self._allocator == CuPyAsyncNumbaManager
        with current_cupy_stream(stream) if cp_ else contextlib.nullcontext():
            for i, from_array in enumerate(from_arrays):
                from_array.copy_to_host(to_arrays[i], stream=stream)

    def sync_stream(self, instance: object) -> None:
        """
        Synchronize the CUDA stream for an instance.

        Parameters
        ----------
        instance
            Instance whose stream to synchronize.

        Returns
        -------
        None
        """
        stream = self.get_stream(instance)
        stream.synchronize()


def get_total_request_size(request: dict[str, ArrayRequest]) -> int:
    """
    Calculate the total memory size of a request in bytes.

    Parameters
    ----------
    request
        Dictionary of array requests to sum.

    Returns
    -------
    int
        Total size in bytes across all requests.
    """
    return sum(
        prod(request.shape) * request.dtype().itemsize
        for request in request.values()
    )
