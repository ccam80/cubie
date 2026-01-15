"""GPU memory management utilities for coordinating cubie allocations.

Notes
-----
Memory allocation chunking is performed along the run axis to handle batches
that exceed available GPU memory. The chunking process is automatic and
coordinated across all instances in a stream group.
"""

from typing import Any, Optional, Callable, Dict, Tuple
from warnings import warn
import contextlib
from copy import deepcopy

from numba import cuda
from attrs import define, Factory as attrsFactory, field
from attrs.validators import (
    in_ as attrsval_in,
    instance_of as attrsval_instance_of,
    optional as attrsval_optional,
)
from numpy import (
    ceil as np_ceil,
    ndarray,
    empty as np_empty,
    floor as np_floor,
)
from math import prod

from cubie.cuda_simsafe import (
    BaseCUDAMemoryManager,
    NumbaCUDAMemoryManager,
    current_mem_info,
    set_cuda_memory_manager,
    CUDA_SIMULATION,
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
    pass


def _ensure_cuda_context() -> None:
    """
    Ensure CUDA context is initialized before memory operations.

    This function validates that a CUDA context exists and is functional,
    triggering initialization if needed. If the context cannot be created
    or is in a bad state, it raises a clear exception rather than causing
    a segfault.

    This is particularly important after cuda.close() calls which can
    leave the context in a state requiring reinitialization.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If CUDA context cannot be initialized or is not functional.
    """
    if not CUDA_SIMULATION:
        try:
            # Attempt to access current context - triggers creation if
            # needed. After cuda.close(), this will create a new context
            ctx = cuda.current_context()
            if ctx is None:
                raise RuntimeError(
                    "CUDA context is None - GPU may not be accessible"
                )
            # Skip memory info check for performance; context existence
            # is sufficient validation for most operations
        except Exception as e:
            # Provide helpful error message instead of segfault
            raise RuntimeError(
                f"Failed to initialize or verify CUDA context: {e}. "
                "This may indicate GPU driver issues, insufficient "
                "permissions, or the GPU may be in an unrecoverable "
                "state after cuda.close(). Try restarting the process "
                "or checking GPU availability."
            ) from e


# These will be keys to a dict, so must be hashable: eq=False
@define(eq=False)
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

    proportion: float = field(
        default=1.0, validator=attrsval_instance_of(float)
    )
    allocations: dict = field(
        default=attrsFactory(dict), validator=attrsval_instance_of(dict)
    )
    invalidate_hook: Callable[[], None] = field(
        default=placeholder_invalidate,
        validator=attrsval_instance_of(Callable),
    )
    allocation_ready_hook: Callable[[ArrayResponse], None] = field(
        default=placeholder_dataready
    )
    cap: Optional[int] = field(
        default=None, validator=attrsval_optional(attrsval_instance_of(int))
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
        If a previous allocation exists with the same key, it is freed
        before adding the new allocation.

        Returns
        -------
        None
        """

        if key in self.allocations:
            # Free the old allocation before adding the new one
            self.free(key)
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
            del self.allocations[key]
        else:
            warn(
                f"Attempted to free allocation for {key}, but "
                f"it was not found in the allocations list."
            )

    def free_all(self) -> None:
        """
        Drop all references to allocated arrays.

        Returns
        -------
        None
        """
        to_free = self.allocations.copy()
        for key in to_free:
            self.free(key)

    @property
    def allocated_bytes(self) -> int:
        """Total bytes allocated across tracked arrays."""
        total = 0
        for arr in self.allocations.values():
            total += arr.nbytes
        return total


@define
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

    totalmem: int = field(
        default=None, validator=attrsval_optional(attrsval_instance_of(int))
    )
    registry: dict[int, InstanceMemorySettings] = field(
        default=attrsFactory(dict),
        validator=attrsval_optional(attrsval_instance_of(dict)),
    )
    stream_groups: StreamGroups = field(default=attrsFactory(StreamGroups))
    _mode: str = field(
        default="passive", validator=attrsval_in(["passive", "active"])
    )
    _allocator: BaseCUDAMemoryManager = field(
        default=NumbaCUDAMemoryManager,
        validator=attrsval_optional(attrsval_instance_of(object)),
    )
    _auto_pool: list[int] = field(
        default=attrsFactory(list), validator=attrsval_instance_of(list)
    )
    _manual_pool: list[int] = field(
        default=attrsFactory(list), validator=attrsval_instance_of(list)
    )
    _queued_allocations: Dict[str, Dict] = field(
        default=attrsFactory(dict), validator=attrsval_instance_of(dict)
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
        # Ensure there's a valid context before change - only relevant if user
        # switches in rapid succession with no interceding operations.
        context = cuda.current_context()
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
        #   Specific excerpt: The invalidation of modules means that all
        #   functions compiled with @cuda.jit prior to context destruction
        #   will need to be redefined, as the code underlying them will also
        #   have been unloaded from the GPU.
        cuda.close()
        self.context = cuda.current_context()
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
            return
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
            return
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

    def _add_manual_proportion(
        self, instance: object, proportion: float
    ) -> None:
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

    def create_host_array(
        self,
        shape: tuple[int, ...],
        dtype: type,
        memory_type: str = "pinned",
        like: Optional[ndarray] = None,
    ) -> ndarray:
        """
        Create a C-contiguous host array.

        Parameters
        ----------
        shape
            Shape of the array to create.
        dtype
            Data type for the array elements.
        memory_type
            Memory type for the host array. Must be ``"pinned"`` or
            ``"host"``. Defaults to ``"pinned"``.
        like
            A source array to copy data from. If provided, the new array has
            the same data as like; if not, it is filled with zeros

        Returns
        -------
        numpy.ndarray
            C-contiguous host array.
        """
        _ensure_cuda_context()
        if memory_type not in ("pinned", "host"):
            raise ValueError(
                f"memory_type must be 'pinned' or 'host', got '{memory_type}'"
            )
        use_pinned = memory_type == "pinned"
        if use_pinned:
            arr = cuda.pinned_array(shape, dtype=dtype)
        else:
            arr = np_empty(shape, dtype=dtype)
        if like is not None:
            arr[:] = like
        else:
            arr.fill(0.0)
        return arr

    def get_available_memory(self, group: str) -> int:
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
            arr = self.allocate(
                shape=request.shape,
                dtype=request.dtype,
                memory_type=request.memory,
                stream=stream,
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
    ) -> object:
        """
        Allocate a single C-contiguous array with specified parameters.

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

        Returns
        -------
        object
            Allocated GPU array.

        Raises
        ------
        ValueError
            If memory_type is not recognized.
        NotImplementedError
            If memory_type is "managed" (not supported).
        """
        _ensure_cuda_context()
        cp_ = self._allocator == CuPyAsyncNumbaManager
        with current_cupy_stream(stream) if cp_ else contextlib.nullcontext():
            if memory_type == "device":
                return cuda.device_array(shape, dtype)
            elif memory_type == "mapped":
                return cuda.mapped_array(shape, dtype)
            elif memory_type == "pinned":
                return cuda.pinned_array(shape, dtype)
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
        _ensure_cuda_context()
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
        _ensure_cuda_context()
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
        _ensure_cuda_context()
        stream = self.get_stream(instance)
        stream.synchronize()

    def _extract_num_runs(
        self,
        queued_requests: Dict[str, Dict[str, ArrayRequest]],
    ) -> int:
        """Extract total_runs from queued allocation requests.
        
        Iterates through all ArrayRequest objects in queued_requests and returns
        the first non-None total_runs value found. Validates that all requests
        with total_runs set have the same value.
        
        Parameters
        ----------
        queued_requests
            Nested dict: instance_id -> {array_label -> ArrayRequest}
        
        Returns
        -------
        int
            The total number of runs for chunking calculations
        
        Raises
        ------
        ValueError
            If no requests contain total_runs, or if inconsistent values found
        
        Notes
        -----
        Requests with total_runs=None are ignored (e.g., driver_coefficients).
        At least one request must provide total_runs for chunking to work.
        """
        total_runs_values = set()
        
        # Iterate through nested dict structure
        for instance_id, requests_dict in queued_requests.items():
            for array_label, request in requests_dict.items():
                if request.total_runs is not None:
                    total_runs_values.add(request.total_runs)
        
        # Validate we found at least one total_runs
        if len(total_runs_values) == 0:
            raise ValueError(
                "No total_runs found in allocation requests. At least one "
                "request must specify total_runs for chunking calculations."
            )
        
        # Validate all total_runs are consistent
        if len(total_runs_values) > 1:
            raise ValueError(
                f"Inconsistent total_runs in requests: found "
                f"{total_runs_values}. All requests with total_runs "
                "must have the same value."
            )
        
        # Return the single value
        return total_runs_values.pop()

    def allocate_queue(
        self,
        triggering_instance: object,
    ) -> None:
        """
        Process all queued requests for a stream group with coordinated chunking.

        Chunking is always performed along the run axis when memory
        constraints require splitting the batch.

        Parameters
        ----------
        triggering_instance
            The instance that triggered queue processing.

        Notes
        -----
        Processes all pending requests in the same stream group, applying
        coordinated chunking based on available memory. Calls
        allocation_ready_hook for each instance with their results.

        The num_runs value is extracted from ArrayRequest.total_runs fields
        rather than from triggering_instance attributes. All requests with
        non-None total_runs must have the same value.

        Returns
        -------
        None
        """
        stream_group = self.get_stream_group(triggering_instance)
        stream = self.get_stream(triggering_instance)
        queued_requests = self._queued_allocations.pop(stream_group, {})

        # Extract num_runs from ArrayRequest total_runs fields
        num_runs = self._extract_num_runs(queued_requests)

        chunk_length, num_chunks = self.get_chunk_parameters(
            queued_requests, num_runs, stream_group
        )
        peers = self.stream_groups.get_instances_in_group(stream_group)
        notaries = set(peers) - set(queued_requests.keys())
        for instance_id, requests_dict in queued_requests.items():
            chunked_shapes = self.compute_chunked_shapes(
                requests_dict,
                chunk_length,
            )

            chunked_requests = deepcopy(requests_dict)
            for key, request in chunked_requests.items():
                request.shape = chunked_shapes[key]

            arrays = self.allocate_all(
                chunked_requests, instance_id, stream=stream
            )
            response = ArrayResponse(
                arr=arrays,
                chunks=num_chunks,
                chunk_length=chunk_length,
                chunked_shapes=chunked_shapes,
            )

            self.registry[instance_id].allocation_ready_hook(response)
            for peer in notaries:
                self.registry[peer].allocation_ready_hook(
                    ArrayResponse(
                        arr={},
                        chunks=num_chunks,
                        chunk_length=chunk_length,
                        chunked_shapes={},
                    )
                )

        return None

    def get_chunk_parameters(
        self,
        requests: Dict[str, Dict],
        axis_length: int,
        stream_group: str,
    ) -> Tuple[int, int]:
        """
        Calculate number of chunks and chunk size for a dict of array requests.

        Chunking is performed along the run axis only.

        Parameters
        ----------
        requests
            Dictionary mapping instance IDs to their array requests.
        axis_length
            Unchunked length of the chunking axis.
        stream_group
            Name of the stream group making the request.

        Returns
        -------
        int, int
            Length of chunked axis and number of chunks needed to fit the
            request.

        Warnings
        --------
        UserWarning
            If request exceeds available VRAM by more than 20x.
        """
        free, _ = self.get_memory_info()
        available_memory = self.get_available_memory(stream_group)
        chunkable_size, unchunkable_size = get_portioned_request_size(
            requests,
        )

        request_size = chunkable_size + unchunkable_size

        if request_size < available_memory:
            return axis_length, 1  # No chunking needed

        if request_size / free > 20:
            warn(
                "This request exceeds available VRAM by more than 20x. "
                f"Available VRAM = {free}, request size = {request_size}.",
                UserWarning,
            )

        # Check for all arrays unchunkable
        if chunkable_size == 0:
            raise ValueError(
                f"All requested arrays are unchunkable, but request size "
                f"({request_size}) exceeds available memory "
                f"({available_memory}). Cannot proceed."
            )

        # Guard: unchunkable arrays alone exceed available memory
        if unchunkable_size >= available_memory:
            raise ValueError(
                f"Unchunkable arrays require {unchunkable_size} bytes but only "
                f"{available_memory} bytes available. Cannot proceed."
            )

        # Calculate chunk size and number of chunks once we know it's eligible
        else:
            available_to_chunk = available_memory - unchunkable_size
            chunk_ratio = chunkable_size / available_to_chunk

            # Maximum chunk size that fits in available memory
            max_chunk_size = int(np_floor(axis_length / chunk_ratio))
            if max_chunk_size == 0:
                raise ValueError(
                    "Can't fit a single run in GPU VRAM. "
                    f"Available memory: {available_memory}. "
                    f"Request size: {request_size}. "
                    f"Chunkable request size: {chunkable_size}."
                )
            # With floor rounding, we might end up with an extra chunk or two
            num_chunks = int(np_ceil(axis_length / max_chunk_size))

        return max_chunk_size, num_chunks

    def compute_chunked_shapes(
        self,
        requests: dict[str, ArrayRequest],
        chunk_size: int,
    ) -> dict[str, Tuple[int, ...]]:
        """
        Compute per-array chunked shapes based on available memory.

        Parameters
        ----------
        requests
            Dictionary mapping labels to array requests.
        chunk_size
            Length of chunked arrays along run axis

        Returns
        -------
        dict[str, tuple[int, ...]]
            Mapping from array labels to their per-chunk shapes.

        Notes
        -----
        Unchunkable arrays retain their original shape.
        """
        chunked_shapes = {}
        for key, request in requests.items():
            if is_request_chunkable(request):
                axis_index = request.chunk_axis_index
                newshape = replace_with_chunked_size(
                    shape=request.shape,
                    axis_index=axis_index,
                    chunked_size=chunk_size,
                )
                chunked_shapes[key] = newshape
            else:
                chunked_shapes[key] = request.shape

        return chunked_shapes





def get_portioned_request_size(
    requests: dict[str, dict[str, ArrayRequest]],
) -> int:
    """
    Calculate total memory requested for the chunkable and unchunkable
    portions of the request.

    Chunking is performed along the run axis only.

    Parameters
    ----------
    requests
        Dictionary of array requests to analyze.

    Returns
    -------
    int, int
        chunkable, unchunkable - Total bytes for arrays that can be chunked
        or not, respectively, along the "run" axis.

    Notes
    -----
    Arrays are chunkable if:
    - request.unchunkable is False
    - The array has a "run" axis
    """
    chunkable = 0
    unchunkable = 0
    for reqs in requests.values():
        chunkable += sum(
            prod(req.shape) * req.dtype().itemsize
            for req in reqs.values()
            if is_request_chunkable(req)
        )
        unchunkable += sum(
            prod(req.shape) * req.dtype().itemsize
            for req in reqs.values()
            if not is_request_chunkable(req)
        )
    return chunkable, unchunkable


def is_request_chunkable(request) -> bool:
    """
    Determine if a single ArrayRequest is chunkable.

    Chunking is always performed along the run axis.

    Parameters
    ----------
    request
        The ArrayRequest to evaluate.

    Returns
    -------
    bool
        True if the request is chunkable, False otherwise.

    Notes
    -----
    A request is considered chunkable if:
    - request.unchunkable is False
    - chunk_axis_index is not None and within bounds
    - run axis has length > 1 (not a degenerate run axis)
    """
    if request.unchunkable:
        return False
    if len(request.shape) == 0:
        return False
    if request.chunk_axis_index is None:
        return False
    if request.chunk_axis_index >= len(request.shape):
        return False
    if request.shape[request.chunk_axis_index] == 1:
        return False
    return True


def replace_with_chunked_size(
    shape: Tuple[int, ...],
    axis_index: int,
    chunked_size: int,
) -> Tuple[int, ...]:
    """
    Replace the "run" axis in shape with chunked size.

    Parameters
    ----------
    shape
        Original shape of the array.
    axis_index
        integer index of the run axis in shape
    chunked_size
        Length of array after chunking along run axis

    Returns
    -------
    tuple[int, ...]
        New shape with chunked size along the "run" axis.
    """
    newshape = tuple(
        dim if i != axis_index else chunked_size for i, dim in enumerate(shape)
    )
    return newshape
