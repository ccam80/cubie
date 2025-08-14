from os import environ
from typing import Optional, Callable, Union, Dict
from warnings import warn
import contextlib
from copy import deepcopy

from numba import cuda
import attrs
import attrs.validators as val
from attrs import Factory
import numpy as np
from math import prod

from cubie.memory.cupy_emm import current_cupy_stream
from cubie.memory.stream_groups import StreamGroups
from cubie.memory.array_requests import ArrayRequest, ArrayResponse
from cubie.memory.cupy_emm import CuPyAsyncNumbaManager, CuPySyncNumbaManager

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from cubie.cudasim_utils import (FakeNumbaCUDAMemoryManager as
                                     NumbaCUDAMemoryManager)
    from cubie.cudasim_utils import (FakeBaseCUDAMemoryManager as
                                     BaseCUDAMemoryManager)
    from cubie.cudasim_utils import fake_get_memory_info as current_mem_info
    from cubie.cudasim_utils import fake_set_manager as set_cuda_memory_manager
else:
    from numba.cuda.cudadrv.driver import NumbaCUDAMemoryManager
    from numba.cuda import BaseCUDAMemoryManager
    def current_mem_info():
        return cuda.current_context().get_memory_info()
    from numba.cuda import set_memory_manager as set_cuda_memory_manager


MIN_AUTOPOOL_SIZE = 0.05

def dummy_invalidate()->None:
    """Default dummy invalidate hook, does nothing."""
    pass

def dummy_dataready(response: ArrayResponse) -> None:
    """Default dummy dataready hook, does nothing."""

# These will be keys to a dict, so must be hashable: eq=False
@attrs.define(eq=False)
class InstanceMemorySettings:
    """Memory registry information for a registered class

    Attributes:
    ==========
        proportion: float
            Proportion of total VRAM assigned to this instance
        allocations: dict
            A dictionary, keyed by a label for the array (given in an
            ArrayRequest), with values a reference to the array itself,
            of all current allocations. Both functions as a "keepalive"
            reference and a way to calculate the total allocated memory.
        invalidate_hook: callable
            A function to call when a change to the CUDA memory system
            occurs; when the allocator/memory manager is changed,
            for example, every stream and array and kernel will be invalidated,
            and will need to be re-allocated or redefined.
        cap: int
            Maximum allocatable bytes for this instance (set by manager based
            on total VRAM and proportion)

    Properties
    ==========
        allocated_bytes: int
            Total number of bytes across all allocated arrays for the instance
    """
    proportion: float = attrs.field(
            default=1.0,
            validator=val.instance_of(float))
    allocations: dict = attrs.field(
            default=Factory(dict),
            validator=val.instance_of(dict))
    invalidate_hook: Callable[[None], None] = attrs.field(
            default=dummy_invalidate,
            validator=val.instance_of(Callable))
    allocation_ready_hook: Callable[[ArrayResponse], None] = attrs.field(
            default=dummy_dataready,
    )
    cap: int = attrs.field(
            default=None,
            validator=val.optional(val.instance_of(int)))


    def add_allocation(self, key, arr):
        """Add an allocation to the instance's allocations list

        NOTE: This will just overwrite the previous allocation, which should
        function as intended, but suggests that the previous batch has not been
        properly deallocated. This emits a warning, so that you're aware."""

        if key in self.allocations:
            warn(
                f"Overwriting previous allocation for {key} at a "
                f"settings level - this suggests that the previous "
                f"array wasn't deallocated properly using the "
                f"memory manager."
            )
        self.allocations[key] = arr

    def free(self, key):
        """Free an allocation by key"""
        if key in self.allocations:
            newalloc = {k: v for k, v in self.allocations.items() if k != key}
        else:
            warn(f"Attempted to free allocation for {key}, but "
                          f"it was not found in the allocations list.")
            newalloc = self.allocations
        self.allocations = newalloc

    def free_all(self):
        """Drop all references to allocated arrays"""
        for key in self.allocations:
            self.free(key)

    @property
    def allocated_bytes(self):
        total = 0
        for arr in self.allocations.values():
            total += arr.nbytes
        return total

@attrs.define
class MemoryManager:
    """Singleton interface for managing memory allocation in cubie,
    and between cubie and other modules. In it's most basic form, it just
    provides a way to change numba's allocator and "chunks" allocation
    requests based on available memory.

    In active management mode, it manages the proportion of total VRAM each
    instance can be allocated, in case of greedy memory processes that can
    be down-prioritised (run over more chunks, more slowly). Processes can
    be manually assigned a proportion of VRAM, or the manager can split the
    memory evenly.

    Any array allocation comes through this module. The MemoryManager
    accepts an ArrayRequest object, and returns an ArrayResponse object,
    which has a reference to the array and the number of chunks to divide
    the problem into.

    MemoryManager assigns each response a stream, so that different areas of
    software can run asynchronously. To combine streams, so that (for
    example) a solver is in the same stream as it's array allocator, assign
    them to a "stream group" when registering.

    Parameters
    ==========

    """

    totalmem: int = attrs.field(default=None,
                                validator=val.optional(val.instance_of(int)))
    registry: dict[int, InstanceMemorySettings] = attrs.field(
            default=Factory(dict),
            validator=val.optional(val.instance_of(dict)))
    stream_groups: StreamGroups = attrs.field(
            default=Factory(StreamGroups))
    _mode: str = attrs.field(
            default="passive",
            validator=val.in_(["passive", "active"]))
    _allocator: BaseCUDAMemoryManager = attrs.field(
            default=NumbaCUDAMemoryManager,
            validator=val.optional(val.instance_of(object)))
    _auto_pool: list[int] = attrs.field(
            default=Factory(list),
            validator=val.instance_of(list))
    _manual_pool: list[int] = attrs.field(
            default=Factory(list),
            validator=val.instance_of(list))
    _stride_order: tuple[str, str, str] = attrs.field(
            default=("time", "run", "variable"),
            validator=val.instance_of(tuple))
    _queued_allocations: Dict[str, Dict] = attrs.field(
        default=Factory(dict),
        validator=val.instance_of(dict))

    def __attrs_post_init__(self):
        free, total = self.get_memory_info()
        self.totalmem = total
        self.registry = {}
        # self.set_allocator("default")

    def register(self,
                 instance,
                 proportion: Optional[float] = None,
                 invalidate_cache_hook: Callable = dummy_invalidate,
                 allocation_ready_hook: Callable = dummy_dataready,
                 stream_group: str = "default",
                 ):
        """
        Register an instance and set its memory allocation settings.

        Parameters
        ----------
        instance: object
            The instance to register.
        proportion: float, optional
            Proportion of VRAM to allocate to this instance. If not specified,
            instance will automatically be assigned an equal portion of the
            total VRAM to other auto-assigned instances
        stream_group: str, optional
        invalidate_cache_hook: callable, optional
        allocation_ready_hook: callable, optional
        """
        instance_id = id(instance)
        if instance_id in self.registry:
            raise ValueError("Instance already registered")

        self.stream_groups.add_instance(instance, stream_group)

        settings = InstanceMemorySettings(
                invalidate_hook=invalidate_cache_hook,
                allocation_ready_hook=allocation_ready_hook)

        self.registry[instance_id] = settings

        if proportion:
            if not 0 <= proportion <= 1:
                raise ValueError("Proportion must be between 0 and 1")
            self._add_manual_proportion(instance, proportion)
        else:
            self._add_auto_proportion(instance)


    def set_allocator(self, name: str):
        """ Set the external memory manager in Numba

        Parameters
        ----------
        name: str
            - "cupy_async": use CuPy's MemoryAsyncPool, see:
            https://docs.cupy.dev/en/latest/reference/generated/cupy.cuda.MemoryAsyncPool.html
            warning: experimental!
            - "cupy": use CuPy's MemoryPool, see:
            https://docs.cupy.dev/en/latest/reference/generated/cupy.cuda.MemoryPool.html
            - "default": use Numba's default memory manager.

        WARNING: A change to the memory manager requires the cuda context to be
        closed and a new one opened. This means that everything previously
        compiled or allocated will become invalidated, and a full rebuild from
        scratch will be required.
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
        # https://nvidia.github.io/numba-cuda/user/external-memory.html#setting-emm-plugin
        # WARNING - this will invalidate all prior streams, arrays, and funcs!
        # CUDA_ERROR_INVALID_CONTEXT or CUDA_ERROR_CONTEXT_IS_DESTROYED
        # suggests you're using an old reference.
        cuda.close()
        self.invalidate_all()
        self.reinit_streams()

    def set_limit_mode(self, mode: str):
        """Sets the memory allocation limiting to active or pasive."""
        if mode not in ["passive", "active"]:
            raise ValueError(f"Unknown mode: {mode}")
        self._mode = mode

    def get_stream(self, instance):
        """Gets the stream associated with an instance"""
        return self.stream_groups.get_stream(instance)

    def change_stream_group(self, instance, new_group):
        """Move instance onto another stream"""
        self.stream_groups.change_group(instance, new_group)

    def reinit_streams(self):
        """Gets a fresh set of streams if the context has been closed."""
        self.stream_groups.reinit_streams()

    def invalidate_all(self):
        """Calls each registered instance's invalidate hook"""
        self.free_all()
        for registered_instance in self.registry.values():
            registered_instance.invalidate_hook()

    def set_manual_proportion(self, instance: object, proportion: float):
        """Set manual allocation proportion for an already-manual instance"""
        instance_id = id(instance)
        if proportion < 0 or proportion > 1:
            raise ValueError("Proportion must be between 0 and 1")
        self._manual_pool.remove(instance_id)
        self._add_manual_proportion(instance, proportion)
        self.registry[instance_id].proportion = proportion

    def set_manual_limit_mode(self, instance: object, proportion: float):
        """Set an auto-limited instance to manual allocation mode"""
        instance_id = id(instance)
        settings = self.registry[instance_id]
        if instance_id in self._manual_pool:
            raise ValueError("Instance is already in manual allocation pool")
        self._auto_pool.remove(instance_id)
        self._add_manual_proportion(instance, proportion)
        settings.proportion = proportion

    def set_auto_limit_mode(self, instance):
        """Set an manual-limited instance to auto allocation mode"""
        instance_id = id(instance)
        settings = self.registry[instance_id]
        if instance_id in self._auto_pool:
            raise ValueError("Instance is already in auto allocation pool")
        self._manual_pool.remove(instance_id)
        settings.proportion = self._add_auto_proportion(instance)

    def proportion(self, instance):
        """Returns the maximum proportion of VRAM allocated to this instance"""
        instance_id = id(instance)
        return self.registry[instance_id].proportion

    def cap(self, instance):
        """Returns the maximum allocatable bytes for this instance"""
        instance_id = id(instance)
        settings = self.registry.get(instance_id)
        return settings.cap

    @property
    def manual_pool_proportion(self):
        """The total proportion of VRAM currently manually assigned"""
        manual_settings = [self.registry[instance_id] for instance_id
                           in self._manual_pool]
        pool_proportion = sum([settings.proportion for settings in manual_settings])
        return pool_proportion

    @property
    def auto_pool_proportion(self):
        """The total proportion of VRAM automatically distributed"""
        auto_settings = [self.registry[instance_id] for instance_id in
                         self._auto_pool]
        pool_proportion = sum(
                [settings.proportion for settings in auto_settings])
        return pool_proportion

    def _add_manual_proportion(self, instance: object, proportion: float):
        """Add an instance to the manual allocation pool"""
        instance_id = id(instance)
        new_manual_pool_size = self.manual_pool_proportion + proportion
        if new_manual_pool_size > 1.0:
            raise ValueError("Manual proportion would exceed total "
                             "available memory")
        elif new_manual_pool_size > 1.0 - MIN_AUTOPOOL_SIZE:
            if len(self._auto_pool) > 0:
                raise ValueError(
                    "Manual proportion would leave less than 5% "
                    "of memory for auto-allocated processes. If "
                    "this is desired, adjust MIN_AUTOPOOL_SIZE in "
                    "mem_manager.py."
                )
            else:
                warn("Manual proportion leaves less than 5% of memory for "
                     "auto allocation if management mode == 'active'.")
        self._manual_pool.append(instance_id)
        self.registry[instance_id].proportion = proportion
        self.registry[instance_id].cap = int(proportion * self.totalmem)

        self._rebalance_auto_pool()

    def _add_auto_proportion(self, instance):
        """Splits non-manually-allocated portion of VRAM amongst the rest"""
        instance_id = id(instance)
        autopool_available = 1.0 - self.manual_pool_proportion
        if autopool_available <= MIN_AUTOPOOL_SIZE:
            raise ValueError("Available auto-allocation pool is less than "
                             "5% of total due to manual allocations. If "
                             "this is desired, adjust MIN_AUTOPOOL_SIZE in "
                             "mem_manager.py.")
        self._auto_pool.append(instance_id)
        self._rebalance_auto_pool()
        return self.registry[instance_id].proportion

    def _rebalance_auto_pool(self):
        available_proportion = 1.0 - self.manual_pool_proportion
        if len(self._auto_pool) == 0:
            return
        each_proportion = available_proportion / len(self._auto_pool)
        cap = int(each_proportion * self.totalmem)
        for instance_id in self._auto_pool:
            self.registry[instance_id].proportion = each_proportion
            self.registry[instance_id].cap = cap


    def set_global_stride_ordering(self, ordering: tuple[str, str, str]):
        """ Sets the ordering of arrays in memory"""
        if not all(elem in ("time", "run", "variable") for elem in ordering):
            raise ValueError("Invalid stride ordering - must containt 'time', "
                             f"'run', 'variable' but got {ordering}")
        self._stride_order = ordering
        # This will also override 2D arrays, which are unaffected, but the
        # overhead is not significant compared to the 3D arrays.
        self.invalidate_all()

    def free(self, array_label: str):
        for settings in self.registry.values():
            if array_label in settings.allocations:
                settings.free(array_label)

    def free_all(self):
        for settings in self.registry.values():
            settings.free_all()

    def _check_requests(self, requests):
        """Check that all requests are valid"""
        if not isinstance(requests, dict):
            raise TypeError(f"Expected dict for requests, got "
                            f"{type(requests)}")
        for key, request in requests.items():
            if not isinstance(request, ArrayRequest):
                raise TypeError(f"Expected ArrayRequest for {key}, "
                                f"got {type(request)}")

    def get_strides(self, request):
        """Determine the strides for a given request"""
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
                dims = {name: size for name, size in
                        zip(array_native_order, shape)}
                strides = {}
                current_stride = itemsize

                # Iterate over the desired order reversed; the last dimension
                # in the order changes fastest so it gets the smallest stride.
                for name in reversed(desired_order):
                    strides[name] = current_stride
                    current_stride *= dims[name]
                strides = tuple(strides[dim] for dim in array_native_order)

        return strides

    def get_available_single(self, instance_id):
        free, total = self.get_memory_info()
        if self._mode == "passive":
            return free
        else:
            settings = self.registry[instance_id]
            cap = settings.cap
            allocated = settings.allocated_bytes
            headroom = cap - allocated
            if headroom / cap < 0.05:
                warn(f"Instance {instance_id} has used more than 95% of it's "
                     "allotted memory already, and future requests will run "
                     "slowly/in many chunks")
            return min(headroom, free)

    def get_available_group(self, group: str):
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
                warn(f"Stream group {group} has used more than 95% of it's "
                     "allotted memory already, and future requests will run "
                     "slowly/in many chunks")
            return min(headroom, free)

    def get_chunks(self, request_size: int, available: int=0):
        """Determine the number of "chunks" required to store a request
        """
        free, total = self.get_memory_info()
        if request_size / free > 20:
            warn("This request exceeds available VRAM by more than 20x. "
                 f"Available VRAM = {free}, request size = {request_size}.",
                 UserWarning)
        return int(np.ceil(request_size / available))

    def get_memory_info(self):
        """ Get free and total memory from GPU context"""
        return current_mem_info()

    def get_stream_group(self, instance):
        """ Get name of the stream group for an instance """
        return self.stream_groups.get_group(instance)

    def is_grouped(self, instance):
        """Returns True if instance is grouped with others in named stream."""
        group = self.get_stream_group(instance)
        if group == 'default':
            return False
        peers = self.stream_groups.get_instances_in_group(group)
        if len(peers) == 1:
            return False
        return True

    def allocate_all(self, requests, instance_id, stream):
        """Allocate a dict of arrays based on a dict of requests"""
        responses={}
        instance_settings = self.registry[instance_id]
        for key, request in requests.items():
            strides = self.get_strides(request)
            arr = self.allocate(
                shape=request.shape,
                dtype=request.dtype,
                memory_type=request.memory,
                stream=stream,
                strides=strides
            )
            instance_settings.add_allocation(key, arr)
            responses[key] = arr
        return responses

    def allocate(self, shape, dtype, memory_type, stream=0, strides=None):
        """Allocate a single array according to request parameters."""
        cupy_ = self._allocator == CuPyAsyncNumbaManager
        with current_cupy_stream(stream) if cupy_ else contextlib.nullcontext():
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

    def queue_request(self, instance, requests: dict[str, ArrayRequest]):
        """Enter a request into the queue for a stream group.

        Adds the request to a pending request dict for a whole stream group,
        so that multiple components can contribute to a single request and
        the whole group can be "chunked" together.

        Parameters
        ----------
        instance: object
            The instance making the request
        requests:
            A dictionary of the same form accepted by process_requests
        """
        self._check_requests(requests)
        stream_group = self.get_stream_group(instance)
        if self._queued_allocations.get(stream_group) is None:
            self._queued_allocations[stream_group] = {}
        instance_id = id(instance)
        self._queued_allocations[stream_group].update({instance_id: requests})

    def chunk_arrays(self,
                     requests: dict[str, ArrayRequest],
                     numchunks: int,
                     axis: str="run"):
        """Divide an array by a number of chunks along a given axis

        Parameters
        ==========
        requests: dict[str, ArrayRequest]
            A dictionary keying a label to an ArrayRequest object.
        numchunks: int
            Factor to divide size by
        axis: str
            An axis label along which to divide the array - for example,
            you might want to batch in time, or batch by runs, depending on
            your system. The string must match a label in _stride_order.

        Returns
        =======
        dict[str, ArrayRequest]
            A new dict of requests with modified shapes.
        """
        chunked_requests = deepcopy(requests)
        for key, request in chunked_requests.items():
            # Divide all "numruns" indices by chunks - numchunks is already
            # conservative (ceiling) rounded, so we take the ceiling of this
            # division to ensure we don't end up with one chunk too many.
            run_index = request.stride_order.index(axis)
            newshape = tuple(
                    int(np.ceil(value / numchunks)) if i == run_index else
                    value for i, value in enumerate(request.shape))
            request.shape = newshape
            chunked_requests[key] = request
        return chunked_requests

    def single_request(self,
                       instance: Union[object, int],
                       requests: dict[str, ArrayRequest],
                       chunk_axis: str="run"):
        """Converts a dictionary of ArrayRequests into allocated arrays

        Accepts a dictionary of ArrayRequests and:

        1) Calculates the memory available to the instance
        2) Divides the request into a number of "chunks" of the size of the
        currently available memory
        3) Divides the requested sizes into chunks
        4) Calculates strides for the array to achieve the memory layout
        dictated by MemoryManager._stride_ordering.
        5) Allocates arrays of requested memory types
        6) Calls the instance's allocation_ready_hook with an ArrayResponse
        containing the arrays and chunk count

        Arguments
        =========
        instance: Union[object, int]
            The instance (or instance id) for which the request is being
            processed. This is used to determine the memory available to the
            instance, to assign the arrays to the correct stream, and to
            record the allocation.
        requests: dict[str, ArrayRequest]
            A dictionary of ArrayRequest objects, where the keys are labels
            for the requests (to return the arrays under so that the caller
            can sort them) and the values are the ArrayRequest objects.
        chunk_axis: str
            An axis label along which to divide the array - for example,
            you might want to batch in time, or batch by runs, depending on
            your system. The string must match a label in _stride_order.

        Returns
        =======
        None
            This method does not return a value. Instead, it calls the
            allocation_ready_hook for the instance with an ArrayResponse
            object containing the allocated arrays and chunk count.

        Raises
        ======
        TypeError
            If requests is not a dict, or the values of a request are not
            ArrayRequests.
        """
        self._check_requests(requests)
        if isinstance(instance, int):
            instance_id = instance
        else:
            instance_id = id(instance)

        request_size= get_total_request_size(requests)
        available_memory = self.get_available_single(id(instance))
        numchunks = self.get_chunks(request_size, available_memory)
        chunked_requests = self.chunk_arrays(
                requests, numchunks, axis=chunk_axis)

        arrays = self.allocate_all(
            chunked_requests,
            instance_id,
            self.get_stream(instance)
        )
        self.registry[instance_id].allocation_ready_hook(ArrayResponse(
                arr=arrays, chunks=numchunks, chunk_axis=chunk_axis))

    def allocate_queue(self,
                       triggering_instance: object,
                       limit_type: str="group",
                       chunk_axis: str="run"):
        """ Process queued requests for the stream group *instance* is in.

        Parameters
        ==========
        triggering_instance: object
            The instance making the request
        limit_type: str
            Whether to calculate the aggregate limit for the whole group (
            "group"), or to ensure that no individual instance exceeds its
            limit ("instance").
        chunk_axis: str
            An axis label along which to divide the array - for example,
            you might want to batch in time, or batch by runs, depending on
            your system. The string must match a label in _stride_order.

        This function does not return, but calls the allocation_ready_hook
        for each instance registered in the same stream group as *instance*
        with the ArrayResponse object for their request.

        If the memory manager is actively limiting memory, allocations will
        be "chunked" so as not to exceed either the combined "group" limit
        or any individuals "instance" limit, depending on the limit_type
        string.
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
                self.single_request(instance=instance_id,
                     requests=requests_dict,
                     chunk_axis=chunk_axis)
        else:
            if limit_type == "group":
                available_memory = self.get_available_group(stream_group)
                request_size = sum([get_total_request_size(request)
                                    for request in queued_requests.values()])
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
                        requests_dict,
                        numchunks,
                        chunk_axis)
                arrays = self.allocate_all(chunked_request,
                                           instance_id,
                                           stream=stream)
                response = ArrayResponse(arr=arrays,
                                         chunks=numchunks,
                                         chunk_axis=chunk_axis)
                self.registry[instance_id].allocation_ready_hook(response)

            for peer in notaries:
                self.registry[peer].allocation_ready_hook(ArrayResponse(
                        arr={}, chunks=numchunks, chunk_axis=chunk_axis))
        return None

    def to_device(self, instance: object, from_arrays: list, to_arrays: list):
        """Copy values to device array in the groups stream. """
        stream = self.get_stream(instance)
        is_cupy = self._allocator == CuPyAsyncNumbaManager
        with (current_cupy_stream(stream) if is_cupy else
              contextlib.nullcontext()):
            for i, from_array in enumerate(from_arrays):
                cuda.to_device(from_array, stream=stream, to=to_arrays[i])

    def from_device(self,
                    instance: object,
                    from_arrays: list,
                    to_arrays: list):
        """Copy values from device array in the groups stream. """
        stream = self.get_stream(instance)
        is_cupy = self._allocator == CuPyAsyncNumbaManager
        with (current_cupy_stream(stream) if is_cupy else
              contextlib.nullcontext()):
            for i, from_array in enumerate(from_arrays):
                from_array.copy_to_host(to_arrays[i], stream=stream)

    def sync_stream(self, instance):
        stream = self.get_stream(instance)
        stream.synchronize()

def get_total_request_size(request: dict[str, ArrayRequest]):
    """Returns the total size of a request in bytes"""
    return sum(prod(request.shape) * request.dtype().itemsize
               for request in request.values())