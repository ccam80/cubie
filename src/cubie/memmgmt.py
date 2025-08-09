from numba import cuda
import attrs
import attrs.validators as val
from attrs import Factory
import cupy
from numba.cuda.cudadrv.driver import Stream, NumbaCUDAMemoryManager
import numpy as np
from typing import Optional

MIN_AUTOPOOL_SIZE = 0.05

# noinspection PyTypeChecker
@attrs.define
class ArrayRequest:
    """ Information required to allocate a CUDA array: shape, dtype, memory"""
    shape: tuple[int, int, int] = attrs.field(
            default=(1, 1, 1),
            validator=val.deep_iterable(val.instance_of(int),
                                        val.instance_of(tuple)))
    dtype: np.dtype = attrs.field(
            default=np.float64,
            validator=val.instance_of(np.dtype))
    memory: [str] = attrs.field(default="device",
                                  validator=val.in_(["device", "mapped",
                                                     "pinned", "managed"]))
    # managed not tested on Windows and will raise NotImplementedError

    @property
    def size(self):
        return np.prod(self.shape) * self.dtype.itemsize

@attrs.define
class ArrayResponse:
    """ Result of an array allocation: an array, and a number of chunks """
    arr: cuda.devicearray.DeviceNDArray = attrs.field()

# These will be keys to a dict, so must be hashable: eq=False
@attrs.define(eq=False)
class InstanceMemorySettings:
    """Memory registry information for a registered class

    Attributes:
    ==========
        proportion: float
            Proportion of total VRAM assigned to this instance
        allocated: int
            Current allocation in bytes for this instance
        _strides_current: bool
            Whether this processes have been reallocated since a strides
            change
        _manual_proportion: bool
            Whether the proportion is manually set or auto-calculated.
            Manual mode might be helpful to manage processes that use a lot
            of memory but can run slowly, freeing up memory for other
            processes.
        _native_stride_order: tuple[str, str, str]
            Native stride ordering for array in memory - cubie uses ("time",
            "run", "variable") by default
        _cap: int
            Maximum allocatable bytes for this instance (set by manager based
            on total VRAM and proportion)
    """
    proportion: float = attrs.field(default=1.0,
                                  validator=val.instance_of(float))

    allocated: int = attrs.field(default=0,
                                 validator=val.instance_of(int))
    strides_current: bool = attrs.field(default=False,
                                         validator=val.instance_of(bool))
    stream_group: str = attrs.field(default="default",
                                    validator=val.instance_of(str))
    _manual_proportion: bool = attrs.field(default=False,
                                           validator=val.instance_of(bool))
    _native_stride_order: Optional[tuple[str, str, str]] = attrs.field(
            default=("time", "run", "variable"))
    _cap: int = attrs.field(default=None,
                            validator=val.optional(
                                       val.instance_of(int)))

    def __attrs_post_init__(self):
        if self._native_stride_order is None:
            self._native_stride_order = ("time", "run", "variable")

@attrs.define
class StreamGroups:
    """Dictionaries which map instances to groups, and groups to a stream"""
    groups: Optional[dict[str, list[object]]] = attrs.field(
            default=Factory(dict),
            validator=val.optional(val.instance_of(dict)))
    streams: dict[str, Stream] = attrs.field(
            default=Factory(dict),
            validator=val.instance_of(dict)
    )

    def __attrs_post_init__(self):
        if self.groups is None:
            self.groups = {'default': []}
        if self.streams is None:
            self.streams = {'default': cuda.default_stream()}

    def add_instance(self, instance, group):
        """Add an instance to a group, and assign it a stream"""
        if group not in self.groups:
            self.groups[group] = []
            self.streams[group] = cuda.stream()
        self.groups[group].append(instance)

    def change_group(self, instance, new_group):
        """Change the group of an instance"""
        old_group = self.groups[instance.stream_group]
        old_group.remove(instance)
        self.add_instance(instance, new_group)

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
    them to a "stream group" when registering."""

    totalmem: int = attrs.field(default=None,
                                validator=val.optional(val.instance_of(int)))
    registered_instances: dict[object, InstanceMemorySettings] = attrs.field(
            default=Factory(dict),
            validator=val.optional(val.instance_of(dict)))
    stream_groups: StreamGroups = attrs.field(default=Factory(StreamGroups))
    _mode: str = attrs.field(default="passive",
                             validator=val.in_(["passive", "active"])
                             )
    _allocator: object = attrs.field(
            default=None,
            validator=val.optional(val.instance_of(object)))
    _stride_ordering: tuple[str, str, str] = attrs.field(
            default=("time", "run", "variable"),
            validator=val.instance_of(tuple))
    _auto_pool: list[object] = attrs.field(
            default=Factory(list),
            validator=val.instance_of(list))
    _manual_pool: list[object] = attrs.field(
            default=Factory(list),
            validator=val.instance_of(list)
    )

    def __attrs_post_init__(self):

        free, total = cuda.current_context().get_memory_info()
        self.totalmem = total
        self.registered_instances = {}
        self.set_allocator("default")

    def set_stride_ordering(self, ordering: tuple[str, str, str]):
        """ Sets the ordering of arrays in memory"""
        if not all(elem in ("time", "run", "variable") for elem in ordering):
            raise ValueError("Invalid stride ordering - must containt 'time', "
                             f"'run', 'variable' but got {ordering}")
        self._stride_ordering = ordering
        for instance_settings in self.registered_instances.values():
            instance_settings._strides_current = False

    def set_allocator(self, name: str):
        """ Set the external memory manager in Numba"""
        if name == "cupy_stream":
            # use CuPy async memory pool
            self._allocator = cupy.cuda.MemoryAsyncPool
        elif name == "cupy":
            self._allocator = cupy.cuda.MemoryPool
        elif name == "default":
            # use numba's default allocator
            self._allocator = NumbaCUDAMemoryManager
        else:
            raise ValueError(f"Unknown allocator: {name}")
        cuda.set_memory_manager(self._allocator)

        # Reset the context:
        # https://nvidia.github.io/numba-cuda/user/external-memory.html#setting-emm-plugin
        # WARNING - this will invalidate all prior streams, arrays, and funcs!
        # CUDA_ERROR_INVALID_CONTEXT or CUDA_ERROR_CONTEXT_IS_DESTROYED
        # suggests you're using an old reference.
        cuda.close()

    def register(self,
                 instance,
                 stream_group: str = "default",
                 proportion: Optional[float] =None,
                 native_stride_order: Optional[tuple[str,str,str]] = None):
        """
        Register an instance with optional manual proportion.
        Returns a CUDA Stream if instance has request_size attribute.
        """
        if proportion:
            manual = True
            if not 0 <= proportion <= 1:
                raise ValueError("Proportion must be between 0 and 1")
            self._add_manual_proportion(instance, proportion)
        else:
            manual = False
            proportion = self._add_auto_proportion(instance)
        cap = int(self.totalmem * proportion)
        self.stream_groups.add_instance(instance, stream_group)

        settings = InstanceMemorySettings(
                proportion=proportion, allocated=0, stream_group=stream_group,
                manual_proportion=manual, cap=cap,
                native_stride_order=native_stride_order, strides_current=False)

        self.registered_instances[instance] = settings

    def set_manual_allocation(self, instance: object, proportion: float):
        """Set manual allocation status for an already-registered instance"""
        settings = self.registered_instances[instance]
        if settings._manual_proportion:
            raise ValueError("Instance is already in manual allocation pool")
        self._auto_pool.remove(instance)
        self._add_manual_proportion(instance, proportion)
        settings.proportion = proportion
        settings._manual_proportion = True

    def set_auto_allocation(self, instance):
        """Sets auto-allocation status for an already-registered instance"""
        settings = self.registered_instances[instance]
        if not settings._manual_proportion:
            raise ValueError("Instance is already in auto allocation pool")
        self._manual_pool.remove(instance)
        settings.proportion = self._add_auto_proportion(instance)
        settings._manual_proportion = False

    def _add_manual_proportion(self, object: object, proportion: float):
        """Adds an instance to the manual pool with a given proportion"""
        new_manual_pool_size = self.manual_pool_proportion + proportion
        if new_manual_pool_size > 1.0:
            raise ValueError("Manual proportion would exceed total "
                             "available memory")
        if len(self._auto_pool) > 0:
            available_auto = 1.0 - new_manual_pool_size
            if available_auto <= MIN_AUTOPOOL_SIZE:
                raise ValueError("Manual proportion would leave less than 5% "
                                 "of memory for auto-allocated processes")
            self._rebalance_auto_pool(available_auto)
        self._manual_pool.append(object)

    def _add_auto_proportion(self, object):
        """Adds an instance to the auto-pool, and return its proportion"""
        new_objects_in_pool = len(self._auto_pool) + 1
        autopool_available = 1.0 - self.manual_pool_proportion
        if autopool_available == 0:
            raise ValueError("All memory has been allocated manually to other "
                             "objects - modify other allocations if you want to "
                             "get this into the GPU")
        new_proportion = autopool_available / new_objects_in_pool
        for settings in self._auto_pool:
            settings.proportion = new_proportion
        self._auto_pool.append(object)
        return new_proportion

    def _rebalance_auto_pool(self, available_proportion):
        """Splits the available portion of VRAM amongst the auto-pool"""
        proportion = available_proportion / len(self._auto_pool)
        for settings in self._auto_pool:
            settings.proportion = proportion

    def check_strides_current(self, instance):
        """Returns False if strides have changed since last allocation"""
        return self.registered_instances.get(instance)._strides_current

    def proportion(self, instance):
        return self.registered_instances[instance].proportion

    def cap(self, instance):
        # return the configured cap in bytes for the instance
        settings = self.registered_instances.get(instance)
        return settings._cap

    @property
    def manual_pool_proportion(self):
        manual_settings = [self.registered_instances[instance] for instance
                           in self._manual_pool]
        pool_proportion = sum([settings.proportion for settings in manual_settings])
        return pool_proportion

    @property
    def auto_pool_proportion(self):
        auto_settings = [self.registered_instances[instance] for instance in
                           self._auto_pool]
        pool_proportion = sum(
                [settings.proportion for settings in auto_settings])
        return pool_proportion

    def allocate(self, request: dict[str, ArrayRequest])  -> (
            dict[str, ArrayResponse]):
        pass

    def get_chunks(self, instance, request_size):
        # determine number of chunks given request size based on global available memory
        free, total = cuda.current_context().get_memory_info()
        settings = self.registered_instances.get(instance)
        cap = settings._cap
        allocated = settings.allocated
        headroom = self.registered_instances.get(instance)._cap - allocated
        available = min(headroom, free)
        return np.ceil(request_size / available)
