"""Base classes for constructing cached CUDA device functions with Numba."""

from hashlib import sha256
from abc import ABC, abstractmethod
from typing import Set, Any, Tuple, Dict

from attrs import define, field, fields, has, Attribute, astuple, asdict
from numpy import (
    array_equal,
    asarray,
    ndarray,
    dtype as np_dtype,
)
from numba import from_dtype

from cubie._utils import (
    in_attr,
    PrecisionDType,
    precision_validator,
    precision_converter,
)
from cubie.cuda_simsafe import from_dtype as simsafe_dtype


def _hash_tuple(input: Tuple) -> str:
    """Serialize a value to a string for hashing.

    Parameters
    ----------
    input
        Tuple to serialize.

    Returns
    -------
    str
        String representation suitable for hashing.
    """
    parts = []
    for value in input:
        if value is None:
            parts.append("None")
        elif isinstance(value, ndarray):
            # Hash array bytes for deterministic result, incorporating shape
            # and dtype
            array_hash = sha256(value.tobytes()).hexdigest()
            parts.append(f"ndarray:{array_hash}")
        else:
            parts.append(str(value))
    combined = "|".join(parts)
    return sha256(combined.encode("utf-8")).hexdigest()


def attribute_is_hashable(attribute: Attribute, value: Any) -> bool:
    """Check if an attribute value is hashable.

    Parameters
    ----------
    attribute
        An attrs field attribute.
    value
        Value of the attribute.

    Returns
    -------
    bool
        True if the value is hashable, False otherwise.

    Notes
    -----
    Only checks the eq flag; it is the user's responsibility to mark
    unhashable objects eq=False. This should be done in Cubie anyway for
    field updates to successfully track changes.
    """
    eq = attribute.eq
    if eq is False:
        return False
    return True


@define
class _CubieConfigBase:
    """Base class for any configuration container which holds session state.
    Contains updating, serialising, and hashing logic."""

    _unhashable_fields: Set[str] = field(
        factory=set, init=False, repr=False, eq=False
    )
    _values_hash: str = field(default="", init=False, repr=False, eq=False)
    _field_map: Dict[str, Attribute] = field(
        factory=dict, init=False, repr=False, eq=False
    )
    _nested_attrs: Set[str] = field(
        factory=set, init=False, repr=False, eq=False
    )

    def __attrs_post_init__(self):
        """Post-initialization to generate initial hash values."""
        field_map = {}
        for fld in fields(type(self)):
            field_map[fld.name] = fld
            if fld.alias is not None:
                field_map[fld.alias] = fld

        self._field_map = field_map
        self._nested_attrs = {
            fld.name for fld in fields(type(self)) if has(fld.type)
        }
        self._unhashable_fields = {
            field for field in fields(type(self)) if field.eq is False
        }
        self._values_hash = self._generate_values_hash()
        from typing import get_origin

        if any(
            (get_origin(fld.type) is dict or fld.type is dict)
            and fld not in self._unhashable_fields
            for fld in field_map.values()
        ):
            raise TypeError(
                "Fields of type 'dict' are not supported in "
                "CUDAFactoryConfig subclasses, as they're not hashable, "
                "cacheable, and their entries are not easily updated by the "
                "update() method. Please create an attrs class for the "
                "compile-critical data you're adding."
            )

    def update(
        self, updates_dict: dict = None, **kwargs
    ) -> Tuple[Set[str], Set[str]]:
        """Update configuration fields with new values.

        Parameters
        ----------
        updates_dict
            Mapping of setting names to new values. Keys should be
            non-underscored field names (e.g., ``"precision"`` not
            ``"_precision"``).
        **kwargs
            Additional settings to update.

        Returns
        -------
        tuple[set[str], set[str]]
            recognized: Names of settings that matched known fields.
            changed: Names of settings whose values were updated.

        Notes
        -----
        Checks field names and field aliases in a single pass. For fields
        with underscore-prefixed names (e.g., ``_precision``), the key in
        updates_dict should be the non-underscored form (``precision``),
        which matches the field's alias. After updates, values_tuple and
        values_hash are regenerated.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        updates_dict.update(kwargs)
        if not updates_dict:
            return set(), set()

        recognized = set()
        changed = set()

        field_map = self._field_map

        for key, value in updates_dict.items():
            fld = field_map.get(key)
            if fld is None:
                continue

            recognized.add(key)
            old_value = getattr(self, fld.name)

            # Determine if value changed, handling arrays
            if isinstance(old_value, ndarray) or isinstance(value, ndarray):
                value_changed = not array_equal(
                    asarray(old_value), asarray(value)
                )
            else:
                value_changed = old_value != value

            if value_changed:
                setattr(self, fld.name, value)
                changed.add(key)

        for name in self._nested_attrs:
            nested_obj = getattr(self, name)

            nested_recognized, nested_changed = nested_obj.update(updates_dict)
            recognized.update(nested_recognized)
            changed.update(nested_changed)

        # Regenerate hash after updates
        if changed:
            self._values_hash = self._generate_values_hash()

        return recognized, changed

    def _generate_values_hash(self) -> str:
        """Generate hash of current Tuple of values from current field values.
        Called automatically after __init__ and update() (only if any fields
        were modified, in the latter case).
        """
        return _hash_tuple(self.values_tuple)

    @property
    def cache_dict(self):
        """Return a dict of all attrs fields without eq=False, for saving
        and loading complete state."""
        return asdict(self, recurse=True, filter=attribute_is_hashable)

    @property
    def values_tuple(self) -> Tuple:
        """Tuple of all attrs field values without eq=False.

        Returns
        -------
        tuple
            Tuple of configuration values representing the current state.
        """
        return astuple(self, recurse=True, filter=attribute_is_hashable)

    @property
    def values_hash(self) -> str:
        """SHA256 hexdigest of the values_tuple.

        Returns
        -------
        str
            64-character hex string representing the configuration state.
        """
        return self._values_hash


@define
class CUDAFactoryConfig(_CubieConfigBase):
    """Base class for CUDAFactory compile settings containers.

    Provides infrastructure for tracking configuration values and computing
    stable hashes for cache key generation. Subclasses should be defined
    with @attrs.define decorator.

    .. warning::

        **All field modifications MUST be done via the :meth:`update` method.**

        Direct attribute assignment (e.g., ``config.field = value``) will
        break cache invalidation and hashing logic. The ``update()`` method
        ensures ``values_tuple`` and ``values_hash`` are regenerated after
        any change, which is required for correct cache key generation.

    Notes
    -----
    The values_tuple and values_hash properties enable efficient cache
    invalidation by comparing configuration states. Fields with eq=False
    are excluded from hashing (typically callables or device functions).
    """

    precision: PrecisionDType = field(
        validator=precision_validator, converter=precision_converter
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def numba_precision(self) -> type:
        """Return the Numba dtype associated with ``precision``."""

        return from_dtype(np_dtype(self.precision))

    @property
    def simsafe_precision(self) -> type:
        """Return the CUDA-simulator-safe dtype for ``precision``."""

        return simsafe_dtype(np_dtype(self.precision))


@define
class CUDADispatcherCache:
    """Base class for CUDAFactory device function Dispatchers."""

    pass


class CUDAFactory(ABC):
    """Factory for creating and caching CUDA device functions.

    Subclasses implement :meth:`build` to construct Numba CUDA device functions
    or other cached outputs. Compile settings are stored as attrs classes and
    any change invalidates the cache to ensure functions are rebuilt when
    needed.

    .. warning::

        **All compile settings modifications MUST be done via
        :meth:`update_compile_settings`.**

        Direct attribute assignment on compile_settings (e.g.,
        ``factory.compile_settings.field = value``) will break cache
        invalidation and hashing logic. The ``update_compile_settings()``
        method ensures the cache is properly invalidated and hash values
        are regenerated.

    Attributes
    ----------
    _compile_settings : attrs class or None
        Current compile settings.
    _cache_valid : bool
        Indicates whether cached outputs are valid.
    _cache : attrs class or None
        Container for cached outputs (CUDADispatcherCache subclass).

    Notes
    -----
    There is potential for a cache mismatch when doing the following:

    ```python
    device_function = self.device_function  # calls build if settings updated
    self.update_compile_settings(new_setting=value)  # updates settings but
    does not rebuild

    device_function(argument_derived_from_new_setting)  # this will use the
    old device function, not the new one
    ```

    The lesson is: Always use CUDAFactory.device_function at the point of
    use, otherwise you'll defeat the cache invalidation logic.

    If your build function returns multiple cached items, create a cache
    class decorated with @attrs.define. For example:
    ```python
    @attrs.define
    class MyCache:
        device_function: callable
        other_output: int
    ```
    Then, in your build method, return an instance of this class:
    ```python
    def build(self):
        return MyCache(device_function=my_device_function, other_output=42)
    ```

    The current cache validity can be checked using the `cache_valid` property,
    which will return True if the cache
    is valid and False otherwise.
    """

    def __init__(self):
        """Initialize the CUDA factory.

        Notes
        -----
        Uses the global default time logger from cubie.time_logger.
        Configure timing via solve_ivp(time_logging_level=...) or
        Solver(time_logging_level=...).
        """
        self._compile_settings = None
        self._cache_valid = True
        self._cache = None

    @abstractmethod
    def build(self):
        """Build and return the CUDA device function.

        This method must be overridden by subclasses.

        Returns
        -------
        callable or attrs class
            Compiled CUDA function or container of cached outputs.
        """
        return None

    def setup_compile_settings(self, compile_settings):
        """Attach a container of compile-critical settings to the object.

        Parameters
        ----------
        compile_settings : attrs class
            Settings object used to configure the CUDA function.

        Notes
        -----
        Any existing settings are replaced.
        """
        if not has(compile_settings):
            raise TypeError(
                "Compile settings must be an attrs class instance."
            )
        self._compile_settings = compile_settings
        self._invalidate_cache()

    @property
    def cache_valid(self):
        """bool: ``True`` if cached outputs are up to date."""

        return self._cache_valid

    @property
    def device_function(self):
        """Return the compiled CUDA device function.

        Returns
        -------
        callable
            Compiled CUDA device function.
        """
        return self.get_cached_output("device_function")

    @property
    def compile_settings(self):
        """Return the current compile settings object."""
        return self._compile_settings

    def update_compile_settings(
        self, updates_dict=None, silent=False, **kwargs
    ) -> Set[str]:
        """Update compile settings with new values.

        Parameters
        ----------
        updates_dict : dict, optional
            Mapping of setting names to new values.
        silent : bool, default=False
            Suppress errors for unrecognised parameters.
        **kwargs
            Additional settings to update.

        Returns
        -------
        set[str]
            Names of settings that were successfully updated.

        Raises
        ------
        ValueError
            If compile settings have not been set up.
        KeyError
            If an unrecognised parameter is supplied and ``silent`` is ``False``.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        if self._compile_settings is None:
            raise ValueError(
                "Compile settings must be set up using "
                "self.setup_compile_settings before updating."
            )
        recognized, changed = self._compile_settings.update(updates_dict)

        unrecognised = set(updates_dict.keys()) - recognized
        if unrecognised and not silent:
            invalid = ", ".join(sorted(unrecognised))
            raise KeyError(
                f"'{invalid}' is not a valid compile setting for this "
                "object, and so was not updated.",
            )
        if changed:
            self._invalidate_cache()

        return recognized

    def _invalidate_cache(self):
        """Mark cached Dispatchers as invalid."""
        self._cache_valid = False

    def _build(self):
        """Rebuild cached outputs if they are invalid."""
        build_result = self.build()

        if not isinstance(build_result, CUDADispatcherCache):
            raise TypeError(
                "build() must return an attrs class (CUDADispatcherCache "
                "subclass)"
            )

        self._cache = build_result
        self._cache_valid = True

    def get_cached_output(self, output_name):
        """Return a named cached output.

        Parameters
        ----------
        output_name : str
            Name of the cached item to retrieve.

        Returns
        -------
        Any
            Cached value associated with ``output_name``.

        Raises
        ------
        KeyError
            If ``output_name`` is not present in the cache.
        NotImplementedError
            If a cache has been filled with a "-1" integer, this indicates
            that the requested object is not implemented in the subclass.
        """
        if not self.cache_valid:
            self._build()
        if self._cache is None:
            raise RuntimeError("Cache has not been initialized by build().")
        if not in_attr(output_name, self._cache):
            raise KeyError(
                f"Output '{output_name}' not found in cached outputs."
            )
        cache_contents = getattr(self._cache, output_name)
        if type(cache_contents) is int and cache_contents == -1:
            raise NotImplementedError(
                f"Output '{output_name}' is not implemented in this class."
            )
        return cache_contents

    @property
    def config_hash(self):
        """Returns the hash of the current compile settings of the factory.
        If the factory has child factories, their hashes will be hashed and
        the individual hashes will be concatenated and re-hashed.
        """
        own_hash = self.compile_settings.values_hash
        child_hashes = tuple()
        for child_factory in self._iter_child_factories():
            #
            child_hashes = child_hashes + (child_factory.config_hash,)
        if child_hashes:
            # Combine all nested hashes and re-hash
            hash_str = own_hash.join(child_hashes)
            hash_str = own_hash.join(child_hashes)
            return sha256(hash_str.encode("utf-8")).hexdigest()
        else:
            return own_hash

    def _iter_child_factories(self):
        """Yield direct attribute values that are CUDAFactory instances.

        Only inspects immediate attributes (no nested attrs/dicts/iterables).
        Each child is yielded once (uniqueness by id). Attributes are sorted
        alphabetically by name for deterministic ordering.
        """
        seen = set()
        for val in sorted(vars(self).keys()):
            if isinstance(val, CUDAFactory):
                oid = id(val)
                if oid not in seen:
                    seen.add(oid)
                    yield val

    @property
    def precision(self) -> type:
        """Return the precision dtype used by compiled device functions."""
        return self.compile_settings.precision

    @property
    def numba_precision(self) -> type:
        """Return the Numba dtype used by compiled device functions."""

        return self.compile_settings.numba_precision

    @property
    def simsafe_precision(self) -> type:
        """Return the CUDA-simulator-safe dtype for the functions."""

        return self.compile_settings.simsafe_precision
