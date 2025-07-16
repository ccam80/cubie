from warnings import warn
import attrs
from CuMC._utils import in_attr, is_attrs_class

class CUDAFactory:
    """
    Factory class for creating CUDA device functions and kernels in Numba. This parent class has common cache
    invalidation and update functionality for parameters that affect the compiled CUDA functions.

    Each CUDAFactory subclass should implement a "build()" factory method that declares and returns the CUDA function.
    An additional class, decorated with @attrs.define, should be created that holds all of the data that is used to
    configure the CUDA function. The CUDAFactory monitors any changes to the compile settings class, and invalidates
    the cache when these have changed. If a cached object (like the device function) is requested but the cache is
    invalid, the object will be built again before returning.

    The build() method can either return a single CUDA dispatcher, which will then be available through the
    self.device_function property, or another @attrs class which only holds the cached outputs (when there are other
    pieces of information that can be invalidated, such as memory sizes), each of which can then be safely accessed
    using the self.get_cached_output(key) method. Apart from the build method, a subclass can include whatever
    non-CUDA support functions are required.

    attrs classes are used to hold compile settings and cached output to avoid the mutability of keys that come with
    using dicts, according to the sage advice from attrs (self-promotion I guess, but I like it!)
        https://www.attrs.org/en/stable/why.html#dicts:~:text=Dictionaries%20are%20not%20for%20fixed%20fields.

    Instantiate your compile_settings class and pass it to the self.setup_compile_settings() method in init,
    then handle any updates to it using the self.update_compile_settings() method. This method will check if the
    attributes requested are in the compile settings class, and will raise a warning if not. Once it's done it's
    updating work, it will invalidate the cache. There is still a potential cache mismatch when doing the following:

        '''
        device_function = self.device_function #calls build if settings updated
        self.update_compile_settings(new_setting=value) #updates settings but does not rebuild
        self.compile_settings.new_setting #fetches updated setting and builds new device functions

        device_function(argument_derived_from_new_setting) #this will use the old device function, not the new one
        '''

    The lesson is: Always use CUDAFactory.device_function at the point of use, otherwise you'll break the
    cache logic.

    If your build function returns multiple cached items, create a cache class decorated with @attrs.define. For
    example:
    ```python
    @attrs.define
    class MyCache:
        device_function: callable
        other_output: int
    ```
    Then, in your build method, return an instance of this class:
    '''python

    def build(self):
        return MyCache(device_function=my_device_function,
                        other_output=42)

    ```python

    The current cache validity can be checked using the `cache_valid` property, which will return True if the cache
     is valid and False otherwise.

    Leave underscored methods and attribute alone - all should be modifiable using the non-underscored methods.
    """

    def __init__(self):
        self._compile_settings = None
        self._cache_valid = True
        self._device_function = None
        self._cache = None

    def build(self):
        """A must-override method for any subclass."""
        return None

    def setup_compile_settings(self, compile_settings):
        """
        Set the compile settings for the factory. This function should be called to initialize the compile settings.
        The compile_settings parameter should be an attrs class instance.

        This method overwrites the _compile_settings attribute wholesale rather than updating, so calling this on an
        existing instance will overwrite any previously set compile settings.
        """
        if not attrs.has(compile_settings):
            raise TypeError("Compile settings must be an attrs class instance.")
        self._compile_settings = compile_settings
        self._invalidate_cache()

    @property
    def cache_valid(self):
        return self._cache_valid

    @property
    def device_function(self):
        """
        Returns the compiled CUDA device function.
        If the cache is invalid, it will rebuild the function.
        """
        if not self._cache_valid:
            self._build()
        return self._device_function

    @property
    def compile_settings(self):
        """
        Returns the current compile settings class instance.
        """
        return self._compile_settings

    def update_compile_settings(self, **kwargs):
        """
        Update the compile settings with new values, specified as keyword arguments.
        This method should be called before building the system to ensure that the latest settings are used.
        """
        if self._compile_settings is None:
            raise ValueError("Compile settings must be set up using self.setup_compile_settings before updating.")
        
        update_successful = False
        for key, value in kwargs.items():
            if in_attr(key, self._compile_settings):
                setattr(self._compile_settings, key, value)
                update_successful = True
            else:
                warn(f"'{key}' is not a valid compile setting for this object, and so was not updated.", 
                     stacklevel=2)

        if update_successful:
            self._invalidate_cache()

    def _invalidate_cache(self):
        self._cache_valid = False

    def _build(self):
        """Build or compile the system if needed."""
        build_result = self.build()

        if is_attrs_class(build_result):
            # Multi-output case (for subclasses that need multiple cached values)
            self._cache = build_result
            # For backward compatibility, if 'device_function' is in the dict, make it an attribute
            if in_attr('device_function', build_result):
                self._device_function = build_result.device_function
        else:
            self._device_function = build_result

        self._cache_valid = True

    def get_cached_output(self, output_name):
        """
        Get a cached output by name. Will trigger a rebuild if the cache is invalid.
        For subclasses with multiple outputs.
        """
        if not self.cache_valid:
            self._build()
        if output_name == 'device_function':
            return self._device_function
        else:
            if not in_attr(output_name, self._cache):
                raise KeyError(f"Output '{output_name}' not found in cached outputs.")
            return getattr(self._cache, output_name)
