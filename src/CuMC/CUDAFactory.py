from warnings import warn
import attrs


class CUDAFactory:
    """
    Factory class for creating CUDA-based system models.
    This class has common cache invalidation and update functionality for compile-sensitive parameters.

    Subclasses must implement the "build" method, which is a factory method for the cache contents. This method can
    either return a single CUDA dispatcher, which will then be available through the self.device_function property,
    or a dict of cached outputs (when there are other pieces of information that can be invalidated), each of which
    can then be accessed using the self.get_cached_output(key) method. Apart from the build method, a subclass can include
    whatever non-CUDA support functions are required.

    Each subclass of this class should save compile-sensitive parameters using the setup_compile_settings method
    after calling super().__init__(), and update these using the `update_compile_settings` method.
    self.setup_compile_settings overwrites any existing config object, and sets the allowed attributes for updates.
    self.update_compile_settings will update attributes of the config object..

    The cached elements of this class can take two forms:
        - The _device_function attribute contains the Numba dispatcher (the result of defining a function with the
    @cuda.jit decorator), which is made available using the device_function property.
     - The _cached_outputs dictionary contains multiple cached functions and parameters. If there is a key in that
     dictionary called 'device_function', that will be saved to _device_function as well by default.

     Which of these is used depends on the return type of the subclass's build method (dict or callable).
     If the cache has been marked invalid by an update to the compile settings, the device_function or
     get_cached_output attributes will call self.build() to refill the cache. The current cache validity can be checked
     using the `cache_valid` property, which will return True if the cache is valid and False otherwise.

    Leave underscored methods and attribute alone - all should be modifiable using the non-underscored methods.
    """

    def __init__(self):
        self._compile_settings = None
        self._cache_valid = True
        self._device_function = None
        self._cached_outputs = {}

    @property
    def cache_valid(self):
        return self._cache_valid

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
        Returns the current compile settings class instance. To avoid a mismatch between fetched parameters and built
        functions, this getter will enforce a build if the cache is invalid. There is still a potential cache
        mismatch when doing the following:

        '''
        device_function = self.device_function #calls build if settings updated
        self.update_compile_settings(new_setting=value) #updates settings but does not rebuild
        self.compile_settings.new_setting #fetches updated setting and builds new device functions

        device_function(argument_derived_from_new_setting) #this will use the old device function, not the new one
        '''
        The lesson is: Always use CUDAFactory.device_function at the point of calling, otherwise you'll break the
        cache logic.
        """
        if not self._cache_valid:
            self._build()
        return self._compile_settings

    def update_compile_settings(self, **kwargs):
        """
        Update the compile settings with new values.
        This method should be called before building the system to ensure that the latest settings are used.
        """
        if self._compile_settings is None:
            raise ValueError("Compile settings must be set up using self.setup_compile_settings before updating.")

        # Check if all kwargs are valid attributes of the config
        valid_fields = {field.name for field in attrs.fields(self._compile_settings.__class__)}
        invalid_keys = set(kwargs.keys()) - valid_fields

        if invalid_keys:
            for key in invalid_keys:
                warn(f"Entry {key} was not found in the compile settings for this object, and so was not "
                     "updated.", UserWarning, stacklevel=2)

        # Filter out invalid keys and update attributes directly
        valid_updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        if valid_updates:
            for key, value in valid_updates.items():
                setattr(self._compile_settings, key, value)
            self._invalidate_cache()

    def _invalidate_cache(self):
        self._cache_valid = False

    def build(self):
        """A must-override method for any subclass."""
        return None

    def _build(self):
        """Build or compile the system if needed."""
        build_result = self.build()

        if isinstance(build_result, dict):
            # Multi-output case (for subclasses that need multiple cached values)
            self._cached_outputs.update(build_result)
            # For backward compatibility, if 'device_function' is in the dict
            if 'device_function' in build_result:
                self._device_function = build_result['device_function']
        else:
            self._device_function = build_result

        self._cache_valid = True

    def get_cached_output(self, output_name):
        """
        Get a cached output by name. Will trigger a rebuild if the cache is invalid.
        For subclasses with multiple outputs.
        """
        if not self._cache_valid:
            self._build()
        if output_name == 'device_function':
            return self._device_function
        else:
            if output_name not in self._cached_outputs:
                raise KeyError(f"Output '{output_name}' not found in cached outputs.")
            return self._cached_outputs.get(output_name)
