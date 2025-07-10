from warnings import warn


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
    setup_compile_settings overwrites any existing dictionary, and sets the set of allowed keys in the
    compile_settings dictionary. self.update_compile_settings will update any values of existing keys, and warn the
    user if they attempt to overwrite a non-existing key.

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
        self._compile_settings = {}
        self._cache_valid = True
        self._device_function = None
        self._cached_outputs = {}

    @property
    def cache_valid(self):
        return self._cache_valid

    def setup_compile_settings(self, compile_settings):
        """
        Set the compile settings for the factory. This function should be called to initialize the compile settings,
        and will determine the set of allowed compile settings for the system. New keys can not be added using the
        "update_compile_settings" method.

        This method overwrites the _compile_settings attribute wholesale rather than updating, so calling this on an
        existing instance will overwrite any previously set compile settings.
        """
        if not isinstance(compile_settings, dict):
            raise TypeError("Compile settings must be a dictionary.")
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
        Returns the current compile settings dictionary.
        This dictionary contains the compile settings that are used to build the CUDA device function.
        """
        return self._compile_settings

    def update_compile_settings(self, **kwargs):
        """
        Update the compile settings with new values.
        This method should be called before building the system to ensure that the latest settings are used.
        """
        updates = {}
        for key, value in kwargs.items():
            if key in self.compile_settings:
                updates[key] = value
            else:
                warn(f"Entry {key} was not found in the compile settings dictionary for this object, and so was not "
                     "updated.", UserWarning, stacklevel=2
                     )

        self.compile_settings.update(updates)
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
