import numpy as np

class SystemValues:
    """ A container for numerical values used to specify ODE systems, such as initial state
    values, parameters, and observables (auxiliary variables).

    This is just a fancy dictionary with some tricks for interacting with the CUDA machinery.
    Set it up with a dictionary of values (just the ones that you don't want to be the default),
    a dictionary of default values (use these if a custom value not set, and a precision). This
    object creates a corresponding array to feed to CUDA functions for compiling, and a dictionary
    of indices to look up that array.

    You can index into this object like a dictionary or an array, i.e. values['key'] or values[index or slice].

    Specify labels and default values (0.0 if not specified) for each value in a dictionary.

    """
    def __init__(self, values_dict, defaults, precision, **kwargs):
        """
        Initialize the system parameters with default values, user-specified values from a dictionary,
        then any keyword arguments. Sets up an array of values and a dictionary mapping parameter names to indices.

        Args:
            values_dict (dict): Dictionary of parameter values to override defaults
            defaults (dict): Dictionary of default parameter values
            precision (numpy.dtype): Data type for the values array (e.g., np.float32, np.float64)
            **kwargs: Additional parameter values that override both defaults and values_dict
        """
        if values_dict is None:
            values_dict = {}


        self.precision = precision
        self.values_array = None
        self.indices_dict = None

        # Instantiate parameters dictionary
        # Handle case where values_dict is a list of strings
        if isinstance(values_dict, (list, tuple)) and all(isinstance(item, str) for item in values_dict):
            values_dict = {k: 0.0 for k in values_dict}
        self.values_dict = {}

        if defaults==None:
            defaults = {}
        # Set detailed values, then overwrite with values provided in dict, then any single-parameter
        # keyword arguments.

        combined_updates = {**defaults, **values_dict, **kwargs}

        # Note: If the same value occurs in the dict and
        # keyword args, the kwargs one will win.
        self.values_dict.update(combined_updates)

        # Initialize values_array and indices_dict
        self.update_param_array_and_indices()

        self.n = len(self.values_array)


    def update_param_array_and_indices(self):
        """
        Extract all values in self.values_dict and save to a numpy array with the specified precision.
        Also create a dict keyed by the values_dict key whose value is the index
        of the parameter in the created array.

        Note: The indices_dict dict will always be in the same order as the values_array,
        as long as the keys are extracted in the same order (insertion order is preserved in Python 3.7+).
        """
        keys = list(self.values_dict.keys())
        self.values_array = np.array([self.values_dict[k] for k in keys], dtype=self.precision)
        self.indices_dict = {k: i for i, k in enumerate(keys)}


    def get_param_index(self, parameter_key):
        """
        Retrieve the index (or indices) of the parameter(s) in the values_array.
        Accepts a single string or a list of strings.
        Raises KeyErrorif a key is not found.
        """
        if isinstance(parameter_key, str):
            if parameter_key in self.indices_dict:
                return self.indices_dict[parameter_key]
            else:
                raise KeyError(f"'{parameter_key}' not found in this SystemValues object. Double check that you're looking " +
                               f"in the right place (i.e. states, or parameters, or constants)")
        elif isinstance(parameter_key, (list, tuple)):
            missing = [k for k in parameter_key if k not in self.indices_dict]
            if missing:
                raise KeyError(f"Parameter key(s) {missing} not found in this SystemValues object. Double check that you're looking " +
                               f"in the right place (i.e. states, or parameters, or constants)")
            return [self.indices_dict[k] for k in parameter_key]
        else:
            raise TypeError("parameter_key must be a string or a list/tuple of strings.")

    def print_param_indices(self):
        """
        Print each parameter and its index, one per line, for terminal readout.
        """
        for key, idx in self.indices_dict.items():
            print(f"{key}: {idx}")

    def get_value(self, key):
        """
        Retrieve the value(s) of the parameter(s) from the values_dict.
        Accepts a single string or a list of strings.
        Raises KeyError if a key is not found.

        Args:
            key: string or list of strings
                The parameter key(s) to retrieve values for

        Returns:
            value: float or list of floats
                The parameter value(s) requested

        Raises:
            KeyError: If the key is not found in the parameters dictionary
            TypeError: If the key is not a string or list/tuple of strings
        """
        if isinstance(key, str):
            if key in self.indices_dict:
                return self.values_dict.get(key)
            else:
                raise KeyError(
                    f"'{key}' not found in this SystemValues object. Double check that you're looking " +
                    f"in the right place (i.e. states, or parameters, or constants)")
        elif isinstance(key, (list, tuple)):
            missing = [k for k in key if k not in self.indices_dict]
            if missing:
                raise KeyError(
                    f"key(s) {missing} not found in this SystemValues object. Double check that you're looking " +
                    f"in the right place (i.e. states, or parameters, or constants)")
            return [self.values_dict.get(k) for k in key]
        else:
            raise TypeError("key must be a string or a list/tuple of strings.")

    def set_values_dict(self, values_dict):
        """
        Update dictionary and values_array with new values
        Updates both the values_dict and the values_array.

        Args:
           values_dict: key-value pairs to update in the values_dict.

        Raises:
            KeyError: If the key is not found in the parameters dictionary
        """
        # Update the dictionary
        missing = [k for k in values_dict.keys() if k not in self.indices_dict]
        if missing:
            raise KeyError(
                f"Parameter key(s) {missing} not found in this SystemValues object. Double check that you're looking " +
                f"in the right place (i.e. states, or parameters, or constants)")
        else:
            self.values_dict.update(values_dict)
            # Update the values_array
            for key, value in values_dict.items():
                index = self.get_param_index(key)
                self.values_array[index] = value

    def __getitem__(self, key):
        """
        Allow dictionary-like and array-like access to the values. Any indexing method will return a value or values only.

        Args:
            key: string, integer, or slice
                If string, the parameter key to retrieve the value for
                If integer, the index in the values_array to retrieve
                If slice, the slice of the values_array to retrieve

        Returns:
            value: The parameter value requested

        Raises:
            KeyError: If the string key is not found in the parameters dictionary
            IndexError: If the integer index is out of bounds
            TypeError: If the key is not a string, integer, or slice
        """
        if isinstance(key, str):
            return self.get_value(key)
        elif isinstance(key, int):
            if 0 <= key < len(self.values_array):
                return self.values_array[key]
            else:
                raise IndexError(f"Index {key} is out of bounds for values_array with length {len(self.values_array)}")
        elif isinstance(key, slice):
            return self.values_array[key]
        else:
            raise TypeError("key must be a string, integer, or slice.")

    def __setitem__(self, key, value):
        """
        Allow dictionary-like and array-like indexing to the values. Both indexing methods will update both the 
        dictionary and the array. 

        Args:
            key: string, integer, or slice
                If string, the parameter key to update
                If integer, the index in the values_array to update
                If slice, the slice of the values_array to update
            value: The new value to set

        Raises:
            KeyError: If the string key is not found in the parameters dictionary
            IndexError: If the integer index is out of bounds
            TypeError: If the key is not a string, integer, or slice
        """
        if isinstance(key, str):
            self.set_values_dict({key: value})
        elif isinstance(key, int):
            if 0 <= key < len(self.values_array):
                # Update the array
                self.values_array[key] = value
                # Find the corresponding key in the dictionary
                for dict_key, idx in self.indices_dict.items():
                    if idx == key:
                        # Update the dictionary
                        self.values_dict[dict_key] = value
                        break
            else:
                raise IndexError(f"Index {key} is out of bounds for values_array with length {len(self.values_array)}")
        elif isinstance(key, slice):
            # Get the indices that would be accessed by this slice
            indices = list(range(*key.indices(len(self.values_array))))

            # Update the array
            self.values_array[key] = value

            # Update the corresponding dictionary entries
            for idx_pos, i in enumerate(indices):
                for dict_key, idx in self.indices_dict.items():
                    if idx == i:
                        # If value is a sequence, use the appropriate element
                        if hasattr(value, '__len__') and not isinstance(value, (str, dict)):
                            # Use the position in the indices list, not the actual index value
                            if idx_pos < len(value):
                                self.values_dict[dict_key] = value[idx_pos]
                        else:
                            # If value is a scalar, use it for all elements
                            self.values_dict[dict_key] = value
                        break
        else:
            raise TypeError("key must be a string, integer, or slice.")
