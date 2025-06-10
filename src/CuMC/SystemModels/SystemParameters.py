import numpy as np

class SystemParametersDict(dict):
    """A dictionary to hold system parameters, allowing for easy retrieval and
    modification of parameters using descriptively named setter and getter functions.

    Custom methods:
    - set_parameter(key, item): Set the value of a parameter identified by `key`.
    - get_parameter(key): Retrieve the value of a parameter identified by `key`.

    Both methods raise a KeyError if the key is not found in the dictionary.
    """
    def set_parameter(self, key, item):
        if key in self:
            self[key] = item
        else:
            raise KeyError(f"Constant {key} not in parameters dictionary")

    def get_parameter(self, key):
        if key in self:
            return self[key]
        else:
            raise KeyError(f"Constant {key} not in parameters dictionary")


class SystemParameters:
    """ A container for system parameters used in simulations.
    For interfacing with CUDA systems, this class replicates the python-side dictionary
    data, creating an array of values, and a dictionary of names to indices.
    """
    def __init__(self, values_dict, defaults, precision, **kwargs):
        """
        Initialize the system parameters with default values, user-specified values from a dictionary,
        then any keyword arguments. Sets up an array of
        """
        if values_dict is None:
            values_dict = {}

        self.precision = precision
        self.param_array = None
        self.param_indices = None

        # Instantiate parameters dictionary
        self.parameterDict = SystemParametersDict()

        # Set detailed values, then overwrite with values provided in dict, then any single-parameter
        # keyword arguments.
        combined_updates = {**defaults, **values_dict, **kwargs}

        # Note: If the same value occurs in the dict and
        # keyword args, the kwargs one will win.
        self.parameterDict.update(combined_updates)

        # Initialize param_array and param_indices
        self.update_param_array_and_indices()



    def update_param_array_and_indices(self):
        """
        Extract all values in self.parameterDict and save to a numpy array with the specified precision.
        Also create a dict keyed by the parameterDict key whose value is the index
        of the parameter in the created array.

        Note: The param_indices dict will always be in the same order as the param_array,
        as long as the keys are extracted in the same order (insertion order is preserved in Python 3.7+).
        """
        keys = list(self.parameterDict.keys())
        self.param_array = np.array([self.parameterDict[k] for k in keys], dtype=self.precision)
        self.param_indices = {k: i for i, k in enumerate(keys)}


    def get_param_index(self, parameter_key):
        """
        Retrieve the index (or indices) of the parameter(s) in the param_array.
        Accepts a single string or a list of strings.
        Raises KeyErrorif a key is not found.
        """
        if isinstance(parameter_key, str):
            if parameter_key in self.param_indices:
                return self.param_indices[parameter_key]
            else:
                raise KeyError(f"Parameter key '{parameter_key}' not found in param_indices.")
        elif isinstance(parameter_key, (list, tuple)):
            missing = [k for k in parameter_key if k not in self.param_indices]
            if missing:
                raise KeyError(f"Parameter key(s) {missing} not found in param_indices.")
            return [self.param_indices[k] for k in parameter_key]
        else:
            raise TypeError("parameter_key must be a string or a list/tuple of strings.")

    def print_param_indices(self):
        """
        Print each parameter and its index, one per line, for terminal readout.
        """
        for key, idx in self.param_indices.items():
            print(f"{key}: {idx}")
