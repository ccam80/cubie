


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
    def __init__(self, values, defaults, **kwargs):
        """
        Initialize the system parameters with default values, user-specified values from a dictionary,
        then any keyword arguments. Sets up an array of
        """
        if values is None:
            values = {}

        # Instantiate parameters dictionary
        self.parameterDict = SystemParametersDict()

        # Set detailed values, then overwrite with values provided in dict, then any single-parameter
        # keyword arguments.
        combined_updates = {**defaults, **constants_dict, **kwargs}

        # Note: If the same value occurs in the dict and
        # keyword args, the kwargs one will win.
        for key, item in combined_updates.items():
            constants.update(combined_updates)

    return constants