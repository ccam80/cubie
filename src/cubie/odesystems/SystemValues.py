from collections.abc import (
    Sized,
)

import numpy as np
from sympy import Symbol

class SystemValues:
    """A container for numerical values used to specify ODE systems.

    A container for numerical values such as initial state values, parameters,
    and observables (auxiliary variables). This object creates a corresponding
    array to feed to CUDA functions for compiling, and a dictionary of indices
    to look up that array.

    Parameters
    ----------
    values_array : np.ndarray
        Array containing the numerical values.
    indices_dict : dict
        Dictionary mapping parameter names to array indices.
    keys_by_index : dict
        Dictionary mapping array indices to parameter names.
    values_dict : dict
        Dictionary containing parameter name-value pairs.
    precision : np.dtype
        Data type for the values array.

    Notes
    -----
    If given a list of strings, it will create a dictionary with those strings
    as keys and 0.0 as the default value.

    You can index into this object like a dictionary or an array, i.e.
    values['key'] or values[index or slice].
    """

    values_array: np.ndarray
    indices_dict: dict
    keys_by_index: dict
    values_dict: dict
    precision: np.dtype

    def __init__(
        self,
        values_dict,
        precision,
        defaults=None,
        name=None,
        **kwargs,
    ):
        """Initialize the system parameters.

        Initialize the system parameters with default values, user-specified
        values from a dictionary, then any keyword arguments. Sets up an array
        of values and a dictionary mapping parameter names to indices.

        Parameters
        ----------
        values_dict : dict or list of str
            Full dictionary of parameter values, or dictionary of a subset
            of parameter values for use with a second dictionary of default
            values. This argument can also be a list of strings, in which case
            it will create a dictionary with those strings as keys.
        precision : numpy.dtype
            Data type for the values array (e.g., np.float32, np.float64).
        defaults : dict, optional
            Dictionary of default parameter values, if you're only updating
            a subset.
        **kwargs : dict
            Additional parameter values that override both defaults and
            values_dict.

        Notes
        -----
        If the same value occurs in the dict and keyword args, the kwargs one
        will win.
        """

        if np.issubdtype(precision, np.integer) or np.issubdtype(
            precision, np.floating
        ):
            self.precision = precision
        else:
            raise TypeError(
                f"precision must be a numpy dtype, you provided a "
                f"{type(precision)}"
            )

        self.values_array = None
        self.indices_dict = None
        self.keys_by_index = None
        self.values_dict = {}

        if values_dict is None:
            values_dict = {}
        if defaults is None:
            defaults = {}

        if isinstance(values_dict, (list, tuple)):
            values_dict = {k: 0.0 for k in values_dict}

        if isinstance(defaults, (list, tuple)):
            defaults = {k: 0.0 for k in defaults}

        defaults = self._convert_symbol_keys(defaults)
        values_dict = self._convert_symbol_keys(values_dict)

        # Set default values, then overwrite with values provided in values
        # dict, then any single-parameter keyword arguments.
        combined_updates = {**defaults, **values_dict, **kwargs}

        # Note: If the same value occurs in the dict and
        # keyword args, the kwargs one will win.
        self.values_dict.update(combined_updates)

        # Initialize values_array and indices_dict
        self.update_param_array_and_indices()

        self.n = len(self.values_array)
        self.name = name
    def __repr__(self):
        if self.name is None:
            name = "System Values"
        else:
            name = self.name
        if all(val == 0.0 for val in self.values_dict.values()):
            return f"{name}: variables ({list(self.values_dict.keys())})"
        else:
            return f"{name}: ({self.values_dict})"

    def _convert_symbol_keys(self, input_dict):
        """Convert symbol keys to strings for an input dictionary"""
        if not isinstance(input_dict, dict):
            return input_dict
        converted = {}
        for key, value in input_dict.items():
            if isinstance(key, Symbol):
                converted[str(key)] = value
            elif isinstance(key, str):
                converted[key] = value
        return converted

    def update_param_array_and_indices(self):
        """Extract values and create array and indices mapping.

        Extract all values in self.values_dict and save to a numpy array with
        the specified precision. Also create a dict keyed by the values_dict
        key whose value is the index of the parameter in the created array.

        Notes
        -----
        The indices_dict dict will always be in the same order as the
        values_array, as long as the keys are extracted in the same order
        (insertion order is preserved in Python 3.7+).
        """
        keys = list(self.values_dict.keys())
        self.values_array = np.array(
            [self.values_dict[k] for k in keys], dtype=self.precision
        )
        self.indices_dict = {k: i for i, k in enumerate(keys)}
        self.keys_by_index = {i: k for i, k in enumerate(keys)}

    def get_index_of_key(self, parameter_key, silent=False):
        """Retrieve the index of a given key in the values_array.

        Parameters
        ----------
        parameter_key : str
            The parameter key to look up.
        silent : bool, optional
            If True, suppresses KeyError if key is not found, by default False.

        Returns
        -------
        int
            The index of the parameter in the values_array.

        Raises
        ------
        KeyError
            If parameter_key is not found and silent is False.
        TypeError
            If parameter_key is not a string.
        """
        if isinstance(parameter_key, str):
            if parameter_key in self.indices_dict:
                return self.indices_dict[parameter_key]
            else:
                if not silent:
                    raise KeyError(
                        f"'{parameter_key}' not found in this SystemValues"
                        f" object. Double check that you're looking in the"
                        f" right place (i.e. states, or parameters, or "
                        f"constants)",
                    )
        else:
            raise TypeError(
                f"parameter_key must be a string, "
                f"you submitted a {type(parameter_key)}."
            )

    def get_indices(self, keys_or_indices, silent=False):
        """Convert parameter identifiers to array indices.

        Convert from the many possible forms that a user might specify
        parameters (str, int, list of either, array of indices) into an array
        of indices that can be passed to the CUDA functions.

        Parameters
        ----------
        keys_or_indices : str, int, slice, list, or numpy.ndarray
            Parameter identifiers that can be:
            - Single string parameter name
            - Single integer index
            - Slice object
            - List of string parameter names
            - List of integer indices
            - Numpy array of indices
        silent : bool, optional
            Flag determining whether to raise KeyError if a key is not found,
            by default False.

        Returns
        -------
        numpy.ndarray
            Array of parameter indices as np.int16 values.

        Raises
        ------
        KeyError
            If a string key is not found and silent is False.
        IndexError
            If an index is out of bounds.
        TypeError
            If the input type is not supported or mixed types are provided.
        """
        if isinstance(keys_or_indices, list):
            if all(isinstance(item, str) for item in keys_or_indices):
                # A list of strings
                indices = np.asarray(
                    [
                        self.get_index_of_key(state, silent)
                        for state in keys_or_indices
                    ],
                    dtype=np.int16,
                )
            elif all(isinstance(item, int) for item in keys_or_indices):
                # A list of ints
                indices = np.asarray(keys_or_indices, dtype=np.int16)
            else:
                # List contains mixed types or unsupported types
                non_str_int_types = [
                    type(item)
                    for item in keys_or_indices
                    if not isinstance(item, (str, int))
                ]
                if non_str_int_types:
                    raise TypeError(
                        f"When specifying a variable to save or modify, "
                        f"you can provide strings that match the labels,"
                        f" or integers that match the indices - you "
                        f"provided a list containing"
                        f" {non_str_int_types[0]}",
                    )
                else:
                    raise TypeError(
                        "When specifying a variable to save or modify, "
                        "you can provide a list of strings or a list of "
                        "integers, but not a mixed list of both"
                    )

        elif isinstance(keys_or_indices, str):
            # A single string
            indices = np.asarray(
                [self.get_index_of_key(keys_or_indices)], dtype=np.int16
            )
        elif isinstance(keys_or_indices, int):
            # A single int
            indices = np.asarray([keys_or_indices], dtype=np.int16)

        elif isinstance(keys_or_indices, slice):
            # A slice object
            indices = np.arange(len(self.values_array))[
                keys_or_indices
            ].astype(np.int16)

        elif isinstance(keys_or_indices, np.ndarray):
            indices = keys_or_indices.astype(np.int16)

        else:
            raise TypeError(
                f"When specifying a variable to save or modify, you can"
                f" provide strings that match the labels,"
                f" or integers that match the indices - you provided a "
                f"{type(keys_or_indices)}"
            )

        if any(
            index < 0 or index >= len(self.values_array) for index in indices
        ):
            raise IndexError(
                f"One or more indices are out of bounds. Valid indices are"
                f" from 0 to {len(self.values_array) - 1}."
            )

        return indices

    def get_values(self, keys_or_indices):
        """Retrieve parameter values.

        Retrieve the value(s) of the parameter(s) from the values_dict.
        Accepts the same range of input types as get_indices.

        Parameters
        ----------
        keys_or_indices : str, int, list, or numpy.ndarray
            Parameter identifiers that can be:
            - Single string parameter name
            - Single integer index
            - List of string parameter names
            - List of integer indices
            - Numpy array of indices

        Returns
        -------
        float or numpy.ndarray
            The parameter value(s) requested.

        Raises
        ------
        KeyError
            If a string key is not found in the parameters dictionary.
        IndexError
            If an integer index is out of bounds.
        TypeError
            If the input type is not supported.
        """
        indices = self.get_indices(keys_or_indices)
        if len(indices) == 1:
            return np.asarray(
                self.values_array[indices[0]], dtype=self.precision
            )
        return np.asarray(
            [self.values_array[index] for index in indices],
            dtype=self.precision,
        )

    def set_values(self, keys, values):
        """Set parameter values.

        Parameters
        ----------
        keys : str, int, list, or numpy.ndarray
            Parameter identifiers.
        values : float, list, or numpy.ndarray
            Values to set for the specified parameters.

        Raises
        ------
        ValueError
            If the number of keys does not match the number of values.
        """
        indices = self.get_indices(keys)

        # Checks for mismatches between lengths of indices and values
        if len(indices) == 1:
            if isinstance(values, Sized):
                # Check for one key, multiple values
                if len(values) != 1:
                    raise ValueError(
                        "The number of indices does not match the number "
                        "of values provided. "
                    )
                else:
                    updates = {self.keys_by_index[indices[0]]: values[0]}
            else:
                updates = {self.keys_by_index[indices[0]]: values}

        elif not isinstance(values, Sized):
            # Check for two keys, one value
            raise ValueError(
                "The number of indices does not match the number of values"
                " provided. "
            )

        elif len(indices) != len(values):
            raise ValueError(
                "The number of indices does not match the number of values"
                " provided. "
            )
        else:
            updates = {
                self.keys_by_index[index]: value
                for index, value in zip(indices, values)
            }
        self.update_from_dict(updates)

    def update_from_dict(self, values_dict, silent=False, **kwargs):
        """Update dictionary and values_array with new values.

        Updates both the values_dict and the values_array.

        Parameters
        ----------
        values_dict : dict
            Key-value pairs to update in the values_dict.
        silent : bool, optional
            If True, suppresses KeyError if a key is not found in the
            parameters dictionary, by default False.
        **kwargs : dict
            Additional key-value pairs to update.

        Returns
        -------
        set of str
            A set of keys that were successfully updated.

        Raises
        ------
        KeyError
            If the key is not found in the parameters dictionary and
            silent is False.
        TypeError
            If any value in the values_dict cannot be cast to the
            specified precision.
        """
        if values_dict is None:
            values_dict = {}
        if kwargs:
            values_dict.update(kwargs)
        if values_dict == {}:
            return set()

        # Update the dictionary
        unrecognised = [
            k for k in values_dict.keys() if k not in self.indices_dict
        ]
        recognised = {
            k: v for k, v in values_dict.items() if k in self.indices_dict
        }
        if unrecognised:
            if not silent:
                raise KeyError(
                    f"Parameter key(s) {unrecognised} not found in this "
                    f"SystemValues object. Double check that "
                    f"you're looking in the right place (i.e. states"
                    f", or parameters, or constants)",
                )
        if any(
            np.can_cast(value, self.precision) is False
            for value in recognised.values()
        ):
            raise TypeError(
                f"One or more values in the provided dictionary cannot be "
                f"cast to the specified precision {self.precision}. "
                f"Please ensure all values are compatible with this "
                f"precision.",
            )
        else:
            # Update the dictionary
            self.values_dict.update(values_dict)
            # Update the values_array
            for key, value in recognised.items():
                index = self.get_index_of_key(key, silent=silent)
                self.values_array[index] = value

        return set(values_dict.keys()) - set(unrecognised)

    @property
    def names(self):
        """Get parameter names.

        Returns
        -------
        list of str
            List of parameter names.
        """
        return list(self.values_dict.keys())

    def get_labels(self, indices):
        """Get parameter labels corresponding to indices.

        Parameters
        ----------
        indices : list or numpy.ndarray
            Array or list of indices.

        Returns
        -------
        list of str
            List of labels corresponding to the provided indices.

        Raises
        ------
        TypeError
            If indices is not a list or numpy array.
        """
        if isinstance(indices, (list, np.ndarray)):
            return [self.keys_by_index[i] for i in indices]
        else:
            raise TypeError(
                f"indices must be a list or numpy array, you provided a "
                f"{type(indices)}."
            )

    def __getitem__(self, key):
        """Allow dictionary-like and array-like access to values.

        Parameters
        ----------
        key : str, int, or slice
            If string, the parameter key to retrieve the value for.
            If integer, the index in the values_array to retrieve.
            If slice, the slice of the values_array to retrieve.

        Returns
        -------
        float or numpy.ndarray
            The parameter value requested.

        Raises
        ------
        KeyError
            If the string key is not found in the parameters dictionary.
        IndexError
            If the integer index is out of bounds.
        TypeError
            If the key is not a string, integer, or slice.
        """
        return self.get_values(key)

    def __setitem__(self, key, value):
        """Allow dictionary-like and array-like indexing to set values.

        Parameters
        ----------
        key : str, int, or slice
            If string, the parameter key to update.
            If integer, the index in the values_array to update.
            If slice, the slice of the values_array to update.
        value : float, list, or numpy.ndarray
            The new value to set.

        Raises
        ------
        KeyError
            If the string key is not found in the parameters dictionary.
        IndexError
            If the integer index is out of bounds.
        TypeError
            If the key is not a string, integer, or slice.

        Notes
        -----
        Both indexing methods will update both the dictionary and the array.
        """
        self.set_values(key, value)
