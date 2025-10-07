"""Containers for the numerical values used to parameterise ODE systems."""

from collections.abc import Mapping, Sequence, Sized
from typing import Any

import numpy as np
from sympy import Symbol

from cubie._utils import PrecisionDType


class SystemValues:
    """Manage keyed parameter values and their packed array representation.

    Parameters
    ----------
    values_dict
        Dictionary defining parameter values. Lists of parameter names are
        expanded to a dictionary with ``0.0`` defaults.
    precision
        Precision factory applied when creating the packed values array.
    defaults
        Optional dictionary supplying baseline parameter values.
    name
        Display label used in ``repr`` output.
    **kwargs
        Individual parameter overrides applied after ``defaults`` and
        ``values_dict``.

    Notes
    -----
    Dictionary-style and array-style indexing both operate on the packed
    ``values_array`` and backing ``values_dict``.
    """

    values_array: np.ndarray | None
    indices_dict: dict[str, int] | None
    keys_by_index: dict[int, str] | None
    values_dict: dict[str, float]
    precision: PrecisionDType
    n: int
    name: str | None

    def __init__(
        self,
        values_dict: Mapping[str, float] | Sequence[str] | None,
        precision: PrecisionDType,
        defaults: Mapping[str, float] | Sequence[str] | None = None,
        name: str | None = None,
        **kwargs: float,
    ) -> None:
        """Initialise the packed values dictionary and array.

        Parameters
        ----------
        values_dict
            Full parameter dictionary or iterable of parameter names. Names
            are expanded to ``0.0`` defaults before precision coercion.
        precision
            Precision used when materialising ``values_array``.
        defaults
            Baseline parameter dictionary applied before ``values_dict``.
        name
            Friendly identifier displayed by ``repr``.
        **kwargs
            Parameter overrides applied after ``values_dict``.

        Notes
        -----
        Keyword arguments replace identically named entries in
        ``values_dict`` and ``defaults``.
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

    def __repr__(self) -> str:
        """Return a readable summary of the stored parameter values."""

        if self.name is None:
            name = "System Values"
        else:
            name = self.name
        if all(val == 0.0 for val in self.values_dict.values()):
            return f"{name}: variables ({list(self.values_dict.keys())})"
        return f"{name}: ({self.values_dict})"

    def _convert_symbol_keys(self, input_dict: Any) -> Any:
        """Return a dictionary whose keys are converted to strings.

        Parameters
        ----------
        input_dict
            Dictionary potentially keyed by :class:`sympy.Symbol` objects.

        Returns
        -------
        Any
            Dictionary with symbol keys converted to strings, or the original
            ``input_dict`` if it is not a dictionary.
        """
        if not isinstance(input_dict, dict):
            return input_dict
        converted: dict[str, float] = {}
        for key, value in input_dict.items():
            if isinstance(key, Symbol):
                converted[str(key)] = value
            elif isinstance(key, str):
                converted[key] = value
        return converted

    def update_param_array_and_indices(self) -> None:
        """Populate ``values_array`` and index mappings from ``values_dict``.

        Notes
        -----
        The mapping order follows dictionary insertion so lookup indices stay
        aligned with ``values_array``.
        """
        keys = list(self.values_dict.keys())
        self.values_array = np.array(
            [self.values_dict[k] for k in keys], dtype=self.precision
        )
        self.indices_dict = {k: i for i, k in enumerate(keys)}
        self.keys_by_index = {i: k for i, k in enumerate(keys)}

    def get_index_of_key(self, parameter_key: str, silent: bool = False) -> int:
        """Return the array index associated with a parameter name.

        Parameters
        ----------
        parameter_key
            Parameter name to locate within ``indices_dict``.
        silent
            When ``True`` missing keys do not trigger an exception.

        Returns
        -------
        int
            Index of ``parameter_key`` within ``values_array``.

        Raises
        ------
        KeyError
            Raised when ``parameter_key`` is not present and ``silent`` is
            ``False``.
        TypeError
            Raised when ``parameter_key`` is not a string.
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

    def get_indices(
        self,
        keys_or_indices: str | int | slice | list[str | int] | np.ndarray,
        silent: bool = False,
    ) -> np.ndarray:
        """Convert parameter identifiers into packed array indices.

        Parameters
        ----------
        keys_or_indices
            Parameter descriptors supplied as names, indices, slices, or
            sequences of either.
        silent
            When ``True`` missing keys do not trigger an exception.

        Returns
        -------
        numpy.ndarray
            Array of ``np.int16`` indices targeting ``values_array``.

        Raises
        ------
        KeyError
            Raised when a requested name is missing and ``silent`` is
            ``False``.
        IndexError
            Raised when an index is outside the valid range.
        TypeError
            Raised when the provided descriptors are of unsupported or mixed
            types.
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

    def get_values(
        self, keys_or_indices: str | int | list[str | int] | np.ndarray
    ) -> np.ndarray:
        """Return parameter values selected by name or index.

        Parameters
        ----------
        keys_or_indices
            Parameter descriptors accepted by :meth:`get_indices`.

        Returns
        -------
        numpy.ndarray
            Precision-coerced view of the requested parameter values.

        Raises
        ------
        KeyError
            Raised when a requested name is missing.
        IndexError
            Raised when an index is outside the valid range.
        TypeError
            Raised when the descriptors are of unsupported types.
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

    def set_values(
        self,
        keys: str | int | slice | list[str | int] | np.ndarray,
        values: float | Sequence[float] | np.ndarray,
    ) -> None:
        """Assign new values to the selected parameters.

        Parameters
        ----------
        keys
            Parameter descriptors accepted by :meth:`get_indices`.
        values
            Replacement values aligned with ``keys``.

        Raises
        ------
        ValueError
            Raised when the number of values does not match ``keys``.
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

    def update_from_dict(
        self,
        values_dict: Mapping[str, float] | None,
        silent: bool = False,
        **kwargs: float,
    ) -> set[str]:
        """Update stored parameter values from dictionaries.

        Parameters
        ----------
        values_dict
            Dictionary of key-value pairs to apply.
        silent
            When ``True`` missing keys do not trigger an exception.
        **kwargs
            Additional key-value pairs to apply after ``values_dict``.

        Returns
        -------
        set[str]
            Keys successfully updated in the stored values.

        Raises
        ------
        KeyError
            Raised when a key is missing and ``silent`` is ``False``.
        TypeError
            Raised when a value cannot be cast to ``precision``.
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
            self.values_dict.update(recognised)
            # Update the values_array
            for key, value in recognised.items():
                index = self.get_index_of_key(key, silent=silent)
                self.values_array[index] = value

        return set(values_dict.keys()) - set(unrecognised)

    @property
    def names(self) -> list[str]:
        """List of parameter names."""
        return list(self.values_dict.keys())

    def get_labels(self, indices: list[int] | np.ndarray) -> list[str]:
        """Return parameter labels for supplied indices.

        Parameters
        ----------
        indices
            Integer indices referencing ``values_array``.

        Returns
        -------
        list[str]
            Labels corresponding to ``indices``.

        Raises
        ------
        TypeError
            Raised when ``indices`` is not a sequence or array.
        """
        if isinstance(indices, (list, np.ndarray)):
            return [self.keys_by_index[i] for i in indices]
        else:
            raise TypeError(
                f"indices must be a list or numpy array, you provided a "
                f"{type(indices)}."
            )

    def __getitem__(self, key: str | int | slice) -> np.ndarray:
        """Return parameter values using dictionary- or array-style access.

        Parameters
        ----------
        key
            Parameter descriptors accepted by :meth:`get_values`.

        Returns
        -------
        numpy.ndarray
            Precision-coerced parameter values selected by ``key``.

        Raises
        ------
        KeyError
            Raised when ``key`` is a missing name.
        IndexError
            Raised when ``key`` references an invalid index.
        TypeError
            Raised when ``key`` is of an unsupported type.
        """
        return self.get_values(key)

    def __setitem__(
        self,
        key: str | int | slice,
        value: float | Sequence[float] | np.ndarray,
    ) -> None:
        """Update parameter values using dictionary- or array-style access.

        Parameters
        ----------
        key
            Parameter descriptor accepted by :meth:`set_values`.
        value
            Replacement value or sequence aligned with ``key``.

        Raises
        ------
        KeyError
            Raised when ``key`` is a missing name.
        IndexError
            Raised when ``key`` references an invalid index.
        TypeError
            Raised when ``key`` is of an unsupported type.

        Notes
        -----
        Both indexing methods update ``values_dict`` and ``values_array``.
        """
        self.set_values(key, value)
