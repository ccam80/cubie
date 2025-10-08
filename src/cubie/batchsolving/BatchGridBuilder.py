"""Batch grid helpers for state and parameter combinations.

This module turns user-supplied dictionaries or arrays into the 2D NumPy
arrays expected by the batch solver. :class:`BatchGridBuilder` is the primary
entry point and is usually accessed through :class:`cubie.batchsolving.solver.Solver`.

Notes
-----
``BatchGridBuilder.__call__`` accepts up to four arguments:

``request``
    Mapping of parameter and state names to value sequences. Provides a
    single dictionary describing all sweep variables.
``params``
    Mapping or array containing parameter values only. One-dimensional
    inputs override defaults for every run, while two-dimensional inputs
    are treated as pre-built grids.
``states``
    Mapping or array containing state values only. Interpretation matches
    ``params``.
``kind``
    Controls how inputs are combined. ``"combinatorial"`` builds the
    Cartesian product, while ``"verbatim"`` preserves row-wise groupings.

When arrays are supplied directly they are treated as fully specified row
sets. Dictionary inputs trigger combinatorial expansion before assembly so
that every value combination is represented in the resulting grid.

Examples
--------
>>> import numpy as np
>>> from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder
>>> from cubie.odesystems.systems.decays import Decays
>>> system = Decays(coefficients=[1.0, 2.0])
>>> grid_builder = BatchGridBuilder.from_system(system)
>>> params = {"p0": [0.1, 0.2], "p1": [10, 20]}
>>> states = {"x0": [1.0, 2.0], "x1": [0.5, 1.5]}
>>> inits, params = grid_builder(
...     params=params, states=states, kind="combinatorial"
... )
>>> print(inits.shape)
(16, 2)
>>> print(inits)
[[1.  0.5]
 [1.  0.5]
 [1.  0.5]
 [1.  0.5]
 [1.  1.5]
 [1.  1.5]
 [1.  1.5]
 [1.  1.5]
 [2.  0.5]
 [2.  0.5]
 [2.  0.5]
 [2.  0.5]
 [2.  1.5]
 [2.  1.5]
 [2.  1.5]
 [2.  1.5]]
>>> print(params.shape)
(16, 2)
>>> print(params)
[[ 0.1 10. ]
 [ 0.1 20. ]
 [ 0.2 10. ]
 [ 0.2 20. ]
 [ 0.1 10. ]
 [ 0.1 20. ]
 [ 0.2 10. ]
 [ 0.2 20. ]
 [ 0.1 10. ]
 [ 0.1 20. ]
 [ 0.2 10. ]
 [ 0.2 20. ]
 [ 0.1 10. ]
 [ 0.1 20. ]
 [ 0.2 10. ]
 [ 0.2 20. ]]

Example 2: verbatim arrays

>>> params = np.array([[0.1, 0.2], [10, 20]])
>>> states = np.array([[1.0, 2.0], [0.5, 1.5]])
>>> inits, params = grid_builder(params=params, states=states, kind="verbatim")
>>> print(inits.shape)
(2, 2)
>>> print(inits)
[[1.  2. ]
 [0.5 1.5]]
>>> print(params.shape)
(2, 2)
>>> print(params)
[[ 0.1  0.2]
 [10.  20. ]]

>>> inits, params = grid_builder(
...     params=params, states=states, kind="combinatorial"
... )
>>> print(inits.shape)
(4, 2)
>>> print(inits)
[[1.  2. ]
 [1.  2. ]
 [0.5 1.5]
 [0.5 1.5]]
>>> print(params.shape)
(4, 2)
>>> print(params)
[[ 0.1  0.2]
 [10.  20. ]
 [ 0.1  0.2]
 [10.  20. ]]

Same as individual dictionaries

>>> request = {
...     "p0": [0.1, 0.2],
...     "p1": [10, 20],
...     "x0": [1.0, 2.0],
...     "x1": [0.5, 1.5],
... }
>>> inits, params = grid_builder(request=request, kind="combinatorial")
>>> print(inits.shape)
(16, 2)
>>> print(params.shape)
(16, 2)

>>> request = {"p0": [0.1, 0.2]}
>>> inits, params = grid_builder(request=request, kind="combinatorial")
>>> print(inits.shape)
(2, 2)
>>> print(inits)  # unspecified variables are filled with defaults from system
[[1. 1.]
 [1. 1.]]
>>> print(params.shape)
(2, 2)
>>> print(params)
[[0.1 2. ]
 [0.2 2. ]]
"""

from itertools import product
from typing import Dict, List, Optional, Union
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.SystemValues import SystemValues


def unique_cartesian_product(arrays: List[np.ndarray]) -> np.ndarray:
    """Return unique combinations of elements from input arrays.

    Each input array can contain duplicates, but the output omits duplicate
    rows while preserving the order of the input arrays.

    Parameters
    ----------
    arrays
        List of one-dimensional NumPy arrays containing elements to combine.

    Returns
    -------
    np.ndarray
        Two-dimensional array in which each row is a unique combination of
        the supplied values.

    Notes
    -----
    Duplicate elements are removed by constructing an ordered dictionary per
    input array. ``itertools.product`` then generates the Cartesian product
    of the deduplicated inputs.

    Examples
    --------
    >>> unique_cartesian_product([np.array([1, 2, 2]), np.array([3, 4])])
    array([[1, 3],
           [1, 4],
           [2, 3],
           [2, 4]])
    """
    deduplicated_inputs = [
        list(dict.fromkeys(a)) for a in arrays
    ]  # preserve order, remove dups
    return np.array([list(t) for t in product(*deduplicated_inputs)])


def combinatorial_grid(
    request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
    values_instance: SystemValues,
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a grid of all unique combinations of requested values.

    Parameters
    ----------
    request
        Dictionary keyed by parameter names or indices whose values are
        scalars or iterables describing sweep values. Value arrays may differ
        in length.
    values_instance
        :class:`SystemValues` instance used to resolve indices for the
        provided keys.
    silent
        When ``True`` suppresses warnings about unrecognised keys.

    Returns
    -------
    tuple of np.ndarray and np.ndarray
        Pair of index and value arrays describing the combinatorial grid.

    Notes
    -----
    Unspecified parameters retain their defaults when the grid is later
    expanded. The number of runs equals the product of all supplied value
    counts.

    Examples
    --------
    >>> combinatorial_grid(
    ...     {"param1": [0.1, 0.2, 0.3], "param2": [10, 20]}, system.parameters
    ... )
    (array([0, 1]),
     array([[ 0.1, 10. ],
            [ 0.1, 20. ],
            [ 0.2, 10. ],
            [ 0.2, 20. ],
            [ 0.3, 10. ],
            [ 0.3, 20. ]]))
    """
    cleaned_request = {
        k: v for k, v in request.items() if np.asarray(v).size > 0
    }
    indices = values_instance.get_indices(
        list(cleaned_request.keys()), silent=silent
    )
    combos = unique_cartesian_product(
        [np.asarray(v) for v in cleaned_request.values()],
    )
    return indices, combos


def verbatim_grid(
    request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
    values_instance: SystemValues,
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a grid that aligns parameter rows without combinatorial expansion.

    Parameters
    ----------
    request
        Dictionary keyed by parameter names or indices whose values are
        scalars or iterables describing sweep values.
    values_instance
        :class:`SystemValues` instance used to resolve indices for the
        provided keys.
    silent
        When ``True`` suppresses warnings about unrecognised keys.

    Returns
    -------
    tuple of np.ndarray and np.ndarray
        Pair of index and value arrays describing the row-wise grid.

    Notes
    -----
    All value arrays must share the same length so rows stay aligned.

    Examples
    --------
    >>> verbatim_grid(
    ...     {"param1": [0.1, 0.2, 0.3], "param2": [10, 20, 30]},
    ...     system.parameters,
    ... )
    (array([0, 1]),
     array([[ 0.1, 10. ],
            [ 0.2, 20. ],
            [ 0.3, 30. ]]))
    """
    cleaned_request = {
        k: v for k, v in request.items() if np.asarray(v).size > 0
    }
    indices = values_instance.get_indices(
        list(cleaned_request.keys()), silent=silent
    )
    combos = np.asarray([item for item in cleaned_request.values()]).T
    return indices, combos


def generate_grid(
    request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
    values_instance: SystemValues,
    kind: str = "combinatorial",
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a parameter grid for batch runs from a request dictionary.

    Parameters
    ----------
    request
        Dictionary keyed by parameter names or indices whose values are
        scalars or iterables describing sweep values.
    values_instance
        :class:`SystemValues` instance used to resolve indices for the
        provided keys.
    kind
        Strategy used to assemble the grid. ``"combinatorial"`` expands all
        combinations while ``"verbatim"`` preserves row groupings.
    silent
        When ``True`` suppresses warnings about unrecognised keys.

    Returns
    -------
    tuple of np.ndarray and np.ndarray
        Pair of index and value arrays describing the generated grid.

    Notes
    -----
    ``kind`` selects between :func:`combinatorial_grid` and
    :func:`verbatim_grid`.
    """
    if kind == "combinatorial":
        return combinatorial_grid(request, values_instance, silent=silent)
    elif kind == "verbatim":
        return verbatim_grid(request, values_instance, silent=silent)
    else:
        raise ValueError(
            f"Unknown grid type '{kind}'. Use 'combinatorial' or 'verbatim'."
        )


def combine_grids(
    grid1: np.ndarray, grid2: np.ndarray, kind: str = "combinatorial"
) -> tuple[np.ndarray, np.ndarray]:
    """Combine two grids according to the requested pairing strategy.

    Parameters
    ----------
    grid1
        First grid, typically parameters.
    grid2
        Second grid, typically initial states.
    kind
        ``"combinatorial"`` builds the Cartesian product and
        ``"verbatim"`` pairs rows directly.

    Returns
    -------
    tuple of np.ndarray and np.ndarray
        Extended versions of ``grid1`` and ``grid2`` aligned to the chosen
        strategy.

    Raises
    ------
    ValueError
        Raised when ``kind`` is ``"verbatim"`` and the inputs have different
        row counts or when ``kind`` is unknown.
    """
    if kind == "combinatorial":
        # Cartesian product: all combinations of rows from each grid
        g1_repeat = np.repeat(grid1, grid2.shape[0], axis=0)
        g2_tile = np.tile(grid2, (grid1.shape[0], 1))
        return g1_repeat, g2_tile
    elif kind == "verbatim":
        if grid1.shape[0] != grid2.shape[0]:
            raise ValueError(
                "For 'verbatim', both grids must have the same number of rows."
            )
        return grid1, grid2
    else:
        raise ValueError(
            f"Unknown grid type '{kind}'. Use 'combinatorial' or 'verbatim'."
        )


def extend_grid_to_array(
    grid: np.ndarray,
    indices: np.ndarray,
    default_values: np.ndarray,
) -> np.ndarray:
    """Join a grid with defaults to create complete parameter arrays.

    Parameters
    ----------
    grid
        Two-dimensional array of gridded parameter values.
    indices
        One-dimensional array describing which parameter indices were swept.
    default_values
        One-dimensional array of default parameter values.

    Returns
    -------
    np.ndarray
        Two-dimensional array containing complete parameter rows for each
        sweep entry.

    Raises
    ------
    ValueError
        Raised when ``grid`` column count does not match ``indices`` length.
    """
    if grid.ndim == 1:
        array = default_values[np.newaxis, :]
    else:
        if grid.shape[1] != indices.shape[0]:
            raise ValueError("Grid shape does not match indices shape.")
        array = np.vstack([default_values] * grid.shape[0])
        array[:, indices] = grid

    return array


def generate_array(
    request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
    values_instance: SystemValues,
    kind: str = "combinatorial",
) -> np.ndarray:
    """Create a complete two-dimensional array from a request dictionary.

    Parameters
    ----------
    request
        Dictionary keyed by parameter names or indices whose values are
        scalars or iterables describing sweep values.
    values_instance
        :class:`SystemValues` instance used to resolve indices for the
        provided keys.
    kind
        Strategy used to assemble the grid. ``"combinatorial"`` expands all
        combinations while ``"verbatim"`` preserves row groupings.

    Returns
    -------
    np.ndarray
        Two-dimensional array whose rows contain complete parameter values
        for each run.
    """
    indices, grid = generate_grid(request, values_instance, kind=kind)
    return extend_grid_to_array(grid, indices, values_instance.values_array)


class BatchGridBuilder:
    """Build grids of parameter and state values for batch runs.

    The builder converts dictionaries or arrays into the solver-ready
    two-dimensional arrays used when planning batch integrations.

    Parameters
    ----------
    interface
        System interface containing parameter and state metadata.

    Attributes
    ----------
    parameters
        Parameter metadata sourced from ``interface``.
    states
        State metadata sourced from ``interface``.
    """

    def __init__(self, interface: SystemInterface):
        """Initialise the builder with a system interface."""
        self.parameters = interface.parameters
        self.states = interface.states

    @classmethod
    def from_system(cls, system: BaseODE) -> "BatchGridBuilder":
        """Create a builder from a system model.

        Parameters
        ----------
        system
            System model providing parameter and state metadata.

        Returns
        -------
        BatchGridBuilder
            Builder configured for ``system``.
        """
        interface = SystemInterface.from_system(system)
        return cls(interface)

    def grid_arrays(
        self,
        request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
        kind: str = "combinatorial",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build parameter and state grids from a mixed request dictionary.

        Parameters
        ----------
        request
            Dictionary keyed by parameter or state identifier whose values
            describe sweep entries.
        kind
            Strategy used to assemble the grid. ``"combinatorial"`` expands
            all combinations while ``"verbatim"`` preserves row groupings.

        Returns
        -------
        tuple of np.ndarray and np.ndarray
            Initial state and parameter arrays aligned for batch execution.
        """
        param_request = {
            k: np.asarray(v)
            for k, v in request.items()
            if k in self.parameters.names
        }
        state_request = {
            k: np.asarray(v)
            for k, v in request.items()
            if k in self.states.names
        }

        params_array = generate_array(
            param_request, self.parameters, kind=kind
        )
        initial_values_array = generate_array(
            state_request, self.states, kind=kind
        )
        initial_values_array, params_array = combine_grids(
            initial_values_array, params_array, kind=kind
        )

        return initial_values_array, params_array

    def __call__(
        self,
        request: Optional[
            Dict[str, Union[float, ArrayLike, np.ndarray]]
        ] = None,
        params: Optional[Union[Dict, ArrayLike]] = None,
        states: Optional[Union[Dict, ArrayLike]] = None,
        kind: str = "combinatorial",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process user input to generate parameter and state arrays.

        Parameters
        ----------
        request
            Optional dictionary keyed by variable name containing a combined
            request for parameters and initial values.
        params
            Optional dictionary or array describing parameter sweeps. A
            one-dimensional array overrides defaults for every run.
        states
            Optional dictionary or array describing initial state sweeps. A
            one-dimensional array overrides defaults for every run.
        kind
            Strategy used to assemble the grid. ``"combinatorial"`` expands
            all combinations while ``"verbatim"`` preserves row groupings.

        Returns
        -------
        tuple of np.ndarray and np.ndarray
            Initial state and parameter arrays aligned for batch execution.

        Notes
        -----
        Passing ``params`` and ``states`` as arrays treats each as a complete
        grid. ``kind="combinatorial"`` computes the Cartesian product of both
        grids, matching the behaviour of a combined request dictionary. When
        arrays already describe paired rows, set ``kind`` to ``"verbatim"`` to
        keep them aligned.
        """
        parray = None
        sarray = None
        if request is not None:
            if states is not None or params is not None:
                raise TypeError(
                    "If a mixed request dictionary is provided, "
                    "states and params requests must be None."
                    "Check that you've input your arguments "
                    "correctly, using keywords for params and "
                    "inits, if you were not trying to provide a "
                    "mixed request dictionary."
                )
            if not isinstance(request, dict):
                raise TypeError(
                    "If provided, a combined request must be provided "
                    f"as a dictionary, got {type(request)}."
                )
            return self.grid_arrays(request, kind=kind)
        else:
            request = {}
            if isinstance(params, dict):
                request.update(params)
            elif isinstance(params, (list, tuple, np.ndarray)):
                parray = self._sanitise_arraylike(params, self.parameters)
            elif params is not None:
                raise TypeError(
                    "Parameters must be provided as a dictionary, "
                    "or a 1D or 2D array-like object."
                )
            if isinstance(states, dict):
                request.update(states)
            elif isinstance(states, (list, tuple, np.ndarray)):
                sarray = self._sanitise_arraylike(states, self.states)
            elif states is not None:
                raise TypeError(
                    "Initial states must be provided as a dictionary, "
                    "or a 1D or 2D array-like object."
                )

            if parray is not None and sarray is not None:
                return combine_grids(sarray, parray, kind=kind)
            elif request:
                if parray is not None:
                    sarray = generate_array(request, self.states, kind=kind)
                    return combine_grids(sarray, parray, kind=kind)
                elif sarray is not None:
                    parray = generate_array(
                        request, self.parameters, kind=kind
                    )
                    return combine_grids(sarray, parray, kind=kind)
                else:
                    return self.grid_arrays(request, kind=kind)
            elif parray is not None:
                sarray = np.full(
                    (parray.shape[0], self.states.n), self.states.values_array
                )
                return sarray, parray
            elif sarray is not None:
                parray = np.full(
                    (sarray.shape[0], self.parameters.n),
                    self.parameters.values_array,
                )
                return sarray, parray
            else:
                return (
                    self.states.values_array[np.newaxis, :],
                    self.parameters.values_array[np.newaxis, :],
                )

    def _trim_or_extend(
        self, arr: np.ndarray, values_object: SystemValues
    ) -> np.ndarray:
        """Extend incomplete arrays with defaults or trim extra values.

        Parameters
        ----------
        arr
            Array requiring adjustment.
        values_object
            System values object containing defaults and dimension metadata.

        Returns
        -------
        np.ndarray
            Array whose column count matches ``values_object.n``.
        """
        if arr.shape[1] < values_object.n:
            # If the array is shorter than the number of values, extend it
            # with default values
            arr = np.pad(
                arr,
                ((0, 0), (0, values_object.n - arr.shape[1])),
                mode="constant",
                constant_values=values_object.values_array[arr.shape[1] :],
            )
        elif arr.shape[1] > values_object.n:
            arr = arr[: values_object.n]
        return arr

    def _sanitise_arraylike(
        self, arr: Optional[ArrayLike], values_object: SystemValues
    ) -> Optional[np.ndarray]:
        """Convert array-likes to 2D arrays and ensure expected dimensions.

        Parameters
        ----------
        arr
            Array-like data describing sweep values.
        values_object
            System values object containing defaults and dimension metadata.

        Returns
        -------
        Optional[np.ndarray]
            Two-dimensional array sized to ``values_object`` or ``None`` when
            no data remain after sanitisation.

        Raises
        ------
        ValueError
            Raised when the input has more than two dimensions.

        Warns
        -----
        UserWarning
            Warned when the number of provided columns differs from the
            expected dimension.
        """
        if arr is None:
            return arr
        elif not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        if arr.ndim > 2:
            raise ValueError(
                f"Input must be a 1D or 2D array, but got a {arr.ndim}D array."
            )
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :]

        if arr.shape[1] != values_object.n:
            warn(
                f"Provided input data has {arr.shape[1]} columns, but there "
                f"are {values_object.n} settable values. Missing values "
                f"will be filled with default values, and extras ignored."
            )
            arr = self._trim_or_extend(arr, values_object)
        if arr.size == 0:
            return None

        return arr  # correctly sized array just falls through untouched

    # ------------------------------------------------------------------
    # Static convenience wrappers
    # ------------------------------------------------------------------
    # These wrappers mirror the module-level helper functions so that when
    # the package re-exports the ``BatchGridBuilder`` *class* under the same
    # name as this module (via ``cubie.batchsolving.__init__``), an import
    # like ``import cubie.batchsolving.BatchGridBuilder as batchgridmodule``
    # that unexpectedly resolves to the class (name shadowing) will still
    # provide access to the expected helper functions used in tests.
    #
    # Keeping the original module-level functions preserves backward
    # compatibility and avoids duplicating logic.

    @staticmethod
    def unique_cartesian_product(arrays: List[np.ndarray]) -> np.ndarray:  # type: ignore[override]
        return unique_cartesian_product(arrays)

    @staticmethod
    def combinatorial_grid(
        request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
        values_instance: SystemValues,
        silent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return combinatorial_grid(request, values_instance, silent=silent)

    @staticmethod
    def verbatim_grid(
        request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
        values_instance: SystemValues,
        silent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return verbatim_grid(request, values_instance, silent=silent)

    @staticmethod
    def generate_grid(
        request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
        values_instance: SystemValues,
        kind: str = "combinatorial",
        silent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return generate_grid(request, values_instance, kind=kind, silent=silent)

    @staticmethod
    def combine_grids(
        grid1: np.ndarray, grid2: np.ndarray, kind: str = "combinatorial"
    ) -> tuple[np.ndarray, np.ndarray]:
        return combine_grids(grid1, grid2, kind=kind)

    @staticmethod
    def extend_grid_to_array(
        grid: np.ndarray, indices: np.ndarray, default_values: np.ndarray
    ) -> np.ndarray:
        return extend_grid_to_array(grid, indices, default_values)

    @staticmethod
    def generate_array(
        request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
        values_instance: SystemValues,
        kind: str = "combinatorial",
    ) -> np.ndarray:
        return generate_array(request, values_instance, kind=kind)
