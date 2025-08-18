"""Utilities for building grids of state and parameter values.

The :class:`BatchGridBuilder` class converts user-supplied dictionaries or
arrays into the 2D numpy arrays expected by the solver. The user primarily
interacts with this class through the Solver class, unless they have very
specific needs. Interaction with the BatchConfigurator is primarily through
the `__call__` method, which accepts four arguments:

- `request`: A dictionary of (potentially mixed) parameter and state names
mapped to sequences of values. Says: "Integrate from combinations of
these named variables"
- 'params': A dictionary of purely parameter variable names, a 1D sequence to
set one set of values for every integration, or a 2D array of all
combinations to integrate from.
- 'states': A dictionary of purely state variable names, a 1D sequence to
set one set of values for every integration, or a 2D array of all
combinations to integrate from.
- 'kind': A string indicating how to combine the parameters and states.
  - 'combinatorial' constructs a grid of all combinations of the values
  provided
  - 'verbatim' constructs a grid of [[all variables[0]],[all variables[1]],...]
  when the conbinations have been intentionally constructed already.

Examples
--------
''''
import numpy as np
from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder
from cubie.systemmodels.systems.decays import Decays
systm = Decays(coefficients=[1.0, 2.0])
grid_builder = BatchGridBuilder.from_system(system)
params = {'p0': [0.1, 0.2], 'p1': [10, 20]}
states = {'x0': [1.0, 2.0], 'x1': [0.5, 1.5]}
inits, params = grid_builder(params=params, states=states,
                                                kind='combinatorial')
'''
>>>print(inits.shape)
(16, 2)
>>>print(inits)
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
>>>print(params.shape)
(16, 2)
>>>print(params)
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

 '''
 # Example 2: verbatim arrays
params = np.array([[0.1, 0.2], [10, 20]])
states = np.array([[1.0, 2.0], [0.5, 1.5]])
inits, params = grid_builder(params=params, states=states,
                                                kind='verbatim')
'''
>>>print(inits.shape)
(2, 2)
>>>print(inits)
[[1.  2. ]
 [0.5 1.5]]
>>>print(params.shape)
(2, 2)
>>>print(params)
[[ 0.1  0.2]
 [10.  20. ]]


>>>inits, params = grid_builder(params=params, states=states,
                                                kind='combinatorial')

>>>print(inits.shape)
(4, 2)
>>>print(inits)
[[1.  2. ]
 [1.  2. ]
 [0.5 1.5]
 [0.5 1.5]]
>>>print(params.shape)
(4, 2)
>>>print(params)
[[ 0.1  0.2]
 [10.  20. ]
 [ 0.1  0.2]
 [10.  20. ]]

#Same as individual dictionaries
>>>request = {'p0': [0.1, 0.2], 'p1': [10, 20], 'x0': [1.0, 2.0],
           'x1': [0.5, 1.5]}
>>>inits, params = grid_builder(request=request,
                                                kind='combinatorial')
>>>print(inits.shape)
(16, 2)
>>>print(params.shape)
(16, 2)

>>>request = {'p0': [0.1, 0.2]}
>>>inits, params = grid_builder(request=request,
                                                kind='combinatorial')
>>>print(inits.shape)
(2, 2)
>>>print(inits)  # unspecified variables are filled with defaults from system
[[1. 1.]
 [1. 1.]]
>>>print(params.shape)
(2, 2)
>>>print(params)
[[0.1 2. ]
 [0.2 2. ]]

Notes
-----
There is a subtle difference between a 'combinatorial' combination of two
input arrays the same values given as per-variable arrays in a dict,
as demonstrated in the examples above. If the user provides arrays as an
argument, it is assumed that all within-array combinations have already been
constructed. When the user provides a dict, it is assumed that they want a
combinatorial combination of the values. In the array case, the method will
return arrays which contain all combinations of the crows of the arrays (
i.e. nrows x nrows combinations). The dictionary case first constructs
combinations of values, resulting in an array of height nvals1 * nvals2 *
... * nvalsk for k variables, then combines these in the same fashion as
the array case.

For more fine-grained control, you can call the grid_arrays and
combine_grids methods directly, or construct the full arrays outside of
this method and let them pass through verbatim."""


from itertools import product
from typing import List, Union, Dict, Optional
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cubie.systemmodels.SystemValues import SystemValues
from cubie.systemmodels.systems.GenericODE import GenericODE
from cubie.batchsolving.SystemInterface import SystemInterface


def unique_cartesian_product(arrays: List[np.ndarray]):
    """Return a 2D array of each unique combination of elements from a list
    of 1d input arrays.
    Each input array can have duplicates, but the output will not contain
    any duplicate rows.
    The order of the input arrays is preserved, and the output will have the
    same order of elements as the input.
    Parameters
    ----------
    arrays : List[np.ndarray]
        A list of 1D numpy arrays, each containing elements to be combined.
    Returns
    -------
    combos : np.ndarray
        A 2D numpy array where each row is a unique combination of elements
        from the inputs
    Examples
    --------
    >>> unique_cartesian_product([np.array([1, 2, 2]), np.array([3, 4])])
    array([[1, 3],
        [1, 4],
        [2, 3],
        [2, 4]])
    Notes
    -----
    This function removes duplicates by creating a dict with the elements of
    the input array as keys. It then casts
    that to a list, getting the de-duplicated values. It  then uses
    `itertools.product` to generate the Cartesian
    product of  the input arrays.
    ."""
    deduplicated_inputs = [list(dict.fromkeys(a)) for a in
                           arrays]  # preserve order, remove dups
    return np.array([list(t) for t in product(*deduplicated_inputs)])


def combinatorial_grid(
        request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
        values_instance: SystemValues, silent: bool = False, ) -> tuple[
    np.ndarray, np.ndarray]:
    """
    Build a grid of all unique combinations of values based on a dictionary
    keyed by parameter name or index,
    and with values comprising the entire set of parameter values.

    Parameters
    ----------
    request : Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]]
        Dictionary where keys are parameter names or indices, and values are
        either a single value or an array of
        values for that parameter.
        For a combinatorial grid, the arrays of values need not be equal in
        length.
    values_instance: SystemValues
        The SystemValues instance in which to find the indices for the keys
        in the request.
    silent: bool
        If True, suppress warnings about unrecognized parameters in the
        request.

    Returns
    -------
    grid: np.ndarray
        A 2D array of shape (n_runs, n_requested_parameters) where each row
        corresponds to a set of parameters
        for a run.
    indices: np.ndarray
        A 1D array of indices corresponding to the gridded parameters.

    Unspecified parameters are filled with their default values from the
    system. n_runs is the combinatorial
    of the lengths of all of the value types - for example, if the request
    contains two parameters with 3, 2,
    and 4 values, then n_runs would be 3 * 2 * 4 = 24
    Examples
    --------
    ```
    >>> combinatorial_grid({
        'param1': [0.1, 0.2, 0.3],
        'param2': [10, 20]
    }, system.parameters)
    ```
    (array([[ 0.1, 10. ],
           [ 0.1, 20. ],
           [ 0.2, 10. ],
           [ 0.2, 20. ],
           [ 0.3, 10. ],
           [ 0.3, 20. ]]),
         array([0, 1]))
    """
    cleaned_request = {k: v for k, v in request.items() if
                       np.asarray(v).size > 0}
    indices = values_instance.get_indices(list(cleaned_request.keys()),
                                          silent=silent)
    combos = unique_cartesian_product(
            [np.asarray(v) for v in cleaned_request.values()], )
    return indices, combos


def verbatim_grid(
        request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
        values_instance: SystemValues, silent: bool = False, ) -> tuple[
    np.ndarray, np.ndarray]:
    """ Build a grid of parameters for a batch of runs based on a dictionary
    keyed by parameter name or index,
    and values the entire set of parameter values. Parameters vary together,
    but not combinatorially. All values
    arrays must be of equal length.
    Parameters
    ----------
    request : Dict[Union[str, int], Union[float, ArrayLike, NDArray]]
        Dictionary where keys are parameter names or indices, and values are
        either a single value or an array of
        values for that parameter.
    values_instance: SystemValues
        The SystemValues instance in which to find the indices for the keys
        in the request.
    silent: bool
        If True, suppress warnings about unrecognized parameters in the
        request.

    Returns
    -------
    grid: np.ndarray
        A 2D array of shape (n_runs, n_requested_parameters) where each row
        corresponds to a set of parameters
        for a run.
    indices: np.ndarray
        A 1D array of indices corresponding to the gridded parameters.
    Unspecified parameters are filled with their default values from the
    system. n_runs is the length of _all_
    value arrays, which must be equal.
    Examples
    --------
    ```
    >>>verbatim_grid({
        'param1': [0.1, 0.2, 0.3],
        'param2': [10, 20, 30]
    }, system.parameters)
    (array([[ 0.1, 10. ],
           [ 0.2, 20. ],
           [ 0.3, 30. ]]),
           array([0, 1]))

    ```
    """
    cleaned_request = {k: v for k, v in request.items() if
                       np.asarray(v).size > 0}
    indices = values_instance.get_indices(list(cleaned_request.keys()),
                                          silent=silent)
    combos = np.asarray([item for item in cleaned_request.values()]).T
    return indices, combos


def generate_grid(
        request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
        values_instance: SystemValues, kind: str = 'combinatorial',
        silent: bool = False, ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a grid of parameters for a batch of runs based on a dictionary
    keyed by parameter name or index,
    and with values comprising the entire set of parameter values.

    Parameters
    ----------
    request : Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]]
        Dictionary where keys are parameter names or indices, and values are
        either a single value or an array of
        values for that parameter.
    values_instance: SystemValues
        The SystemValues instance in which to find the indices for the keys
        in the request.
    kind : str
        The type of grid to generate. Can be 'combinatorial' or 'verbatim'.
    silent: bool
        If True, suppress warnings about unrecognized parameters in the
        request.

    Returns
    -------
    grid: np.ndarray
        A 2D array of shape (n_runs, n_requested_parameters) where each row
        corresponds to a set of parameters
        for a run.
    indices: np.ndarray
        A 1D array of indices corresponding to the gridded parameters.

    Notes
    -----
    The `kind` parameter determines how the grid is constructed:
    - 'combinatorial': see BatchGridBuilder.combinatorial_grid
    - 'verbatim': see BatchGridBuilder.verbatim_grid
    """
    if kind == 'combinatorial':
        return combinatorial_grid(request, values_instance, silent=silent)
    elif kind == 'verbatim':
        return verbatim_grid(request, values_instance, silent=silent)
    else:
        raise ValueError(
                f"Unknown grid type '{kind}'. Use 'combinatorial' or "
                f"'verbatim'.")


def combine_grids(grid1: np.ndarray, grid2: np.ndarray,
                  kind: str = 'combinatorial') -> tuple[
    np.ndarray, np.ndarray]:
    """
    Combine two grids (e.g., parameter and state grids) into a single grid.

    Parameters
    ----------
    grid1 : np.ndarray
        First grid (e.g., parameter grid).
    grid2 : np.ndarray
        Second grid (e.g., state grid).
    kind : str
        'combinatorial' for cartesian product, 'verbatim' for row-wise pairing.

    Returns
    -------
    np.ndarray, np.ndarray
        Extended grids grid1, grid2
    """
    if kind == 'combinatorial':
        # Cartesian product: all combinations of rows from each grid
        g1_repeat = np.repeat(grid1, grid2.shape[0], axis=0)
        g2_tile = np.tile(grid2, (grid1.shape[0], 1))
        return g1_repeat, g2_tile
    elif kind == 'verbatim':
        if grid1.shape[0] != grid2.shape[0]:
            raise ValueError(
                    "For 'verbatim', both grids must have the same number of "
                    "rows.")
        return grid1, grid2
    else:
        raise ValueError(
                f"Unknown grid type '{kind}'. Use 'combinatorial' or "
                f"'verbatim'.")


def extend_grid_to_array(grid: np.ndarray, indices: np.ndarray,
                         default_values: np.ndarray, ):
    """Join a grid of values with the an array of default values, creating a
    2D array where each row has a full
    set of parameters, and non-gridded parameters are set to their default
    values.
    Parameters
    ----------
    grid : np.ndarray
        A 2D array of shape (n_runs, n_requested_parameters) where each row
        corresponds to a set of parameters for a run.
    indices : np.ndarray
        A 1D array of indices corresponding to the gridded parameters.
    default_values : np.ndarray
        A 1D array of default values for the parameters.
    Returns
    -------
    np.ndarray
        A 2D array of shape (n_runs, n_parameters) where each row
        corresponds to a set of parameters for a run.
        Parameters not specified in the grid are filled with their default
        values from the system.
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
        kind: str = 'combinatorial', ) -> np.ndarray:
    """
    Create a 2D array of requested parameters or states based on a
    dictionary of requests.
    Parameters
    ----------
    request : Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]]
        Dictionary where keys are parameter names or indices, and values are
        either a single value or an array of
        values for that parameter.
    values_instance: SystemValues
        The SystemValues instance in which to find the indices for the keys
        in the request.
    kind : str
        The type of grid to generate. Can be 'combinatorial' or 'verbatim'.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_runs, n_parameters) where each row
        corresponds to a set of parameters
        for a run. Parameters not specified in the request are filled with
        their default values from the system.
    """
    indices, grid = generate_grid(request, values_instance, kind=kind)
    return extend_grid_to_array(grid, indices, values_instance.values_array)


class BatchGridBuilder:
    """Build grids of parameter and state values for batch runs."""

    def __init__(self, interface: SystemInterface):
        self.parameters = interface.parameters
        self.states = interface.states

    @classmethod
    def from_system(cls, system: GenericODE):
        """Create a ``BatchGridBuilder`` from a system model."""
        interface = SystemInterface.from_system(system)
        return cls(interface)

    def grid_arrays(self,
                    request: Dict[Union[str, int],
                                  Union[float, ArrayLike, np.ndarray]],
                    kind: str = 'combinatorial'
                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a grid of parameters for a batch of runs based on a dictionary
        keyed by parameter name or index,
        Parameters
        ----------
        request
        kind

        Returns
        -------
        np.ndarray, np.ndarray
            2d state and parameter arrays for input into the integrator
        """
        param_request = {k: v for k, v in request.items() if
                         k in self.parameters.names}
        state_request = {k: v for k, v in request.items() if
                         k in self.states.names}

        params_array = generate_array(param_request, self.parameters,
                                      kind=kind)
        initial_values_array = generate_array(state_request, self.states,
                                              kind=kind)
        initial_values_array, params_array = combine_grids(
                initial_values_array, params_array, kind=kind)

        return initial_values_array, params_array

    def __call__(
            self,
             request: Optional[Dict[str,
                               Union[float, ArrayLike, np.ndarray]]] = None,
             params: Optional[Union[Dict, ArrayLike]] = None,
             states: Optional[Union[Dict, ArrayLike]] = None,
             kind: str = 'combinatorial'
             ) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes user input to generate parameter and state arrays for a
        batch run.

        This method acts as the main entry point for the user. It accepts
        parameters and initial states in various formats (dictionaries or
        arrays), processes them, and returns two 2D arrays: one for initial
        states and one for parameters, ready to be used in a simulation.

        Parameters
        ----------
        request: Optional[Dict[str, Union[float, ArrayLike, np.ndarray]]]
            A dictionary keyed by variable name containing a combined
            request for parameters and initial values.
        params : Optional[Union[Dict, ArrayLike]], optional
            The parameters to be varied in the batch run. Can be a
            dictionary mapping parameter names to values
            or an array-like object. Defaults to None. If a 1d sequence or
            array is given, this is assumed to be a set of parameters to
            override the defaults for every run.
        states : Optional[Union[Dict, ArrayLike]], optional
            The initial states to be varied in the batch run. Can be a
            dictionary mapping state names to values
            or an array-like object. Defaults to None. If a 1d sequence or
            array is given, this is assumed to be a set of parameters to
            override the defaults for every run.
        kind : str, optional
            The method for generating the grid of runs. Can be
            'combinatorial' or 'verbatim'.
            - 'combinatorial': Creates a run for every combination of the
            provided parameter and state values.
            - 'verbatim': Creates runs based on a direct pairing of values.
            All value arrays must have the same length.
            Defaults to 'combinatorial'.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two 2D numpy arrays:
            1. The initial state values for each run.
            2. The parameter values for each run.

        Examples
        --------
        See BatchGridBuilder module docstring for examples.
        """
        parray = None
        sarray = None
        if request is not None:
            if states is not None or params is not None:
                raise TypeError("If a mixed request dictionary is provided, "
                                "states and params requests must be None."
                                "Check that you've input your arguments "
                                "correctly, using keywords for params and "
                                "inits, if you were not trying to provide a "
                                "mixed request dictionary.")
            if not isinstance(request, dict):
                raise TypeError(
                        "If provided, a combined request must be provided "
                        f"as a dictionary, got {type(request)}.")
            return self.grid_arrays(request, kind=kind)
        else:
            request = {}
            if isinstance(params, dict):
                request.update(params)
            elif isinstance(params, (list, tuple, np.ndarray)):
                parray = self._sanitise_arraylike(params, self.parameters)
            elif params is not None:
                raise TypeError("Parameters must be provided as a dictionary, "
                                "or a 1D or 2D array-like object.")
            if isinstance(states, dict):
                request.update(states)
            elif isinstance(states, (list, tuple, np.ndarray)):
                sarray = self._sanitise_arraylike(states, self.states)
            elif states is not None:
                raise TypeError(
                        "Initial states must be provided as a dictionary, "
                        "or a 1D or 2D array-like object.")

            if parray is not None and sarray is not None:
                return combine_grids(sarray, parray, kind=kind)
            elif request:
                if parray is not None:
                    sarray = generate_array(request, self.states, kind=kind)
                    return combine_grids(sarray, parray, kind=kind)
                elif sarray is not None:
                    parray = generate_array(request, self.parameters,
                                            kind=kind)
                    return combine_grids(sarray, parray, kind=kind)
                else:
                    return self.grid_arrays(request, kind=kind)
            elif parray is not None:
                sarray = np.full((parray.shape[0], self.states.n),
                                 self.states.values_array)
                return sarray, parray
            elif sarray is not None:
                parray = np.full((sarray.shape[0], self.parameters.n),
                                 self.parameters.values_array)
                return sarray, parray

            else:
                return (self.states.values_array[np.newaxis, :],
                        self.parameters.values_array[np.newaxis, :])

    def _trim_or_extend(self, arr: NDArray, values_object: SystemValues):
        """ Extends an incomplete array with defaults or trims extra values."""
        if arr.shape[1] < values_object.n:
            # If the array is shorter than the number of values, extend it
            # with default values
            arr = np.pad(arr, ((0, 0), (0, values_object.n - arr.shape[1])),
                         mode='constant',
                         constant_values=values_object.values_array[
                                         arr.shape[1]:])
        elif arr.shape[1] > values_object.n:
            arr = arr[:values_object.n]
        return arr

    def _sanitise_arraylike(self, arr, values_object: SystemValues):
        """Converts to 2D array if <2D and ensures the correct height."""
        if arr is None:
            return arr
        elif not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        if arr.ndim > 2:
            raise ValueError(
                    f"Input must be a 1D or 2D array, but got a {arr.ndim}D "
                    f"array.")
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :]

        if arr.shape[1] != values_object.n:
            warn(f"Provided input data has {arr.shape[1]} columns, but there "
                 f"are {values_object.n} settable values. Missing values "
                 f"will be filled with default values, and extras ignored.")
            arr = self._trim_or_extend(arr, values_object)
        if arr.size == 0:
            return None

        return arr  # correctly sized array just falls through untouched
