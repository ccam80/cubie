"""
Module to configure batch runs by building parameter and initial value grids,
index generators for saved states/observables, and splitting summary outputs.
"""
from itertools import product
from typing import List, Union, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from CuMC.SystemModels.SystemValues import SystemValues
from CuMC.SystemModels.Systems.GenericODE import GenericODE


def unique_cartesian_product(arrays: List[np.ndarray]):
    """Return a 2D array of each unique combination of elements from a list of 1d input arrays.
    Each input array can have duplicates, but the output will not contain any duplicate rows.
    The order of the input arrays is preserved, and the output will have the same order of elements as the input.
    Parameters
    ----------
    arrays : List[np.ndarray]
        A list of 1D numpy arrays, each containing elements to be combined.
    Returns
    -------
    combos : np.ndarray
        A 2D numpy array where each row is a unique combination of elements from the inputs
    Examples
    --------
    >>> unique_cartesian_product([np.array([1, 2, 2]), np.array([3, 4])])
    array([[1, 3],
        [1, 4],
        [2, 3],
        [2, 4]])
    Notes
    -----
    This function removes duplicates by creating a dict with the elements of the input array as keys. It then casts
    that to a list, getting the de-duplicated values. It  then uses `itertools.product` to generate the Cartesian
    product of  the input arrays.
    ."""
    deduplicated_inputs = [list(dict.fromkeys(a)) for a in arrays]  # preserve order, remove dups
    return np.array([list(t) for t in product(*deduplicated_inputs)])


def combinatorial_grid(request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
                       values_instance: SystemValues,
                       silent: bool = False,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a grid of all unique combinations of values based on a dictionary keyed by parameter name or index,
    and with values comprising the entire set of parameter values.

    Parameters
    ----------
    request : Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]]
        Dictionary where keys are parameter names or indices, and values are either a single value or an array of
        values for that parameter.
        For a combinatorial grid, the arrays of values need not be equal in length.
    values_instance: SystemValues
        The SystemValues instance in which to find the indices for the keys in the request.
    silent: bool
        If True, suppress warnings about unrecognized parameters in the request.

    Returns
    -------
    grid: np.ndarray
        A 2D array of shape (n_runs, n_requested_parameters) where each row corresponds to a set of parameters
        for a run.
    indices: np.ndarray
        A 1D array of indices corresponding to the gridded parameters.

    Unspecified parameters are filled with their default values from the system. n_runs is the combinatorial
    of the lengths of all of the value types - for example, if the request contains two parameters with 3, 2,
    and 4 values, then n_runs would be 3 * 2 * 4 = 24
    Examples
    --------
    ```
    combinatorial_grid({
        'param1': [0.1, 0.2, 0.3],
        'param2': [10, 20]
    }, system.parameters)
    ```
    >>> (array([[ 0.1, 10. ],
           [ 0.1, 20. ],
           [ 0.2, 10. ],
           [ 0.2, 20. ],
           [ 0.3, 10. ],
           [ 0.3, 20. ]]),
         array([0, 1]))
    """
    cleaned_request = {k: v for k, v in request.items() if np.asarray(v).size > 0}
    indices = values_instance.get_indices(list(cleaned_request.keys()), silent=silent)
    combos = unique_cartesian_product(
            [np.asarray(v) for v in cleaned_request.values()],
            )
    return indices, combos


def verbatim_grid(request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
                  values_instance: SystemValues,
                  silent: bool = False,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """ Build a grid of parameters for a batch of runs based on a dictionary keyed by parameter name or index,
    and values the entire set of parameter values. Parameters vary together, but not combinatorially. All values
    arrays must be of equal length.
    Parameters
    ----------
    request : Dict[Union[str, int], Union[float, ArrayLike, NDArray]]
        Dictionary where keys are parameter names or indices, and values are either a single value or an array of
        values for that parameter.
    values_instance: SystemValues
        The SystemValues instance in which to find the indices for the keys in the request.
    silent: bool
        If True, suppress warnings about unrecognized parameters in the request.

    Returns
    -------
    grid: np.ndarray
        A 2D array of shape (n_runs, n_requested_parameters) where each row corresponds to a set of parameters
        for a run.
    indices: np.ndarray
        A 1D array of indices corresponding to the gridded parameters.
    Unspecified parameters are filled with their default values from the system. n_runs is the length of _all_
    value arrays, which must be equal.
    Examples
    --------
    ```
    verbatim_grid({
        'param1': [0.1, 0.2, 0.3],
        'param2': [10, 20, 30]
    }, system.parameters)
    >>> (array([[ 0.1, 10. ],
           [ 0.2, 20. ],
           [ 0.3, 30. ]]),
           array([0, 1]))

    ```
    """
    cleaned_request = {k: v for k, v in request.items() if np.asarray(v).size > 0}
    indices = values_instance.get_indices(list(cleaned_request.keys()), silent=silent)
    combos = np.asarray([item for item in cleaned_request.values()]).T
    return indices, combos


def generate_grid(request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
                  values_instance: SystemValues,
                  kind: str = 'combinatorial',
                  silent: bool = False,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a grid of parameters for a batch of runs based on a dictionary keyed by parameter name or index,
    and with values comprising the entire set of parameter values.

    Parameters
    ----------
    request : Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]]
        Dictionary where keys are parameter names or indices, and values are either a single value or an array of
        values for that parameter.
    values_instance: SystemValues
        The SystemValues instance in which to find the indices for the keys in the request.
    kind : str
        The type of grid to generate. Can be 'combinatorial' or 'verbatim'.
    silent: bool
        If True, suppress warnings about unrecognized parameters in the request.

    Returns
    -------
    grid: np.ndarray
        A 2D array of shape (n_runs, n_requested_parameters) where each row corresponds to a set of parameters
        for a run.
    indices: np.ndarray
        A 1D array of indices corresponding to the gridded parameters.

    Notes
    -----
    The `kind` parameter determines how the grid is constructed:
    - 'combinatorial': see BatchConfigurator.combinatorial_grid
    - 'verbatim': see BatchConfigurator.verbatim_grid
    """
    if kind == 'combinatorial':
        return combinatorial_grid(request, values_instance, silent=silent)
    elif kind == 'verbatim':
        return verbatim_grid(request, values_instance, silent=silent)
    else:
        raise ValueError(f"Unknown grid type '{kind}'. Use 'combinatorial' or 'verbatim'.")


def combine_grids(grid1: np.ndarray, grid2: np.ndarray, kind: str = 'combinatorial') -> tuple[np.ndarray,
np.ndarray]:
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
            raise ValueError("For 'verbatim', both grids must have the same number of rows.")
        return grid1, grid2
    else:
        raise ValueError(f"Unknown grid type '{kind}'. Use 'combinatorial' or 'verbatim'.")


def extend_grid_to_array(grid: np.ndarray,
                         indices: np.ndarray,
                         default_values: np.ndarray,
                         ):
    """Join a grid of values with the an array of default values, creating a 2D array where each row has a full
    set of parameters, and non-gridded parameters are set to their default values.
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
        A 2D array of shape (n_runs, n_parameters) where each row corresponds to a set of parameters for a run.
        Parameters not specified in the grid are filled with their default values from the system.
    """
    if grid.ndim == 1:
        array = default_values[np.newaxis, :]
    else:
        if grid.shape[1] != indices.shape[0]:
            raise ValueError("Grid shape does not match indices shape.")
        array = np.vstack([default_values] * grid.shape[0])
        array[:, indices] = grid

    return array


def generate_array(request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
                   values_instance: SystemValues,
                   kind: str = 'combinatorial',
                   ) -> np.ndarray:
    """
    Create a 2D array of requested parameters or states based on a dictionary of requests.
    Parameters
    ----------
    request : Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]]
        Dictionary where keys are parameter names or indices, and values are either a single value or an array of
        values for that parameter.
    values_instance: SystemValues
        The SystemValues instance in which to find the indices for the keys in the request.
    kind : str
        The type of grid to generate. Can be 'combinatorial' or 'verbatim'.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_runs, n_parameters) where each row corresponds to a set of parameters
        for a run. Parameters not specified in the request are filled with their default values from the system.
    """
    indices, grid = generate_grid(request, values_instance, kind=kind)
    return extend_grid_to_array(grid, indices, values_instance.values_array)


class BatchConfigurator:
    """
    Holds system parameters and initial values for integrator runs, as well as observable labels for a convenient
    to provide convenience indexing-by-label to the system's components. Takes default values from the system at
    instantiation, and all writing to the parameters and states is done in this module from that point on.
    """

    def __init__(self,
                 system_parameters: SystemValues,
                 system_inits: SystemValues,
                 system_observables: SystemValues,
                 ):

        self.parameters = system_parameters
        self.states = system_inits
        self.observables = system_observables

    @classmethod
    def from_system(cls, system: GenericODE):
        """
        Create a BatchConfigurator instance from a GenericODE system.
        """
        parameters = system.parameters
        inits = system.initial_values
        observables = system.observables

        return cls(system_parameters=parameters,
                   system_inits=inits,
                   system_observables=observables,
                   )

    def update(self, updates=None, **kwargs):
        """
        Update the parameters, states, and/or observables with a dictionary of updates or keyword arguments.
        The keys should be the names of the parameters/states/observables.
        """
        if updates is None:
            updates = {}
        if kwargs:
            updates.update(kwargs)
        if updates == {}:
            return

        all_unrecognized = set(updates.keys())
        recognized = []
        for values_object in (self.parameters, self.states):
            recognized = values_object.update_from_dict(updates, silent=True)
            all_unrecognized -= recognized

        # Check if any parameters were unrecognized (indicating an entry error)
        if all_unrecognized:
            unrecognized_list = sorted(all_unrecognized)
            raise KeyError(f"The following updates were not recognized by the system. Was this a typo?:"
                           f" {unrecognized_list}",
                           )

    def state_indices(self,
                      keys_or_indices: Union[List[Union[str, int]], str, int],
                      silent: bool = False,
                      ) -> np.ndarray:
        """
        Convert user-specified state names or indices to numpy array of int indices.
        """
        return self.states.get_indices(keys_or_indices, silent=silent)

    def observable_indices(self,
                           keys_or_indices: Union[List[Union[str, int]], str, int],
                           silent: bool = False,
                           ) -> np.ndarray:
        """
        Convert user-specified observable names or indices to numpy array of int indices.
        """
        return self.observables.get_indices(keys_or_indices, silent=silent)

    def parameter_indices(self,
                          keys_or_indices: Union[List[Union[str, int]], str, int],
                          silent: bool = False,
                          ) -> np.ndarray:
        """
        Convert user-specified parameter names or indices to numpy array of int indices.
        Parameters
        ----------
        keys_or_indices : Union[List[Union[str, int]], str, int]
            A list of parameter names or indices, or a single name or index.
            If a list is provided, it can contain strings (parameter names) or integers (indices).

        Returns
        -------
        indices: np.ndarray
            A numpy array of integer indices corresponding to the provided parameter names or indices.
            If a single name or index is provided, returns a 1D array with that single index.
        """
        return self.parameters.get_indices(keys_or_indices, silent=silent)

    def grid_arrays(self,
                    request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
                    kind: str = 'combinatorial',
                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a grid of parameters for a batch of runs based on a dictionary keyed by parameter name or index,
        Parameters
        ----------
        request
        kind

        Returns
        -------
        np.ndarray, np.ndarray
            2d state and parameter arrays for input into the integrator
        """
        param_request = {k: v for k, v in request.items() if k in self.parameters.names}
        state_request = {k: v for k, v in request.items() if k in self.states.names}

        params_array = generate_array(param_request, self.parameters, kind=kind)
        initial_values_array = generate_array(state_request, self.states, kind=kind)
        initial_values_array, params_array = combine_grids(initial_values_array, params_array, kind=kind)

        return initial_values_array, params_array

    def get_labels(self, values_object, indices: np.ndarray) -> List[str]:
        """
        Get the labels of the states corresponding to the provided indices.
        Parameters
        ----------
        indices : np.ndarray
            A 1D array of state indices.

        Returns
        -------
        List[str]
            A list of state labels corresponding to the provided indices.
        """
        return values_object.get_labels(indices)

    def state_labels(self, indices: Optional[np.ndarray]) -> List[str]:
        """
        Get the labels of the states corresponding to the provided indices.
        Parameters
        ----------
        indices : np.ndarray
            A 1D array of state indices. If None, return all state labels.

        Returns
        -------
        List[str]
            A list of state labels corresponding to the provided indices.
        """
        if indices is None:
            return self.states.names
        return self.get_labels(self.states, indices)

    def observable_labels(self, indices: Optional[np.ndarray]) -> List[str]:
        """
        Get the labels of the observables corresponding to the provided indices.
        Parameters
        ----------
        indices : np.ndarray
            A 1D array of observable indices. If None, return all observable labels.

        Returns
        -------
        List[str]
            A list of observable labels corresponding to the provided indices.
        """
        if indices is None:
            return self.observables.names
        return self.get_labels(self.observables, indices)

    def parameter_labels(self, indices: Optional[np.ndarray]) -> List[str]:
        """
        Get the labels of the parameters corresponding to the provided indices.
        Parameters
        ----------
        indices : np.ndarray
            A 1D array of parameter indices. If None, return all parameter labels.

        Returns
        -------
        List[str]
            A list of parameter labels corresponding to the provided indices.
        """
        if indices is None:
            return self.parameters.names
        return self.get_labels(self.parameters, indices)