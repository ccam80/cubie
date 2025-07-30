"""
Module to configure batch runs by building parameter and initial value grids,
index generators for saved states/observables, and splitting summary outputs.
"""
from typing import List, Union, Dict

import numpy as np
from numpy.typing import ArrayLike

from CuMC.ForwardSim.OutputHandling import summary_metrics
from CuMC.SystemModels.SystemValues import SystemValues
from CuMC.SystemModels.Systems.GenericODE import GenericODE
from itertools import product

def unique_cartesian_product(arrays):
    """All combos picking one item from each array; remove duplicates from inputs on the way in."""
    deduplicated_inputs = [list(dict.fromkeys(a)) for a in arrays]  # preserve order, remove dups
    return [list(t) for t in product(*deduplicated_inputs)]

class BatchConfigurator:
    """
    Holds system parameters and initial values for integrator runs. Takes the default values from the system at
    instantiation, and all writing to the parameters and states is done in this module from that point on.
    """

    def __init__(self,
                 system_parameters: SystemValues,
                 system_inits: SystemValues,
                 system_observables: SystemValues
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
                   system_observables=observables
                   )

    def set(self, updates):
        """
        Update the parameters, states, or observables with a dictionary of updates.
        The keys should be the names or indices of the parameters/states/observables.
        """
        #  Again, try each with all tags, keep al ist of unrecognised ones.
        if 'parameters' in updates:
            self.parameters.update(updates['parameters'])
        if 'states' in updates:
            self.states.update(updates['states'])
        if 'observables' in updates:
            self.observables.update(updates['observables'])

    def state_indices(self,
                      keys_or_indices: Union[List[Union[str, int]], str, int]
                      ) -> np.ndarray:
        """
        Convert user-specified state names or indices to numpy array of int indices.
        """
        return self.states.get_indices(keys_or_indices)

    def observable_indices(self,
                           keys_or_indices: Union[List[Union[str, int]], str, int]
                           ) -> np.ndarray:
        """
        Convert user-specified observable names or indices to numpy array of int indices.
        """
        return self.observables.get_indices(keys_or_indices)

    def parameter_indices(self,
                          keys_or_indices: Union[List[Union[str, int]], str, int]
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
        return self.parameters.get_indices(keys_or_indices)

    def combinatorial_grid(self,
                           request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
                           values_instance: SystemValues
                           ) -> tuple[np.ndarray]:
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
        indices = values_instance.get_indices(request.keys())
        combos = unique_cartesian_product(
            [np.asarray(v, dtype=self.parameters.precision) for v in request.values()]
        )
        return indices, combos

    def verbatim_grid(self,
                        request: Dict[Union[str, int], Union[float, ArrayLike, NDArray]],
                        values_instance: SystemValues
                      ) -> np.ndarray:
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
        indices = values_instance.get_indices(request.keys())
        combos = [item for key, item in request.values()]
        return indices, combos

    def combined_input_grid(self,
                            request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
                            kind: str = 'combinatorial') -> np.ndarray:
        """
        Build a grid of parameters for a batch of runs based on a dictionary keyed by parameter name or index,
        Parameters
        ----------
        request
        kind

        Returns
        -------

        """
        # use similar logic to update_parameters - add silent flag to systemvalues updates, generate grids of each,
        # then combinate em up and fill the remainder with default values.
        pass

class SummarySplitter:
    """
    Split the flat summary output arrays into separate arrays for each summary metric.
    """

    def __init__(self, summary_types: List[str]):
        self.summary_types = summary_types
        # get output sizes per summary metric (width per variable)
        self.heights = summary_metrics.output_sizes(summary_types) # Use a sizes object or something

    def split_state_summaries(self, state_summary_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split state summary array of shape (n_runs, n_summary_samples, total_state_out_per_var)
        into dict of summary_type -> array of shape (n_runs, n_summary_samples, out_per_var_i).
        """
        splits = {}
        offset = 0
        for s_type, h in zip(self.summary_types, self.heights):
            splits[s_type] = state_summary_array[..., offset:offset + h]
            offset += h
        return splits

    def split_observable_summaries(self, obs_summary_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Same as split_state_summaries but for observables.
        """
        splits = {}
        offset = 0
        for s_type, h in zip(self.summary_types, self.heights):
            splits[s_type] = obs_summary_array[..., offset:offset + h]
            offset += h
        return splits