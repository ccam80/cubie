"""
Module to configure batch runs by building parameter and initial value grids,
index generators for saved states/observables, and splitting summary outputs.
"""
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Union, Dict
from CuMC.SystemModels.SystemValues import SystemValues
from CuMC.ForwardSim.OutputHandling import summary_metrics


class BatchConfigurator:
    """
    Build grids of parameters or initial values for batch runs and generate index arrays.
    """
    def __init__(self, system):
        self.system = system
        # system must expose .parameters and .initial_values as SystemValues

    def build_parameter_grid(self, param_keys: List[str], values: ArrayLike) -> np.ndarray:
        """
        Given a list of parameter names and a 2D array of values (n_runs x n_keys),
        return a full parameter array of shape (n_runs x n_total_parameters),
        filling unspecified parameters with defaults.
        """
        params_sv: SystemValues = self.system.parameters
        # resolve indices for keys
        idx = params_sv.get_indices(param_keys)
        values = np.asarray(values, dtype=params_sv.precision)
        if values.ndim == 1:
            values = values.reshape(-1, len(idx))
        n_runs = values.shape[0]
        grid = np.tile(params_sv.values_array, (n_runs, 1))
        grid[:, idx] = values
        return grid

    def build_initial_values_grid(self, state_keys: List[str], values: ArrayLike) -> np.ndarray:
        """
        Given state names and values, build full initial values array for each run.
        """
        states_sv: SystemValues = self.system.initial_values
        idx = states_sv.get_indices(state_keys)
        values = np.asarray(values, dtype=states_sv.precision)
        if values.ndim == 1:
            values = values.reshape(-1, len(idx))
        n_runs = values.shape[0]
        grid = np.tile(states_sv.values_array, (n_runs, 1))
        grid[:, idx] = values
        return grid

    def generate_saved_state_indices(self, keys_or_indices: Union[List[Union[str,int]], str, int]) -> np.ndarray:
        """
        Convert user-specified state names or indices to numpy array of int indices.
        """
        return self.system.initial_values.get_indices(keys_or_indices)

    def generate_saved_observable_indices(self, keys_or_indices: Union[List[Union[str,int]], str, int]) -> np.ndarray:
        """
        Convert user-specified observable names or indices to numpy array of int indices.
        """
        return self.system.observables.get_indices(keys_or_indices)


class SummarySplitter:
    """
    Split the flat summary output arrays into separate arrays for each summary metric.
    """
    def __init__(self, summary_types: List[str]):
        self.summary_types = summary_types
        # get output sizes per summary metric (width per variable)
        self.heights = summary_metrics.output_sizes(summary_types)

    def split_state_summaries(self, state_summary_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split state summary array of shape (n_runs, n_summary_samples, total_state_out_per_var)
        into dict of summary_type -> array of shape (n_runs, n_summary_samples, out_per_var_i).
        """
        splits = {}
        offset = 0
        for s_type, h in zip(self.summary_types, self.heights):
            splits[s_type] = state_summary_array[..., offset:offset+h]
            offset += h
        return splits

    def split_observable_summaries(self, obs_summary_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Same as split_state_summaries but for observables.
        """
        splits = {}
        offset = 0
        for s_type, h in zip(self.summary_types, self.heights):
            splits[s_type] = obs_summary_array[..., offset:offset+h]
            offset += h
        return splits
