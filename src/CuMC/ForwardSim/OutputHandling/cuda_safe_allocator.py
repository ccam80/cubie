"""
CUDA-safe array allocation utilities that prevent zero-sized array issues.
"""

import numpy as np
from numba import cuda
from typing import Dict, Tuple, Any, Optional
from .output_config import OutputConfig


class CUDASafeAllocator:
    """

    """

    def __init__(self, config: OutputConfig):
        self.config = config

    def get_array_shapes(self, n_samples: int, n_summary_samples: int) -> Dict[str, Tuple[int, ...]]:
        """
        Calculate array shapes for all output types, ensuring CUDA compatibility.

        Args:
            n_samples: Number of time-domain samples
            n_summary_samples: Number of summary samples

        Returns:
            Dictionary of array names to shapes
        """
        cuda_dims = self.config.get_cuda_safe_dimensions()  # remove cuda safe naming

        shapes = {}

        # Time-domain outputs
        if self.config.save_state or self.config.save_time:
            time_cols = cuda_dims["n_saved_states"]
            if self.config.save_time:
                time_cols += 1
            shapes["state_output"] = (n_samples, time_cols)
        else:
            shapes["state_output"] = (n_samples, 1)  # Minimal fallback

        if self.config.save_observables:
            shapes["observables_output"] = (n_samples, cuda_dims["n_saved_observables"])
        else:
            shapes["observables_output"] = (n_samples, 1)  # Minimal fallback

        # Summary outputs
        if self.config.save_summaries:
            shapes["state_summaries_output"] = (n_summary_samples, cuda_dims["state_summary_output"])
            shapes["observables_summaries_output"] = (n_summary_samples, cuda_dims["obs_summary_output"])
        else:
            shapes["state_summaries_output"] = (n_summary_samples, 1)
            shapes["observables_summaries_output"] = (n_summary_samples, 1)

        # Temporary arrays (for CUDA kernels)
        shapes["temp_state_summaries"] = (cuda_dims["state_summary_temp"],)
        shapes["temp_obs_summaries"] = (cuda_dims["obs_summary_temp"],)

        return shapes

    def create_output_arrays(self, shapes: Dict[str, Tuple[int, ...]], precision: np.dtype) -> Dict[str, np.ndarray]:
        """
        Create all output arrays with the given shapes and precision.
        Review: This one's bollocks
        Args:
            shapes: Dictionary of array names to shapes
            precision: NumPy dtype (float32 or float64)

        Returns:
            Dictionary of array names to allocated arrays
        """
        arrays = {}

        for name, shape in shapes.items():
            arrays[name] = np.zeros(shape, dtype=precision)

        return arrays

    def extract_user_data(self, arrays: Dict[str, np.ndarray],
                          n_samples: int, n_summary_samples: int,
                          ) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract only the data the user actually requested, filtering out padding.

        Args:
            arrays: Dictionary of raw output arrays (potentially with padding)
            n_samples: Number of actual time-domain samples
            n_summary_samples: Number of actual summary samples

        Returns:
            Dictionary of filtered arrays with only requested data
        """
        user_data = {}

        # Time-domain data
        if self.config.save_state or self.config.save_time:
            state_array = arrays["state_output"][:n_samples]

            if self.config.save_state and self.config.save_time:
                # Split into time and state components
                user_data["time"] = state_array[:, -1]  # Last column is time
                user_data["state"] = state_array[:, :-1]  # All but last column
            elif self.config.save_time:
                user_data["time"] = state_array[:, 0]  # Only time
            elif self.config.save_state:
                n_states = self.config.effective_n_saved_states
                user_data["state"] = state_array[:, :n_states]

        if self.config.save_observables:
            n_obs = self.config.effective_n_saved_observables
            user_data["observables"] = arrays["observables_output"][:n_samples, :n_obs]

        # Summary data
        # TODO: This one needs to get labels for each summary type
        if self.config.save_summaries:
            if self.config.needs_state_summaries:
                req = self.config.get_memory_requirements()
                n_state_summaries = req["state_summaries"]["output"]
                user_data["state_summaries"] = arrays["state_summaries_output"][:n_summary_samples, :n_state_summaries]

            if self.config.needs_observable_summaries:
                req = self.config.get_memory_requirements()
                n_obs_summaries = req["observable_summaries"]["output"]
                user_data["observable_summaries"] = arrays["observables_summaries_output"][:n_summary_samples,
                                                    :n_obs_summaries]

        return user_data

    def get_kernel_parameters(self) -> Dict[str, Any]:
        """
        Get parameters needed by CUDA kernels to handle the allocation strategy.
        #TODO: Review - this seems like double handling, why is this separate.
        Returns:
            Dictionary of parameters for kernel functions
        """
        cuda_dims = self.config.get_cuda_safe_dimensions()

        return {
            "actual_n_saved_states":      self.config.effective_n_saved_states,
            "actual_n_saved_observables": self.config.effective_n_saved_observables,
            "cuda_n_saved_states":        cuda_dims["n_saved_states"],
            "cuda_n_saved_observables":   cuda_dims["n_saved_observables"],
            "save_state":                 self.config.save_state,
            "save_observables":           self.config.save_observables,
            "save_time":                  self.config.save_time,
            "save_summaries":             self.config.save_summaries,
            "needs_state_summaries":      self.config.needs_state_summaries,
            "needs_observable_summaries": self.config.needs_observable_summaries,
            }