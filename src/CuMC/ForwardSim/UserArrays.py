from CuMC.ForwardSim.BatchOutputArrays import OutputArrays
from CuMC.ForwardSim.BatchConfigurator import BatchConfigurator
from CuMC.ForwardSim.BatchSolverKernel import BatchSolverKernel
from CuMC.ForwardSim import summary_metrics
import attrs
import numpy as np
from numpy.typing import NDArray
from typing import Optional

class UserArrays:
    time_domain: Optional[NDArray] = attrs.field(
        default=attrs.Factory(lambda: np.array([])),
        validator=attrs.validators.optional(attrs.validators.instance_of(NDArray)))
    summaries: Optional[NDArray] = attrs.field(
        default=attrs.Factory(lambda: np.array([])),
        validator=attrs.validators.optional(attrs.validators.instance_of(NDArray)))
    time_domain_legend: Optional[dict[int, str]] = attrs.field(default=attrs.Factory(dict),
        validator=attrs.validators.optional(attrs.validators.instance_of(dict)))
    summaries_legend: Optional[dict[int, str]] = attrs.field(default=attrs.Factory(dict),
        validator=attrs.validators.optional(attrs.validators.instance_of(dict)))

    @classmethod
    def from_solver(cls, solver: BatchSolverKernel) -> "UserArrays": #noqa: F821
        """
        Create UserArrays from a BatchSolverKernel instance.

        Args:
            solver (BatchSolverKernel): The solver instance to extract user arrays from.

        Returns:
            UserArrays: An instance of UserArrays containing the data from the solver.
        """
        user_arrays =  1

        return user_arrays

    def get_time_domain_legend(self) -> dict[int, str]:
        """
        Get the legend for the time domain.

        Returns:
            dict[int, str]: A dictionary mapping indices to time domain labels.
        """
        return self.time_domain_legend if self.time_domain_legend else {}



def summary_legend_from_solver(solver: "Solver") -> dict[int, str]: #noqa: F821
    """
    Get the summary array legend from a Solver instance.

    Args:
        solver (BatchSolverKernel): The solver instance to extract the time domain legend from.

    Returns:
        dict[int, str]: A dictionary mapping indices to time domain labels.
    """
    # Add toggles to avoid errors on no-summary runs

    summaries_legend = {}
    singlevar_legend = solver.summary_legend_per_variable
    saved_states = solver.saved_state_indices
    saved_observables = solver.saved_observable_indices
    state_labels = solver.batch_config.state_names(saved_states)
    obs_labels = solver.batch_config.observable_names(saved_observables)
    len_state_legend = len(state_labels) * len(singlevar_legend)

    for i, label in enumerate(state_labels):
        for j, (key, val) in enumerate(singlevar_legend.items()):
            string = f"{label} {val}"
            index = i * len(singlevar_legend) + j
            summaries_legend[index] = string
    for i, label in enumerate(obs_labels):
        for j, (key, val) in enumerate(singlevar_legend.items()):
            string = f"{label} {val}"
            index = len_state_legend + i * len(singlevar_legend) + j
            summaries_legend[index] = string

    #Copilots concise but less readable version:
    # # A more concise version using dictionary comprehensions:
    # len_state_legend = len(state_labels) * len(singlevar_legend)
    # state_dict = {
    #     i * len(singlevar_legend) + j: f"{label} {val}"
    #     for i, label in enumerate(state_labels)
    #     for j, (_, val) in enumerate(singlevar_legend.items())
    #     }
    # obs_dict = {
    #     len_state_legend + i * len(singlevar_legend) + j: f"{label} {val}"
    #     for i, label in enumerate(obs_labels)
    #     for j, (_, val) in enumerate(singlevar_legend.items())
    #     }
    # summaries_legend = {**state_dict, **obs_dict}

def time_domain_legend_from_solver(solver: "Solver") -> dict[int, str]:  # noqa: F821
    """
    Get the time domain legend from a Solver instance.

    Args:
        solver (BatchSolverKernel): The solver instance to extract the time domain legend from.

    Returns:
        dict[int, str]: A dictionary mapping indices to time domain labels.
    """
    # Add toggles to avoid errors on no-time-domain runs
    time_domain_legend = {}
    saved_states = solver.saved_state_indices
    saved_observables = solver.saved_observable_indices
    state_labels = solver.batch_config.state_names(saved_states) # hoik up into solver
    obs_labels = solver.batch_config.observable_names(saved_observables) #hoik up into solver

    for i, label in enumerate(state_labels):
        time_domain_legend[i] = f"{label}"

    for i, label in enumerate(obs_labels):
        time_domain_legend[len(state_labels) + i] = f"{label}"

    return time_domain_legend

# class ThisClassStopsErrorsAndDoesNothingElse:
#     def time_domain_array(self) -> np.ndarray:
#         active_outputs = self._active_outputs
#         include_state = active_outputs.state
#         include_observables = active_outputs.observables
#
#         if include_state and include_observables:
#             return np.hstack((self.state, self.observables))
#         elif include_state:
#             return self.state
#         elif include_observables:
#             return self.observables
#         else:
#             return np.array([])
#
#     def summaries_array(self) -> np.ndarray:
#         active_outputs = self._active_outputs
#         include_state = active_outputs.state_summaries
#         include_observables = active_outputs.observable_summaries
#
#         if include_state and include_observables:
#             return np.hstack((self.state_summaries, self.observable_summaries))
#         elif include_state:
#             return self.state_summaries
#         elif include_observables:
#             return self.observable_summaries
#         else:
#             return np.array([])
#
#     def split_summaries_by_type(self, solver_instance: "BatchSolverKernel"):  # noqa: F821
#         """
#         Return dicts of summary arrays split by metric type for states and observables.
#
#         Args:
#             solver_instance: The solver instance containing output types information
#
#         Returns:
#             A tuple of two dictionaries (state_splits, obs_splits) where each dictionary
#             maps summary type to the corresponding slice of the summary array
#         """
#         output_types = solver_instance.integrator.summary_types
#         if not output_types:
#             return {}, {}
#
#         heights = summary_metrics.output_sizes(output_types)
#
#         # Split state_summaries
#         state_splits = {}
#         offset = 0
#         for s_type, h in zip(output_types, heights):
#             state_splits[s_type] = self.state_summaries[..., offset:offset + h]
#             offset += h
#
#         # Split observable_summaries
#         obs_splits = {}
#         offset = 0
#         for s_type, h in zip(output_types, heights):
#             obs_splits[s_type] = self.observable_summaries[..., offset:offset + h]
#             offset += h
#
#         return state_splits, obs_splits
#
#     def legend(self, solver_instance: "BatchSolverKernel"):  # noqa: F821
#         """
#         Return a dict mapping row index to variable name and type for state, output, and summaries.
#
#         Args:
#             solver_instance: The solver instance containing system and integrator information
#
#         Returns:
#             A dictionary with the following structure:
#             {
#                 'state': {index: {'variable': variable_name, 'type': 'state'}},
#                 'observables': {index: {'variable': variable_name, 'type': 'observable'}},
#                 'state_summaries': {index: {'variable': variable_name, 'summary_type': summary_type}},
#                 'observable_summaries': {index: {'variable': variable_name, 'summary_type': summary_type}}
#             }
#         """
#         # Get variable names from the solver instance
#         state_names = solver_instance.system.state_names
#         observable_names = solver_instance.system.observable_names
#
#         # Get summary types from the solver instance
#         summary_types = solver_instance.integrator.summary_types
#
#         # Create legends for state and observables
#         state_legend = {}
#         for idx, name in enumerate(state_names):
#             state_legend[idx] = {"variable": name, "type": "state"}
#
#         observable_legend = {}
#         for idx, name in enumerate(observable_names):
#             observable_legend[idx] = {"variable": name, "type": "observable"}
#
#         # Create legends for state summaries and observable summaries
#         state_summaries_legend = {}
#         observable_summaries_legend = {}
#
#         if summary_types:
#             # Get heights for each summary type
#             heights = summary_metrics.output_sizes(summary_types)
#
#             # Create legend for state summaries
#             state_idx = 0
#             for s_type, height in zip(summary_types, heights):
#                 for var_name in state_names:
#                     for i in range(height):
#                         state_summaries_legend[state_idx] = {
#                             "variable":     var_name,
#                             "summary_type": s_type,
#                             "index":        i
#                             }
#                         state_idx += 1
#
#             # Create legend for observable summaries
#             obs_idx = 0
#             for s_type, height in zip(summary_types, heights):
#                 for var_name in observable_names:
#                     for i in range(height):
#                         observable_summaries_legend[obs_idx] = {
#                             "variable":     var_name,
#                             "summary_type": s_type,
#                             "index":        i
#                             }
#                         obs_idx += 1
#
#         return {
#             "state":                state_legend,
#             "observables":          observable_legend,
#             "state_summaries":      state_summaries_legend,
#             "observable_summaries": observable_summaries_legend
#             }

    # def output_arrays_with_legend(self, solver_instance: "BatchSolverKernel") -> dict:
    #     """
    #     Return a dictionary of output arrays with legends attached as SolverResult objects
    #
    #     Args:
    #         solver_instance: The solver instance containing system and integrator information
    #
    #     Returns:
    #         A dictionary with keys 'state', 'observables', 'state_summaries', 'observable_summaries',
    #         'time_domain', and 'summaries', each containing a SolverResult object with the
    #         corresponding array data and legend attached.
    #     """
    #     # Get the legend
    #     legend_dict = self.legend(solver_instance)
    #
    #     # Create SolverResult objects with legends attached
    #     result = {}
    #
    #     # Handle state array
    #     if self.state is not None and self.state.size > 1:
    #         result['state'] = SolverResult(self.state, legend_dict['state'])
    #
    #     # Handle observables array
    #     if self.observables is not None and self.observables.size > 1:
    #         result['observables'] = SolverResult(self.observables, legend_dict['observables'])
    #
    #     # Handle state_summaries array
    #     if self.state_summaries is not None and self.state_summaries.size > 1:
    #         result['state_summaries'] = SolverResult(self.state_summaries, legend_dict['state_summaries'])
    #
    #     # Handle observable_summaries array
    #     if self.observable_summaries is not None and self.observable_summaries.size > 1:
    #         result['observable_summaries'] = SolverResult(self.observable_summaries,
    #                                                       legend_dict['observable_summaries']
    #                                                       )
    #
    #     # Handle time_domain array
    #     time_domain = self.time_domain_array()
    #     if time_domain.size > 0:
    #         # Combine state and observables legends
    #         time_domain_legend = {}
    #         state_legend = legend_dict['state']
    #         obs_legend = legend_dict['observables']
    #
    #         # Add state legend entries
    #         for idx, entry in state_legend.items():
    #             time_domain_legend[idx] = entry
    #
    #         # Add observables legend entries with offset
    #         state_size = len(state_legend)
    #         for idx, entry in obs_legend.items():
    #             time_domain_legend[idx + state_size] = entry
    #
    #         result['time_domain'] = SolverResult(time_domain, time_domain_legend)
    #
    #     # Handle summaries array
    #     summaries = self.summaries_array()
    #     if summaries.size > 0:
    #         # Combine state_summaries and observable_summaries legends
    #         summaries_legend = {}
    #         state_summaries_legend = legend_dict['state_summaries']
    #         obs_summaries_legend = legend_dict['observable_summaries']
    #
    #         # Add state_summaries legend entries
    #         for idx, entry in state_summaries_legend.items():
    #             summaries_legend[idx] = entry
    #
    #         # Add observable_summaries legend entries with offset
    #         state_summaries_size = len(state_summaries_legend)
    #         for idx, entry in obs_summaries_legend.items():
    #             summaries_legend[idx + state_summaries_size] = entry
    #
    #         result['summaries'] = SolverResult(summaries, summaries_legend)
    #
    #     return result