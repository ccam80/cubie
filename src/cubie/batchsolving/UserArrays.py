from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cubie.batchsolving.solver import Solver
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

import attrs
import numpy as np
from numpy.typing import NDArray
from copy import deepcopy
from cubie.batchsolving.BatchOutputArrays import ActiveOutputs


@attrs.define
class UserArrays:
    time_domain: Optional[NDArray] = attrs.field(
            default=attrs.Factory(lambda: np.array([])),
            validator=attrs.validators.optional(
                    attrs.validators.instance_of(np.ndarray)),
            eq=attrs.cmp_using(eq=np.array_equal))
    summaries: Optional[NDArray] = attrs.field(
            default=attrs.Factory(lambda: np.array([])),
            validator=attrs.validators.optional(
                    attrs.validators.instance_of(np.ndarray)),
            eq=attrs.cmp_using(eq=np.array_equal))
    time_domain_legend: Optional[dict[int, str]] = attrs.field(
            default=attrs.Factory(dict), validator=attrs.validators.optional(
                    attrs.validators.instance_of(dict)))
    summaries_legend: Optional[dict[int, str]] = attrs.field(
            default=attrs.Factory(dict), validator=attrs.validators.optional(
                    attrs.validators.instance_of(dict)))
    _singlevar_summary_legend: Optional[dict[int, str]] = attrs.field(
            default=attrs.Factory(dict), validator=attrs.validators.optional(
                    attrs.validators.instance_of(dict)))
    active_outputs: Optional[ActiveOutputs] = attrs.field(
            default=attrs.Factory(lambda: ActiveOutputs()))

    @classmethod
    def from_solver(cls, solver: Union[
        "Solver", "BatchSolverKernel"]) -> "UserArrays":
        """
        Create user_arrays from a Solver instance.

        Args:
            solver (BatchSolverKernel): The solver instance to extract user
            arrays from.

        Returns:
            UserArrays: An instance of user_arrays containing the data from
            the solver.
        """
        active_outputs = solver.active_output_arrays
        time_domain_array = cls.time_domain_array(active_outputs,
                                                  solver.state,
                                                  solver.observables)
        summaries_array = cls.summaries_array(active_outputs,
                                              solver.state_summaries,
                                              solver.observable_summaries)
        time_domain_legend = cls.time_domain_legend_from_solver(solver)
        summaries_legend = cls.summary_legend_from_solver(solver)
        _singlevar_summary_legend = solver.summary_legend_per_variable

        user_arrays = cls(time_domain=time_domain_array,
                summaries=summaries_array,
                time_domain_legend=time_domain_legend,
                summaries_legend=summaries_legend,
                active_outputs=active_outputs,
                singlevar_summary_legend=_singlevar_summary_legend, )

        return user_arrays

    def update_from_solver(self, solver: "Solver") -> "UserArrays":
        """
        Create UserArrays from a BatchSolverKernel instance.

        Args:
            solver (BatchSolverKernel): The solver instance to extract user
            arrays from.

        Returns:
            UserArrays: An instance of UserArrays containing the data from
            the solver.
        """
        self.active_outputs = solver.active_output_arrays

        self.time_domain = UserArrays.time_domain_array(
                self.active_outputs,
                solver.state,
                solver.observables, )
        self.summaries = UserArrays.summaries_array(self.active_outputs,
                                                    solver.state_summaries,
                                                    solver.observable_summaries)
        self._singlevar_summary_legend = solver.summary_legend_per_variable
        self.time_domain_legend = UserArrays.time_domain_legend_from_solver(
                solver)
        self.summaries_legend = UserArrays.summary_legend_from_solver(solver)

    @property
    def as_pandas(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                    "Pandas is required to convert UserArrays to DataFrames. "
                    "To keep the dependencies list low, Pandas isn’t "
                    "included. Install Pandas to use this feature.")

        # Construct multi-indexed DataFrame for time domain where each slice
        # [:, i, :] is stacked under run index i
        time_dfs = []
        n_runs = self.time_domain.shape[1] if self.time_domain.ndim == 3 else 1
        for run_idx in range(n_runs):
            slice_array = self.time_domain[:, run_idx, :]
            df = pd.DataFrame(slice_array,
                              columns=list(self.time_domain_legend.values()))
            if "time" in df.index:
                new_columns = df.loc["time"].tolist()
                df = df.drop("time")
                df.columns = new_columns
            df.index = pd.MultiIndex.from_product([[run_idx], df.index])
            time_dfs.append(df)
        time_domain_df = pd.concat(time_dfs)

        # Construct multi-indexed DataFrame for summaries by stacking each
        # 2d slice [:, i, :]
        summaries_dfs = []
        n_runs_summaries = self.summaries.shape[
            1] if self.summaries.ndim == 3 else 1
        for run_idx in range(n_runs_summaries):
            slice_array = self.summaries[:, run_idx, :]
            df = pd.DataFrame(slice_array,
                              columns=list(self.summaries_legend.values()))
            summaries_dfs.append(df)
        summaries_df = pd.concat(summaries_dfs, keys=range(n_runs_summaries))

        return time_domain_df, summaries_df

    @property
    def as_numpy(self) -> dict[str, NDArray]:
        """
        Returns the arrays instance as a dictionary of NumPy arrays.

        Returns:
            dict[str, NDArray]: A dictionary containing the time domain and
            summaries arrays.
        """
        return {"time_domain"       : np.asarray(deepcopy(self.time_domain)),
                "summaries"         : np.asarray(deepcopy(self.summaries)),
                "time_domain_legend": self.time_domain_legend.copy(),
                "summaries_legend"  : self.summaries_legend.copy(), }

    @property
    def per_summary_arrays(self) -> dict[str, NDArray]:
        """
        Returns each summary as a separate array, keyed by summary type.

        Returns:
            dict[str, NDArray]: A dictionary containing one array for each
            summary type. If a summary type has
            multiple entries, such as multiple peaks, then each entry is
            returned as a separate array.
        """
        if self.summaries.size == 0:
            return {}

        # Split summaries by type
        variable_legend = self.time_domain_legend
        singlevar_legend = self._singlevar_summary_legend
        indices_per_var = np.max([k for k in singlevar_legend.keys()]) + 1

        if "time" in variable_legend.values():
            time_key = next(
                    (k for k, v in variable_legend.items() if v == "time"),
                    None)
            variable_legend = {(k - 1 if k > time_key else k): v for k, v in
                               variable_legend.items() if v != "time"}

        per_summary_arrays = {}
        for offset, label in singlevar_legend.items():
            per_summary_arrays[label] = self.summaries[:, :,
                                        offset::indices_per_var]

        per_summary_arrays["legend"] = variable_legend

        return per_summary_arrays

    @property
    def active_outputs(self):
        """
        Flags indicating which device arrays are nonzero.
        """
        return self.active_outputs

    @staticmethod
    def time_domain_array(active_outputs, state, observables) -> np.ndarray:

        active_outputs = active_outputs
        include_state = active_outputs.state
        include_observables = active_outputs.observables

        if include_state and include_observables:
            return np.concatenate((state, observables), axis=-1)
        elif include_state:
            return state
        elif include_observables:
            return observables
        else:
            return np.array([])

    @staticmethod
    def summaries_array(active_outputs, state_summaries,
                        observable_summaries) -> np.ndarray:
        include_state = active_outputs.state_summaries
        include_observables = active_outputs.observable_summaries

        if include_state and include_observables:
            return np.concatenate((state_summaries, observable_summaries),
                                  axis=-1)
        elif include_state:
            return state_summaries
        elif include_observables:
            return observable_summaries
        else:
            return np.array([])

    @staticmethod
    def summary_legend_from_solver(solver: "Solver") -> dict[int, str]:
        """
        Get the summary array legend from a Solver instance.


        Args:
            solver (BatchSolverKernel): The solver instance to extract the
            time domain legend from.

        Returns:
            dict[int, str]: A dictionary mapping indices to time domain labels.
        """
        singlevar_legend = solver.summary_legend_per_variable
        saved_states = solver.saved_state_indices
        saved_observables = solver.saved_observable_indices
        state_labels = solver.batch_configurator.state_labels(saved_states)
        obs_labels = solver.batch_configurator.observable_labels(
                saved_observables)
        summaries_legend = {}
        # state summaries
        for i, label in enumerate(state_labels):
            for j, (key, val) in enumerate(singlevar_legend.items()):
                index = i * len(singlevar_legend) + j
                summaries_legend[index] = f"{label} {val}"
        # observable summaries
        len_state_legend = len(state_labels) * len(singlevar_legend)
        for i, label in enumerate(obs_labels):
            for j, (key, val) in enumerate(singlevar_legend.items()):
                index = len_state_legend + i * len(singlevar_legend) + j
                summaries_legend[index] = f"{label} {val}"
        return summaries_legend

    @staticmethod
    def time_domain_legend_from_solver(solver: "Solver") -> dict[int, str]:
        """
        Get the time domain legend from a Solver instance.
        Returns a dict mapping time domain indices to labels, including time
        if saved.

        Args:
            solver (BatchSolverKernel): The solver instance to extract the
            time domain legend from.

        Returns:
            dict[int, str]: A dictionary mapping indices to time domain labels.
        """
        time_domain_legend = {}
        saved_states = solver.saved_state_indices
        saved_observables = solver.saved_observable_indices
        state_labels = solver.batch_configurator.state_labels(saved_states)
        obs_labels = solver.batch_configurator.observable_labels(
                saved_observables)  # hoik up into solver
        offset = 0

        for i, label in enumerate(state_labels):
            time_domain_legend[i] = f"{label}"
            offset = i

        if solver.save_time:
            offset += 1
            time_domain_legend[offset] = "time"

        for i, label in enumerate(obs_labels):
            offset += 1
            time_domain_legend[offset + i] = label
        return time_domain_legend
