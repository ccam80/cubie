from typing import Optional, TYPE_CHECKING, Union, List, Any

if TYPE_CHECKING:
    from cubie.batchsolving.solver import Solver
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

import attrs
import attrs.validators as val
import numpy as np
from numpy.typing import NDArray
from cubie.batchsolving.arrays.BatchOutputArrays import ActiveOutputs
from cubie.batchsolving import ArrayTypes
from cubie._utils import slice_variable_dimension


@attrs.define
class SolveSpec:
    """Container for details of the solver run for return with data"""
    dt_min: float = attrs.field(validator=val.instance_of(float))
    dt_max: float = attrs.field(validator=val.instance_of(float))
    dt_save: float = attrs.field(validator=val.instance_of(float))
    dt_summarise: float = attrs.field(validator=val.instance_of(float))
    atol: float = attrs.field(validator=val.instance_of(float))
    rtol: float = attrs.field(validator=val.instance_of(float))
    duration: float = attrs.field(validator=val.instance_of(float))
    warmup: float = attrs.field(validator=val.instance_of(float))
    algorithm: str = attrs.field(validator=val.instance_of(str))
    saved_states: Optional[List[str]] = attrs.field()
    saved_observables: Optional[List[str]] = attrs.field()
    summarised_states: Optional[List[str]] = attrs.field()
    summarised_observables: Optional[List[str]] = attrs.field()
    output_types: Optional[List[str]] = attrs.field()
    precision: type = attrs.field(validator=val.instance_of(type))


@attrs.define
class SolveResult:
    time_domain_array: Optional[NDArray] = attrs.field(
            default=attrs.Factory(lambda: np.array([])),
            validator=val.optional(val.instance_of(np.ndarray)),
            eq=attrs.cmp_using(eq=np.array_equal))
    summaries_array: Optional[NDArray] = attrs.field(
            default=attrs.Factory(lambda: np.array([])),
            validator=val.optional(val.instance_of(np.ndarray)),
            eq=attrs.cmp_using(eq=np.array_equal))
    time: Optional[NDArray] = attrs.field(
            default=attrs.Factory(lambda: np.array([])),
            validator=val.optional(val.instance_of(np.ndarray)))
    time_domain_legend: Optional[dict[int, str]] = attrs.field(
            default=attrs.Factory(dict),
            validator=val.optional(val.instance_of(dict)))
    summaries_legend: Optional[dict[int, str]] = attrs.field(
            default=attrs.Factory(dict),
            validator=val.optional(val.instance_of(dict)))
    solve_settings: Optional[SolveSpec] = attrs.field(
            default=None,
            validator=val.optional(val.instance_of(SolveSpec)
            ))
    _singlevar_summary_legend: Optional[dict[int, str]] = attrs.field(
            default=attrs.Factory(dict),
            validator=val.optional(val.instance_of(dict)))
    _active_outputs: Optional[ActiveOutputs] = attrs.field(
            default=attrs.Factory(lambda: ActiveOutputs()))
    _stride_order: tuple[str, ...] = attrs.field(
            default=("time", "run", "variable"))

    @classmethod
    def from_solver(cls,
                    solver: Union["Solver", "BatchSolverKernel"],
                    results_type: str = 'full',
                    ) -> Union["SolveResult", dict[str,Any]]:
        """
        Create user_arrays from a Solver instance.

        Args:
            solver (Solver): The solver instance to extract results from.
            results_type(str): 'full' or 'numpy' or 'pandas'
                - 'full' returns a SolveResult instance with all arrays.
                - 'numpy' returns a dictionary of NumPy Arrays.
                - "numpy_per_summary" returns a dictionary of NumPy Arrays,
                where the summaries array is split into one array per
                summary type.
                - 'pandas' returns a dictionary of Pandas DataFrames.

        Returns:
            SolveResult: An instance of user_arrays containing the data from
            the solver.
        """

        active_outputs = solver.active_output_arrays
        state_active = active_outputs.state
        observables_active = active_outputs.observables
        state_summaries_active = active_outputs.state_summaries
        observable_summaries_active = active_outputs.observable_summaries
        solve_settings = solver.solve_info

        time, state_less_time = cls.cleave_time(
                solver.state,
                time_saved=solver.save_time,
                stride_order=solver.output_stride_order)

        time_domain_array = cls.combine_time_domain_arrays(
                state_less_time,
                solver.observables,
                state_active,
                observables_active)

        summaries_array = cls.combine_summaries_array(
                solver.state_summaries,
                solver.observable_summaries,
                state_summaries_active,
                observable_summaries_active)

        time_domain_legend = cls.time_domain_legend_from_solver(solver)

        summaries_legend = cls.summary_legend_from_solver(solver)
        singlevar_summary_legend = solver.summary_legend_per_variable

        user_arrays = cls(time_domain_array=time_domain_array,
                          summaries_array=summaries_array,
                          time=time,
                          time_domain_legend=time_domain_legend,
                          summaries_legend=summaries_legend,
                          active_outputs=active_outputs,
                          solve_settings=solve_settings,
                          stride_order=solver.output_stride_order,
                          singlevar_summary_legend=singlevar_summary_legend)

        if results_type == "full":
            return user_arrays
        elif results_type == "numpy":
            return user_arrays.as_numpy
        elif results_type == "numpy_per_summary":
            return user_arrays.as_numpy_per_summary
        elif results_type == "pandas":
            return user_arrays.as_pandas
        else:
            return user_arrays

    @property
    def as_pandas(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                    "Pandas is required to convert SolveResult to DataFrames. "
                    "Pandas is an optional dependency- it's only used here "
                    "to make analysis of data easier. Install Pandas to "
                    "use this feature.")

        run_index = self._stride_order.index("run")
        ndim = len(self._stride_order)
        time_dfs = []
        summaries_dfs = []
        any_summaries = (self.active_outputs.state_summaries or
                        self.active_outputs.observable_summaries)

        n_runs = (self.time_domain_array.shape[run_index] if ndim == 3 else 1)
        time_headings = list(self.time_domain_legend.values())
        summary_headings = list(self.summaries_legend.values())
        
        for run in range(n_runs):
            run_slice = slice_variable_dimension(slice(run, run+1, None),
                                                 run_index, ndim)

            singlerun_array = np.squeeze(self.time_domain_array[run_slice],
                                         axis=run_index)
            df = pd.DataFrame(singlerun_array, columns=time_headings)

            # Use time as index if extant
            if self.time is not None:
                if self.time.ndim > 1:
                    time_for_run = (self.time[:, run]
                                    if self.time.shape[1] > run
                                    else self.time[:, 0])
                else:
                    time_for_run = self.time
                df.index = time_for_run

            # Create MultiIndex columns with run number as first level
            df.columns = pd.MultiIndex.from_product([[f"run_{run}"],
                                                     df.columns])
            time_dfs.append(df)

            if any_summaries:
                singlerun_array = np.squeeze(self.summaries_array[run_slice],
                                             axis=run_index)
                df = pd.DataFrame(singlerun_array, columns=summary_headings)
                summaries_dfs.append(df)
                df.columns = pd.MultiIndex.from_product([[f"run_{run}"],
                                                         df.columns])
            else:
                summaries_dfs.append(pd.DataFrame)

            time_domain_df = pd.concat(time_dfs, axis=1)
            summaries_df = pd.concat(summaries_dfs, axis=1)

        return {"time_domain": time_domain_df,
                "summaries": summaries_df}

    @property
    def as_numpy(self) -> dict[str, Optional[NDArray]]:
        """
        Returns the arrays instance as a dictionary of NumPy arrays.

        Returns:
            dict[str, NDArray]: A dictionary containing the time domain and
            summaries_array arrays.
        """
        return {"time": self.time.copy() if self.time is not None else None,
                "time_domain_array": self.time_domain_array.copy(),
                "summaries_array": self.summaries_array.copy(),
                "time_domain_legend": self.time_domain_legend.copy(),
                "summaries_legend": self.summaries_legend.copy()}


    @property
    def as_numpy_per_summary(self) -> dict[str, Optional[NDArray]]:
        """
        Returns the arrays as a dictionary of NumPy arrays, where the
        summaries array has been split into one array per summary type.

        Returns:
            dict[str, NDArray]: A dictionary containing one array for
            time-domain results, time, and each summary type keyed by summary
            type.
        """
        arrays = {"time": self.time.copy() if self.time is not None else None,
                  "time_domain_array":  self.time_domain_array.copy(),
                  "time_domain_legend": self.time_domain_legend.copy()}
        arrays.update(**self.per_summary_arrays)

        return arrays


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
        if (self._active_outputs.state_summaries is False and
                self._active_outputs.observable_summaries is False):
            return {}

        variable_index = self._stride_order.index("variable")

        # Split summaries_array by type
        variable_legend = self.time_domain_legend
        singlevar_legend = self._singlevar_summary_legend
        indices_per_var = np.max([k for k in singlevar_legend.keys()]) + 1
        per_summary_arrays = {}

        for offset, label in singlevar_legend.items():
            summ_slice = slice(offset,None,indices_per_var)
            summ_slice = slice_variable_dimension(summ_slice,
                                                  variable_index,
                                                 len(self._stride_order))
            per_summary_arrays[label] = self.summaries_array[summ_slice].copy()
        per_summary_arrays["summary_legend"] = variable_legend

        return per_summary_arrays

    @property
    def active_outputs(self):
        """
        Flags indicating which device arrays are nonzero.
        """
        return self._active_outputs

    @staticmethod
    def cleave_time(state: ArrayTypes,
                    time_saved: bool = False,
                    stride_order: Optional[list[str]] = None) \
            -> (tuple[Optional[NDArray], NDArray]):
        """If time has been saved, remove it from the state array and return.

        Parameters:
            state (ArrayTypes): The state array to cleave.
            time_saved (bool): Whether time has been saved in the state array.
            stride_order (Optional[list[str]]): The order of dimensions in the
            host arrays.

        Returns:
            tuple[Optional[NDArray], NDArray]: A tuple containing the time (
            if saved, otherwise None) and the state array with time removed.
        """
        if stride_order is None:
            stride_order = ["time", "run", "variable"]
        if time_saved:
            var_index = stride_order.index("variable")
            ndim = len(state.shape)

            time_slice = slice_variable_dimension(slice(-1,None,None),
                                                  var_index, ndim)
            state_slice = slice_variable_dimension(slice(None, -1), var_index,
                                                   ndim)

            time = np.squeeze(state[time_slice], axis=var_index)
            state_less_time = state[state_slice]
            return time, state_less_time
        else:
            return None, state

    @staticmethod
    def combine_time_domain_arrays(state,
                                   observables,
                                   state_active=True,
                                   observables_active=True) -> NDArray:
        if state_active and observables_active:
            return np.concatenate((state, observables), axis=-1)
        elif state_active:
            return state.copy()
        elif observables_active:
            return observables.copy()
        else:
            return np.array([])

    @staticmethod
    def combine_summaries_array(state_summaries,
                        observable_summaries, summarise_states,
                                summarise_observables
                                ) -> np.ndarray:
        """ Combine state and observable summaries_array into a single array. """

        if summarise_states and summarise_observables:
            return np.concatenate((state_summaries, observable_summaries),
                                  axis=-1)
        elif summarise_states:
            return state_summaries.copy()
        elif summarise_observables:
            return observable_summaries.copy()
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
        state_labels = solver.saved_states
        obs_labels = solver.saved_observables
        summaries_legend = {}

        # state summaries_array
        for i, label in enumerate(state_labels):
            for j, (key, val) in enumerate(singlevar_legend.items()):
                index = i * len(singlevar_legend) + j
                summaries_legend[index] = f"{label} {val}"
        # observable summaries_array
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
        state_labels = solver.saved_states
        obs_labels = solver.saved_observables
        offset = 0

        for i, label in enumerate(state_labels):
            time_domain_legend[i] = f"{label}"
            offset = i

        for i, label in enumerate(obs_labels):
            offset += 1
            time_domain_legend[offset + i] = label
        return time_domain_legend
