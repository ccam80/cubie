"""Batch solver run specifications and result containers.

This module exposes :class:`SolveSpec` to describe solver configuration and
:class:`SolveResult` to aggregate output arrays, legends, and metadata once a
batch integration completes.

Published Classes
-----------------
:class:`SolveSpec`
    Frozen attrs dataclass describing the configuration used for a solver run.

:class:`RawSolveResult`
    Owns copied raw output arrays.

:class:`SolveResult`
    Aggregates output arrays, legends, and metadata from a completed batch
    integration.

See Also
--------
:class:`~cubie.batchsolving.solver.Solver`
    User-facing solver whose :meth:`~Solver.solve` returns a
    :class:`SolveResult`.
:func:`~cubie.batchsolving.solver.solve_ivp`
    Convenience wrapper returning a :class:`SolveResult`.
"""

from typing import Optional, TYPE_CHECKING, Union, List, Any, Tuple

if TYPE_CHECKING:
    from cubie.batchsolving.solver import Solver
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
    import pandas as pd

from attrs import (
    cmp_using as attrs_cmp_using,
    define,
    Factory as attrsFactory,
    field,
)
from attrs.validators import (
    instance_of as attrsval_instance_of,
    optional as attrsval_optional,
    or_ as attrsval_or,
)
from numpy import (
    amax as np_amax,
    array as np_array,
    array_equal as np_array_equal,
    concatenate as np_concatenate,
    nan as np_nan,
    ndarray,
    memmap as np_memmap,
    squeeze as np_squeeze,
    where as np_where,
)
from numpy.typing import NDArray
from cubie.batchsolving.BatchSolverConfig import ActiveOutputs
from cubie.batchsolving import ArrayTypes
from cubie.result_codes import decode_status_codes
from cubie.memory.mem_manager import HOST_STAGING_BYTES
from cubie._utils import (
    slice_variable_dimension,
    opt_gttype_validator,
    opt_getype_validator,
    getype_validator,
    gttype_validator,
    PrecisionDType,
)


def _format_time_domain_label(label: str, unit: str) -> str:
    """Format a time-domain legend label with unit if not dimensionless.

    Parameters
    ----------
    label
        Variable label.
    unit
        Unit string for the variable.

    Returns
    -------
    str
        Formatted label. If unit is "dimensionless", returns just the label.
        Otherwise returns "label [unit]".
    """
    if unit != "dimensionless":
        return f"{label} [{unit}]"
    return label


def _release_spill_arrays(memory_manager, arrays) -> None:
    """Release each spill mapping once."""
    cleanups = set()
    for array in arrays:
        owner = array
        while isinstance(owner, ndarray):
            cleanup = getattr(owner, "_cubie_spill_cleanup", None)
            if cleanup is not None and id(cleanup) not in cleanups:
                cleanups.add(id(cleanup))
                memory_manager.release_host_array(owner)
            owner = getattr(owner, "base", None)


class RawSolveResult(dict):
    """Own independent raw output arrays."""

    def __init__(self, arrays, memory_manager) -> None:
        super().__init__(arrays)
        self._memory_manager = memory_manager
        self._closed = False

    def close(self) -> None:
        """Release spill files owned by this result."""
        if self._closed:
            return
        _release_spill_arrays(self._memory_manager, self.values())
        self.clear()
        self._closed = True

    def __enter__(self) -> "RawSolveResult":
        """Return this result."""
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Release spill files on context exit."""
        self.close()


@define
class SolveSpec:
    """Describe the configuration of a solver run.

    Attributes
    ----------
    dt
        Fixed step size, or ``None`` for adaptive controllers.
    dt_min
        Minimum time step size.
    dt_max
        Maximum time step size.
    save_every
        Interval at which state values are stored.
    summarise_every
        Interval for computing summary outputs.
    sample_summaries_every
        Interval for sampling summary metric updates.
    atol
        Absolute error tolerance when configured.
    rtol
        Relative error tolerance when configured.
    duration
        Total integration time.
    warmup
        Initial warm-up period prior to recording outputs.
    t0
        Initial integration time supplied to the solver.
    algorithm
        Name of the integration algorithm.
    saved_states
        Labels of states saved verbatim or ``None`` when disabled.
    saved_observables
        Labels of observables saved verbatim or ``None`` when disabled.
    summarised_states
        Labels of states with summaries computed or ``None`` when disabled.
    summarised_observables
        Labels of observables with summaries computed or ``None`` when disabled.
    output_types
        Types of output arrays generated during the run or ``None``.
    precision
        Floating-point precision factory used for host conversions.
    """

    dt: Optional[float] = field(validator=opt_gttype_validator(float, 0.0))
    dt_min: float = field(validator=gttype_validator(float, 0.0))
    dt_max: float = field(validator=gttype_validator(float, 0.0))
    save_every: Optional[float] = field(
        validator=opt_gttype_validator(float, 0.0)
    )
    summarise_every: Optional[float] = field(
        validator=opt_getype_validator(float, 0.0)
    )
    sample_summaries_every: Optional[float] = field(
        validator=opt_getype_validator(float, 0.0)
    )
    atol: Optional[float] = field(
        validator=attrsval_or(
            opt_gttype_validator(float, 0.0), attrsval_instance_of(ndarray)
        ),
    )
    rtol: Optional[float] = field(
        validator=attrsval_or(
            opt_gttype_validator(float, 0.0), attrsval_instance_of(ndarray)
        ),
    )
    duration: float = field(validator=gttype_validator(float, 0.0))
    warmup: float = field(validator=getype_validator(float, 0.0))
    t0: float = field(validator=getype_validator(float, float("-inf")))
    algorithm: str = field(validator=attrsval_instance_of(str))
    saved_states: Optional[List[str]] = field()
    saved_observables: Optional[List[str]] = field()
    summarised_states: Optional[List[str]] = field()
    summarised_observables: Optional[List[str]] = field()
    output_types: Optional[List[str]] = field()
    precision: PrecisionDType = field()


@define
class SolveResult:
    """Aggregate output arrays and related metadata for a solver run.

    Parameters
    ----------
    time_domain_array
        NumPy array containing time-domain results.
    summaries_array
        NumPy array containing summary results.
    time
        Optional NumPy array containing time values.
    iteration_counters
        NumPy array containing iteration counts per run.
    status_codes
        Optional NumPy array containing solver status codes per run (0 for
        success, nonzero for errors). Shape is (n_runs,) with dtype int32.
    time_domain_legend
        Optional mapping from time-domain indices to labels.
    summaries_legend
        Optional mapping from summary indices to labels.
    solve_settings
        Optional solver run configuration.
    active_outputs
        Optional :class:`ActiveOutputs` instance describing enabled arrays.
    stride_order
        Sequence describing the order of axes in host arrays.
    singlevar_summary_legend
        Optional mapping from summary offsets to legend labels.
    """

    time_domain_array: NDArray = field(
        default=attrsFactory(lambda: np_array([])),
        validator=attrsval_instance_of(ndarray),
        eq=attrs_cmp_using(eq=np_array_equal),
    )
    summaries_array: NDArray = field(
        default=attrsFactory(lambda: np_array([])),
        validator=attrsval_instance_of(ndarray),
        eq=attrs_cmp_using(eq=np_array_equal),
    )
    time: Optional[NDArray] = field(
        default=attrsFactory(lambda: np_array([])),
        validator=attrsval_optional(attrsval_instance_of(ndarray)),
    )
    iteration_counters: NDArray = field(
        default=attrsFactory(lambda: np_array([])),
        validator=attrsval_instance_of(ndarray),
        eq=attrs_cmp_using(eq=np_array_equal),
    )
    status_codes: Optional[NDArray] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(ndarray)),
        eq=attrs_cmp_using(eq=np_array_equal),
    )
    time_domain_legend: Optional[dict[int, str]] = field(
        default=attrsFactory(dict),
        validator=attrsval_optional(attrsval_instance_of(dict)),
    )
    summaries_legend: Optional[dict[int, str]] = field(
        default=attrsFactory(dict),
        validator=attrsval_optional(attrsval_instance_of(dict)),
    )
    solve_settings: Optional[SolveSpec] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(SolveSpec)),
    )
    _singlevar_summary_legend: Optional[dict[int, str]] = field(
        default=attrsFactory(dict),
        validator=attrsval_optional(attrsval_instance_of(dict)),
    )
    _active_outputs: Optional[ActiveOutputs] = field(
        default=attrsFactory(ActiveOutputs)
    )
    _stride_order: Union[tuple[str, ...], list[str]] = field(
        default=("time", "variable", "run")
    )
    _memory_manager: Optional[Any] = field(
        default=None, repr=False, eq=False
    )
    _closed: bool = field(default=False, init=False, repr=False, eq=False)

    @classmethod
    def from_solver(
        cls,
        solver: Union["Solver", "BatchSolverKernel"],
        results_type: str = "full",
        nan_error_trajectories: bool = True,
    ) -> Union["SolveResult", dict[str, Any]]:
        """Create a :class:`SolveResult` from a solver instance.

        Parameters
        ----------
        solver
            Object providing access to output arrays and metadata.
        results_type
            Format of the returned results. Options are ``"full"``, ``"numpy"``,
            ``"numpy_per_summary"``, ``raw``, and ``"pandas"``. Defaults to
            ``"full"``. ``raw`` returns independent spill-aware copies without
            legends or supporting information.
        nan_error_trajectories
            When ``True`` (default), trajectories with nonzero status codes
            are set to NaN. When ``False``, all trajectories are returned
            with original values regardless of status. This parameter is
            ignored when ``results_type`` is ``"raw"``.

        Returns
        -------
        SolveResult or dict[str, Any]
            ``SolveResult`` when ``results_type`` is ``"full"``; otherwise a
            dictionary containing the requested representation.
        """
        if results_type == "raw":
            arrays = {
                "state": cls._copy_result_array(solver, solver.state),
                "observables": cls._copy_result_array(
                    solver, solver.observables
                ),
                "state_summaries": cls._copy_result_array(
                    solver, solver.state_summaries
                ),
                "observable_summaries": cls._copy_result_array(
                    solver, solver.observable_summaries
                ),
                "iteration_counters": cls._copy_result_array(
                    solver, solver.iteration_counters
                ),
                "status_codes": cls._copy_result_array(
                    solver, solver.status_codes
                ),
            }
            return RawSolveResult(arrays, solver.kernel.memory_manager)
        active_outputs = solver.active_outputs
        state_active = active_outputs.state
        observables_active = active_outputs.observables
        state_summaries_active = active_outputs.state_summaries
        observable_summaries_active = active_outputs.observable_summaries
        solve_settings = solver.solve_info

        # Retrieve status codes for non-raw results
        status_codes = cls._copy_result_array(solver, solver.status_codes)
        iteration_counters = cls._copy_result_array(
            solver, solver.iteration_counters
        )
        time, state_less_time = cls.cleave_time(
            solver.state,
            time_saved=solver.save_time,
            stride_order=solver.kernel.output_arrays.host.state.stride_order,
        )
        if time is not None:
            time = cls._copy_result_array(solver, time)

        time_domain_array = cls._combine_result_arrays(
            solver,
            (state_less_time, solver.observables),
            (state_active, observables_active),
        )

        summaries_array = cls._combine_result_arrays(
            solver,
            (solver.state_summaries, solver.observable_summaries),
            (state_summaries_active, observable_summaries_active),
        )

        # Process error trajectories when enabled
        if (
            nan_error_trajectories
            and status_codes is not None
            and status_codes.size > 0
        ):
            # Find runs with nonzero status codes
            error_run_indices = np_where(status_codes != 0)[0]

            if len(error_run_indices) > 0:
                # Get stride order and find run dimension
                stride_order = (
                    solver.kernel.output_arrays.host.state.stride_order
                )
                run_index = stride_order.index("run")

                # Set error trajectories to NaN using vectorized indexing
                if time_domain_array.size > 0:
                    if run_index == 0:
                        time_domain_array[error_run_indices, :, :] = np_nan
                    elif run_index == 1:
                        time_domain_array[:, error_run_indices, :] = np_nan
                    else:  # run_index == 2
                        time_domain_array[:, :, error_run_indices] = np_nan

                if summaries_array.size > 0:
                    if run_index == 0:
                        summaries_array[error_run_indices, :, :] = np_nan
                    elif run_index == 1:
                        summaries_array[:, error_run_indices, :] = np_nan
                    else:  # run_index == 2
                        summaries_array[:, :, error_run_indices] = np_nan

        time_domain_legend = cls.time_domain_legend_from_solver(solver)

        summaries_legend = cls.summary_legend_from_solver(solver)
        singlevar_summary_legend = solver.summary_legend_per_variable

        user_arrays = cls(
            time_domain_array=time_domain_array,
            summaries_array=summaries_array,
            time=time,
            iteration_counters=iteration_counters,
            status_codes=status_codes,
            time_domain_legend=time_domain_legend,
            summaries_legend=summaries_legend,
            active_outputs=active_outputs,
            solve_settings=solve_settings,
            stride_order=solver.kernel.output_arrays.host.state.stride_order,
            singlevar_summary_legend=singlevar_summary_legend,
            memory_manager=solver.kernel.memory_manager,
        )

        if results_type == "full":
            return user_arrays
        try:
            if results_type == "numpy":
                result = user_arrays.as_numpy
            elif results_type == "numpy_per_summary":
                result = user_arrays.as_numpy_per_summary
            elif results_type == "pandas":
                result = user_arrays.as_pandas
            else:
                return user_arrays
        finally:
            if results_type in ("numpy", "numpy_per_summary", "pandas"):
                user_arrays.close()
        return result

    def close(self) -> None:
        """Release spill files owned by this result."""
        if self._closed:
            return
        arrays = (
            self.time_domain_array,
            self.summaries_array,
            self.time,
            self.iteration_counters,
            self.status_codes,
        )
        if self._memory_manager is not None:
            _release_spill_arrays(self._memory_manager, arrays)
        self.time_domain_array = np_array([])
        self.summaries_array = np_array([])
        self.time = None
        self.iteration_counters = np_array([])
        self.status_codes = None
        self._closed = True

    def __enter__(self) -> "SolveResult":
        """Return this result."""
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Release spill files on context exit."""
        self.close()

    @staticmethod
    def _copy_result_array(solver, source: ndarray) -> ndarray:
        """Copy an output without forcing a spill file into RAM."""
        manager = solver.kernel.memory_manager
        owner = solver.kernel.output_arrays
        memory_type = "memmap" if isinstance(source, np_memmap) else "host"
        result = manager.create_host_array(
            source.shape,
            source.dtype,
            memory_type,
            instance=owner,
        )
        source_flat = source.reshape(-1)
        result_flat = result.reshape(-1)
        block_length = max(1, HOST_STAGING_BYTES // source.dtype.itemsize)
        for start in range(0, source_flat.size, block_length):
            stop = min(start + block_length, source_flat.size)
            result_flat[start:stop] = source_flat[start:stop]
        return result

    @classmethod
    def _combine_result_arrays(cls, solver, arrays, enabled) -> ndarray:
        """Combine enabled outputs into a spill-aware result array."""
        active = [array for array, use in zip(arrays, enabled) if use]
        if not active:
            return np_array([])
        if len(active) == 1:
            return cls._copy_result_array(solver, active[0])

        shape = list(active[0].shape)
        shape[1] = sum(array.shape[1] for array in active)
        manager = solver.kernel.memory_manager
        owner = solver.kernel.output_arrays
        memory_type = (
            "memmap"
            if any(isinstance(array, np_memmap) for array in active)
            else "host"
        )
        result = manager.create_host_array(
            tuple(shape),
            active[0].dtype,
            memory_type,
            instance=owner,
        )
        row_bytes = max(1, result[0].nbytes)
        rows = max(1, HOST_STAGING_BYTES // row_bytes)
        offset = 0
        for source in active:
            target = slice(offset, offset + source.shape[1])
            for start in range(0, source.shape[0], rows):
                stop = min(start + rows, source.shape[0])
                result[start:stop, target, ...] = source[start:stop]
            offset += source.shape[1]
        return result

    @property
    def as_pandas(self) -> dict[str, "pd.DataFrame"]:
        """Convert the results to pandas DataFrames.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Dictionary containing ``time_domain`` and ``summaries`` DataFrames.

        Raises
        ------
        ImportError
            Raised when pandas is not available.

        Notes
        -----
        Pandas is an optional dependency that is imported lazily.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is required to convert SolveResult to DataFrames. "
                "Pandas is an optional dependency- it's only used here "
                "to make analysis of data easier. Install Pandas to "
                "use this feature."
            )

        run_index = self._stride_order.index("run")
        ndim = len(self._stride_order)
        time_dfs = []
        summaries_dfs = []
        any_summaries = (
            self.active_outputs.state_summaries
            or self.active_outputs.observable_summaries
        )

        n_runs = self.time_domain_array.shape[run_index] if ndim == 3 else 1
        time_headings = list(self.time_domain_legend.values())
        summary_headings = list(self.summaries_legend.values())

        # Resolve time index once (use first run's time for multi-run)
        time_index = None
        if self.time is not None and self.time.size > 0:
            if self.time.ndim > 1:
                time_index = np_array(
                    self.time[:, 0], copy=True, subok=False
                )
            else:
                time_index = np_array(self.time, copy=True, subok=False)

        for run in range(n_runs):
            run_slice = slice_variable_dimension(
                slice(run, run + 1, None), run_index, ndim
            )

            singlerun_array = np_array(
                np_squeeze(
                    self.time_domain_array[run_slice], axis=run_index
                ),
                copy=True,
                subok=False,
            )
            df = pd.DataFrame(singlerun_array, columns=time_headings)

            # Create MultiIndex columns with run number as first level
            df.columns = pd.MultiIndex.from_product(
                [[f"run_{run}"], df.columns]
            )
            time_dfs.append(df)

            if any_summaries:
                singlerun_array = np_array(
                    np_squeeze(
                        self.summaries_array[run_slice], axis=run_index
                    ),
                    copy=True,
                    subok=False,
                )
                df = pd.DataFrame(singlerun_array, columns=summary_headings)
                summaries_dfs.append(df)
                df.columns = pd.MultiIndex.from_product(
                    [[f"run_{run}"], df.columns]
                )
            else:
                summaries_dfs.append(pd.DataFrame())

        time_domain_df = pd.concat(time_dfs, axis=1)
        summaries_df = pd.concat(summaries_dfs, axis=1)

        # Set time index after concat to avoid reindexing errors from
        # float32 duplicate values during alignment
        if time_index is not None:
            time_domain_df.index = time_index

        return {"time_domain": time_domain_df, "summaries": summaries_df}

    @property
    def as_numpy(self) -> dict[str, Optional[NDArray]]:
        """
        Return the results as copies of NumPy arrays.

        Returns
        -------
        dict[str, Optional[NDArray]]
            Dictionary containing copies of time, time_domain_array,
            summaries_array, time_domain_legend, summaries_legend, and
            iteration_counters.
        """
        return {
            "time": (
                np_array(self.time, copy=True, subok=False)
                if self.time is not None
                else None
            ),
            "time_domain_array": np_array(
                self.time_domain_array, copy=True, subok=False
            ),
            "summaries_array": np_array(
                self.summaries_array, copy=True, subok=False
            ),
            "time_domain_legend": self.time_domain_legend.copy(),
            "summaries_legend": self.summaries_legend.copy(),
            "iteration_counters": np_array(
                self.iteration_counters, copy=True, subok=False
            ),
        }

    @property
    def as_numpy_per_summary(self) -> dict[str, Optional[NDArray]]:
        """
        Return the results as separate NumPy arrays per summary type.

        Returns
        -------
        dict[str, Optional[NDArray]]
            Dictionary containing time, time_domain_array, time_domain_legend,
            iteration counters, and individual summary arrays.
        """
        arrays = {
            "time": (
                np_array(self.time, copy=True, subok=False)
                if self.time is not None
                else None
            ),
            "time_domain_array": np_array(
                self.time_domain_array, copy=True, subok=False
            ),
            "time_domain_legend": self.time_domain_legend.copy(),
            "iteration_counters": np_array(
                self.iteration_counters, copy=True, subok=False
            ),
        }
        arrays.update(**self.per_summary_arrays)

        return arrays

    @property
    def per_summary_arrays(self) -> dict[str, NDArray]:
        """
        Split summaries_array into separate arrays keyed by summary type.

        Returns
        -------
        dict[str, NDArray]
            Dictionary where each key is a summary type and the value is the
            corresponding NumPy array. The dictionary also includes a key
            'summary_legend' mapping to the variable legend.
        """
        if (
            self._active_outputs.state_summaries is False
            and self._active_outputs.observable_summaries is False
        ):
            return {}

        variable_index = self._stride_order.index("variable")

        # Split summaries_array by type
        variable_legend = self.time_domain_legend
        singlevar_legend = self._singlevar_summary_legend
        indices_per_var = np_amax([k for k in singlevar_legend.keys()]) + 1
        per_summary_arrays = {}

        for offset, label in singlevar_legend.items():
            summ_slice = slice(offset, None, indices_per_var)
            summ_slice = slice_variable_dimension(
                summ_slice, variable_index, len(self._stride_order)
            )
            per_summary_arrays[label] = np_array(
                self.summaries_array[summ_slice], copy=True, subok=False
            )
        per_summary_arrays["summary_legend"] = variable_legend

        return per_summary_arrays

    @property
    def active_outputs(self) -> ActiveOutputs:
        """Return the active output flags."""
        return self._active_outputs

    @property
    def status_messages(self) -> dict[int, List[str]]:
        """Decode nonzero run status codes into named result flags.

        Returns
        -------
        dict[int, list[str]]
            Mapping from run index to the list of
            :class:`~cubie.result_codes.CUBIE_RESULT_CODES` member names set
            in that run's status word. Runs that completed successfully
            (status ``0``) are omitted, so an empty mapping means every run
            succeeded.
        """
        return decode_status_codes(self.status_codes)

    @staticmethod
    def cleave_time(
        state: ArrayTypes,
        time_saved: bool = False,
        stride_order: Optional[Tuple[str, ...]] = None,
    ) -> tuple[Optional[NDArray], NDArray]:
        """Remove time from the state array when present.

        Parameters
        ----------
        state
            State array potentially containing a time column.
        time_saved
            Flag indicating if time is saved in the state array.
        stride_order
            Optional order of dimensions in the array. Defaults to
            ``["time", "variable", "run"]`` when ``None``.

        Returns
        -------
        tuple[Optional[NDArray], NDArray]
            Pair containing the time array (or ``None``) and the state array
            with time removed.
        """
        if stride_order is None:
            stride_order = ["time", "variable", "run"]
        if time_saved:
            var_index = stride_order.index("variable")
            ndim = len(state.shape)

            time_slice = slice_variable_dimension(
                slice(-1, None, None), var_index, ndim
            )
            state_slice = slice_variable_dimension(
                slice(None, -1), var_index, ndim
            )

            time = np_squeeze(state[time_slice], axis=var_index)
            state_less_time = state[state_slice]
            return time, state_less_time
        else:
            return None, state

    @staticmethod
    def combine_time_domain_arrays(
        state: ArrayTypes,
        observables: ArrayTypes,
        state_active: bool = True,
        observables_active: bool = True,
    ) -> NDArray:
        """Combine state and observable arrays into a single time-domain array.

        Parameters
        ----------
        state
            Array of state values.
        observables
            Array of observable values.
        state_active
            Flag indicating if state values are active.
        observables_active
            Flag indicating if observable values are active.

        Returns
        -------
        NDArray
            Combined array along the variable axis (axis=1) for 3D arrays,
            or a copy of the active array.
        """
        if state_active and observables_active:
            # Concatenate along variable axis (axis=1) for (time, variable, run)
            return np_concatenate((state, observables), axis=1)
        elif state_active:
            return state.copy()
        elif observables_active:
            return observables.copy()
        else:
            return np_array([])

    @staticmethod
    def combine_summaries_array(
        state_summaries: ArrayTypes,
        observable_summaries: ArrayTypes,
        summarise_states: bool,
        summarise_observables: bool,
    ) -> ndarray:
        """Combine state and observable summary arrays into a single array.

        Parameters
        ----------
        state_summaries
            Array containing state summaries.
        observable_summaries
            Array containing observable summaries.
        summarise_states
            Flag indicating if state summaries are active.
        summarise_observables
            Flag indicating if observable summaries are active.

        Returns
        -------
        ndarray
            Combined summary array along the variable axis (axis=1).
        """
        if summarise_states and summarise_observables:
            # Concatenate along variable axis (axis=1) for (time, variable, run)
            return np_concatenate(
                (state_summaries, observable_summaries), axis=1
            )
        elif summarise_states:
            return state_summaries.copy()
        elif summarise_observables:
            return observable_summaries.copy()
        else:
            return np_array([])

    @staticmethod
    def summary_legend_from_solver(solver: "Solver") -> dict[int, str]:
        """Generate a summary legend from the solver instance.

        Parameters
        ----------
        solver
            Solver instance providing saved states, observables, and summary
            legends.

        Returns
        -------
        dict[int, str]
            Dictionary mapping summary array indices to labels with units.
        """
        singlevar_legend = solver.summary_legend_per_variable
        unit_modifications = solver.summary_unit_modifications
        state_labels = solver.summarised_states
        obs_labels = solver.summarised_observables
        summaries_legend = {}

        state_units = {}
        obs_units = {}

        if hasattr(solver.system, "state_units"):
            state_units = solver.system.state_units
        if hasattr(solver.system, "observable_units"):
            obs_units = solver.system.observable_units

        # state summaries_array
        for i, label in enumerate(state_labels):
            unit = state_units.get(label, "dimensionless")
            for j, (key, summary_type) in enumerate(singlevar_legend.items()):
                index = i * len(singlevar_legend) + j
                unit_mod = unit_modifications.get(j, "[unit]")

                # Apply unit modification and format legend
                if unit != "dimensionless":
                    # Replace 'unit' placeholder (not '[unit]') to preserve brackets
                    modified_unit = unit_mod.replace("unit", unit)
                    summaries_legend[index] = (
                        f"{label} {modified_unit} {summary_type}"
                    )
                else:
                    summaries_legend[index] = f"{label} {summary_type}"

        # observable summaries_array
        len_state_legend = len(state_labels) * len(singlevar_legend)
        for i, label in enumerate(obs_labels):
            unit = obs_units.get(label, "dimensionless")
            for j, (key, summary_type) in enumerate(singlevar_legend.items()):
                index = len_state_legend + i * len(singlevar_legend) + j
                unit_mod = unit_modifications.get(j, "[unit]")

                # Apply unit modification and format legend
                if unit != "dimensionless":
                    # Replace 'unit' placeholder (not '[unit]') to preserve brackets
                    modified_unit = unit_mod.replace("unit", unit)
                    summaries_legend[index] = (
                        f"{label} {modified_unit} {summary_type}"
                    )
                else:
                    summaries_legend[index] = f"{label} {summary_type}"

        return summaries_legend

    @staticmethod
    def time_domain_legend_from_solver(solver: "Solver") -> dict[int, str]:
        """Generate a time-domain legend from the solver instance.

        Parameters
        ----------
        solver
            Solver instance providing saved states and observables.

        Returns
        -------
        dict[int, str]
            Dictionary mapping time-domain indices to labels with units.
        """
        time_domain_legend = {}
        state_labels = solver.saved_states
        obs_labels = solver.saved_observables

        state_units = {}
        obs_units = {}

        if hasattr(solver.system, "state_units"):
            state_units = solver.system.state_units
        if hasattr(solver.system, "observable_units"):
            obs_units = solver.system.observable_units

        offset = 0

        for i, label in enumerate(state_labels):
            unit = state_units.get(label, "dimensionless")
            time_domain_legend[i] = _format_time_domain_label(label, unit)

        offset = len(state_labels)
        for i, label in enumerate(obs_labels):
            unit = obs_units.get(label, "dimensionless")
            time_domain_legend[offset + i] = _format_time_domain_label(
                label, unit
            )
        return time_domain_legend
