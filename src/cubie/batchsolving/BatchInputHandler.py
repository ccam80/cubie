"""Batch input handling for state and parameter processing.

This module processes user-supplied dictionaries or arrays into the 2D NumPy
arrays expected by the batch solver. :class:`BatchInputHandler` is the primary
entry point and is usually accessed through :class:`cubie.batchsolving.solver.Solver`.

The handler provides:
- Input classification to determine optimal processing paths
- Array validation and precision casting
- Grid construction for combinatorial or verbatim runs

Notes
-----
``BatchInputHandler.__call__`` accepts three arguments:

``states``
    Mapping or array containing state values only. One-dimensional
    inputs override defaults for every run, while two-dimensional inputs
    are treated as pre-built grids in (variable, run) format.
``params``
    Mapping or array containing parameter values only. Interpretation matches
    ``states``.
``kind``
    Controls how inputs are combined. ``"combinatorial"`` builds the
    Cartesian product, while ``"verbatim"`` preserves column-wise groupings.

When arrays are supplied directly they are treated as fully specified grids
in (variable, run) format where each column represents a run configuration.
Dictionary inputs trigger combinatorial expansion before assembly so
that every value combination is represented in the resulting grid.

``BatchInputHandler.__call__`` processes states and params through
independent paths, combining only at the final alignment step:

1. Each input is processed via ``_process_input()`` to produce
   a 2D array in (variable, run) format
2. Arrays are aligned via ``_align_run_counts()`` using the
   specified ``kind`` strategy
3. Results are cast to system precision

This architecture keeps states and params separate throughout,
improving code clarity and reducing unnecessary transformations.

Examples
--------
>>> import numpy as np
>>> import cubie as qb
>>> from cubie.batchsolving.BatchInputHandler import BatchInputHandler
>>> system = qb.create_ODE_system(
...    dxdt=["dx = p0 * p1 * y","dy = p1 * x"],
...    parameters = {'p0': 2.0, 'p1': 1.5}
... )
>>> handler = BatchInputHandler.from_system(system)
>>> params = {"p0": [0.1, 0.2], "p1": [10, 20]}
>>> states = {"x0": [1.0, 2.0], "x1": [0.5, 1.5]}
>>> inits, params = handler(
...     states=states, params=params, kind="combinatorial"
... )
>>> print(inits.shape)
(2, 16)
>>> print(inits)
[[1.  1.  1.  1.  1.  1.  1.  1.  2.  2.  2.  2.  2.  2.  2.  2. ]
 [0.5 0.5 0.5 0.5 1.5 1.5 1.5 1.5 0.5 0.5 0.5 0.5 1.5 1.5 1.5 1.5]]
>>> print(params.shape)
(2, 16)
>>> print(params)
[[ 0.1  0.1  0.2  0.2  0.1  0.1  0.2  0.2  0.1  0.1  0.2  0.2  0.1  0.1  0.2  0.2]
 [10.  20.  10.  20.  10.  20.  10.  20.  10.  20.  10.  20.  10.  20.  10.  20. ]]

Example 2: verbatim arrays

>>> params = np.array([[0.1, 0.2], [10, 20]])
>>> states = np.array([[1.0, 2.0], [0.5, 1.5]])
>>> inits, params = handler(states=states, params=params, kind="verbatim")
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

>>> inits, params = handler(
...     states=states, params=params, kind="combinatorial"
... )
>>> print(inits.shape)
(2, 4)
>>> print(inits)
[[1.  1.  2.  2. ]
 [0.5 0.5 1.5 1.5]]
>>> print(params.shape)
(2, 4)
>>> print(params)
[[ 0.1  0.2  0.1  0.2]
 [10.  20.  10.  20. ]]

Example 3: single parameter sweep (unspecified filled with defaults)

>>> params = {"p0": [0.1, 0.2]}
>>> inits, params = handler(params=params, kind="combinatorial")
>>> print(inits.shape)
(2, 2)
>>> print(inits)  # unspecified variables are filled with defaults from system
[[1. 1.]
 [1. 1.]]
>>> print(params.shape)
(2, 2)
>>> print(params)
[[0.1 0.2]
 [2.  2. ]]
"""
from itertools import product
from typing import Dict, List, Optional, Tuple, Union
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
        Two-dimensional array in (variable, run) format where each column
        is a unique combination of the supplied values.

    Notes
    -----
    Duplicate elements are removed by constructing an ordered dictionary per
    input array. ``itertools.product`` then generates the Cartesian product
    of the deduplicated inputs.

    Examples
    --------
    >>> unique_cartesian_product([np.array([1, 2, 2]), np.array([3, 4])])
    array([[1, 1, 2, 2],
           [3, 4, 3, 4]])
    """
    deduplicated_inputs = [
        list(dict.fromkeys(a)) for a in arrays
    ]  # preserve order, remove dups
    # Build array in (variable, run) format: rows are variables, columns runs
    return np.array([list(t) for t in product(*deduplicated_inputs)]).T


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
        Value array is in (variable, run) format.

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
     array([[ 0.1,  0.1,  0.2,  0.2,  0.3,  0.3],
            [10. , 20. , 10. , 20. , 10. , 20. ]]))
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
        Value array is in (variable, run) format.

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
     array([[ 0.1,  0.2,  0.3],
            [10. , 20. , 30. ]]))
    """
    cleaned_request = {
        k: v for k, v in request.items() if np.asarray(v).size > 0
    }
    indices = values_instance.get_indices(
        list(cleaned_request.keys()), silent=silent
    )
    # Build in (variable, run) format: rows are swept variables, columns runs
    combos = np.asarray([item for item in cleaned_request.values()])
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
        Value array is in (variable, run) format.

    Notes
    -----
    ``kind`` selects between :func:`combinatorial_grid` and
    :func:`verbatim_grid`.
    """
    # When kind == 'combinatorial' use combinatorial expansion of values
    if kind == "combinatorial":
        return combinatorial_grid(request, values_instance, silent=silent)
    # When kind == 'verbatim' preserve row-wise groupings without expansion
    elif kind == "verbatim":
        return verbatim_grid(request, values_instance, silent=silent)
    # Any other kind is invalid
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
        First grid in (variable, run) format, typically parameters.
    grid2
        Second grid in (variable, run) format, typically initial states.
    kind
        ``"combinatorial"`` builds the Cartesian product and
        ``"verbatim"`` pairs columns directly.

    Returns
    -------
    tuple of np.ndarray and np.ndarray
        Extended versions of ``grid1`` and ``grid2`` in (variable, run)
        format aligned to the chosen strategy.

    Raises
    ------
    ValueError
        Raised when ``kind`` is ``"verbatim"`` and the inputs have different
        run counts or when ``kind`` is unknown.
    """
    # For 'combinatorial' return the Cartesian product of runs (columns)
    if kind == "combinatorial":
        # Cartesian product: all combinations of runs from each grid
        # Repeat each column of grid1 for each column in grid2
        g1_repeat = np.repeat(grid1, grid2.shape[1], axis=1)
        # Tile grid2 columns for each column in grid1
        g2_tile = np.tile(grid2, (1, grid1.shape[1]))
        return g1_repeat, g2_tile
    # For 'verbatim' pair columns directly and error if run counts differ
    elif kind == "verbatim":
        # Capture original sizes before any broadcast
        g1_runs = grid1.shape[1]
        g2_runs = grid2.shape[1]
        # Broadcast single-run grids to match the other grid's size
        if g1_runs == 1 and g2_runs > 1:
            grid1 = np.repeat(grid1, g2_runs, axis=1)
        elif g2_runs == 1 and g1_runs > 1:
            grid2 = np.repeat(grid2, g1_runs, axis=1)
        # After broadcasting, check dimensions match
        if grid1.shape[1] != grid2.shape[1]:
            raise ValueError(
                "For 'verbatim', both grids must have the same number "
                "of runs, or one grid must have exactly 1 run so it can be "
                "broadcast to match the other."
            )
        return grid1, grid2
    # Any other kind is invalid
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
        Two-dimensional array of gridded parameter values in (variable, run)
        format.
    indices
        One-dimensional array describing which parameter indices were swept.
    default_values
        One-dimensional array of default parameter values.

    Returns
    -------
    np.ndarray
        Two-dimensional array in (variable, run) format containing complete
        parameter values for each run.

    Raises
    ------
    ValueError
        Raised when ``grid`` row count does not match ``indices`` length.
    """
    # Handle empty indices: no variables swept, return defaults for all runs
    if indices.size == 0:
        n_runs = grid.shape[1] if grid.ndim > 1 else 1
        return np.tile(default_values[:, np.newaxis], (1, n_runs))

    # If grid is 1D it represents a single column of default values
    if grid.ndim == 1:
        array = default_values[:, np.newaxis]
    else:
        # When multidimensional ensure the grid row count matches indices
        if grid.shape[0] != indices.shape[0]:
            raise ValueError("Grid shape does not match indices shape.")
        if default_values.shape[0] == indices.shape[0]:
            # All indices swept, just pass the array straight through
            array = grid
        else:
            # Create array with default values for all runs
            n_runs = grid.shape[1]
            array = np.column_stack([default_values] * n_runs)
            array[indices, :] = grid

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
        Two-dimensional array in (variable, run) format with complete
        parameter values for each run.
    """
    indices, grid = generate_grid(request, values_instance, kind=kind)
    return extend_grid_to_array(grid, indices, values_instance.values_array)


class BatchInputHandler:
    """Process and validate solver inputs for batch runs.

    The handler converts dictionaries or arrays into solver-ready
    two-dimensional arrays, classifies input types for optimal
    processing paths, and validates array shapes and dtypes.

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
    precision
        Floating-point precision for returned arrays.
    """

    def __init__(self, interface: SystemInterface):
        """Initialise the handler with a system interface."""
        self.parameters = interface.parameters
        self.states = interface.states
        self.precision = interface.parameters.precision

    @classmethod
    def from_system(cls, system: BaseODE) -> "BatchInputHandler":
        """Create a handler from a system model.

        Parameters
        ----------
        system
            System model providing parameter and state metadata.

        Returns
        -------
        BatchInputHandler
            Handler configured for ``system``.
        """
        interface = SystemInterface.from_system(system)
        return cls(interface)

    def __call__(
        self,
        states: Optional[Union[Dict, ArrayLike]] = None,
        params: Optional[Union[Dict, ArrayLike]] = None,
        kind: str = "combinatorial",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process user input to generate state and parameter arrays.

        Parameters
        ----------
        states
            Optional dictionary or array describing initial state sweeps.
        params
            Optional dictionary or array describing parameter sweeps.
        kind
            Strategy for grid assembly. ``"combinatorial"`` expands
            all combinations while ``"verbatim"`` preserves pairings.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Initial state and parameter arrays aligned for batch execution.

        Notes
        -----
        Passing ``states`` and ``params`` as arrays treats each as a
        complete grid. ``kind="combinatorial"`` computes the Cartesian
        product of both grids. When arrays already describe paired runs,
        set ``kind`` to ``"verbatim"`` to keep them aligned.
        """
        # Update precision from current system state
        self.precision = self.states.precision

        # Fast path - if right-sized arrays, return straight away.
        if self._are_right_sized_arrays(states, params):
            # Handle empty parameters case
            if self.parameters.empty and params is None and states is not None:
                n_runs = states.shape[1]
                params = np.empty((0, n_runs), dtype=self.precision)
            return self._cast_to_precision(states, params)

        # Fast path arrays - if a single right-sized array and a None,
        # or 1d array-like, extend the small one and return quickly.
        fast_result = self._try_fast_path_arrays(states, params, kind)
        if fast_result is not None:
            return fast_result

        # Process each category independently
        states_array = self._process_input(states, self.states, kind)
        params_array = self._process_input(params, self.parameters, kind)

        # Align run counts
        states_array, params_array = self._align_run_counts(
            states_array, params_array, kind
        )

        # Cast to system precision
        return self._cast_to_precision(states_array, params_array)

    def classify_inputs(
        self,
        states: Union[np.ndarray, Dict[str, Union[float, np.ndarray]], None],
        params: Union[np.ndarray, Dict[str, Union[float, np.ndarray]], None],
    ) -> str:
        """Classify input types to determine optimal processing path.

        Parameters
        ----------
        states
            Initial state values as dict, array, or None.
        params
            Parameter values as dict, array, or None.

        Returns
        -------
        str
            Classification: 'dict', 'array', or 'device'.

        Notes
        -----
        Returns 'dict' when either input is a dictionary, triggering
        full grid construction. Returns 'array' when both inputs are
        numpy arrays with matching run counts in (n_vars, n_runs) format.
        Returns 'device' when both have __cuda_array_interface__.
        """
        # If either input is a dict, use grid builder path
        if isinstance(states, dict) or isinstance(params, dict):
            return 'dict'

        # Check for device arrays (CUDA arrays with interface)
        states_is_device = hasattr(states, '__cuda_array_interface__')
        params_is_device = hasattr(params, '__cuda_array_interface__')
        if states_is_device and params_is_device:
            return 'device'

        # Check for numpy arrays with correct shapes
        if isinstance(states, np.ndarray) and isinstance(params, np.ndarray):
            # Must be 2D arrays in (n_vars, n_runs) format
            if states.ndim == 2 and params.ndim == 2:
                n_states = self.states.n
                n_params = self.parameters.n
                # Verify variable counts match system expectations
                if (states.shape[0] == n_states and
                        params.shape[0] == n_params):
                    # Verify run counts match
                    if states.shape[1] == params.shape[1]:
                        return 'array'

        # Default to dict path (grid builder handles conversion)
        return 'dict'

    def validate_arrays(
        self,
        states: np.ndarray,
        params: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and prepare pre-built arrays for kernel execution.

        Parameters
        ----------
        states
            Initial state array in (n_states, n_runs) format.
        params
            Parameter array in (n_params, n_runs) format.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Validated arrays cast to system precision in (states, params)
            order.

        Notes
        -----
        Arrays are cast to the system precision dtype when needed.
        Returned as contiguous arrays for optimal kernel performance.
        """
        # Update precision from current system state
        self.precision = self.states.precision

        # Cast to correct dtype if needed
        if states.dtype != self.precision:
            states = states.astype(self.precision, copy=False)
        if params.dtype != self.precision:
            params = params.astype(self.precision, copy=False)

        # Ensure contiguous layout for optimal kernel performance
        states = np.ascontiguousarray(states)
        params = np.ascontiguousarray(params)

        return states, params

    def _trim_or_extend(
        self, arr: np.ndarray, values_object: SystemValues
    ) -> np.ndarray:
        """Extend incomplete arrays with defaults or trim extra values.

        Parameters
        ----------
        arr
            Array in (variable, run) format requiring adjustment.
        values_object
            System values object containing defaults and dimension metadata.

        Returns
        -------
        np.ndarray
            Array in (variable, run) format whose row count matches
            ``values_object.n``.
        """
        # If the array has fewer rows than the number of values, extend it
        # with default values
        if arr.shape[0] < values_object.n:
            n_runs = arr.shape[1]
            # Create padding with default values for missing variables
            padding = np.tile(
                values_object.values_array[arr.shape[0]:, np.newaxis],
                (1, n_runs)
            )
            arr = np.vstack([arr, padding])
        # If the array has more rows than expected, trim the extras
        elif arr.shape[0] > values_object.n:
            arr = arr[:values_object.n, :]
        return arr

    def _sanitise_arraylike(
        self, arr: Optional[ArrayLike], values_object: SystemValues
    ) -> Optional[np.ndarray]:
        """Convert array-likes to 2D arrays in (variable, run) format.

        Parameters
        ----------
        arr
            Array-like data describing sweep values. If 2D, expected in
            (variable, run) format.
        values_object
            System values object containing defaults and dimension metadata.

        Returns
        -------
        Optional[np.ndarray]
            Two-dimensional array in (variable, run) format sized to
            ``values_object`` or ``None`` when no data remain after
            sanitisation.

        Raises
        ------
        ValueError
            Raised when the input has more than two dimensions.

        Warns
        -----
        UserWarning
            Warned when the number of provided rows differs from the
            expected dimension.
        """
        # If no array provided, pass through None
        if arr is None:
            return arr
        # If the input is not already an ndarray, coerce it to one
        elif not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        # Reject inputs with more than two dimensions explicitly
        if arr.ndim > 2:
            raise ValueError(
                f"Input must be a 1D or 2D array, but got a {arr.ndim}D array."
            )
        # Convert 1D vectors to single-column 2D arrays (one run)
        elif arr.ndim == 1:
            arr = arr[:, np.newaxis]

        # Warn and adjust arrays whose row count differs from expected
        if arr.shape[0] != values_object.n:
            warn(
                f"Provided input data has {arr.shape[0]} variables, but there "
                f"are {values_object.n} settable values. Missing values "
                f"will be filled with default values, and extras ignored."
            )
            arr = self._trim_or_extend(arr, values_object)
        # Empty arrays collapse to None
        if arr.size == 0:
            return None

        return arr  # correctly sized array just falls through untouched

    def _process_input(
        self,
        input_data: Optional[Union[Dict, ArrayLike]],
        values_object: SystemValues,
        kind: str,
    ) -> np.ndarray:
        """Process a single input category to a 2D array.

        Handles None, dict, or array-like inputs for either params
        or states, returning a complete 2D array in (variable, run)
        format with all variables included.

        Parameters
        ----------
        input_data
            Input as None (use defaults), dict (expand to grid),
            or array-like (sanitize).
        values_object
            SystemValues instance for this category (params or states).
        kind
            Grid type: "combinatorial" or "verbatim".

        Returns
        -------
        np.ndarray
            2D array in (variable, run) format with all variables.

        Raises
        ------
        TypeError
            Raised when input_data is not None, dict, or array-like.
        ValueError
            Raised when non-empty input_data is provided but values_object
            has no variables (is empty).
        """
        # Handle empty SystemValues (system has no variables of this type)
        if values_object.empty:
            if input_data is not None:
                # Check if input is truly empty or has actual data
                is_empty_input = False
                if isinstance(input_data, dict) and len(input_data) == 0:
                    is_empty_input = True
                elif isinstance(input_data, np.ndarray) and input_data.size == 0:
                    is_empty_input = True
                elif isinstance(input_data, (list, tuple)) and len(input_data) == 0:
                    is_empty_input = True

                if not is_empty_input:
                    raise ValueError(
                        f"Grid values were provided but the system has no "
                        f"settable variables of this type. Expected None or "
                        f"empty input, got {type(input_data).__name__}."
                    )
            # Return empty 2D array with 0 rows and 1 column
            return np.empty((0, 1), dtype=values_object.precision)

        # None -> single-column defaults
        if input_data is None:
            return values_object.values_array[:, np.newaxis]

        # Dict -> expand to grid, extend with defaults
        if isinstance(input_data, dict):
            # Ensure all values are iterable by wrapping scalars
            input_data = {k: np.atleast_1d(v) for k, v in input_data.items()}
            indices, grid = generate_grid(input_data, values_object, kind=kind)
            return extend_grid_to_array(
                grid, indices, values_object.values_array
            )

        # Array-like -> sanitize to 2D
        if isinstance(input_data, (list, tuple, np.ndarray)):
            sanitised = self._sanitise_arraylike(input_data, values_object)
            if sanitised is None:
                # Treat empty inputs like None: use single-column defaults
                return values_object.values_array[:, np.newaxis]
            return sanitised

        # Unsupported type
        raise TypeError(
            f"Input must be None, dict, or array-like, got {type(input_data)}"
        )

    def _align_run_counts(
        self,
        states_array: np.ndarray,
        params_array: np.ndarray,
        kind: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align run counts between states and params arrays.

        For combinatorial: computes Cartesian product of runs.
        For verbatim: pairs directly (single-run broadcasts).

        Parameters
        ----------
        states_array
            States in (variable, run) format.
        params_array
            Params in (variable, run) format.
        kind
            Grid type: "combinatorial" or "verbatim".

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Aligned (states_array, params_array) with matching run counts.
        """
        return combine_grids(states_array, params_array, kind=kind)

    def _cast_to_precision(
        self, states: np.ndarray, params: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cast state and parameter arrays to the system precision.

        Parameters
        ----------
        states
            Initial state array in (variable, run) format.
        params
            Parameter array in (variable, run) format.

        Returns
        -------
        tuple of np.ndarray and np.ndarray
            State and parameter arrays with ``dtype`` matching
            ``self.precision``.
        """
        return (
            np.ascontiguousarray(states.astype(self.precision, copy=False)),
            np.ascontiguousarray(params.astype(self.precision, copy=False)),
        )

    def _are_right_sized_arrays(
        self,
        states: Optional[Union[ArrayLike, Dict]],
        params: Optional[Union[ArrayLike, Dict]],
    ) -> bool:
        """Check if both inputs are pre-formatted arrays ready for the solver.

        This method only returns True when both inputs are 2D numpy arrays
        with matching run counts and correct variable counts for their
        respective SystemValues objects. Returns False for None, dicts,
        or arrays that need further processing.

        Also handles the special case where a SystemValues object is empty
        (no variables), in which case None or an empty 2D array is acceptable.

        Parameters
        ----------
        states
            Initial states as array or dict.
        params
            Parameters as array, dict, or None.

        Returns
        -------
        bool
            True if both inputs are correctly sized 2D arrays with matching
            run counts.
        """
        # Handle empty parameters case: states must be right-sized array,
        # params can be None
        if self.parameters.empty:
            if isinstance(states, np.ndarray) and states.ndim == 2:
                if states.shape[0] == self.states.n:
                    if params is None:
                        return True
                    if isinstance(params, np.ndarray) and params.ndim == 2:
                        return (params.shape[0] == 0
                                and params.shape[1] == states.shape[1])
            return False

        # Normal case: both must be 2D arrays
        if isinstance(states, np.ndarray) and isinstance(params, np.ndarray):
            # Both arrays: check run counts match and arrays are system-sized
            if states.ndim != 2 or params.ndim != 2:
                return False
            states_runs = states.shape[1]
            states_variables = states.shape[0]
            params_runs = params.shape[1]
            params_variables = params.shape[0]
            if states_runs == params_runs:
                if (states_variables == self.states.n
                        and params_variables == self.parameters.n):
                    return True
        return False

    def _is_right_sized_array(
        self,
        arr: Optional[Union[ArrayLike, Dict]],
        values_object: SystemValues,
    ) -> bool:
        """Check if input is a right-sized 2D array.

        Parameters
        ----------
        arr
            Input to check.
        values_object
            SystemValues instance for dimension comparison.

        Returns
        -------
        bool
            True if arr is a 2D ndarray with correct variable count,
            or True if values_object is empty and arr is None.
        """
        # If the SystemValues is empty (no variables), consider it right-sized
        # if the array is None or an empty 2D array with 0 rows
        if values_object.empty:
            if arr is None:
                return True
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == 0:
                return True
            return False
        if not isinstance(arr, np.ndarray):
            return False
        if arr.ndim != 2:
            return False
        return arr.shape[0] == values_object.n

    def _is_1d_or_none(
        self,
        arr: Optional[Union[ArrayLike, Dict]],
    ) -> bool:
        """Check if input is None or a 1D array-like.

        Parameters
        ----------
        arr
            Input to check.

        Returns
        -------
        bool
            True if arr is None or a 1D array-like (list, tuple, 1D ndarray).
        """
        if arr is None:
            return True
        if isinstance(arr, dict):
            return False
        if isinstance(arr, np.ndarray):
            return arr.ndim == 1
        if isinstance(arr, (list, tuple)):
            # Check if flat (1D) - no nested lists/tuples
            # Use hasattr('__len__') to check for iterables, excluding scalars
            return not any(
                isinstance(x, (list, tuple)) or
                (isinstance(x, np.ndarray) and x.ndim > 0)
                for x in arr
            )
        return False

    def _to_defaults_column(
        self,
        values_object: SystemValues,
        n_runs: int,
    ) -> np.ndarray:
        """Create a 2D defaults array with n_runs columns.

        Parameters
        ----------
        values_object
            SystemValues instance containing default values.
        n_runs
            Number of run columns to create.

        Returns
        -------
        np.ndarray
            2D array in (variable, run) format with defaults.
        """
        return np.tile(values_object.values_array[:, np.newaxis], (1, n_runs))

    def _try_fast_path_arrays(
        self,
        states: Optional[Union[ArrayLike, Dict]],
        params: Optional[Union[ArrayLike, Dict]],
        kind: str,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Try fast path for single right-sized array with None or 1D input.

        Parameters
        ----------
        states
            States input (array, dict, or None).
        params
            Params input (array, dict, or None).
        kind
            Grid type: "combinatorial" or "verbatim".

        Returns
        -------
        Optional[tuple[np.ndarray, np.ndarray]]
            Aligned (states, params) arrays if fast path applies, else None.
        """
        states_ok = self._is_right_sized_array(states, self.states)
        params_ok = self._is_right_sized_array(params, self.parameters)
        states_small = self._is_1d_or_none(states)
        params_small = self._is_1d_or_none(params)

        # Case 1: states is right-sized array, params is None or 1D
        if states_ok and params_small:
            n_runs = states.shape[1]
            if params is None:
                params_array = self._to_defaults_column(self.parameters, n_runs)
            else:
                # 1D array: convert to column, extend with defaults
                params_array = self._sanitise_arraylike(params, self.parameters)
                # _sanitise_arraylike guarantees shape[0] == self.parameters.n
                if params_array.shape[1] == 1:
                    params_array = np.repeat(params_array, n_runs, axis=1)
            states_array, params_array = self._align_run_counts(
                states, params_array, kind
            )
            return self._cast_to_precision(states_array, params_array)

        # Case 2: params is right-sized array, states is None or 1D
        if params_ok and states_small:
            n_runs = params.shape[1]
            if states is None:
                states_array = self._to_defaults_column(self.states, n_runs)
            else:
                # 1D array: convert to column, extend with defaults
                states_array = self._sanitise_arraylike(states, self.states)
                # _sanitise_arraylike guarantees shape[0] == self.states.n
                if states_array.shape[1] == 1:
                    states_array = np.repeat(states_array, n_runs, axis=1)
            states_array, params_array = self._align_run_counts(
                states_array, params, kind
            )
            return self._cast_to_precision(states_array, params_array)

        # Fast path doesn't apply
        return None
