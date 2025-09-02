"""
Summary metrics system for CUDA-accelerated batch integration.

This module provides the core infrastructure for summary metrics in the cubie
integrator system, including registration and function dispatch for CUDA
device functions.
"""

from typing import Optional, Callable, Union, Any
from warnings import warn
from abc import abstractmethod
import attrs

from cubie.CUDAFactory import CUDAFactory

@attrs.define
class MetricFuncCache:
    """Cache container for compiled metric functions"""
    update: Callable = attrs.field(default=None)
    save: Callable = attrs.field(default=None)

@attrs.define
class CompileSettingsPlaceholder:
    """Placeholder for compile settings"""
    empty_dict: dict = attrs.field(factory=dict)

def register_metric(registry):
    """
    Decorator factory for registering summary metrics.

    Parameters
    ----------
    registry : SummaryMetrics
        The registry instance to register the metric with.

    Returns
    -------
    callable
        Decorator function that registers a metric class.

    Notes
    -----
    This decorator automatically creates an instance of the decorated
    class and registers it with the provided registry.
    """

    def decorator(cls):
        instance = cls()
        registry.register_metric(instance)
        return cls

    return decorator


class SummaryMetric(CUDAFactory):
    """
    Abstract base class for summary metrics in the cubie integrator system.

    This class defines the interface for all summary metrics and holds memory
    requirements for buffer and output arrays, as well as dispatchers for update
    and save functions. All concrete metric implementations must inherit from
    this class and implement the abstract methods.

    Attributes
    ----------
    buffer_size : int or callable, default 0
        Size required in the summary buffer, or a callable that computes
        the size based on parameters. For parameterized metrics (like peaks),
        use a lambda function: `lambda n: 3 + n`.
    output_size : int or callable, default 0
        Size required in the output array, or a callable that computes
        the size based on parameters. For parameterized metrics (like peaks),
        use a lambda function: `lambda n: n`.
    name : str, default ""
        Name identifier for the metric (e.g., "max", "mean", "peaks").
    update_device_func : callable
        CUDA device function for updating the metric during integration.
        Property; compiled and cached when requested.
    save_device_func : callable
        CUDA device function for saving final metric results.
        Property; compiled and cached when requested.

    Notes
    -----
    All concrete implementations must:
    1. Implement build() to generate the required CUDA device
    functions
    2. Follow the exact function signatures for update and save functions
    3. Register the metric using the @register_metric(summary_metrics)
    decorator
    4. Import the metric after summary_metrics is instantiated in
    summary_metrics.__init__.py

    The CUDA device functions must have these signatures:
    - update(value, buffer, current_index, customisable_variable)
    - save(buffer, output_array, summarise_every, customisable_variable)

    Subclasses should call super().__init__() with name, buffer_size and
    output_size, and overload the build method.

    Examples
    --------
    For a simple metric with fixed sizes:
    ```python
    def __init__(self):
        super().__init__(
            name="mean",
            buffer_size=1,
            output_size=1,
        )
    ```

    For a parameterized metric:
    ```python
    def __init__(self):
        super().__init__(
            name="peaks",
            buffer_size=lambda n: 3 + n,
            output_size=lambda n: n,
        )
    ```

    Examples
    --------
    See the implemented metrics (Max, Mean, RMS, Peaks) for concrete examples
    of how to properly implement this interface.
    """

    def __init__(self,
                 buffer_size: Union[int, Callable],
                 output_size: Union[int, Callable],
                 name: str):
        super().__init__()
        self.buffer_size = buffer_size
        self.output_size = output_size
        self.name = name

        # Instantiate empty settings object for CUDAFactory compatibility
        self.setup_compile_settings(CompileSettingsPlaceholder())


    @abstractmethod
    def build(self):
        """
        Generate CUDA device functions for the metric.

        This method must be implemented by all concrete metric classes.
        It should create and return the CUDA-compiled update and save
        functions with the exact required signatures.

        Returns
        -------
        MetricFuncCache[update: callable, save: callable]
            Cache object containing update and save functions for CUDA
            execution. Both functions must be compiled with @cuda.jit
            decorators.

        Notes
        -----
        The generated functions must have these exact signatures:

        update(value, buffer, current_index, customisable_variable):
            Called during each integration step to update running
            calculations.

            Parameters:
            - value (float): New variable value to process
            - buffer (array): Buffer for storing intermediate calculations
            - current_index (int): Current integration step number
            - customisable_variable (int): Parameter for metric configuration

        save(buffer, output_array, summarise_every, customisable_variable):
            Called at summary intervals to compute final results and reset
            buffers.

            Parameters:
            - buffer (array): Buffer containing intermediate calculations
            - output_array (array): Array to store final metric results
            - summarise_every (int): Number of steps between summary saves
            - customisable_variable (int): Parameter for metric configuration

        Both functions must be decorated with:
        @cuda.jit(["float32, float32[::1], int64, int64",
                   "float64, float64[::1], int64, int64"],
                  device=True, inline=True)

        Examples
        --------
        For a simple maximum value metric:
        ```python
        def build(self):
            @cuda.jit([...], device=True, inline=True)
            def update(value, buffer, current_index, customisable_variable):
                if value > buffer[0]:
                    buffer[0] = value

            @cuda.jit([...], device=True, inline=True)
            def save(
                buffer, output_array, summarise_every, customisable_variable
            ):
                output_array[0] = buffer[0]
                buffer[0] = -1.0e30  # Reset for next period

            return MetricFuncCache(update = update, save = save)
        ```

        For a mean calculation metric:
        ```python
        def build(self):
            @cuda.jit([...], device=True, inline=True)
            def update(value, buffer, current_index, customisable_variable):
                buffer[0] += value  # Accumulate sum

            @cuda.jit([...], device=True, inline=True)
            def save(
                buffer, output_array, summarise_every, customisable_variable
            ):
                output_array[0] = buffer[0] / summarise_every
                buffer[0] = 0.0  # Reset for next period

            return MetricFuncCache(update = update, save = save)
        ```
        """
        pass

    @property
    def update_device_func(self):
        return self.get_cached_output("update")

    @property
    def save_device_func(self):
        return self.get_cached_output("save")

@attrs.define
class SummaryMetrics:
    """
    Registry and dispatcher for summary metrics.

    This class holds the complete set of implemented summary metrics and
    provides summary information to other modules. It manages memory
    layout, function dispatch, and parameter handling for all registered
    metrics.

    Attributes
    ----------
    _names : list[str]
        List of registered metric names.
    _buffer_sizes : dict[str, int or callable]
        Buffer size requirements for each metric.
    _output_sizes : dict[str, int or callable]
        Output size requirements for each metric.
    _save_functions : dict[str, callable]
        Save functions for each metric.
    _update_functions : dict[str, callable]
        Update functions for each metric.
    _metric_objects : dict[str, SummaryMetric]
        Complete metric objects for each registered metric.
    _params : dict[str, Any]
        Parameters for each metric.

    Notes
    -----
    All methods consistently return data only for requested metrics,
    not for all implemented metrics. This allows efficient compilation
    of only the needed functionality.

    The class provides:
    - implemented_metrics: List of available metric names
    - buffer_offsets/sizes: Memory layout for summary buffers
    - output_offsets/sizes: Memory layout for output arrays
    - save/update_functions: CUDA device functions for metrics
    - params: Parameter values for parameterized metrics
    """

    _names: list[str] = attrs.field(
        validator=attrs.validators.instance_of(list), factory=list, init=False
    )
    _buffer_sizes: dict[str, Union[int, Callable]] = attrs.field(
        validator=attrs.validators.instance_of(dict),
        factory=dict,
        init=False,
    )
    _output_sizes: dict[str, Union[int, Callable]] = attrs.field(
        validator=attrs.validators.instance_of(dict),
        factory=dict,
        init=False,
    )
    _metric_objects = attrs.field(
        validator=attrs.validators.instance_of(dict), factory=dict, init=False
    )
    _params: dict[str, Optional[Any]] = attrs.field(
        validator=attrs.validators.instance_of(dict),
        factory=dict,
        init=False,
    )

    def __attrs_post_init__(self):
        """Initialize the parameters dictionary."""
        self._params = {}

    def register_metric(self, metric: SummaryMetric):
        """
        Register a new summary metric with the system.

        Parameters
        ----------
        metric : SummaryMetric
            A SummaryMetric instance to register.

        Raises
        ------
        ValueError
            If a metric with the same name is already registered.

        Notes
        -----
        Once registered, the metric becomes available for use in output
        configurations and will be automatically included in update and
        save function chains when requested.
        """

        if metric.name in self._names:
            raise ValueError(f"Metric '{metric.name}' is already registered.")

        self._names.append(metric.name)
        self._buffer_sizes[metric.name] = metric.buffer_size
        self._output_sizes[metric.name] = metric.output_size
        self._metric_objects[metric.name] = metric
        self._params[metric.name] = 0

    def preprocess_request(self, request):
        """
        Parse parameters from metric specifications and validate.

        Parameters
        ----------
        request : list[str]
            List of metric specifications, potentially with parameters.

        Returns
        -------
        list[str]
            List of validated metric names with parameters extracted.

        Notes
        -----
        Parses metric specifications like 'peaks[3]' to extract parameters
        and validates that all requested metrics are registered. Issues
        warnings for unregistered metrics.
        """
        clean_request = self.parse_string_for_params(request)
        # Validate that all metrics exist and filter out unregistered ones
        validated_request = []
        for metric in clean_request:
            if metric not in self._names:
                warn(
                    f"Metric '{metric}' is not registered. Skipping.",
                    stacklevel=2,
                )
            else:
                validated_request.append(metric)
        return validated_request

    @property
    def implemented_metrics(self):
        """
        Get list of all registered summary metric names.

        Returns
        -------
        list[str]
            Names of all registered summary metrics.
        """
        return self._names

    def summaries_buffer_height(self, output_types_requested):
        """
        Calculate total buffer size for requested summary metrics.

        Parameters
        ----------
        output_types_requested : list[str]
            List of metric names to calculate total buffer size for.

        Returns
        -------
        int
            Total buffer size needed for all requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)

        offset = 0
        for metric in parsed_request:
            size = self._get_size(metric, self._buffer_sizes)
            offset += size
        return offset

    def buffer_offsets(self, output_types_requested):
        """
        Get buffer starting offsets for requested summary metrics.

        Parameters
        ----------
        output_types_requested : list[str]
            List of metric names to generate offsets for.

        Returns
        -------
        tuple[int]
            Buffer starting offsets for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)

        offset = 0
        offsets_dict = {}
        for metric in parsed_request:
            offsets_dict[metric] = offset
            size = self._get_size(metric, self._buffer_sizes)
            offset += size
        return tuple(offsets_dict[metric] for metric in parsed_request)

    def buffer_sizes(self, output_types_requested):
        """
        Get buffer sizes for requested summary metrics.

        Parameters
        ----------
        output_types_requested : list[str]
            List of metric names to generate sizes for.

        Returns
        -------
        tuple[int]
            Buffer sizes for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        return tuple(
            self._get_size(metric, self._buffer_sizes)
            for metric in parsed_request
        )

    def output_offsets(self, output_types_requested):
        """
        Get output array starting offsets for requested summary metrics.

        Parameters
        ----------
        output_types_requested : list[str]
            List of metric names to generate offsets for.

        Returns
        -------
        tuple[int]
            Output array starting offsets for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)

        offset = 0
        offsets_dict = {}
        for metric in parsed_request:
            offsets_dict[metric] = offset
            size = self._get_size(metric, self._output_sizes)
            offset += size
        return tuple(offsets_dict[metric] for metric in parsed_request)

    def output_offsets_dict(self, output_types_requested):
        """
        Get output array offsets as a dictionary for requested metrics.

        Parameters
        ----------
        output_types_requested : list[str]
            List of metric names to generate offsets for.

        Returns
        -------
        dict[str, int]
            Dictionary with metric names as keys and their offsets as values.
        """
        parsed_request = self.preprocess_request(output_types_requested)

        offset = 0
        offsets_dict = {}
        for metric in parsed_request:
            offsets_dict[metric] = offset
            size = self._get_size(metric, self._output_sizes)
            offset += size
        return offsets_dict

    def summaries_output_height(self, output_types_requested):
        """
        Calculate total output size for requested summary metrics.

        Parameters
        ----------
        output_types_requested : list[str]
            List of metric names to calculate total output size for.

        Returns
        -------
        int
            Total output size needed for all requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)

        total_size = 0
        for metric in parsed_request:
            size = self._get_size(metric, self._output_sizes)
            total_size += size
        return total_size

    def _get_size(self, metric_name, size_dict):
        """
        Calculate size based on parameters if needed.

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        size_dict : dict
            Dictionary containing size specifications.

        Returns
        -------
        int
            Calculated size for the metric.

        Warnings
        --------
        UserWarning
            If a callable size has parameter set to 0.
        """
        size = size_dict.get(metric_name)
        if callable(size):
            param = self._params.get(metric_name)
            if param == 0:
                warn(
                    f"Metric '{metric_name}' has a callable size "
                    f"but parameter is set to 0. This results in a size"
                    "of 0, which is likely not what you want",
                    UserWarning,
                    stacklevel=2,
                )
            return size(param)

        return size

    def legend(self, output_types_requested):
        """
        Generate column headings for requested summary metrics.

        Parameters
        ----------
        output_types_requested : list[str]
            List of metric names to generate headings for.

        Returns
        -------
        list[str]
            Column headings for the metrics in order.

        Notes
        -----
        For metrics with output_size=1, the heading is just the metric name.
        For metrics with output_size>1, the headings are {name}_1, {name}_2,
        etc.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        headings = []

        for metric in parsed_request:
            output_size = self._get_size(metric, self._output_sizes)

            if output_size == 1:
                headings.append(metric)
            else:
                for i in range(output_size):
                    headings.append(f"{metric}_{i + 1}")

        return headings

    def output_sizes(self, output_types_requested):
        """
        Get output array sizes for requested summary metrics.

        Parameters
        ----------
        output_types_requested : list[str]
            List of metric names to generate sizes for.

        Returns
        -------
        tuple[int]
            Output array sizes for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        return tuple(
            self._get_size(metric, self._output_sizes)
            for metric in parsed_request
        )

    def save_functions(self, output_types_requested):
        """
        Get save functions for requested summary metrics.

        Parameters
        ----------
        output_types_requested : list[str]
            List of metric names to get save functions for.

        Returns
        -------
        tuple[callable]
            CUDA device save functions for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        # Retrieve device functions from metric objects at call time
        return tuple(self._metric_objects[metric].save_device_func for metric in parsed_request)

    def update_functions(self, output_types_requested):
        """
        Get update functions for requested summary metrics.

        Parameters
        ----------
        output_types_requested : list[str]
            List of metric names to get update functions for.

        Returns
        -------
        tuple[callable]
            CUDA device update functions for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        # Retrieve device functions from metric objects at call time
        return tuple(self._metric_objects[metric].update_device_func for metric in parsed_request)

    def params(self, output_types_requested: list[str]):
        """
        Get parameters for requested summary metrics.

        Parameters
        ----------
        output_types_requested : list[str]
            List of metric names to get parameters for.

        Returns
        -------
        tuple[Any]
            Parameter values for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        return tuple(self._params[metric] for metric in parsed_request)

    def parse_string_for_params(self, dirty_request: list[str]):
        """
        Extract parameters from metric specification strings.

        Parameters
        ----------
        dirty_request : list[str]
            List of metric specifications that may contain parameters
            in the format 'metric[param]'.

        Returns
        -------
        list[str]
            List of metrics with any [param] stripped

        Notes
        -----
        Any params stripped from the string are saved in self._params, keyed
        by 'metric' name.
        """
        clean_request = []
        self._params = {}
        for string in dirty_request:
            if "[" in string:
                name, param_part = string.split("[", 1)
                param_str = param_part.split("]")[0]

                try:
                    param_value = int(param_str)
                except ValueError:
                    raise ValueError(
                        f"Parameter in '{string}' must be an integer."
                    )

                self._params[name] = param_value
                clean_request.append(name)
            else:
                clean_request.append(string)
                self._params[string] = 0

        return clean_request
