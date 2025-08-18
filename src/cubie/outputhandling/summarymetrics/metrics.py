from typing import Optional, Callable, Union, Any
from warnings import warn

import attrs


def register_metric(registry):
    def decorator(cls):
        instance = cls()
        registry.register_metric(instance)
        return cls

    return decorator


@attrs.define
class SummaryMetric:
    """
    Base class for summary metrics in the cubie integrator system. Holds
    memory requirements in buffer and output
    arrays, as well as dispatchers for the update and save functions. Not intended to be mutable or even instantiated
    by the user, but as a dataclass to provide compile-critical information with less boilerplate.
    """

    buffer_size: Union[int, Callable] = attrs.field(default=0,
                                                    validator=attrs.validators.instance_of(
                                                            Union[
                                                                int, Callable], ), )
    output_size: Union[int, Callable] = attrs.field(default=0,
                                                    validator=attrs.validators.instance_of(
                                                            Union[
                                                                int, Callable]), )
    update_device_func: Callable = attrs.field(
            validator=attrs.validators.instance_of(Callable), default=None)
    save_device_func: Callable = attrs.field(
            validator=attrs.validators.instance_of(Callable), default=None)
    name: str = attrs.field(validator=attrs.validators.instance_of(str),
                            default="")
    input_variable: Optional[dict[str, int]] = attrs.field(
            validator=attrs.validators.instance_of(Optional[dict]),
            default=None, )


@attrs.define
class SummaryMetrics:
    """
    Holds the full set of implemented summary metrics, and presents summary information to the rest of the modules.
    Presents:
    - .implemented_metrics: a list of strings to check requested metric types against (done internally for other
    requests)
    - .buffer_offsets(output_types_requested): Returns (total_buffer_size, offsets_tuple) for requested metrics only
    - .output_offsets(output_types_requested): Returns (total_output_size, offsets_tuple) for requested metrics only
    - .buffer_sizes(output_types_requested): Returns sizes tuple for requested metrics only
    - .output_sizes(output_types_requested): Returns sizes tuple for requested metrics only
    - .save_functions(output_types_requested): Returns function tuple for requested metrics only
    - .update_functions(output_types_requested): Returns function tuple for requested metrics only
    - .params(output_types_requested): Returns parameter tuple for requested metrics only

    All methods consistently return data only for the requested metrics, not for all implemented metrics.
    """
    _names: list[str] = attrs.field(
            validator=attrs.validators.instance_of(list), factory=list,
            init=False)
    _buffer_sizes: dict[str, Union[int, Callable]] = attrs.field(
            validator=attrs.validators.instance_of(dict), factory=dict,
            init=False, )
    _output_sizes: dict[str, Union[int, Callable]] = attrs.field(
            validator=attrs.validators.instance_of(dict), factory=dict,
            init=False, )
    _save_functions: dict[str, Callable] = attrs.field(
            validator=attrs.validators.instance_of(dict), factory=dict,
            init=False, )
    _update_functions: dict[str, Callable] = attrs.field(
            validator=attrs.validators.instance_of(dict), factory=dict,
            init=False, )
    _metric_objects = attrs.field(validator=attrs.validators.instance_of(dict),
                                  factory=dict, init=False)
    _params: dict[str, Optional[Any]] = attrs.field(
            validator=attrs.validators.instance_of(dict), factory=dict,
            init=False, )

    def __attrs_post_init__(self):
        self._params = {}

    def register_metric(self, metric: SummaryMetric):
        """
        Register a new summary metric. Once you've created a SummaryMetric, register it with the total set of
        summarymetrics by calling this method. It will then be included in the list of summary metrics available,
        and slot into the update and save functions automatically when included in an outputs list.

        Args:
            metric: An instance of SummaryMetric to register.
        """

        if metric.name in self._names:
            raise ValueError(f"Metric '{metric.name}' is already registered.")

        self._names.append(metric.name)
        self._buffer_sizes[metric.name] = metric.buffer_size
        self._output_sizes[metric.name] = metric.output_size
        self._metric_objects[metric.name] = metric
        self._update_functions[metric.name] = metric.update_device_func
        self._save_functions[metric.name] = metric.save_device_func
        self._params[metric.name] = 0

    def preprocess_request(self, request):
        """Parse parameters and validate the request."""
        clean_request = self.parse_string_for_params(request)
        # Validate that all metrics exist and filter out unregistered ones
        validated_request = []
        for metric in clean_request:
            if metric not in self._names:
                warn(f"Metric '{metric}' is not registered. Skipping.",
                     stacklevel=2)
            else:
                validated_request.append(metric)
        return validated_request

    @property
    def implemented_metrics(self):
        """
        Returns a list of names of all registered summary metrics.
        """
        return self._names

    def summaries_buffer_height(self, output_types_requested):
        """
        Returns the total buffer size for the requested summary metrics.

        Args:
            output_types_requested: A list of metric names to calculate total buffer size for.

        Returns:
            An integer representing the total buffer size needed.
        """
        parsed_request = self.preprocess_request(output_types_requested)

        offset = 0
        for metric in parsed_request:
            size = self._get_size(metric, self._buffer_sizes)
            offset += size
        return offset

    def buffer_offsets(self, output_types_requested):
        """
        Returns a tuple of buffer starting offsets for the requested summary metrics.

        Args:
            output_types_requested: A list of metric names to generate offsets for.

        Returns:
            A tuple of offsets for the requested metrics.
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
        Returns a tuple of buffer sizes for the requested summary metrics.

        Args:
            output_types_requested: A list of metric names to generate sizes for.

        Returns:
            A tuple with metric sizes in the buffer.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        return tuple(self._get_size(metric, self._buffer_sizes) for metric in
                     parsed_request)

    def output_offsets(self, output_types_requested):
        """
        Returns a tuple of output array starting offsets for the requested summary metrics.

        Args:
            output_types_requested: A list of metric names to generate offsets for.

        Returns:
            A tuple of offsets for the requested metrics.
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
        Returns a dictionary of output array starting offsets for the requested summary metrics.

        Args:
            output_types_requested: A list of metric names to generate offsets for.

        Returns:
            A dictionary with metric names as keys and their offsets as values.
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
        Returns the total output size for the requested summaries_array metrics.

        Args:
            output_types_requested: A list of metric names to calculate total output size for.

        Returns:
            An integer representing the total output size needed.
        """
        parsed_request = self.preprocess_request(output_types_requested)

        total_size = 0
        for metric in parsed_request:
            size = self._get_size(metric, self._output_sizes)
            total_size += size
        return total_size

    def _get_size(self, metric_name, size_dict):
        """Calculate size based on parameters if needed."""
        size = size_dict.get(metric_name)
        if callable(size):
            param = self._params.get(metric_name)
            if param == 0:
                warn(f"Metric '{metric_name}' has a callable size but parameter is set to 0. This resuts in a size"
                     "of 0, which is likely not what you want", UserWarning,
                        stacklevel=2, )
            return size(param)

        return size

    def legend(self, output_types_requested):
        """
        Returns a list of column headings for the requested summary metrics.

        For metrics with output_size=1, the heading is just the metric name.
        For metrics with output_size>1, the headings are {name}_1, {name}_2, etc.

        Args:
            output_types_requested: A list of metric names to generate headings for.

        Returns:
            A list of column headings for the metrics in the order they appear.
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
        Returns a tuple of output array sizes for the requested summary metrics.

        Args:
            output_types_requested: A list of metric names to generate sizes for.

        Returns:
            A tuple with metric sizes in the output array.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        return tuple(self._get_size(metric, self._output_sizes) for metric in
                     parsed_request)

    def save_functions(self, output_types_requested):
        """
        Returns a tuple of save functions for the requested summary metrics.

        Args:
            output_types_requested: A list of metric names to generate save functions for.

        Returns:
            A tuple with save functions for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        return tuple(self._save_functions[metric] for metric in parsed_request)

    def update_functions(self, output_types_requested):
        """
        Returns a tuple of update functions for the requested summary metrics.

        Args:
            output_types_requested: A list of metric names to generate update functions for.

        Returns:
            A tuple with update functions for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        return tuple(
                self._update_functions[metric] for metric in parsed_request)

    def params(self, output_types_requested: list[str]):
        """
        Returns a tuple of params from the provided request string.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        return tuple(self._params[metric] for metric in parsed_request)

    def parse_string_for_params(self, dirty_request: list[str]):
        """Get single integer metric from specification string like 'peaks[3]', and return the list of strings with
        parameters removed, saving the parameter to self._params under the cleaned key."""
        clean_request = []
        self._params = {}
        for string in dirty_request:
            if '[' in string:
                name, param_part = string.split('[', 1)
                param_str = param_part.split(']')[0]

                try:
                    param_value = int(param_str)
                except ValueError:
                    raise ValueError(
                            f"Parameter in '{string}' must be an integer.")

                self._params[name] = param_value
                clean_request.append(name)
            else:
                clean_request.append(string)
                self._params[string] = 0

        return clean_request
