import attrs
from warnings import warn
from typing import Optional
@attrs.define
class SummaryMetric:
    """
    Base class for summary metrics in the CuMC integrator system. Holds memory requirements in temporary and output
    arrays, as well as dispatchers for the update and save functions. Not intended to be mutable or even instantiated
    by the user, but as a dataclass to provide compile-critical information with less boilerplate.
    """

    temp_size: int = attrs.field(validator=attrs.validators.instance_of(int), default=0)
    output_size: int = attrs.field(validator=attrs.validators.instance_of(int), default=0)
    update_device_func: callable = attrs.field(validator=attrs.validators.instance_of(callable), default=None)
    save_device_func: callable = attrs.field(validator=attrs.validators.instance_of(callable), default=None)
    name: str = attrs.field(validator=attrs.validators.instance_of(str), default="")
    input_variable: Optional[dict[str, int]] = attrs.field(validator=attrs.validators.instance_of(dict), default=None)

@attrs.define
class SummaryMetrics:
    """
    Holds the full set of implemented summary metrics, and presents summary information to the rest of the modules.
    Presents:
    - .names: a list of strings to check use requested metric types against
    - .temp_offsets(output_types_requested): A dict of "metric name": starting index in temporary (running) array
    - .output_offsets(output_types_requested): A dict of "metric name": starting index in output (save) array
    - .flags(output_types_requested): A dict of "metric name": boolean flag indicating if the metric is requested

    """
    _names: list[str] = attrs.field(validator=attrs.validators.instance_of(list), default=[], init=False)
    _temp_offsets: dict[str, int] = attrs.field(validator=attrs.validators.instance_of(dict), default={}, init=False)
    _output_offsets: dict[str, int] = attrs.field(validator=attrs.validators.instance_of(dict), default={}, init=False)
    _flags: dict[str, bool] = attrs.field(validator=attrs.validators.instance_of(dict), default={}, init=False)
    _save_functions: dict[str, callable] = attrs.field(validator=attrs.validators.instance_of(dict), default={},
                                                    init=False)
    _update_functions
    _metric_objects = attrs.field(validator=attrs.validators.instance_of(dict), default={}, init=False)

    def register_metric(self, metric: SummaryMetric):
        """
        Register a new summary metric. Once you've created a SummaryMetric, register it with the total set of
        SummaryMetrics by calling this method. It will then be included in the list of summary metrics available,
        and slot into the update and save functions automatically when included in an outputs list.

        Args:
            metric: An instance of SummaryMetric to register.
        """

        if metric.name in self._names:
            raise ValueError(f"Metric '{metric.name}' is already registered.")

        self._names.append(metric.name)
        self._temp_offsets[metric.name] = metric.temp_size
        self._output_offsets[metric.name] = metric.output_size
        self._flags[metric.name] = False
        self._metric_objects[metric.name] = metric

    @property
    def implemented_metrics(self):
        """
        Returns a list of names of all registered summary metrics.
        """
        return self._names

    def