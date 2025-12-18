"""Infrastructure for registering CUDA summary metrics.

The module exposes the registry used by output handling to collate integration
summaries. It compiles CUDA device callbacks through ``CUDAFactory`` and keeps
per-metric metadata such as buffer sizes, parameterisation, and device
function dispatch tables.
"""

from typing import Any, Callable, Optional, Union
from warnings import warn
from abc import abstractmethod
import attrs
import attrs.validators as val
import numpy as np

from cubie._utils import (
    gttype_validator,
    PrecisionDType,
    precision_converter,
    precision_validator,
)
from cubie.CUDAFactory import CUDAFactory, CUDAFunctionCache


@attrs.define
class MetricFuncCache(CUDAFunctionCache):
    """Cache container for compiled metric functions.

    Attributes
    ----------
    update
        Callable update device function.
    save
        Callable save device function.
    """

    update: Callable = attrs.field(default=None)
    save: Callable = attrs.field(default=None)

@attrs.define
class MetricConfig:
    """Configuration for summary metric compilation.
    
    Attributes
    ----------
    dt_save
        Time interval between saved states. Used by derivative
        metrics to scale finite differences. Defaults to 0.01.
    precision
        Numerical precision for metric calculations. Defaults to
        np.float32.
    """
    _precision: PrecisionDType = attrs.field(
        converter=precision_converter,
        validator=precision_validator,
    )
    _dt_save: float = attrs.field(
        default=0.01,
        validator=val.optional(gttype_validator(float, 0.0))
    )

    
    @property
    def dt_save(self) -> float:
        """Time interval between saved states."""
        return self._dt_save
    
    @property
    def precision(self) -> type[np.floating]:
        """Numerical precision for metric calculations."""
        return self._precision


def register_metric(registry: "SummaryMetrics") -> Callable:
    """Create a decorator that registers a metric on instantiation.

    Parameters
    ----------
    registry
        SummaryMetrics instance that accepts the metric.

    Returns
    -------
    Callable
        Decorator that instantiates and registers the wrapped metric class.

    Notes
    -----
    The decorator immediately instantiates the class so that registration
    occurs when the module importing the metric executes.
    """

    def decorator(cls):
        instance = cls(registry.precision)
        registry.register_metric(instance)
        return cls

    return decorator


class SummaryMetric(CUDAFactory):
    """Abstract base class for summary metrics.

    Attributes
    ----------
    buffer_size
        int or Callable. Memory required per metric buffer entry. Parameterised
        metrics should supply a callable that accepts the metric parameter.
    output_size
        int or Callable. Memory required for persisted metric results.
    name
        str. Identifier used in registries and configuration strings.
    unit_modification
        str. Format string for unit modification in legends.
    update_device_func
        Callable. Compiled CUDA device update function for the metric.
    save_device_func
        Callable. Compiled CUDA device save function for the metric.
    dt_save
        save interval. Defaults to 0.01.
    precision
        Numerical precision for metric calculations. Defaults to np.float32.

    Notes
    -----
    Subclasses must implement :meth:`build` to provide CUDA device callbacks
    with the signatures ``update(value, buffer, current_index,
    customisable_variable)`` and ``save(buffer, output_array, summarise_every,
    customisable_variable)``. Metrics need to be imported only after the global
    registry has been created so that decoration registers the implementation.
    """

    def __init__(
        self,
        buffer_size: Union[int, Callable],
        output_size: Union[int, Callable],
        name: str,
        precision: PrecisionDType,
        unit_modification: str = "[unit]",
        dt_save: float = 0.01,
    ) -> None:
        """Initialise core metadata for a summary metric.

        Parameters
        ----------
        buffer_size
            int or Callable. Buffer footprint or callable that accepts the
            metric parameter.
        output_size
            int or Callable. Output footprint or callable that accepts the
            metric parameter.
        name
            str. Identifier used for registration.
        unit_modification
            str. Format string for unit modification in legends.
            Use "[unit]" as placeholder. Defaults to "[unit]".
        dt_save
            float. Time interval for save operations. Defaults to 0.01.
        precision
            PrecisionDType. Numerical precision for metric calculations.
            Defaults to np.float32.
        """

        super().__init__()
        self.buffer_size = buffer_size
        self.output_size = output_size
        self.name = name
        self.unit_modification = unit_modification

        # Instantiate empty settings object for CUDAFactory compatibility
        self.setup_compile_settings(
            MetricConfig(dt_save=dt_save, precision=precision)
        )


    @abstractmethod
    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for the metric.

        Returns
        -------
        MetricFuncCache
            Cache containing device update and save functions compiled for
            CUDA execution.

        Notes
        -----
        Implementations must return functions with the signatures
        ``update(value, buffer, current_index, customisable_variable)`` and
        ``save(buffer, output_array, summarise_every, customisable_variable)``.
        Each callback needs ``@cuda.jit(..., device=True, inline=True)``
        decoration supporting both single- and double-precision input.
        """

        pass

    @property
    def update_device_func(self) -> Callable:
        """CUDA device update function for the metric."""

        return self.get_cached_output("update")

    @property
    def save_device_func(self) -> Callable:
        """CUDA device save function for the metric."""

        return self.get_cached_output("save")

    @property
    def precision(self) -> type[np.floating]:
        """Numerical precision for metric calculations."""
        return self.compile_settings.precision

    def update(self, **kwargs) -> None:
        """Update metric compile settings.
        
        Parameters
        ----------
        **kwargs
            Compile settings to update (e.g., dt_save=0.02).
            
        Returns
        -------
        None
            Returns None.
            
        Notes
        -----
        Updates the MetricConfig and invalidates cache if values change.
        Triggers recompilation on next device_function access.
        """
        self.update_compile_settings(kwargs, silent=True)

@attrs.define
class SummaryMetrics:
    """Registry and dispatcher for summary metrics.

    Attributes
    ----------
    _names
        list[str]. Registered metric names.
    _buffer_sizes
        dict[str, int | Callable]. Buffer size requirements keyed by name.
    _output_sizes
        dict[str, int | Callable]. Output size requirements keyed by name.
    _metric_objects
        dict[str, SummaryMetric]. Registered metric instances.
    _params
        dict[str, Any]. Parameters parsed from configuration strings.

    Notes
    -----
    Methods only report information for metrics explicitly requested so the
    caller can compile device functions tailored to the active configuration.
    """
    precision: PrecisionDType = attrs.field()
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
    _combined_metrics: dict[frozenset[str], str] = attrs.field(
        validator=attrs.validators.instance_of(dict),
        factory=dict,
        init=False,
    )


    def __attrs_post_init__(self) -> None:
        """Reset the parsed parameter cache and define combined metrics."""

        self._params = {}
        # Define combined metrics registry:
        # Maps frozenset of individual metrics to the combined metric name
        # Only combine when ALL constituent parts are requested
        # This ensures user gets exactly what they requested
        self._combined_metrics = {
            frozenset(["mean", "std", "rms"]): "mean_std_rms",
            frozenset(["mean", "std"]): "mean_std",
            frozenset(["std", "rms"]): "std_rms",
            frozenset(["max", "min"]): "extrema",
            frozenset(["dxdt_max", "dxdt_min"]): "dxdt_extrema",
            frozenset(["d2xdt2_max", "d2xdt2_min"]): "d2xdt2_extrema",
        }

    def update(self, **kwargs) -> None:
        """Update compile settings for all registered metrics.
        
        Parameters
        ----------
        **kwargs
            Compile settings to update (e.g., dt_save=0.02, precision=np.float64).
            
        Returns
        -------
        None
            Returns None.
            
        Notes
        -----
        Propagates updates to all registered metric objects.
        Each metric invalidates its cache if values change.
        """
        if "precision" in kwargs:
            self.precision = kwargs["precision"]
        for metric in self._metric_objects.values():
            metric.update(**kwargs)

    def register_metric(self, metric: SummaryMetric) -> None:
        """Register a new summary metric with the system.

        Parameters
        ----------
        metric
            SummaryMetric instance to register.

        Raises
        ------
        ValueError
            If a metric with the same name is already registered.

        Notes
        -----
        Registration makes the metric available for summary buffer planning
        and function dispatch during integration.
        """

        if metric.name in self._names:
            raise ValueError(f"Metric '{metric.name}' is already registered.")

        self._names.append(metric.name)
        self._buffer_sizes[metric.name] = metric.buffer_size
        self._output_sizes[metric.name] = metric.output_size
        self._metric_objects[metric.name] = metric
        self._params[metric.name] = 0

    def _apply_combined_metrics(self, request: list[str]) -> list[str]:
        """Substitute individual metrics with combined metrics when beneficial.

        Parameters
        ----------
        request
            List of metric names to check for substitution.

        Returns
        -------
        list[str]
            Modified list with combined metrics substituted where applicable.

        Notes
        -----
        Checks if subsets of requested metrics match any combined metric
        patterns and substitutes them with the more efficient combined version.
        Prioritizes larger combinations (more metrics combined).
        Preserves the original order of metrics in the request.
        """
        result = []
        used = set()
        
        # Sort by size (descending) to prefer larger combinations
        sorted_combinations = sorted(
            self._combined_metrics.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        # Process each metric in the original request order
        for metric in request:
            if metric in used:
                # Already processed as part of a combination
                continue
                
            # Check if this metric is part of any combination
            combined_found = False
            for metric_set, combined_name in sorted_combinations:
                if metric in metric_set and metric_set.issubset(request):
                    # Check if combined metric is registered and not already used
                    if combined_name in self._names and combined_name not in result:
                        result.append(combined_name)
                        used.update(metric_set)
                        # Add parameter entry for combined metric (always 0)
                        self._params[combined_name] = 0
                        combined_found = True
                        break
            
            if not combined_found:
                # No combination found, add the metric as-is
                result.append(metric)
                used.add(metric)
        
        return result

    def preprocess_request(self, request: list[str]) -> list[str]:
        """Parse parameters from metric specifications and validate.

        Parameters
        ----------
        request
            List of metric specification strings that may include
            ``[parameter]`` suffixes.

        Returns
        -------
        list[str]
            Validated metric names after parameter parsing and combined
            metric substitution.

        Notes
        -----
        Invalid metric names trigger a warning and are removed from the
        returned list. Combined metrics are automatically substituted when
        multiple individual metrics can be computed more efficiently together.
        """
        clean_request = self.parse_string_for_params(request)
        
        # Apply combined metric substitutions
        clean_request = self._apply_combined_metrics(clean_request)
        
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
    def implemented_metrics(self) -> list[str]:
        """Registered summary metric names."""

        return self._names

    def summaries_buffer_height(
        self,
        output_types_requested: list[str],
    ) -> int:
        """Calculate total buffer size for requested summary metrics.

        Parameters
        ----------
        output_types_requested
            Metric names used to determine buffer requirements.

        Returns
        -------
        int
            Total buffer size needed for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)

        offset = 0
        for metric in parsed_request:
            size = self._get_size(metric, self._buffer_sizes)
            offset += size
        return offset

    def buffer_offsets(
        self,
        output_types_requested: list[str],
    ) -> tuple[int, ...]:
        """Get buffer starting offsets for requested summary metrics.

        Parameters
        ----------
        output_types_requested
            Metric names to generate offsets for.

        Returns
        -------
        tuple[int, ...]
            Buffer starting offsets in the order requested.
        """
        parsed_request = self.preprocess_request(output_types_requested)

        offset = 0
        offsets_dict = {}
        for metric in parsed_request:
            offsets_dict[metric] = offset
            size = self._get_size(metric, self._buffer_sizes)
            offset += size
        return tuple(offsets_dict[metric] for metric in parsed_request)

    def buffer_sizes(
        self,
        output_types_requested: list[str],
    ) -> tuple[int, ...]:
        """Get buffer sizes for requested summary metrics.

        Parameters
        ----------
        output_types_requested
            Metric names to generate sizes for.

        Returns
        -------
        tuple[int, ...]
            Buffer sizes for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        return tuple(
            self._get_size(metric, self._buffer_sizes)
            for metric in parsed_request
        )

    def output_offsets(
        self,
        output_types_requested: list[str],
    ) -> tuple[int, ...]:
        """Get output array starting offsets for requested summary metrics.

        Parameters
        ----------
        output_types_requested
            Metric names to generate offsets for.

        Returns
        -------
        tuple[int, ...]
            Output array starting offsets in the order requested.
        """
        parsed_request = self.preprocess_request(output_types_requested)

        offset = 0
        offsets_dict = {}
        for metric in parsed_request:
            offsets_dict[metric] = offset
            size = self._get_size(metric, self._output_sizes)
            offset += size
        return tuple(offsets_dict[metric] for metric in parsed_request)

    def output_offsets_dict(
        self,
        output_types_requested: list[str],
    ) -> dict[str, int]:
        """Get output array offsets as a dictionary for requested metrics.

        Parameters
        ----------
        output_types_requested
            Metric names to generate offsets for.

        Returns
        -------
        dict[str, int]
            Mapping of metric names to output offsets.
        """
        parsed_request = self.preprocess_request(output_types_requested)

        offset = 0
        offsets_dict = {}
        for metric in parsed_request:
            offsets_dict[metric] = offset
            size = self._get_size(metric, self._output_sizes)
            offset += size
        return offsets_dict

    def summaries_output_height(
        self,
        output_types_requested: list[str],
    ) -> int:
        """Calculate total output size for requested summary metrics.

        Parameters
        ----------
        output_types_requested
            Metric names used to determine output requirements.

        Returns
        -------
        int
            Total output size needed for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)

        total_size = 0
        for metric in parsed_request:
            size = self._get_size(metric, self._output_sizes)
            total_size += size
        return total_size

    def _get_size(
        self,
        metric_name: str,
        size_dict: dict[str, Union[int, Callable]],
    ) -> int:
        """Calculate size based on parameters if needed.

        Parameters
        ----------
        metric_name
            Name of the metric to look up.
        size_dict
            Dictionary containing size specifications for metrics.

        Returns
        -------
        int
            Calculated size for the metric.

        Warnings
        --------
        UserWarning
            Issued when a callable size is provided without a parameter value.
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

    def legend(self, output_types_requested: list[str]) -> list[str]:
        """Generate column headings for requested summary metrics.

        Parameters
        ----------
        output_types_requested
            Metric names to generate headings for.

        Returns
        -------
        list[str]
            Column headings for the metrics in order.

        Notes
        -----
        Metrics with multi-element outputs produce numbered headings such as
        ``{name}_1`` and ``{name}_2``.
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

    def unit_modifications(self, output_types_requested: list[str]) -> list[str]:
        """Generate unit modification strings for requested summary metrics.

        Parameters
        ----------
        output_types_requested
            Metric names to generate unit modifications for.

        Returns
        -------
        list[str]
            Unit modification strings for the metrics in order.

        Notes
        -----
        Returns one unit modification per output element. For multi-element
        outputs, the same unit modification is repeated for each element.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        modifications = []

        for metric in parsed_request:
            output_size = self._get_size(metric, self._output_sizes)
            unit_mod = self._metric_objects[metric].unit_modification

            for _ in range(output_size):
                modifications.append(unit_mod)

        return modifications

    def output_sizes(
        self,
        output_types_requested: list[str],
    ) -> tuple[int, ...]:
        """Get output array sizes for requested summary metrics.

        Parameters
        ----------
        output_types_requested
            Metric names to generate sizes for.

        Returns
        -------
        tuple[int, ...]
            Output array sizes for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        return tuple(
            self._get_size(metric, self._output_sizes)
            for metric in parsed_request
        )

    def save_functions(
        self,
        output_types_requested: list[str],
    ) -> tuple[Callable, ...]:
        """Get save functions for requested summary metrics.

        Parameters
        ----------
        output_types_requested
            Metric names to get save functions for.

        Returns
        -------
        tuple[Callable, ...]
            CUDA device save functions for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        # Retrieve device functions from metric objects at call time
        return tuple(
            self._metric_objects[metric].save_device_func
            for metric in parsed_request
        )

    def update_functions(
        self,
        output_types_requested: list[str],
    ) -> tuple[Callable, ...]:
        """Get update functions for requested summary metrics.

        Parameters
        ----------
        output_types_requested
            Metric names to get update functions for.

        Returns
        -------
        tuple[Callable, ...]
            CUDA device update functions for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        # Retrieve device functions from metric objects at call time
        return tuple(
            self._metric_objects[metric].update_device_func
            for metric in parsed_request
        )

    def params(self, output_types_requested: list[str]) -> tuple[Any, ...]:
        """Get parameters for requested summary metrics.

        Parameters
        ----------
        output_types_requested
            Metric names to get parameters for.

        Returns
        -------
        tuple[Any, ...]
            Parameter values for the requested metrics.
        """
        parsed_request = self.preprocess_request(output_types_requested)
        return tuple(self._params[metric] for metric in parsed_request)

    def parse_string_for_params(self, dirty_request: list[str]) -> list[str]:
        """Extract parameters from metric specification strings.

        Parameters
        ----------
        dirty_request
            Metric specifications that may contain ``[param]`` suffixes.

        Returns
        -------
        list[str]
            Metrics with any parameter notation removed.

        Notes
        -----
        Parsed parameter values are cached in ``self._params`` keyed by metric
        name.
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
