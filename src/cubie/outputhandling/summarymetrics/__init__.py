"""Summary metric registry and built-in CUDA reductions.

The module instantiates the registry that coordinates summary metric device
functions and imports the included implementations so they self-register.
Third-party metrics should use :func:`register_metric` to add themselves when
their module is imported.
"""

from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetrics,
    register_metric,
)

summary_metrics: SummaryMetrics = SummaryMetrics()

# Import each metric once, to register it with the summary_metrics object.
from cubie.outputhandling.summarymetrics import mean  # noqa
from cubie.outputhandling.summarymetrics import max   # noqa
from cubie.outputhandling.summarymetrics import rms   # noqa
from cubie.outputhandling.summarymetrics import peaks # noqa
from cubie.outputhandling.summarymetrics import std   # noqa
from cubie.outputhandling.summarymetrics import min   # noqa
from cubie.outputhandling.summarymetrics import max_magnitude  # noqa
from cubie.outputhandling.summarymetrics import extrema  # noqa
from cubie.outputhandling.summarymetrics import negative_peaks  # noqa
from cubie.outputhandling.summarymetrics import mean_std_rms  # noqa
from cubie.outputhandling.summarymetrics import mean_std  # noqa
from cubie.outputhandling.summarymetrics import std_rms  # noqa

__all__ = ["summary_metrics", "register_metric"]
