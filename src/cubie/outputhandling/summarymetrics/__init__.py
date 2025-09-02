"""Summary metrics.

This module provides a collection of summary metrics for calculating
statistics during integration."""

from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetrics,
    register_metric,
)

summary_metrics = SummaryMetrics()

# Import each metric once, to register it with the summary_metrics object.
from cubie.outputhandling.summarymetrics import mean  # noqa
from cubie.outputhandling.summarymetrics import max   # noqa
from cubie.outputhandling.summarymetrics import rms   # noqa
from cubie.outputhandling.summarymetrics import peaks # noqa

__all__ = ["summary_metrics", "register_metric"]
