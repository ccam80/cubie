from cubie.outputhandling.summarymetrics.metrics import \
    SummaryMetrics, register_metric

summary_metrics = SummaryMetrics()

# Import each metric once, to register it with the summary_metrics object.
from cubie.outputhandling.summarymetrics import mean
from cubie.outputhandling.summarymetrics import max
from cubie.outputhandling.summarymetrics import rms
from cubie.outputhandling.summarymetrics import peaks

__all__ = ["summary_metrics", "register_metric"]
