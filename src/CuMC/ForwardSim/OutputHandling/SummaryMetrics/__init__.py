from CuMC.ForwardSim.OutputHandling.SummaryMetrics.metrics import \
    SummaryMetrics, register_metric

summary_metrics = SummaryMetrics()

# Import each metric once, to register it with the summary_metrics object.
from CuMC.ForwardSim.OutputHandling.SummaryMetrics import mean
from CuMC.ForwardSim.OutputHandling.SummaryMetrics import max
from CuMC.ForwardSim.OutputHandling.SummaryMetrics import rms
from CuMC.ForwardSim.OutputHandling.SummaryMetrics import peaks

__all__ = ["summary_metrics", "register_metric"]
