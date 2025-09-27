src/cubie/outputhandling/summarymetrics
=======================================

.. currentmodule:: cubie.outputhandling.summarymetrics

The ``summarymetrics`` package houses the summary metric registry used by
output handling to accumulate reductions during integration. Importing the
package instantiates :data:`summary_metrics` and eagerly imports the built-in
metrics so that each registers its CUDA device update and save functions.
External packages extend the system by decorating new metric classes with
:func:`register_metric`.

Public interface
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   summary_metrics
   register_metric
   cubie.outputhandling.summarymetrics.metrics.SummaryMetric
   cubie.outputhandling.summarymetrics.metrics.SummaryMetrics
   cubie.outputhandling.summarymetrics.metrics.MetricFuncCache
   cubie.outputhandling.summarymetrics.mean.Mean
   cubie.outputhandling.summarymetrics.max.Max
   cubie.outputhandling.summarymetrics.rms.RMS
   cubie.outputhandling.summarymetrics.peaks.Peaks

Dependencies
------------

* Compiles device functions via :class:`cubie.CUDAFactory.CUDAFactory` and
  :mod:`numba.cuda`.
* Consumes save/update cadence configuration from
  :mod:`cubie.outputhandling`.

