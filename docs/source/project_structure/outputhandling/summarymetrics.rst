Summary metrics
===============

``cubie.outputhandling.summarymetrics``
---------------------------------------

.. currentmodule:: cubie.outputhandling.summarymetrics

The ``summarymetrics`` package houses the summary metric registry used by output
handling to accumulate reductions during integration. Importing the package
instantiates :data:`summary_metrics` and eagerly imports the built-in metrics so
that each registers its CUDA device update and save functions. External packages
extend the system by decorating new metric classes with :func:`register_metric`.

Public interface
----------------

* :data:`summary_metrics` – registry storing metric factories and compiled
  device callables.
* :func:`register_metric` – decorator used by metric modules to register
  implementations.
* :class:`SummaryMetric <cubie.outputhandling.summarymetrics.metrics.SummaryMetric>` –
  base class describing summary metric interfaces.
* :class:`SummaryMetrics <cubie.outputhandling.summarymetrics.metrics.SummaryMetrics>` –
  registry container that stores metrics and compiled functions.
* :class:`MetricFuncCache <cubie.outputhandling.summarymetrics.metrics.MetricFuncCache>` –
  caches compiled CUDA functions per metric.
* :class:`Mean <cubie.outputhandling.summarymetrics.mean.Mean>` – built-in
  average metric.
* :class:`Max <cubie.outputhandling.summarymetrics.max.Max>` – built-in maximum
  metric.
* :class:`RMS <cubie.outputhandling.summarymetrics.rms.RMS>` – built-in root-mean-square
  metric.
* :class:`Peaks <cubie.outputhandling.summarymetrics.peaks.Peaks>` – built-in
  peak-detection metric.

Dependencies
------------

* Compiles device functions via :class:`cubie.CUDAFactory` and :mod:`numba.cuda`.
* Consumes save/update cadence configuration from :mod:`cubie.outputhandling`.
