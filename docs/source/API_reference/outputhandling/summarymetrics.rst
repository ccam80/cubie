Summary metrics
===============

``cubie.outputhandling.summarymetrics``
---------------------------------------

.. currentmodule:: cubie.outputhandling.summarymetrics

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   summary_metrics
   register_metric
   summarymetrics/metrics/summary_metric
   summarymetrics/metrics/summary_metrics
   summarymetrics/metrics/metric_func_cache
   summarymetrics/metrics/mean
   summarymetrics/metrics/max
   summarymetrics/metrics/rms
   summarymetrics/metrics/peaks

The ``summarymetrics`` package houses the summary metric registry used by output
handling to accumulate reductions during integration. Importing the package
creates :data:`summary_metrics` and eagerly imports the built-in metrics so that
each registers its CUDA device update and save functions. External packages
extend the system by decorating new metric classes with :func:`register_metric`.

Public interface
----------------

* :doc:`summary_metrics <summary_metrics>` – registry storing metric factories and compiled device callables.
* :doc:`register_metric <register_metric>` – decorator used by metric modules to register implementations.
* :doc:`SummaryMetric <summarymetrics/metrics/summary_metric>` – base class describing summary metric interfaces.
* :doc:`SummaryMetrics <summarymetrics/metrics/summary_metrics>` – registry container that stores metrics and compiled functions.
* :doc:`MetricFuncCache <summarymetrics/metrics/metric_func_cache>` – caches compiled CUDA functions per metric.
* :doc:`Mean <summarymetrics/metrics/mean>` – built-in average metric.
* :doc:`Max <summarymetrics/metrics/max>` – built-in maximum metric.
* :doc:`RMS <summarymetrics/metrics/rms>` – built-in root-mean-square metric.
* :doc:`Peaks <summarymetrics/metrics/peaks>` – built-in peak-detection metric.

Dependencies
------------

* Compiles device functions via :class:`cubie.CUDAFactory` and :mod:`numba.cuda`.
* Consumes save/update cadence configuration from :mod:`cubie.outputhandling`.
