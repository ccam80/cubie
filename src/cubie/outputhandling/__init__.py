"""This module manages the saving and summarising of the solver system.

This module provides the core infrastructure for saving and summarising the
solver system, including registration and function dispatch for CUDA device
functions."""

from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.output_functions import *
from cubie.outputhandling.output_sizes import *
from cubie.outputhandling.output_config import *


__all__ = ["summary_metrics", "OutputFunctions", "OutputConfig"]
