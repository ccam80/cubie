"""Shared fixtures for batch solving tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder


Array = np.ndarray


@pytest.fixture(scope="function")
def batchconfig_instance(system) -> BatchGridBuilder:
    """Return a batch grid builder for the configured system."""

    return BatchGridBuilder.from_system(system)


@pytest.fixture(scope="function")
def square_drive(system, solver_settings, precision, request) -> Array:
    """Generate a square driver waveform for forcing vectors."""

    if hasattr(request, "param"):
        if "cycles" in request.param:
            cycles = request.getattr("cycles", 5)
    else:
        cycles = 5

    numvecs = system.sizes.drivers
    length = int(solver_settings["duration"] // solver_settings["dt_min"])
    driver = np.zeros((length, numvecs), dtype=precision)
    half_period = length // (2 * cycles)

    for idx in range(cycles):
        driver[idx * half_period : (idx + 1) * half_period, :] = 1.0

    return driver


@pytest.fixture(scope="function")
def batch_settings_override(request) -> dict:
    """Override values for batch grid settings when parametrised."""

    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def batch_settings(batch_settings_override) -> dict:
    """Return default batch grid settings merged with overrides."""

    defaults = {
        "num_state_vals_0": 2,
        "num_state_vals_1": 0,
        "num_param_vals_0": 2,
        "num_param_vals_1": 0,
        "kind": "combinatorial",
    }
    defaults.update({k: v for k, v in batch_settings_override.items() if k in defaults})
    return defaults


@pytest.fixture(scope="function")
def batch_request(system, batch_settings) -> dict[str, Array]:
    """Build a request dictionary describing the batch sweep."""

    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    return {
        state_names[0]: np.linspace(0.1, 1.0, batch_settings["num_state_vals_0"]),
        state_names[1]: np.linspace(0.1, 1.0, batch_settings["num_state_vals_1"]),
        param_names[0]: np.linspace(0.1, 1.0, batch_settings["num_param_vals_0"]),
        param_names[1]: np.linspace(0.1, 1.0, batch_settings["num_param_vals_1"]),
    }


@pytest.fixture(scope="function")
def batch_input_arrays(
    batch_request,
    batch_settings,
    batchconfig_instance,
) -> tuple[Array, Array]:
    """Return the initial state and parameter arrays for the batch run."""

    return batchconfig_instance.grid_arrays(
        batch_request, kind=batch_settings["kind"]
    )


@dataclass
class BatchResult:
    """Container for CPU reference outputs for a single batch run."""

    state: Array
    observables: Array
    state_summaries: Array
    observable_summaries: Array
    status: int


@pytest.fixture(scope="function")
def batch_results(
    batch_input_arrays,
    cpu_loop_runner,
    square_drive,
    precision,
) -> list[BatchResult]:
    """Compute CPU reference outputs for each run in the requested batch."""

    initial_sets, parameter_sets = batch_input_arrays
    driver_matrix = np.array(square_drive, dtype=precision, copy=True)

    results: list[BatchResult] = []
    for idx in range(initial_sets.shape[0]):
        loop_result = cpu_loop_runner(
            initial_values=initial_sets[idx, :],
            parameters=parameter_sets[idx, :],
            forcing_vectors=driver_matrix,
        )
        results.append(
            BatchResult(
                state=loop_result["state"],
                observables=loop_result["observables"],
                state_summaries=loop_result["state_summaries"],
                observable_summaries=loop_result["observable_summaries"],
                status=int(loop_result["status"]),
            )
        )
    return results
