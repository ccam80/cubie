"""Shared fixtures for batch solving tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder
from tests._utils import _driver_sequence


Array = np.ndarray


@pytest.fixture(scope="function")
def batchconfig_instance(system) -> BatchGridBuilder:
    """Return a batch grid builder for the configured system."""

    return BatchGridBuilder.from_system(system)


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
def cpu_batch_results(
    batch_input_arrays,
    cpu_loop_runner,
    system,
    solver_settings,
    precision,
) -> BatchResult:
    """Compute CPU reference outputs for each run in the requested batch."""

    initial_sets, parameter_sets = batch_input_arrays
    num_sets = initial_sets.shape[0]
    samples = int(np.ceil(solver_settings["duration"]
                    / (float(solver_settings["dt_save"]))))
    driver_matrix = _driver_sequence(
        samples=samples,
        total_time=solver_settings["duration"],
        n_drivers=system.num_drivers,
        precision=precision,
    )

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

    state_stack = np.stack([r.state for r in results], axis=1)
    observables_stack = np.stack([r.observables for r in results], axis=1)
    state_summaries_stack = np.stack(
        [r.state_summaries for r in results], axis=1
    )
    observable_summaries_stack = np.stack(
        [r.observable_summaries for r in results], axis=1
    )
    status_or = 0
    for r in results:
        status_or |= r.status

    return BatchResult(
        state=state_stack,
        observables=observables_stack,
        state_summaries=state_summaries_stack,
        observable_summaries=observable_summaries_stack,
        status=status_or,
    )
