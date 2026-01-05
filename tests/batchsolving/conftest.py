"""Shared fixtures for batch solving tests."""

from __future__ import annotations

import attrs

import numpy as np
import pytest

from cubie.batchsolving.BatchInputHandler import BatchInputHandler

# Backward compatibility alias
BatchGridBuilder = BatchInputHandler


Array = np.ndarray


@pytest.fixture(scope="session")
def input_handler(system) -> BatchInputHandler:
    """Return a batch input handler for the configured system."""

    return BatchInputHandler.from_system(system)


# Backward compatibility alias
@pytest.fixture(scope="session")
def batchconfig_instance(input_handler) -> BatchInputHandler:
    """Backward compatibility alias for input_handler."""
    return input_handler


@pytest.fixture(scope="session")
def batch_settings_override(request) -> dict:
    """Override values for batch grid settings when parametrised."""

    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def batch_request(system, batch_settings, precision) -> dict[str, Array]:
    """Build a request dictionary describing the batch sweep."""

    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    #Generate n samples as a linspace, but also concatenate the default value on the end for comparison
    return {
        state_names[0]: np.concatenate([
            np.linspace(0.1, 1.0, batch_settings["num_state_vals_0"], dtype=precision),
            [system.initial_values.values_dict[state_names[0]]]
        ]),
        state_names[1]: np.concatenate([
            np.linspace(0.1, 1.0, batch_settings["num_state_vals_1"], dtype=precision),
            [system.initial_values.values_dict[state_names[1]]]
        ]),
        param_names[0]: np.concatenate([
            np.linspace(0.1, 1.0, batch_settings["num_param_vals_0"], dtype=precision),
            [system.parameters.values_dict[param_names[0]]]
        ]),
        param_names[1]: np.concatenate([
            np.linspace(0.1, 1.0, batch_settings["num_param_vals_1"], dtype=precision),
            [system.parameters.values_dict[param_names[1]]]
        ]),
    }


@pytest.fixture(scope="session")
def batch_input_arrays(
    batch_request,
    batch_settings,
    input_handler,
    system,
) -> tuple[Array, Array]:
    """Return the initial state and parameter arrays for the batch run."""
    # Separate batch_request into states and params based on system names
    state_names = set(system.initial_values.names)
    param_names = set(system.parameters.names)

    states_dict = {k: v for k, v in batch_request.items() if k in state_names}
    params_dict = {k: v for k, v in batch_request.items() if k in param_names}

    return input_handler(
        states=states_dict,
        params=params_dict,
        kind=batch_settings["kind"]
    )


@attrs.define
class BatchResult:
    """Container for CPU reference outputs for a single batch run."""

    state: Array
    observables: Array
    state_summaries: Array
    observable_summaries: Array
    status: int


@pytest.fixture(scope="session")
def cpu_batch_results(
    batch_input_arrays,
    cpu_loop_runner,
    system,
    solver_settings,
    precision,
    driver_array,
) -> BatchResult:
    """Compute CPU reference outputs for each run in the requested batch."""

    initial_sets, parameter_sets = batch_input_arrays
    results: list[BatchResult] = []
    coefficients = driver_array.coefficients if driver_array is not None else None
    # Arrays are in (variable, run) format - iterate over run dimension (axis 1)
    n_runs = initial_sets.shape[1]
    for idx in range(n_runs):
        loop_result = cpu_loop_runner(
            initial_values=initial_sets[:, idx],
            parameters=parameter_sets[:, idx],
            driver_coefficients=coefficients
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

    # Stack along axis=2 to get (time, variable, run) format
    state_stack = np.stack([r.state for r in results], axis=2)
    observables_stack = np.stack([r.observables for r in results], axis=2)
    state_summaries_stack = np.stack(
        [r.state_summaries for r in results], axis=2
    )
    observable_summaries_stack = np.stack(
        [r.observable_summaries for r in results], axis=2
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
