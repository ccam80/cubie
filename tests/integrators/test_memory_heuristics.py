"""Tests for measured auto buffer-placement heuristics."""

import pytest

from cubie.buffer_registry import buffer_registry
from cubie.integrators.memory_heuristics import auto_memory_locations


def loop_and_algo_shared_buffers(solver):
    """Return non-child shared buffer names for the loop and step."""
    run = solver.kernel.single_integrator
    names = set()
    for parent in (run._loop, run._algo_step):
        group = buffer_registry._groups.get(parent)
        if group is None:
            continue
        for name, entry in group.entries.items():
            if name.endswith(("_shared", "_persistent")):
                continue
            if entry.location == "shared" and entry.size > 0:
                names.add(name)
    return names


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {"algorithm": "euler"},
        {"algorithm": "tsit5"},
        {"algorithm": "backwards_euler"},
    ],
    indirect=True,
)
def test_small_system_keeps_all_buffers_local(solver):
    """No placement fires below the spill gate: small systems stay
    all-local, where shared placements measured neutral-to-slower."""
    assert loop_and_algo_shared_buffers(solver) == set()


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "system_type": "large",
            "algorithm": "tsit5",
            "output_types": ["state"],
            "saved_observable_indices": [],
            "summarised_observable_indices": [],
        },
        {
            "system_type": "large",
            "algorithm": "dirk",
            "output_types": ["state"],
            "saved_observable_indices": [],
            "summarised_observable_indices": [],
        },
        {
            "system_type": "large",
            "algorithm": "backwards_euler",
            "output_types": ["state"],
            "saved_observable_indices": [],
            "summarised_observable_indices": [],
        },
    ],
    indirect=True,
)
def test_large_system_moves_state_pair_to_shared(solver):
    """Heavily spilled kernels with a sub-1-KiB state pair get the
    measured state/proposed_state shared placement."""
    assert loop_and_algo_shared_buffers(solver) == {
        "state",
        "proposed_state",
    }


@pytest.mark.parametrize(
    "solver_settings_override",
    [{
        "system_type": "large",
        "algorithm": "tsit5",
        "output_types": ["state"],
        "saved_observable_indices": [],
        "summarised_observable_indices": [],
        "state_location": "local",
    }],
    indirect=True,
)
def test_user_location_key_blocks_whole_group(solver):
    """Pinning one key of a placement group keeps the whole group
    local: partially relocated groups were never benchmarked."""
    assert loop_and_algo_shared_buffers(solver) == set()


@pytest.mark.parametrize(
    "solver_settings_override",
    [{
        "system_type": "large",
        "algorithm": "tsit5",
        "output_types": ["state"],
        "saved_observable_indices": [],
        "summarised_observable_indices": [],
        "auto_memory": False,
    }],
    indirect=True,
)
def test_auto_memory_false_keeps_all_buffers_local(solver):
    """auto_memory=False disables every heuristic placement."""
    assert loop_and_algo_shared_buffers(solver) == set()


@pytest.mark.parametrize(
    "solver_settings_override",
    [{
        "system_type": "large",
        "algorithm": "backwards_euler",
        "output_types": ["state"],
        "saved_observable_indices": [],
        "summarised_observable_indices": [],
        "state_location": "local",
    }],
    indirect=True,
)
def test_blocked_group_falls_through_to_next_candidate(solver):
    """When the user pins the state pair local, the next measured
    candidate (the work-buffer group) fires instead."""
    assert loop_and_algo_shared_buffers(solver) == {"increment_cache"}


def test_resolver_skips_unmeasured_families(solver):
    """The resolver returns nothing for the default euler config and
    respects explicitly supplied keys."""
    run = solver.kernel.single_integrator
    assert auto_memory_locations(run) == {}
