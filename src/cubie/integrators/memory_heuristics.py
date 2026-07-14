"""Measured heuristics for default CUDA buffer memory locations.

Selects shared-memory placements that beat the all-local defaults in
paired A/B kernel benchmarks (issue #329, RTX 4070 SUPER). The rules
fire only on configurations matching a measured winning pattern;
everything else keeps the all-local defaults, which the same sweep
measured as best (or within noise of best) whenever the per-thread
working set fits on-chip.

Mechanism, as measured: once the per-thread local working set spills
past the register file, kernels become DRAM-bandwidth-bound and
moving a bounded slice of the hottest buffers into shared memory
relieves the spill traffic. Placements above ~2.5 KiB per run shrink
the block size through the kernel's 32 KiB dynamic-shared target and
lose more residency than they save, so oversized groups stay local.

All thresholds operate on registry-declared buffer sizes - the only
quantities available before compilation. They were calibrated so
that every winning benchmark cell fires its rule and every losing
cell does not.
"""

from typing import Dict, List, Tuple

from numpy import dtype as np_dtype

from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms import (
    BackwardsEulerPCStep,
    BackwardsEulerStep,
    DIRKStep,
    ERKStep,
    FIRKStep,
)

HEAVY_SPILL_BYTES = 3072
"""Predicted per-thread local footprint that marks a spilled kernel.

Paired sweeps show shared placements only win once the declared
local+persistent footprint reaches ~3.3 KiB (dirk n=64); the largest
all-shared-losing cell sits at ~2.6 KiB (backwards Euler n=64).
"""

SPILL_FLOOR_BYTES = 512
"""Minimum footprint for the explicit stage-buffer rule to fire.

Below this the working set register-promotes and shared placement
was measured as a pure loss (tsit5 n=4: 2.0x slower).
"""

STATE_PAIR_MAX_BYTES = 1024
"""Largest state+proposed_state pair moved to shared.

Pairs at or under 1 KiB won 1.6-2.9x on spilled kernels (tsit5,
dirk, firk, backwards Euler at n=64-128); 2 KiB pairs collapsed the
block size to 16 and lost up to 1.7x (n=256).
"""

WORK_GROUP_MAX_BYTES = 2560
"""Largest algorithm work-buffer group moved to shared.

The 2.5 KiB dirk n=128 group won 1.75x; the 5 KiB n=256 group lost
1.9x.
"""

EXPLICIT_WORK_MAX_BYTES = 512
"""Largest explicit stage-buffer group moved to shared.

The 448 B tsit5 n=16 group won 1.25x; the 1.8 KiB n=64 group lost
2.2x.
"""

PARAMS_MIN_BYTES = 512
"""Smallest parameter buffer worth moving to shared.

512 B parameter sets won 1.1-2.1x on spilled kernels; smaller sets
were inconsistent across algorithms.
"""

STATE_PAIR_KEYS = ("state_location", "proposed_state_location")

WORK_BUFFER_LOCATION_KEYS: Dict[type, Tuple[str, ...]] = {
    ERKStep: ("stage_rhs_location", "stage_accumulator_location"),
    DIRKStep: (
        "stage_increment_location",
        "stage_base_location",
        "accumulator_location",
        "stage_rhs_location",
    ),
    FIRKStep: (
        "stage_increment_location",
        "stage_driver_stack_location",
        "stage_state_location",
    ),
    BackwardsEulerStep: ("increment_cache_location",),
    BackwardsEulerPCStep: ("increment_cache_location",),
}
"""Work-buffer location kwargs per measured algorithm family.

Families absent from this mapping (crank_nicolson, rosenbrock) have
no measured winning placement and keep all-local defaults.
"""


def _chain_local_bytes(loop, itemsize: int) -> int:
    """Return the declared per-thread local footprint in bytes.

    Sums plain-local buffer sizes across the loop and every child
    component recorded in the registry, plus the loop's persistent
    total (children's persistent totals already roll up into the
    loop's child entries).
    """
    groups = buffer_registry._groups

    def local_elements(parent) -> int:
        group = groups.get(parent)
        if group is None:
            return 0
        total = group.local_buffer_size()
        for child in group.children.values():
            total += local_elements(child)
        return total

    persistent = buffer_registry.persistent_local_buffer_size(loop)
    return (local_elements(loop) + persistent) * itemsize


def _group_bytes(parent, keys: Tuple[str, ...], itemsize: int) -> int:
    """Return the shared bytes a location-key group would occupy.

    Aliased buffers overlap their parent's allocation, so only
    non-aliased entries contribute.
    """
    group = buffer_registry._groups.get(parent)
    if group is None:
        return 0
    total = 0
    for key in keys:
        name = key.removesuffix("_location")
        entry = group.entries.get(name)
        if entry is not None and entry.aliases is None:
            total += entry.size
    return total * itemsize


def auto_memory_locations(
    single_integrator_run,
    user_location_keys=frozenset(),
) -> Dict[str, str]:
    """Return shared placements measured faster than all-local.

    Parameters
    ----------
    single_integrator_run
        Constructed integrator core whose loop, algorithm step, and
        solver chain have registered their buffers.
    user_location_keys
        ``*_location`` settings the user supplied explicitly. A
        buffer group containing any of these keys is skipped whole:
        partially relocated groups were never benchmarked.

    Returns
    -------
    Dict[str, str]
        ``*_location`` settings to apply, empty when the all-local
        defaults are already the measured best. At most one buffer
        group is selected so every returned placement corresponds to
        a benchmarked operating point.
    """
    loop = single_integrator_run._loop
    algo_step = single_integrator_run._algo_step
    itemsize = np_dtype(single_integrator_run.precision).itemsize

    work_keys: Tuple[str, ...] = ()
    for step_type, keys in WORK_BUFFER_LOCATION_KEYS.items():
        if type(algo_step) is step_type:
            work_keys = keys
            break
    else:
        return {}

    footprint = _chain_local_bytes(loop, itemsize)
    n_states = loop.compile_settings.n_states
    n_parameters = loop.compile_settings.n_parameters
    state_pair_bytes = 2 * n_states * itemsize
    work_bytes = _group_bytes(algo_step, work_keys, itemsize)
    params_bytes = n_parameters * itemsize

    candidates: List[Tuple[bool, Tuple[str, ...]]] = []
    if algo_step.is_implicit:
        if footprint < HEAVY_SPILL_BYTES:
            return {}
        candidates = [
            (
                itemsize == 4
                and state_pair_bytes <= STATE_PAIR_MAX_BYTES,
                STATE_PAIR_KEYS,
            ),
            (0 < work_bytes <= WORK_GROUP_MAX_BYTES, work_keys),
            (
                itemsize == 4 and params_bytes >= PARAMS_MIN_BYTES,
                ("parameters_location",),
            ),
        ]
    else:
        if footprint >= HEAVY_SPILL_BYTES:
            candidates = [
                (
                    state_pair_bytes <= STATE_PAIR_MAX_BYTES * (
                        itemsize // 4
                    ),
                    STATE_PAIR_KEYS,
                ),
                (
                    itemsize == 4 and params_bytes >= PARAMS_MIN_BYTES,
                    ("parameters_location",),
                ),
            ]
        elif footprint >= SPILL_FLOOR_BYTES:
            candidates = [
                (
                    0 < work_bytes <= EXPLICIT_WORK_MAX_BYTES,
                    work_keys,
                ),
            ]

    for fires, keys in candidates:
        if fires and not (set(keys) & set(user_location_keys)):
            return {key: "shared" for key in keys}
    return {}
