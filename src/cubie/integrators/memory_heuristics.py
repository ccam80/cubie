"""Measured heuristics for default CUDA buffer memory locations.

Selects shared-memory placements that beat the all-local defaults in
paired A/B kernel benchmarks (issue #329). The rules fire only on
configurations matching a measured winning pattern; everything else
keeps the all-local defaults, which the same sweep measured as best
(or within noise of best) whenever the per-thread working set fits
on-chip.

Mechanism, as measured: once the per-thread local working set spills
past the register file, kernels become DRAM-bandwidth-bound and
moving a bounded slice of the hottest buffers into shared memory
relieves the spill traffic. Placements above ~2.5 KiB per run shrink
the block size through the kernel's 32 KiB dynamic-shared target and
lose more residency than they save, so oversized groups stay local.

All thresholds operate on registry-declared buffer sizes - the only
quantities available before compilation - and are calibrated per GPU
architecture. Each :class:`MemoryThresholds` entry in
``THRESHOLDS_BY_ARCH`` is derived from a sweep dataset by
``benchmarks/memory_location_sweep.py --fit``; to calibrate a new
card, run the sweep there and paste the emitted entry. Unknown
architectures fall back to the ``DEFAULT_ARCH`` entry.
"""

from typing import Dict, FrozenSet, List, Optional, Tuple

from attrs import define
from numpy import dtype as np_dtype

from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import compute_capability_code
from cubie.integrators.algorithms import (
    BackwardsEulerStep,
    DIRKStep,
    ERKStep,
    FIRKStep,
)

MEASURED_STEP_TYPES = (
    ERKStep,
    DIRKStep,
    FIRKStep,
    BackwardsEulerStep,
)
"""Algorithm families with benchmarked placement rules.

Subclasses (e.g. the predict-correct backwards Euler step) inherit
their base family's rules: their buffer groups are read from the
registry, so a subclass that registers different buffers is still
placed against its own declared sizes. Families not in this tuple
(crank_nicolson, rosenbrock) have no measured winning placement and
keep all-local defaults.
"""

STATE_PAIR_KEYS = ("state_location", "proposed_state_location")

PARAMS_KEYS = ("parameters_location",)


@define(frozen=True)
class MemoryThresholds:
    """Placement gates calibrated for one GPU architecture.

    All values are bytes of registry-declared buffer sizes.

    Attributes
    ----------
    heavy_spill_bytes : int
        Per-thread local footprint that marks a spilled kernel.
        Shared placements only win once the declared footprint
        passes this; the largest all-shared-losing cell sits below
        it.
    spill_floor_bytes : int
        Minimum footprint for the explicit stage-buffer rule.
        Below this the working set register-promotes and shared
        placement was measured as a pure loss.
    state_pair_max_bytes : int
        Largest state+proposed_state pair moved to shared. Larger
        pairs collapse the block size and lose.
    work_group_max_bytes : int
        Largest implicit algorithm work-buffer group moved to
        shared.
    explicit_work_max_bytes : int
        Largest explicit stage-buffer group moved to shared.
    params_min_bytes : int
        Smallest parameter buffer worth moving to shared on a
        spilled kernel; smaller sets were inconsistent.
    """

    heavy_spill_bytes: int
    spill_floor_bytes: int
    state_pair_max_bytes: int
    work_group_max_bytes: int
    explicit_work_max_bytes: int
    params_min_bytes: int


THRESHOLDS_BY_ARCH: Dict[str, MemoryThresholds] = {
    # RTX 4070 SUPER, 206 paired configurations (issue #329),
    # emitted by memory_location_sweep.py --fit. Boundary cells:
    # footprint 3.9 KiB won vs 3.3 KiB lost; state pair 1 KiB won
    # 1.7-2.9x vs 2 KiB lost; work group 2 KiB won 1.75x vs 4 KiB
    # lost; explicit stage buffers 448 B won 1.25x vs 1.8 KiB
    # lost; params 512 B won 1.1-2.1x.
    "8.9": MemoryThresholds(
        heavy_spill_bytes=3584,
        spill_floor_bytes=512,
        state_pair_max_bytes=1024,
        work_group_max_bytes=2048,
        explicit_work_max_bytes=512,
        params_min_bytes=512,
    ),
}

DEFAULT_ARCH = "8.9"
"""Fallback architecture for cards without a calibrated entry."""


def resolve_thresholds(
    arch: Optional[str] = None,
) -> MemoryThresholds:
    """Return the thresholds calibrated for a GPU architecture.

    Parameters
    ----------
    arch
        Compute-capability code such as ``"8.9"``. When None, the
        current device is queried (CUDASIM reports no
        architecture and receives the default entry).

    Returns
    -------
    MemoryThresholds
        The calibrated entry for ``arch``, falling back to
        ``DEFAULT_ARCH`` when the architecture has no entry.
    """
    if arch is None:
        arch = compute_capability_code()
    if arch is None or arch not in THRESHOLDS_BY_ARCH:
        arch = DEFAULT_ARCH
    return THRESHOLDS_BY_ARCH[arch]


@define(frozen=True)
class DeclaredSizes:
    """Registry-declared quantities the placement gates operate on.

    Attributes
    ----------
    itemsize : int
        Bytes per element of the run's precision.
    is_implicit : bool
        Whether the algorithm step embeds a nonlinear solve.
    footprint_bytes : int
        Declared per-thread local plus persistent footprint.
    state_pair_bytes : int
        Size of the state and proposed-state pair.
    work_group_bytes : int
        Non-aliased size of the algorithm step's own buffers.
    params_bytes : int
        Size of the parameter buffer.
    work_location_keys : Tuple[str, ...]
        ``{name}_location`` settings for the algorithm step's own
        registered buffers, read from the registry so renamed or
        restructured buffers are picked up automatically.
    """

    itemsize: int
    is_implicit: bool
    footprint_bytes: int
    state_pair_bytes: int
    work_group_bytes: int
    params_bytes: int
    work_location_keys: Tuple[str, ...]


def declared_sizes(single_integrator_run) -> DeclaredSizes:
    """Measure the declared buffer sizes of a constructed run.

    Parameters
    ----------
    single_integrator_run
        Constructed integrator core whose loop, algorithm step, and
        solver chain have registered their buffers.

    Returns
    -------
    DeclaredSizes
        Registry-declared quantities for the placement gates.
    """
    loop = single_integrator_run._loop
    algo_step = single_integrator_run._algo_step
    itemsize = np_dtype(single_integrator_run.precision).itemsize

    work_names = buffer_registry.relocatable_buffer_names(algo_step)
    work_elements = buffer_registry.nonaliased_elements(
        algo_step, work_names
    )
    footprint_elements = buffer_registry.declared_local_elements(loop)

    n_states = loop.compile_settings.n_states
    n_parameters = loop.compile_settings.n_parameters

    return DeclaredSizes(
        itemsize=itemsize,
        is_implicit=algo_step.is_implicit,
        footprint_bytes=footprint_elements * itemsize,
        state_pair_bytes=2 * n_states * itemsize,
        work_group_bytes=work_elements * itemsize,
        params_bytes=n_parameters * itemsize,
        work_location_keys=tuple(
            f"{name}_location" for name in work_names
        ),
    )


def placement_candidates(
    sizes: DeclaredSizes,
    thresholds: MemoryThresholds,
) -> List[Tuple[str, ...]]:
    """Return the placement groups whose gates fire, best first.

    Parameters
    ----------
    sizes
        Registry-declared quantities for one configuration.
    thresholds
        Calibrated gates for the target architecture.

    Returns
    -------
    List[Tuple[str, ...]]
        ``*_location`` key groups measured faster than all-local,
        in priority order. Empty when all-local is the measured
        best.
    """
    itemsize = sizes.itemsize
    footprint = sizes.footprint_bytes
    candidates: List[Tuple[bool, Tuple[str, ...]]] = []

    if sizes.is_implicit:
        if footprint < thresholds.heavy_spill_bytes:
            return []
        candidates = [
            (
                itemsize == 4
                and sizes.state_pair_bytes
                <= thresholds.state_pair_max_bytes,
                STATE_PAIR_KEYS,
            ),
            (
                0
                < sizes.work_group_bytes
                <= thresholds.work_group_max_bytes,
                sizes.work_location_keys,
            ),
            (
                itemsize == 4
                and sizes.params_bytes >= thresholds.params_min_bytes,
                PARAMS_KEYS,
            ),
        ]
    else:
        if footprint >= thresholds.heavy_spill_bytes:
            candidates = [
                (
                    sizes.state_pair_bytes
                    <= thresholds.state_pair_max_bytes
                    * (itemsize // 4),
                    STATE_PAIR_KEYS,
                ),
                (
                    itemsize == 4
                    and sizes.params_bytes
                    >= thresholds.params_min_bytes,
                    PARAMS_KEYS,
                ),
            ]
        elif footprint >= thresholds.spill_floor_bytes:
            candidates = [
                (
                    0
                    < sizes.work_group_bytes
                    <= thresholds.explicit_work_max_bytes,
                    sizes.work_location_keys,
                ),
            ]

    return [keys for fires, keys in candidates if fires]


def auto_memory_locations(
    single_integrator_run,
    user_location_keys: FrozenSet[str] = frozenset(),
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
        group is selected so every applied placement corresponds to
        a benchmarked operating point.
    """
    algo_step = single_integrator_run._algo_step
    if not isinstance(algo_step, MEASURED_STEP_TYPES):
        return {}

    sizes = declared_sizes(single_integrator_run)
    thresholds = resolve_thresholds()

    for keys in placement_candidates(sizes, thresholds):
        if not (set(keys) & set(user_location_keys)):
            return {key: "shared" for key in keys}
    return {}
