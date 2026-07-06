"""Tests for cubie.buffer_registry."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.buffer_registry import (
    BufferGroup,
    BufferRegistry,
    CUDABuffer,
    buffer_registry,
)


# ── Fixtures ─────────────────────────────────────────────── #
# BufferRegistry is a library singleton. Per Rule 9, a session-
# scoped fixture provides a fresh instance shared across tests.
# Each test that needs isolation registers under a unique parent
# (BufferRegistry treats each parent as an independent group).


@pytest.fixture(scope="session")
def fresh_registry():
    """Fresh BufferRegistry instance (not the module singleton)."""
    return BufferRegistry()


# ── CUDABuffer construction and type properties ──────────── #


@pytest.mark.parametrize(
    "location, persistent, expected_shared, expected_local, "
    "expected_persistent",
    [
        pytest.param(
            "shared", False, True, False, False, id="shared",
        ),
        pytest.param(
            "local", False, False, True, False, id="local",
        ),
        pytest.param(
            "local", True, False, False, True, id="persistent",
        ),
    ],
)
def test_cuda_buffer_type_properties(
    location, persistent, expected_shared, expected_local,
    expected_persistent,
):
    """CUDABuffer type properties reflect location and persistent.

    Inline construction justified: testing CUDABuffer __init__ and
    derived boolean properties directly.
    """
    buf = CUDABuffer(
        name="buf", size=10, location=location,
        persistent=persistent,
    )
    assert buf.is_shared is expected_shared
    assert buf.is_local is expected_local
    assert buf.is_persistent_local is expected_persistent


def test_cuda_buffer_construction_stores_fields():
    """CUDABuffer stores all constructor arguments.

    Inline construction justified: testing __init__ field storage.
    """
    buf = CUDABuffer(
        name="test", size=42, location="shared",
        persistent=False, aliases="parent", precision=np.float64,
    )
    assert buf.name == "test"
    assert buf.size == 42
    assert buf.location == "shared"
    assert buf.persistent is False
    assert buf.aliases == "parent"
    assert buf.precision == np.float64


def test_cuda_buffer_defaults():
    """CUDABuffer defaults: persistent=False, aliases=None, precision=float32.

    Inline construction justified: testing __init__ defaults.
    """
    buf = CUDABuffer(name="d", size=5, location="local")
    assert buf.persistent is False
    assert buf.aliases is None
    assert buf.precision == np.float32


def test_cuda_buffer_invalid_location_raises():
    """CUDABuffer raises ValueError for invalid location.

    Inline construction justified: testing __init__ validation.
    """
    with pytest.raises(ValueError):
        CUDABuffer(name="x", size=1, location="invalid")


def test_cuda_buffer_invalid_precision_raises():
    """CUDABuffer raises ValueError for unsupported precision.

    Inline construction justified: testing __init__ validation.
    """
    with pytest.raises(ValueError, match="float16, float32, float64"):
        CUDABuffer(
            name="x", size=1, location="shared",
            precision=np.complex64,
        )


@pytest.mark.parametrize(
    "precision",
    [
        pytest.param(np.float32, id="float32"),
        pytest.param(np.float64, id="float64"),
        pytest.param(np.float16, id="float16"),
        pytest.param(np.int32, id="int32"),
        pytest.param(np.int64, id="int64"),
    ],
)
def test_cuda_buffer_valid_precisions(precision):
    """CUDABuffer accepts all supported buffer dtype types.

    Inline construction justified: testing __init__ precision validation.
    """
    buf = CUDABuffer(
        name="x", size=1, location="shared", precision=precision,
    )
    assert buf.precision == precision


# ── CUDABuffer.build_allocator ────────────────────────────── #


def test_build_allocator_shared_slice():
    """build_allocator with shared_slice returns callable allocator.

    Inline construction justified: testing CUDABuffer.build_allocator
    directly; no fixture provides bare CUDABuffer instances.
    """
    buf = CUDABuffer(name="s", size=10, location="shared")
    alloc = buf.build_allocator(
        shared_slice=slice(0, 10),
        persistent_slice=None,
        local_size=None,
        zero=False,
    )
    assert callable(alloc)
    assert alloc.__name__ == "allocate_buffer"


def test_build_allocator_persistent_slice():
    """build_allocator with persistent_slice (no shared) returns callable.

    Inline construction justified: testing CUDABuffer.build_allocator.
    """
    buf = CUDABuffer(
        name="p", size=5, location="local", persistent=True,
    )
    alloc = buf.build_allocator(
        shared_slice=None,
        persistent_slice=slice(0, 5),
        local_size=None,
        zero=False,
    )
    assert callable(alloc)
    assert alloc.__name__ == "allocate_buffer"


def test_build_allocator_local_size():
    """build_allocator with local_size (no shared/persistent) returns callable.

    Inline construction justified: testing CUDABuffer.build_allocator.
    """
    buf = CUDABuffer(name="l", size=3, location="local")
    alloc = buf.build_allocator(
        shared_slice=None,
        persistent_slice=None,
        local_size=3,
        zero=False,
    )
    assert callable(alloc)
    assert alloc.__name__ == "allocate_buffer"


def test_build_allocator_zero_flag():
    """build_allocator with zero=True produces a different closure.

    Inline construction justified: testing CUDABuffer.build_allocator
    zero-flag branch.
    """
    buf = CUDABuffer(name="z", size=4, location="shared")
    alloc_no_zero = buf.build_allocator(
        shared_slice=slice(0, 4),
        persistent_slice=None,
        local_size=None,
        zero=False,
    )
    alloc_zero = buf.build_allocator(
        shared_slice=slice(0, 4),
        persistent_slice=None,
        local_size=None,
        zero=True,
    )
    assert callable(alloc_no_zero)
    assert callable(alloc_zero)
    assert alloc_no_zero is not alloc_zero


# ── BufferGroup.register validation ──────────────────────── #


def test_register_empty_name_raises(single_integrator_run):
    """BufferGroup.register raises ValueError for empty name."""
    group = BufferGroup(parent=single_integrator_run)
    with pytest.raises(ValueError, match="cannot be empty"):
        group.register("", 10, "shared")


def test_register_self_alias_raises(single_integrator_run):
    """BufferGroup.register raises ValueError for self-aliasing."""
    group = BufferGroup(parent=single_integrator_run)
    with pytest.raises(ValueError, match="cannot alias itself"):
        group.register("buf", 10, "shared", aliases="buf")


def test_register_missing_alias_target_raises(single_integrator_run):
    """BufferGroup.register raises ValueError for missing alias target."""
    group = BufferGroup(parent=single_integrator_run)
    with pytest.raises(ValueError, match="not registered"):
        group.register("child", 10, "shared", aliases="parent")


def test_register_adds_entry_and_invalidates(single_integrator_run):
    """Registration adds entry to group and invalidates layouts."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("buf", 10, "shared")
    group.build_layouts()
    assert group._shared_layout is not None

    group.register("buf2", 5, "shared")
    assert "buf2" in group.entries
    assert group.entries["buf2"].size == 5
    assert group.entries["buf2"].location == "shared"
    assert group._shared_layout is None


# ── BufferGroup.update_buffer ─────────────────────────────── #


def test_update_buffer_unregistered_returns_false_false(
    single_integrator_run,
):
    """update_buffer returns (False, False) for unknown buffer."""
    group = BufferGroup(parent=single_integrator_run)
    recognized, changed = group.update_buffer("missing", size=10)
    assert recognized is False
    assert changed is False


def test_update_buffer_no_change_returns_true_false(
    single_integrator_run,
):
    """update_buffer returns (True, False) when values unchanged."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("buf", 10, "shared")
    recognized, changed = group.update_buffer("buf", size=10)
    assert recognized is True
    assert changed is False


def test_update_buffer_changed_returns_true_true(
    single_integrator_run,
):
    """update_buffer returns (True, True) and invalidates on change."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("buf", 10, "shared")
    group.build_layouts()
    assert group._shared_layout is not None

    recognized, changed = group.update_buffer("buf", size=20)
    assert recognized is True
    assert changed is True
    assert group.entries["buf"].size == 20
    assert group._shared_layout is None


# ── BufferGroup.invalidate_layouts ────────────────────────── #


def test_invalidate_layouts_clears_all(single_integrator_run):
    """invalidate_layouts sets all caches to None."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("s", 10, "shared")
    group.register("p", 5, "local", persistent=True)
    group.register("l", 3, "local")
    group.build_layouts()

    group.invalidate_layouts()
    assert group._shared_layout is None
    assert group._persistent_layout is None
    assert group._local_sizes is None
    assert group._alias_consumption == {}


# ── BufferGroup.build_layouts ─────────────────────────────── #


def test_build_layouts_shared_sequential_offsets(
    single_integrator_run,
):
    """build_layouts assigns sequential shared slices."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("a", 10, "shared")
    group.register("b", 20, "shared")
    group.build_layouts()

    assert group.shared_layout["a"] == slice(0, 10)
    assert group.shared_layout["b"] == slice(10, 30)


def test_build_layouts_persistent_sequential_offsets(
    single_integrator_run,
):
    """build_layouts assigns sequential persistent slices."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("a", 15, "local", persistent=True)
    group.register("b", 25, "local", persistent=True)
    group.build_layouts()

    assert group.persistent_layout["a"] == slice(0, 15)
    assert group.persistent_layout["b"] == slice(15, 40)


def test_build_layouts_local_sizes_min_one(single_integrator_run):
    """build_layouts uses max(size, 1) for local buffers."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("zero", 0, "local")
    group.register("nonzero", 7, "local")
    group.build_layouts()

    assert group.local_sizes["zero"] == 1
    assert group.local_sizes["nonzero"] == 7


def test_build_layouts_short_circuits_when_populated(
    single_integrator_run,
):
    """build_layouts returns early when all caches already built."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("s", 10, "shared")
    group.build_layouts()
    original = group._shared_layout

    group.build_layouts()
    assert group._shared_layout is original


# ── BufferGroup.layout_aliases ────────────────────────────── #


def test_alias_overlaps_shared_parent(single_integrator_run):
    """Aliased buffer overlaps within shared parent when space."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("parent", 100, "shared")
    group.register("child", 30, "shared", aliases="parent")
    group.build_layouts()

    assert group.shared_layout["parent"] == slice(0, 100)
    assert group.shared_layout["child"] == slice(0, 30)


def test_alias_exceeds_parent_falls_back(single_integrator_run):
    """Aliased buffer exceeding parent gets own shared allocation."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("parent", 50, "shared")
    group.register("child", 80, "shared", aliases="parent")
    group.build_layouts()

    assert group.shared_layout["parent"] == slice(0, 50)
    assert group.shared_layout["child"] == slice(50, 130)
    assert group.shared_buffer_size() == 130


def test_alias_fallback_persistent(single_integrator_run):
    """Persistent aliased buffer falls back to persistent layout."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("parent", 10, "local")
    group.register(
        "child", 5, "local", persistent=True, aliases="parent",
    )
    group.build_layouts()

    assert group.persistent_layout["child"] == slice(0, 5)
    assert group.persistent_local_buffer_size() == 5


def test_alias_fallback_local(single_integrator_run):
    """Local aliased buffer falls back to local pile."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("parent", 10, "local", persistent=True)
    group.register("child", 5, "local", aliases="parent")
    group.build_layouts()

    assert group.local_sizes["child"] == 5


def test_alias_local_child_of_shared_parent_overlaps(
    single_integrator_run,
):
    """Local child aliasing shared parent overlaps in shared."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("parent", 100, "shared")
    group.register("child", 30, "local", aliases="parent")
    group.build_layouts()

    assert group.shared_layout["child"] == slice(0, 30)
    assert group.local_buffer_size() == 0


def test_multiple_aliases_sequential_consumption(
    single_integrator_run,
):
    """Multiple aliases consume parent space sequentially."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("parent", 100, "shared")
    group.register("c1", 40, "shared", aliases="parent")
    group.register("c2", 40, "shared", aliases="parent")
    group.register("c3", 40, "shared", aliases="parent")
    group.build_layouts()

    assert group.shared_layout["c1"] == slice(0, 40)
    assert group.shared_layout["c2"] == slice(40, 80)
    # c3 doesn't fit (only 20 left), gets own allocation
    assert group.shared_layout["c3"] == slice(100, 140)
    assert group.shared_buffer_size() == 140


# ── BufferGroup lazy property triggers ────────────────────── #


@pytest.mark.parametrize(
    "prop",
    [
        pytest.param("shared_layout", id="shared"),
        pytest.param("persistent_layout", id="persistent"),
        pytest.param("local_sizes", id="local"),
    ],
)
def test_layout_property_triggers_build(prop, single_integrator_run):
    """Accessing layout property triggers build when None."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("s", 5, "shared")
    group.register("p", 3, "local", persistent=True)
    group.register("l", 2, "local")

    assert group._shared_layout is None
    _ = getattr(group, prop)
    assert group._shared_layout is not None
    assert group._persistent_layout is not None
    assert group._local_sizes is not None


# ── BufferGroup size methods ──────────────────────────────── #


def test_shared_buffer_size_empty(single_integrator_run):
    """shared_buffer_size returns 0 for empty layout."""
    group = BufferGroup(parent=single_integrator_run)
    assert group.shared_buffer_size() == 0


def test_shared_buffer_size_returns_max_stop(single_integrator_run):
    """shared_buffer_size returns max slice stop."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("a", 10, "shared")
    group.register("b", 20, "shared")
    assert group.shared_buffer_size() == 30


def test_local_buffer_size_returns_sum(single_integrator_run):
    """local_buffer_size returns sum of local sizes."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("a", 5, "local")
    group.register("b", 8, "local")
    assert group.local_buffer_size() == 13


def test_persistent_buffer_size_empty(single_integrator_run):
    """persistent_local_buffer_size returns 0 for empty layout."""
    group = BufferGroup(parent=single_integrator_run)
    assert group.persistent_local_buffer_size() == 0


def test_persistent_buffer_size_returns_max_stop(
    single_integrator_run,
):
    """persistent_local_buffer_size returns max slice stop."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("a", 30, "local", persistent=True)
    group.register("b", 40, "local", persistent=True)
    assert group.persistent_local_buffer_size() == 70


# ── BufferGroup.get_allocator ─────────────────────────────── #


def test_get_allocator_unregistered_raises(single_integrator_run):
    """get_allocator raises KeyError for unregistered buffer."""
    group = BufferGroup(parent=single_integrator_run)
    with pytest.raises(KeyError, match="not registered"):
        group.get_allocator("missing")


def test_get_allocator_returns_allocator_for_registered(
    single_integrator_run,
):
    """get_allocator returns allocator with correct name."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("buf", 10, "shared")
    alloc = group.get_allocator("buf")
    assert callable(alloc)
    assert alloc.__name__ == "allocate_buffer"


# ── BufferRegistry central registry ──────────────────────── #


def test_registry_register_creates_group(
    fresh_registry, step_controller,
):
    """register creates new BufferGroup for unknown parent."""
    fresh_registry.register("buf", step_controller, 10, "shared")
    assert step_controller in fresh_registry._groups
    entry = fresh_registry._groups[step_controller].entries["buf"]
    assert entry.size == 10
    assert entry.location == "shared"
    # Cleanup: remove group so other tests start clean
    fresh_registry.clear_parent(step_controller)


def test_registry_register_reuses_group(
    fresh_registry, step_controller,
):
    """register reuses existing group for known parent."""
    fresh_registry.register("a", step_controller, 10, "shared")
    fresh_registry.register("b", step_controller, 5, "shared")
    assert len(fresh_registry._groups) >= 1
    entries = fresh_registry._groups[step_controller].entries
    assert "a" in entries
    assert "b" in entries
    fresh_registry.clear_parent(step_controller)


def test_registry_update_buffer_unknown_parent(
    fresh_registry, output_functions,
):
    """update_buffer returns (False, False) for unknown parent."""
    # output_functions is not registered, so it's genuinely unknown
    recognized, changed = fresh_registry.update_buffer(
        "buf", output_functions,
    )
    assert recognized is False
    assert changed is False


def test_registry_update_buffer_delegates(
    fresh_registry, step_controller,
):
    """update_buffer delegates to group for known parent."""
    fresh_registry.register("buf", step_controller, 10, "shared")
    recognized, changed = fresh_registry.update_buffer(
        "buf", step_controller, size=20,
    )
    assert recognized is True
    assert changed is True
    assert (
        fresh_registry._groups[step_controller].entries["buf"].size
        == 20
    )
    fresh_registry.clear_parent(step_controller)


def test_registry_clear_layout_known_parent(
    fresh_registry, step_controller,
):
    """clear_layout invalidates layouts for known parent."""
    fresh_registry.register("buf", step_controller, 10, "shared")
    _ = fresh_registry.shared_buffer_size(step_controller)
    group = fresh_registry._groups[step_controller]
    assert group._shared_layout is not None

    fresh_registry.clear_layout(step_controller)
    assert group._shared_layout is None
    fresh_registry.clear_parent(step_controller)


def test_registry_clear_layout_unknown_parent_noop(
    fresh_registry, output_functions,
):
    """clear_layout is a no-op for unknown parent."""
    fresh_registry.clear_layout(output_functions)  # should not raise


def test_registry_clear_parent_removes_group(
    fresh_registry, step_controller,
):
    """clear_parent removes group for known parent."""
    fresh_registry.register("buf", step_controller, 10, "shared")
    fresh_registry.clear_parent(step_controller)
    assert step_controller not in fresh_registry._groups


def test_registry_clear_parent_unknown_noop(
    fresh_registry, output_functions,
):
    """clear_parent is a no-op for unknown parent."""
    fresh_registry.clear_parent(output_functions)  # should not raise


def test_registry_reset_clears_all(
    fresh_registry, step_controller, output_functions,
):
    """reset clears all groups."""
    fresh_registry.register("a", step_controller, 10, "shared")
    fresh_registry.register("b", output_functions, 5, "local")
    fresh_registry.reset()
    assert len(fresh_registry._groups) == 0


# ── BufferRegistry.update ─────────────────────────────────── #


def test_registry_update_empty_returns_empty(
    fresh_registry, step_controller,
):
    """update returns empty set for empty updates."""
    fresh_registry.register("buf", step_controller, 10, "local")
    assert fresh_registry.update(step_controller) == set()
    fresh_registry.clear_parent(step_controller)


def test_registry_update_unknown_parent_returns_empty(
    fresh_registry, output_functions,
):
    """update returns empty set for unknown parent."""
    result = fresh_registry.update(
        output_functions, buf_location="shared",
    )
    assert result == set()


def test_registry_update_recognizes_location_keys(
    fresh_registry, step_controller,
):
    """update recognizes keys ending in _location."""
    fresh_registry.register("buf", step_controller, 10, "local")
    recognized = fresh_registry.update(
        step_controller, buf_location="shared",
    )
    assert "buf_location" in recognized
    fresh_registry.clear_parent(step_controller)


def test_registry_update_invalid_location_raises(
    fresh_registry, step_controller,
):
    """update raises ValueError for invalid location value."""
    fresh_registry.register("buf", step_controller, 10, "local")
    try:
        with pytest.raises(ValueError, match="Invalid location"):
            fresh_registry.update(
                step_controller, buf_location="invalid",
            )
    finally:
        fresh_registry.clear_parent(step_controller)


def test_registry_update_changes_location_and_invalidates(
    fresh_registry, step_controller,
):
    """update changes location and invalidates layouts."""
    fresh_registry.register("buf", step_controller, 10, "local")
    _ = fresh_registry.local_buffer_size(step_controller)
    group = fresh_registry._groups[step_controller]
    assert group._local_sizes is not None

    fresh_registry.update(step_controller, buf_location="shared")
    assert group.entries["buf"].location == "shared"
    assert group._local_sizes is None
    fresh_registry.clear_parent(step_controller)


def test_registry_update_returns_all_recognized(
    fresh_registry, step_controller,
):
    """update returns set of all recognized keys."""
    fresh_registry.register("a", step_controller, 10, "local")
    fresh_registry.register("b", step_controller, 5, "local")
    recognized = fresh_registry.update(
        step_controller,
        updates_dict={"a_location": "shared"},
        b_location="shared",
    )
    assert recognized == {"a_location", "b_location"}
    fresh_registry.clear_parent(step_controller)


def test_registry_update_ignores_non_location_keys(
    fresh_registry, step_controller,
):
    """update ignores params not ending in _location."""
    fresh_registry.register("buf", step_controller, 10, "local")
    recognized = fresh_registry.update(
        step_controller, other_param="value",
    )
    assert recognized == set()
    fresh_registry.clear_parent(step_controller)


def test_registry_update_no_change_preserves_layout(
    fresh_registry, step_controller,
):
    """update preserves layout when location unchanged."""
    fresh_registry.register("buf", step_controller, 10, "local")
    _ = fresh_registry.local_buffer_size(step_controller)
    group = fresh_registry._groups[step_controller]
    assert group._local_sizes is not None

    fresh_registry.update(step_controller, buf_location="local")
    assert group._local_sizes is not None
    fresh_registry.clear_parent(step_controller)


# ── BufferRegistry size delegation ────────────────────────── #


@pytest.mark.parametrize(
    "method",
    [
        pytest.param("shared_buffer_size", id="shared"),
        pytest.param("local_buffer_size", id="local"),
        pytest.param("persistent_local_buffer_size", id="persistent"),
    ],
)
def test_registry_size_unknown_parent_returns_zero(
    fresh_registry, output_functions, method,
):
    """Size methods return 0 for unknown parent."""
    assert getattr(fresh_registry, method)(output_functions) == 0


def test_registry_size_delegates_to_group(
    fresh_registry, step_controller,
):
    """Size methods delegate to group methods for known parent."""
    fresh_registry.register("s", step_controller, 10, "shared")
    fresh_registry.register("l", step_controller, 5, "local")
    fresh_registry.register(
        "p", step_controller, 3, "local", persistent=True,
    )

    assert fresh_registry.shared_buffer_size(step_controller) == 10
    assert fresh_registry.local_buffer_size(step_controller) == 5
    assert (
        fresh_registry.persistent_local_buffer_size(step_controller)
        == 3
    )
    fresh_registry.clear_parent(step_controller)


# ── BufferRegistry.get_allocator ──────────────────────────── #


def test_registry_get_allocator_unknown_parent_raises(
    fresh_registry, output_functions,
):
    """get_allocator raises KeyError for unknown parent."""
    with pytest.raises(KeyError, match="no registered"):
        fresh_registry.get_allocator("buf", output_functions)


def test_registry_get_allocator_delegates(
    fresh_registry, step_controller,
):
    """get_allocator delegates to group for known parent."""
    fresh_registry.register("buf", step_controller, 10, "shared")
    alloc = fresh_registry.get_allocator("buf", step_controller)
    assert callable(alloc)
    assert alloc.__name__ == "allocate_buffer"
    fresh_registry.clear_parent(step_controller)


# ── BufferRegistry separate parent contexts ───────────────── #


def test_separate_parents_independent(
    fresh_registry, step_controller, output_functions,
):
    """Different parents have independent buffer groups."""
    fresh_registry.register("buf", step_controller, 100, "shared")
    fresh_registry.register("buf", output_functions, 50, "shared")
    assert (
        fresh_registry.shared_buffer_size(step_controller) == 100
    )
    assert (
        fresh_registry.shared_buffer_size(output_functions) == 50
    )
    fresh_registry.clear_parent(step_controller)
    fresh_registry.clear_parent(output_functions)


def test_clear_one_parent_preserves_others(
    fresh_registry, step_controller, output_functions,
):
    """Clearing one parent does not affect others."""
    fresh_registry.register("buf", step_controller, 100, "shared")
    fresh_registry.register("buf", output_functions, 50, "shared")
    fresh_registry.clear_parent(step_controller)
    assert step_controller not in fresh_registry._groups
    assert (
        fresh_registry.shared_buffer_size(output_functions) == 50
    )
    fresh_registry.clear_parent(output_functions)


# ── BufferRegistry.get_child_allocators ───────────────────── #


def test_get_child_allocators_registers_buffers(
    single_integrator_run,
):
    """get_child_allocators registers child shared/persistent."""
    reg = BufferRegistry()
    parent = single_integrator_run
    child = single_integrator_run._loop

    child_shared = buffer_registry.shared_buffer_size(child)
    child_persistent = buffer_registry.persistent_local_buffer_size(
        child,
    )

    alloc_s, alloc_p = reg.get_child_allocators(
        parent=parent, child=child, name="loop",
    )
    entries = reg._groups[parent].entries
    assert entries["loop_shared"].size == child_shared
    assert entries["loop_persistent"].size == child_persistent
    assert callable(alloc_s)
    assert callable(alloc_p)
    assert alloc_s.__name__ == "allocate_buffer"
    assert alloc_p.__name__ == "allocate_buffer"


def test_get_child_allocators_with_name(single_integrator_run):
    """get_child_allocators uses provided name for buffer names."""
    reg = BufferRegistry()
    parent = single_integrator_run
    child = single_integrator_run._algo_step

    reg.get_child_allocators(
        parent=parent, child=child, name="solver",
    )
    assert "solver_shared" in reg._groups[parent].entries
    assert "solver_persistent" in reg._groups[parent].entries


def test_get_child_allocators_default_name(single_integrator_run):
    """get_child_allocators uses child_{id} when name=None."""
    reg = BufferRegistry()
    parent = single_integrator_run
    child = single_integrator_run._algo_step

    reg.get_child_allocators(
        parent=parent, child=child, name=None,
    )
    child_id = id(child)
    expected_shared = f"child_{child_id}_shared"
    expected_persistent = f"child_{child_id}_persistent"
    assert expected_shared in reg._groups[parent].entries
    assert expected_persistent in reg._groups[parent].entries


# ── BufferRegistry.get_toplevel_allocators ────────────────── #


def test_get_toplevel_allocators_returns_callables(solverkernel):
    """get_toplevel_allocators returns (alloc_shared, alloc_persistent)."""
    reg = BufferRegistry()
    alloc_shared, alloc_persistent = reg.get_toplevel_allocators(
        solverkernel,
    )
    assert callable(alloc_shared)
    assert callable(alloc_persistent)
    assert alloc_shared.__name__ == "alloc_shared"
    assert alloc_persistent.__name__ == "alloc_persistent"


# ── Deterministic layout order ────────────────────────────── #


def test_layout_deterministic_regardless_of_access_order(
    single_integrator_run,
):
    """Layout is deterministic regardless of property access order."""
    group = BufferGroup(parent=single_integrator_run)
    group.register("parent", 100, "shared")
    group.register("child", 30, "shared", aliases="parent")
    group.register("local", 20, "local")
    group.register("persist", 10, "local", persistent=True)

    group.build_layouts()
    shared1 = dict(group.shared_layout)
    persistent1 = dict(group.persistent_layout)
    local1 = dict(group.local_sizes)

    group.invalidate_layouts()

    # Access in different order
    local2 = dict(group.local_sizes)
    persistent2 = dict(group.persistent_layout)
    shared2 = dict(group.shared_layout)

    assert shared1 == shared2
    assert persistent1 == persistent2
    assert local1 == local2
