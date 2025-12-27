"""Tests for the centralized buffer registry."""

import pytest
import numpy as np

from cubie.buffer_registry import (
    CUDABuffer,
    BufferRegistry,
)


class MockFactory:
    """Mock factory for testing buffer registration."""
    pass


class TestCUDABuffer:
    """Tests for CUDABuffer attrs class."""

    def test_create_shared_entry(self):
        entry = CUDABuffer(
            name='test_buffer',
            size=100,
            location='shared',
        )
        assert entry.is_shared
        assert not entry.is_local
        assert not entry.is_persistent_local

    def test_create_local_entry(self):
        entry = CUDABuffer(
            name='test_buffer',
            size=50,
            location='local',
            persistent=False,
        )
        assert entry.is_local
        assert not entry.is_shared
        assert not entry.is_persistent_local

    def test_create_persistent_local_entry(self):
        entry = CUDABuffer(
            name='test_buffer',
            size=25,
            location='local',
            persistent=True,
        )
        assert entry.is_persistent_local
        assert not entry.is_local
        assert not entry.is_shared

    def test_invalid_location_raises(self):
        with pytest.raises(ValueError):
            CUDABuffer(
                name='test',
                size=10,
                location='invalid',
            )


class TestBufferRegistry:
    """Tests for BufferRegistry singleton."""

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        """Create a fresh registry for each test."""
        self.registry = BufferRegistry()
        self.factory = MockFactory()
        yield

    def test_register_creates_context(self):
        self.registry.register(
            'buffer1', self.factory, 100, 'shared'
        )
        assert self.factory in self.registry._groups
        assert 'buffer1' in self.registry._groups[self.factory].entries

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            self.registry.register('', self.factory, 100, 'shared')

    def test_self_alias_raises(self):
        with pytest.raises(ValueError, match="cannot alias itself"):
            self.registry.register(
                'buffer1', self.factory, 100, 'shared', aliases='buffer1'
            )

    def test_alias_nonexistent_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            self.registry.register(
                'buffer1', self.factory, 50, 'shared', aliases='parent'
            )

    def test_valid_aliasing(self):
        self.registry.register('parent', self.factory, 100, 'shared')
        self.registry.register(
            'child', self.factory, 30, 'shared', aliases='parent'
        )
        context = self.registry._groups[self.factory]
        assert context.entries['child'].aliases == 'parent'

    def test_shared_buffer_size_excludes_aliases(self):
        self.registry.register('parent', self.factory, 100, 'shared')
        self.registry.register(
            'child', self.factory, 30, 'shared', aliases='parent'
        )
        # Only parent counts toward total
        size = self.registry.shared_buffer_size(self.factory)
        assert size == 100

    def test_local_buffer_size_minimum_one(self):
        self.registry.register('zero_size', self.factory, 0, 'local')
        size = self.registry.local_buffer_size(self.factory)
        assert size == 1  # max(0, 1) = 1

    def test_persistent_local_size(self):
        self.registry.register(
            'persist', self.factory, 50, 'local', persistent=True
        )
        size = self.registry.persistent_local_buffer_size(self.factory)
        assert size == 50

    def test_unregistered_factory_returns_zero(self):
        other_factory = MockFactory()
        assert self.registry.shared_buffer_size(other_factory) == 0
        assert self.registry.local_buffer_size(other_factory) == 0
        assert self.registry.persistent_local_buffer_size(other_factory) == 0

    def test_update_buffer_silent_ignore_unregistered(self):
        other_factory = MockFactory()
        # Should not raise
        self.registry.update_buffer('buffer', other_factory, size=200)

    def test_clear_factory(self):
        self.registry.register('buffer1', self.factory, 100, 'shared')
        self.registry.clear_parent(self.factory)
        assert self.factory not in self.registry._groups


class TestLayoutComputation:
    """Tests for lazy layout computation."""

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        self.registry = BufferRegistry()
        self.factory = MockFactory()
        yield

    def test_shared_layout_computed_on_access(self):
        self.registry.register('buf1', self.factory, 100, 'shared')
        self.registry.register('buf2', self.factory, 50, 'shared')
        context = self.registry._groups[self.factory]

        # Layout should be None initially after registration
        assert context._shared_layout is None

        # Access triggers computation
        _ = self.registry.shared_buffer_size(self.factory)
        assert context._shared_layout is not None

    def test_aliasing_slices_computed_correctly(self):
        self.registry.register('parent', self.factory, 100, 'shared')
        self.registry.register(
            'child1', self.factory, 30, 'shared', aliases='parent'
        )
        self.registry.register(
            'child2', self.factory, 20, 'shared', aliases='parent'
        )

        # Trigger layout computation
        _ = self.registry.shared_buffer_size(self.factory)
        context = self.registry._groups[self.factory]
        layout = context._shared_layout

        # parent: slice(0, 100)
        # child1: slice(0, 30) - aliases first 30 of parent
        # child2: slice(30, 50) - aliases next 20 of parent
        assert layout['parent'] == slice(0, 100)
        assert layout['child1'] == slice(0, 30)
        assert layout['child2'] == slice(30, 50)

    def test_layout_invalidated_on_new_registration(self):
        self.registry.register('buf1', self.factory, 100, 'shared')
        _ = self.registry.shared_buffer_size(self.factory)
        context = self.registry._groups[self.factory]
        assert context._shared_layout is not None

        # New registration invalidates
        self.registry.register('buf2', self.factory, 50, 'shared')
        assert context._shared_layout is None

    def test_build_layouts_populates_all_caches(self):
        """build_layouts() populates all three layout caches."""
        self.registry.register('shared', self.factory, 100, 'shared')
        self.registry.register(
            'persist', self.factory, 50, 'local', persistent=True
        )
        self.registry.register('local', self.factory, 25, 'local')

        group = self.registry._groups[self.factory]

        # All caches should be None initially
        assert group._shared_layout is None
        assert group._persistent_layout is None
        assert group._local_sizes is None

        # Call build_layouts
        group.build_layouts()

        # All caches should be populated
        assert group._shared_layout is not None
        assert group._persistent_layout is not None
        assert group._local_sizes is not None

        # Verify contents
        assert 'shared' in group._shared_layout
        assert 'persist' in group._persistent_layout
        assert 'local' in group._local_sizes

    def test_build_layouts_early_return_when_populated(self):
        """build_layouts() returns early when all caches populated."""
        self.registry.register('shared', self.factory, 100, 'shared')
        group = self.registry._groups[self.factory]

        # First call populates
        group.build_layouts()
        original_layout = group._shared_layout

        # Second call should be no-op
        group.build_layouts()

        # Should be exact same object (not rebuilt)
        assert group._shared_layout is original_layout


class TestGetAllocator:
    """Tests for allocator generation."""

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        self.registry = BufferRegistry()
        self.factory = MockFactory()
        yield

    def test_get_allocator_unregistered_factory_raises(self):
        other_factory = MockFactory()
        with pytest.raises(KeyError, match="no registered"):
            self.registry.get_allocator('buffer', other_factory)

    def test_get_allocator_unregistered_buffer_raises(self):
        self.registry.register('buffer1', self.factory, 100, 'shared')
        with pytest.raises(KeyError, match="not registered"):
            self.registry.get_allocator('nonexistent', self.factory)

    def test_get_allocator_returns_callable(self):
        self.registry.register('buffer1', self.factory, 100, 'shared')
        allocator = self.registry.get_allocator('buffer1', self.factory)
        assert callable(allocator)

class TestMultipleFactories:
    """Tests for multiple factory contexts."""

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        self.registry = BufferRegistry()
        yield

    def test_separate_contexts_for_different_factories(self):
        factory1 = MockFactory()
        factory2 = MockFactory()

        self.registry.register('buf', factory1, 100, 'shared')
        self.registry.register('buf', factory2, 50, 'shared')

        assert self.registry.shared_buffer_size(factory1) == 100
        assert self.registry.shared_buffer_size(factory2) == 50

    def test_clear_one_factory_doesnt_affect_others(self):
        factory1 = MockFactory()
        factory2 = MockFactory()

        self.registry.register('buf', factory1, 100, 'shared')
        self.registry.register('buf', factory2, 50, 'shared')

        self.registry.clear_parent(factory1)

        assert factory1 not in self.registry._groups
        assert factory2 in self.registry._groups
        assert self.registry.shared_buffer_size(factory2) == 50


class TestPersistentLocal:
    """Tests for persistent local buffer handling."""

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        self.registry = BufferRegistry()
        self.factory = MockFactory()
        yield

    def test_persistent_flag_distinguishes_local_types(self):
        self.registry.register('local_buf', self.factory, 10, 'local')
        self.registry.register(
            'persist_buf', self.factory, 20, 'local', persistent=True
        )

        assert self.registry.local_buffer_size(self.factory) == 10
        assert self.registry.persistent_local_buffer_size(self.factory) == 20

    def test_persistent_layout_computed_correctly(self):
        self.registry.register(
            'persist1', self.factory, 30, 'local', persistent=True
        )
        self.registry.register(
            'persist2', self.factory, 40, 'local', persistent=True
        )

        # Trigger layout computation
        _ = self.registry.persistent_local_buffer_size(self.factory)
        context = self.registry._groups[self.factory]
        layout = context._persistent_layout

        # Both should have valid slices
        assert 'persist1' in layout
        assert 'persist2' in layout
        # Total size is 70
        total = self.registry.persistent_local_buffer_size(self.factory)
        assert total == 70

class TestBufferUpdate:
    """Tests for buffer update functionality."""

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        self.registry = BufferRegistry()
        self.factory = MockFactory()
        yield

    def test_update_buffer_changes_size(self):
        self.registry.register('buf', self.factory, 100, 'shared')
        self.registry.update_buffer('buf', self.factory, size=200)

        context = self.registry._groups[self.factory]
        assert context.entries['buf'].size == 200

    def test_update_invalidates_layout(self):
        self.registry.register('buf', self.factory, 100, 'shared')
        _ = self.registry.shared_buffer_size(self.factory)
        context = self.registry._groups[self.factory]
        assert context._shared_layout is not None

        self.registry.update_buffer('buf', self.factory, size=200)
        assert context._shared_layout is None

    def test_update_nonexistent_buffer_silent(self):
        self.registry.register('buf', self.factory, 100, 'shared')
        # Should not raise
        self.registry.update_buffer('other', self.factory, size=200)


class TestCrossTypeAliasing:
    """Tests for cross-type aliasing validation."""

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        self.registry = BufferRegistry()
        self.factory = MockFactory()
        yield

    def test_persistent_alias_of_persistent_allowed(self):
        self.registry.register(
            'parent', self.factory, 100, 'local', persistent=True
        )
        self.registry.register(
            'child', self.factory, 30, 'local',
            persistent=True, aliases='parent'
        )
        context = self.registry._groups[self.factory]
        assert context.entries['child'].aliases == 'parent'

    def test_local_alias_of_local_allowed(self):
        self.registry.register('parent', self.factory, 100, 'local')
        self.registry.register(
            'child', self.factory, 30, 'local', aliases='parent'
        )
        context = self.registry._groups[self.factory]
        assert context.entries['child'].aliases == 'parent'

    def test_shared_alias_of_shared_allowed(self):
        self.registry.register('parent', self.factory, 100, 'shared')
        self.registry.register(
            'child', self.factory, 30, 'shared', aliases='parent'
        )
        context = self.registry._groups[self.factory]
        assert context.entries['child'].aliases == 'parent'


class TestCrossLocationAliasing:
    """Tests for cross-location aliasing behavior."""

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        self.registry = BufferRegistry()
        self.parent = MockFactory()
        yield

    def test_shared_buffer_can_alias_local_parent(self):
        """Shared buffer aliasing local parent falls back to own allocation."""
        self.registry.register('parent', self.parent, 100, 'local')
        self.registry.register(
            'child', self.parent, 30, 'shared', aliases='parent'
        )
        # Child should be allocated in shared memory (fallback)
        size = self.registry.shared_buffer_size(self.parent)
        assert size == 30  # Child allocated separately

    def test_local_buffer_can_alias_shared_parent(self):
        """Local buffer aliasing shared parent overlaps in shared memory."""
        self.registry.register('parent', self.parent, 100, 'shared')
        self.registry.register(
            'child', self.parent, 30, 'local', aliases='parent'
        )
        # Child overlaps parent in shared memory when space available
        shared_size = self.registry.shared_buffer_size(self.parent)
        local_size = self.registry.local_buffer_size(self.parent)
        assert shared_size == 100  # Only parent (child overlaps)
        assert local_size == 0  # Child in shared, not local

    def test_alias_fallback_when_parent_too_small(self):
        """Alias falls back to own allocation when parent insufficient."""
        self.registry.register('parent', self.parent, 50, 'shared')
        self.registry.register(
            'child', self.parent, 80, 'shared', aliases='parent'
        )
        # Child needs 80 but parent only has 50, so allocates separately
        size = self.registry.shared_buffer_size(self.parent)
        assert size == 130  # Total allocated space including fallback

        # But layout should have both
        group = self.registry._groups[self.parent]
        _ = self.registry.shared_buffer_size(self.parent)
        layout = group._shared_layout
        assert 'parent' in layout
        assert 'child' in layout
        # Child should have its own allocation after parent
        assert layout['parent'] == slice(0, 50)
        assert layout['child'] == slice(50, 130)

    def test_multiple_aliases_first_come_first_serve(self):
        """Multiple aliases consume parent space sequentially."""
        self.registry.register('parent', self.parent, 100, 'shared')
        self.registry.register(
            'child1', self.parent, 40, 'shared', aliases='parent'
        )
        self.registry.register(
            'child2', self.parent, 40, 'shared', aliases='parent'
        )
        self.registry.register(
            'child3', self.parent, 40, 'shared', aliases='parent'
        )

        group = self.registry._groups[self.parent]
        size = self.registry.shared_buffer_size(self.parent)
        layout = group._shared_layout

        # parent: slice(0, 100)
        # child1: slice(0, 40) within parent
        # child2: slice(40, 80) within parent
        # child3: doesn't fit (only 20 left), gets own allocation
        assert layout['parent'] == slice(0, 100)
        assert layout['child1'] == slice(0, 40)
        assert layout['child2'] == slice(40, 80)
        assert layout['child3'] == slice(100, 140)
        # Total size should be 140 (max slice.stop)
        assert size == 140

    def test_persistent_alias_of_nonpersistent_local_allowed(self):
        """Persistent buffer can now alias non-persistent local."""
        self.registry.register('parent', self.parent, 100, 'local')
        self.registry.register(
            'child', self.parent, 30, 'local',
            persistent=True, aliases='parent'
        )
        # Should not raise; child gets own persistent allocation
        group = self.registry._groups[self.parent]
        assert group.entries['child'].aliases == 'parent'
        persist_size = self.registry.persistent_local_buffer_size(
            self.parent
        )
        assert persist_size == 30

    def test_aliased_shared_child_of_local_parent_gets_shared_allocation(self):
        """Shared child aliasing local parent allocates in shared layout."""
        self.registry.register('parent', self.parent, 100, 'local')
        self.registry.register(
            'child', self.parent, 30, 'shared', aliases='parent'
        )

        shared_size = self.registry.shared_buffer_size(self.parent)
        local_size = self.registry.local_buffer_size(self.parent)

        # Child should be in shared (fallback)
        assert shared_size == 30
        # Parent in local
        assert local_size == 100

    def test_aliased_persistent_child_of_local_parent_gets_persistent(self):
        """Persistent child aliasing local parent allocates in persistent."""
        self.registry.register('parent', self.parent, 100, 'local')
        self.registry.register(
            'child', self.parent, 30, 'local',
            persistent=True, aliases='parent'
        )

        persist_size = self.registry.persistent_local_buffer_size(
            self.parent
        )
        local_size = self.registry.local_buffer_size(self.parent)

        # Child should be in persistent (fallback)
        assert persist_size == 30
        # Parent in local
        assert local_size == 100

    def test_aliased_local_child_of_shared_parent_gets_shared(self):
        """Local child aliasing shared parent overlaps in shared memory."""
        self.registry.register('parent', self.parent, 100, 'shared')
        self.registry.register(
            'child', self.parent, 30, 'local', aliases='parent'
        )

        shared_size = self.registry.shared_buffer_size(self.parent)
        local_size = self.registry.local_buffer_size(self.parent)

        # Parent in shared
        assert shared_size == 100
        # Child overlaps parent in shared, not in local
        assert local_size == 0


class TestDeterministicLayouts:
    """Tests for deterministic layout building order."""

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        self.registry = BufferRegistry()
        self.parent = MockFactory()
        yield

    def test_property_access_order_shared_first(self):
        """Accessing shared_layout first produces same result."""
        self.registry.register('parent', self.parent, 100, 'shared')
        self.registry.register(
            'child', self.parent, 30, 'shared', aliases='parent'
        )
        self.registry.register('local', self.parent, 20, 'local')

        group = self.registry._groups[self.parent]

        # Access shared first
        shared = group.shared_layout.copy()
        persistent = group.persistent_layout.copy()
        local = group.local_sizes.copy()

        # Invalidate and access in different order
        group.invalidate_layouts()

        # Access local first
        local2 = group.local_sizes.copy()
        persistent2 = group.persistent_layout.copy()
        shared2 = group.shared_layout.copy()

        assert shared == shared2
        assert persistent == persistent2
        assert local == local2

    def test_property_access_order_persistent_first(self):
        """Accessing persistent_layout first produces same result."""
        self.registry.register(
            'persist', self.parent, 50, 'local', persistent=True
        )
        self.registry.register('shared', self.parent, 100, 'shared')
        self.registry.register(
            'child', self.parent, 30, 'shared', aliases='shared'
        )

        group = self.registry._groups[self.parent]

        # Access persistent first
        persistent = group.persistent_layout.copy()
        shared = group.shared_layout.copy()
        local = group.local_sizes.copy()

        # Invalidate and access shared first
        group.invalidate_layouts()

        shared2 = group.shared_layout.copy()
        local2 = group.local_sizes.copy()
        persistent2 = group.persistent_layout.copy()

        assert shared == shared2
        assert persistent == persistent2
        assert local == local2


class TestPrecisionValidation:
    """Tests for precision validation."""

    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError, match="float16, float32, float64"):
            CUDABuffer(
                name='test',
                size=10,
                location='shared',
                precision=np.complex64,
            )

    def test_valid_float32_precision(self):
        entry = CUDABuffer(
            name='test',
            size=10,
            location='shared',
            precision=np.float32,
        )
        assert entry.precision == np.float32

    def test_valid_float64_precision(self):
        entry = CUDABuffer(
            name='test',
            size=10,
            location='shared',
            precision=np.float64,
        )
        assert entry.precision == np.float64

    def test_valid_float16_precision(self):
        entry = CUDABuffer(
            name='test',
            size=10,
            location='shared',
            precision=np.float16,
        )
        assert entry.precision == np.float16

    def test_valid_int32_precision(self):
        entry = CUDABuffer(
            name='test',
            size=10,
            location='shared',
            precision=np.int32,
        )
        assert entry.precision == np.int32

    def test_valid_int64_precision(self):
        entry = CUDABuffer(
            name='test',
            size=10,
            location='shared',
            precision=np.int64,
        )
        assert entry.precision == np.int64


class TestBufferRegistryUpdate:
    """Tests for BufferRegistry.update() method."""

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        """Create a fresh registry for each test."""
        self.registry = BufferRegistry()
        self.parent = MockFactory()
        yield

    def test_update_recognizes_location_params(self):
        """BufferRegistry.update() recognizes [buffer_name]_location params."""
        self.registry.register('test_buffer', self.parent, 100, 'local')
        recognized = self.registry.update(
            self.parent, test_buffer_location='shared'
        )
        assert 'test_buffer_location' in recognized

    def test_update_changes_buffer_location(self):
        """BufferRegistry.update() changes buffer location in CUDABuffer."""
        self.registry.register('test_buffer', self.parent, 100, 'local')
        self.registry.update(self.parent, test_buffer_location='shared')
        entry = self.registry._groups[self.parent].entries['test_buffer']
        assert entry.location == 'shared'

    def test_update_invalid_location_raises(self):
        """BufferRegistry.update() raises ValueError for invalid location."""
        self.registry.register('test_buffer', self.parent, 100, 'local')
        with pytest.raises(ValueError, match="Invalid location"):
            self.registry.update(self.parent, test_buffer_location='invalid')

    def test_update_unregistered_parent_silent(self):
        """BufferRegistry.update() returns empty set for unregistered parent."""
        other_parent = MockFactory()
        recognized = self.registry.update(
            other_parent, test_buffer_location='shared'
        )
        assert recognized == set()

    def test_update_with_updates_dict(self):
        """BufferRegistry.update() accepts updates_dict parameter."""
        self.registry.register('buf1', self.parent, 100, 'local')
        self.registry.register('buf2', self.parent, 50, 'local')
        recognized = self.registry.update(
            self.parent,
            updates_dict={'buf1_location': 'shared', 'buf2_location': 'shared'}
        )
        assert 'buf1_location' in recognized
        assert 'buf2_location' in recognized

    def test_update_ignores_non_location_params(self):
        """BufferRegistry.update() ignores params not ending in _location."""
        self.registry.register('test_buffer', self.parent, 100, 'local')
        recognized = self.registry.update(
            self.parent, some_other_param='value'
        )
        assert recognized == set()

    def test_update_ignores_unknown_buffer_names(self):
        """BufferRegistry.update() ignores location for unknown buffers."""
        self.registry.register('test_buffer', self.parent, 100, 'local')
        recognized = self.registry.update(
            self.parent, unknown_buffer_location='shared'
        )
        assert 'unknown_buffer_location' not in recognized

    def test_update_invalidates_layout(self):
        """BufferRegistry.update() invalidates layouts when locations change."""
        self.registry.register('test_buffer', self.parent, 100, 'local')
        _ = self.registry.local_buffer_size(self.parent)
        group = self.registry._groups[self.parent]
        assert group._local_sizes is not None

        self.registry.update(self.parent, test_buffer_location='shared')
        assert group._local_sizes is None

    def test_update_no_change_doesnt_invalidate(self):
        """BufferRegistry.update() preserves layout if location unchanged."""
        self.registry.register('test_buffer', self.parent, 100, 'local')
        _ = self.registry.local_buffer_size(self.parent)
        group = self.registry._groups[self.parent]
        assert group._local_sizes is not None

        self.registry.update(self.parent, test_buffer_location='local')
        # Layout should still be valid since location didn't change
        assert group._local_sizes is not None

    def test_update_empty_dict_returns_empty_set(self):
        """BufferRegistry.update() returns empty set for empty updates."""
        self.registry.register('test_buffer', self.parent, 100, 'local')
        recognized = self.registry.update(self.parent)
        assert recognized == set()

    def test_update_merges_dict_and_kwargs(self):
        """BufferRegistry.update() merges updates_dict and kwargs."""
        self.registry.register('buf1', self.parent, 100, 'local')
        self.registry.register('buf2', self.parent, 50, 'local')
        recognized = self.registry.update(
            self.parent,
            updates_dict={'buf1_location': 'shared'},
            buf2_location='shared'
        )
        assert 'buf1_location' in recognized
        assert 'buf2_location' in recognized
