"""Tests for BufferSettings infrastructure migration status.

Note: The BufferSettings classes have been replaced by the buffer_registry API.
These tests verify the deprecated base classes still exist for backwards
compatibility, but the concrete subclasses (LinearSolverBufferSettings, etc.)
have been removed.
"""
import pytest

from cubie.BufferSettings import (
    BufferSettings,
    LocalSizes,
    SliceIndices,
)


class TestDeprecatedBaseClasses:
    """Tests for deprecated base classes in BufferSettings module.
    
    These base classes are kept for backwards compatibility with any external
    code that may have subclassed them. The concrete implementations have been
    migrated to use buffer_registry.
    """

    def test_local_sizes_base_class_exists(self):
        """LocalSizes base class should still be importable."""
        sizes = LocalSizes()
        # Should have nonzero method
        assert hasattr(sizes, 'nonzero')

    def test_slice_indices_base_class_exists(self):
        """SliceIndices base class should still be importable."""
        indices = SliceIndices()
        assert indices is not None

    def test_buffer_settings_is_abstract(self):
        """BufferSettings base class should be abstract."""
        # BufferSettings has abstract methods, cannot instantiate directly
        with pytest.raises(TypeError):
            BufferSettings()
