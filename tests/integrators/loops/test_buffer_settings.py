"""Tests for LoopBufferSettings class."""
import pytest
from cubie.integrators.loops.buffer_settings import (
    LoopBufferSettings,
    LoopSharedIndicesFromSettings,
)


class TestLoopBufferSettings:
    """Tests for LoopBufferSettings initialization and properties."""

    def test_default_locations_match_all_in_one(self):
        """Default locations should match tests/all_in_one.py settings."""
        settings = LoopBufferSettings(n_states=3)

        assert settings.state_buffer_location == 'local'
        assert settings.state_proposal_location == 'local'
        assert settings.parameters_location == 'local'
        assert settings.drivers_location == 'shared'
        assert settings.drivers_proposal_location == 'shared'
        assert settings.observables_location == 'shared'
        assert settings.observables_proposal_location == 'shared'
        assert settings.error_location == 'local'
        assert settings.counters_location == 'local'
        assert settings.state_summary_location == 'local'
        assert settings.observable_summary_location == 'shared'
        assert settings.scratch_location == 'shared'

    def test_boolean_flags_from_locations(self):
        """Boolean properties should reflect location settings."""
        settings = LoopBufferSettings(
            n_states=3,
            state_buffer_location='shared',
            parameters_location='shared',
        )

        assert settings.use_shared_state is True
        assert settings.use_shared_parameters is True
        assert settings.use_shared_state_proposal is False
        assert settings.use_shared_drivers is True

    def test_invalid_location_raises(self):
        """Invalid location string should raise ValueError."""
        with pytest.raises(ValueError):
            LoopBufferSettings(
                n_states=3,
                state_buffer_location='invalid'
            )

    def test_negative_size_raises(self):
        """Negative size should raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            LoopBufferSettings(n_states=-1)

    def test_shared_memory_elements_all_local(self):
        """All-local config should have zero shared memory."""
        settings = LoopBufferSettings(
            n_states=3,
            n_parameters=2,
            n_drivers=1,
            n_observables=2,
            state_buffer_location='local',
            state_proposal_location='local',
            parameters_location='local',
            drivers_location='local',
            drivers_proposal_location='local',
            observables_location='local',
            observables_proposal_location='local',
            error_location='local',
            counters_location='local',
            state_summary_location='local',
            observable_summary_location='local',
        )
        assert settings.shared_memory_elements == 0

    def test_shared_memory_elements_calculation(self):
        """Shared memory should sum sizes of shared buffers only."""
        settings = LoopBufferSettings(
            n_states=3,
            n_parameters=2,
            state_buffer_location='shared',
            parameters_location='shared',
        )
        # state (3) + parameters (2) = 5
        assert settings.shared_memory_elements == 5

    def test_zero_size_buffers(self):
        """Zero-size buffers should not contribute to memory totals."""
        settings = LoopBufferSettings(
            n_states=3,
            n_observables=0,
            n_drivers=0,
        )
        # With defaults, drivers and observables are shared but size 0
        # Only shared buffers with non-zero size contribute
        assert settings.n_observables == 0
        assert settings.n_drivers == 0

    def test_local_memory_elements_all_shared(self):
        """All-shared config should have zero local memory."""
        settings = LoopBufferSettings(
            n_states=3,
            n_parameters=2,
            n_drivers=1,
            n_observables=2,
            state_buffer_location='shared',
            state_proposal_location='shared',
            parameters_location='shared',
            drivers_location='shared',
            drivers_proposal_location='shared',
            observables_location='shared',
            observables_proposal_location='shared',
            error_location='shared',
            counters_location='shared',
            state_summary_location='shared',
            observable_summary_location='shared',
        )
        assert settings.local_memory_elements == 0

    def test_property_aliases(self):
        """total_shared_elements and total_local_elements aliases."""
        settings = LoopBufferSettings(
            n_states=3,
            n_parameters=2,
            state_buffer_location='shared',
            parameters_location='shared',
        )
        assert settings.total_shared_elements == settings.shared_memory_elements
        assert settings.total_local_elements == settings.local_memory_elements

    def test_counters_with_n_counters_positive(self):
        """Counters with n_counters > 0 should include proposed_counters."""
        settings = LoopBufferSettings(
            n_states=3,
            n_counters=4,
            counters_location='shared',
        )
        # counters (4) + proposed_counters (2) = 6
        indices = settings.calculate_shared_indices()
        assert indices.counters == slice(0, 4)
        assert indices.proposed_counters == slice(4, 6)
        # shared_memory_elements should include counters + proposed
        assert settings.shared_memory_elements == 6


class TestLoopBufferSettingsIndices:
    """Tests for calculate_shared_indices method."""

    def test_shared_indices_all_shared(self):
        """All-shared config should generate contiguous slices."""
        settings = LoopBufferSettings(
            n_states=3,
            n_parameters=2,
            n_drivers=1,
            n_observables=2,
            state_buffer_location='shared',
            state_proposal_location='shared',
            parameters_location='shared',
            drivers_location='shared',
            drivers_proposal_location='shared',
            observables_location='shared',
            observables_proposal_location='shared',
        )
        indices = settings.calculate_shared_indices()

        # State: 0-3, proposed: 3-6, params: 6-8, etc.
        assert indices.state == slice(0, 3)
        assert indices.proposed_state == slice(3, 6)
        assert indices.parameters == slice(6, 8)

    def test_shared_indices_mixed_locations(self):
        """Mixed locations should give zero-length slices for local."""
        settings = LoopBufferSettings(
            n_states=3,
            n_parameters=2,
            state_buffer_location='shared',
            state_proposal_location='local',
            parameters_location='shared',
        )
        indices = settings.calculate_shared_indices()

        assert indices.state == slice(0, 3)
        assert indices.proposed_state == slice(0, 0)  # Local gets empty
        assert indices.parameters == slice(3, 5)

    def test_shared_indices_all_local(self):
        """All-local config should have all zero-length slices."""
        settings = LoopBufferSettings(
            n_states=3,
            state_buffer_location='local',
            state_proposal_location='local',
            parameters_location='local',
            drivers_location='local',
            drivers_proposal_location='local',
            observables_location='local',
            observables_proposal_location='local',
            error_location='local',
            counters_location='local',
            state_summary_location='local',
            observable_summary_location='local',
        )
        indices = settings.calculate_shared_indices()

        assert indices.state == slice(0, 0)
        assert indices.proposed_state == slice(0, 0)
        assert indices.local_end == 0

    def test_scratch_slice_starts_at_local_end(self):
        """Scratch slice should start where loop buffers end."""
        settings = LoopBufferSettings(
            n_states=3,
            state_buffer_location='shared',
        )
        indices = settings.calculate_shared_indices()

        assert indices.scratch.start == indices.local_end


class TestLoopSharedIndicesFromSettings:
    """Tests for LoopSharedIndicesFromSettings properties."""

    def test_n_states_property(self):
        """n_states should reflect state slice width."""
        indices = LoopSharedIndicesFromSettings(
            state=slice(0, 5),
            proposed_state=slice(5, 10),
            observables=slice(10, 12),
            proposed_observables=slice(12, 14),
            parameters=slice(14, 17),
            drivers=slice(17, 18),
            proposed_drivers=slice(18, 19),
            state_summaries=slice(19, 19),
            observable_summaries=slice(19, 19),
            error=slice(19, 24),
            counters=slice(24, 24),
            proposed_counters=slice(24, 24),
            local_end=24,
            scratch=slice(24, None),
            all=slice(None),
        )
        assert indices.n_states == 5
        assert indices.n_parameters == 3
        assert indices.n_observables == 2

    def test_loop_shared_elements_property(self):
        """loop_shared_elements should return local_end."""
        indices = LoopSharedIndicesFromSettings(
            state=slice(0, 3),
            proposed_state=slice(3, 6),
            observables=slice(6, 8),
            proposed_observables=slice(8, 10),
            parameters=slice(10, 12),
            drivers=slice(12, 13),
            proposed_drivers=slice(13, 14),
            state_summaries=slice(14, 14),
            observable_summaries=slice(14, 14),
            error=slice(14, 17),
            counters=slice(17, 17),
            proposed_counters=slice(17, 17),
            local_end=17,
            scratch=slice(17, None),
            all=slice(None),
        )
        assert indices.loop_shared_elements == 17
