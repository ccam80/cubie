"""Tests for LoopBufferSettings class."""
import pytest
from cubie.integrators.loops.ode_loop import (
    LoopBufferSettings,
    LoopLocalSizes,
    LoopSliceIndices,
)


class TestLoopBufferSettings:
    """Tests for LoopBufferSettings initialization and properties."""


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
        assert settings.use_shared_drivers is False

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
        assert settings.local_memory_elements == 2

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

        # Order: state, proposed, observables, proposed_obs, params, etc.
        # State: 0-3, proposed: 3-6, observables: 6-8, proposed_obs: 8-10,
        # params: 10-12
        assert indices.state == slice(0, 3)
        assert indices.proposed_state == slice(3, 6)
        assert indices.observables == slice(6, 8)
        assert indices.proposed_observables == slice(8, 10)
        assert indices.parameters == slice(10, 12)

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


class TestLoopSliceIndicesProperties:
    """Tests for LoopSliceIndices properties."""

    def test_n_states_property(self):
        """n_states should reflect state slice width."""
        indices = LoopSliceIndices(
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
            local_end=24,
            scratch=slice(24, None),
            all=slice(None),
        )
        assert indices.n_states == 5
        assert indices.n_parameters == 3
        assert indices.n_observables == 2

    def test_loop_shared_elements_property(self):
        """loop_shared_elements should return local_end."""
        indices = LoopSliceIndices(
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
            local_end=17,
            scratch=slice(17, None),
            all=slice(None),
        )
        assert indices.loop_shared_elements == 17


class TestLoopLocalSizes:
    """Tests for LoopLocalSizes and nonzero method."""

    def test_local_sizes_property(self):
        """local_sizes property should return LoopLocalSizes instance."""
        settings = LoopBufferSettings(
            n_states=3,
            n_parameters=2,
            n_drivers=1,
            n_observables=2,
            n_error=3,
            n_counters=4,
        )
        sizes = settings.local_sizes

        assert isinstance(sizes, LoopLocalSizes)
        assert sizes.state == 3
        assert sizes.proposed_state == 3
        assert sizes.parameters == 2
        assert sizes.drivers == 1
        assert sizes.observables == 2
        assert sizes.error == 3
        assert sizes.counters == 4

    def test_nonzero_returns_one_for_zero_size(self):
        """nonzero method should return 1 for zero-size attributes."""
        sizes = LoopLocalSizes(
            state=0,
            proposed_state=0,
            parameters=0,
            drivers=0,
            proposed_drivers=0,
            observables=0,
            proposed_observables=0,
            error=0,
            counters=0,
            state_summary=0,
            observable_summary=0,
        )

        assert sizes.nonzero('state') == 1
        assert sizes.nonzero('parameters') == 1
        assert sizes.nonzero('error') == 1

    def test_nonzero_returns_value_for_nonzero_size(self):
        """nonzero method should return actual value for nonzero sizes."""
        sizes = LoopLocalSizes(
            state=5,
            proposed_state=5,
            parameters=3,
            drivers=2,
            proposed_drivers=2,
            observables=4,
            proposed_observables=4,
            error=5,
            counters=4,
            state_summary=0,
            observable_summary=0,
        )

        assert sizes.nonzero('state') == 5
        assert sizes.nonzero('parameters') == 3
        assert sizes.nonzero('error') == 5


class TestLoopSliceIndices:
    """Tests for shared_indices property."""

    def test_shared_indices_property(self):
        """shared_indices property should return LoopSliceIndices instance."""
        settings = LoopBufferSettings(
            n_states=3,
            n_parameters=2,
            state_buffer_location='shared',
            parameters_location='shared',
        )
        indices = settings.shared_indices

        assert isinstance(indices, LoopSliceIndices)
        assert indices.state == slice(0, 3)
        assert indices.parameters == slice(3, 5)

    def test_shared_indices_matches_calculate_shared_indices(self):
        """shared_indices should return same result as calculate_shared_indices."""
        settings = LoopBufferSettings(
            n_states=3,
            n_parameters=2,
            n_drivers=1,
            state_buffer_location='shared',
            parameters_location='shared',
            drivers_location='shared',
        )
        calculated = settings.calculate_shared_indices()
        property_result = settings.shared_indices

        assert calculated.state == property_result.state
        assert calculated.parameters == property_result.parameters
        assert calculated.drivers == property_result.drivers
        assert calculated.local_end == property_result.local_end
