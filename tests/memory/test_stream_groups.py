from os import environ
import pytest
from cubie.memory.stream_groups import StreamGroups

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from cubie.cudasim_utils import FakeStream as Stream
else:
    from numba.cuda.cudadrv.driver import Stream


class DummyClass:
    def __init__(self, proportion=None, invalidate_all_hook=None):
        self.proportion = proportion
        self.invalidate_all_hook = invalidate_all_hook

@pytest.fixture(scope="function")
def stream_groups(array_request_settings):
    return StreamGroups()

class TestStreamGroups:
    @pytest.mark.nocudasim
    def test_add_instance(self, stream_groups):
        inst = DummyClass()
        stream_groups.add_instance(inst, "group1")
        assert "group1" in stream_groups.groups
        assert id(inst) in stream_groups.groups["group1"]
        assert isinstance(stream_groups.streams["group1"], Stream)
        with pytest.raises(ValueError):
            stream_groups.add_instance(inst, "group2")

        inst2 = DummyClass()
        stream_groups.add_instance(inst2, "group2")
        assert id(inst2) in stream_groups.groups["group2"]
        assert id(inst) not in stream_groups.groups["group2"]
        assert id(inst2) not in stream_groups.groups["group1"]

    def test_get_group(self, stream_groups):
        """Test that get_group returns the correct group for an instance"""
        inst = DummyClass()
        stream_groups.add_instance(inst, "group1")
        assert stream_groups.get_group(inst) == "group1"
        with pytest.raises(ValueError):
            assert stream_groups.get_group(DummyClass()) is None
        inst1 = DummyClass()
        stream_groups.add_instance(inst1, "group2")
        assert stream_groups.get_group(inst1) == "group2"
        assert stream_groups.get_group(inst) != "group2"

    @pytest.mark.nocudasim
    def test_change_group(self, stream_groups):
        """Test that change_group removes the instance from the old group,
        adds it to the new one, and that the instances stream has changed"""
        inst = DummyClass()
        stream_groups.add_instance(inst, 'group1')
        old_stream = stream_groups.get_stream(inst)
        # move to new group
        stream_groups.change_group(inst, 'group2')
        assert id(inst) not in stream_groups.groups['group1']
        assert id(inst) in stream_groups.groups['group2']
        new_stream = stream_groups.get_stream(inst)
        assert int(new_stream.handle.value) != int(old_stream.handle.value)
        # error when instance not in any group
        with pytest.raises(ValueError):
            stream_groups.change_group(DummyClass(), 'groupX')


    @pytest.mark.nocudasim
    def test_get_stream(self, stream_groups):
        inst = DummyClass()
        stream_groups.add_instance(inst, "group1")
        stream1 = int(stream_groups.get_stream(inst).handle.value)
        stream2 = int(stream_groups.get_stream(inst).handle.value)
        assert stream1 == stream2
        assert stream1 != stream_groups.streams["group1"]

        inst2 = DummyClass()
        stream_groups.add_instance(inst2, "group2")
        stream3 = int(stream_groups.get_stream(inst2).handle.value)
        assert stream3 != stream1
        assert stream3 != stream2
        assert stream3 != stream_groups.streams["group1"]
        assert stream3 != stream_groups.streams["group2"]

    @pytest.mark.nocudasim
    def test_reinit_streams(self, stream_groups):
        """ test that two instances have different streams in different
        groups, then reinit, and check that streams don't match old ones or
        each other."""
        inst1 = DummyClass()
        inst2 = DummyClass()
        stream_groups.add_instance(inst1, 'g1')
        stream_groups.add_instance(inst2, 'g2')
        # ensure different initial streams
        s1_old = stream_groups.get_stream(inst1)
        s2_old = stream_groups.get_stream(inst2)
        assert s1_old != s2_old
        # reinitialize streams
        stream_groups.reinit_streams()
        s1_new = stream_groups.get_stream(inst1)
        s2_new = stream_groups.get_stream(inst2)
        assert s1_new != s1_old
        assert s2_new != s2_old
        assert s1_new != s2_new