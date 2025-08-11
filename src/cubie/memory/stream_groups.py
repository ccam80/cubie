from os import environ
from typing import Optional, Union
from numba import cuda
import attrs
import attrs.validators as val

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from cubie.cudasim_utils import FakeStream as Stream
else:
    from numba.cuda.cudadrv.driver import Stream

@attrs.define
class StreamGroups:
    """Dictionaries which map instances to groups, and groups to a stream"""
    groups: Optional[dict[str, list[int]]] = attrs.field(
            default=attrs.Factory(dict),
            validator=val.optional(val.instance_of(dict)))
    streams: dict[str, Union[Stream, int]] = attrs.field(
            default=attrs.Factory(dict),
            validator=val.instance_of(dict))

    def __attrs_post_init__(self):
        if self.groups is None:
            self.groups = {'default': []}
        if self.streams is None:
            self.streams = {'default': cuda.default_stream()}

    def add_instance(self, instance, group):
        """Add an instance to a stream group"""
        instance_id = id(instance)
        if any(instance_id in group for group in self.groups.values()):
            raise ValueError("Instance already in a stream group. Call "
                             "change_group instead")
        if group not in self.groups:
            self.groups[group] = []
            self.streams[group] = cuda.stream()
        self.groups[group].append(instance_id)

    def get_group(self, instance):
        """Gets stream group associated with an instance"""
        instance_id = id(instance)
        try:
            return [key for key, value in self.groups.items()
                    if instance_id in value][0]
        except IndexError:
            raise ValueError("Instance not in any stream groups")

    def get_stream(self, instance):
        """Getd the stream associated with an instance"""
        return self.streams[self.get_group(instance)]

    def change_group(self, instance, new_group):
        """Move instance onto another stream group"""
        instance_id = id(instance)
        old_group = self.get_group(instance)
        self.groups[old_group].remove(instance_id)
        self.add_instance(instance, new_group)

    def reinit_streams(self):
        """Get a fresh set of streams if the context has been closed."""
        for group, stream in self.streams.items():
            self.streams[group] = cuda.stream()