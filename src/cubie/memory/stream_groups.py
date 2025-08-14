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
        if isinstance(instance, int):
            instance_id = instance
        else:
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
        if isinstance(instance, int):
            instance_id = instance
        else:
            instance_id = id(instance)
        try:
            return [key for key, value in self.groups.items()
                    if instance_id in value][0]
        except IndexError:
            raise ValueError("Instance not in any stream groups")

    def get_stream(self, instance):
        """Gets the stream associated with an instance"""
        return self.streams[self.get_group(instance)]

    def get_instances_in_group(self, group):
        """Get all instances in a stream group"""
        if group not in self.groups:
            return []

        return self.groups[group]

    def change_group(self, instance, new_group):
        """Move instance to another stream group"""
        if isinstance(instance, int):
            instance_id = instance
        else:
            instance_id = id(instance)

        # Remove from current group
        current_group = self.get_group(instance)
        self.groups[current_group].remove(instance_id)

        # Add to new group
        if new_group not in self.groups:
            self.groups[new_group] = []
            self.streams[new_group] = cuda.stream()
        self.groups[new_group].append(instance_id)

    def reinit_streams(self):
        """Reinitialize all streams (called after context reset)"""
        for group in self.streams:
            self.streams[group] = cuda.stream()
