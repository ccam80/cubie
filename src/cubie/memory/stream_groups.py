"""
Stream group management for organizing instances and CUDA streams.

This module provides functionality to group instances together and assign them
to shared CUDA streams for coordinated memory operations and kernel execution.
"""

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
    """
    Container for organizing instances into groups with shared CUDA streams.

    Parameters
    ----------
    groups : dict of str to list of int, optional
        Dictionary mapping group names to lists of instance IDs.
    streams : dict of str to Stream or int, optional
        Dictionary mapping group names to CUDA streams.

    Attributes
    ----------
    groups : dict of str to list of int
        Dictionary mapping group names to lists of instance IDs.
    streams : dict of str to Stream or int
        Dictionary mapping group names to CUDA streams.

    Notes
    -----
    Each group has an associated CUDA stream that all instances in the group
    share for coordinated operations. The 'default' group is created
    automatically.
    """

    groups: Optional[dict[str, list[int]]] = attrs.field(
        default=attrs.Factory(dict),
        validator=val.optional(val.instance_of(dict)),
    )
    streams: dict[str, Union[Stream, int]] = attrs.field(
        default=attrs.Factory(dict), validator=val.instance_of(dict)
    )

    def __attrs_post_init__(self):
        """Initialize default group and stream if not provided."""
        if self.groups is None:
            self.groups = {"default": []}
        if self.streams is None:
            self.streams = {"default": cuda.default_stream()}

    def add_instance(self, instance, group):
        """
        Add an instance to a stream group.

        Parameters
        ----------
        instance : object or int
            The instance to add to the group, or its ID.
        group : str
            Name of the group to add the instance to.

        Raises
        ------
        ValueError
            If the instance is already in a stream group.

        Notes
        -----
        If the group doesn't exist, it will be created with a new CUDA
        stream.
        """
        if isinstance(instance, int):
            instance_id = instance
        else:
            instance_id = id(instance)
        if any(instance_id in group for group in self.groups.values()):
            raise ValueError(
                "Instance already in a stream group. Call change_group instead"
            )
        if group not in self.groups:
            self.groups[group] = []
            self.streams[group] = cuda.stream()
        self.groups[group].append(instance_id)

    def get_group(self, instance):
        """
        Get the stream group associated with an instance.

        Parameters
        ----------
        instance : object or int
            The instance to find the group for, or its ID.

        Returns
        -------
        str
            Name of the group containing the instance.

        Raises
        ------
        ValueError
            If the instance is not in any stream groups.
        """
        if isinstance(instance, int):
            instance_id = instance
        else:
            instance_id = id(instance)
        try:
            return [
                key
                for key, value in self.groups.items()
                if instance_id in value
            ][0]
        except IndexError:
            raise ValueError("Instance not in any stream groups")

    def get_stream(self, instance):
        """
        Get the CUDA stream associated with an instance.

        Parameters
        ----------
        instance : object or int
            The instance to get the stream for, or its ID.

        Returns
        -------
        Stream or int
            CUDA stream associated with the instance's group.
        """
        return self.streams[self.get_group(instance)]

    def get_instances_in_group(self, group):
        """
        Get all instances in a stream group.

        Parameters
        ----------
        group : str
            Name of the group to get instances for.

        Returns
        -------
        list of int
            List of instance IDs in the group, or empty list if group
            doesn't exist.
        """
        if group not in self.groups:
            return []

        return self.groups[group]

    def change_group(self, instance, new_group):
        """
        Move an instance to another stream group.

        Parameters
        ----------
        instance : object or int
            The instance to move, or its ID.
        new_group : str
            Name of the group to move the instance to.

        Notes
        -----
        If the new group doesn't exist, it will be created with a new CUDA
        stream.
        """
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
        """
        Reinitialize all streams after a context reset.

        Notes
        -----
        Called after CUDA context reset to create fresh streams for all groups.
        """
        for group in self.streams:
            self.streams[group] = cuda.stream()
