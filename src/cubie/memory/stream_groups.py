"""Stream group management for coordinating CUDA work queues.

This module groups host-side objects and identifiers under shared CUDA streams
so that related kernels, transfers, and memory operations execute together. The
default group always exists and receives a fresh stream after CUDA context
resets.
"""

from typing import Any, Optional, Union
from numba import cuda
import attrs
import attrs.validators as val

from cubie.cuda_simsafe import Stream


@attrs.define
class StreamGroups:
    """Container for organizing instances into groups with shared streams.

    Parameters
    ----------
    groups
        Dictionary mapping group names to lists of instance identifiers. When
        omitted, an empty mapping is created and populated with the "default"
        group.
    streams
        Dictionary mapping group names to CUDA streams. When omitted, each
        group receives a new stream from :func:`numba.cuda.stream` and the
        "default" group is backed by :func:`numba.cuda.default_stream`.

    Attributes
    ----------
    groups
        Dictionary mapping group names to lists of instance identifiers.
    streams
        Dictionary mapping group names to CUDA streams.

    Notes
    -----
    Each group has an associated CUDA stream that all instances in the group
    share for coordinated operations. The "default" group is created
    automatically.
    """

    groups: Optional[dict[str, list[int]]] = attrs.field(
        default=attrs.Factory(dict),
        validator=val.optional(val.instance_of(dict)),
    )
    streams: dict[str, Union[Stream, int]] = attrs.field(
        default=attrs.Factory(dict), validator=val.instance_of(dict)
    )

    def __attrs_post_init__(self) -> None:
        """
        Initialize default group and stream if not provided.

        Returns
        -------
        None
            ``None``.
        """
        if self.groups is None:
            self.groups = {"default": []}
        if self.streams is None:
            self.streams = {"default": cuda.default_stream()}

    def add_instance(self, instance: Any, group: str) -> None:
        """
        Add an instance to a stream group.

        Parameters
        ----------
        instance
            Host object or integer identifier to register with a stream group.
        group
            Name of the destination group.

        Returns
        -------
        None
            ``None``.

        Raises
        ------
        ValueError
            If the instance is already in a stream group.

        Notes
        -----
        If the group does not exist, it is created with a new CUDA stream.
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

    def get_group(self, instance: Any) -> str:
        """
        Get the stream group associated with an instance.

        Parameters
        ----------
        instance
            Host object or integer identifier whose group is requested.

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

    def get_stream(self, instance: Any) -> Union[Stream, int]:
        """
        Get the CUDA stream associated with an instance.

        Parameters
        ----------
        instance
            Host object or integer identifier whose stream is requested.

        Returns
        -------
        Stream or int
            CUDA stream associated with the instance's group.
        """
        return self.streams[self.get_group(instance)]

    def get_instances_in_group(self, group: str) -> list[int]:
        """
        Get all instances in a stream group.

        Parameters
        ----------
        group
            Name of the group to inspect.

        Returns
        -------
        list of int
            List of instance identifiers associated with the group, or an empty
            list when the group has not been created.
        """
        if group not in self.groups:
            return []

        return self.groups[group]

    def change_group(self, instance: Any, new_group: str) -> None:
        """
        Move an instance to another stream group.

        Parameters
        ----------
        instance
            Host object or integer identifier to move.
        new_group
            Name of the destination group.

        Returns
        -------
        None
            ``None``.

        Notes
        -----
        If the new group does not exist, it is created with a new CUDA stream.
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

    def reinit_streams(self) -> None:
        """
        Reinitialize all streams after a context reset.

        Returns
        -------
        None
            ``None``.

        Notes
        -----
        Called after CUDA context reset to create fresh streams for all groups.
        """
        for group in self.streams:
            self.streams[group] = cuda.stream()
