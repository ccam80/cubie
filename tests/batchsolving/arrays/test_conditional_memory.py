"""Tests for conditional memory type selection based on chunking."""

from numpy import float32 as np_float32

import attrs

from cubie.batchsolving.arrays.BaseArrayManager import (
    BaseArrayManager,
    ArrayContainer,
    ManagedArray,
)
from cubie.batchsolving.arrays.BatchOutputArrays import (
    OutputArrays,
    OutputArrayContainer,
)
from cubie.batchsolving.arrays.BatchInputArrays import InputArrayContainer
from cubie.memory.array_requests import ArrayResponse


@attrs.define
class ConcreteArrayManager(BaseArrayManager):
    """Concrete implementation of BaseArrayManager for testing."""

    def finalise(self, chunk_index):
        return chunk_index

    def initialise(self, chunk_index):
        return chunk_index

    def update(self):
        return


@attrs.define(slots=False)
class TestArrayContainer(ArrayContainer):
    """Simple test container with a single managed array."""

    state: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("time", "variable", "run"),
            default_shape=(10, 3, 100),
            memory_type="pinned",
        )
    )


class TestIsChunkedProperty:
    """Test the is_chunked property on BaseArrayManager."""

    def test_is_chunked_false_when_single_chunk(self):
        """Verify is_chunked returns False when chunks <= 1."""
        manager = ConcreteArrayManager(
            host=TestArrayContainer(),
            device=TestArrayContainer(),
        )
        # Default is 0 chunks
        assert manager._chunks == 0
        assert manager.is_chunked is False

        # Set to 1 chunk
        manager._chunks = 1
        assert manager.is_chunked is False

    def test_is_chunked_true_when_multiple_chunks(self):
        """Verify is_chunked returns True when chunks > 1."""
        manager = ConcreteArrayManager(
            host=TestArrayContainer(),
            device=TestArrayContainer(),
        )
        manager._chunks = 2
        assert manager.is_chunked is True

        manager._chunks = 10
        assert manager.is_chunked is True


class TestGetHostMemoryType:
    """Test the get_host_memory_type method on BaseArrayManager."""

    def test_get_host_memory_type_returns_pinned_non_chunked(self):
        """Verify non-chunked arrays use pinned memory."""
        manager = ConcreteArrayManager(
            host=TestArrayContainer(),
            device=TestArrayContainer(),
        )
        memory_type = manager.get_host_memory_type(is_chunked=False)
        assert memory_type == "pinned"

    def test_get_host_memory_type_returns_host_chunked(self):
        """Verify chunked arrays use regular numpy (host) memory."""
        manager = ConcreteArrayManager(
            host=TestArrayContainer(),
            device=TestArrayContainer(),
        )
        memory_type = manager.get_host_memory_type(is_chunked=True)
        assert memory_type == "host"


class TestHostFactoryMemoryType:
    """Test host_factory methods accept memory_type parameter."""

    def test_output_container_host_factory_default_pinned(self):
        """Verify OutputArrayContainer.host_factory defaults to pinned."""
        container = OutputArrayContainer.host_factory()
        for _, slot in container.iter_managed_arrays():
            assert slot.memory_type == "pinned"

    def test_output_container_host_factory_accepts_host(self):
        """Verify OutputArrayContainer.host_factory accepts host type."""
        container = OutputArrayContainer.host_factory(memory_type="host")
        for _, slot in container.iter_managed_arrays():
            assert slot.memory_type == "host"

    def test_input_container_host_factory_default_pinned(self):
        """Verify InputArrayContainer.host_factory defaults to pinned."""
        container = InputArrayContainer.host_factory()
        for _, slot in container.iter_managed_arrays():
            assert slot.memory_type == "pinned"

    def test_input_container_host_factory_accepts_host(self):
        """Verify InputArrayContainer.host_factory accepts host type."""
        container = InputArrayContainer.host_factory(memory_type="host")
        for _, slot in container.iter_managed_arrays():
            assert slot.memory_type == "host"


class TestOutputArraysConvertToNumpyWhenChunked:
    """Test OutputArrays converts pinned to numpy when chunked."""

    def test_output_arrays_converts_to_numpy_when_chunked(self):
        """Verify OutputArrays converts pinned to numpy in chunked mode."""
        output_arrays = OutputArrays()

        # Initially arrays are pinned
        for name, slot in output_arrays.host.iter_managed_arrays():
            assert slot.memory_type == "pinned"

        # Set up larger shapes so chunking produces different shapes
        chunks = 3
        for name, slot in output_arrays.device.iter_managed_arrays():
            if slot.is_chunked and "run" in slot.stride_order:
                run_idx = slot.stride_order.index("run")
                new_shape = list(slot.shape)
                new_shape[run_idx] = 30  # 30 runs, divisible by 3 chunks
                slot.shape = tuple(new_shape)

        # Prepare chunked_shapes for each managed array
        # Divide the "run" axis by chunk count to simulate chunking
        chunked_shapes = {}
        for name, slot in output_arrays.device.iter_managed_arrays():
            full_shape = slot.shape
            if slot.is_chunked and "run" in slot.stride_order:
                run_idx = slot.stride_order.index("run")
                chunked = list(full_shape)
                chunked[run_idx] = full_shape[run_idx] // chunks
                chunked_shapes[name] = tuple(chunked)

        # Simulate allocation response with multiple chunks
        response = ArrayResponse(
            arr={},
            chunks=chunks,
            chunk_axis="run",
            chunked_shapes=chunked_shapes,
        )
        output_arrays._on_allocation_complete(response)

        # After chunked allocation, chunked arrays should be host type
        assert output_arrays.is_chunked is True
        for name, slot in output_arrays.host.iter_managed_arrays():
            device_slot = output_arrays.device.get_managed_array(name)
            if device_slot.needs_chunked_transfer:
                assert slot.memory_type == "host"
            else:
                # Non-chunked arrays (like status_codes) stay pinned
                assert slot.memory_type == "pinned"

    def test_output_arrays_stays_pinned_when_not_chunked(self):
        """Verify OutputArrays stays pinned when not chunked."""
        output_arrays = OutputArrays()

        # Initially arrays are pinned
        for name, slot in output_arrays.host.iter_managed_arrays():
            assert slot.memory_type == "pinned"

        # Simulate allocation response with single chunk
        response = ArrayResponse(
            arr={},
            chunks=1,
            chunk_axis="run",
        )
        output_arrays._on_allocation_complete(response)

        # After single-chunk allocation, arrays should stay pinned
        assert output_arrays.is_chunked is False
        for name, slot in output_arrays.host.iter_managed_arrays():
            assert slot.memory_type == "pinned"
