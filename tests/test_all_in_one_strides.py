"""Tests for stride ordering utilities in all_in_one.py."""

import numpy as np

from tests.all_in_one import get_strides, DEFAULT_STRIDE_ORDER


class TestGetStrides:
    """Test get_strides stride calculation function."""

    def test_default_stride_order_constant(self):
        """DEFAULT_STRIDE_ORDER matches memory manager default."""
        # Memory manager uses ("time", "run", "variable") as default
        # Represented as indices: 0=time, 1=run, 2=variable
        assert DEFAULT_STRIDE_ORDER == (0, 1, 2)

    def test_returns_none_for_2d_arrays(self):
        """2D arrays should not get custom strides."""
        result = get_strides((100, 10), np.float32, (0, 1))
        assert result is None

    def test_returns_none_for_1d_arrays(self):
        """1D arrays should not get custom strides."""
        result = get_strides((100,), np.int32, (0,))
        assert result is None

    def test_returns_none_when_orders_match(self):
        """Same native and desired order returns None."""
        # state_output case: native order matches desired order
        result = get_strides(
            (100, 1024, 4),
            np.float32,
            array_native_order=(0, 1, 2),
            desired_order=(0, 1, 2),
        )
        assert result is None

    def test_returns_strides_when_orders_differ(self):
        """Different native and desired order returns custom strides."""
        # iteration_counters case: shape=(runs, samples, counters)
        # native order=(run, time, var) = (1, 0, 2)
        # desired order=(time, run, var) = (0, 1, 2)
        result = get_strides(
            (1024, 100, 4),
            np.int32,
            array_native_order=(1, 0, 2),
            desired_order=(0, 1, 2),
        )
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_stride_calculation_iteration_counters(self):
        """Verify stride calculation for iteration_counters layout."""
        # shape=(runs=1024, samples=100, counters=4)
        # native order=(run=1, time=0, var=2)
        # desired order=(time=0, run=1, var=2)
        #
        # itemsize = 4 bytes (int32)
        # desired reversed = (2, 1, 0)
        # variable (idx=2): stride = 4
        # run (idx=1): stride = 4 * 4 = 16 (4 counters)
        # time (idx=0): stride = 16 * 1024 = 16384 (1024 runs)
        #
        # native order (1, 0, 2) -> dims lookup:
        # idx=1 (run) -> size=1024, stride=16
        # idx=0 (time) -> size=100, stride=16384
        # idx=2 (var) -> size=4, stride=4
        # Result: (16, 16384, 4)

        result = get_strides(
            (1024, 100, 4),
            np.int32,
            array_native_order=(1, 0, 2),
            desired_order=(0, 1, 2),
        )

        assert result == (16, 16384, 4)

    def test_stride_calculation_float64(self):
        """Verify stride calculation with float64 dtype."""
        # shape=(10, 20, 5)
        # native order=(0, 1, 2), desired order=(1, 0, 2)
        # itemsize = 8 bytes (float64)
        #
        # Swap time and run in memory layout:
        # desired reversed = (2, 0, 1)
        # variable (idx=2): stride = 8
        # time (idx=0): stride = 8 * 5 = 40
        # run (idx=1): stride = 40 * 10 = 400
        #
        # native order (0, 1, 2) -> result:
        # idx=0: stride=40
        # idx=1: stride=400
        # idx=2: stride=8
        # Result: (40, 400, 8)

        result = get_strides(
            (10, 20, 5),
            np.float64,
            array_native_order=(0, 1, 2),
            desired_order=(1, 0, 2),
        )

        assert result == (40, 400, 8)

    def test_uses_default_desired_order(self):
        """Default desired_order parameter uses DEFAULT_STRIDE_ORDER."""
        # Test with only array_native_order provided
        result1 = get_strides(
            (100, 50, 10),
            np.float32,
            array_native_order=(1, 0, 2),
        )
        result2 = get_strides(
            (100, 50, 10),
            np.float32,
            array_native_order=(1, 0, 2),
            desired_order=DEFAULT_STRIDE_ORDER,
        )
        assert result1 == result2

    def test_itemsize_affects_strides(self):
        """Different dtypes produce different strides."""
        shape = (10, 20, 5)
        native_order = (0, 1, 2)
        desired_order = (2, 1, 0)

        strides_f32 = get_strides(shape, np.float32, native_order, desired_order)
        strides_f64 = get_strides(shape, np.float64, native_order, desired_order)

        # float32 has 4-byte itemsize, float64 has 8-byte
        assert strides_f32 is not None
        assert strides_f64 is not None
        # Smallest stride should be itemsize
        assert min(strides_f32) == 4
        assert min(strides_f64) == 8


class TestStrideOrderIntegration:
    """Test stride ordering matches production batch solver patterns."""

    def test_state_output_order(self):
        """state_output uses native order (time, run, variable)."""
        # In production: stride_order=("time", "run", "variable")
        # shape=(n_samples, n_runs, n_states)
        result = get_strides(
            (100, 1024, 4),
            np.float32,
            array_native_order=(0, 1, 2),  # (time, run, var)
            desired_order=(0, 1, 2),       # same
        )
        # Same order -> no custom strides needed
        assert result is None

    def test_iteration_counters_order(self):
        """iteration_counters uses native order (run, time, variable)."""
        # In production: stride_order=("run", "time", "variable")
        # shape=(n_runs, n_samples, n_counters)
        # This differs from default (time, run, var) so needs custom strides
        result = get_strides(
            (1024, 100, 4),
            np.int32,
            array_native_order=(1, 0, 2),  # (run, time, var)
            desired_order=(0, 1, 2),       # (time, run, var)
        )
        assert result is not None
