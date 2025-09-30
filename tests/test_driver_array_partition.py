"""Tests for CUDA driver-array interpolation helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numba import cuda

from cubie.integrators.driver_array import DriverArray


@pytest.fixture(scope="function")
def quadratic_driver() -> DriverArray:
    """Driver array sampling ``f(t) = t^2`` on unit spacing."""

    times = np.arange(0.0, 6.0, 1.0, dtype=np.float64)
    values = times ** 2
    return DriverArray(values, times, order=2, wrap=False)


@pytest.fixture(scope="function")
def cubic_drivers() -> tuple[DriverArray, np.ndarray]:
    """Two driver signals that are exactly represented by cubic polynomials."""

    times = np.linspace(0.0, 5.0, 11, dtype=np.float64)
    resolution = times[1] - times[0]
    t = times
    drivers = np.column_stack((t ** 3 - 2.0 * t, 0.5 * t ** 3 + t))
    driver = DriverArray(drivers, times, order=3, wrap=False)
    # Evaluation times avoid the final endpoint to prevent wrap behaviour.
    query_times = times[:-1] + 0.37 * resolution
    return driver, query_times


@pytest.fixture(scope="function")
def wrapping_drivers() -> tuple[DriverArray, DriverArray]:
    """Return clamp and wrap driver arrays sharing identical samples."""

    times = np.linspace(0.0, 4.0, 5, dtype=np.float64)
    values = np.array([0.0, 1.5, -0.75, 2.25, -3.0], dtype=np.float64)
    clamp = DriverArray(values, times, order=3, wrap=False)
    wrap = DriverArray(values, times, order=3, wrap=True)
    return clamp, wrap


def _cpu_partition(time: float, start: float, inv_res: float, segments: int) -> int:
    """Clamp time to the closest segment index."""

    scaled = (time - start) * inv_res
    idx = math.floor(scaled)
    if idx < 0:
        return 0
    if idx >= segments:
        return segments - 1
    return idx


def _cpu_evaluate(
    time: float,
    coefficients: np.ndarray,
    start: float,
    resolution: float,
    wrap: bool,
) -> np.ndarray:
    """Evaluate all driver polynomials on the CPU."""

    inv_res = 1.0 / resolution
    scaled = (time - start) * inv_res
    idx = math.floor(scaled)
    if wrap:
        segment = idx % coefficients.shape[0]
        if segment < 0:
            segment += coefficients.shape[0]
    else:
        segment = _cpu_partition(time, start, inv_res, coefficients.shape[0])
    base_time = start + resolution * float(segment)
    tau = (time - base_time) * inv_res
    out = np.empty(coefficients.shape[1], dtype=coefficients.dtype)
    for driver_index in range(coefficients.shape[1]):
        coeffs = coefficients[segment, driver_index]
        acc = 0.0
        for k in range(coeffs.size - 1, -1, -1):
            acc = acc * tau + coeffs[k]
        out[driver_index] = acc
    return out


def _run_evaluate(device_fn, coefficients: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Execute the device evaluation function across supplied samples."""

    n_times = query_times.size
    n_drivers = coefficients.shape[1]
    out_host = np.empty((n_times, n_drivers), dtype=coefficients.dtype)

    @cuda.jit
    def kernel(times, coeffs, out):
        idx = cuda.grid(1)
        if idx < times.size:
            device_fn(times[idx], coeffs, out[idx])

    d_times = cuda.to_device(query_times)
    d_coeffs = cuda.to_device(coefficients)
    d_out = cuda.to_device(out_host)
    threads_per_block = 64
    blocks = (n_times + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](d_times, d_coeffs, d_out)
    d_out.copy_to_host(out_host)
    return out_host


def test_build_coefficients_matches_polynomial(quadratic_driver):
    """Segment-wise coefficients should reproduce the quadratic exactly."""

    cache = quadratic_driver.build()
    coefficients = cache.coefficients[:, 0, :]
    segment_starts = quadratic_driver.time_array[:-1]
    expected = np.column_stack(
        (
            segment_starts ** 2,
            2.0 * segment_starts,
            np.ones_like(segment_starts),
        )
    )
    assert np.allclose(coefficients, expected)


def test_device_interpolation_matches_cpu(cubic_drivers) -> None:
    """Device evaluation must agree with a CPU Horner evaluation."""

    driver, query_times = cubic_drivers
    cache = driver.build()
    coefficients = cache.coefficients
    device_fn = cache.device_function

    gpu = _run_evaluate(device_fn, coefficients, query_times)

    cpu = np.vstack(
        [
            _cpu_evaluate(
                t,
                coefficients,
                driver.time_array[0],
                driver.resolution,
                wrap=False,
            )
            for t in query_times
        ]
    )

    assert np.allclose(gpu, cpu)


def test_wrap_vs_clamp_evaluation(wrapping_drivers) -> None:
    """Wrapping alters extrapolation compared to clamping semantics."""

    clamp_driver, wrap_driver = wrapping_drivers
    clamp_cache = clamp_driver.build()
    wrap_cache = wrap_driver.build()

    query_times = np.array([-0.5, 0.0, 1.5, 4.0, 4.5, 6.2], dtype=np.float64)

    clamp_gpu = _run_evaluate(
        clamp_cache.device_function, clamp_cache.coefficients, query_times
    )
    wrap_gpu = _run_evaluate(
        wrap_cache.device_function, wrap_cache.coefficients, query_times
    )

    clamp_cpu = np.vstack(
        [
            _cpu_evaluate(
                t,
                clamp_cache.coefficients,
                clamp_driver.time_array[0],
                clamp_driver.resolution,
                wrap=False,
            )
            for t in query_times
        ]
    )
    wrap_cpu = np.vstack(
        [
            _cpu_evaluate(
                t,
                wrap_cache.coefficients,
                wrap_driver.time_array[0],
                wrap_driver.resolution,
                wrap=True,
            )
            for t in query_times
        ]
    )

    assert np.allclose(clamp_gpu, clamp_cpu)
    assert np.allclose(wrap_gpu, wrap_cpu)
    assert not np.allclose(clamp_gpu, wrap_gpu)
