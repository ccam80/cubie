"""Tests for CUDA driver-array interpolation helpers."""

import math
from typing import Tuple

import numpy as np
import pytest
from numba import cuda

from cubie.integrators.driver_array import DriverArray


@pytest.fixture(scope="function")
def quadratic_driver(precision) -> DriverArray:
    """Driver array sampling ``f(t) = t^2`` on unit spacing."""

    times = np.arange(0.0, 6.0, 1.0, dtype=precision)
    values = times**2
    drivers_dict = {"values": values, "time": times}
    driver = DriverArray(
        precision=precision,
        drivers_dict=drivers_dict,
        order=2,
        wrap=False,
    )
    return driver


@pytest.fixture(scope="function")
def cubic_drivers(precision) -> DriverArray:
    """Two driver signals that are exactly represented by cubic polynomials."""

    times = np.linspace(0.0, 5.0, 11, dtype=precision)
    t = times
    drivers_dict = {
        "cubic1": t**3 - 2.0 * t,
        "cubic2": 0.5 * t**3 + 2 * t**2 + t,
        "time": times,
    }
    driver = DriverArray(
        precision=precision,
        drivers_dict=drivers_dict,
        order=3,
        wrap=False,
    )
    # Evaluation times avoid the final endpoint to prevent wrap behaviour.
    return driver


@pytest.fixture(scope="function")
def wrapping_drivers(precision) -> Tuple[DriverArray, DriverArray]:
    """Return clamp and wrap driver arrays sharing identical samples."""

    times = np.linspace(0.0, 4.0, 5, dtype=precision)
    values = np.array([0.0, 1.5, -0.75, 2.25, -3.0], dtype=precision)
    drivers_dict = {"values": values, "time": times}
    clamp = DriverArray(
        precision=precision,
        drivers_dict=drivers_dict,
        order=3,
        wrap=False,
    )
    wrap = DriverArray(
        precision=precision,
        drivers_dict=drivers_dict,
        order=3,
        wrap=True,
    )
    return clamp, wrap


def _cpu_evaluate(
    time: float,
    coefficients: np.ndarray,
    start: float,
    resolution: float,
    wrap: bool,
) -> np.ndarray:
    """Evaluate all driver polynomials on the CPU.

    Parameters
    ----------
    time : float
        Query time to evaluate.
    coefficients : numpy.ndarray
        Segment-major polynomial coefficients.
    start : float
        Start time of the driver samples.
    resolution : float
        Temporal resolution between consecutive samples.
    wrap : bool
        Whether the driver wraps beyond the final segment.

    Returns
    -------
    numpy.ndarray
        Evaluated driver values at ``time``.
    """

    inv_res = 1.0 / resolution
    scaled = (time - start) * inv_res
    if wrap:
        segment_float = math.floor(scaled)
        tau = scaled - segment_float
        segment = int(segment_float) % coefficients.shape[0]
        if segment < 0:
            segment += coefficients.shape[0]
    else:
        if scaled < 0.0 or scaled > float(coefficients.shape[0]):
            return np.zeros(coefficients.shape[1], dtype=coefficients.dtype)
        segment = math.floor(scaled)
        if segment < 0:
            segment = 0
        elif segment >= coefficients.shape[0]:
            segment = coefficients.shape[0] - 1
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


def _run_evaluate(
    device_fn, coefficients: np.ndarray, query_times: np.ndarray
) -> np.ndarray:
    """Execute the device evaluation function across supplied samples.

    Parameters
    ----------
    device_fn : callable
        Device function to execute.
    coefficients : numpy.ndarray
        Segment-major polynomial coefficients.
    query_times : numpy.ndarray
        Time samples to evaluate on the device.

    Returns
    -------
    numpy.ndarray
        Evaluated driver values for each query time.
    """

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

    coefficients = quadratic_driver.coefficients[:, 0, :]
    segment_starts = (np.arange(
            quadratic_driver.num_segments,
            dtype=quadratic_driver.precision)
            * quadratic_driver.dt)
    expected = np.column_stack(
        (
            segment_starts ** 2,
            2.0 * segment_starts,
            np.ones_like(segment_starts),
        )
    )
    np.testing.assert_allclose(
        coefficients,
        expected,
        err_msg=(
            "device coefficients diverged from quadratic reference\n"
            f"device:\n{np.array2string(coefficients)}\n"
            f"expected:\n{np.array2string(expected)}"
        ),
    )


def test_device_interpolation_matches_cpu(cubic_drivers) -> None:
    """Device evaluation must agree with a CPU Horner evaluation."""

    driver = cubic_drivers
    query_times = np.arange(driver.num_samples + 2, dtype=np.int32)
    query_times = query_times.astype(driver.precision) * driver.dt

    coefficients = driver.coefficients
    device_fn = driver.driver_function

    gpu = _run_evaluate(device_fn, coefficients, query_times)

    cpu = np.vstack(
        [
            _cpu_evaluate(
                t,
                coefficients,
                driver.t0,
                driver.dt,
                wrap=False,
            )
            for t in query_times
        ]
    )

    np.testing.assert_allclose(
        gpu,
        cpu,
        err_msg=(
            "device vs cpu Horner evaluation mismatch\n"
            f"gpu:\n{np.array2string(gpu)}\n"
            f"cpu:\n{np.array2string(cpu)}"
        ),
    )


def test_wrap_vs_clamp_evaluation(wrapping_drivers) -> None:
    """Wrapping alters extrapolation compared to clamping semantics."""

    clamp_driver, wrap_driver = wrapping_drivers

    query_times = np.array(
        [-0.5, 0.0, 1.5, 4.0, 4.5, 6.2], dtype=clamp_driver.precision
    )

    clamp_gpu = _run_evaluate(
        clamp_driver.driver_function, clamp_driver.coefficients, query_times
    )
    wrap_gpu = _run_evaluate(
        wrap_driver.driver_function, wrap_driver.coefficients, query_times
    )

    clamp_cpu = np.vstack(
        [
            _cpu_evaluate(
                t,
                clamp_driver.coefficients,
                clamp_driver.t0,
                clamp_driver.dt,
                wrap=False,
            )
            for t in query_times
        ]
    )
    wrap_cpu = np.vstack(
        [
            _cpu_evaluate(
                t,
                wrap_driver.coefficients,
                wrap_driver.t0,
                wrap_driver.dt,
                wrap=True,
            )
            for t in query_times
        ]
    )

    np.testing.assert_allclose(
        clamp_gpu,
        clamp_cpu,
        err_msg=(
            "clamp device evaluation diverged from CPU reference\n"
            f"gpu:\n{np.array2string(clamp_gpu)}\n"
            f"cpu:\n{np.array2string(clamp_cpu)}"
        ),
    )
    np.testing.assert_allclose(
        wrap_gpu,
        wrap_cpu,
        err_msg=(
            "wrap device evaluation diverged from CPU reference\n"
            f"gpu:\n{np.array2string(wrap_gpu)}\n"
            f"cpu:\n{np.array2string(wrap_cpu)}"
        ),
    )
    assert not np.allclose(clamp_gpu, wrap_gpu)


def test_non_wrap_returns_zero_outside_range(quadratic_driver) -> None:
    """Non-wrapping drivers must output zeros beyond sampled times."""

    driver = quadratic_driver
    coefficients = driver.coefficients
    device_fn = driver.driver_function
    dtype = driver.precision
    query_times = np.array(
        [
            driver.t0 - driver.dt,
            driver.t0 + 0.5 * driver.dt,
            driver.t0 + driver.num_segments * driver.dt + driver.dt,
        ],
        dtype=dtype,
    )

    gpu = _run_evaluate(device_fn, coefficients, query_times)
    np.testing.assert_allclose(
        gpu[0],
        0.0,
        err_msg=(
            "leading extrapolation should clamp to zero\n"
            f"gpu:\n{np.array2string(gpu[0])}"
        ),
    )
    np.testing.assert_allclose(
        gpu[2],
        0.0,
        err_msg=(
            "trailing extrapolation should clamp to zero\n"
            f"gpu:\n{np.array2string(gpu[2])}"
        ),
    )
    assert not np.allclose(gpu[1], 0.0)


def test_wrap_repeats_periodically(wrapping_drivers) -> None:
    """Wrapping drivers must repeat their samples beyond the final index."""

    _, wrap_driver = wrapping_drivers
    coefficients = wrap_driver.coefficients
    device_fn = wrap_driver.driver_function
    period = wrap_driver.num_segments * wrap_driver.dt
    dtype = wrap_driver.precision
    query_times = np.array(
        [
            wrap_driver.t0 + 0.25 * period,
            wrap_driver.t0 + 1.25 * period,
            wrap_driver.t0 - 0.75 * period,
        ],
        dtype=dtype,
    )

    gpu = _run_evaluate(device_fn, coefficients, query_times)
    np.testing.assert_allclose(
        gpu[0],
        gpu[1],
        err_msg=(
            "wrap evaluation should repeat forward period\n"
            f"gpu0:\n{np.array2string(gpu[0])}\n"
            f"gpu1:\n{np.array2string(gpu[1])}"
        ),
    )
    np.testing.assert_allclose(
        gpu[0],
        gpu[2],
        err_msg=(
            "wrap evaluation should repeat backward period\n"
            f"gpu0:\n{np.array2string(gpu[0])}\n"
            f"gpu2:\n{np.array2string(gpu[2])}"
        ),
    )


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_interpolation_exact_for_polynomials(order, precision) -> None:
    """Interpolation of polynomials up to ``order`` should be exact."""

    times = np.linspace(0.0, 6.0, 25, dtype=precision)
    coeffs = np.arange(order + 1, dtype=precision) + 1.0
    values = np.zeros_like(times)
    for power, coef in enumerate(coeffs):
        values += coef * times**power
    drivers_dict = {"values": values, "time": times}
    driver = DriverArray(
        precision=precision,
        drivers_dict=drivers_dict,
        order=order,
        wrap=False,
    )
    query = np.linspace(times[0], times[-1], 51, dtype=precision)
    gpu = _run_evaluate(driver.driver_function, driver.coefficients, query)
    reference = np.zeros_like(query)
    for power, coef in enumerate(coeffs):
        reference += coef * query**power
    eps = np.finfo(precision).eps
    tol = 1_000.0 * eps
    np.testing.assert_allclose(
        gpu[:, 0],
        reference,
        rtol=tol,
        atol=tol,
        err_msg=(
            "polynomial interpolation lost exactness\n"
            f"gpu:\n{np.array2string(gpu[:, 0])}\n"
            f"reference:\n{np.array2string(reference)}"
        ),
    )

@pytest.mark.parametrize("bc", ["natural", "periodic"])
def test_cubic_matches_scipy_reference(precision, bc) -> None:
    """Order-three interpolation should match SciPy's cubic spline."""

    scipy = pytest.importorskip("scipy.interpolate")
    rng = np.random.default_rng(1234)
    times = np.linspace(0.0, 2.0, 21, dtype=precision)
    samples = np.sin(2.3 * times) + 0.3 * np.cos(4.1 * times)
    samples += 0.05 * rng.standard_normal(times.size).astype(precision)
    samples[0] = samples[-1]
    drivers_dict = {"drive": samples, "time": times}
    driver = DriverArray(
        precision=precision,
        drivers_dict=drivers_dict,
        order=3,
        wrap=True,
        boundary_condition=bc
    )
    query = np.linspace(times[0], times[-1], 257, dtype=precision)
    gpu = _run_evaluate(driver.driver_function, driver.coefficients, query)
    spline = scipy.CubicSpline(times, samples, bc_type=bc)
    scipy_values = np.asarray(spline(query), dtype=precision)
    np.testing.assert_allclose(
        gpu[:, 0],
        scipy_values,
        rtol=5e-4,
        atol=5e-4,
        err_msg=(
            "cubic interpolation diverged from SciPy reference\n"
            f"gpu:\n{np.array2string(gpu[:, 0])}\n"
            f"scipy:\n{np.array2string(scipy_values)}"
        ),
    )


def _falling_factorial(power: int, derivative: int) -> int:
    """Return the falling factorial ``power! / (power - derivative)!``."""

    result = 1
    for step in range(derivative):
        result *= power - step
    return result


def _evaluate_derivative(
    coefficients: np.ndarray,
    segment: int,
    driver_index: int,
    tau: float,
    derivative: int,
    dt: float,
) -> float:
    """Evaluate a spline derivative at ``tau`` in the supplied segment."""

    total = 0.0
    for power in range(derivative, coefficients.shape[-1]):
        factor = _falling_factorial(power, derivative)
        total += (
            coefficients[segment, driver_index, power]
            * factor
            * (tau ** (power - derivative))
        )
    return total / (dt ** derivative)


def test_natural_boundary_supports_higher_orders(precision) -> None:
    """Natural constraints should zero high derivatives for any order."""

    order = 4
    times = np.linspace(0.0, 3.0, 9, dtype=precision)
    samples = np.sin(times) + 0.25 * times**2
    drivers_dict = {"drive": samples, "time": times}
    driver = DriverArray(
        precision=precision,
        drivers_dict=drivers_dict,
        order=order,
        wrap=False,
        boundary_condition="natural",
    )

    coefficients = driver.coefficients
    dt = driver.dt
    expected_zero = []
    remaining = order - 1
    derivative = 2
    while remaining > 0 and derivative <= order:
        expected_zero.append((0, derivative))
        remaining -= 1
        if remaining == 0:
            break
        expected_zero.append((driver.num_segments - 1, derivative))
        remaining -= 1
        derivative += 1

    for segment, derivative in expected_zero:
        tau = 0.0 if segment == 0 else 1.0
        value = _evaluate_derivative(coefficients, segment, 0, tau, derivative, dt)
        np.testing.assert_allclose(
            value,
            0.0,
            atol=1e-6,
            err_msg=(
                "natural boundary derivative failed to vanish\n"
                f"segment={segment} derivative={derivative} value={value}"
            ),
        )

    gpu = _run_evaluate(driver.driver_function, driver.coefficients, times)
    reference = samples
    np.testing.assert_allclose(
        gpu[:, 0],
        reference,
        rtol=1e-6,
        atol=1e-6,
        err_msg="natural boundary spline failed to reproduce samples",
    )


def test_periodic_boundary_respects_general_order(precision) -> None:
    """Periodic constraints should hold for arbitrary spline order."""

    order = 5
    num_samples = 12
    times = np.linspace(0.0, 2.0 * np.pi, num_samples, dtype=precision)
    values = np.column_stack(
        (
            np.sin(times),
            0.75 * np.cos(2.0 * times),
        )
    )
    values[0] = values[-1]
    drivers_dict = {"s": values[:, 0], "c": values[:, 1], "time": times}
    driver = DriverArray(
        precision=precision,
        drivers_dict=drivers_dict,
        order=order,
        wrap=True,
        boundary_condition="periodic",
    )

    coefficients = driver.coefficients
    dt = driver.dt
    last_segment = driver.num_segments - 1
    for derivative in range(1, order):
        left = _evaluate_derivative(coefficients, 0, 0, 0.0, derivative, dt)
        right = _evaluate_derivative(
            coefficients,
            last_segment,
            0,
            1.0,
            derivative,
            dt,
        )
        np.testing.assert_allclose(
            left,
            right,
            atol=1e-6,
            err_msg=(
                "periodic derivative mismatch for sine driver\n"
                f"derivative={derivative} left={left} right={right}"
            ),
        )

    for derivative in range(1, order):
        left = _evaluate_derivative(coefficients, 0, 1, 0.0, derivative, dt)
        right = _evaluate_derivative(
            coefficients,
            last_segment,
            1,
            1.0,
            derivative,
            dt,
        )
        np.testing.assert_allclose(
            left,
            right,
            atol=1e-6,
            err_msg=(
                "periodic derivative mismatch for cosine driver\n"
                f"derivative={derivative} left={left} right={right}"
            ),
        )

    gpu = _run_evaluate(driver.driver_function, driver.coefficients, times)
    reference = values
    np.testing.assert_allclose(
        gpu,
        reference,
        rtol=1e-6,
        atol=1e-6,
        err_msg="periodic spline failed to reproduce samples",
    )
