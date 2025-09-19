"""Reusable tester for single-step integration algorithms."""

from typing import Any, Iterable

import numpy as np
import pytest
from numba import cuda
from numba import from_dtype
from numba.cuda.simulator.kernel import FakeCUDAKernel

setattr(FakeCUDAKernel, "targetoptions", {"device": True})

from cubie.integrators.algorithms_.backwards_euler import BackwardsEulerStep
from cubie.integrators.algorithms_.backwards_euler_predict_correct import (
    BackwardsEulerPredictCorrectStep,
)
from cubie.integrators.algorithms_.crank_nicolson import CrankNicolsonStep
from cubie.integrators.algorithms_.explicit_euler import ExplicitEulerStep
from tests.integrators.cpu_reference import (
    build_system_evaluator,
    run_reference_stepper,
)


_CPU_REFERENCE: list[tuple[type, tuple[str, bool]]] = [
    (
        BackwardsEulerPredictCorrectStep,
        ("backward_euler_predict_correct", False),
    ),
    (BackwardsEulerStep, ("backward_euler", False)),
    (CrankNicolsonStep, ("crank_nicolson", True)),
    (ExplicitEulerStep, ("explicit_euler", False)),
]


def _resolve_cpu_reference(algorithm_class: type) -> tuple[str, bool]:
    """Return the CPU stepper name and adaptivity for ``algorithm_class``."""

    for registered, info in _CPU_REFERENCE:
        if issubclass(algorithm_class, registered):
            return info
    raise KeyError(
        f"No CPU reference stepper registered for {algorithm_class.__name__}."
    )


class StepAlgorithmTester:
    """Base class for single-step algorithm tests.

    Subclasses should override ``algorithm_class`` and ``step_kwargs`` to
    provide the algorithm under test. Tests can be parametrised with
    different systems, step sizes and tolerances through fixtures.
    """

    # ------------------------------------------------------------------
    # Parameter fixtures
    # ------------------------------------------------------------------
    @pytest.fixture(scope="function")
    def step_size_override(self, request: Any) -> float:
        return request.param if hasattr(request, "param") else 0.1

    @pytest.fixture(scope="function")
    def step_size(self, step_size_override: float) -> float:
        return step_size_override

    @pytest.fixture(scope="function")
    def x0_override(self, request: Any):
        return request.param if hasattr(request, "param") else None

    @pytest.fixture(scope="function")
    def x0(
        self,
        x0_override: Iterable[float] | None,
        system,
        precision: np.dtype,
    ) -> np.ndarray:
        dtype = np.dtype(precision)
        if x0_override is None:
            return system.initial_values.values_array.astype(dtype, copy=True)
        return np.asarray(x0_override, dtype=dtype)

    @pytest.fixture(scope="function")
    def tolerances_override(self, request: Any) -> dict:
        return request.param if hasattr(request, "param") else {}

    @pytest.fixture(scope="function")
    def tolerances(
        self,
        tolerances_override: dict,
        system_override: str,
    ) -> dict:
        defaults = {
            "linear": {"rtol": 1e-2, "atol": 1e-2},
            "nonlinear": {"rtol": 1e-1, "atol": 1e-1},
        }
        base = defaults.get(system_override, defaults["linear"]).copy()
        base.update(tolerances_override)
        return base

    # ------------------------------------------------------------------
    # Algorithm fixtures
    # ------------------------------------------------------------------
    @pytest.fixture(scope="function")
    def algorithm_class(self):  # pragma: no cover - override in subclasses
        raise NotImplementedError

    @pytest.fixture(scope="function")
    def step_kwargs(self, system, precision: np.dtype, step_size: float):
        """Default step kwargs for explicit algorithms."""

        return {
            "dxdt_function": system.dxdt_function,
            "precision": precision,
            "n": system.sizes.states,
            "step_size": step_size,
        }

    @pytest.fixture(scope="function")
    def step_obj(self, algorithm_class, step_kwargs):
        return algorithm_class(**step_kwargs)

    def _run_step(
        self,
        step_obj,
        precision: np.dtype,
        x0: np.ndarray,
        step_size: float,
        system,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Execute a single step on the device."""

        dtype = np.dtype(precision)
        step_obj.build()
        step_fn = step_obj.step_function
        n_states = step_obj.compile_settings.n
        params = np.array(system.parameters.values_array, dtype=dtype, copy=True)
        drivers = np.zeros(system.sizes.drivers, dtype=dtype)
        observables = np.zeros(system.sizes.observables, dtype=dtype)
        state = np.array(x0[:n_states], dtype=dtype, copy=True)
        proposed_state = np.zeros_like(state)
        work_len = max(step_obj.local_scratch_required, n_states)
        work_buffer = np.zeros(work_len, dtype=dtype)
        error = np.zeros(n_states, dtype=dtype)
        flag = np.full(1, -1, dtype=np.int32)
        persistent_len = max(1, step_obj.persistent_local_required)
        shared_elems = max(0, step_obj.shared_memory_required)
        shared_bytes = int(shared_elems * dtype.itemsize)
        threads = max(1, step_obj.threads_per_step)

        numba_precision = from_dtype(dtype)
        dt_value = numba_precision(step_size)

        @cuda.jit
        def kernel(
            state_vec,
            proposed_vec,
            work_vec,
            params_vec,
            drivers_vec,
            observables_vec,
            error_vec,
            flag_vec,
            dt_scalar,
        ):
            shared = cuda.shared.array(0, dtype=numba_precision)
            persistent = cuda.local.array(persistent_len, dtype=numba_precision)
            status = step_fn(
                state_vec,
                proposed_vec,
                work_vec,
                params_vec,
                drivers_vec,
                observables_vec,
                error_vec,
                dt_scalar,
                shared,
                persistent,
            )
            flag_vec[0] = status

        kernel[1, threads, 0, shared_bytes](
            state,
            proposed_state,
            work_buffer,
            params,
            drivers,
            observables,
            error,
            flag,
            dt_value,
        )
        cuda.synchronize()
        return proposed_state, error, int(flag[0])

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_build(self, step_obj, step_size, tolerances) -> None:
        step_obj.build()
        assert step_obj.step_function is not None

    def test_step(
        self,
        algorithm_class,
        step_obj,
        precision: np.dtype,
        x0: np.ndarray,
        step_size: float,
        tolerances: dict,
        system,
    ) -> None:
        dtype = np.dtype(precision)
        state, error, flag = self._run_step(
            step_obj, precision, x0, step_size, system
        )
        cpu_state = np.array(
            x0[: system.sizes.states], dtype=dtype, copy=True
        )
        evaluator = build_system_evaluator(system)
        params = system.parameters.values_array.astype(dtype, copy=True)
        drivers = np.zeros(system.sizes.drivers, dtype=dtype)
        stepper_name, cpu_is_adaptive = _resolve_cpu_reference(algorithm_class)
        cpu_result = run_reference_stepper(
            stepper=stepper_name,
            evaluator=evaluator,
            state=cpu_state,
            params=params,
            drivers_now=drivers,
            drivers_next=drivers,
            dt=float(step_size),
        )

        np.testing.assert_allclose(state, cpu_result.state, **tolerances)
        np.testing.assert_allclose(error, cpu_result.error, **tolerances)

        assert (flag == 0) == cpu_result.converged
        assert step_obj.is_adaptive == cpu_is_adaptive
        if not cpu_is_adaptive and hasattr(
            step_obj.compile_settings, "step_size"
        ):
            assert step_obj.compile_settings.step_size == pytest.approx(
                step_size
            )

