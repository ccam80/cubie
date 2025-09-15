"""Reusable tester for single-step integration algorithms."""

from math import log
from typing import Any, Iterable

import numpy as np
import pytest
from numba import cuda
from numba.cuda.simulator.kernel import FakeCUDAKernel
from numba import from_dtype

setattr(FakeCUDAKernel, "targetoptions", {"device": True})

from cubie.odesystems.symbolic.symbolicODE import create_ODE_system


class StepAlgorithmTester:
    """Base class for single-step algorithm tests.

    Subclasses should override ``algorithm_class`` and ``step_kwargs`` to
    provide the algorithm under test. Tests can be parametrised with
    different systems, step sizes and tolerances through fixtures.
    """

    # ------------------------------------------------------------------
    # System fixtures
    # ------------------------------------------------------------------
    @pytest.fixture(scope="function")
    def linearsystem(self, precision: np.dtype):
        """Two-state linear system with parameters and constants."""

        dxdt = [
            "dx0 = -a*x0 + c0",
            "dx1 = -b*x1 + c1",
        ]
        params = {"a": 1.0, "b": 0.5}
        consts = {"c0": 1.0, "c1": -0.5}
        system = create_ODE_system(
            dxdt,
            states=["x0", "x1"],
            parameters=params,
            constants=consts,
            precision=precision,
        )

        return system

    @pytest.fixture(scope="function")
    def nonlinearsystem(self, precision: np.dtype):
        """Nonlinear two-state system using symbolic parameters."""

        dxdt = [
            "dx0 = -a*x0**2",
            "dx1 = b*x0*x1**2",
        ]
        params = {"a": 1.0}
        consts = {"b": 2.0}
        system = create_ODE_system(
            dxdt,
            states=["x0", "x1"],
            parameters=params,
            constants=consts,
            precision=precision,
        )

        return system

    @pytest.fixture(scope="function")
    def system_type(self, request: Any) -> str:
        return request.param if hasattr(request, "param") else "linear"

    @pytest.fixture(scope="function")
    def system(self, system_type: str, linearsystem, nonlinearsystem):
        return nonlinearsystem if system_type == "nonlinear" else linearsystem

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
    def x0_override(self, request: Any) -> Iterable[float]:
        return request.param if hasattr(request, "param") else [2.0, 0.5]

    @pytest.fixture(scope="function")
    def x0(self, x0_override: Iterable[float]) -> np.ndarray:
        return np.array(list(x0_override))

    @pytest.fixture(scope="function")
    def tolerances_override(self, request: Any) -> dict:
        return request.param if hasattr(request, "param") else {}

    @pytest.fixture(scope="function")
    def tolerances(self, tolerances_override: dict, system_type: str) -> dict:
        defaults = {
            "linear": {"rtol": 1e-2, "atol": 1e-2},
            "nonlinear": {"rtol": 1e-1, "atol": 1e-1},
        }
        tol = defaults[system_type].copy()
        tol.update(tolerances_override)
        return tol

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

    # ------------------------------------------------------------------
    # Expected solution
    # ------------------------------------------------------------------
    @pytest.fixture(scope="function")
    def expected_solution(
        self, system, system_type: str, x0: np.ndarray, step_size: float
    ) -> np.ndarray:
        if system_type == "linear":
            params = system.parameters.values_dict
            consts = system.constants.values_dict
            a, b = params["a"], params["b"]
            c0, c1 = consts["c0"], consts["c1"]
            x_0 = (x0[0] - c0 / a) * np.exp(-a * step_size) + c0 / a
            x_1 = (x0[1] - c1 / b) * np.exp(-b * step_size) + c1 / b
            return np.array([x_0, x_1])
        params = system.parameters.values_dict
        consts = system.constants.values_dict
        a, b = params["a"], consts["b"]
        x_0 = x0[0] / (1 + a * x0[0] * step_size)
        x_1 = 1.0 / (
            1 / x0[1] - (b / a) * log(1 + a * x0[0] * step_size)
        )
        return np.array([x_0, x_1])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_step(
        self,
        step_obj,
        precision: np.dtype,
        x0: np.ndarray,
        step_size: float,
        system,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute a single step on the device."""

        step_obj.build()
        step_fn = step_obj.step_function
        n = step_obj.compile_settings.n
        p = system.sizes.parameters
        d = system.sizes.drivers
        o = max(1, system.sizes.observables)

        state = np.array(x0[:n], dtype=precision)
        params = np.array(
            list(system.parameters.values_dict.values()), dtype=precision
        )
        drivers = np.zeros(d, dtype=precision)
        observables = np.zeros(o, dtype=precision)
        dxdt_buffer = np.zeros(n, dtype=precision)
        local_req = max(1, step_obj.persistent_local_required)
        sharedmem = max(1, step_obj.shared_memory_required)
        flag = np.ones(1, dtype=np.int32) * -1

        numba_precision = from_dtype(precision)

        @cuda.jit
        def kernel(state, params, drivers, observables, dxdt_buffer, flag):
            shared = cuda.shared.array(0, dtype=numba_precision)
            local = cuda.local.array(local_req, dtype=numba_precision)
            dt = cuda.local.array(1, dtype=numba_precision)
            dt[0] = step_size
            error = cuda.local.array(n, dtype=numba_precision)
            flag[0] = step_fn(
                state,
                params,
                drivers,
                observables,
                dxdt_buffer,
                error,
                dt,
                shared,
                local,
            )

        kernel[1, 1, 0, sharedmem](
            state, params, drivers, observables, dxdt_buffer, flag
        )
        return state, flag

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_build(self, step_obj) -> None:
        step_obj.build()
        assert step_obj.step_function is not None

    def test_step(
        self,
        step_obj,
        precision: np.dtype,
        x0: np.ndarray,
        expected_solution: np.ndarray,
        step_size: float,
        tolerances: dict,
        system,
    ) -> None:
        state, flag = self._run_step(
            step_obj, precision, x0, step_size, system
        )
        assert flag[0] == 0
        assert np.allclose(state, expected_solution, **tolerances)

