"""Runtime configuration settings for numerical integration algorithms.

This module provides the :class:`IntegratorRunSettings` class which manages
timing, tolerance, and controller configuration for ODE integration runs.
It performs light dependency injection by instantiating algorithm step
objects and step-size controllers used by the modular IVP loop.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import attrs
from attrs import setters
import numba
import numpy as np
from numpy import float32

# from cubie.integrators.algorithms.LoopStepConfig import LoopStepConfig


_ALGORITHM_ALIASES: Dict[str, str] = {
    "euler": "explicit_euler",
    "explicit_euler": "explicit_euler",
    "explicit": "explicit_euler",
    "backward_euler": "backwards_euler",
    "backwards_euler": "backwards_euler",
    "crank_nicolson": "crank_nicolson",
}

_CONTROLLER_ALIASES: Dict[str, str] = {
    "fixed": "fixed",
    "i": "i",
    "pi": "pi",
    "pid": "pid",
    "gustafsson": "gustafsson",
}

_KNOWN_ALGORITHM_KEYS: set[str] = {
    "dt",
    "linsolve_tolerance",
    "max_linear_iters",
    "linear_correction_type",
    "nonlinear_tolerance",
    "max_newton_iters",
    "newton_damping",
    "newton_max_backtracks",
    "norm_type",
    "preconditioner_order",
}

_KNOWN_CONTROLLER_KEYS: set[str] = {
    "algorithm_order",
    "atol",
    "dt",
    "dt_max",
    "dt_min",
    "kd",
    "ki",
    "kp",
    "max_gain",
    "min_gain",
    "norm",
    "norm_kwargs",
    "order",
    "rtol",
}


def _normalise_algorithm(name: str) -> str:
    """Return the canonical algorithm identifier."""

    try:
        return _ALGORITHM_ALIASES[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown integrator algorithm '{name}'.") from exc


def _normalise_controller(kind: str) -> str:
    """Return the canonical controller identifier."""

    try:
        return _CONTROLLER_ALIASES[kind.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown step controller '{kind}'.") from exc


@attrs.define
class IntegratorRunSettings:
    """Container for runtime/timing settings grouped for IVP loops.

    Parameters
    ----------
    precision
        Numerical precision used for timing comparisons.
    algorithm
        Name of the integration step algorithm.
    step_controller_kind
        Name of the step-size controller.
    algorithm_parameters
        Keyword arguments forwarded to the algorithm step constructor.
    step_controller_parameters
        Keyword arguments forwarded to the controller constructor.
    dt_min, dt_max
        Minimum and maximum step size targets.
    dt_save, dt_summarise
        Output cadence for saved values and summary statistics.
    atol, rtol
        Error tolerances used by adaptive controllers/algorithms.
    output_types
        Output selections requested by the run.
    """

    precision: type = attrs.field(
        default=float32,
        validator=attrs.validators.in_([np.float32, np.float64, np.float16]),
    )
    algorithm: str = attrs.field(
        default="explicit_euler",
        converter=_normalise_algorithm,
        on_setattr=setters.convert,
    )
    step_controller_kind: str = attrs.field(
        default="fixed",
        converter=_normalise_controller,
        on_setattr=setters.convert,
    )


    def __attrs_post_init__(self) -> None:
        """Validate configuration after initialisation."""

        self._apply_parameter_defaults()
        self.validate_settings()

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def numba_precision(self) -> type:
        """Return the Numba-compatible precision."""

        return numba.from_dtype(self.precision)

    @property
    def dt_min(self) -> float:
        """Return the minimum step size."""

        return float(self._dt_min)

    @dt_min.setter
    def dt_min(self, value: float) -> None:
        self._dt_min = float(value)
        self.step_controller_parameters["dt_min"] = self._dt_min
        if self.step_controller_kind == "fixed":
            self.step_controller_parameters["dt"] = self._dt_min
        if self.algorithm == "explicit_euler":
            self.algorithm_parameters["dt"] = self._dt_min
        else:
            self.algorithm_parameters.pop("dt", None)

    @property
    def dt_max(self) -> float:
        """Return the maximum step size."""

        return float(self._dt_max)

    @dt_max.setter
    def dt_max(self, value: float) -> None:
        self._dt_max = float(value)
        self.step_controller_parameters["dt_max"] = self._dt_max

    @property
    def dt_save(self) -> float:
        """Return the save interval."""

        return float(self._dt_save)

    @dt_save.setter
    def dt_save(self, value: float) -> None:
        self._dt_save = float(value)

    @property
    def dt_summarise(self) -> float:
        """Return the summary interval."""

        return float(self._dt_summarise)

    @dt_summarise.setter
    def dt_summarise(self, value: float) -> None:
        self._dt_summarise = float(value)

    @property
    def atol(self) -> float:
        """Return the absolute tolerance."""

        return float(self._atol)

    @atol.setter
    def atol(self, value: float) -> None:
        self._atol = float(value)
        self.step_controller_parameters["atol"] = self._atol
        self.algorithm_parameters["atol"] = self._atol

    @property
    def rtol(self) -> float:
        """Return the relative tolerance."""

        return float(self._rtol)

    @rtol.setter
    def rtol(self, value: float) -> None:
        self._rtol = float(value)
        self.step_controller_parameters["rtol"] = self._rtol
        self.algorithm_parameters["rtol"] = self._rtol

    # @property
    # def loop_step_config(self) -> LoopStepConfig:
    #     """Return the step-size configuration for the integration loop."""
    #
    #     return LoopStepConfig(
    #         dt_min=self.dt_min,
    #         dt_max=self.dt_max,
    #         dt_save=self.dt_save,
    #         dt_summarise=self.dt_summarise,
    #         atol=self.atol,
    #         rtol=self.rtol,
    #     )

    # ------------------------------------------------------------------
    # Helper construction
    # ------------------------------------------------------------------
    def create_step_controller(
        self, precision: type, n_states: int
    ) -> object:
        """Instantiate the configured step-size controller."""

        from cubie.integrators.step_control.adaptive_I_controller import (
            AdaptiveIController,
        )
        from cubie.integrators.step_control.adaptive_PI_controller import (
            AdaptivePIController,
        )
        from cubie.integrators.step_control.adaptive_PID_controller import (
            AdaptivePIDController,
        )
        from cubie.integrators.step_control.fixed_step_controller import (
            FixedStepController,
        )
        from cubie.integrators.step_control.gustafsson_controller import (
            GustafssonController,
        )

        params = dict(self.step_controller_parameters)
        dt_min = float(params.get("dt_min", self._dt_min))
        dt_max = float(params.get("dt_max", self._dt_max))
        atol = float(params.get("atol", self._atol))
        rtol = float(params.get("rtol", self._rtol))
        order = int(params.get("algorithm_order", params.get("order", 1)))
        min_gain = params.get("min_gain", 0.2)
        max_gain = params.get("max_gain", 5.0)
        norm = params.get("norm", "l2")
        norm_kwargs = params.get("norm_kwargs")

        kind = self.step_controller_kind
        if kind == "fixed":
            dt = float(params.get("dt", dt_min))
            return FixedStepController(precision, dt)
        if kind == "i":
            return AdaptiveIController(
                precision,
                dt_min=dt_min,
                dt_max=dt_max,
                atol=atol,
                rtol=rtol,
                algorithm_order=order,
                n=n_states,
                min_gain=min_gain,
                max_gain=max_gain,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
        if kind == "pi":
            return AdaptivePIController(
                precision,
                dt_min=dt_min,
                dt_max=dt_max,
                atol=atol,
                rtol=rtol,
                algorithm_order=order,
                n=n_states,
                kp=params.get("kp", 0.7),
                ki=params.get("ki", 0.4),
                min_gain=min_gain,
                max_gain=max_gain,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
        if kind == "pid":
            return AdaptivePIDController(
                precision,
                dt_min=dt_min,
                dt_max=dt_max,
                atol=atol,
                rtol=rtol,
                algorithm_order=order,
                n=n_states,
                kp=params.get("kp", 0.7),
                ki=params.get("ki", 0.4),
                kd=params.get("kd", 0.2),
                min_gain=min_gain,
                max_gain=max_gain,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
        return GustafssonController(
            precision,
            dt_min=dt_min,
            dt_max=dt_max,
            atol=atol,
            rtol=rtol,
            algorithm_order=order,
            n=n_states,
            min_gain=min_gain,
            max_gain=max_gain,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    def create_step_object(self, system) -> object:
        """Instantiate the configured algorithm step object."""

        from cubie.integrators.algorithms import (
            BackwardsEulerStep,
            CrankNicolsonStep,
            ExplicitEulerStep,
        )

        params = dict(self.algorithm_parameters)
        precision = system.precision
        n_states = system.sizes.states

        if self.algorithm == "explicit_euler":
            step = params.get("dt", self._dt_min)
            return ExplicitEulerStep(
                system.dxdt_function,
                precision,
                n_states,
                step,
            )

        defaults: Dict[str, Any] = {
            "atol": params.pop("atol", self._atol),
            "rtol": params.pop("rtol", self._rtol),
        }
        defaults.update(params)

        if self.algorithm == "backwards_euler":
            return BackwardsEulerStep(
                precision=precision,
                n=n_states,
                dxdt_function=system.dxdt_function,
                get_solver_helper_fn=system.get_solver_helper,
                **defaults,
            )

        if self.algorithm == "crank_nicolson":
            return CrankNicolsonStep(
                precision=precision,
                n=n_states,
                dxdt_function=system.dxdt_function,
                get_solver_helper_fn=system.get_solver_helper,
                **defaults,
            )

        raise KeyError(f"Unsupported algorithm '{self.algorithm}'.")

    # ------------------------------------------------------------------
    # Updates and validation
    # ------------------------------------------------------------------
    def apply_updates(self, updates: Dict[str, Any]) -> Tuple[set[str], bool]:
        """Apply parameter updates and report recognised keys.

        Parameters
        ----------
        updates
            Mapping of configuration names to new values.

        Returns
        -------
        set[str]
            Keys handled by this settings object.
        bool
            Whether dependent objects must be re-instantiated.
        """

        recognised: set[str] = set()
        needs_rebuild = False

        for key, value in updates.items():
            if key == "algorithm":
                new_value = _normalise_algorithm(value)
                if new_value != self.algorithm:
                    self.algorithm = new_value
                    self._ensure_algorithm_defaults()
                    needs_rebuild = True
                recognised.add(key)
            elif key in {"step_controller", "step_controller_kind"}:
                new_kind = _normalise_controller(value)
                if new_kind != self.step_controller_kind:
                    self.step_controller_kind = new_kind
                    self._apply_parameter_defaults()
                    needs_rebuild = True
                recognised.add(key)
            elif key in {
                "dt_min",
                "dt_max",
                "dt_save",
                "dt_summarise",
                "atol",
                "rtol",
            }:
                setattr(self, key, value)
                needs_rebuild = True
                recognised.add(key)
            elif key == "algorithm_parameters" and isinstance(value, dict):
                if value:
                    self.algorithm_parameters.update(value)
                    needs_rebuild = True
                recognised.add(key)
            elif (
                key == "step_controller_parameters"
                and isinstance(value, dict)
            ):
                if value:
                    self.step_controller_parameters.update(value)
                    self._apply_parameter_defaults()
                    needs_rebuild = True
                recognised.add(key)
            elif (
                key in _KNOWN_ALGORITHM_KEYS
                or key in self.algorithm_parameters
            ):
                self.algorithm_parameters[key] = value
                needs_rebuild = True
                recognised.add(key)
            elif (
                key in _KNOWN_CONTROLLER_KEYS
                or key in self.step_controller_parameters
            ):
                self.step_controller_parameters[key] = value
                self._apply_parameter_defaults()
                needs_rebuild = True
                recognised.add(key)

        if needs_rebuild:
            self.validate_settings()

        return recognised, needs_rebuild
