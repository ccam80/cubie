"""Characterise when the Neumann series preconditioner converges.

The Neumann series approximates ``(betaI - gamma h (A (x) J))^-1`` via
the truncated expansion ``beta^-1 (I + T + T^2 + ...)`` where
``T = (gamma h / beta)(A (x) J)``. This converges if and only if the
spectral radius ``rho(T) < 1``.

Since ``T`` scales linearly with the step factor, the static diagnostic
reports its spectral radius per unit ``h`` for FIRK, or per unit
``a_ij * h`` for a single-stage helper whose runtime ``a_ij`` is not
known. An exact convergence verdict is available only when the full
step factor is supplied.

The Jacobian is evaluated at the initial state by central finite
differences of the system's **compiled** ``dxdt`` device function,
launched on the device at the system's compiled precision, so the
diagnostic reflects the behaviour of the production device code rather
than a host-side reconstruction. Finite-differencing the guarded
right-hand side also takes removable singularities numerically: the
analytic derivative of a CellML gating term such as
``(V - E) / (exp((V - E) / k) - 1)`` is ``0 / 0`` at the resting
potential even though the term itself is finite there, and the guarded
device code evaluates it cleanly.

Published Objects
-----------------
:class:`NeumannRHSEvaluator`
    CUDAFactory building the finite-difference Jacobian kernel that
    runs the compiled ``dxdt``.
:func:`neumann_spectral_radius`
    Pure-numeric Neumann-series spectral-radius diagnostic as a function
    of the Jacobian and the ``beta``/``gamma``/tableau parameters.
:func:`check_neumann_convergence`
    Evaluate convergence diagnostics for the Neumann preconditioner and
    emit a warning when divergence is likely.
"""

import logging
import warnings
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import sympy as sp
from attrs import define, field, validators as val

from cubie.CUDAFactory import (
    CUDADispatcherCache,
    CUDAFactory,
    CUDAFactoryConfig,
)
from cubie._utils import PrecisionDType
from cubie.cubie_cache import CacheConfig, CubieCacheHandler
from cubie.cuda_simsafe import CUDA_SIMULATION, cuda
from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases

logger = logging.getLogger(__name__)


@define
class NeumannEvaluatorConfig(CUDAFactoryConfig):
    """Compile settings for the Jacobian evaluation kernel.

    Parameters
    ----------
    dxdt_function
        Compiled ``dxdt`` device function the kernel wraps. Excluded
        from hashing; a change still invalidates the build.
    dxdt_settings_hash
        Hash of the owning system's own settings and constants, so the
        disk-cache key changes whenever ``dxdt`` is regenerated with
        different semantics under the same equations.
    cache_config
        Disk-cache configuration. Excluded from hashing so cache
        relocation never alters the disk-cache key; a change still
        invalidates the build, which reattaches a fresh cache.
    """

    dxdt_function: Optional[Callable] = field(default=None, eq=False)
    dxdt_settings_hash: str = field(
        default="",
        validator=val.instance_of(str),
    )
    cache_config: CacheConfig = field(
        factory=CacheConfig,
        validator=val.instance_of(CacheConfig),
        eq=False,
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()


@define
class NeumannEvaluatorCache(CUDADispatcherCache):
    """Container for the compiled Jacobian evaluation kernel."""

    evaluation_kernel: Union[int, Callable] = field(default=-1)


class NeumannRHSEvaluator(CUDAFactory):
    """Finite-difference Jacobian evaluator for a compiled system.

    Launches the system's compiled ``dxdt`` device function at
    perturbed initial states and forms the Jacobian by central finite
    differences, so the diagnostic sees exactly the device code the
    solver runs, at the compiled precision. The owning
    :class:`SymbolicODE` refreshes ``dxdt_function`` and
    ``dxdt_settings_hash`` before each use, so the kernel rebuilds
    through the standard compile-settings invalidation whenever the
    system's device code changes. Cache settings arrive through the
    same ``update`` chain; the build attaches a configured disk cache
    when caching is enabled.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__()
        # The handler must exist before setup_compile_settings, which
        # invalidates the build and reaches the handler through
        # _invalidate_cache.
        self._cache_handler = CubieCacheHandler(cache_config)
        self.setup_compile_settings(
            NeumannEvaluatorConfig(
                precision=precision,
                cache_config=cache_config,
            )
        )

    def build(self) -> NeumannEvaluatorCache:
        """Compile the evaluation kernel for the configured ``dxdt``."""
        config = self.compile_settings
        dxdt_function = config.dxdt_function
        if dxdt_function is None:
            return NeumannEvaluatorCache()

        # no cover: start
        @cuda.jit(**self.jit_kwargs)
        def evaluate_rhs(
            states, parameters, drivers, observables, out, t
        ):
            i = cuda.grid(1)
            if i < states.shape[0]:
                dxdt_function(
                    states[i],
                    parameters,
                    drivers,
                    observables[i],
                    out[i],
                    t,
                )
        # no cover: end
        if not CUDA_SIMULATION:
            disk_cache = self._cache_handler.configured_cache(
                config.cache_config.system_hash,
                config.values_hash,
            )
            if disk_cache is not None:
                evaluate_rhs._cache = disk_cache
        return NeumannEvaluatorCache(evaluation_kernel=evaluate_rhs)

    def _invalidate_cache(self) -> None:
        """Invalidate the built kernel, flushing in flush-on-change mode."""
        super()._invalidate_cache()
        self._cache_handler.invalidate()

    def jacobian(
        self,
        index_map: IndexedBases,
        t0: float = 0.0,
    ) -> np.ndarray:
        """Return the state Jacobian at the initial state.

        Parameters
        ----------
        index_map
            Index maps supplying current state/parameter values.
        t0
            Time at which to evaluate the right-hand side.

        Returns
        -------
        numpy.ndarray
            The ``state_count x state_count`` Jacobian in float64.
            Entries that cannot be evaluated are returned as ``nan``.
        """
        precision = np.dtype(self.precision)
        state_map = index_map.states.index_map
        n = len(state_map)
        base = np.zeros(n, dtype=np.float64)
        state_defaults = index_map.states.default_values
        for sym, idx in state_map.items():
            base[idx] = float(state_defaults[sym])

        parameters = np.zeros(1, dtype=precision)
        if hasattr(index_map, "parameters"):
            parameter_map = index_map.parameters.index_map
            if parameter_map:
                parameters = np.zeros(
                    len(parameter_map), dtype=precision
                )
                defaults = index_map.parameters.default_values
                for sym, idx in parameter_map.items():
                    parameters[idx] = float(defaults[sym])

        n_drivers = 1
        if hasattr(index_map, "drivers"):
            n_drivers = max(1, len(index_map.drivers.index_map))
        drivers = np.zeros(n_drivers, dtype=precision)

        n_observables = max(1, len(index_map.observables.index_map))
        observables = np.zeros((2 * n, n_observables), dtype=precision)
        out = np.zeros((2 * n, n), dtype=precision)

        # Central-difference step: cube-root of the compiled
        # precision's epsilon balances round-off against truncation
        # error for a second-order stencil at that precision.
        fd_step = float(np.finfo(precision).eps) ** (1.0 / 3.0)
        states = np.empty((2 * n, n), dtype=precision)
        for col in range(n):
            step = fd_step * max(abs(base[col]), 1.0)
            forward = base.copy()
            backward = base.copy()
            forward[col] += step
            backward[col] -= step
            states[2 * col] = forward
            states[2 * col + 1] = backward

        kernel = self.get_cached_output("evaluation_kernel")
        threads = 32
        blocks = (2 * n + threads - 1) // threads

        def launch():
            device_states = cuda.to_device(states)
            device_parameters = cuda.to_device(parameters)
            device_drivers = cuda.to_device(drivers)
            device_observables = cuda.to_device(observables)
            device_out = cuda.to_device(out)
            kernel[blocks, threads](
                device_states,
                device_parameters,
                device_drivers,
                device_observables,
                device_out,
                precision.type(t0),
            )
            return device_out.copy_to_host()

        if CUDA_SIMULATION:
            # The simulator executes device code as host Python,
            # where domain errors raise instead of producing the
            # non-finite values device math returns; translate them
            # into the caller's could-not-evaluate path.
            try:
                evaluated = launch()
            except (
                ZeroDivisionError,
                FloatingPointError,
                OverflowError,
                ValueError,
            ):
                return np.full((n, n), np.nan)
        else:
            evaluated = launch()

        evaluated = evaluated.astype(np.float64)
        jacobian = np.empty((n, n), dtype=np.float64)
        for col in range(n):
            # Effective step from the precision-cast states so the
            # divisor matches the perturbation the device code saw.
            denominator = float(states[2 * col, col]) - float(
                states[2 * col + 1, col]
            )
            jacobian[:, col] = (
                evaluated[2 * col] - evaluated[2 * col + 1]
            ) / denominator
        return jacobian


def neumann_spectral_radius(
    jacobian: np.ndarray,
    beta: float = 1.0,
    gamma: float = 1.0,
    stage_coefficients: Optional[
        Sequence[Sequence[Union[float, sp.Expr]]]
    ] = None,
    stage_coefficient: Optional[float] = None,
    step_factor_value: Optional[float] = None,
) -> Dict[str, object]:
    """Return the Neumann-series spectral radius and step limit.

    The series matrix is ``T = (gamma / beta) h (A (x) J)``. This
    routine evaluates its spectral radius per unit step factor and,
    when ``step_factor_value`` is supplied, the radius at that value.

    Parameters
    ----------
    jacobian
        The ``n x n`` state Jacobian.
    beta, gamma
        Transformation parameters from the FIRK formulation.
    stage_coefficients
        Butcher-tableau ``A`` matrix. When omitted, the single-stage
        helper is analysed.
    stage_coefficient
        Single-stage ``a_ij`` coefficient. When omitted alongside
        ``stage_coefficients``, the reported step factor is
        ``a_ij * h`` rather than ``h``.
    step_factor_value
        Value of the reported step factor at which to evaluate the
        infinite series. When omitted, no convergence verdict is made.

    Returns
    -------
    dict
        ``rho_per_unit_step_factor`` -- radius per unit step factor.
        ``critical_step_factor`` -- largest factor magnitude.
        ``step_factor`` -- ``"h"`` or ``"a_ij * h"``.
        ``rho_series`` -- radius at ``step_factor_value``, or ``None``.
        ``series_converges`` -- exact infinite-series verdict, or
        ``None`` without a factor value.
        ``n_states`` / ``n_stages`` -- flattened-system dimensions.
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    n = jacobian.shape[0]
    if beta == 0.0:
        raise ValueError("beta must be nonzero for a Neumann inverse")

    if stage_coefficients is not None:
        if stage_coefficient is not None:
            raise ValueError(
                "stage_coefficient and stage_coefficients are exclusive"
            )
        coefficients = np.array(
            [[float(c) for c in row] for row in stage_coefficients],
            dtype=np.float64,
        )
        s = coefficients.shape[0]
        coupling = np.kron(coefficients, jacobian)
        step_factor = "h"
    else:
        s = 1
        if stage_coefficient is None:
            coupling = jacobian
            step_factor = "a_ij * h"
        else:
            coupling = float(stage_coefficient) * jacobian
            step_factor = "h"

    series_per_unit_step = (gamma / beta) * coupling
    eigenvalues = np.linalg.eigvals(series_per_unit_step)
    rho_per_unit_step_factor = float(np.max(np.abs(eigenvalues)))
    critical_step_factor = (
        float("inf")
        if rho_per_unit_step_factor == 0.0
        else 1.0 / rho_per_unit_step_factor
    )
    rho_series = (
        None
        if step_factor_value is None
        else abs(float(step_factor_value)) * rho_per_unit_step_factor
    )

    return {
        "rho_per_unit_step_factor": rho_per_unit_step_factor,
        "critical_step_factor": critical_step_factor,
        "step_factor": step_factor,
        "rho_series": rho_series,
        "series_converges": (
            None if rho_series is None else rho_series < 1.0
        ),
        "n_states": n,
        "n_stages": s,
    }


def check_neumann_convergence(
    index_map: IndexedBases,
    evaluator: NeumannRHSEvaluator,
    stage_coefficients: Optional[
        Sequence[Sequence[Union[float, sp.Expr]]]
    ] = None,
    stage_nodes: Optional[Sequence[Union[float, sp.Expr]]] = None,
    beta: float = 1.0,
    gamma: float = 1.0,
    t0: float = 0.0,
    step_size: Optional[float] = None,
    stage_coefficient: Optional[float] = None,
) -> Dict[str, object]:
    """Characterise Neumann convergence at the initial state.

    Evaluates the state Jacobian at the initial state by finite
    differences of the compiled ``dxdt`` device function and computes
    the spectral radius of the implemented Neumann series matrix.

    Parameters
    ----------
    index_map
        Index maps (states, parameters, etc.) with default values.
    evaluator
        Finite-difference Jacobian evaluator wrapping the system's
        compiled ``dxdt`` device function.
    stage_coefficients
        Butcher-tableau ``A`` matrix for the FIRK method.
    stage_nodes
        Butcher-tableau ``c`` nodes. These influence the Jacobian only
        through the ``O(h)`` per-stage time/state offset, which this
        h-independent diagnostic neglects, so they are not used.
    beta, gamma
        Transformation parameters from the FIRK formulation.
    t0
        Time at which to evaluate the Jacobian.
    step_size
        Optional step size for an exact local convergence verdict.
    stage_coefficient
        Runtime single-stage ``a_ij`` value. Required with
        ``step_size`` for a single-stage exact verdict.

    Returns
    -------
    dict
        The :func:`neumann_spectral_radius` result extended with
        ``J_numeric``. ``series_converges`` is ``None`` when the full
        step factor is unavailable or the Jacobian cannot be evaluated.
    """
    jacobian = evaluator.jacobian(index_map, t0=t0)

    if not np.isfinite(jacobian).all():
        n_bad = int(np.count_nonzero(~np.isfinite(jacobian)))
        logger.warning(
            "Neumann preconditioner convergence not verified: the "
            "Jacobian at the initial state has %d non-finite entries "
            "(likely a genuine singularity at t0).",
            n_bad,
        )
        n_stages = (
            len(stage_coefficients)
            if stage_coefficients is not None
            else 1
        )
        return {
            "rho_per_unit_step_factor": float("nan"),
            "critical_step_factor": float("nan"),
            "step_factor": (
                "h"
                if stage_coefficients is not None
                or stage_coefficient is not None
                else "a_ij * h"
            ),
            "rho_series": None,
            "series_converges": None,
            "n_states": jacobian.shape[0],
            "n_stages": n_stages,
            "J_numeric": jacobian,
        }

    step_factor_value = None
    if step_size is not None:
        if stage_coefficients is not None or stage_coefficient is not None:
            step_factor_value = step_size

    result = neumann_spectral_radius(
        jacobian, beta=beta, gamma=gamma,
        stage_coefficients=stage_coefficients,
        stage_coefficient=stage_coefficient,
        step_factor_value=step_factor_value,
    )
    result["J_numeric"] = jacobian

    if (
        result["series_converges"] is None
        and result["rho_per_unit_step_factor"] >= 1.0
    ):
        rho_per_step = result["rho_per_unit_step_factor"]
        critical_step = result["critical_step_factor"]
        step_factor = result["step_factor"].replace(" ", "")
        message = (
            "Neumann-series convergence depends on the runtime step "
            f"factor. At the initial state, rho(T) / abs({step_factor}) "
            f"= {rho_per_step:.3g}; convergence requires "
            f"abs({step_factor}) < "
            f"{critical_step:.3g}. The complete runtime step factor "
            "was unavailable to this static check, so this is not a "
            "divergence verdict."
        )
        warnings.warn(message, stacklevel=3)
        logger.warning(message)
    elif result["series_converges"] is False:
        rho_series = result["rho_series"]
        critical_step = result["critical_step_factor"]
        step_factor = result["step_factor"].replace(" ", "")
        message = (
            "The infinite Neumann series does not converge at the "
            f"initial-state Jacobian for abs({step_factor}) = "
            f"{abs(float(step_factor_value)):.3g}: rho(T) = "
            f"{rho_series:.3g} >= 1. Convergence requires "
            f"abs({step_factor}) < {critical_step:.3g}."
        )
        warnings.warn(message, stacklevel=3)
        logger.warning(message)

    return result
