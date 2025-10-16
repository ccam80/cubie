"""Symbolic evaluation helpers for CPU reference integrators."""

from typing import Dict, Optional, Set

import numpy as np
import sympy as sp

from cubie import SymbolicODE
from cubie.odesystems.symbolic.jacobian import generate_jacobian
from cubie.odesystems.symbolic.sym_utils import topological_sort

from .cpu_utils import Array


TIME_SYMBOL = sp.Symbol("t", real=True)


class CPUODESystem:
    """Evaluator for symbolic systems using compiled numerical functions."""

    def __init__(self, system: SymbolicODE) -> None:
        self.system = system
        self.precision = system.precision
        self.n_states = system.sizes.states
        self.n_observables = system.sizes.observables
        self._state_template = np.zeros(self.n_states, dtype=self.precision)
        self._observable_template = np.zeros(
            self.n_observables, dtype=self.precision
        )

        indexed = system.indices
        self._state_index = indexed.states.index_map
        self._parameter_index = indexed.parameters.index_map
        self._constant_index = indexed.constants.index_map
        self._driver_index = indexed.drivers.index_map
        self._observable_index = indexed.observables.index_map

        self._dx_index = indexed.dxdt.index_map
        ordered_equations = topological_sort(system.equations)
        self._equations = ordered_equations
        self._jacobian_expr = generate_jacobian(
            system.equations,
            self._state_index,
            self._dx_index,
        )

        self._base_symbols: Set[sp.Symbol] = set().union(
            self._state_index.keys(),
            self._parameter_index.keys(),
            self._constant_index.keys(),
            self._driver_index.keys(),
            {TIME_SYMBOL},
        )
        self._observable_symbols: Set[sp.Symbol] = set(
            self._observable_index.keys()
        )
        self._dx_symbols: Set[sp.Symbol] = set(self._dx_index.keys())

        self._compile_expressions()

    def _compile_expressions(self) -> None:
        """Compile symbolic expressions into fast numerical functions."""

        self._compiled_equations = {}
        self._equation_symbols = {}

        for lhs, rhs in self._equations:
            free_vars = list(rhs.free_symbols)
            if free_vars:
                compiled_fn = sp.lambdify(free_vars, rhs, modules=["numpy"])
            else:
                compiled_fn = self.precision(rhs)
            self._equation_symbols[lhs] = free_vars
            self._compiled_equations[lhs] = compiled_fn

        if self._jacobian_expr.shape[0] > 0:
            jacobian_entries = []
            jacobian_symbols = []
            jac_rows, jac_cols = self._jacobian_expr.shape

            for i in range(jac_rows):
                row_entries = []
                row_symbols = []
                for j in range(jac_cols):
                    expr = self._jacobian_expr[i, j]
                    expr_syms = list(expr.free_symbols)
                    if expr_syms:
                        compiled_entry = sp.lambdify(
                            expr_syms, expr, modules=["numpy"]
                        )
                    else:
                        compiled_entry = self.precision(expr)

                    row_entries.append(compiled_entry)
                    row_symbols.append(expr_syms)
                jacobian_entries.append(row_entries)
                jacobian_symbols.append(row_symbols)

            self._compiled_jacobian = jacobian_entries
            self._jacobian_symbols = jacobian_symbols
        else:
            self._compiled_jacobian = []
            self._jacobian_symbols = []

        self._observable_eval_order = self._resolve_dependencies(
            self._observable_symbols
        )
        self._dx_eval_order = self._resolve_dependencies(
            self._dx_symbols, skip=self._observable_symbols
        )

    def _resolve_dependencies(
        self,
        targets: Set[sp.Symbol],
        *,
        skip: Optional[Set[sp.Symbol]] = None,
    ) -> list[sp.Symbol]:
        """Return topologically ordered symbols needed to evaluate ``targets``."""

        if skip is None:
            skip = set()

        closure: Set[sp.Symbol] = set()

        def visit(symbol: sp.Symbol) -> None:
            if (
                symbol in skip
                or symbol in closure
                or symbol not in self._equation_symbols
            ):
                return

            closure.add(symbol)
            for dependency in self._equation_symbols[symbol]:
                if dependency in skip or dependency in self._base_symbols:
                    continue
                visit(dependency)

        for target in targets:
            visit(target)

        ordered = [lhs for lhs, _ in self._equations if lhs in closure]
        return ordered

    def _get_symbol_values(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        time_scalar: float,
        observables: Optional[Array] = None,
    ) -> Dict[sp.Symbol, float]:
        precision = self.precision
        values: Dict[sp.Symbol, float] = {}
        values.update(
            {
                **{
                    sym: precision(state[index])
                    for sym, index in self._state_index.items()
                },
                **{
                    sym: precision(params[index])
                    for sym, index in self._parameter_index.items()
                },
                **{
                    sym: precision(
                        self.system.constants.values_dict[str(sym)]
                    )
                    for sym in self._constant_index.keys()
                },
                **{
                    sym: precision(drivers[index])
                    for sym, index in self._driver_index.items()
                },
            }
        )

        if observables is not None:
            values.update(
                {
                    sym: precision(observables[index])
                    for sym, index in self._observable_index.items()
                }
            )

        values[TIME_SYMBOL] = self.precision(time_scalar)
        return values

    def observables(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        time_scalar: float,
    ) -> Array:
        """Evaluate the observable expressions for the current state."""

        observables = self._observable_template.copy()
        symbol_values = self._get_symbol_values(
            state,
            params,
            drivers,
            time_scalar,
        )

        for lhs in self._observable_eval_order:
            argsymbols = self._equation_symbols[lhs]
            if argsymbols:
                args = tuple(symbol_values[sym] for sym in argsymbols)
                value = self.precision(
                    self._compiled_equations[lhs](*args)
                )
            else:
                value = self.precision(self._compiled_equations[lhs])

            symbol_values[lhs] = value
            if lhs in self._observable_index:
                observables[self._observable_index[lhs]] = value

        return observables

    def rhs(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        observables: Array,
        time_scalar: float,
    ) -> tuple[Array, Dict[sp.Symbol, float]]:
        """Evaluate ``dx/dt`` using pre-computed observables."""

        dxdt = self._state_template.copy()
        symbol_values = self._get_symbol_values(
            state,
            params,
            drivers,
            time_scalar,
            observables=observables,
        )

        for lhs in self._dx_eval_order:
            argsymbols = self._equation_symbols[lhs]
            if argsymbols:
                args = tuple(symbol_values[sym] for sym in argsymbols)
                value = self.precision(
                    self._compiled_equations[lhs](*args)
                )
            else:
                value = self.precision(self._compiled_equations[lhs])

            symbol_values[lhs] = value
            if lhs in self._dx_index:
                dxdt[self._dx_index[lhs]] = value

        return dxdt, symbol_values

    def jacobian(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        observables: Array,
        time_scalar: float,
    ) -> Array:
        if not self._compiled_jacobian:
            return np.zeros((self.n_states, self.n_states), dtype=self.precision)

        _, symbol_values = self.rhs(
            state,
            params,
            drivers,
            observables,
            time_scalar,
        )
        jac_rows = len(self._compiled_jacobian)
        jac_cols = len(self._compiled_jacobian[0]) if jac_rows > 0 else 0
        jacobian = np.zeros((jac_rows, jac_cols), dtype=self.precision)

        for i, (row, row_symbols) in enumerate(
            zip(self._compiled_jacobian, self._jacobian_symbols)
        ):
            for j, (compiled_entry, expr_symbols) in enumerate(
                zip(row, row_symbols)
            ):
                if expr_symbols:
                    args = tuple(symbol_values[sym] for sym in expr_symbols)
                    jacobian[i, j] = self.precision(compiled_entry(*args))
                else:
                    jacobian[i, j] = compiled_entry
        return jacobian


__all__ = ["CPUODESystem", "TIME_SYMBOL"]
