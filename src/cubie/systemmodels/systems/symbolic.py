import numpy as np
import sympy as sp
from numba import cuda, from_dtype

from cubie.systemmodels.systems.GenericODE import GenericODE


class SymbolicODESystem(GenericODE):
    """Create an ODE system from SymPy expressions.

    Parameters are provided as SymPy symbols.  The differential equations are
    provided as a list of ``sympy.Eq`` objects where the left hand side is a
    state or observable symbol and the right hand side is an expression
    composed of states, parameters, constants and previously defined
    observables.
    """

    def __init__(self, *, constants=None, parameters=None, states=None,
                 observables=None, equations=None, precision=np.float64,
                 num_drivers=1):
        if equations is None:
            raise ValueError("equations must be provided")

        # Store the original SymPy symbols for later use
        self.state_symbols, state_vals = self._prepare_container(states)
        self.parameter_symbols, param_vals = self._prepare_container(parameters)
        self.constant_symbols, const_vals = self._prepare_container(constants)
        self.observable_symbols, obs_vals = self._prepare_container(observables)

        self.equations = equations

        super().__init__(initial_values=state_vals,
                         parameters=param_vals,
                         constants=const_vals,
                         observables=obs_vals,
                         precision=precision,
                         num_drivers=num_drivers)

        self._validate_equations()

    @staticmethod
    def _prepare_container(container):
        """Convert a container of SymPy symbols to a form suitable for
        :class:`SystemValues`.

        Returns a tuple ``(symbols, sv_input)`` where ``symbols`` is a list of
        SymPy symbols and ``sv_input`` is either a list of names or a dict of
        ``name -> value``.
        """
        if container is None:
            return [], {}
        if isinstance(container, dict):
            symbols = list(container.keys())
            sv_input = {s.name: container[s] for s in symbols}
            return symbols, sv_input
        else:
            symbols = list(container)
            sv_input = [s.name for s in symbols]
            return symbols, sv_input

    def _validate_equations(self):
        allowed = set(self.state_symbols + self.parameter_symbols +
                      self.constant_symbols + self.observable_symbols)
        for eq in self.equations:
            if not isinstance(eq, sp.Eq):
                raise TypeError("equations must be sympy.Eq instances")
            if eq.lhs not in allowed:
                raise ValueError("Equation LHS must be a state or observable")
            undefined = eq.rhs.free_symbols - allowed
            if undefined:
                raise ValueError(f"Equation RHS contains undefined symbols: {undefined}")

    def build(self):
        """Compile the system into a CUDA device function."""
        global global_constants
        global_constants = self.compile_settings.constants.values_array.astype(self.precision)

        numba_precision = from_dtype(self.precision)

        code_lines = self._generate_code_lines()
        func_code = ["def sympy_dxdt(state, parameters, driver, observables, dxdt):"]
        func_code.extend([f"    {line}" for line in code_lines])
        ns = {}
        exec("\n".join(func_code), {}, ns)
        python_func = ns["sympy_dxdt"]

        jitted = cuda.jit((numba_precision[:], numba_precision[:], numba_precision[:],
                           numba_precision[:], numba_precision[:]),
                          device=True, inline=True)(python_func)
        return jitted

    def _generate_code_lines(self):
        state_idx = {s: i for i, s in enumerate(self.state_symbols)}
        obs_idx = {s: i for i, s in enumerate(self.observable_symbols)}
        param_idx = {s: i for i, s in enumerate(self.parameter_symbols)}
        const_idx = {s: i for i, s in enumerate(self.constant_symbols)}

        state_arr = sp.IndexedBase('state')
        param_arr = sp.IndexedBase('parameters')
        const_arr = sp.IndexedBase('global_constants')
        obs_arr = sp.IndexedBase('observables')

        repl = {s: state_arr[i] for s, i in state_idx.items()}
        repl.update({s: param_arr[i] for s, i in param_idx.items()})
        repl.update({s: const_arr[i] for s, i in const_idx.items()})
        repl.update({s: obs_arr[i] for s, i in obs_idx.items()})

        lines = []
        for eq in self.equations:
            rhs = eq.rhs.subs(repl)
            rhs_code = sp.pycode(rhs)
            if eq.lhs in state_idx:
                lines.append(f"dxdt[{state_idx[eq.lhs]}] = {rhs_code}")
            else:
                lines.append(f"observables[{obs_idx[eq.lhs]}] = {rhs_code}")
        return lines

    def correct_answer_python(self, states, parameters, drivers):
        state_idx = {s: i for i, s in enumerate(self.state_symbols)}
        param_idx = {s: i for i, s in enumerate(self.parameter_symbols)}
        const_idx = {s: i for i, s in enumerate(self.constant_symbols)}
        obs_idx = {s: i for i, s in enumerate(self.observable_symbols)}

        values = {}
        for sym, i in state_idx.items():
            values[sym] = states[i]
        for sym, i in param_idx.items():
            values[sym] = parameters[i]
        for sym, i in const_idx.items():
            values[sym] = self.compile_settings.constants.values_array[i]

        dxdt = np.zeros(self.num_states, dtype=self.precision)
        observables = np.zeros(self.num_observables, dtype=self.precision)

        for eq in self.equations:
            rhs_val = float(eq.rhs.subs(values))
            if eq.lhs in state_idx:
                dxdt[state_idx[eq.lhs]] = rhs_val
            else:
                observables[obs_idx[eq.lhs]] = rhs_val
                values[eq.lhs] = rhs_val
        return dxdt, observables
