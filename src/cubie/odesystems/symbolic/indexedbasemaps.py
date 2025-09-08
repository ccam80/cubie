"""Indexed base mapping classes for symbolic ODE definitions."""

from typing import Dict, Iterable, Optional, Union

import sympy as sp
from numpy import asarray
from numpy.typing import ArrayLike, NDArray


class IndexedBaseMap:
    def __init__(
        self,
        base_name: str,
        symbol_labels: Iterable[str],
        input_defaults: Optional[Union[ArrayLike, NDArray]] = None,
        length=0,
        real=True,
    ):
        if length == 0:
            length = len(list(symbol_labels))

        self.length = length
        self.base_name = base_name
        self.real = real
        self.base = sp.IndexedBase(base_name, shape=(length,), real=real)
        self.index_map = {
            sp.Symbol(name, real=real): index
            for index, name in enumerate(symbol_labels)
        }
        self.ref_map = {
            sp.Symbol(name, real=real): self.base[index]
            for index, name in enumerate(symbol_labels)
        }
        self.symbol_map = {
            name: sp.Symbol(name, real=real) for name in symbol_labels
        }
        if input_defaults is None:
            input_defaults = asarray([0.0] * length)
        elif len(input_defaults) != length:
            raise ValueError(
                "Input defaults must be the same length as the list of symbols"
            )
        self.default_values = dict(zip(self.ref_map.keys(), input_defaults))

    def pop(self, sym):
        """Remove a symbol from this object"""
        self.ref_map.pop(sym)
        self.index_map.pop(sym)
        self.symbol_map.pop(str(sym))
        self.default_values.pop(sym)
        self.base = sp.IndexedBase(
            self.base_name, shape=(len(self.ref_map),), real=self.real
        )
        self.length = len(self.ref_map)

    def push(self, sym, default_value=0.0):
        """Adds a symbol to this object"""
        index = self.length
        self.base = sp.IndexedBase(
            self.base_name, shape=(index + 1,), real=self.real
        )
        self.length += 1
        self.ref_map[sym] = self.base[index]
        self.index_map[sym] = index
        self.symbol_map[str(sym)] = sym
        self.default_values[sym] = default_value

    def update_values(
        self, updates_dict: Dict[Union[str, sp.Symbol], float] = None, **kwargs
    ):
        """Update the default values of the indexed base.

        Parameters
        ----------
            updates_dict : dict of strings, floats
                A dictionary mapping constant names to their new values.
            **kwargs: key-value pairs
                Additional constant updates in key=value form, overrides
                updates_dict.

        Notes
        -----
        Silently ignores keys that are not found in the indexed base map.
        """
        if updates_dict is None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return

        if any(isinstance(key, sp.Symbol) for key in updates_dict.keys()):
            symbol_update_dict = {
                key: value
                for key, value in updates_dict.items()
                if key in self.ref_map
            }
        else:
            symbol_update_dict = {
                self.symbol_map[key]: value
                for key, value in updates_dict.items()
                if key in self.symbol_map
            }

        for sym, val in symbol_update_dict.items():
            self.default_values[sym] = val
        return


class IndexedBases:
    def __init__(
        self,
        states: IndexedBaseMap,
        parameters: IndexedBaseMap,
        constants: IndexedBaseMap,
        observables: IndexedBaseMap,
        drivers: IndexedBaseMap,
        dxdt: IndexedBaseMap,
    ):
        self.states = states
        self.parameters = parameters
        self.constants = constants
        self.observables = observables
        self.drivers = drivers
        self.dxdt = dxdt
        self.all_indices = {
            **self.states.ref_map,
            **self.parameters.ref_map,
            **self.observables.ref_map,
            **self.drivers.ref_map,
            **self.dxdt.ref_map,
        }

    @classmethod
    def from_user_inputs(
        cls,
        states: Union[dict[str, float], Iterable[str]],
        parameters: Union[dict, Iterable[str]],
        constants: Union[dict, Iterable[str]],
        observables: Iterable[str],
        drivers: Iterable[str],
        real=True,
    ):
        if isinstance(states, dict):
            state_names = list(states.keys())
            state_defaults = list(states.values())
        else:
            state_names = list(states)
            state_defaults = None

        if isinstance(parameters, dict):
            param_names = list(parameters.keys())
            param_defaults = list(parameters.values())
        else:
            param_names = list(parameters)
            param_defaults = None

        if isinstance(constants, dict):
            const_names = list(constants.keys())
            const_defaults = list(constants.values())
        else:
            const_names = list(constants)
            const_defaults = None

        states_ = IndexedBaseMap(
            "state", state_names, input_defaults=state_defaults, real=real
        )
        parameters_ = IndexedBaseMap(
            "parameters", param_names, input_defaults=param_defaults, real=real
        )
        constants_ = IndexedBaseMap(
            "constants", const_names, input_defaults=const_defaults, real=real
        )
        observables_ = IndexedBaseMap("observables", observables, real=real)
        drivers_ = IndexedBaseMap("drivers", drivers, real=real)
        dxdt_ = IndexedBaseMap(
            "out", [f"d{s}" for s in state_names], real=real
        )
        return cls(
            states_, parameters_, constants_, observables_, drivers_, dxdt_
        )

    def update_constants(
        self, updates_dict: Dict[str, float] = None, **kwargs
    ):
        """Update values in the constants object

        Parameters
        ----------
            updates_dict : dict of strings, floats
                A dictionary mapping constant names to their new values.
            **kwargs: key-value pairs
                Additional constant updates in key=value form, overrides
                updates_dict.

        Notes
        -----
        Silently ignores keys that are not found in the constants symbol table.
        """
        self.constants.update_values(updates_dict, **kwargs)

    @property
    def state_names(self):
        return list(self.states.symbol_map.keys())

    @property
    def state_symbols(self):
        return list(self.states.ref_map.keys())

    @property
    def state_values(self):
        return self.states.default_values

    @property
    def parameter_names(self):
        return list(self.parameters.symbol_map.keys())

    @property
    def parameter_symbols(self):
        return list(self.parameters.ref_map.keys())

    @property
    def parameter_values(self):
        return self.parameters.default_values

    @property
    def constant_names(self):
        return list(self.constants.symbol_map.keys())

    @property
    def constant_symbols(self):
        return list(self.constants.ref_map.keys())

    @property
    def constant_values(self):
        return self.constants.default_values

    @property
    def observable_names(self):
        return list(self.observables.symbol_map.keys())

    @property
    def observable_symbols(self):
        return list(self.observables.ref_map.keys())

    @property
    def driver_names(self):
        return list(self.drivers.symbol_map.keys())

    @property
    def driver_symbols(self):
        return list(self.drivers.ref_map.keys())

    @property
    def dxdt_names(self) -> Iterable[str]:
        return list(self.dxdt.symbol_map.keys())

    @property
    def dxdt_symbols(self):
        return list(self.dxdt.ref_map.keys())

    @property
    def all_arrayrefs(self) -> dict[str, sp.Symbol]:
        return {
            **self.states.ref_map,
            **self.parameters.ref_map,
            **self.observables.ref_map,
            **self.drivers.ref_map,
            **self.dxdt.ref_map,
        }

    @property
    def all_symbols(self) -> dict[str, sp.Symbol]:
        return {
            **self.states.symbol_map,
            **self.parameters.symbol_map,
            **self.constants.symbol_map,
            **self.observables.symbol_map,
            **self.drivers.symbol_map,
            **self.dxdt.symbol_map,
        }

    def __getitem__(self, item):
        """Returns a reference to the indexed base for any symbol in the map"""
        return self.all_indices[item]
