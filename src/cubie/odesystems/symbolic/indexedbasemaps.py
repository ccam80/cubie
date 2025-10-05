"""Helpers that build indexed SymPy arrays for symbolic ODE metadata."""

from typing import Any, Dict, Iterable, Optional, Union

import sympy as sp


class IndexedBaseMap:
    """Map named scalar symbols onto a fixed-size SymPy indexed base."""

    def __init__(
        self,
        base_name: str,
        symbol_labels: Iterable[str],
        input_defaults: Optional[Iterable[Any]] = None,
        length: int = 0,
        real: bool = True,
    ) -> None:
        """Initialise an indexed base with optional default values.

        Parameters
        ----------
        base_name
            Base symbol name used for the generated :class:`sympy.IndexedBase`.
        symbol_labels
            Symbol names that define the entries in the indexed base.
        input_defaults
            Optional default numeric values aligned with ``symbol_labels``.
        length
            Length override for the indexed base when ``symbol_labels`` is
            provided as an iterator.
        real
            Whether to create real-only symbols in the indexed base.

        Returns
        -------
        None
        """
        labels = list(symbol_labels)
        if length == 0:
            length = len(labels)

        self.length = length
        self.base_name = base_name
        self.real = real
        self.base = sp.IndexedBase(base_name, shape=(length,), real=real)
        self.index_map = {
            sp.Symbol(name, real=real): index
            for index, name in enumerate(labels)
        }
        self.ref_map = {
            sp.Symbol(name, real=real): self.base[index]
            for index, name in enumerate(labels)
        }
        self.symbol_map = {name: sp.Symbol(name, real=real) for name in labels}

        self._passthrough_defaults = False
        if input_defaults is None:
            defaults = [0.0] * length
            self.default_values = dict(zip(self.ref_map.keys(), defaults))
            self.defaults = {
                str(symbol): value
                for symbol, value in self.default_values.items()
            }
        elif isinstance(input_defaults, dict):
            self._passthrough_defaults = True
            self.default_values = input_defaults
            self.defaults = input_defaults
        else:
            defaults = list(input_defaults)
            if len(defaults) != length:
                raise ValueError(
                    "Input defaults must be the same length as the list of "
                    "symbols"
                )
            self.default_values = dict(zip(self.ref_map.keys(), defaults))
            self.defaults = {
                str(symbol): value
                for symbol, value in self.default_values.items()
            }

    def pop(self, sym: sp.Symbol) -> None:
        """Remove a symbol from the indexed base."""
        self.ref_map.pop(sym)
        self.index_map.pop(sym)
        self.symbol_map.pop(str(sym))
        if not self._passthrough_defaults:
            self.default_values.pop(sym)
            self.defaults.pop(str(sym))
        self.base = sp.IndexedBase(
            self.base_name, shape=(len(self.ref_map),), real=self.real
        )
        self.length = len(self.ref_map)

    def push(self, sym: sp.Symbol, default_value: float = 0.0) -> None:
        """Append a new symbol to the indexed base."""
        index = self.length
        self.base = sp.IndexedBase(
            self.base_name, shape=(index + 1,), real=self.real
        )
        self.length += 1
        self.ref_map[sym] = self.base[index]
        self.index_map[sym] = index
        self.symbol_map[str(sym)] = sym
        if not self._passthrough_defaults:
            self.default_values[sym] = default_value
            self.defaults[str(sym)] = default_value

    def update_values(
        self,
        updates_dict: Optional[Dict[Union[str, sp.Symbol], float]] = None,
        **kwargs: float,
    ) -> None:
        """Update the stored default values.

        Parameters
        ----------
        updates_dict
            Mapping of symbol names or symbols to replacement values.
        **kwargs
            Additional symbol updates provided as keyword arguments. Entries
            take precedence over those in ``updates_dict``.

        Returns
        -------
        None

        Notes
        -----
        Silently ignores keys that are not found in the indexed base map.
        """
        if self._passthrough_defaults:
            return

        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
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
            self.defaults[str(sym)] = val
        return

    def set_passthrough_defaults(self, defaults: dict[str, Any]) -> None:
        """Replace defaults with a direct dictionary mapping."""

        self._passthrough_defaults = True
        self.default_values = defaults
        self.defaults = defaults


class IndexedBases:
    """Bundle of indexed bases describing a symbolic ODE definition."""

    def __init__(
        self,
        states: IndexedBaseMap,
        parameters: IndexedBaseMap,
        constants: IndexedBaseMap,
        observables: IndexedBaseMap,
        drivers: IndexedBaseMap,
        dxdt: IndexedBaseMap,
    ) -> None:
        """Initialise the combined index maps.

        Parameters
        ----------
        states
            Indexed base describing the system state vector.
        parameters
            Indexed base describing tunable model parameters.
        constants
            Indexed base describing compile-time constants.
        observables
            Indexed base describing recorded observables.
        drivers
            Indexed base describing driver signals.
        dxdt
            Indexed base describing the ``dx/dt`` outputs.

        Returns
        -------
        None
        """
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
        real: bool = True,
    ) -> "IndexedBases":
        """Construct indexed bases from user-provided metadata.

        Parameters
        ----------
        states
            Either a mapping of state names to default values or an iterable
            of state names.
        parameters
            Either a mapping of parameter names to default values or an
            iterable of parameter names.
        constants
            Either a mapping of constant names to default values or an
            iterable of constant names.
        observables
            Iterable of observable names.
        drivers
            Iterable of driver names.
        real
            Whether to constrain the generated symbols to real values.

        Returns
        -------
        IndexedBases
            Combined bundle of indexed bases for the symbolic ODE system.
        """
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
        self, updates_dict: Optional[Dict[str, float]] = None, **kwargs: float
    ) -> None:
        """Update the constant defaults while preserving other entries.

        Parameters
        ----------
        updates_dict
            Mapping of constant names to replacement values.
        **kwargs
            Additional constant updates provided as keyword arguments.

        Returns
        -------
        None

        Notes
        -----
        Silently ignores keys that are not found in the constants symbol table.
        """
        self.constants.update_values(updates_dict, **kwargs)

    @property
    def state_names(self) -> list[str]:
        """List of state symbol names."""
        return list(self.states.symbol_map.keys())

    @property
    def state_symbols(self) -> list[sp.Symbol]:
        """List of state symbols."""
        return list(self.states.ref_map.keys())

    @property
    def state_values(self) -> Dict[sp.Symbol, float]:
        """Mapping of state symbols to default values."""
        return self.states.default_values

    @property
    def parameter_names(self) -> list[str]:
        """List of parameter symbol names."""
        return list(self.parameters.symbol_map.keys())

    @property
    def parameter_symbols(self) -> list[sp.Symbol]:
        """List of parameter symbols."""
        return list(self.parameters.ref_map.keys())

    @property
    def parameter_values(self) -> Dict[sp.Symbol, float]:
        """Mapping of parameter symbols to default values."""
        return self.parameters.default_values

    @property
    def constant_names(self) -> list[str]:
        """List of constant symbol names."""
        return list(self.constants.symbol_map.keys())

    @property
    def constant_symbols(self) -> list[sp.Symbol]:
        """List of constant symbols."""
        return list(self.constants.ref_map.keys())

    @property
    def constant_values(self) -> Dict[sp.Symbol, float]:
        """Mapping of constant symbols to default values."""
        return self.constants.default_values

    @property
    def observable_names(self) -> list[str]:
        """List of observable symbol names."""
        return list(self.observables.symbol_map.keys())

    @property
    def observable_symbols(self) -> list[sp.Symbol]:
        """List of observable symbols."""
        return list(self.observables.ref_map.keys())

    @property
    def driver_names(self) -> list[str]:
        """List of driver symbol names."""
        return list(self.drivers.symbol_map.keys())

    @property
    def driver_symbols(self) -> list[sp.Symbol]:
        """List of driver symbols."""
        return list(self.drivers.ref_map.keys())

    @property
    def dxdt_names(self) -> Iterable[str]:
        """List of ``dx/dt`` output symbol names."""
        return list(self.dxdt.symbol_map.keys())

    @property
    def dxdt_symbols(self) -> list[sp.Symbol]:
        """List of ``dx/dt`` output symbols."""
        return list(self.dxdt.ref_map.keys())

    @property
    def all_arrayrefs(self) -> dict[str, sp.Symbol]:
        """Dictionary of all indexed base references keyed by symbol."""
        return {
            **self.states.ref_map,
            **self.parameters.ref_map,
            **self.observables.ref_map,
            **self.drivers.ref_map,
            **self.dxdt.ref_map,
        }

    @property
    def all_symbols(self) -> dict[str, sp.Symbol]:
        """Dictionary of all scalar symbols keyed by name."""
        return {
            **self.states.symbol_map,
            **self.parameters.symbol_map,
            **self.constants.symbol_map,
            **self.observables.symbol_map,
            **self.drivers.symbol_map,
            **self.dxdt.symbol_map,
        }

    def __getitem__(self, item: sp.Symbol) -> sp.Symbol:
        """Return the indexed reference associated with ``item``."""
        return self.all_indices[item]
