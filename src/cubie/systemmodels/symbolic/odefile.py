"""Utilities for generating Python functions from SymPy code."""

from importlib import util
from os import getcwd
from pathlib import Path
from typing import Iterable, Tuple

import sympy as sp

from cubie.systemmodels.symbolic.dxdt import (
    DXDT_TEMPLATE,
    generate_dxdt_fac_code,
)
from cubie.systemmodels.symbolic.jacobian import (
    JVP_TEMPLATE,
    VJP_TEMPLATE,
    generate_jvp_code,
    generate_vjp_code,
)
from cubie.systemmodels.symbolic.parser import IndexedBases

DXDT_MATCHLINE = DXDT_TEMPLATE.splitlines()[1]
JVP_MATCHLINE = JVP_TEMPLATE.splitlines()[1]
VJP_MATCHLINE = VJP_TEMPLATE.splitlines()[1]

cwd = getcwd()
GENERATED_DIR = Path(cwd) / "generated"

HEADER = ("\n# This file was generated automatically by Cubie. Don't make "
          "changes in here - they'll just be overwritten! Instead, modify "
          "the sympy input which you used to define the file.\n"
          "from numba import cuda\n"
          "\n\n\n")

class ODEFile:
    """Class for managing generated files."""
    def __init__(self, system_name, fn_hash):
        GENERATED_DIR.mkdir(exist_ok=True)
        self.file_path = GENERATED_DIR / f"{system_name}.py"
        self._init_file(fn_hash)

    def _init_file(self, fn_hash):
        if not self.cached_file_valid(fn_hash):
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(f"#{fn_hash}")
                f.write("\n")
                f.write(HEADER)
            return True
        else:
            return False

    @property
    def _dxdt_generated(self):
        """Returns True if a dxdt function exists in this file."""
        if DXDT_MATCHLINE in self.file_path.read_text():
            return True
        return False

    @property
    def _jvp_generated(self):
        """Returns True if a JVP function exists in this file."""
        if JVP_MATCHLINE in self.file_path.read_text():
            return True
        return False

    @property
    def _vjp_generated(self):
        """Returns True if a VJP function exists in this file."""
        if VJP_MATCHLINE in self.file_path.read_text():
            return True
        return False

    def cached_file_valid(self, fn_hash):
        if self.file_path.exists():
            with open(self.file_path, "r", encoding="utf-8") as f:
                existing_hash = f.readline().strip().lstrip("#")
                if existing_hash == fn_hash:
                    return True
        return False

    def generate_dxdt_fac(self,
                          equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                          index_map: IndexedBases,
                          cse = True):
        if not self._dxdt_generated:
            func_name = "dxdt_factory"
            code = generate_dxdt_fac_code(equations,index_map,
                                          func_name,
                                          cse=cse)
            self.add_function(code, func_name)

    def get_dxdt_fac(self,
                     equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                     index_map: IndexedBases,
                     cse = True):
        self.generate_dxdt_fac(equations, index_map, cse=cse)
        return self._import_function("dxdt_factory")

    def generate_jvp_fac(self,
                         equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                         index_map: IndexedBases,
                         cse = True):
        if not self._jvp_generated:
            func_name = "jvp_factory"
            code = generate_jvp_code(equations, index_map, func_name, cse=cse)
            self.add_function(code, func_name)

    def get_jvp_fac(self,
                     equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                     index_map: IndexedBases,
                     cse = True):
        self.generate_jvp_fac(equations, index_map, cse=cse)
        return self._import_function("jvp_factory")

    def generate_vjp_fac(self,
                         equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                         index_map: IndexedBases,
                         cse = True):
        if not self._vjp_generated:
            func_name = "vjp_factory"
            code = generate_vjp_code(equations, index_map, func_name, cse=cse)
            self.add_function(code, func_name)

    def get_vjp_fac(
        self,
        equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
        index_map: IndexedBases,
        cse=True,
    ):
        self.generate_vjp_fac(equations, index_map, cse=cse)
        return self._import_function("vjp_factory")

    def _import_function(self,
                         func_name):
        """ Import func_name from the generated file"""
        spec = util.spec_from_file_location(func_name, self.file_path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name)

    def get_factory(self, func_type, equations, index_map,
                    cse = True):
        if func_type == "dxdt":
            return self.get_dxdt_fac(equations, index_map, cse=cse)
        elif func_type == "jvp":
            return self.get_jvp_fac(equations, index_map, cse=cse)
        elif func_type == "vjp":
            return self.get_vjp_fac(equations, index_map, cse=cse)
        else:
            raise ValueError(f"Invalid function type: {func_type}")

    def generate_and_import(self, code_lines, func_name, template):
        """Codegen a function and import it."""
        if not self.cache_valid:
            self.add_function(code_lines, func_name)
        return self._import_function(func_name)

    def add_function(self,
                     printed_code: str,
                     func_name: str) -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(printed_code)
