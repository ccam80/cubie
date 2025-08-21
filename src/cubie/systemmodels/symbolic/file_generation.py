"""Utilities for generating Python functions from SymPy code."""

from pathlib import Path
from importlib import util
from os import getcwd

cwd = getcwd()
GENERATED_DIR = Path(cwd) / "generated"

HEADER = ("# This file was generated automatically by Cubie. Don't make "
          "changes in here - they'll just be overwritten! Instead, modify "
          "the sympy input which you used to define the file.\n"
          "from numba import cuda\n"
          "from math import *\n\n\n")
#TODO: define allowed functions somewhere else.

class GeneratedFile:
    """Class for managing generated files."""
    def __init__(self, system_name):
        GENERATED_DIR.mkdir(exist_ok=True)
        self.file_path = GENERATED_DIR / f"{system_name}.py"
        self.prior_file_exists = self._init_file()

    def _init_file(self):
        if GENERATED_DIR.exists():
            file_path = self.file_path
            if not file_path.exists():
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(HEADER)
                    return False
            else:
                # TODO: Currently always overwrites. This branch is a
                #  placeholder for checking if the file's dxdt matches. (i.e.
                #  cache valid)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(HEADER)
                return True

    def _import_function(self, func_name):
        """ Import func_name from the generated file"""
        spec = util.spec_from_file_location(func_name, self.file_path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name)

    def generate_import(self, code_lines, func_name, template):
        """Codegen a function and import it."""
        self.generate_function(code_lines, func_name, template)
        return self._import_function(func_name)

    def generate_function(self, code_lines, func_name, template):
        body = "\n    ".join(code_lines)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(template.format(func_name=func_name, body=body))



