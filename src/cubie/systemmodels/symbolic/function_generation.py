"""Utilities for generating Python functions from SymPy code."""

from pathlib import Path
from importlib import util
from uuid import uuid4

GENERATED_DIR = Path(__file__).with_name("generated")
GENERATED_DIR.mkdir(exist_ok=True)

DXDT_TEMPLATE = (
    "def {func_name}(state, parameters, driver, observables, dxdt):\n"
    "    {body}\n"
)

JACOBIAN_TEMPLATE = (
    "def {func_name}(state, parameters, driver, observables, J):\n"
    "    {body}\n"
)

def _import_function(file_path, func_name):
    spec = util.spec_from_file_location(func_name, file_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)

def generate_function(code_lines, func_name, template):
    body = "\n    ".join(code_lines)
    file_name = f"{func_name}_{uuid4().hex}.py"
    file_path = GENERATED_DIR / file_name
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(template.format(func_name=func_name, body=body))
    return _import_function(file_path, func_name)

def generate_dxdt_function(code_lines, template=DXDT_TEMPLATE):
    return generate_function(code_lines, "sympy_dxdt", template)

def generate_jacobian_function(code_lines, template=JACOBIAN_TEMPLATE):
    return generate_function(code_lines, "sympy_jacobian", template)
