"""Utilities for generating Python functions from SymPy code."""

from pathlib import Path
from importlib import util
from uuid import uuid4

GENERATED_DIR = Path(__file__).with_name("generated")
GENERATED_DIR.mkdir(exist_ok=True)

DXDT_TEMPLATE = (
    "from numba import cuda\n"
    "def {func_name}(constants, precision):\n"
    "    \"\"\"Auto-generated dxdt factory.\"\"\"\n"    
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def dxdt(state, parameters, driver, observables, dxdt):\n"
    "    {body}\n"
    "    \n"
    "    return dxdt\n"
)

JACOBIAN_TEMPLATE = (
    "\n\n\ndef {func_name}(constants, precision):\n"
    "    \"\"\"Auto-generated Jacobian factory.\"\"\"\n"
    
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "             inline=True)\n"
    "    def jac_v(state, parameters, driver, Jv):\n"
    "        {body}\n"
    "    \n"
    "    return jac_v\n"
)


def _import_function(file_path, func_name):
    spec = util.spec_from_file_location(func_name, file_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def generate_function(code_lines, func_name, template):
    body = "\n    ".join(code_lines)
    file_name = f"generated_system_{uuid4().hex[0:6]}.py"
    # TODO: store generated in user directory, cache
    file_path = GENERATED_DIR / file_name
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(template.format(func_name=func_name, body=body))
    return _import_function(file_path, func_name)

def generate_dxdt_function(code_lines, template=DXDT_TEMPLATE):
    return generate_function(code_lines, "_build_symbolic_dxdt", template)

def generate_jacobian_function(code_lines, template=JACOBIAN_TEMPLATE):
    return generate_function(code_lines, "_build_symbolic_jacobian", template)
