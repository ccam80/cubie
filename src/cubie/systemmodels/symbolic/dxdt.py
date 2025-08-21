from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cubie.systemmodels.symbolic.file_generation import GeneratedFile

DXDT_TEMPLATE = (
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

def generate_dxdt_function(code_lines, file: "GeneratedFile"):
    return file.generate_import(code_lines,
                               "_build_symbolic_dxdt",
                               DXDT_TEMPLATE)