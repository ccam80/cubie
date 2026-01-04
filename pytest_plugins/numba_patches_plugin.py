"""Pytest plugin applying early Numba monkeypatches.

The goal is to patch Numba before test modules (and project modules) import it.
This file must stay lightweight: no Numba imports at module scope.
"""

import threading
from contextlib import contextmanager
from typing import Any, List

def _apply_numba_patches() -> None:
    """Apply monkeypatches to Numba CUDA components."""
    import copy

    from numba.cuda.core import ir, ir_utils
    from numba.cuda.core.ir_utils import (
        add_offset_to_labels,
        find_topo_order,
        next_label,
        replace_vars,
        simplify_CFG,
    )
    from numba.cuda.core.inline_closurecall import (
        _add_definitions,
        _created_inlined_var_name,
        _debug_dump,
        _get_all_scopes,
        _get_callee_args,
        _replace_args_with,
        _replace_returns,
    )

    def inline_ir_patched(
        self,
        caller_ir,
        block,
        i,
        callee_ir,
        callee_freevars,
        arg_typs=None,
        preserve_ir=True,
    ):
        """Inline `callee_ir` into `caller_ir` without persisting IR copies."""
        callee_ir_original = callee_ir

        if preserve_ir:

            def copy_ir(the_ir):
                kernel_copy = the_ir.copy()
                kernel_copy.blocks = {}
                for block_label in the_ir.blocks:
                    new_block = copy.deepcopy(the_ir.blocks[block_label])
                    kernel_copy.blocks[block_label] = new_block
                return kernel_copy

            callee_ir = copy_ir(callee_ir)

        if self.validator is not None:
            self.validator(callee_ir)

        scope = block.scope
        instr = block.body[i]
        call_expr = instr.value
        callee_blocks = callee_ir.blocks

        max_label = max(
            ir_utils._the_max_label.next(),
            max(caller_ir.blocks.keys()),
        )
        callee_blocks = add_offset_to_labels(callee_blocks, max_label + 1)
        callee_blocks = simplify_CFG(callee_blocks)
        callee_ir.blocks = callee_blocks

        min_label = min(callee_blocks.keys())
        max_label = max(callee_blocks.keys())
        ir_utils._the_max_label.update(max_label)

        self.debug_print("After relabel")
        _debug_dump(callee_ir)

        callee_scopes = _get_all_scopes(callee_blocks)
        self.debug_print("callee_scopes = ", callee_scopes)
        assert len(callee_scopes) == 1
        callee_scope = callee_scopes[0]

        var_dict = {}
        for var in tuple(callee_scope.localvars._con.values()):
            if var.name not in callee_freevars:
                inlined_name = _created_inlined_var_name(
                    callee_ir.func_id.unique_name,
                    var.name,
                )
                new_var = scope.redefine(inlined_name, loc=var.loc)
                callee_scope.redefine(inlined_name, loc=var.loc)
                var_dict[var.name] = new_var

        self.debug_print("var_dict = ", var_dict)
        replace_vars(callee_blocks, var_dict)

        self.debug_print("After local var rename")
        _debug_dump(callee_ir)

        callee_func = callee_ir.func_id.func
        args = _get_callee_args(call_expr, callee_func, instr.loc, caller_ir)

        if self._permit_update_type_and_call_maps:
            if arg_typs is None:
                raise TypeError("arg_typs should have a value not None")
            self.update_type_and_call_maps(callee_ir, arg_typs)
            callee_blocks = callee_ir.blocks

        self.debug_print("After arguments rename: ")
        _debug_dump(callee_ir)

        _replace_args_with(callee_blocks, args)

        new_block = ir.Block(scope, block.loc)
        new_block.body = block.body[i + 1 :]
        new_label = next_label()
        caller_ir.blocks[new_label] = new_block

        block.body = block.body[:i]
        block.body.append(ir.Jump(min_label, instr.loc))

        topo_order = find_topo_order(callee_blocks)
        _replace_returns(callee_blocks, instr.target, new_label)

        if (
            instr.target.name in caller_ir._definitions
            and call_expr in caller_ir._definitions[instr.target.name]
        ):
            caller_ir._definitions[instr.target.name].remove(call_expr)

        new_blocks = [(new_label, new_block)]
        for label in topo_order:
            callee_block = callee_blocks[label]
            callee_block.scope = scope
            _add_definitions(caller_ir, callee_block)
            caller_ir.blocks[label] = callee_block
            new_blocks.append((label, callee_block))

        self.debug_print("After merge in")
        _debug_dump(caller_ir)

        return callee_ir_original, callee_blocks, var_dict, new_blocks

    def inline_function_patched(
        self,
        caller_ir,
        block,
        i,
        function,
        arg_typs=None,
    ):
        """Inline a Python function without preserving callee IR."""
        callee_ir = self.run_untyped_passes(function)
        freevars = function.__code__.co_freevars
        return self.inline_ir(
            caller_ir,
            block,
            i,
            callee_ir,
            freevars,
            arg_typs=arg_typs,
            preserve_ir=False,
        )

    swap_state_lock = threading.Lock()
    swap_state = {}

    @contextmanager
    def swapped_cuda_module_patched(fn, fake_cuda_module):
        """Swap the cuda` module reference in a thread-safe manner."""
        from numba import cuda

        fn_globs = fn.__globals__
        globals_id = id(fn_globs)

        with swap_state_lock:
            keys_to_swap = [k for k, v in fn_globs.items() if v is cuda]
            module_swaps = swap_state.get(globals_id)
            if module_swaps is None:
                swap_state[globals_id] = (keys_to_swap, 0)
                for name in keys_to_swap:
                    fn_globs[name] = fake_cuda_module
            else:
                swapped_keys, refcount = module_swaps
                swap_state[globals_id] = (swapped_keys, refcount + 1)

        try:
            yield
        finally:
            with swap_state_lock:
                swapped_keys, refcount = swap_state.get(globals_id)
                if refcount >= 1:
                    swap_state[globals_id] = (swapped_keys, refcount - 1)
                else:
                    for name in swapped_keys:
                        fn_globs[name] = cuda
                    del swap_state[globals_id]

    import numba.cuda.core.inline_closurecall as inline_closurecall
    inline_closurecall.InlineWorker.inline_ir = inline_ir_patched
    inline_closurecall.InlineWorker.inline_function = inline_function_patched

    import numba.cuda.simulator.kernelapi as kernelapi
    kernelapi.swapped_cuda_module = swapped_cuda_module_patched


def pytest_load_initial_conftests(
    early_config: Any,
    parser: Any,
    args: List[str],
) -> None:
    """Apply patches at pytest startup."""

    _apply_numba_patches()