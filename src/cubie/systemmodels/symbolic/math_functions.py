# """Placeholder math functions for CUDA-safe symbolic expressions.
#
# Adapted from the `_math_functions.py` module in
# `chaste-codegen <https://github.com/ModellingWebLab/chaste-codegen>`_ (MIT licence).
# """
#
# from sympy import Function, cos, exp, log, sin, sqrt
#
#
# class RealFunction(Function):
#     def _eval_is_real(self):
#         return self.args[0].is_real
#
#
# class exp_(RealFunction):
#     def fdiff(self, argindex=1):
#         assert argindex == 1
#         return self
#
#
# class sin_(RealFunction):
#     def fdiff(self, argindex=1):
#         assert argindex == 1
#         return cos_(self.args[0])
#
#
# class cos_(RealFunction):
#     def fdiff(self, argindex=1):
#         assert argindex == 1
#         return -sin_(self.args[0])
#
#
# class sqrt_(RealFunction):
#     def fdiff(self, argindex=1):
#         assert argindex == 1
#         return 1 / (2 * sqrt_(self.args[0]))
#
#
# class log_(RealFunction):
#     def fdiff(self, argindex=1):
#         assert argindex == 1
#         return 1 / self.args[0]
#
#
# MATH_FUNC_SYMPY_MAPPING = {
#     exp_: exp,
#     sin_: sin,
#     cos_: cos,
#     sqrt_: sqrt,
#     log_: log,
# }
#
#
# def subs_math_func_placeholders(expr):
#     """Replace placeholder functions in *expr* with SymPy equivalents."""
#     for placeholder, sym_func in MATH_FUNC_SYMPY_MAPPING.items():
#         expr = expr.replace(placeholder, sym_func)
#     return expr