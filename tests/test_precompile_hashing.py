"""Tests for the precompile plugin's function-identity hashing."""

from tests._precompile_hashing import _function_key


def test_function_identity_includes_closure_values():
    """Function keys distinguish serialized closure constants."""
    def factory(value):
        def kernel(argument=1):
            return argument + value

        return kernel

    assert _function_key(factory(1)) != _function_key(factory(2))


def test_function_identity_ignores_generated_source_location():
    """Generated source location does not affect function identity."""
    source = (
        "def kernel(value):\n"
        "    def inner(argument):\n"
        "        return argument + 1\n"
        "    return inner(value)\n"
    )

    def compile_kernel(source_text, line_offset):
        namespace = {"__name__": "generated_cache_test"}
        code = compile(
            "\n" * line_offset + source_text,
            "generated_helpers.py",
            "exec",
        )
        exec(code, namespace)
        return namespace["kernel"]

    first = compile_kernel(source, 0)
    relocated = compile_kernel(source, 20)
    changed = compile_kernel(source.replace("+ 1", "+ 2"), 0)

    assert first.__code__.co_firstlineno != relocated.__code__.co_firstlineno
    assert _function_key(first) == _function_key(relocated)
    assert _function_key(first) != _function_key(changed)
