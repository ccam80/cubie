"""Chained-preconditioner composition factory.

Implicit algorithms may configure two preconditioners applied as
``P1(P0(v))``. The chain is an ordinary owned CUDA factory: the two
concrete helper callables arrive as ``eq=False`` derived settings, the
chained kinds and signature variant are semantic settings, and the
composed device function rebuilds whenever either input callable is
replaced. Algorithm policy — resolving ``preconditioner_type`` into
concrete helper kinds — lives with the owning implicit step, not with
the ODE system.

Published Classes
-----------------
:class:`PreconditionerChainConfig`
    Compile settings for the composed closure.
:class:`PreconditionerChain`
    Factory emitting the chained preconditioner device function.

See Also
--------
:class:`~cubie.integrators.algorithms.ode_implicitstep.ODEImplicitStep`
    Owns a chain instance when two preconditioners are configured.
"""

from typing import Callable, Optional

from attrs import define, field, frozen, validators

from cubie.CUDAFactory import (
    CUDADispatcherCache,
    CUDAFactory,
    CUDAFactoryConfig,
)
from cubie.cuda_simsafe import cuda


@frozen
class PreconditionerChainConfig(CUDAFactoryConfig):
    """Compile settings for a two-stage preconditioner chain.

    Attributes
    ----------
    kinds
        Concrete helper kind names composed by the chain, in
        application order. Semantic identity of the composition.
    cached
        Whether the chained signature carries a ``cached_aux``
        argument (Rosenbrock-W's cached helper family).
    p0
        First preconditioner device function. Derived setting.
    p1
        Second preconditioner device function. Derived setting.
    """

    kinds: tuple = field(
        default=(),
        converter=tuple,
    )
    cached: bool = field(
        default=False, validator=validators.instance_of(bool)
    )
    p0: Optional[Callable] = field(default=None, eq=False)
    p1: Optional[Callable] = field(default=None, eq=False)


@define
class PreconditionerChainCache(CUDADispatcherCache):
    """Hold the chained preconditioner device function."""

    device_function: Callable = field()


class PreconditionerChain(CUDAFactory):
    """Compile a device function chaining two preconditioners.

    Notes
    -----
    The chained signature carries a trailing ``chain_scratch`` buffer
    in addition to the standard ``scratch``: ``scratch`` holds the
    intermediate P0 result, so P0 borrows ``out`` (dead until P1
    writes it) as its scratch slot and P1 receives ``chain_scratch``.
    Every buffer each stage sees is therefore distinct, so chained
    preconditioners may freely use their scratch arguments. The
    consuming linear solver allocates ``chain_scratch`` from the
    buffer registry when ``preconditioner_is_chained`` is set — the
    solver materialises the argument, so the buffer stays with the
    solver.
    """

    def __init__(self, precision, jit_flags=None) -> None:
        super().__init__()
        kwargs = {}
        if jit_flags is not None:
            kwargs["jit_flags"] = jit_flags
        self.setup_compile_settings(
            PreconditionerChainConfig(precision=precision, **kwargs)
        )

    def build(self) -> PreconditionerChainCache:
        """Compile the chained preconditioner for the current inputs."""
        config = self.compile_settings
        p0 = config.p0
        p1 = config.p1
        jit_kwargs = self.jit_kwargs

        # no cover: start
        if config.cached:

            @cuda.jit(device=True, inline=True, **jit_kwargs)
            def chained(
                state, parameters, drivers, cached_aux, base_state,
                t, h, a_ij, v, out, jvp, scratch, chain_scratch,
            ):
                p0(
                    state, parameters, drivers, cached_aux,
                    base_state, t, h, a_ij,
                    v, scratch, jvp, out,
                )
                p1(
                    state, parameters, drivers, cached_aux,
                    base_state, t, h, a_ij,
                    scratch, out, jvp, chain_scratch,
                )
        else:

            @cuda.jit(device=True, inline=True, **jit_kwargs)
            def chained(
                state, parameters, drivers, base_state,
                t, h, a_ij, v, out, jvp, scratch, chain_scratch,
            ):
                p0(
                    state, parameters, drivers, base_state,
                    t, h, a_ij, v, scratch, jvp, out,
                )
                p1(
                    state, parameters, drivers, base_state,
                    t, h, a_ij, scratch, out, jvp, chain_scratch,
                )
        # no cover: end
        return PreconditionerChainCache(device_function=chained)
