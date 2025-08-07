from attrs import define

from cubie.integrators.algorithms.euler import Euler
from cubie.integrators.algorithms.genericIntegratorAlgorithm import \
    GenericIntegratorAlgorithm


@define
class _ImplementedAlgorithms:
    """Container for implemented integrator algorithms."""

    euler = Euler
    generic = GenericIntegratorAlgorithm

    def __getitem__(self, item):
        """Allow access to algorithms by name."""
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(f"Algorithm '{item}' is not implemented.")


ImplementedAlgorithms = _ImplementedAlgorithms()

__all__ = ['Euler', 'GenericIntegratorAlgorithm', 'ImplementedAlgorithms']
