"""
Integration algorithm implementations and registry.

This module provides concrete implementations of numerical integration
algorithms for CUDA-based ODE solving, along with a registry system
for accessing available algorithms. It includes the base algorithm
framework and specific implementations like the Euler method.
"""

from attrs import define

from cubie.integrators.algorithms.euler import Euler
from cubie.integrators.algorithms.genericIntegratorAlgorithm import (
    GenericIntegratorAlgorithm,
)


@define
class _ImplementedAlgorithms:
    """
    Container for implemented integrator algorithms.

    This class provides a registry of available integration algorithms
    that can be accessed by name. It supports both attribute and
    dictionary-style access to algorithms.

    Attributes
    ----------
    euler : Euler
        Euler integration algorithm implementation.
    generic : GenericIntegratorAlgorithm
        Base generic integration algorithm class.
    """

    euler = Euler
    generic = GenericIntegratorAlgorithm

    def __getitem__(self, item):
        """
        Allow access to algorithms by name.

        Parameters
        ----------
        item : str
            Name of the algorithm to retrieve.

        Returns
        -------
        class
            The algorithm class corresponding to the given name.

        Raises
        ------
        KeyError
            If the algorithm name is not implemented.
        """
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(f"Algorithm '{item}' is not implemented.")


ImplementedAlgorithms = _ImplementedAlgorithms()

__all__ = ["Euler", "GenericIntegratorAlgorithm", "ImplementedAlgorithms"]
