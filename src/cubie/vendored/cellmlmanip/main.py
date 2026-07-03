"""Main entry-points and functions for interacting with the cellmlmanip library."""
from .parser import Parser


def load_model(path, unit_store=None):
    """Parses a CellML file and returns a :class:`cellmlmanip.model.Model`.

    :param unit_store: Optional :class:`cellmlmanip.units.UnitStore` instance; if given the model will share the
        underlying registry so that conversions between model units and those from the provided store work.
    """
    return Parser(path).parse(unit_store=unit_store)
