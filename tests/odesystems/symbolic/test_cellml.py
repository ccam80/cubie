import pytest

from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model


def test_cellml_import_error():
    """Missing dependency raises ImportError."""

    with pytest.raises(ImportError):
        load_cellml_model("dummy.cellml")

