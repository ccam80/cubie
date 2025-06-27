import pytest
import numpy as np
from copy import deepcopy
# from CuMC.SystemModels.SystemValues import SystemValues
from CuMC._utils import update_dicts_from_kwargs

_DICT_UPDATE_TESTS = [
    ({"c": 3}, True, None, "update single item" ),
    ({"a": 5}, True, "found in multiple dictionaries", "key in multiple dicts"),
    ({"z": 10}, False, "not found", "key not found"),
    ({"a": 7, "c": 2.5}, True, None, "update across multiple dicts"),
    ({"d": 0.5}, False, None, "save as existing"),
    # Unclean inputs and realistic user-error edge cases
    ({"c": None}, True, None, "None value"),
    ({"a": [1, 2, 3]}, True, None, "list as value"),
    ({"b": {"nested": "dict"}}, True, None, "dict as value"),
    ({}, False, None, "empty kwargs"),
    ({" a ": 10}, False, "not found", "key with spaces"),
    ({"a-b": 15}, False, "not found", "key with special chars"),
]

def test_dict_update_single():
    """Test that a single key-value pair can be updated in a dictionary."""
    d = {'a': 1, 'b': 2}
    update_dicts_from_kwargs(d, a=3)
    assert d['a'] == 3, "Single key update failed"

    with pytest.warns(UserWarning):
        update_dicts_from_kwargs(d, c=5, b=4)
        assert d['b'] == 4, "Single key update failed after bogus key attempted"
@pytest.mark.parametrize("kwargs, expected_modified, expected_warning, test_name",
                         _DICT_UPDATE_TESTS,
                         ids=[config[3] for config in _DICT_UPDATE_TESTS])
def test_dict_updates_all(kwargs, expected_modified, expected_warning, test_name):
    INTDICT = dict(a=1,
                   b=2)
    NON_NUM_DICT = dict(a='test',
                        b=np.asarray)
    FLOAT_DICT = dict(c=1.2e30,
                      d=0.5)
    dicts = [INTDICT, NON_NUM_DICT, FLOAT_DICT]

    original_values = {
        dict_idx: {k: v for k, v in d.items() if k in kwargs}
        for dict_idx, d in enumerate(dicts)
    }

    if expected_warning:
        with pytest.warns(UserWarning, match=expected_warning):
            result = update_dicts_from_kwargs(dicts, **kwargs)
    else:
        result = update_dicts_from_kwargs(dicts, **kwargs)

    assert result == expected_modified, f"Expected modified={expected_modified}, got {result}"

    #Test that updates were successful, and no other values were changed
    for dict_idx, d in enumerate(dicts):
        for key, new_value in kwargs.items():
            if key in d:
                if original_values[dict_idx][key] != new_value:
                    assert d[key] == new_value, f"Dict {dict_idx}, key {key} not updated correctly"
                else:
                    assert d[key] == original_values[dict_idx][
                        key], f"Dict {dict_idx}, key {key} changed unexpectedly"
