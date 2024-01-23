import pytest
from xm_solarlib import tools
import numpy as np
import pandas as pd

@pytest.mark.parametrize('keys, input_dict, expected', [
    (['a', 'b'], {'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2}),
    (['a', 'b', 'd'], {'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2}),
    (['a'], {}, {}),
    (['a'], {'b': 2}, {})
])
def test_build_kwargs(keys, input_dict, expected):
    kwargs = tools._build_kwargs(keys, input_dict)
    assert kwargs == expected


def _obj_test_golden_sect(params, loc):
    return params[loc] * (1. - params['c'] * params[loc]**params['n'])


@pytest.mark.parametrize('params, lb, ub, expected, func', [
    ({'c': 1., 'n': 1.}, 0., 1., 0.5, _obj_test_golden_sect),
    ({'c': 1e6, 'n': 6.}, 0., 1., 0.07230200263994839, _obj_test_golden_sect),
    ({'c': 0.2, 'n': 0.3}, 0., 100., 89.14332727531685, _obj_test_golden_sect)
])
def test__golden_sect_dataframe(params, lb, ub, expected, func):
    v, x = tools._golden_sect_dataframe(params, lb, ub, func)
    assert np.isclose(x, expected, atol=1e-8)


def test__golden_sect_dataframe_atol():
    params = {'c': 0.2, 'n': 0.3}
    expected = 89.14332727531685
    v, x = tools._golden_sect_dataframe(
        params, 0., 100., _obj_test_golden_sect, atol=1e-12)
    assert np.isclose(x, expected, atol=1e-12)


def test__golden_sect_dataframe_vector():
    params = {'c': np.array([1., 2.]), 'n': np.array([1., 1.])}
    lower = np.array([0., 0.001])
    upper = np.array([1.1, 1.2])
    expected = np.array([0.5, 0.25])
    v, x = tools._golden_sect_dataframe(params, lower, upper,
                                        _obj_test_golden_sect)
    assert np.allclose(x, expected, atol=1e-8)
    # some upper and lower bounds equal
    params = {'c': np.array([1., 2., 1.]), 'n': np.array([1., 1., 1.])}
    lower = np.array([0., 0.001, 1.])
    upper = np.array([1., 1.2, 1.])
    expected = np.array([0.5, 0.25, 1.0])  # x values for maxima
    v, x = tools._golden_sect_dataframe(params, lower, upper,
                                        _obj_test_golden_sect)
    assert np.allclose(x, expected, atol=1e-8)
    # all upper and lower bounds equal, arrays of length 1
    params = {'c': np.array([1.]), 'n': np.array([1.])}
    lower = np.array([1.])
    upper = np.array([1.])
    expected = np.array([1.])  # x values for maxima
    v, x = tools._golden_sect_dataframe(params, lower, upper,
                                        _obj_test_golden_sect)
    assert np.allclose(x, expected, atol=1e-8)


def test__golden_sect_dataframe_nans():
    # nan in bounds
    params = {'c': np.array([1., 2., 1.]), 'n': np.array([1., 1., 1.])}
    lower = np.array([0., 0.001, np.nan])
    upper = np.array([1.1, 1.2, 1.])
    expected = np.array([0.5, 0.25, np.nan])
    v, x = tools._golden_sect_dataframe(params, lower, upper,
                                        _obj_test_golden_sect)
    assert np.allclose(x, expected, atol=1e-8, equal_nan=True)
    # nan in function values
    params = {'c': np.array([1., 2., np.nan]), 'n': np.array([1., 1., 1.])}
    lower = np.array([0., 0.001, 0.])
    upper = np.array([1.1, 1.2, 1.])
    expected = np.array([0.5, 0.25, np.nan])
    v, x = tools._golden_sect_dataframe(params, lower, upper,
                                        _obj_test_golden_sect)
    assert np.allclose(x, expected, atol=1e-8, equal_nan=True)
    # all nan in bounds
    params = {'c': np.array([1., 2., 1.]), 'n': np.array([1., 1., 1.])}
    lower = np.array([np.nan, np.nan, np.nan])
    upper = np.array([1.1, 1.2, 1.])
    expected = np.array([np.nan, np.nan, np.nan])
    v, x = tools._golden_sect_dataframe(params, lower, upper,
                                        _obj_test_golden_sect)
    assert np.allclose(x, expected, atol=1e-8, equal_nan=True)


def test_degrees_to_index_1():
    """Test that _degrees_to_index raises an error when something other than
    'latitude' or 'longitude' is passed."""
    with pytest.raises(IndexError):  # invalid value for coordinate argument
        tools._degrees_to_index(degrees=22.0, coordinate='width')

def test_degrees_to_index_latitude():
    # Prueba para 'latitude' con valores dentro del rango
    degrees = 45.0
    index = tools._degrees_to_index(degrees, 'latitude')
    assert index == 540  # El índice esperado para 45 grados de latitud

def test_degrees_to_index_longitude():
    # Prueba para 'longitude' con valores dentro del rango
    degrees = -90.0
    index = tools._degrees_to_index(degrees, 'longitude')
    assert index == 1080  # El índice esperado para -90 grados de longitud

def test_degrees_to_index_out_of_range():
    # Prueba para valores fuera del rango esperando IndexError
    with pytest.raises(IndexError):
        tools._degrees_to_index(200.0, 'latitude')

    with pytest.raises(IndexError):
        tools._degrees_to_index(-200.0, 'longitude')

def test_degrees_to_index_invalid_coordinate():
    # Prueba para un tipo de coordenada inválido esperando IndexError
    with pytest.raises(IndexError):
        tools._degrees_to_index(45.0, 'invalid_coordinate')


@pytest.mark.parametrize('args, args_idx', [
    # no pandas.Series or pandas.DataFrame args
    ((1,), None),
    (([1],), None),
    ((np.array(1),), None),
    ((np.array([1]),), None),
    # has pandas.Series or pandas.DataFrame args
    ((pd.DataFrame([1], index=[1]),), 0),
    ((pd.Series([1], index=[1]),), 0),
    ((1, pd.Series([1], index=[1]),), 1),
    ((1, pd.DataFrame([1], index=[1]),), 1),
    # first pandas.Series or pandas.DataFrame is used
    ((1, pd.Series([1], index=[1]), pd.DataFrame([2], index=[2]),), 1),
    ((1, pd.DataFrame([1], index=[1]), pd.Series([2], index=[2]),), 1),
])
def test_get_pandas_index(args, args_idx):
    index = tools.get_pandas_index(*args)

    if args_idx is None:
        assert index is None
    else:
        pd.testing.assert_index_equal(args[args_idx].index, index)


def test_build_args_missing_key():
    keys = ['key1', 'key2', 'key3']
    input_dict = {'key1': 'value1', 'key2': 'value2', 'key4': 'value4'}
    dict_name = 'test_dict'

    with pytest.raises(KeyError) as exc_info:
        tools._build_args(keys, input_dict, dict_name)

    assert str(exc_info.value) == "\"Missing required parameter 'key3'. Found {'key1': 'value1', 'key2': 'value2', 'key4': 'value4'} in test_dict.\""

def test_build_kwargs():
    keys = ['key1', 'key2', 'key3']
    input_dict = {'key1': 'value1', 'key2': 'value2', 'key4': 'value4'}

    result = tools._build_kwargs(keys, input_dict)

    assert result == {'key1': 'value1', 'key2': 'value2'}