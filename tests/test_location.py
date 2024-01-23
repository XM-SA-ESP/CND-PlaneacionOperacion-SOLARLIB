import datetime
from unittest.mock import ANY

import numpy as np
from numpy import nan
import pandas as pd
from .conftest import assert_frame_equal, assert_index_equal

import pytest

import pytz
from pytz.exceptions import UnknownTimeZoneError

import xm_solarlib
from xm_solarlib.location import Location
from .conftest import requires_ephem

from xm_solarlib.constantes import TZ_AMERICA__PHOENIX, TZ_US_ARIZONA

def test_location_required():
    Location(32.2, -111)


def test_location_all():
    Location(32.2, -111, TZ_US_ARIZONA, 700, 'Tucson')


@pytest.mark.parametrize('tz', [
    pytz.timezone(TZ_US_ARIZONA), TZ_AMERICA__PHOENIX,  -7, -7.0,
    datetime.timezone.utc
])
def test_location_tz(tz):
    Location(32.2, -111, tz)


def test_location_invalid_tz():
    with pytest.raises(UnknownTimeZoneError):
        Location(32.2, -111, 'invalid')


def test_location_invalid_tz_type():
    with pytest.raises(TypeError):
        Location(32.2, -111, [5])

@pytest.fixture
def expected_location_str():
    return '\n'.join([
        'Location: ',
        '  name: Tucson',
        '  latitude: 32.2',
        '  longitude: -111',
        '  altitude: 700',
        '  tz: US/Arizona'
    ])


def test_location_print_all(expected_location_str):
    tus = Location(32.2, -111, TZ_US_ARIZONA, 700, 'Tucson')
    assert tus.__str__() == expected_location_str


def test_location_print_pytz(expected_location_str):
    tus = Location(32.2, -111, pytz.timezone(TZ_US_ARIZONA), 700, 'Tucson')
    assert tus.__str__() == expected_location_str


@pytest.fixture
def times():
    return pd.date_range(start='20160101T0600-0700',
                         end='20160101T1800-0700',
                         freq='3H')


def test_get_clearsky_ineichen_supply_linke(mocker):
    tus = Location(32.2, -111, TZ_US_ARIZONA, 700)
    times = pd.date_range(start='2014-06-24-0700', end='2014-06-25-0700',
                          freq='3h')
    mocker.spy(xm_solarlib.clearsky, 'ineichen')
    out = tus.get_clearsky(times, linke_turbidity=3)
    # we only care that the LT is passed in this test
    xm_solarlib.clearsky.ineichen.assert_called_once_with(ANY, ANY, 3, ANY, ANY)
    assert_index_equal(out.index, times)
    # check that values are 0 before sunrise and after sunset
    assert out.iloc[0:2, :].sum().sum() == 0
    assert out.iloc[-2:, :].sum().sum() == 0
    # check that values are > 0 during the day
    assert (out.iloc[2:-2, :] > 0).all().all()
    assert (out.columns.values == ['ghi', 'dni', 'dhi']).all()


def test_get_clearsky_haurwitz(times):
    tus = Location(32.2, -111, TZ_US_ARIZONA, 700, 'Tucson')
    clearsky = tus.get_clearsky(times, model='haurwitz')
    expected = pd.DataFrame(data=np.array(
                            [[   0.        ],
                             [ 242.30085588],
                             [ 559.38247117],
                             [ 384.6873791 ],
                             [   0.        ]]),
                            columns=['ghi'],
                            index=times)
    assert_frame_equal(expected, clearsky)



def test_get_clearsky_simplified_solis_dni_extra(times):
    tus = Location(32.2, -111, TZ_US_ARIZONA, 700, 'Tucson')
    clearsky = tus.get_clearsky(times, model='simplified_solis',
                                dni_extra=1370)
    expected = pd.DataFrame(data=np.
        array([[   0.        ,    0.        ,    0.        ],
               [  67.82281485,  618.15469596,  229.34422063],
               [  98.53217848,  825.98663808,  559.15039353],
               [  83.48619937,  732.45218243,  373.59500313],
               [   0.        ,    0.        ,    0.        ]]),
                            columns=['dhi', 'dni', 'ghi'],
                            index=times)
    expected = expected[['ghi', 'dni', 'dhi']]
    assert_frame_equal(expected, clearsky)


def test_get_solarposition(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 12, 30, 30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = golden_mst.get_solarposition(times, temperature=11)
    ephem_data = np.round(ephem_data, 3)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 3)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_location___repr__(expected_location_str):
    tus = Location(32.2, -111, TZ_US_ARIZONA, 700, 'Tucson')
    assert tus.__repr__() == expected_location_str


def test_extra_kwargs():
    with pytest.raises(TypeError, match='arbitrary_kwarg'):
        Location(32.2, -111, arbitrary_kwarg='value')   


def test_get_airmass_valueerror(times):
    tus = Location(32.2, -111, TZ_US_ARIZONA, 700, 'Tucson')
    with pytest.raises(ValueError):
        tus.get_airmass(times, model='invalid_model')