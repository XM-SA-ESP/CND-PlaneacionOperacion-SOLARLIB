import calendar
import datetime
import warnings

import numpy as np
import pandas as pd

from .conftest import assert_frame_equal, assert_series_equal
import pytest

from xm_solarlib.location import Location
from xm_solarlib import solarposition, spa

from .conftest import requires_ephem, requires_spa_c, requires_numba

from xm_solarlib.constantes import TZ_US_ARIZONA, SOL_HORIZONTE


# setup times and locations to be tested.
times = pd.date_range(start=datetime.datetime(2014, 6, 24),
                      end=datetime.datetime(2014, 6, 26), freq='15Min')

tus = Location(32.2, -111, TZ_US_ARIZONA, 700)  # no DST issues possible
times_localized = times.tz_localize(tus.tz)

tol = 5


@pytest.fixture()
def expected_solpos_multi():
    return pd.DataFrame({'elevation': [39.872046, 39.505196],
                         'apparent_zenith': [50.111622, 50.478260],
                         'azimuth': [194.340241, 194.311132],
                         'apparent_elevation': [39.888378, 39.521740]},
                        index=['2003-10-17T12:30:30Z', '2003-10-18T12:30:30Z'])


@pytest.fixture()
def expected_rise_set_spa():
    # for Golden, CO, from NREL SPA website
    times = pd.DatetimeIndex([datetime.datetime(2015, 1, 2),
                              datetime.datetime(2015, 8, 2),
                              ]).tz_localize('MST')
    sunrise = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 7, 21, 55),
                                datetime.datetime(2015, 8, 2, 5, 0, 27)
                                ]).tz_localize('MST').tolist()
    sunset = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 16, 47, 43),
                               datetime.datetime(2015, 8, 2, 19, 13, 58)
                               ]).tz_localize('MST').tolist()
    transit = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 12, 4, 45),
                                datetime.datetime(2015, 8, 2, 12, 6, 58)
                                ]).tz_localize('MST').tolist()
    return pd.DataFrame({'sunrise': sunrise,
                         'sunset': sunset,
                         'transit': transit},
                        index=times)


@pytest.fixture()
def expected_rise_set_ephem():
    # for Golden, CO, from USNO websites
    times = pd.DatetimeIndex([datetime.datetime(2015, 1, 1),
                              datetime.datetime(2015, 1, 2),
                              datetime.datetime(2015, 1, 3),
                              datetime.datetime(2015, 8, 2),
                              ]).tz_localize('MST')
    sunrise = pd.DatetimeIndex([datetime.datetime(2015, 1, 1, 7, 22, 0),
                                datetime.datetime(2015, 1, 2, 7, 22, 0),
                                datetime.datetime(2015, 1, 3, 7, 22, 0),
                                datetime.datetime(2015, 8, 2, 5, 0, 0)
                                ]).tz_localize('MST').tolist()
    sunset = pd.DatetimeIndex([datetime.datetime(2015, 1, 1, 16, 47, 0),
                               datetime.datetime(2015, 1, 2, 16, 48, 0),
                               datetime.datetime(2015, 1, 3, 16, 49, 0),
                               datetime.datetime(2015, 8, 2, 19, 13, 0)
                               ]).tz_localize('MST').tolist()
    transit = pd.DatetimeIndex([datetime.datetime(2015, 1, 1, 12, 4, 0),
                                datetime.datetime(2015, 1, 2, 12, 5, 0),
                                datetime.datetime(2015, 1, 3, 12, 5, 0),
                                datetime.datetime(2015, 8, 2, 12, 7, 0)
                                ]).tz_localize('MST').tolist()
    return pd.DataFrame({'sunrise': sunrise,
                         'sunset': sunset,
                         'transit': transit},
                        index=times)


# the physical tests are run at the same time as the NREL SPA test.
# pyephem reproduces the NREL result to 2 decimal places.
# this doesn't mean that one code is better than the other.

@requires_spa_c
def test_spa_c_physical(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 12, 30, 30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.spa_c(times, golden_mst.latitude,
                                     golden_mst.longitude,
                                     pressure=82000,
                                     temperature=11)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


@requires_spa_c
def test_spa_c_physical_dst(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.spa_c(times, golden.latitude,
                                     golden.longitude,
                                     pressure=82000,
                                     temperature=11)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_spa_python_numpy_physical(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 12, 30, 30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.spa_python(times, golden_mst.latitude,
                                          golden_mst.longitude,
                                          pressure=82000,
                                          temperature=11, delta_t=67,
                                          atmos_refract=0.5667,
                                          how='numpy')
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_spa_python_numpy_physical_dst(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.spa_python(times, golden.latitude,
                                          golden.longitude,
                                          pressure=82000,
                                          temperature=11, delta_t=67,
                                          atmos_refract=0.5667,
                                          how='numpy')
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_sun_rise_set_transit_spa(expected_rise_set_spa, golden):
    # solution from NREL SAP web calculator
    south = Location(-35.0, 0.0, tz='UTC')
    times = pd.DatetimeIndex([datetime.datetime(1996, 7, 5, 0),
                              datetime.datetime(2004, 12, 4, 0)]
                             ).tz_localize('UTC')
    sunrise = pd.DatetimeIndex([datetime.datetime(1996, 7, 5, 7, 8, 15),
                                datetime.datetime(2004, 12, 4, 4, 38, 57)]
                               ).tz_localize('UTC').tolist()
    sunset = pd.DatetimeIndex([datetime.datetime(1996, 7, 5, 17, 1, 4),
                               datetime.datetime(2004, 12, 4, 19, 2, 3)]
                              ).tz_localize('UTC').tolist()
    transit = pd.DatetimeIndex([datetime.datetime(1996, 7, 5, 12, 4, 36),
                                datetime.datetime(2004, 12, 4, 11, 50, 22)]
                               ).tz_localize('UTC').tolist()
    frame = pd.DataFrame({'sunrise': sunrise,
                          'sunset': sunset,
                          'transit': transit}, index=times)

    result = solarposition.sun_rise_set_transit_spa(times, south.latitude,
                                                    south.longitude,
                                                    delta_t=65.0)
    result_rounded = pd.DataFrame(index=result.index)
    # need to iterate because to_datetime does not accept 2D data
    # the rounding fails on pandas < 0.17
    for col, data in result.items():
        result_rounded[col] = data.dt.round('1s')

    assert_frame_equal(frame, result_rounded)

    # test for Golden, CO compare to NREL SPA
    result = solarposition.sun_rise_set_transit_spa(
        expected_rise_set_spa.index, golden.latitude, golden.longitude,
        delta_t=65.0)

    # round to nearest minute
    result_rounded = pd.DataFrame(index=result.index)
    # need to iterate because to_datetime does not accept 2D data
    for col, data in result.items():
        result_rounded[col] = data.dt.round('s').tz_convert('MST')

    assert_frame_equal(expected_rise_set_spa, result_rounded)


@requires_ephem
def test_sun_rise_set_transit_ephem(expected_rise_set_ephem, golden):
    # test for Golden, CO compare to USNO, using local midnight
    result = solarposition.sun_rise_set_transit_ephem(
        expected_rise_set_ephem.index, golden.latitude, golden.longitude,
        next_or_previous='next', altitude=golden.altitude, pressure=0,
        temperature=11, horizon=SOL_HORIZONTE)
    # round to nearest minute
    result_rounded = pd.DataFrame(index=result.index)
    for col, data in result.items():
        result_rounded[col] = data.dt.round('min').tz_convert('MST')
    assert_frame_equal(expected_rise_set_ephem, result_rounded)

    # test next sunrise/sunset with times
    times = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 3, 0, 0),
                              datetime.datetime(2015, 1, 2, 10, 15, 0),
                              datetime.datetime(2015, 1, 2, 15, 3, 0),
                              datetime.datetime(2015, 1, 2, 21, 6, 7)
                              ]).tz_localize('MST')
    expected = pd.DataFrame(index=times,
                            columns=['sunrise', 'sunset'],
                            dtype='datetime64[ns]')
    idx_sunrise = pd.to_datetime(['2015-01-02', '2015-01-03', '2015-01-03',
                                  '2015-01-03']).tz_localize('MST')
    expected['sunrise'] = \
        expected_rise_set_ephem.loc[idx_sunrise, 'sunrise'].tolist()
    idx_sunset = pd.to_datetime(['2015-01-02', '2015-01-02', '2015-01-02',
                                 '2015-01-03']).tz_localize('MST')
    expected['sunset'] = \
        expected_rise_set_ephem.loc[idx_sunset, 'sunset'].tolist()
    idx_transit = pd.to_datetime(['2015-01-02', '2015-01-02', '2015-01-03',
                                  '2015-01-03']).tz_localize('MST')
    expected['transit'] = \
        expected_rise_set_ephem.loc[idx_transit, 'transit'].tolist()

    result = solarposition.sun_rise_set_transit_ephem(times,
                                                      golden.latitude,
                                                      golden.longitude,
                                                      next_or_previous='next',
                                                      altitude=golden.altitude,
                                                      pressure=0,
                                                      temperature=11,
                                                      horizon=SOL_HORIZONTE)
    # round to nearest minute
    result_rounded = pd.DataFrame(index=result.index)
    for col, data in result.items():
        result_rounded[col] = data.dt.round('min').tz_convert('MST')
    assert_frame_equal(expected, result_rounded)

    # test previous sunrise/sunset with times
    times = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 3, 0, 0),
                              datetime.datetime(2015, 1, 2, 10, 15, 0),
                              datetime.datetime(2015, 1, 3, 3, 0, 0),
                              datetime.datetime(2015, 1, 3, 13, 6, 7)
                              ]).tz_localize('MST')
    expected = pd.DataFrame(index=times,
                            columns=['sunrise', 'sunset'],
                            dtype='datetime64[ns]')
    idx_sunrise = pd.to_datetime(['2015-01-01', '2015-01-02', '2015-01-02',
                                  '2015-01-03']).tz_localize('MST')
    expected['sunrise'] = \
        expected_rise_set_ephem.loc[idx_sunrise, 'sunrise'].tolist()
    idx_sunset = pd.to_datetime(['2015-01-01', '2015-01-01', '2015-01-02',
                                 '2015-01-02']).tz_localize('MST')
    expected['sunset'] = \
        expected_rise_set_ephem.loc[idx_sunset, 'sunset'].tolist()
    idx_transit = pd.to_datetime(['2015-01-01', '2015-01-01', '2015-01-02',
                                  '2015-01-03']).tz_localize('MST')
    expected['transit'] = \
        expected_rise_set_ephem.loc[idx_transit, 'transit'].tolist()

    result = solarposition.sun_rise_set_transit_ephem(
        times,
        golden.latitude, golden.longitude, next_or_previous='previous',
        altitude=golden.altitude, pressure=0, temperature=11, horizon=SOL_HORIZONTE)
    # round to nearest minute
    result_rounded = pd.DataFrame(index=result.index)
    for col, data in result.items():
        result_rounded[col] = data.dt.round('min').tz_convert('MST')
    assert_frame_equal(expected, result_rounded)

    # test with different timezone
    times = times.tz_convert('UTC')
    expected = expected.tz_convert('UTC')  # resuse result from previous
    for col, data in expected.items():
        expected[col] = data.dt.tz_convert('UTC')
    result = solarposition.sun_rise_set_transit_ephem(
        times,
        golden.latitude, golden.longitude, next_or_previous='previous',
        altitude=golden.altitude, pressure=0, temperature=11, horizon=SOL_HORIZONTE)
    # round to nearest minute
    result_rounded = pd.DataFrame(index=result.index)
    for col, data in result.items():
        result_rounded[col] = data.dt.round('min').tz_convert(times.tz)
    assert_frame_equal(expected, result_rounded)


@requires_ephem
def test_sun_rise_set_transit_ephem_error(expected_rise_set_ephem, golden):
    with pytest.raises(ValueError):
        solarposition.sun_rise_set_transit_ephem(expected_rise_set_ephem.index,
                                                 golden.latitude,
                                                 golden.longitude,
                                                 next_or_previous='other')
    tz_naive = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 3, 0, 0)])
    with pytest.raises(ValueError):
        solarposition.sun_rise_set_transit_ephem(tz_naive,
                                                 golden.latitude,
                                                 golden.longitude,
                                                 next_or_previous='next')


@requires_ephem
def test_sun_rise_set_transit_ephem_horizon(golden):
    times = pd.DatetimeIndex([datetime.datetime(2016, 1, 3, 0, 0, 0)
                              ]).tz_localize('MST')
    # center of sun disk
    center = solarposition.sun_rise_set_transit_ephem(
        times,
        latitude=golden.latitude, longitude=golden.longitude)
    edge = solarposition.sun_rise_set_transit_ephem(
        times,
        latitude=golden.latitude, longitude=golden.longitude, horizon=SOL_HORIZONTE)
    result_rounded = (edge['sunrise'] - center['sunrise']).dt.round('min')

    sunrise_delta = datetime.datetime(2016, 1, 3, 7, 17, 11) - \
        datetime.datetime(2016, 1, 3, 7, 21, 33)
    expected = pd.Series(index=times,
                         data=[sunrise_delta],
                         name='sunrise').dt.round('min')
    assert_series_equal(expected, result_rounded)


@requires_ephem
def test_pyephem_physical(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 12, 30, 30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.pyephem(times, golden_mst.latitude,
                                       golden_mst.longitude, pressure=82000,
                                       temperature=11)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos.round(2),
                       ephem_data[expected_solpos.columns].round(2))


@requires_ephem
def test_pyephem_physical_dst(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.pyephem(times, golden.latitude,
                                       golden.longitude, pressure=82000,
                                       temperature=11)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos.round(2),
                       ephem_data[expected_solpos.columns].round(2))


def test_ephemeris_physical(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 12, 30, 30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.ephemeris(times, golden_mst.latitude,
                                         golden_mst.longitude,
                                         pressure=82000,
                                         temperature=11)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_ephemeris_physical_dst(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.ephemeris(times, golden.latitude,
                                         golden.longitude, pressure=82000,
                                         temperature=11)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_ephemeris_physical_no_tz(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 19, 30, 30),
                          periods=1, freq='D')
    ephem_data = solarposition.ephemeris(times, golden_mst.latitude,
                                         golden_mst.longitude,
                                         pressure=82000,
                                         temperature=11)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_get_solarposition_error(golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    with pytest.raises(ValueError):
        solarposition.get_solarposition(times, golden.latitude,
                                        golden.longitude,
                                        pressure=82000,
                                        temperature=11,
                                        method='error this')


@pytest.mark.parametrize("pressure, expected", [
    (82000, 'expected_solpos'),
    (90000, pd.DataFrame(
        np.array([[39.88997,   50.11003,  194.34024,   39.87205,   14.64151,
                   50.12795]]),
        columns=['apparent_elevation', 'apparent_zenith', 'azimuth',
                 'elevation', 'equation_of_time', 'zenith'],
        index=['2003-10-17T12:30:30Z']))
    ])
def test_get_solarposition_pressure(
        pressure, expected, golden, expected_solpos):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude,
                                                 pressure=pressure,
                                                 temperature=11)
    if isinstance(expected, str) and expected == 'expected_solpos':
        expected = expected_solpos
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 5)
    ephem_data = np.round(ephem_data, 5)
    assert_frame_equal(this_expected, ephem_data[this_expected.columns])


@pytest.mark.parametrize("altitude, expected", [
    (1830.14, 'expected_solpos'),
    (2000, pd.DataFrame(
        np.array([[39.88788,   50.11212,  194.34024,   39.87205,   14.64151,
                   50.12795]]),
        columns=['apparent_elevation', 'apparent_zenith', 'azimuth',
                 'elevation', 'equation_of_time', 'zenith'],
        index=['2003-10-17T12:30:30Z']))
    ])
def test_get_solarposition_altitude(
        altitude, expected, golden, expected_solpos):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude,
                                                 altitude=altitude,
                                                 temperature=11)
    if isinstance(expected, str) and expected == 'expected_solpos':
        expected = expected_solpos
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 5)
    ephem_data = np.round(ephem_data, 5)
    assert_frame_equal(this_expected, ephem_data[this_expected.columns])


def test_get_solarposition_no_kwargs(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


@requires_ephem
def test_get_solarposition_method_pyephem(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude,
                                                 method='pyephem')
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


# put numba tests at end of file to minimize reloading

@requires_numba
def test_spa_python_numba_physical(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 12, 30, 30),
                          periods=1, freq='D', tz=golden_mst.tz)
    with warnings.catch_warnings():
        # don't warn on method reload or num threads
        # ensure that numpy is the most recently used method so that
        # we can use the warns filter below
        warnings.simplefilter("ignore")
        ephem_data = solarposition.spa_python(times, golden_mst.latitude,
                                              golden_mst.longitude,
                                              pressure=82000,
                                              temperature=11, delta_t=67,
                                              atmos_refract=0.5667,
                                              how='numpy', numthreads=1)
    with pytest.warns(UserWarning):
        ephem_data = solarposition.spa_python(times, golden_mst.latitude,
                                              golden_mst.longitude,
                                              pressure=82000,
                                              temperature=11, delta_t=67,
                                              atmos_refract=0.5667,
                                              how='numba', numthreads=1)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


@requires_numba
def test_spa_python_numba_physical_dst(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)

    with warnings.catch_warnings():
        # don't warn on method reload or num threads
        warnings.simplefilter("ignore")
        ephem_data = solarposition.spa_python(times, golden.latitude,
                                              golden.longitude, pressure=82000,
                                              temperature=11, delta_t=67,
                                              atmos_refract=0.5667,
                                              how='numba', numthreads=1)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])

    with pytest.warns(UserWarning):
        # test that we get a warning when reloading to use numpy only
        solarposition.spa_python(times, golden.latitude,
                                              golden.longitude,
                                              pressure=82000,
                                              temperature=11, delta_t=67,
                                              atmos_refract=0.5667,
                                              how='numpy', numthreads=1)