import datetime
from collections import OrderedDict
import warnings

import numpy as np
from numpy import array, nan
import pandas as pd

import pytest
from numpy.testing import (assert_almost_equal,
                           assert_allclose)
from xm_solarlib import irradiance

from .conftest import (
    assert_frame_equal,
    assert_series_equal,
    requires_ephem,
    requires_numba
)

from xm_solarlib.constantes import TZ_AMERICA__PHOENIX, TZ_US_ARIZONA


# fixtures create realistic test input data
# test input data generated at Location(32.2, -111, 'US/Arizona', 700)
# test input data is hard coded to avoid dependencies on other parts of pvlib


@pytest.fixture
def times():
    # must include night values
    return pd.date_range(start='20140624', freq='6H', periods=4,
                         tz=TZ_US_ARIZONA)


@pytest.fixture
def irrad_data(times):
    return pd.DataFrame(np.array(
        [[0.,    0.,    0.],
         [79.73860422,  316.1949056,   40.46149818],
         [1042.48031487,  939.95469881,  118.45831879],
         [257.20751138,  646.22886049,   62.03376265]]),
        columns=['ghi', 'dni', 'dhi'], index=times)


@pytest.fixture
def ephem_data(times):
    return pd.DataFrame(np.array(
        [[124.0390863, 124.0390863, -34.0390863, -34.0390863,
          352.69550699,  -2.36677158],
         [82.85457044,  82.97705621,   7.14542956,   7.02294379,
          66.71410338,  -2.42072165],
         [10.56413562,  10.56725766,  79.43586438,  79.43274234,
          144.76567754,  -2.47457321],
         [72.41687122,  72.46903556,  17.58312878,  17.53096444,
          287.04104128,  -2.52831909]]),
        columns=['apparent_zenith', 'zenith', 'apparent_elevation',
                 'elevation', 'azimuth', 'equation_of_time'],
        index=times)


@pytest.fixture
def dni_et(times):
    return np.array(
        [1321.1655834833093, 1321.1655834833093, 1321.1655834833093,
         1321.1655834833093])


@pytest.fixture
def relative_airmass(times):
    return pd.Series([np.nan, 7.58831596, 1.01688136, 3.27930443], times)


# setup for et rad test. put it here for readability
timestamp = pd.Timestamp('20161026')
dt_index = pd.DatetimeIndex([timestamp])
doy = timestamp.dayofyear
dt_date = timestamp.date()
dt_datetime = datetime.datetime.combine(dt_date, datetime.time(0))
dt_np64 = np.datetime64(dt_datetime)
value = 1383.636203


@pytest.mark.parametrize('testval, expected', [
    (doy, value),
    (np.float64(doy), value),
    (dt_date, value),
    (dt_datetime, value),
    (dt_np64, value),
    (np.array([doy]), np.array([value])),
    (pd.Series([doy]), np.array([value])),
    (dt_index, pd.Series([value], index=dt_index)),
    (timestamp, value)
])
def test_get_extra_radiation(testval, expected):
    out = irradiance.get_extra_radiation(testval)
    assert_allclose(out, expected, atol=10)


def test_get_ground_diffuse_simple_float():
    result = irradiance.get_ground_diffuse(40, 900)
    assert_allclose(result, 26.32000014911496)


def test_get_ground_diffuse_simple_series(irrad_data):
    ground_irrad = irradiance.get_ground_diffuse(40, irrad_data['ghi'])
    assert ground_irrad.name == 'diffuse_ground'


def test_get_ground_diffuse_albedo_0(irrad_data):
    ground_irrad = irradiance.get_ground_diffuse(
        40, irrad_data['ghi'], albedo=0)
    assert (0 == ground_irrad).all()


def test_get_ground_diffuse_albedo_series(times):
    albedo = pd.Series(0.2, index=times)
    ground_irrad = irradiance.get_ground_diffuse(
        45, pd.Series(1000, index=times), albedo)
    expected = albedo * 0.5 * (1 - np.sqrt(2) / 2.) * 1000
    expected.name = 'diffuse_ground'
    assert_series_equal(ground_irrad, expected)


def test_isotropic_float():
    result = irradiance.isotropic(40, 100)
    assert_allclose(result, 88.30222215594891)


def test_isotropic_series(irrad_data):
    result = irradiance.isotropic(40, irrad_data['dhi'])
    assert_allclose(result, [0, 35.728402, 104.601328, 54.777191], atol=1e-4)


def test_klucher_series_float():
    # klucher inputs
    surface_tilt, surface_azimuth = 40.0, 180.0
    dhi, ghi = 100.0, 900.0
    solar_zenith, solar_azimuth = 20.0, 180.0
    # expect same result for floats and pd.Series
    expected = irradiance.klucher(
        surface_tilt, surface_azimuth,
        pd.Series(dhi), pd.Series(ghi),
        pd.Series(solar_zenith), pd.Series(solar_azimuth)
    )  # 94.99429931664851
    result = irradiance.klucher(
        surface_tilt, surface_azimuth, dhi, ghi, solar_zenith, solar_azimuth
    )
    assert_allclose(result, expected[0])


def test_klucher_series(irrad_data, ephem_data):
    result = irradiance.klucher(40, 180, irrad_data['dhi'], irrad_data['ghi'],
                                ephem_data['apparent_zenith'],
                                ephem_data['azimuth'])
    # pvlib matlab 1.4 does not contain the max(cos_tt, 0) correction
    # so, these values are different
    assert_allclose(result, [0., 36.789794, 109.209347, 56.965916], atol=1e-4)
    # expect same result for np.array and pd.Series
    expected = irradiance.klucher(
        40, 180, irrad_data['dhi'].values, irrad_data['ghi'].values,
        ephem_data['apparent_zenith'].values, ephem_data['azimuth'].values
    )
    assert_allclose(result, expected, atol=1e-4)


def test_haydavies(irrad_data, ephem_data, dni_et):
    result = irradiance.haydavies(
        40, 180, irrad_data['dhi'], irrad_data['dni'], dni_et,
        ephem_data['apparent_zenith'], ephem_data['azimuth'])
    # values from matlab 1.4 code
    assert_allclose(result, [0, 27.1775, 102.9949, 33.1909], atol=1e-4)


def test_haydavies_components(irrad_data, ephem_data, dni_et):
    expected = pd.DataFrame(np.array(
        [[0, 27.1775, 102.9949, 33.1909],
         [0, 27.1775, 30.1818, 27.9837],
         [0, 0, 72.8130, 5.2071],
         [0, 0, 0, 0]]).T,
        columns=['sky_diffuse', 'isotropic', 'circumsolar', 'horizon'],
        index=irrad_data.index
    )
    # pandas
    result = irradiance.haydavies(
        40, 180, irrad_data['dhi'], irrad_data['dni'], dni_et,
        ephem_data['apparent_zenith'], ephem_data['azimuth'],
        return_components=True)
    assert_frame_equal(result, expected, check_less_precise=4)
    # numpy
    result = irradiance.haydavies(
        40, 180, irrad_data['dhi'].values, irrad_data['dni'].values, dni_et,
        ephem_data['apparent_zenith'].values, ephem_data['azimuth'].values,
        return_components=True)
    assert_allclose(result['sky_diffuse'], expected['sky_diffuse'], atol=1e-4)
    assert_allclose(result['isotropic'], expected['isotropic'], atol=1e-4)
    assert_allclose(result['circumsolar'], expected['circumsolar'], atol=1e-4)
    assert_allclose(result['horizon'], expected['horizon'], atol=1e-4)
    assert isinstance(result, dict)
    # scalar
    result = irradiance.haydavies(
        40, 180, irrad_data['dhi'].values[-1], irrad_data['dni'].values[-1],
        dni_et[-1], ephem_data['apparent_zenith'].values[-1],
        ephem_data['azimuth'].values[-1], return_components=True)
    assert_allclose(result['sky_diffuse'], expected['sky_diffuse'].iloc[-1],
                    atol=1e-4)
    assert_allclose(result['isotropic'], expected['isotropic'].iloc[-1],
                    atol=1e-4)
    assert_allclose(result['circumsolar'], expected['circumsolar'].iloc[-1],
                    atol=1e-4)
    assert_allclose(result['horizon'], expected['horizon'].iloc[-1], atol=1e-4)
    assert isinstance(result, dict)


def test_reindl(irrad_data, ephem_data, dni_et):
    result = irradiance.reindl(
        40, 180, irrad_data['dhi'], irrad_data['dni'], irrad_data['ghi'],
        dni_et, ephem_data['apparent_zenith'], ephem_data['azimuth'])
    # values from matlab 1.4 code
    assert_allclose(result, [0., 27.9412, 104.1317, 34.1663], atol=1e-4)


def test_king(irrad_data, ephem_data):
    result = irradiance.king(40, irrad_data['dhi'], irrad_data['ghi'],
                             ephem_data['apparent_zenith'])
    assert_allclose(result, [0, 44.629352, 115.182626, 79.719855], atol=1e-4)


def test_perez(irrad_data, ephem_data, dni_et, relative_airmass):
    dni = irrad_data['dni'].copy()
    dni.iloc[2] = np.nan
    out = irradiance.perez(40, 180, irrad_data['dhi'], dni,
                           dni_et, ephem_data['apparent_zenith'],
                           ephem_data['azimuth'], relative_airmass)
    expected = pd.Series(np.array(
        [0.,   31.46046871,  np.nan,   45.45539877]),
        index=irrad_data.index)
    assert_series_equal(out, expected, check_less_precise=2)


def test_perez_components(irrad_data, ephem_data, dni_et, relative_airmass):
    dni = irrad_data['dni'].copy()
    dni.iloc[2] = np.nan
    out = irradiance.perez(40, 180, irrad_data['dhi'], dni,
                           dni_et, ephem_data['apparent_zenith'],
                           ephem_data['azimuth'], relative_airmass,
                           return_components=True)
    expected = pd.DataFrame(np.array(
        [[0.,   31.46046871,  np.nan,   45.45539877],
         [0.,  26.84138589,          np.nan,  31.72696071],
         [0.,  0.,         np.nan,  4.47966439],
         [0.,  4.62212181,         np.nan,  9.25316454]]).T,
        columns=['sky_diffuse', 'isotropic', 'circumsolar', 'horizon'],
        index=irrad_data.index
    )
    expected_for_sum = expected['sky_diffuse'].copy()
    expected_for_sum.iloc[2] = 0
    sum_components = out.iloc[:, 1:].sum(axis=1)
    sum_components.name = 'sky_diffuse'

    assert_frame_equal(out, expected, check_less_precise=2)
    assert_series_equal(sum_components, expected_for_sum, check_less_precise=2)


def test_perez_negative_horizon():
    times = pd.date_range(start='20190101 11:30:00', freq='1H',
                          periods=5, tz='US/Central')

    # Avoid test dependencies on functionality not being tested by hard-coding
    # the inputs. This data corresponds to Goodwin Creek in the afternoon on
    # 1/1/2019.
    # dni_e is slightly rounded from irradiance.get_extra_radiation
    # airmass from atmosphere.get_relative_airmas
    inputs = pd.DataFrame(np.array(
        [[158,         19,          1,          0,          0],
         [249,        165,        136,         93,         50],
         [57.746951,  57.564205,  60.813841,  66.989435,  75.353368],
         [171.003315, 187.346924, 202.974357, 216.725599, 228.317233],
         [1414,       1414,       1414,       1414,       1414],
         [1.869315,   1.859981,   2.044429,   2.544943,   3.900136]]).T,
        columns=['dni', 'dhi', 'solar_zenith',
                 'solar_azimuth', 'dni_extra', 'airmass'],
        index=times
    )

    out = irradiance.perez(34, 180, inputs['dhi'], inputs['dni'],
                           inputs['dni_extra'], inputs['solar_zenith'],
                           inputs['solar_azimuth'], inputs['airmass'],
                           model='allsitescomposite1990',
                           return_components=True)

    # sky_diffuse can be less than isotropic under certain conditions as
    # horizon goes negative
    expected = pd.DataFrame(np.array(
        [[281.410185, 152.20879, 123.867898, 82.836412, 43.517015],
         [166.785419, 142.24475, 119.173875, 83.525150, 45.725931],
         [113.548755,  16.09757,   9.956174,  3.142467,  0],
         [1.076010,  -6.13353,  -5.262151, -3.831230, -2.208923]]).T,
        columns=['sky_diffuse', 'isotropic', 'circumsolar', 'horizon'],
        index=times
    )

    expected_for_sum = expected['sky_diffuse'].copy()
    sum_components = out.iloc[:, 1:].sum(axis=1)
    sum_components.name = 'sky_diffuse'

    assert_frame_equal(out, expected, check_less_precise=2)
    assert_series_equal(sum_components, expected_for_sum, check_less_precise=2)


def test_perez_arrays(irrad_data, ephem_data, dni_et, relative_airmass):
    dni = irrad_data['dni'].copy()
    dni.iloc[2] = np.nan
    out = irradiance.perez(40, 180, irrad_data['dhi'].values, dni.values,
                           dni_et, ephem_data['apparent_zenith'].values,
                           ephem_data['azimuth'].values,
                           relative_airmass.values)
    expected = np.array(
        [0.,   31.46046871,  np.nan,   45.45539877])
    assert_allclose(out, expected, atol=1e-2)
    assert isinstance(out, np.ndarray)


def test_perez_scalar():
    # copied values from fixtures
    out = irradiance.perez(40, 180, 118.45831879, 939.95469881,
                           1321.1655834833093, 10.56413562, 144.76567754,
                           1.01688136)
    # this will fail. out is ndarry with ndim == 0. fix in future version.
    # assert np.isscalar(out)
    assert_allclose(out, 109.084332)


@pytest.mark.parametrize('model', ['isotropic', 'haydavies', 'perez'])
def test_sky_diffuse_zenith_close_to_90(model):
    # GH 432
    sky_diffuse = irradiance.get_sky_diffuse(
        30, 180, 89.999, 230,
        dni=10, dhi=50, dni_extra=1360, airmass=12, model=model)
    assert sky_diffuse < 100


def test_get_sky_diffuse_model_invalid():
    with pytest.raises(ValueError):
        irradiance.get_sky_diffuse(
            30, 180, 0, 180, 1000, 100, dni_extra=1360, airmass=1,
            model='invalid')


def test_get_sky_diffuse_missing_dni_extra():
    msg = 'dni_extra is required'
    with pytest.raises(ValueError, match=msg):
        irradiance.get_sky_diffuse(
            30, 180, 0, 180, 1000, 100, airmass=1,
            model='haydavies')


def test_get_sky_diffuse_missing_airmass(irrad_data, ephem_data, dni_et):
    # test assumes location is Tucson, AZ
    # calculated airmass should be the equivalent to fixture airmass
    dni = irrad_data['dni'].copy()
    dni.iloc[2] = np.nan
    out = irradiance.get_sky_diffuse(
        40, 180, ephem_data['apparent_zenith'], ephem_data['azimuth'], dni
        , irrad_data['dhi'], dni_et,  model='perez')
    expected = pd.Series(np.array(
        [0., 31.46046871, np.nan, 45.45539877]),
        index=irrad_data.index)
    assert_series_equal(out, expected, check_less_precise=2)


def test_get_total_irradiance(irrad_data, ephem_data, dni_et,
                              relative_airmass):
    models = ['isotropic', 'haydavies', 'perez']

    for model in models:
        total = irradiance.get_total_irradiance(
            32, 180,
            ephem_data['apparent_zenith'], ephem_data['azimuth'],
            dni=irrad_data['dni'], ghi=irrad_data['ghi'],
            dhi=irrad_data['dhi'],
            dni_extra=dni_et, airmass=relative_airmass,
            model=model,
            surface_type='urban')

        assert total.columns.tolist() == ['poa_global', 'poa_direct',
                                          'poa_diffuse', 'poa_sky_diffuse',
                                          'poa_ground_diffuse']


@pytest.mark.parametrize('model', ['isotropic', 'haydavies',
                                   'perez'])
def test_get_total_irradiance_albedo(
        irrad_data, ephem_data, dni_et, relative_airmass, model):
    albedo = pd.Series(0.2, index=ephem_data.index)
    total = irradiance.get_total_irradiance(
        32, 180,
        ephem_data['apparent_zenith'], ephem_data['azimuth'],
        dni=irrad_data['dni'], ghi=irrad_data['ghi'],
        dhi=irrad_data['dhi'],
        dni_extra=dni_et, airmass=relative_airmass,
        model=model,
        albedo=albedo)

    assert total.columns.tolist() == ['poa_global', 'poa_direct',
                                      'poa_diffuse', 'poa_sky_diffuse',
                                      'poa_ground_diffuse']


@pytest.mark.parametrize('model', ['isotropic', 'haydavies',
                                   'perez'])
def test_get_total_irradiance_scalars(model):
    total = irradiance.get_total_irradiance(
        32, 180,
        10, 180,
        dni=1000, ghi=1100,
        dhi=100,
        dni_extra=1400, airmass=1,
        model=model,
        surface_type='urban')

    assert list(total.keys()) == ['poa_global', 'poa_direct',
                                  'poa_diffuse', 'poa_sky_diffuse',
                                  'poa_ground_diffuse']
    # test that none of the values are nan
    assert np.isnan(np.array(list(total.values()))).sum() == 0


def test_get_total_irradiance_missing_dni_extra():
    msg = 'dni_extra is required'
    with pytest.raises(ValueError, match=msg):
        irradiance.get_total_irradiance(
            32, 180,
            10, 180,
            dni=1000, ghi=1100,
            dhi=100,
            model='haydavies')


def test_get_total_irradiance_missing_airmass():
    total = irradiance.get_total_irradiance(
        32, 180,
        10, 180,
        dni=1000, ghi=1100,
        dhi=100,
        dni_extra=1400,
        model='perez')
    assert list(total.keys()) == ['poa_global', 'poa_direct',
                                  'poa_diffuse', 'poa_sky_diffuse',
                                  'poa_ground_diffuse']

@pytest.mark.parametrize('pressure,expected', [
    (93193,  [[830.46567,   0.79742,   0.93505],
              [676.09497,   0.63776,   3.02102]]),
    (None,   [[868.72425,   0.79742,   1.01664],
              [680.66679,   0.63776,   3.28463]]),
    (101325, [[868.72425,   0.79742,   1.01664],
              [680.66679,   0.63776,   3.28463]])
])
def test_disc_value(pressure, expected):
    # see GH 449 for pressure=None vs. 101325.
    columns = ['dni', 'kt', 'airmass']
    times = pd.DatetimeIndex(['2014-06-24T1200', '2014-06-24T1800'],
                             tz=TZ_AMERICA__PHOENIX)
    ghi = pd.Series([1038.62, 254.53], index=times)
    zenith = pd.Series([10.567, 72.469], index=times)
    out = irradiance.disc(ghi, zenith, times, pressure=pressure)
    expected_values = np.array(expected)
    expected = pd.DataFrame(expected_values, columns=columns, index=times)
    # check the pandas dataframe. check_less_precise is weird
    assert_frame_equal(out, expected, check_less_precise=True)
    # use np.assert_allclose to check values more clearly
    assert_allclose(out.values, expected_values, atol=1e-5)


def test_disc_overirradiance():
    columns = ['dni', 'kt', 'airmass']
    ghi = np.array([3000])
    solar_zenith = np.full_like(ghi, 0)
    times = pd.date_range(start='2016-07-19 12:00:00', freq='1s',
                          periods=len(ghi), tz=TZ_AMERICA__PHOENIX)
    out = irradiance.disc(ghi=ghi, solar_zenith=solar_zenith,
                          datetime_or_doy=times)
    expected = pd.DataFrame(np.array(
        [[8.72544336e+02, 1.00000000e+00, 9.99493933e-01]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)


def test_disc_min_cos_zenith_max_zenith():
    # map out behavior under difficult conditions with various
    # limiting kwargs settings
    columns = ['dni', 'kt', 'airmass']
    times = pd.DatetimeIndex(['2016-07-19 06:11:00'], tz=TZ_AMERICA__PHOENIX)
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times)
    expected = pd.DataFrame(np.array(
        [[0.00000000e+00, 1.16046346e-02, 12.0]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # max_zenith and/or max_airmass keep these results reasonable
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times,
                          min_cos_zenith=0)
    expected = pd.DataFrame(np.array(
        [[0.00000000e+00, 1.0, 12.0]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # still get reasonable values because of max_airmass=12 limit
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times,
                          max_zenith=100)
    expected = pd.DataFrame(np.array(
        [[0., 1.16046346e-02, 12.0]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # still get reasonable values because of max_airmass=12 limit
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times,
                          min_cos_zenith=0, max_zenith=100)
    expected = pd.DataFrame(np.array(
        [[277.50185968, 1.0, 12.0]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # max_zenith keeps this result reasonable
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times,
                          min_cos_zenith=0, max_airmass=100)
    expected = pd.DataFrame(np.array(
        [[0.00000000e+00, 1.0, 36.39544757]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # allow zenith to be close to 90 and airmass to be infinite
    # and we get crazy values
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times,
                          max_zenith=100, max_airmass=100)
    expected = pd.DataFrame(np.array(
        [[6.68577449e+03, 1.16046346e-02, 3.63954476e+01]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # allow min cos zenith to be 0, zenith to be close to 90,
    # and airmass to be very big and we get even higher DNI values
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times,
                          min_cos_zenith=0, max_zenith=100, max_airmass=100)
    expected = pd.DataFrame(np.array(
        [[7.21238390e+03, 1., 3.63954476e+01]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)



@pytest.mark.parametrize(
    'surface_tilt,surface_azimuth,solar_zenith,' +
    'solar_azimuth,aoi_expected,aoi_proj_expected',
    [(0, 0, 0, 0, 0, 1),
     (30, 180, 30, 180, 0, 1),
     (30, 180, 150, 0, 180, -1),
     (90, 0, 30, 60, 75.5224878, 0.25),
     (90, 0, 30, 170, 119.4987042, -0.4924038)])
def test_aoi_and_aoi_projection(surface_tilt, surface_azimuth, solar_zenith,
                                solar_azimuth, aoi_expected,
                                aoi_proj_expected):
    aoi = irradiance.aoi(surface_tilt, surface_azimuth, solar_zenith,
                         solar_azimuth)
    assert_allclose(aoi, aoi_expected, atol=1e-5)

    aoi_projection = irradiance.aoi_projection(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    assert_allclose(aoi_projection, aoi_proj_expected, atol=1e-6)


def test_aoi_projection_precision():
    # GH 1185 -- test that aoi_projection does not exceed 1.0, and when
    # given identical inputs, the returned projection is very close to 1.0

    # scalars
    zenith = 89.26778228223463
    azimuth = 60.932028605997004
    projection = irradiance.aoi_projection(zenith, azimuth, zenith, azimuth)
    assert projection <= 1
    assert np.isclose(projection, 1)

    # arrays
    zeniths = np.array([zenith])
    azimuths = np.array([azimuth])
    projections = irradiance.aoi_projection(zeniths, azimuths,
                                            zeniths, azimuths)
    assert all(projections <= 1)
    assert all(np.isclose(projections, 1))
    assert projections.dtype == np.dtype('float64')


@pytest.fixture
def airmass_kt():
    # disc algorithm stopped at am=12. test am > 12 for out of range behavior
    return np.array([1, 5, 12, 20])


def test_clearness_index():
    ghi = np.array([-1, 0, 1, 1000])
    solar_zenith = np.array([180, 90, 89.999, 0])
    ghi, solar_zenith = np.meshgrid(ghi, solar_zenith)
    # default min_cos_zenith
    out = irradiance.clearness_index(ghi, solar_zenith, 1370)
    # np.set_printoptions(precision=3, floatmode='maxprec', suppress=True)
    expected = np.array(
        [[0., 0., 0.011, 2.],
         [0., 0., 0.011, 2.],
         [0., 0., 0.011, 2.],
         [0., 0., 0.001, 0.73]])
    assert_allclose(out, expected, atol=0.001)
    # specify min_cos_zenith
    with np.errstate(invalid='ignore', divide='ignore'):
        out = irradiance.clearness_index(ghi, solar_zenith, 1400,
                                         min_cos_zenith=0)
    expected = np.array(
        [[0.,   nan, 2., 2.],
         [0., 0., 2., 2.],
         [0., 0., 2., 2.],
         [0., 0., 0.001, 0.714]])
    assert_allclose(out, expected, atol=0.001)
    # specify max_clearness_index
    out = irradiance.clearness_index(ghi, solar_zenith, 1370,
                                     max_clearness_index=0.82)
    expected = np.array(
        [[0.,  0.,  0.011,  0.82],
         [0.,  0.,  0.011,  0.82],
         [0.,  0.,  0.011,  0.82],
         [0.,  0.,  0.001,  0.73]])
    assert_allclose(out, expected, atol=0.001)
    # specify min_cos_zenith and max_clearness_index
    with np.errstate(invalid='ignore', divide='ignore'):
        out = irradiance.clearness_index(ghi, solar_zenith, 1400,
                                         min_cos_zenith=0,
                                         max_clearness_index=0.82)
    expected = np.array(
        [[0.,    nan,  0.82,  0.82],
         [0.,  0.,  0.82,  0.82],
         [0.,  0.,  0.82,  0.82],
         [0.,  0.,  0.001,  0.714]])
    assert_allclose(out, expected, atol=0.001)
    # scalars
    out = irradiance.clearness_index(1000, 10, 1400)
    expected = 0.725
    assert_allclose(out, expected, atol=0.001)
    # series
    times = pd.date_range(start='20180601', periods=2, freq='12H')
    ghi = pd.Series([0, 1000], index=times)
    solar_zenith = pd.Series([90, 0], index=times)
    extra_radiation = pd.Series([1360, 1400], index=times)
    out = irradiance.clearness_index(ghi, solar_zenith, extra_radiation)
    expected = pd.Series([0, 0.714285714286], index=times)
    assert_series_equal(out, expected)