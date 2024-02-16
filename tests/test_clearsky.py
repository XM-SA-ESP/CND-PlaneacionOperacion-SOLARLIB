from collections import OrderedDict

import numpy as np
from numpy import nan
import pandas as pd
import pytz
from scipy.linalg import hankel

import pytest
from numpy.testing import assert_allclose
from .conftest import assert_frame_equal

from xm_solarlib import clearsky

from xm_solarlib.constantes import TZ_AMERICA__PHOENIX


def test_ineichen_series():
    times = pd.date_range(start='2014-06-24', end='2014-06-25', freq='3h',
                          tz=TZ_AMERICA__PHOENIX)
    apparent_zenith = pd.Series(np.array(
        [124.0390863, 113.38779941, 82.85457044, 46.0467599, 10.56413562,
         34.86074109, 72.41687122, 105.69538659, 124.05614124]),
        index=times)
    am = pd.Series(np.array(
        [nan, nan, 6.97935524, 1.32355476, 0.93527685,
         1.12008114, 3.01614096, nan, nan]),
        index=times)
    expected = pd.DataFrame(np.
        array([[   0.        ,    0.        ,    0.        ],
               [   0.        ,    0.        ,    0.        ],
               [  65.49426624,  321.16092181,   25.54562017],
               [ 704.6968125 ,  888.90147035,   87.73601277],
               [1044.1230677 ,  953.24925854,  107.03109696],
               [ 853.02065704,  922.06124712,   96.42909484],
               [ 251.99427693,  655.44925241,   53.9901349 ],
               [   0.        ,    0.        ,    0.        ],
               [   0.        ,    0.        ,    0.        ]]),
                            columns=['ghi', 'dni', 'dhi'],
                            index=times)

    out = clearsky.ineichen(apparent_zenith, am, 3)
    assert_frame_equal(expected, out)


def test_ineichen_series_perez_enhancement():
    times = pd.date_range(start='2014-06-24', end='2014-06-25', freq='3h',
                          tz=TZ_AMERICA__PHOENIX)
    apparent_zenith = pd.Series(np.array(
        [124.0390863, 113.38779941, 82.85457044, 46.0467599, 10.56413562,
         34.86074109, 72.41687122, 105.69538659, 124.05614124]),
        index=times)
    am = pd.Series(np.array(
        [nan, nan, 6.97935524, 1.32355476, 0.93527685,
         1.12008114, 3.01614096, nan, nan]),
        index=times)
    expected = pd.DataFrame(np.
        array([[   0.        ,    0.        ,    0.        ],
               [   0.        ,    0.        ,    0.        ],
               [  91.1249279 ,  321.16092171,   51.17628184],
               [ 716.46580547,  888.9014706 ,   99.50500553],
               [1053.42066073,  953.24925905,  116.3286895 ],
               [ 863.54692748,  922.06124652,  106.9553658 ],
               [ 271.06382275,  655.44925213,   73.05968076],
               [   0.        ,    0.        ,    0.        ],
               [   0.        ,    0.        ,    0.        ]]),
                            columns=['ghi', 'dni', 'dhi'],
                            index=times)

    out = clearsky.ineichen(apparent_zenith, am, 3, perez_enhancement=True)
    assert_frame_equal(expected, out)


def test_ineichen_scalar_input():
    expected = OrderedDict()
    expected['ghi'] = 1038.159219
    expected['dni'] = 942.2081860378344
    expected['dhi'] = 110.26529293612793

    out = clearsky.ineichen(10., 1., 3.)
    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_ineichen_nans():
    length = 4

    apparent_zenith = np.full(length, 10.)
    apparent_zenith[0] = np.nan

    linke_turbidity = np.full(length, 3.)
    linke_turbidity[1] = np.nan

    dni_extra = np.full(length, 1370.)
    dni_extra[2] = np.nan

    airmass_absolute = np.full(length, 1.)

    expected = OrderedDict()
    expected['ghi'] = np.full(length, np.nan)
    expected['dni'] = np.full(length, np.nan)
    expected['dhi'] = np.full(length, np.nan)

    expected['ghi'][length-1] = 1042.72590228
    expected['dni'][length-1] = 946.35279683
    expected['dhi'][length-1] = 110.75033088

    out = clearsky.ineichen(apparent_zenith, airmass_absolute,
                            linke_turbidity, dni_extra=dni_extra)

    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_ineichen_arrays():
    expected = OrderedDict()

    expected['ghi'] = (np.
        array([[[1095.77074798, 1054.17449885, 1014.15727338],
                [ 839.40909243,  807.54451692,  776.88954373],
                [ 190.27859353,  183.05548067,  176.10656239]],

               [[ 773.49041181,  625.19479557,  505.33080493],
                [ 592.52803177,  478.92699901,  387.10585505],
                [ 134.31520045,  108.56393694,   87.74977339]],

               [[ 545.9968869 ,  370.78162375,  251.79449885],
                [ 418.25788117,  284.03520249,  192.88577665],
                [  94.81136442,   64.38555328,   43.72365587]]]))

    expected['dni'] = (np.
        array([[[1014.38807396,  942.20818604,  861.11344424],
                [1014.38807396,  942.20818604,  861.11344424],
                [1014.38807396,  942.20818604,  861.11344424]],

               [[ 687.61305142,  419.14891162,  255.50098235],
                [ 687.61305142,  419.14891162,  255.50098235],
                [ 687.61305142,  419.14891162,  255.50098235]],

               [[ 458.62196014,  186.46177428,   75.80970012],
                [ 458.62196014,  186.46177428,   75.80970012],
                [ 458.62196014,  186.46177428,   75.80970012]]]))

    expected['dhi'] = (np.
        array([[[ 81.38267402, 111.96631281, 153.04382915],
                [ 62.3427452 ,  85.77117175, 117.23837487],
                [ 14.13195304,  19.44274618,  26.57578203]],

               [[ 85.87736039, 206.04588395, 249.82982258],
                [ 65.78587472, 157.84030442, 191.38074731],
                [ 14.91244713,  35.77949226,  43.38249342]],

               [[ 87.37492676, 184.31984947, 175.98479873],
                [ 66.93307711, 141.19719644, 134.81217714],
                [ 15.17249681,  32.00680597,  30.5594396 ]]]))

    apparent_zenith = np.linspace(0, 80, 3)
    airmass_absolute = np.linspace(1, 10, 3)
    linke_turbidity = np.linspace(2, 4, 3)

    apparent_zenith, airmass_absolute, linke_turbidity = \
        np.meshgrid(apparent_zenith, airmass_absolute, linke_turbidity)

    out = clearsky.ineichen(apparent_zenith, airmass_absolute, linke_turbidity)

    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_ineichen_dni_extra():
    expected = pd.DataFrame(
        np.array([[1042.72590228,  946.35279683,  110.75033088]]),
        columns=['ghi', 'dni', 'dhi'])

    out = clearsky.ineichen(10, 1, 3, dni_extra=pd.Series(1370))
    assert_frame_equal(expected, out)


def test_ineichen_altitude():
    expected = pd.DataFrame(
        np.array([[1134.24312405,  994.95377835,  154.40492924]]),
        columns=['ghi', 'dni', 'dhi'])

    out = clearsky.ineichen(10, 1, 3, altitude=pd.Series(2000))
    assert_frame_equal(expected, out)


def test_haurwitz():
    apparent_solar_elevation = np.array([-20, -0.05, -0.001, 5, 10, 30, 50, 90])
    apparent_solar_zenith = 90 - apparent_solar_elevation
    data_in = pd.DataFrame(data=apparent_solar_zenith,
                           index=apparent_solar_zenith,
                           columns=['apparent_zenith'])
    expected = pd.DataFrame(np.array([0.,
                                      0.,
                                      0.,
                                      48.6298687941956,
                                      135.741748091813,
                                      487.894132885425,
                                      778.766689344363,
                                      1035.09203253450]),
                            columns=['ghi'],
                            index=apparent_solar_zenith)
    out = clearsky.haurwitz(data_in['apparent_zenith'])
    assert_frame_equal(expected, out)


def test_simplified_solis_scalar_elevation():
    expected = OrderedDict()
    expected['ghi'] = 1064.653145
    expected['dni'] = 959.335463
    expected['dhi'] = 129.125602

    out = clearsky.simplified_solis(80)
    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_simplified_solis_scalar_neg_elevation():
    expected = OrderedDict()
    expected['ghi'] = 0
    expected['dni'] = 0
    expected['dhi'] = 0

    out = clearsky.simplified_solis(-10)
    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_simplified_solis_series_elevation():
    expected = pd.DataFrame(
        np.array([[959.335463,  1064.653145,  129.125602]]),
        columns=['dni', 'ghi', 'dhi'])
    expected = expected[['ghi', 'dni', 'dhi']]

    out = clearsky.simplified_solis(pd.Series(80))
    assert_frame_equal(expected, out)


def test_simplified_solis_dni_extra():
    expected = pd.DataFrame(np.array([[963.555414,  1069.33637,  129.693603]]),
                            columns=['dni', 'ghi', 'dhi'])
    expected = expected[['ghi', 'dni', 'dhi']]

    out = clearsky.simplified_solis(80, dni_extra=pd.Series(1370))
    assert_frame_equal(expected, out)


def test_simplified_solis_pressure():
    expected = pd.DataFrame(np.
        array([[  964.26930718,  1067.96543669,   127.22841797],
               [  961.88811874,  1066.36847963,   128.1402539 ],
               [  959.58112234,  1064.81837558,   129.0304193 ]]),
                            columns=['dni', 'ghi', 'dhi'])
    expected = expected[['ghi', 'dni', 'dhi']]

    out = clearsky.simplified_solis(
        80, pressure=pd.Series([95000, 98000, 101000]))
    assert_frame_equal(expected, out)


def test_simplified_solis_aod700():
    expected = pd.DataFrame(np.
        array([[ 1056.61710493,  1105.7229086 ,    64.41747323],
               [ 1007.50558875,  1085.74139063,   102.96233698],
               [  959.3354628 ,  1064.65314509,   129.12560167],
               [  342.45810926,   638.63409683,    77.71786575],
               [   55.24140911,     7.5413313 ,     0.        ]]),
                            columns=['dni', 'ghi', 'dhi'])
    expected = expected[['ghi', 'dni', 'dhi']]

    aod700 = pd.Series([0.0, 0.05, 0.1, 1, 10])
    out = clearsky.simplified_solis(80, aod700=aod700)
    assert_frame_equal(expected, out)


def test_simplified_solis_precipitable_water():
    expected = pd.DataFrame(np.
        array([[ 1001.15353307,  1107.84678941,   128.58887606],
               [ 1001.15353307,  1107.84678941,   128.58887606],
               [  983.51027357,  1089.62306672,   129.08755996],
               [  959.3354628 ,  1064.65314509,   129.12560167],
               [  872.02335029,   974.18046717,   125.63581346]]),
                            columns=['dni', 'ghi', 'dhi'])
    expected = expected[['ghi', 'dni', 'dhi']]

    out = clearsky.simplified_solis(
        80, precipitable_water=pd.Series([0.0, 0.2, 0.5, 1.0, 5.0]))
    assert_frame_equal(expected, out)


def test_simplified_solis_small_scalar_pw():

    expected = OrderedDict()
    expected['ghi'] = 1107.84678941
    expected['dni'] = 1001.15353307
    expected['dhi'] = 128.58887606

    out = clearsky.simplified_solis(80, precipitable_water=0.1)
    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_simplified_solis_return_arrays():
    expected = OrderedDict()

    expected['ghi'] = np.array([[ 1148.40081325,   913.42330823],
                                [  965.48550828,   760.04527609]])

    expected['dni'] = np.array([[ 1099.25706525,   656.24601381],
                                [  915.31689149,   530.31697378]])

    expected['dhi'] = np.array([[   64.1063074 ,   254.6186615 ],
                                [   62.75642216,   232.21931597]])

    aod700 = np.linspace(0, 0.5, 2)
    precipitable_water = np.linspace(0, 10, 2)

    aod700, precipitable_water = np.meshgrid(aod700, precipitable_water)

    out = clearsky.simplified_solis(80, aod700, precipitable_water)

    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_simplified_solis_nans_arrays():

    # construct input arrays that each have 1 nan offset from each other,
    # the last point is valid for all arrays

    length = 6

    apparent_elevation = np.full(length, 80.)
    apparent_elevation[0] = np.nan

    aod700 = np.full(length, 0.1)
    aod700[1] = np.nan

    precipitable_water = np.full(length, 0.5)
    precipitable_water[2] = np.nan

    pressure = np.full(length, 98000.)
    pressure[3] = np.nan

    dni_extra = np.full(length, 1370.)
    dni_extra[4] = np.nan

    expected = OrderedDict()
    expected['ghi'] = np.full(length, np.nan)
    expected['dni'] = np.full(length, np.nan)
    expected['dhi'] = np.full(length, np.nan)

    expected['ghi'][length-1] = 1096.022736
    expected['dni'][length-1] = 990.306854
    expected['dhi'][length-1] = 128.664594

    out = clearsky.simplified_solis(apparent_elevation, aod700,
                                    precipitable_water, pressure, dni_extra)

    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_simplified_solis_nans_series():

    # construct input arrays that each have 1 nan offset from each other,
    # the last point is valid for all arrays

    length = 6

    apparent_elevation = pd.Series(np.full(length, 80.))
    apparent_elevation[0] = np.nan

    aod700 = np.full(length, 0.1)
    aod700[1] = np.nan

    precipitable_water = np.full(length, 0.5)
    precipitable_water[2] = np.nan

    pressure = np.full(length, 98000.)
    pressure[3] = np.nan

    dni_extra = np.full(length, 1370.)
    dni_extra[4] = np.nan

    expected = OrderedDict()
    expected['ghi'] = np.full(length, np.nan)
    expected['dni'] = np.full(length, np.nan)
    expected['dhi'] = np.full(length, np.nan)

    expected['ghi'][length-1] = 1096.022736
    expected['dni'][length-1] = 990.306854
    expected['dhi'][length-1] = 128.664594

    expected = pd.DataFrame.from_dict(expected)

    out = clearsky.simplified_solis(apparent_elevation, aod700,
                                    precipitable_water, pressure, dni_extra)

    assert_frame_equal(expected, out)






