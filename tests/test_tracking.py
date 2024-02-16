import numpy as np
import pandas as pd
import pytest
from .conftest import assert_series_equal
import xm_solarlib
from xm_solarlib import tracking



def test_singleaxis_aoi_gh1221():
    # vertical tracker
    loc = xm_solarlib.location.Location(40.1134, -88.3695)
    dr = pd.date_range(
        start='02-Jun-1998 00:00:00', end='02-Jun-1998 23:55:00', freq='5T',
        tz='Etc/GMT+6')
    sp = loc.get_solarposition(dr)
    tr = xm_solarlib.tracking.singleaxis(
        sp['apparent_zenith'], sp['azimuth'], axis_tilt=90, axis_azimuth=180,
        max_angle=0.001, backtrack=False)
    fixed = xm_solarlib.irradiance.aoi(90, 180, sp['apparent_zenith'], sp['azimuth'])
    fixed[np.isnan(tr['aoi'])] = np.nan
    assert np.allclose(tr['aoi'], fixed, equal_nan=True)


def test_calc_surface_orientation_types():
    # numpy arrays
    rotations = np.array([-10, 0, 10])
    expected_tilts = np.array([10, 0, 10], dtype=float)
    expected_azimuths = np.array([270, 90, 90], dtype=float)
    out = tracking.calc_surface_orientation(tracker_theta=rotations)
    np.testing.assert_allclose(expected_tilts, out['surface_tilt'])
    np.testing.assert_allclose(expected_azimuths, out['surface_azimuth'])

    # pandas Series
    rotations = pd.Series(rotations)
    expected_tilts = pd.Series(expected_tilts).rename('surface_tilt')
    expected_azimuths = pd.Series(expected_azimuths).rename('surface_azimuth')
    out = tracking.calc_surface_orientation(tracker_theta=rotations)
    assert_series_equal(expected_tilts, out['surface_tilt'])
    assert_series_equal(expected_azimuths, out['surface_azimuth'])

    # float
    for rotation, expected_tilt, expected_azimuth in zip(
            rotations, expected_tilts, expected_azimuths):
        out = tracking.calc_surface_orientation(rotation)
        assert out['surface_tilt'] == pytest.approx(expected_tilt)
        assert out['surface_azimuth'] == pytest.approx(expected_azimuth)


def test_calc_surface_orientation_kwargs():
    # non-default axis tilt & azimuth
    rotations = np.array([-10, 0, 10])
    expected_tilts = np.array([22.2687445, 20.0, 22.2687445])
    expected_azimuths = np.array([152.72683041, 180.0, 207.27316959])
    out = tracking.calc_surface_orientation(rotations,
                                            axis_tilt=20,
                                            axis_azimuth=180)
    np.testing.assert_allclose(out['surface_tilt'], expected_tilts)
    np.testing.assert_allclose(out['surface_azimuth'], expected_azimuths)


def test_calc_surface_orientation_special():
    # special cases for rotations
    rotations = np.array([-180, -90, -0, 0, 90, 180])
    expected_tilts = np.array([180, 90, 0, 0, 90, 180], dtype=float)
    expected_azimuths = [270, 270, 90, 90, 90, 90]
    out = tracking.calc_surface_orientation(rotations)
    np.testing.assert_allclose(out['surface_tilt'], expected_tilts)
    np.testing.assert_allclose(out['surface_azimuth'], expected_azimuths)

    # special case for axis_tilt
    rotations = np.array([-10, 0, 10])
    expected_tilts = np.array([90, 90, 90], dtype=float)
    expected_azimuths = np.array([350, 0, 10], dtype=float)
    out = tracking.calc_surface_orientation(rotations, axis_tilt=90)
    np.testing.assert_allclose(out['surface_tilt'], expected_tilts)
    np.testing.assert_allclose(out['surface_azimuth'], expected_azimuths)

    # special cases for axis_azimuth
    rotations = np.array([-10, 0, 10])
    expected_tilts = np.array([10, 0, 10], dtype=float)
    expected_azimuth_offsets = np.array([-90, 90, 90], dtype=float)
    for axis_azimuth in [0, 90, 180, 270, 360]:
        expected_azimuths = (axis_azimuth + expected_azimuth_offsets) % 360
        out = tracking.calc_surface_orientation(rotations,
                                                axis_azimuth=axis_azimuth)
        np.testing.assert_allclose(out['surface_tilt'], expected_tilts)
        # the rounding is a bit ugly, but necessary to test approximately equal
        # in a modulo-360 sense.
        np.testing.assert_allclose(np.round(out['surface_azimuth'], 4) % 360,
                                   expected_azimuths, rtol=1e-5, atol=1e-5)