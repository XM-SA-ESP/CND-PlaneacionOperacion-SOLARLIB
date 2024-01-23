import numpy as np
from  xm_solarlib.pvfactors.irradiance.utils import \
    perez_diffuse_luminance, breakup_df_inputs, \
    calculate_circumsolar_shading, calculate_horizon_band_shading


def test_calculate_circumsolar_shading():
    """
    Test that the disk shading function stays consistent
    """
    # Test for one value of 20% of the diameter being covered
    percentage_distance_covered = 20.
    percent_shading = calculate_circumsolar_shading(
        percentage_distance_covered, model='uniform_disk')

    # Compare to expected
    expected_disk_shading_perc = 14.2378489933
    atol = 0
    rtol = 1e-8
    np.testing.assert_allclose(expected_disk_shading_perc, percent_shading,
                               atol=atol, rtol=rtol)


def test_calculate_horizon_band_shading():
    """Test that calculation of horizon band shading percentage is correct """

    shading_angle = np.array([-10., 0., 3., 9., 20.])
    horizon_band_angle = 15.
    percent_shading = calculate_horizon_band_shading(shading_angle,
                                                     horizon_band_angle)
    expected_percent_shading = [0., 0., 20., 60., 100.]
    np.testing.assert_allclose(expected_percent_shading, percent_shading)