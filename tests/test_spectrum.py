import pytest
from numpy.testing import assert_allclose
import pandas as pd
import numpy as np
from xm_solarlib.spectrum.mismatch import spectral_factor_firstsolar, spectral_factor_sapm

from .conftest import assert_series_equal


@pytest.mark.parametrize("module_type,expect", [
    ('cdte', np.array(
        [[ 0.99051020, 0.97640320, 0.93975028],
         [ 1.02928735, 1.01881074, 0.98578821],
         [ 1.04750335, 1.03814456, 1.00623986]])),
    ('monosi', np.array(
        [[ 0.97769770, 1.02043409, 1.03574032],
         [ 0.98630905, 1.03055092, 1.04736262],
         [ 0.98828494, 1.03299036, 1.05026561]])),
    ('polysi', np.array(
        [[ 0.97704080, 1.01705849, 1.02613202],
         [ 0.98992828, 1.03173953, 1.04260662],
         [ 0.99352435, 1.03588785, 1.04730718]])),
    ('cigs', np.array(
        [[ 0.97459190, 1.02821696, 1.05067895],
         [ 0.97529378, 1.02967497, 1.05289307],
         [ 0.97269159, 1.02730558, 1.05075651]])),
    ('asi', np.array(
        [[ 1.05552750, 0.87707583, 0.72243772],
         [ 1.11225204, 0.93665901, 0.78487953],
         [ 1.14555295, 0.97084011, 0.81994083]]))
])
def test_spectral_factor_firstsolar(module_type, expect):
    ams = np.array([1, 3, 5])
    pws = np.array([1, 3, 5])
    ams, pws = np.meshgrid(ams, pws)
    out = spectral_factor_firstsolar(pws, ams, module_type)
    assert_allclose(out, expect, atol=0.001)


def test_spectral_factor_firstsolar_supplied():
    # use the cdte coeffs
    coeffs = (0.87102, -0.040543, -0.00929202, 0.10052, 0.073062, -0.0034187)
    out = spectral_factor_firstsolar(1, 1, coefficients=coeffs)
    expected = 0.99134828
    assert_allclose(out, expected, atol=1e-3)


def test_spectral_factor_firstsolar_ambiguous():
    with pytest.raises(TypeError):
        spectral_factor_firstsolar(1, 1)


def test_spectral_factor_firstsolar_ambiguous_both():
    # use the cdte coeffs
    coeffs = (0.87102, -0.040543, -0.00929202, 0.10052, 0.073062, -0.0034187)
    with pytest.raises(TypeError):
        spectral_factor_firstsolar(1, 1, 'cdte', coefficients=coeffs)


def test_spectral_factor_firstsolar_large_airmass():
    # test that airmass > 10 is treated same as airmass==10
    m_eq10 = spectral_factor_firstsolar(1, 10, 'monosi')
    m_gt10 = spectral_factor_firstsolar(1, 15, 'monosi')
    assert_allclose(m_eq10, m_gt10)


def test_spectral_factor_firstsolar_low_airmass():
    with pytest.warns(UserWarning, match='Exceptionally low air mass'):
        _ = spectral_factor_firstsolar(1, 0.1, 'monosi')


def test_spectral_factor_firstsolar_range():
    with pytest.warns(UserWarning, match='Exceptionally high pw values'):
        out = spectral_factor_firstsolar(np.array([.1, 3, 10]),
                                                  np.array([1, 3, 5]),
                                                  module_type='monosi')
    expected = np.array([0.96080878, 1.03055092, np.nan])
    assert_allclose(out, expected, atol=1e-3)
    with pytest.warns(UserWarning, match='Exceptionally high pw values'):
        out = spectral_factor_firstsolar(6, 1.5,
                                                  max_precipitable_water=5,
                                                  module_type='monosi')
    with pytest.warns(UserWarning, match='Exceptionally low pw values'):
        out = spectral_factor_firstsolar(np.array([0, 3, 8]),
                                                  np.array([1, 3, 5]),
                                                  module_type='monosi')
    expected = np.array([0.96080878, 1.03055092, 1.04932727])
    assert_allclose(out, expected, atol=1e-3)
    with pytest.warns(UserWarning, match='Exceptionally low pw values'):
        spectral_factor_firstsolar(0.2, 1.5,
                                                  min_precipitable_water=1,
                                                  module_type='monosi')
        
@pytest.mark.parametrize('airmass,expected', [
    (1.5, 1.00028714375),
    (np.array([[10, np.nan]]), np.array([[0.999535, 0]])),
    (pd.Series([5]), pd.Series([1.0387675]))
])
def test_spectral_factor_sapm(sapm_module_params, airmass, expected):

    out = spectral_factor_sapm(airmass, sapm_module_params)

    if isinstance(airmass, pd.Series):
        assert_series_equal(out, expected, check_less_precise=4)
    else:
        assert_allclose(out, expected, atol=1e-4)
