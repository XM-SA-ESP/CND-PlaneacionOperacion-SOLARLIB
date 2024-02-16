import numpy as np
import pandas as pd
import pytest
import unittest.mock as mock
from .conftest import assert_series_equal
from numpy.testing import assert_allclose
from xm_solarlib import pvsystem
from xm_solarlib import iam as _iam
from xm_solarlib import spectrum
from xm_solarlib.pvsystem import FixedMount
from xm_solarlib import temperature
from xm_solarlib.constantes import LENGTH_MISMATCH



@pytest.fixture
def two_array_system(pvsyst_module_params, cec_module_params):
    """Two-array PVSystem.

    Both arrays are identical.
    """
    temperature_model = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][
        'open_rack_glass_glass'
    ]
    # Need u_v to be non-zero so wind-speed changes cell temperature
    # under the pvsyst model.
    temperature_model['u_v'] = 1.0
    # parameter for fuentes temperature model
    temperature_model['noct_installed'] = 45
    # parameters for noct_sam temperature model
    temperature_model['noct'] = 45.
    temperature_model['module_efficiency'] = 0.2
    module_params = {**pvsyst_module_params, **cec_module_params}
    return pvsystem.PVSystem(
        arrays=[
            pvsystem.Array(
                mount=pvsystem.FixedMount(0, 180),
                temperature_model_parameters=temperature_model,
                module_parameters=module_params
            ),
            pvsystem.Array(
                mount=pvsystem.FixedMount(0, 180),
                temperature_model_parameters=temperature_model,
                module_parameters=module_params
            )
        ]
    )

def test_pvsystem_get_iam_sapm(sapm_module_params, mocker):
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)
    mocker.spy(_iam, 'sapm')
    aoi = 0
    out = system.get_iam(aoi, 'sapm')
    _iam.sapm.assert_called_once_with(aoi, sapm_module_params)
    assert_allclose(out, 1.0, atol=0.01)


def test_pvsystem_get_iam_invalid(sapm_module_params, mocker):
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)
    with pytest.raises(ValueError):
        system.get_iam(45, iam_model='not_a_model')


def test_pvsystem_sapm(sapm_module_params, mocker):
    mocker.spy(pvsystem, 'sapm')
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)
    effective_irradiance = 500
    temp_cell = 25
    out = system.sapm(effective_irradiance, temp_cell)
    pvsystem.sapm.assert_called_once_with(effective_irradiance, temp_cell,
                                          sapm_module_params)
    assert_allclose(out['p_mp'], 100, atol=100)


def test_pvsystem_multi_array_sapm(sapm_module_params):
    system = pvsystem.PVSystem(
        arrays=[pvsystem.Array(pvsystem.FixedMount(0, 180),
                               module_parameters=sapm_module_params),
                pvsystem.Array(pvsystem.FixedMount(0, 180),
                               module_parameters=sapm_module_params)]
    )
    effective_irradiance = (100, 500)
    temp_cell = (15, 25)
    sapm_one, sapm_two = system.sapm(effective_irradiance, temp_cell)
    assert sapm_one['p_mp'] != sapm_two['p_mp']
    sapm_one_flip, sapm_two_flip = system.sapm(
        (effective_irradiance[1], effective_irradiance[0]),
        (temp_cell[1], temp_cell[0])
    )
    assert sapm_one_flip['p_mp'] == sapm_two['p_mp']
    assert sapm_two_flip['p_mp'] == sapm_one['p_mp']
    with pytest.raises(ValueError,
                       match=LENGTH_MISMATCH):
        system.sapm(effective_irradiance, 10)
    with pytest.raises(ValueError,
                       match=LENGTH_MISMATCH):
        system.sapm(500, temp_cell)


def test_pvsystem_sapm_spectral_loss(sapm_module_params, mocker):
    mocker.spy(spectrum, 'spectral_factor_sapm')
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)
    airmass = 2
    out = system.sapm_spectral_loss(airmass)
    spectrum.spectral_factor_sapm.assert_called_once_with(airmass,
                                                          sapm_module_params)
    assert_allclose(out, 1, atol=0.5)


def test_pvsystem_multi_array_sapm_spectral_loss(sapm_module_params):
    system = pvsystem.PVSystem(
        arrays=[pvsystem.Array(pvsystem.FixedMount(0, 180),
                               module_parameters=sapm_module_params),
                pvsystem.Array(pvsystem.FixedMount(0, 180),
                               module_parameters=sapm_module_params)]
    )
    loss_one, loss_two = system.sapm_spectral_loss(2)
    assert loss_one == loss_two


def test_pvsystem_sapm_effective_irradiance(sapm_module_params, mocker):
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)
    mocker.spy(pvsystem, 'sapm_effective_irradiance')

    poa_direct = 900
    poa_diffuse = 100
    airmass_absolute = 1.5
    aoi = 0
    p = (sapm_module_params['A4'], sapm_module_params['A3'],
         sapm_module_params['A2'], sapm_module_params['A1'],
         sapm_module_params['A0'])
    f1 = np.polyval(p, airmass_absolute)
    expected = f1 * (poa_direct + sapm_module_params['FD'] * poa_diffuse)
    out = system.sapm_effective_irradiance(
        poa_direct, poa_diffuse, airmass_absolute, aoi)
    pvsystem.sapm_effective_irradiance.assert_called_once_with(
        poa_direct, poa_diffuse, airmass_absolute, aoi, sapm_module_params)
    assert_allclose(out, expected, atol=0.1)


def test_pvsystem_multi_array_sapm_effective_irradiance(sapm_module_params):
    system = pvsystem.PVSystem(
        arrays=[pvsystem.Array(pvsystem.FixedMount(0, 180),
                               module_parameters=sapm_module_params),
                pvsystem.Array(pvsystem.FixedMount(0, 180),
                               module_parameters=sapm_module_params)]
    )
    poa_direct = (500, 900)
    poa_diffuse = (50, 100)
    aoi = (0, 10)
    airmass_absolute = 1.5
    irrad_one, irrad_two = system.sapm_effective_irradiance(
        poa_direct, poa_diffuse, airmass_absolute, aoi
    )
    assert irrad_one != irrad_two


@pytest.mark.parametrize("poa_direct, poa_diffuse, aoi",
                         [(20, (10, 10), (20, 20)),
                          ((20, 20), (10,), (20, 20)),
                          ((20, 20), (10, 10), 20)])
def test_pvsystem_sapm_effective_irradiance_value_error(
        poa_direct, poa_diffuse, aoi, two_array_system):
    with pytest.raises(ValueError,
                       match=LENGTH_MISMATCH):
        two_array_system.sapm_effective_irradiance(
            poa_direct, poa_diffuse, 10, aoi
        )


def test_pvsystem_get_cell_temperature_invalid():
    system = pvsystem.PVSystem()
    with pytest.raises(ValueError, match='not a valid'):
        system.get_cell_temperature(1000, 25, 1, 'not_a_model')


@pytest.mark.parametrize("model",
                         ['faiman', 'pvsyst', 'sapm', 'fuentes', 'noct_sam'])
def test_pvsystem_multi_array_celltemp_temp_too_short(
        model, two_array_system):
    with pytest.raises(ValueError,
                       match=LENGTH_MISMATCH):
        two_array_system.get_cell_temperature((1000, 1000), (1,), 1,
                                              model=model)


@pytest.mark.parametrize("model",
                         ['faiman', 'pvsyst', 'sapm', 'fuentes', 'noct_sam'])
def test_pvsystem_multi_array_celltemp_temp_too_long(
        model, two_array_system):
    with pytest.raises(ValueError,
                       match=LENGTH_MISMATCH):
        two_array_system.get_cell_temperature((1000, 1000), (1, 1, 1), 1,
                                              model=model)


@pytest.mark.parametrize("model",
                         ['faiman', 'pvsyst', 'sapm', 'fuentes', 'noct_sam'])
def test_pvsystem_multi_array_celltemp_wind_too_short(
        model, two_array_system):
    with pytest.raises(ValueError,
                       match=LENGTH_MISMATCH):
        two_array_system.get_cell_temperature((1000, 1000), 25, (1,),
                                              model=model)


@pytest.mark.parametrize("model",
                         ['faiman', 'pvsyst', 'sapm', 'fuentes', 'noct_sam'])
def test_pvsystem_multi_array_celltemp_wind_too_long(
        model, two_array_system):
    with pytest.raises(ValueError,
                       match=LENGTH_MISMATCH):
        two_array_system.get_cell_temperature((1000, 1000), 25, (1, 1, 1),
                                              model=model)


@pytest.mark.parametrize("model",
                         ['faiman', 'pvsyst', 'sapm', 'fuentes', 'noct_sam'])
def test_pvsystem_multi_array_celltemp_poa_length_mismatch(
        model, two_array_system):
    with pytest.raises(ValueError,
                       match=LENGTH_MISMATCH):
        two_array_system.get_cell_temperature(1000, 25, 1, model=model)


@pytest.fixture
def single_axis_tracker_mount():
    return pvsystem.SingleAxisTrackerMount(axis_tilt=10, axis_azimuth=170,
                                           max_angle=45, backtrack=False,
                                           gcr=0.4, cross_axis_tilt=-5)

def test_singleaxistrackermount_constructor(single_axis_tracker_mount):
    expected = dict(axis_tilt=10, axis_azimuth=170, max_angle=45,
                    backtrack=False, gcr=0.4, cross_axis_tilt=-5)
    for attr_name, expected_value in expected.items():
        assert getattr(single_axis_tracker_mount, attr_name) == expected_value


def test_singleaxistrackermount_get_orientation(single_axis_tracker_mount):
    expected = {'surface_tilt': 19.29835284, 'surface_azimuth': 229.7643755}
    actual = single_axis_tracker_mount.get_orientation(45, 190)
    for key, expected_value in expected.items():
        err_msg = f"{key} value incorrect"
        assert actual[key] == pytest.approx(expected_value), err_msg


def test_singleaxistrackermount_get_orientation_asymmetric_max():
    mount = pvsystem.SingleAxisTrackerMount(max_angle=(-30, 45))
    expected = {'surface_tilt': [45, 30], 'surface_azimuth': [90, 270]}
    actual = mount.get_orientation([60, 60], [90, 270])
    for key, expected_value in expected.items():
        err_msg = f"{key} value incorrect"
        assert actual[key] == pytest.approx(expected_value), err_msg



def test_calcparams_desoto(cec_module_params):
    times = pd.date_range(start='2015-01-01', periods=3, freq='12H')
    df = pd.DataFrame({
        'effective_irradiance': [0.0, 800.0, 800.0],
        'temp_cell': [25, 25, 50]
    }, index=times)

    IL, I0, rs, rsh, nnsvth = pvsystem.calcparams_desoto(
        df['effective_irradiance'],
        df['temp_cell'],
        alpha_sc=cec_module_params['alpha_sc'],
        a_ref=cec_module_params['a_ref'],
        i_l_ref=cec_module_params['i_l_ref'],
        i_o_ref=cec_module_params['i_o_ref'],
        r_sh_ref=cec_module_params['r_sh_ref'],
        r_s=cec_module_params['r_s'],
        egref=1.121,
        degdt=-0.0002677
    )

    assert_series_equal(IL, pd.Series([0.0, 6.036, 6.096], index=times),
                        check_less_precise=3)
    assert_series_equal(I0, pd.Series([0.0, 1.94e-9, 7.419e-8], index=times),
                        check_less_precise=3)
    assert_series_equal(rs, pd.Series([0.094, 0.094, 0.094], index=times),
                        check_less_precise=3)
    assert_series_equal(rsh, pd.Series([np.inf, 19.65, 19.65], index=times),
                        check_less_precise=3)
    assert_series_equal(nnsvth, pd.Series([0.473, 0.473, 0.5127], index=times),
                        check_less_precise=3)


def test_calcparams_cec(cec_module_params):
    times = pd.date_range(start='2015-01-01', periods=3, freq='12H')
    df = pd.DataFrame({
        'effective_irradiance': [0.0, 800.0, 800.0],
        'temp_cell': [25, 25, 50]
    }, index=times)

    IL, I0, rs, rsh, nnsvth = pvsystem.calcparams_cec(
        df['effective_irradiance'],
        df['temp_cell'],
        alpha_sc=cec_module_params['alpha_sc'],
        a_ref=cec_module_params['a_ref'],
        i_l_ref=cec_module_params['i_l_ref'],
        i_o_ref=cec_module_params['i_o_ref'],
        r_sh_ref=cec_module_params['r_sh_ref'],
        r_s=cec_module_params['r_s'],
        adjust=cec_module_params['adjust'],
        egref=1.121,
        degdt=-0.0002677
    )

    assert_series_equal(IL, pd.Series([0.0, 6.036, 6.0896], index=times),
                        check_less_precise=3)
    assert_series_equal(I0, pd.Series([0.0, 1.94e-9, 7.419e-8], index=times),
                        check_less_precise=3)
    assert_series_equal(rs, pd.Series([0.094, 0.094, 0.094], index=times),
                        check_less_precise=3)
    assert_series_equal(rsh, pd.Series([np.inf, 19.65, 19.65], index=times),
                        check_less_precise=3)
    assert_series_equal(nnsvth, pd.Series([0.473, 0.473, 0.5127], index=times),
                        check_less_precise=3)