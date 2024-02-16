import numpy as np
import pandas as pd
import scipy
from xm_solarlib import pvsystem
from xm_solarlib.singlediode import (bishop88_mpp, estimate_voc, VOLTAGE_BUILTIN,
                               bishop88, bishop88_i_from_v, bishop88_v_from_i)
from xm_solarlib._deprecation import xm_solarlib_deprecation_warning
import pytest
from .conftest import DATA_DIR

POA = 888
TCELL = 55

@pytest.fixture(scope="module")
def pandas_module():
    import pandas as pd
    return pd


def build_precise_iv_curve_dataframe(file_csv, file_json):
    """
    Reads a precise IV curve parameter set CSV and JSON to create a DataFrame.
    The CSV contains the parameters of the single diode equation which are used
    to generate the JSON data. The data are calculated using [1]_ with 40
    decimal digits of precision in order have at least 16 decimal digits of
    precision when they are stored in JSON. The precision is sufficient for the
    difference between the left and right side of the single diode equation to
    be less than :math:`1 \times 10^{-16}` when the numbers from the JSON are
    read as mpmath floats. The code to generate these IV curve data is from
    [2]_. The data and tests that use this function were added in :pull:`1573`.

    Parameters
    ----------
    file_csv: str
        Path to a CSV file of IV curve parameter sets.

    file_json: str
        Path to a JSON file of precise IV curves.

    Returns
    -------
        A DataFrame with these columns: ``Index``, ``photocurrent``,
        ``saturation_current``, ``resistance_series``, ``resistance_shunt``,
        ``n``, ``cells_in_series``, ``Voltages``, ``Currents``,
        ``diode_voltage``, ``v_oc``, ``i_sc``, ``v_mp``, ``i_mp``, ``p_mp``,
        ``i_x``, ``i_xx`, ``Temperature``, ``Irradiance``, ``Sweep Direction``,
        ``Datetime``, ``Boltzmann``, ``Elementary Charge``, and ``Vth``. The
        columns ``Irradiance``, ``Sweep Direction`` are None or empty strings.

    References
    ----------
    .. [1] The mpmath development team. (2023). mpmath: a Python library for
       arbitrary-precision floating-point arithmetic (version 1.2.1).
       `mpmath <mpmath.org>`_

    .. [2] The ivcurves development team. (2022). Code to generate precise
       solutions to the single diode equation.
       `ivcurves <github.com/cwhanse/ivcurves>`_
    """
    params = pd.read_csv(file_csv)
    curves_metadata = pd.read_json(file_json)
    curves = pd.DataFrame(curves_metadata['IV Curves'].values.tolist())
    curves['cells_in_series'] = curves_metadata['cells_in_series']
    joined = params.merge(curves, on='Index', how='inner',
                          suffixes=(None, '_drop'), validate='one_to_one')
    joined = joined[(c for c in joined.columns if not c.endswith('_drop'))]

    # parse strings to np.float64
    is_array = ['Currents', 'Voltages', 'diode_voltage']
    joined[is_array] = joined[is_array].applymap(
        lambda a: np.asarray(a, dtype=np.float64)
    )
    is_number = ['v_oc', 'i_sc', 'v_mp', 'i_mp', 'p_mp', 'i_x', 'i_xx',
                 'Temperature']
    joined[is_number] = joined[is_number].applymap(np.float64)

    joined['Boltzmann'] = scipy.constants.Boltzmann
    joined['Elementary Charge'] = scipy.constants.elementary_charge
    joined['Vth'] = (
        joined['Boltzmann'] * joined['Temperature']
        / joined['Elementary Charge']
    )

    return joined


@pytest.fixture(scope='function', params=[
    {
        'csv': f'{DATA_DIR}/precise_iv_curves_parameter_sets1.csv',
        'json': f'{DATA_DIR}/precise_iv_curves1.json'
    },
    {
        'csv': f'{DATA_DIR}/precise_iv_curves_parameter_sets2.csv',
        'json': f'{DATA_DIR}/precise_iv_curves2.json'
    }
], ids=[1, 2])
def precise_iv_curves(request):
    file_csv, file_json = request.param['csv'], request.param['json']
    pc = build_precise_iv_curve_dataframe(file_csv, file_json)
    params = ['photocurrent', 'saturation_current', 'resistance_series',
              'resistance_shunt']
    singlediode_params = pc.loc[:, params]
    singlediode_params['nnsvth'] = pc['n'] * pc['cells_in_series'] * pc['Vth']
    return singlediode_params, pc


@pytest.mark.parametrize('method', ['lambertw', 'brentq', 'newton'])
def test_singlediode_precision(method, precise_iv_curves):
    """
    Tests the accuracy of singlediode. ivcurve_pnts is not tested.
    """
    x, pc = precise_iv_curves
    outs = pvsystem.singlediode(method=method, **x)

    assert np.allclose(pc['i_sc'], outs['i_sc'], atol=1e-10, rtol=0)
    assert np.allclose(pc['v_oc'], outs['v_oc'], atol=1e-10, rtol=0)
    assert np.allclose(pc['i_mp'], outs['i_mp'], atol=7e-8, rtol=0)
    assert np.allclose(pc['v_mp'], outs['v_mp'], atol=1e-6, rtol=0)
    assert np.allclose(pc['p_mp'], outs['p_mp'], atol=1e-10, rtol=0)
    assert np.allclose(pc['i_x'], outs['i_x'], atol=1e-10, rtol=0)

    # This test should pass with atol=9e-8 on MacOS and Windows.
    # The atol was lowered to pass on Linux when the vectorized umath module
    # introduced in NumPy 1.22.0 is used.
    assert np.allclose(pc['i_xx'], outs['i_xx'], atol=1e-6, rtol=0)


def test_singlediode_lambert_negative_voc():

    # Those values result in a negative v_oc out of `_lambertw_v_from_i`
    x = np.array([0., 1.480501e-11, 0.178, 8000., 1.797559])
    outs = pvsystem.singlediode(*x, method='lambertw')
    assert outs['v_oc'] == 0

    # Testing for an array
    x  = np.array([x, x]).T
    outs = pvsystem.singlediode(*x, method='lambertw')
    assert np.array_equal(outs['v_oc'], [0, 0])


@pytest.mark.parametrize('method', ['lambertw'])
def test_ivcurve_pnts_precision(method, precise_iv_curves):
    """
    Tests the accuracy of the IV curve points calcuated by singlediode. Only
    methods of singlediode that linearly spaced points are tested.
    """
    x, pc = precise_iv_curves
    pc_i, pc_v = np.stack(pc['Currents']), np.stack(pc['Voltages'])
    ivcurve_pnts = len(pc['Currents'][0])

    with pytest.warns(xm_solarlib_deprecation_warning, match='ivcurve_pnts'):
        outs = pvsystem.singlediode(method=method, ivcurve_pnts=ivcurve_pnts,
                                    **x)

    assert np.allclose(pc_i, outs['i'], atol=1e-10, rtol=0)
    assert np.allclose(pc_v, outs['v'], atol=1e-10, rtol=0)


def get_pvsyst_fs_495():
    """
    PVsyst parameters for First Solar FS-495 module from PVSyst-6.7.2 database.

    i_l_ref derived from Isc_ref conditions::

        i_l_ref = (I_sc_ref + Id + Ish) / (1 - d2mutau/(Vbi*N_s - Vd))

    where::

        Vd = I_sc_ref * r_s
        Id = i_o_ref * (exp(Vd / nNsVt) - 1)
        Ish = Vd / r_sh_ref

    """
    return {
        'd2mutau': 1.31, 'alpha_sc': 0.00039, 'gamma_ref': 1.48,
        'mu_gamma': 0.001, 'i_o_ref': 9.62e-10, 'r_sh_ref': 5000,
        'r_sh_0': 12500, 'r_sh_exp': 3.1, 'r_s': 4.6, 'beta_oc': -0.2116,
        'egref': 1.5, 'cells_in_series': 108, 'cells_in_parallel': 2,
        'I_sc_ref': 1.55, 'V_oc_ref': 86.5, 'I_mp_ref': 1.4, 'V_mp_ref': 67.85,
        'temp_ref': 25, 'irrad_ref': 1000, 'i_l_ref': 1.5743233463848496
    }

# DeSoto @(888[W/m**2], 55[degC]) = {Pmp: 72.71, Isc: 1.402, Voc: 75.42)




