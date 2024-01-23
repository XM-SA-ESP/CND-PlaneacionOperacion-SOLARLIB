import pandas as pd
import pytest
from packaging.version import Version
from xm_solarlib.location import Location
from pathlib import Path

try:
    from xm_solarlib import pvfactors
    has_pvfactors = True
except ImportError:
    has_pvfactors = False

requires_pvfactors = pytest.mark.skipif(not has_pvfactors,
                                        reason='requires pvfactors')

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / 'xm_solarlib/data'

def _check_pandas_assert_kwargs(kwargs):
    # handles the change in API related to default
    # tolerances in pandas 1.1.0.  See xm_solarlib GH #1018
    if Version(pd.__version__) >= Version('1.1.0'):
        if kwargs.pop('check_less_precise', False):
            kwargs['atol'] = 1e-3
            kwargs['rtol'] = 1e-3
        else:
            kwargs['atol'] = 1e-5
            kwargs['rtol'] = 1e-5
    else:
        kwargs.pop('rtol', None)
        kwargs.pop('atol', None)
    return kwargs

def assert_series_equal(left, right, **kwargs):
    kwargs = _check_pandas_assert_kwargs(kwargs)
    pd.testing.assert_series_equal(left, right, **kwargs)

def assert_frame_equal(left, right, **kwargs):
    kwargs = _check_pandas_assert_kwargs(kwargs)
    pd.testing.assert_frame_equal(left, right, **kwargs)

def assert_index_equal(left, right, **kwargs):
    kwargs = _check_pandas_assert_kwargs(kwargs)
    pd.testing.assert_index_equal(left, right, **kwargs)

try:
    import ephem
    has_ephem = True
except ImportError:
    has_ephem = False

requires_ephem = pytest.mark.skipif(not has_ephem, reason='requires ephem')


def has_spa_c():
    try:
        from xm_solarlib.spa_c_files.spa_py import spa_calc
    except ImportError:
        return False
    else:
        return True


requires_spa_c = pytest.mark.skipif(not has_spa_c(), reason="requires spa_c")


def has_numba():
    try:
        import numba
    except ImportError:
        return False
    else:
        vers = numba.__version__.split('.')
        if int(vers[0] + vers[1]) < 17:
            return False
        else:
            return True


requires_numba = pytest.mark.skipif(not has_numba(), reason="requires numba")


@pytest.fixture(scope="function")
def pvsyst_module_params():
    """
    Define some PVSyst module parameters for testing.

    The scope of the fixture is set to ``'function'`` to allow tests to modify
    parameters if required without affecting other tests.
    """
    parameters = {
        'gamma_ref': 1.05,
        'mu_gamma': 0.001,
        'i_l_ref': 6.0,
        'i_o_ref': 5e-9,
        'egref': 1.121,
        'r_sh_ref': 300,
        'r_sh_0': 1000,
        'r_s': 0.5,
        'r_sh_exp': 5.5,
        'cells_in_series': 60,
        'alpha_sc': 0.001,
    }
    return parameters


@pytest.fixture(scope='function')
def cec_module_params():
    """
    Define some CEC module parameters for testing.

    The scope of the fixture is set to ``'function'`` to allow tests to modify
    parameters if required without affecting other tests.
    """
    parameters = {
        'Name': 'Example Module',
        'BIPV': 'Y',
        'Date': '4/28/2008',
        'T_NOCT': 65,
        'A_c': 0.67,
        'N_s': 18,
        'I_sc_ref': 7.5,
        'V_oc_ref': 10.4,
        'I_mp_ref': 6.6,
        'V_mp_ref': 8.4,
        'alpha_sc': 0.003,
        'beta_oc': -0.04,
        'a_ref': 0.473,
        'i_l_ref': 7.545,
        'i_o_ref': 1.94e-09,
        'r_s': 0.094,
        'r_sh_ref': 15.72,
        'adjust': 10.6,
        'gamma_r': -0.5,
        'Version': 'MM105',
        'PTC': 48.9,
        'Technology': 'Multi-c-Si',
    }
    return parameters


@pytest.fixture()
def expected_solpos():
    return pd.DataFrame({'elevation': 39.872046,
                         'apparent_zenith': 50.111622,
                         'azimuth': 194.340241,
                         'apparent_elevation': 39.888378},
                        index=['2003-10-17T12:30:30Z'])


@pytest.fixture()
def golden_mst():
    return Location(39.742476, -105.1786, 'MST', 1830.14)

@pytest.fixture()
def golden():
    return Location(39.742476, -105.1786, 'America/Denver', 1830.14)


@pytest.fixture(scope='function')
def sapm_module_params():
    """
    Define SAPM model parameters for Canadian Solar CS5P 220M module.

    The scope of the fixture is set to ``'function'`` to allow tests to modify
    parameters if required without affecting other tests.
    """
    parameters = {'Material': 'c-Si',
                  'Cells_in_Series': 96,
                  'Parallel_Strings': 1,
                  'A0': 0.928385,
                  'A1': 0.068093,
                  'A2': -0.0157738,
                  'A3': 0.0016606,
                  'A4': -6.93E-05,
                  'B0': 1,
                  'B1': -0.002438,
                  'B2': 0.0003103,
                  'B3': -0.00001246,
                  'B4': 2.11E-07,
                  'B5': -1.36E-09,
                  'C0': 1.01284,
                  'C1': -0.0128398,
                  'C2': 0.279317,
                  'C3': -7.24463,
                  'C4': 0.996446,
                  'C5': 0.003554,
                  'C6': 1.15535,
                  'C7': -0.155353,
                  'Isco': 5.09115,
                  'Impo': 4.54629,
                  'Voco': 59.2608,
                  'Vmpo': 48.3156,
                  'Aisc': 0.000397,
                  'Aimp': 0.000181,
                  'bvoco': -0.21696,
                  'Mbvoc': 0.0,
                  'bvmpo': -0.235488,
                  'Mbvmp': 0.0,
                  'N': 1.4032,
                  'IXO': 4.97599,
                  'IXXO': 3.18803,
                  'FD': 1}
    return parameters