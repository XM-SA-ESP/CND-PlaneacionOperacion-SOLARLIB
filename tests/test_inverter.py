import numpy as np
import pandas as pd

from .conftest import assert_series_equal
from numpy.testing import assert_allclose

from xm_solarlib import inverter

def test_sandia_multi():

    cec_inverter_parameters = {
        'Name': 'Inversor Test',
        'Vac': 208.0,
        'Paco': 250.0,
        'Pdco': 250.0,
        'Vdco': 30.0,
        'Pso': 1.0,
        'C0': -5.0e-05,
        'C1': -6.0e-04,
        'C2': 0.08,
        'C3': -0.1,
        'Pnt': 0.02,
        'Vdcmax': 48.0,
        'Idcmax': 9.8,
        'Mppt_low': 27.0,
        'Mppt_high': 39.0,
    }
    
     
    vdcs = pd.Series(np.linspace(0, 50, 3))
    idcs = pd.Series(np.linspace(0, 11, 3)) / 2
    pdcs = idcs * vdcs
    pacs = inverter.sandia_multi((vdcs, vdcs), (pdcs, pdcs),
                                 cec_inverter_parameters)
    assert_series_equal(pacs, pd.Series([-0.020000, 137.98070362944736, 250.000000]))
    # with lists instead of tuples
    pacs = inverter.sandia_multi([vdcs, vdcs], [pdcs, pdcs],
                                 cec_inverter_parameters)
    assert_series_equal(pacs, pd.Series([-0.020000, 137.98070362944736, 250.000000]))
    # with arrays instead of tuples
    pacs = inverter.sandia_multi(np.array([vdcs, vdcs]),
                                 np.array([pdcs, pdcs]),
                                 cec_inverter_parameters)
    assert_allclose(pacs, np.array([-0.020000, 137.98070362944736, 250.000000]))