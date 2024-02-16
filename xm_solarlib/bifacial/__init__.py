"""
The ``bifacial`` module contains functions to model irradiance for bifacial
modules.

"""
from xm_solarlib._deprecation import deprecated
from xm_solarlib.bifacial import pvfactors

pvfactors_timeseries = deprecated(
    since='0.9.1',
    name='xm_solarlib.bifacial.pvfactors_timeseries',
    alternative='xm_solarlib.bifacial.pvfactors.pvfactors_timeseries'
)(pvfactors.pvfactors_timeseries)