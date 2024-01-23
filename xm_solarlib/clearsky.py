"""
The ``clearsky`` module contains several methods
to calculate clear sky GHI, DNI, and DHI.
"""

import os
from collections import OrderedDict
import calendar

import numpy as np
import pandas as pd
import h5py

from xm_solarlib import tools
from xm_solarlib.tools import _degrees_to_index


def ineichen(apparent_zenith, airmass_absolute, linke_turbidity,
             altitude=0, dni_extra=1364., perez_enhancement=False):
    '''
    Determine clear sky GHI, DNI, and DHI from Ineichen/Perez model.

    Implements the Ineichen and Perez clear sky model for global
    horizontal irradiance (GHI), direct normal irradiance (DNI), and
    calculates the clear-sky diffuse horizontal (DHI) component as the
    difference between GHI and DNI*cos(zenith) as presented in [1, 2]. A
    report on clear sky models found the Ineichen/Perez model to have
    excellent performance with a minimal input data set [3].

    Default values for monthly Linke turbidity provided by SoDa [4, 5].

    Parameters
    -----------
    apparent_zenith : numeric
        Refraction corrected solar zenith angle in degrees.

    airmass_absolute : numeric
        Pressure corrected airmass.

    linke_turbidity : numeric
        Linke Turbidity.

    altitude : numeric, default 0
        Altitude above sea level in meters.

    dni_extra : numeric, default 1364
        Extraterrestrial irradiance. The units of ``dni_extra``
        determine the units of the output.

    perez_enhancement : bool, default False
        Controls if the Perez enhancement factor should be applied.
        Setting to True may produce spurious results for times when
        the Sun is near the horizon and the airmass is high.
        See https://github.com/xm_solarlib/xm_solarlib-python/issues/435

    Returns
    -------
    clearsky : DataFrame (if Series input) or OrderedDict of arrays
        DataFrame/OrderedDict contains the columns/keys
        ``'dhi', 'dni', 'ghi'``.

    See also
    --------
    lookup_linke_turbidity
    xm_solarlib.location.Location.get_clearsky

    References
    ----------
    .. [1] P. Ineichen and R. Perez, "A New airmass independent formulation for
       the Linke turbidity coefficient", Solar Energy, vol 73, pp. 151-157,
       2002.

    .. [2] R. Perez et. al., "A New Operational Model for Satellite-Derived
       Irradiances: Description and Validation", Solar Energy, vol 73, pp.
       307-317, 2002.

    .. [3] M. Reno, C. Hansen, and J. Stein, "Global Horizontal Irradiance
       Clear Sky Models: Implementation and Analysis", Sandia National
       Laboratories, SAND2012-2389, 2012.

    .. [4] http://www.soda-is.com/eng/services/climat_free_eng.php#c5 (obtained
       July 17, 2012).

    .. [5] J. Remund, et. al., "Worldwide Linke Turbidity Information", Proc.
       ISES Solar World Congress, June 2003. Goteborg, Sweden.
    '''

    # ghi is calculated using either the equations in [1] by setting
    # perez_enhancement=False (default behavior) or using the model
    # in [2] by setting perez_enhancement=True.

    # The NaN handling is a little subtle. The AM input is likely to
    # have NaNs that we'll want to map to 0s in the output. However, we
    # want NaNs in other inputs to propagate through to the output. This
    # is accomplished by judicious use and placement of np.maximum,
    # np.minimum, and np.fmax

    # use max so that nighttime values will result in 0s instead of
    # negatives. propagates nans.
    cos_zenith = np.maximum(tools.cosd(apparent_zenith), 0)

    tl = linke_turbidity

    fh1 = np.exp(-altitude/8000.)
    fh2 = np.exp(-altitude/1250.)
    cg1 = 5.09e-05 * altitude + 0.868
    cg2 = 3.92e-05 * altitude + 0.0387

    ghi = np.exp(-cg2*airmass_absolute*(fh1 + fh2*(tl - 1)))

    # https://github.com/xm_solarlib/xm_solarlib-python/issues/435
    if perez_enhancement:
        ghi *= np.exp(0.01*airmass_absolute**1.8)

    # use fmax to map airmass nans to 0s. multiply and divide by tl to
    # reinsert tl nans
    ghi = cg1 * dni_extra * cos_zenith * tl / tl * np.fmax(ghi, 0)

    # From [1] (Following [2] leads to 0.664 + 0.16268 / fh1)
    # See https://github.com/xm_solarlib/xm_solarlib-python/pull/808
    b = 0.664 + 0.163/fh1
    bnci = b * np.exp(-0.09 * airmass_absolute * (tl - 1))
    bnci = dni_extra * np.fmax(bnci, 0)

    # "empirical correction" SE 73, 157 & SE 73, 312.
    bnci_2 = ((1 - (0.1 - 0.2*np.exp(-tl))/(0.1 + 0.882/fh1)) /
              cos_zenith)
    bnci_2 = ghi * np.fmin(np.fmax(bnci_2, 0), 1e20)

    dni = np.minimum(bnci, bnci_2)

    dhi = ghi - dni*cos_zenith

    irrads = OrderedDict()
    irrads['ghi'] = ghi
    irrads['dni'] = dni
    irrads['dhi'] = dhi

    if isinstance(dni, pd.Series):
        irrads = pd.DataFrame.from_dict(irrads)

    return irrads


def lookup_linke_turbidity(time, latitude, longitude, filepath=None,
                           interp_turbidity=True):
    """
    Look up the Linke Turibidity from the ``LinkeTurbidities.h5``
    data file supplied with xm_solarlib.

    Parameters
    ----------
    time : pandas.DatetimeIndex

    latitude : float or int

    longitude : float or int

    filepath : None or string, default None
        The path to the ``.h5`` file.

    interp_turbidity : bool, default True
        If ``True``, interpolates the monthly Linke turbidity values
        found in ``LinkeTurbidities.h5`` to daily values.

    Returns
    -------
    turbidity : Series
    """

    # The .h5 file 'LinkeTurbidities.h5' contains a single 2160 x 4320 x 12
    # matrix of type uint8 called 'LinkeTurbidity'. The rows represent global
    # latitudes from 90 to -90 degrees; the columns represent global longitudes
    # from -180 to 180; and the depth (third dimension) represents months of
    # the year from January (1) to December (12). To determine the Linke
    # turbidity for a position on the Earth's surface for a given month do the
    # following: LT = LinkeTurbidity(LatitudeIndex, LongitudeIndex, month).
    # Note that the numbers within the matrix are 20 * Linke Turbidity,
    # so divide the number from the file by 20 to get the
    # turbidity.

    # The nodes of the grid are 5' (1/12=0.0833[arcdeg]) apart.
    # From Section 8 of Aerosol optical depth and Linke turbidity climatology
    # http://www.meteonorm.com/images/uploads/downloads/ieashc36_report_TL_AOD_climatologies.pdf
    # 1st row: 89.9583 S, 2nd row: 89.875 S
    # 1st column: 179.9583 W, 2nd column: 179.875 W

    if filepath is None:
        xm_solarlib_path = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(xm_solarlib_path, 'data', 'LinkeTurbidities.h5')

    latitude_index = _degrees_to_index(latitude, coordinate='latitude')
    longitude_index = _degrees_to_index(longitude, coordinate='longitude')

    with h5py.File(filepath, 'r') as lt_h5_file:
        lts = lt_h5_file['LinkeTurbidity'][latitude_index, longitude_index]

    if interp_turbidity:
        linke_turbidity = _interpolate_turbidity(lts, time)
    else:
        months = time.month - 1
        linke_turbidity = pd.Series(lts[months], index=time)

    linke_turbidity /= 20.

    return linke_turbidity


def _interpolate_turbidity(lts, time):
    """
    Interpolated monthly Linke turbidity onto daily values.

    Parameters
    ----------
    lts : np.array
        Monthly Linke turbidity values.
    time : pd.DatetimeIndex
        Times to be interpolated onto.

    Returns
    -------
    linke_turbidity : pd.Series
        The interpolated turbidity.
    """
    # Data covers 1 year. Assume that data corresponds to the value at the
    # middle of each month. This means that we need to add previous Dec and
    # next Jan to the array so that the interpolation will work for
    # Jan 1 - Jan 15 and Dec 16 - Dec 31.
    lts_concat = np.concatenate([[lts[-1]], lts, [lts[0]]])

    # handle leap years
    try:
        isleap = time.is_leap_year
    except AttributeError:
        year = time.year
        isleap = _is_leap_year(year)

    dayofyear = time.dayofyear
    days_leap = _calendar_month_middles(2016)
    days_no_leap = _calendar_month_middles(2015)

    # Then we map the month value to the day of year value.
    # Do it for both leap and non-leap years.
    lt_leap = np.interp(dayofyear, days_leap, lts_concat)
    lt_no_leap = np.interp(dayofyear, days_no_leap, lts_concat)
    linke_turbidity = np.where(isleap, lt_leap, lt_no_leap)

    linke_turbidity = pd.Series(linke_turbidity, index=time)

    return linke_turbidity


def _calendar_month_middles(year):
    """List of middle day of each month, used by Linke turbidity lookup"""
    # remove mdays[0] since January starts at mdays[1]
    # make local copy of mdays since we need to change
    # February for leap years
    mdays = np.array(calendar.mdays[1:])
    ydays = 365
    # handle leap years
    if calendar.isleap(year):
        mdays[1] = mdays[1] + 1
        ydays = 366
    middles = np.concatenate(
        [[-calendar.mdays[-1] / 2.0],  # Dec last year
         np.cumsum(mdays) - np.array(mdays) / 2.,  # this year
         [ydays + calendar.mdays[1] / 2.0]])  # Jan next year
    return middles


def _is_leap_year(year):
    """Determine if a year is leap year.

    Parameters
    ----------
    year : numeric

    Returns
    -------
    isleap : array of bools
    """
    isleap = ((np.mod(year, 4) == 0) &
              ((np.mod(year, 100) != 0) | (np.mod(year, 400) == 0)))
    return isleap


def haurwitz(apparent_zenith):
    '''
    Determine clear sky GHI using the Haurwitz model.

    Implements the Haurwitz clear sky model for global horizontal
    irradiance (GHI) as presented in [1, 2]. A report on clear
    sky models found the Haurwitz model to have the best performance
    in terms of average monthly error among models which require only
    zenith angle [3].

    Parameters
    ----------
    apparent_zenith : Series
        The apparent (refraction corrected) sun zenith angle
        in degrees.

    Returns
    -------
    ghi : DataFrame
        The modeled global horizonal irradiance in W/m^2 provided
        by the Haurwitz clear-sky model.

    References
    ----------

    .. [1] B. Haurwitz, "Insolation in Relation to Cloudiness and Cloud
       Density," Journal of Meteorology, vol. 2, pp. 154-166, 1945.

    .. [2] B. Haurwitz, "Insolation in Relation to Cloud Type," Journal of
       Meteorology, vol. 3, pp. 123-124, 1946.

    .. [3] M. Reno, C. Hansen, and J. Stein, "Global Horizontal Irradiance
       Clear Sky Models: Implementation and Analysis", Sandia National
       Laboratories, SAND2012-2389, 2012.
    '''

    cos_zenith = tools.cosd(apparent_zenith.values)
    clearsky_ghi = np.zeros_like(apparent_zenith.values)
    cos_zen_gte_0 = cos_zenith > 0
    clearsky_ghi[cos_zen_gte_0] = (1098.0 * cos_zenith[cos_zen_gte_0] *
                                   np.exp(-0.059/cos_zenith[cos_zen_gte_0]))

    df_out = pd.DataFrame(index=apparent_zenith.index,
                          data=clearsky_ghi,
                          columns=['ghi'])

    return df_out


def simplified_solis(apparent_elevation, aod700=0.1, precipitable_water=1.,
                     pressure=101325., dni_extra=1364.):
    """
    Calculate the clear sky GHI, DNI, and DHI according to the
    simplified Solis model.

    Reference [1]_ describes the accuracy of the model as being 15, 20,
    and 18 W/m^2 for the beam, global, and diffuse components. Reference
    [2]_ provides comparisons with other clear sky models.

    Parameters
    ----------
    apparent_elevation : numeric
        The apparent elevation of the sun above the horizon (deg).

    aod700 : numeric, default 0.1
        The aerosol optical depth at 700 nm (unitless).
        Algorithm derived for values between 0 and 0.45.

    precipitable_water : numeric, default 1.0
        The precipitable water of the atmosphere (cm).
        Algorithm derived for values between 0.2 and 10 cm.
        Values less than 0.2 will be assumed to be equal to 0.2.

    pressure : numeric, default 101325.0
        The atmospheric pressure (Pascals).
        Algorithm derived for altitudes between sea level and 7000 m,
        or 101325 and 41000 Pascals.

    dni_extra : numeric, default 1364.0
        Extraterrestrial irradiance. The units of ``dni_extra``
        determine the units of the output.

    Returns
    -------
    clearsky : DataFrame (if Series input) or OrderedDict of arrays
        DataFrame/OrderedDict contains the columns/keys
        ``'dhi', 'dni', 'ghi'``.

    References
    ----------
    .. [1] P. Ineichen, "A broadband simplified version of the
       Solis clear sky model," Solar Energy, 82, 758-762 (2008).

    .. [2] P. Ineichen, "Validation of models that estimate the clear
       sky global and beam solar irradiance," Solar Energy, 132,
       332-344 (2016).
    """

    p = pressure

    w = precipitable_water

    # algorithm fails for pw < 0.2
    w = np.maximum(w, 0.2)

    # this algorithm is reasonably fast already, but it could be made
    # faster by precalculating the powers of aod700, the log(p/p0), and
    # the log(w) instead of repeating the calculations as needed in each
    # function

    i0p = _calc_i0p(dni_extra, w, aod700, p)

    taub = _calc_taub(w, aod700, p)
    b = _calc_b(w, aod700)

    taug = _calc_taug(w, aod700, p)
    g = _calc_g(w, aod700)

    taud = _calc_taud(w, aod700, p)
    d = _calc_d(aod700, p)

    # this prevents the creation of nans at night instead of 0s
    # it's also friendly to scalar and series inputs
    sin_elev = np.maximum(1.e-30, np.sin(np.radians(apparent_elevation)))

    dni = i0p * np.exp(-taub/sin_elev**b)
    ghi = i0p * np.exp(-taug/sin_elev**g) * sin_elev
    dhi = i0p * np.exp(-taud/sin_elev**d)

    irrads = OrderedDict()
    irrads['ghi'] = ghi
    irrads['dni'] = dni
    irrads['dhi'] = dhi

    if isinstance(dni, pd.Series):
        irrads = pd.DataFrame.from_dict(irrads)

    return irrads


def _calc_i0p(i0, w, aod700, p):
    """Calculate the "enhanced extraterrestrial irradiance"."""
    p0 = 101325.
    io0 = 1.08 * w**0.0051
    i01 = 0.97 * w**0.032
    i02 = 0.12 * w**0.56
    i0p = i0 * (i02*aod700**2 + i01*aod700 + io0 + 0.071*np.log(p/p0))

    return i0p


def _calc_taub(w, aod700, p):
    """Calculate the taub coefficient"""
    p0 = 101325.
    tb1 = 1.82 + 0.056*np.log(w) + 0.0071*np.log(w)**2
    tb0 = 0.33 + 0.045*np.log(w) + 0.0096*np.log(w)**2
    tbp = 0.0089*w + 0.13

    taub = tb1*aod700 + tb0 + tbp*np.log(p/p0)

    return taub


def _calc_taub(w, aod700, p):
    """Calculate the taub coefficient"""
    p0 = 101325.
    tb1 = 1.82 + 0.056*np.log(w) + 0.0071*np.log(w)**2
    tb0 = 0.33 + 0.045*np.log(w) + 0.0096*np.log(w)**2
    tbp = 0.0089*w + 0.13

    taub = tb1*aod700 + tb0 + tbp*np.log(p/p0)

    return taub


def _calc_b(w, aod700):
    """Calculate the b coefficient."""

    b1 = 0.00925*aod700**2 + 0.0148*aod700 - 0.0172
    b0 = -0.7565*aod700**2 + 0.5057*aod700 + 0.4557

    b = b1 * np.log(w) + b0

    return b


def _calc_taug(w, aod700, p):
    """Calculate the taug coefficient"""
    p0 = 101325.
    tg1 = 1.24 + 0.047*np.log(w) + 0.0061*np.log(w)**2
    tg0 = 0.27 + 0.043*np.log(w) + 0.0090*np.log(w)**2
    tgp = 0.0079*w + 0.1
    taug = tg1*aod700 + tg0 + tgp*np.log(p/p0)

    return taug


def _calc_taub(w, aod700, p):
    """Calculate the taub coefficient"""
    p0 = 101325.
    tb1 = 1.82 + 0.056*np.log(w) + 0.0071*np.log(w)**2
    tb0 = 0.33 + 0.045*np.log(w) + 0.0096*np.log(w)**2
    tbp = 0.0089*w + 0.13

    taub = tb1*aod700 + tb0 + tbp*np.log(p/p0)

    return taub


def _calc_g(w, aod700):
    """Calculate the g coefficient."""

    g = -0.0147*np.log(w) - 0.3079*aod700**2 + 0.2846*aod700 + 0.3798

    return g


def _calc_taud(w, aod700, p):
    """Calculate the taud coefficient."""

    # isscalar tests needed to ensure that the arrays will have the
    # right shape in the tds calculation.
    # there's probably a better way to do this.

    if np.isscalar(w) and np.isscalar(aod700):
        w = np.array([w])
        aod700 = np.array([aod700])
    elif np.isscalar(w):
        w = np.full_like(aod700, w)
    elif np.isscalar(aod700):
        aod700 = np.full_like(w, aod700)

    # set up nan-tolerant masks
    aod700_lt_0p05 = np.full_like(aod700, False, dtype='bool')
    np.less(aod700, 0.05, where=~np.isnan(aod700), out=aod700_lt_0p05)
    aod700_mask = np.array([aod700_lt_0p05, ~aod700_lt_0p05], dtype=int)

    # create tuples of coefficients for
    # aod700 < 0.05, aod700 >= 0.05
    td4 = 86*w - 13800, -0.21*w + 11.6
    td3 = -3.11*w + 79.4, 0.27*w - 20.7
    td2 = -0.23*w + 74.8, -0.134*w + 15.5
    td1 = 0.092*w - 8.86, 0.0554*w - 5.71
    td0 = 0.0042*w + 3.12, 0.0057*w + 2.94
    tdp = -0.83*(1+aod700)**(-17.2), -0.71*(1+aod700)**(-15.0)

    tds = (np.array([td0, td1, td2, td3, td4, tdp]) * aod700_mask).sum(axis=1)

    p0 = 101325.
    taud = (tds[4]*aod700**4 + tds[3]*aod700**3 + tds[2]*aod700**2 +
            tds[1]*aod700 + tds[0] + tds[5]*np.log(p/p0))

    # be polite about matching the output type to the input type(s)
    if len(taud) == 1:
        taud = taud[0]

    return taud


def _calc_d(aod700, p):
    """Calculate the d coefficient."""

    p0 = 101325.
    dp = 1/(18 + 152*aod700)
    d = -0.337*aod700**2 + 0.63*aod700 + 0.116 + dp*np.log(p/p0)

    return d


