import numpy as np
import pandas as pd
from warnings import warn
import logging  

def spectral_factor_sapm(airmass_absolute, module):
    """
    Calculates the SAPM spectral loss coefficient, F1.

    Parameters
    ----------
    airmass_absolute : numeric
        Absolute airmass

    module : dict-like
        A dict, Series, or DataFrame defining the SAPM performance
        parameters. See the :py:func:`sapm` notes section for more
        details.

    Returns
    -------
    F1 : numeric
        The SAPM spectral loss coefficient.

    Notes
    -----
    nan airmass values will result in 0 output.
    """

    am_coeff = [module['A4'], module['A3'], module['A2'], module['A1'],
                module['A0']]

    spectral_loss = np.polyval(am_coeff, airmass_absolute)

    spectral_loss = np.where(np.isnan(spectral_loss), 0, spectral_loss)

    spectral_loss = np.maximum(0, spectral_loss)

    if isinstance(airmass_absolute, pd.Series):
        spectral_loss = pd.Series(spectral_loss, airmass_absolute.index)

    return spectral_loss


def spectral_factor_firstsolar(precipitable_water, airmass_absolute,
                               module_type=None, coefficients=None,
                               min_precipitable_water=0.1,
                               max_precipitable_water=8):
    r"""
    Spectral mismatch modifier based on precipitable water and absolute
    (pressure-adjusted) airmass.

    Estimates a spectral mismatch modifier :math:`M` representing the effect on
    module short circuit current of variation in the spectral
    irradiance. :math:`M`  is estimated from absolute (pressure currected) air
    mass, :math:`AM_a`, and precipitable water, :math:`Pw`, using the following
    function:

    .. math::

        M = c_1 + c_2 AM_a  + c_3 Pw  + c_4 AM_a^{0.5}
            + c_5 Pw^{0.5} + c_6 \frac{AM_a} {Pw^{0.5}}

    Default coefficients are determined for several cell types with
    known quantum efficiency curves, by using the Simple Model of the
    Atmospheric Radiative Transfer of Sunshine (SMARTS) [1]_. Using
    SMARTS, spectrums are simulated with all combinations of AMa and
    Pw where:

       * :math:`0.5 \textrm{cm} <= Pw <= 5 \textrm{cm}`
       * :math:`1.0 <= AM_a <= 5.0`
       * Spectral range is limited to that of CMP11 (280 nm to 2800 nm)
       * spectrum simulated on a plane normal to the sun
       * All other parameters fixed at G173 standard

    From these simulated spectra, M is calculated using the known
    quantum efficiency curves. Multiple linear regression is then
    applied to fit Eq. 1 to determine the coefficients for each module.

    Based on the xm_solarlib Matlab function ``pvl_FSspeccorr`` by Mitchell
    Lee and Alex Panchula of First Solar, 2016 [2]_.

    Parameters
    ----------
    precipitable_water : numeric
        atmospheric precipitable water. [cm]

    airmass_absolute : numeric
        absolute (pressure-adjusted) airmass. [unitless]

    module_type : str, optional
        a string specifying a cell type. Values of 'cdte', 'monosi', 'xsi',
        'multisi', and 'polysi' (can be lower or upper case). If provided,
        module_type selects default coefficients for the following modules:

            * 'cdte' - First Solar Series 4-2 CdTe module.
            * 'monosi', 'xsi' - First Solar TetraSun module.
            * 'multisi', 'polysi' - anonymous multi-crystalline silicon module.
            * 'cigs' - anonymous copper indium gallium selenide module.
            * 'asi' - anonymous amorphous silicon module.

        The module used to calculate the spectral correction
        coefficients corresponds to the Multi-crystalline silicon
        Manufacturer 2 Model C from [3]_. The spectral response (SR) of CIGS
        and a-Si modules used to derive coefficients can be found in [4]_

    coefficients : array-like, optional
        Allows for entry of user-defined spectral correction
        coefficients. Coefficients must be of length 6. Derivation of
        coefficients requires use of SMARTS and PV module quantum
        efficiency curve. Useful for modeling PV module types which are
        not included as defaults, or to fine tune the spectral
        correction to a particular PV module. Note that the parameters for
        modules with very similar quantum efficiency should be similar,
        in most cases limiting the need for module specific coefficients.

    min_precipitable_water : float, default 0.1
        minimum atmospheric precipitable water. Any ``precipitable_water``
        value lower than ``min_precipitable_water``
        is set to ``min_precipitable_water`` to avoid model divergence. [cm]

    max_precipitable_water : float, default 8
        maximum atmospheric precipitable water. Any ``precipitable_water``
        value greater than ``max_precipitable_water``
        is set to ``np.nan`` to avoid model divergence. [cm]

    Returns
    -------
    modifier: array-like
        spectral mismatch factor (unitless) which can be multiplied
        with broadband irradiance reaching a module's cells to estimate
        effective irradiance, i.e., the irradiance that is converted to
        electrical current.

    References
    ----------
    .. [1] Gueymard, Christian. SMARTS2: a simple model of the atmospheric
       radiative transfer of sunshine: algorithms and performance
       assessment. Cocoa, FL: Florida Solar Energy Center, 1995.
    .. [2] Lee, Mitchell, and Panchula, Alex. "Spectral Correction for
       Photovoltaic Module Performance Based on Air Mass and Precipitable
       Water." IEEE Photovoltaic Specialists Conference, Portland, 2016
    .. [3] Marion, William F., et al. User's Manual for Data for Validating
       Models for PV Module Performance. National Renewable Energy
       Laboratory, 2014. http://www.nrel.gov/docs/fy14osti/61610.pdf
    .. [4] Schweiger, M. and Hermann, W, Influence of Spectral Effects
       on Energy Yield of Different PV Modules: Comparison of Pwat and
       MMF Approach, TUV Rheinland Energy GmbH report 21237296.003,
       January 2017
    """

    # --- Screen Input Data ---

    # *** Pw ***
    # Replace Pw Values below 0.1 cm with 0.1 cm to prevent model from
    # diverging"
    pw = np.atleast_1d(precipitable_water)
    pw = pw.astype('float64')
    if np.min(pw) < min_precipitable_water:
        pw = np.maximum(pw, min_precipitable_water)
        warn('Exceptionally low pw values replaced with '
             f'{min_precipitable_water} cm to prevent model divergence')

    # Warn user about Pw data that is exceptionally high
    if np.max(pw) > max_precipitable_water:
        pw[pw > max_precipitable_water] = np.nan
        warn('Exceptionally high pw values replaced by np.nan: '
             'check input data.')

    # *** AMa ***
    # Replace Extremely High AM with AM 10 to prevent model divergence
    # AM > 10 will only occur very close to sunset
    if np.max(airmass_absolute) > 10:
        airmass_absolute = np.minimum(airmass_absolute, 10)

    # Warn user about AMa data that is exceptionally low
    if np.min(airmass_absolute) < 0.58:
        warn('Exceptionally low air mass: ' +
             'model not intended for extra-terrestrial use')
        # pvl_absoluteairmass(1,pvl_alt2pres(4340)) = 0.58 Elevation of
        # Mina Pirquita, Argentian = 4340 m. Highest elevation city with
        # population over 50,000.

    _coefficients = {}
    _coefficients['cdte'] = (
        0.86273, -0.038948, -0.012506, 0.098871, 0.084658, -0.0042948)
    _coefficients['monosi'] = (
        0.85914, -0.020880, -0.0058853, 0.12029, 0.026814, -0.0017810)
    _coefficients['xsi'] = _coefficients['monosi']
    _coefficients['polysi'] = (
        0.84090, -0.027539, -0.0079224, 0.13570, 0.038024, -0.0021218)
    _coefficients['multisi'] = _coefficients['polysi']
    _coefficients['cigs'] = (
        0.85252, -0.022314, -0.0047216, 0.13666, 0.013342, -0.0008945)
    _coefficients['asi'] = (
        1.12094, -0.047620, -0.0083627, -0.10443, 0.098382, -0.0033818)

    if module_type is not None and coefficients is None:
        coefficients = _coefficients[module_type.lower()]
    elif module_type is None and coefficients is not None:
        logging.warning('pass')
    elif module_type is None and coefficients is None:
        raise TypeError('No valid input provided, both module_type and ' +
                        'coefficients are None')
    else:
        raise TypeError('Cannot resolve input, must supply only one of ' +
                        'module_type and coefficients')

    # Evaluate Spectral Shift
    coeff = coefficients
    ama = airmass_absolute
    modifier = (
        coeff[0] + coeff[1]*ama + coeff[2]*pw + coeff[3]*np.sqrt(ama) +
        coeff[4]*np.sqrt(pw) + coeff[5]*ama/np.sqrt(pw))

    return modifier
