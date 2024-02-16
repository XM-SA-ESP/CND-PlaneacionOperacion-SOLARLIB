import numpy as np
import pandas as pd
from collections import OrderedDict
from xm_solarlib import solarposition
from functools import partial
import datetime
import logging



from . import tools, atmosphere

SURFACE_ALBEDOS = {
    "urban": 0.18,
    "grass": 0.20,
    "fresh grass": 0.26,
    "soil": 0.17,
    "sand": 0.40,
    "snow": 0.65,
    "fresh snow": 0.75,
    "asphalt": 0.12,
    "concrete": 0.30,
    "aluminum": 0.85,
    "copper": 0.74,
    "fresh steel": 0.35,
    "dirty steel": 0.08,
    "sea": 0.06,
}



def get_extra_radiation(datetime_or_doy, solar_constant=1366.1,
                        epoch_year=2014):
    """
    Determina la radiación extraterrestre a partir del día del año.

    Parámetros
    ----------
    datetime_or_doy : numérico, array, fecha, datetime, Timestamp, DatetimeIndex
        Día del año, array de días del año o objeto similar a datetime.

    solar_constant : float, por defecto 1366.1
        La constante solar.

    epoch_year : int, por defecto 2014
        El año en el que se calculará una entrada del día del año. Solo
        aplica a la entrada del día del año utilizada con los métodos pyephem o nrel.

    kwargs :
        Pasados a solarposition.nrel_earthsun_distance

    Retorna
    -------
    dni_extra : float, array, o Serie
        La radiación extraterrestre presente en vatios por metro cuadrado
        en una superficie que es normal al sol. Las entradas de Timestamp y
        DatetimeIndex de Pandas producirán una Serie de Pandas. Todas las demás
        entradas producirán un float o un array de floats.

    Referencias
    ----------
    .. [1] M. Reno, C. Hansen, y J. Stein, "Modelos de Cielo Claro de Irradiación Horizontal Global: Implementación y Análisis", Laboratorios Nacionales de Sandia, SAND2012-2389, 2012.

    .. [2] <http://solardat.uoregon.edu/SolarRadiationBasics.html>, Ecs. SR1 y SR2

    .. [3] Partridge, G. W. y Platt, C. M. R. 1976. Procesos Radiativos en Meteorología y Climatología.

    .. [4] Duffie, J. A. y Beckman, W. A. 1991. Ingeniería Solar de Procesos Térmicos, 2a ed. J. Wiley and Sons, Nueva York.

    .. [5] ASCE, 2005. La Ecuación de Referencia Estándar de Evapotranspiración de la ASCE, Instituto de Recursos Hídricos y Ambientales de la Sociedad Americana de Ingenieros Civiles, Ed. R. G. Allen et al.
    """

    to_doy, to_datetimeindex, to_output = _handle_extra_radiation_types(
        datetime_or_doy, epoch_year
    )


    B = solarposition._calculate_simple_day_angle(to_doy(datetime_or_doy))
    roverr0sqrd = (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) +
                    0.000719 * np.cos(2 * B) + 7.7e-05 * np.sin(2 * B))

    ea = solar_constant * roverr0sqrd

    ea = to_output(ea)

    return ea


def _handle_extra_radiation_types(datetime_or_doy, epoch_year):
    # Este bloque establecerá las funciones que se pueden usar para convertir
    # las entradas en día del año o pandas DatetimeIndex, y las
    # funciones que producirán el tipo de salida adecuado. Es
    # complicado porque hay muchos tipos de entrada similares al día del año,
    # y los diferentes algoritmos necesitan diferentes tipos. Tal vez tengas
    # una mejor manera de hacerlo.
    if isinstance(datetime_or_doy, pd.DatetimeIndex):
        to_doy = tools._pandas_to_doy  # won't be evaluated unless necessary

        def to_datetimeindex(x):
            return x  # noqa: E306

        to_output = partial(pd.Series, index=datetime_or_doy)
    elif isinstance(datetime_or_doy, pd.Timestamp):
        to_doy = tools._pandas_to_doy
        to_datetimeindex = tools._datetimelike_scalar_to_datetimeindex
        to_output = tools._scalar_out
    elif isinstance(datetime_or_doy, (datetime.date, datetime.datetime, np.datetime64)):
        to_doy = tools._datetimelike_scalar_to_doy
        to_datetimeindex = tools._datetimelike_scalar_to_datetimeindex
        to_output = tools._scalar_out
    elif np.isscalar(datetime_or_doy):  # enteros y flotantes de varios tipos

        def to_doy(x):
            return x  # noqa: E306

        to_datetimeindex = partial(tools._doy_to_datetimeindex, epoch_year=epoch_year)
        to_output = tools._scalar_out
    else:  # assume that we have an array-like object of doy

        def to_doy(x):
            return x  # noqa: E306

        to_datetimeindex = partial(tools._doy_to_datetimeindex, epoch_year=epoch_year)
        to_output = tools._array_out

    return to_doy, to_datetimeindex, to_output


def get_total_irradiance(
    surface_tilt,
    surface_azimuth,
    solar_zenith,
    solar_azimuth,
    dni,
    ghi,
    dhi,
    dni_extra=None,
    airmass=None,
    albedo=0.25,
    surface_type=None,
    model="isotropic",
    model_perez="allsitescomposite1990",
):
    r"""
    Determina la irradiación total en plano y sus componentes de haz, difusión
    del cielo y reflexión del suelo, utilizando el modelo de irradiación difusa
    del cielo especificado.

    .. math::

       I_{tot} = I_{beam} + I_{sky diffuse} + I_{ground}

    Modelos de difusión del cielo incluyen:
        * isotrópico (predeterminado)
        * klucher
        * haydavies
        * reindl
        * king
        * perez

    Parámetros
    ----------
    surface_tilt : numérico
        Inclinación del panel desde la horizontal. [grado]
    surface_azimuth : numérico
        Azimut del panel desde el norte. [grado]
    solar_zenith : numérico
        Ángulo cenital solar. [grado]
    solar_azimuth : numérico
        Ángulo de azimut solar. [grado]
    dni : numérico
        Irradiancia Normal Directa. [W/m2]
    ghi : numérico
        Irradiancia horizontal global. [W/m2]
    dhi : numérico
        Irradiancia horizontal difusa. [W/m2]
    dni_extra : Ninguno o numérico, por defecto Ninguno
        Irradiancia normal directa extraterrestre. [W/m2]
    airmass : Ninguno o numérico, por defecto Ninguno
        Masa de aire relativa (no ajustada por presión). [sin unidades]
    albedo : numérico, por defecto 0.25
        Albedo de la superficie del suelo. [sin unidades]
    surface_type : Ninguno o str, por defecto Ninguno
        Tipo de superficie. Ver :py:func:`~pvlib.irradiance.get_ground_diffuse` para la lista de valores aceptados.
    model : str, por defecto 'isotrópico'
        Modelo de irradiación. Puede ser uno de ``'isotrópico'``, ``'klucher'``, ``'haydavies'``, ``'reindl'``, ``'king'``, ``'perez'``.
    model_perez : str, por defecto 'allsitescomposite1990'
        Usado solo si ``model='perez'``. Ver :py:func:`~pvlib.irradiance.perez`.

    Devuelve
    -------
    total_irrad : OrderedDict o DataFrame
        Contiene claves/columnas ``'poa_global', 'poa_direct', 'poa_diffuse', 'poa_sky_diffuse', 'poa_ground_diffuse'``.

    Notas
    -----
    Los modelos ``'haydavies'``, ``'reindl'``, o ``'perez'`` requieren ``'dni_extra'``. Los valores pueden calcularse usando :py:func:`~pvlib.irradiance.get_extra_radiation`.

    El modelo ``'perez'`` requiere masa de aire relativa (``airmass``) como entrada. Si ``airmass`` no se proporciona, se calcula usando los valores predeterminados en :py:func:`~pvlib.atmosphere.get_relative_airmass`.

    """

    poa_sky_diffuse = get_sky_diffuse(
        surface_tilt,
        surface_azimuth,
        solar_zenith,
        solar_azimuth,
        dni,
        dhi,
        dni_extra=dni_extra,
        airmass=airmass,
        model=model,
        model_perez=model_perez,
    )

    poa_ground_diffuse = get_ground_diffuse(surface_tilt, ghi, albedo, surface_type)
    aoi_ = aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    irrads = poa_components(aoi_, dni, poa_sky_diffuse, poa_ground_diffuse)
    return irrads


def get_sky_diffuse(
    surface_tilt,
    surface_azimuth,
    solar_zenith,
    solar_azimuth,
    dni,
    dhi,
    dni_extra=None,
    airmass=None,
    model="isotropic",
    model_perez="allsitescomposite1990",
):
    r"""
    Determina la componente de irradiación difusa del cielo en un plano utilizando el modelo de irradiación difusa del cielo especificado.

    Modelos de difusión del cielo incluyen:
        * isotrópico (predeterminado)
        * klucher
        * haydavies
        * reindl
        * king
        * perez

    Parámetros
    ----------
    surface_tilt : numérico
        Inclinación del panel desde la horizontal. [grado]
    surface_azimuth : numérico
        Azimut del panel desde el norte. [grado]
    solar_zenith : numérico
        Ángulo cenital solar. [grado]
    solar_azimuth : numérico
        Ángulo de azimut solar. [grado]
    dni : numérico
        Irradiancia Normal Directa. [W/m2]
    ghi : numérico
        Irradiancia horizontal global. [W/m2]
    dhi : numérico
        Irradiancia horizontal difusa. [W/m2]
    dni_extra : Ninguno o numérico, por defecto Ninguno
        Irradiancia normal directa extraterrestre. [W/m2]
    airmass : Ninguno o numérico, por defecto Ninguno
        Masa de aire relativa (no ajustada por presión). [sin unidades]
    model : str, por defecto 'isotrópico'
        Modelo de irradiación. Puede ser uno de ``'isotrópico'``, ``'klucher'``, ``'haydavies'``, ``'reindl'``, ``'king'``, ``'perez'``.
    model_perez : str, por defecto 'allsitescomposite1990'
        Usado solo si ``model='perez'``. Ver :py:func:`~pvlib.irradiance.perez`.

    Retorna
    -------
    poa_sky_diffuse : numérico
        Irradiación difusa del cielo en el plano del arreglo. [W/m2]

    Lanza
    ------
    ValueError
        Si el modelo es uno de ``'haydavies'``, ``'reindl'``, o ``'perez'`` y ``dni_extra`` es ``Ninguno``.

    Notas
    -----
    Los modelos ``'haydavies'``, ``'reindl'``, y ``'perez'``` requieren 'dni_extra'. Los valores pueden calcularse usando :py:func:`~pvlib.irradiance.get_extra_radiation`.

    El modelo ``'perez'`` requiere masa de aire relativa (``airmass``) como entrada. Si ``airmass`` no se proporciona, se calcula usando los valores predeterminados en :py:func:`~pvlib.atmosphere.get_relative_airmass`.

    """

    model = model.lower()

    if (model in {"haydavies", "reindl", "perez"}) and (dni_extra is None):
        raise ValueError(f"dni_extra is required for model {model}")

    if model == "isotropic":
        sky = isotropic(surface_tilt, dhi)

    elif model == 'haydavies':
        sky = haydavies(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                        solar_zenith, solar_azimuth)
    elif model == 'perez':
        if airmass is None:
            airmass = atmosphere.get_relative_airmass(solar_zenith)
        sky = perez(
            surface_tilt,
            surface_azimuth,
            dhi,
            dni,
            dni_extra,
            solar_zenith,
            solar_azimuth,
            airmass,
            model=model_perez,
        )
    else:
        raise ValueError(f"invalid model selection {model}")

    return sky


def isotropic(surface_tilt, dhi):
    r"""
    Determina la irradiación difusa del cielo en una superficie inclinada utilizando
    el modelo de cielo isotrópico.

    .. math::

       I_{d} = DHI \frac{1 + \cos\beta}{2}

    El modelo de Hottel y Woertz trata el cielo como una fuente uniforme de
    irradiación difusa. Por lo tanto, la irradiación difusa del cielo (la irradiación reflejada por el suelo no está incluida en este algoritmo) en una superficie inclinada se puede encontrar a partir de la irradiación horizontal difusa y el ángulo de inclinación de la superficie. Una discusión sobre el origen del modelo isotrópico se puede encontrar en [2]_.

    Parámetros
    ----------
    surface_tilt : numérico
        Ángulo de inclinación de la superficie en grados decimales. La inclinación debe ser >=0 y <=180. El ángulo de inclinación se define como grados desde la horizontal (por ejemplo, superficie hacia arriba = 0, superficie hacia el horizonte = 90)

    dhi : numérico
        Irradiación horizontal difusa en W/m^2. DHI debe ser >=0.

    Retorna
    -------
    diffuse : numérico
        La componente difusa del cielo de la radiación solar.

    Referencias
    ----------
    .. [1] Loutzenhiser P.G. et al. "Validación empírica de modelos para calcular la irradiación solar en superficies inclinadas para la simulación de energía en edificios" 2007, Solar Energy vol. 81. pp. 254-267 :doi:`10.1016/j.solener.2006.03.009`

    .. [2] Kamphuis, N.R. et al. "Perspectivas sobre el origen, derivación, significado e importancia del modelo de cielo isotrópico" 2020, Solar Energy vol. 201. pp. 8-12 :doi:`10.1016/j.solener.2020.02.067`
    """
    sky_diffuse = dhi * (1 + tools.cosd(surface_tilt)) * 0.5

    return sky_diffuse


def klucher(surface_tilt, surface_azimuth, dhi, ghi, solar_zenith, solar_azimuth):
    r"""
    Determina la irradiación difusa del cielo en una superficie inclinada
    utilizando el modelo de Klucher de 1979.

    .. math::

       I_{d} = DHI \frac{1 + \cos\beta}{2} (1 + F' \sin^3(\beta/2))
       (1 + F' \cos^2\theta\sin^3\theta_z)

    donde

    .. math::

       F' = 1 - (I_{d0} / GHI)^2

    El modelo de Klucher de 1979 determina la irradiación difusa del cielo
    (la irradiación reflejada por el suelo no está incluida en este algoritmo) en una superficie inclinada utilizando el ángulo de inclinación de la superficie, el ángulo de azimut de la superficie, la irradiación horizontal difusa, la irradiación normal directa, la irradiación global horizontal, la irradiación extraterrestre, el ángulo cenital solar y el ángulo de azimut solar.

    Parámetros
    ----------
    surface_tilt : numérico
        Ángulos de inclinación de la superficie en grados decimales. surface_tilt debe ser >=0 y <=180. El ángulo de inclinación se define como grados desde la horizontal (por ejemplo, superficie hacia arriba = 0, superficie hacia el horizonte = 90).

    surface_azimuth : numérico
        Ángulos de azimut de la superficie en grados decimales. surface_azimuth debe ser >=0 y <=360. La convención de azimut se define como grados al este del norte (por ejemplo, Norte = 0, Sur = 180, Este = 90, Oeste = 270).

    dhi : numérico
        Irradiación horizontal difusa en W/m^2. DHI debe ser >=0.

    ghi : numérico
        Irradiación global en W/m^2. DNI debe ser >=0.

    solar_zenith : numérico
        Ángulos cenitales solares aparentes (corregidos por refracción) en grados decimales. solar_zenith debe ser >=0 y <=180.

    solar_azimuth : numérico
        Ángulos de azimut solar en grados decimales. solar_azimuth debe ser >=0 y <=360. La convención de azimut se define como grados al este del norte (por ejemplo, Norte = 0, Este = 90, Oeste = 270).

    Retorna
    -------
    diffuse : numérico
        La componente difusa del cielo de la radiación solar.

    Referencias
    ----------
    .. [1] Loutzenhiser P.G. et al. "Validación empírica de modelos para calcular la irradiación solar en superficies inclinadas para la simulación de energía en edificios" 2007, Solar Energy vol. 81. pp. 254-267.

    .. [2] Klucher, T.M., 1979. Evaluación de modelos para predecir la insolación en superficies inclinadas. Solar Energy 23 (2), 111-114.
    """

    # zenith angle with respect to panel normal.
    cos_tt = aoi_projection(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    cos_tt = np.maximum(cos_tt, 0)  # GH 526

    # silence warning from 0 / 0
    with np.errstate(invalid="ignore"):
        F = 1 - ((dhi / ghi) ** 2)

    try:
        # fails with single point input
        F.fillna(0, inplace=True)
    except AttributeError:
        F = np.where(np.isnan(F), 0, F)

    term1 = 0.5 * (1 + tools.cosd(surface_tilt))
    term2 = 1 + F * (tools.sind(0.5 * surface_tilt) ** 3)
    term3 = 1 + F * (cos_tt**2) * (tools.sind(solar_zenith) ** 3)

    sky_diffuse = dhi * term1 * term2 * term3

    return sky_diffuse


def king(surface_tilt, dhi, ghi, solar_zenith):
    """
    Determina la irradiación difusa del cielo en una superficie inclinada utilizando
    el modelo King.

    El modelo de King determina la irradiación difusa del cielo (la irradiación reflejada por el suelo no está incluida en este algoritmo) en una superficie inclinada utilizando el ángulo de inclinación de la superficie, la irradiación horizontal difusa, la irradiación global horizontal y el ángulo cenital solar. Cabe destacar que este modelo no está bien documentado y no ha sido publicado de ninguna forma (hasta enero de 2012).

    Parámetros
    ----------
    surface_tilt : numérico
        Ángulos de inclinación de la superficie en grados decimales. El ángulo de inclinación se define como grados desde la horizontal (por ejemplo, superficie hacia arriba = 0, superficie hacia el horizonte = 90).

    dhi : numérico
        Irradiación horizontal difusa en W/m^2.

    ghi : numérico
        Irradiación global horizontal en W/m^2.

    solar_zenith : numérico
        Ángulos cenitales solares aparentes (corregidos por refracción) en grados decimales.

    Retorna
    --------
    poa_sky_diffuse : numérico
        La componente difusa de la radiación solar.
    """

    sky_diffuse = (
        dhi * (1 + tools.cosd(surface_tilt)) / 2
        + ghi * (0.012 * solar_zenith - 0.04) * (1 - tools.cosd(surface_tilt)) / 2
    )
    sky_diffuse = np.maximum(sky_diffuse, 0)

    return sky_diffuse


def perez(
    surface_tilt,
    surface_azimuth,
    dhi,
    dni,
    dni_extra,
    solar_zenith,
    solar_azimuth,
    airmass,
    model="allsitescomposite1990",
    return_components=False,
):
    """
    Determina la irradiación difusa del cielo en una superficie inclinada utilizando
    uno de los modelos de Perez.

    Los modelos de Perez determinan la irradiación difusa del cielo (la irradiación reflejada por el suelo no está incluida en este algoritmo) en una superficie inclinada utilizando el ángulo de inclinación de la superficie, el ángulo de azimut de la superficie, la irradiación horizontal difusa, la irradiación normal directa, la irradiación extraterrestre, el ángulo cenital solar, el ángulo de azimut solar y la masa de aire relativa (no corregida por presión). Opcionalmente, se puede usar un selector para utilizar cualquiera de los conjuntos de coeficientes del modelo de Perez.

    Parámetros
    ----------
    surface_tilt : numérico
        Ángulos de inclinación de la superficie en grados decimales. surface_tilt debe ser >=0 y <=180. El ángulo de inclinación se define como grados desde la horizontal (por ejemplo, superficie hacia arriba = 0, superficie hacia el horizonte = 90).

    surface_azimuth : numérico
        Ángulos de azimut de la superficie en grados decimales. surface_azimuth debe ser >=0 y <=360. La convención de azimut se define como grados al este del norte (por ejemplo, Norte = 0, Sur = 180, Este = 90, Oeste = 270).

    dhi : numérico
        Irradiación horizontal difusa en W/m^2. DHI debe ser >=0.

    dni : numérico
        Irradiación normal directa en W/m^2. DNI debe ser >=0.

    dni_extra : numérico
        Irradiación normal extraterrestre en W/m^2.

    solar_zenith : numérico
        Ángulos cenitales solares aparentes (corregidos por refracción) en grados decimales. solar_zenith debe ser >=0 y <=180.

    solar_azimuth : numérico
        Ángulos de azimut solar en grados decimales. solar_azimuth debe ser >=0 y <=360. La convención de azimut se define como grados al este del norte (por ejemplo, Norte = 0, Este = 90, Oeste = 270).

    airmass : numérico
        Valores de masa de aire relativa (no corregida por presión). Si AM es un DataFrame, debe ser del mismo tamaño que todas las demás entradas de DataFrame. AM debe ser >=0 (tener cuidado al usar el modelo 1/sec(z) para la generación de AM).

    model : string (opcional, por defecto='allsitescomposite1990')
        Un string que selecciona el conjunto deseado de coeficientes de Perez. Si el modelo no se proporciona como entrada, se usará el predeterminado, '1990'. Todas las posibles selecciones de modelo son:

        * '1990'
        * 'allsitescomposite1990' (igual que '1990')
        * 'allsitescomposite1988'
        * 'sandiacomposite1988'
        * 'usacomposite1988'
        * 'france1988'
        * 'phoenix1988'
        * 'elmonte1988'
        * 'osage1988'
        * 'albuquerque1988'
        * 'capecanaveral1988'
        * 'albany1988'

    return_components: bool (opcional, por defecto=False)
        Bandera utilizada para decidir si se devuelven o no los componentes difusos calculados.

    Retorna
    --------
    numérico, OrderedDict, o DataFrame
        Tipo de retorno controlado por el argumento `return_components`.
        Si ``return_components=False``, se devuelve `sky_diffuse`.
        Si ``return_components=True``, se devuelve `diffuse_components`.

    sky_diffuse : numérico
        La componente difusa del cielo de la radiación solar en una superficie inclinada.

    diffuse_components : OrderedDict (entrada de array) o DataFrame (entrada de Series)
        Claves/columnas son:
            * sky_diffuse: Difuso total del cielo
            * isotrópico
            * circumsolar
            * horizonte

    Referencias
    ----------
    .. [1] Loutzenhiser P.G. et al. "Validación empírica de modelos para calcular la irradiación solar en superficies inclinadas para la simulación de energía en edificios" 2007, Solar Energy vol. 81. pp. 254-267

    .. [2] Perez, R., Seals, R., Ineichen, P., Stewart, R., Menicucci, D.,
       1987. Una nueva versión simplificada del modelo de irradiación difusa de Perez para superficies inclinadas. Solar Energy 39(3), 221-232.

    .. [3] Perez, R., Ineichen, P., Seals, R., Michalsky, J., Stewart, R.,
       1990. Modelado de la disponibilidad de luz diurna y los componentes de irradiación a partir de la irradiación directa y global. Solar Energy 44 (5), 271-289.

    .. [4] Perez, R. et al 1988. "El Desarrollo y Verificación del Modelo de Radiación Difusa de Perez". SAND88-7030
    """

    kappa = 1.041  # for solar_zenith in radians
    z = np.radians(solar_zenith)  # convert to radians

    # delta is the sky's "brightness"
    delta = dhi * airmass / dni_extra

    # epsilon is the sky's "clearness"
    with np.errstate(invalid="ignore"):
        eps = ((dhi + dni) / dhi + kappa * (z**3)) / (1 + kappa * (z**3))

    # numpy indexing below will not work with a Series
    if isinstance(eps, pd.Series):
        eps = eps.values

    # Perez et al define clearness bins according to the following
    # rules. 1 = overcast ... 8 = clear (these names really only make
    # sense for small zenith angles, but...) these values will
    # eventually be used as indicies for coeffecient look ups
    ebin = np.digitize(eps, (0.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2))
    ebin = np.array(ebin)  # GH 642
    ebin[np.isnan(eps)] = 0

    # correct for 0 indexing in coeffecient lookup
    # later, ebin = -1 will yield nan coefficients
    ebin -= 1

    # The various possible sets of Perez coefficients are contained
    # in a subfunction to clean up the code.
    f1c, f2c = _get_perez_coefficients(model)

    # results in invalid eps (ebin = -1) being mapped to nans
    nans = np.array([np.nan, np.nan, np.nan])
    f1c = np.vstack((f1c, nans))
    f2c = np.vstack((f2c, nans))

    f1 = f1c[ebin, 0] + f1c[ebin, 1] * delta + f1c[ebin, 2] * z
    f1 = np.maximum(f1, 0)

    f2 = f2c[ebin, 0] + f2c[ebin, 1] * delta + f2c[ebin, 2] * z

    A = aoi_projection(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    A = np.maximum(A, 0)

    B = tools.cosd(solar_zenith)
    B = np.maximum(B, tools.cosd(85))

    # Calculate Diffuse POA from sky dome
    term1 = 0.5 * (1 - f1) * (1 + tools.cosd(surface_tilt))
    term2 = f1 * A / B
    term3 = f2 * tools.sind(surface_tilt)

    sky_diffuse = np.maximum(dhi * (term1 + term2 + term3), 0)

    # we've preserved the input type until now, so don't ruin it!
    if isinstance(sky_diffuse, pd.Series):
        sky_diffuse[np.isnan(airmass)] = 0
    else:
        sky_diffuse = np.where(np.isnan(airmass), 0, sky_diffuse)

    if return_components:
        diffuse_components = OrderedDict()
        diffuse_components["sky_diffuse"] = sky_diffuse

        # Calculate the different components
        diffuse_components["isotropic"] = dhi * term1
        diffuse_components["circumsolar"] = dhi * term2
        diffuse_components["horizon"] = dhi * term3

        # Set values of components to 0 when sky_diffuse is 0
        mask = sky_diffuse == 0
        if isinstance(sky_diffuse, pd.Series):
            diffuse_components = pd.DataFrame(diffuse_components)
            diffuse_components.loc[mask] = 0
        else:
            diffuse_components = {
                k: np.where(mask, 0, v) for k, v in diffuse_components.items()
            }
        return diffuse_components
    else:
        return sky_diffuse


def aoi_projection(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    """
    Calcula el producto punto del vector unitario de posición solar y el vector
    unitario normal a la superficie; en otras palabras, el coseno del ángulo de incidencia.

    Nota de uso: Cuando el sol está detrás de la superficie, el valor devuelto es
    negativo. Para muchos usos, los valores negativos deben establecerse en cero.

    Introduce todos los ángulos en grados.

    Parámetros
    ----------
    surface_tilt : numérico
        Inclinación del panel desde la horizontal.
    surface_azimuth : numérico
        Azimut del panel desde el norte.
    solar_zenith : numérico
        Ángulo cenital solar.
    solar_azimuth : numérico
        Ángulo de azimut solar.

    Devuelve
    -------
    projection : numérico
        Producto punto de la normal del panel y el ángulo solar.
    """

    projection = tools.cosd(surface_tilt) * tools.cosd(solar_zenith) + tools.sind(
        surface_tilt
    ) * tools.sind(solar_zenith) * tools.cosd(solar_azimuth - surface_azimuth)

    # GH 1185
    projection = np.clip(projection, -1, 1)

    try:
        projection.name = "aoi_projection"
    except AttributeError:
        pass

    return projection


def haydavies(
    surface_tilt,
    surface_azimuth,
    dhi,
    dni,
    dni_extra,
    solar_zenith=None,
    solar_azimuth=None,
    projection_ratio=None,
    return_components=False,
):
    """
    Determina la irradiancia difusa del cielo en una superficie inclinada utilizando
    el modelo de Hay & Davies de 1980.

    .. math::
        I_{d} = DHI ( A R_b + (1 - A) (\frac{1 + \cos\beta}{2}) )

    El modelo de Hay y Davies de 1980 determina la irradiancia difusa del
    cielo (la irradiancia reflejada del suelo no está incluida en este
    algoritmo) en una superficie inclinada utilizando el ángulo de inclinación
    de la superficie, el ángulo de azimut de la superficie, la irradiancia
    horizontal difusa, la irradiancia normal directa, la irradiancia
    extraterrestre normal, el ángulo cenital solar y el ángulo de azimut solar.

    Parámetros
    ----------
    surface_tilt : numérico
        Ángulos de inclinación de la superficie en grados decimales. El ángulo
        de inclinación se define como grados desde la horizontal (por ejemplo,
        superficie mirando hacia arriba = 0, superficie mirando hacia el horizonte = 90)

    surface_azimuth : numérico
        Ángulos de azimut de la superficie en grados decimales. La convención
        de azimut se define como grados al este del norte (por ejemplo, Norte = 0,
        Sur = 180, Este = 90, Oeste = 270).

    dhi : numérico
        Irradiancia horizontal difusa en W/m^2.

    dni : numérico
        Irradiancia normal directa en W/m^2.

    dni_extra : numérico
        Irradiancia normal extraterrestre en W/m^2.

    solar_zenith : Ninguno o numérico, por defecto Ninguno
        Ángulos cenitales solares aparentes (corregidos por refracción) en grados
        decimales. Se debe proporcionar ``solar_zenith`` y ``solar_azimuth`` o
        proporcionar ``projection_ratio``.

    solar_azimuth : Ninguno o numérico, por defecto Ninguno
        Ángulos de azimut solar en grados decimales. Se debe proporcionar
        ``solar_zenith`` y ``solar_azimuth`` o proporcionar ``projection_ratio``.

    projection_ratio : Ninguno o numérico, por defecto Ninguno
        Relación de proyección del ángulo de incidencia con la proyección del ángulo
        cenital solar. Se debe proporcionar ``solar_zenith`` y ``solar_azimuth`` o
        proporcionar ``projection_ratio``.

    return_components : bool, por defecto Falso
        Bandera utilizada para decidir si se devuelven o no los componentes difusos
        calculados.

    Devuelve
    --------
    numérico, OrderedDict o DataFrame
        Tipo de retorno controlado por el argumento `return_components`.
        Si ``return_components=False``, se devuelve `sky_diffuse`.
        Si ``return_components=True``, se devuelven `diffuse_components`.

    sky_diffuse : numérico
        Componente difuso del cielo de la radiación solar en una superficie inclinada.

    diffuse_components : OrderedDict (entrada de array) o DataFrame (entrada de Series)
        Claves/columnas son:
            * sky_diffuse: Difuso total del cielo
            * isotrópico
            * circumsolar
            * horizonte

    Notas
    ------
    Al proporcionar ``projection_ratio``, considere restringir sus valores
    cuando el ángulo cenital se acerca a 90 grados o la proyección del ángulo
    de incidencia es negativa. Ver código para detalles.

    Referencias
    -----------
    .. [1] Loutzenhiser P.G. et. al. "Validación empírica de modelos para
       calcular la irradiancia solar en superficies inclinadas para la simulación
       de la energía del edificio" 2007, Solar Energy vol. 81. pp. 254-267

    .. [2] Hay, J.E., Davies, J.A., 1980. Cálculos de la radiación solar
       incidente en una superficie inclinada. En: Hay, J.E., Won, T.K.
       (Eds.), Actas del Primer Taller de Datos de Radiación Solar de Canadá, 59.
       Ministerio de Suministros y Servicios, Canadá.
    """

    # if necessary, calculate ratio of titled and horizontal beam irradiance
    if projection_ratio is None:
        cos_tt = aoi_projection(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth
        )
        cos_tt = np.maximum(cos_tt, 0)  # GH 526
        cos_solar_zenith = tools.cosd(solar_zenith)
        rb = cos_tt / np.maximum(cos_solar_zenith, 0.01745)  # GH 432
    else:
        rb = projection_ratio

    # Anisotropy Index
    AI = dni / dni_extra

    # these are the () and [] sub-terms of the second term of eqn 7
    term1 = 1 - AI
    term2 = 0.5 * (1 + tools.cosd(surface_tilt))

    poa_isotropic = np.maximum(dhi * term1 * term2, 0)
    poa_circumsolar = np.maximum(dhi * (AI * rb), 0)
    sky_diffuse = poa_isotropic + poa_circumsolar

    if return_components:
        diffuse_components = OrderedDict()
        diffuse_components["sky_diffuse"] = sky_diffuse

        # Calculate the individual components
        diffuse_components["isotropic"] = poa_isotropic
        diffuse_components["circumsolar"] = poa_circumsolar
        diffuse_components["horizon"] = np.where(
            np.isnan(diffuse_components["isotropic"]), np.nan, 0.0
        )

        if isinstance(sky_diffuse, pd.Series):
            diffuse_components = pd.DataFrame(diffuse_components)
        return diffuse_components
    else:
        return sky_diffuse


def reindl(
    surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra, solar_zenith, solar_azimuth
):
    r"""
    Determina la irradiancia difusa del cielo en una superficie inclinada usando
    el modelo de Reindl de 1990.

    .. math::

       I_{d} = DHI (A R_b + (1 - A) (\frac{1 + \cos\beta}{2})
       (1 + \sqrt{\frac{I_{hb}}{I_h}} \sin^3(\beta/2)) )

    El modelo de Reindl de 1990 determina la irradiancia difusa del cielo
    (la irradiancia reflejada por el suelo no está incluida en este algoritmo) en una
    superficie inclinada utilizando el ángulo de inclinación de la superficie, el ángulo de acimut de la superficie,
    irradiancia horizontal difusa, irradiancia normal directa, irradiancia
    horizontal global, irradiancia extraterrestre, ángulo cenital solar
    y ángulo de acimut solar.

    Parámetros
    ----------
    surface_tilt : numérico
        Ángulos de inclinación de la superficie en grados decimales. El ángulo de inclinación se
        define como grados desde la horizontal (por ejemplo, superficie hacia arriba = 0,
        superficie hacia el horizonte = 90).

    surface_azimuth : numérico
        Ángulos de acimut de la superficie en grados decimales. La convención de acimut se
        define como grados al este del norte (por ejemplo, Norte = 0,
        Sur = 180, Este = 90, Oeste = 270).

    dhi : numérico
        Irradiancia horizontal difusa en W/m^2.

    dni : numérico
        Irradiancia normal directa en W/m^2.

    ghi: numérico
        Irradiancia global en W/m^2.

    dni_extra : numérico
        Irradiancia normal extraterrestre en W/m^2.

    solar_zenith : numérico
        Ángulos cenitales solares aparentes (corregidos por refracción) en grados decimales.

    solar_azimuth : numérico
        Ángulos de acimut solar en grados decimales. La convención de acimut se
        define como grados al este del norte (por ejemplo, Norte = 0, Este = 90,
        Oeste = 270).

    Retorna
    -------
    poa_sky_diffuse : numérico
        La componente difusa del cielo de la radiación solar.

    Notas
    -----
    El cálculo de poa_sky_diffuse se genera a partir del artículo de Loutzenhiser et al.
    (2007), ecuación 8. Nótese que he eliminado la porción de radiación directa y de
    reflectancia del suelo de la ecuación y esto genera ÚNICAMENTE la radiación
    difusa del cielo y circumsolar, por lo que la forma de la ecuación
    varía ligeramente de la ecuación 8.

    Referencias
    ----------
    .. [1] Loutzenhiser P.G. et. al. "Validación empírica de modelos para
       calcular la irradiancia solar en superficies inclinadas para la simulación de energía
       en edificios" 2007, Solar Energy vol. 81. pp. 254-267

    .. [2] Reindl, D.T., Beckmann, W.A., Duffie, J.A., 1990a. Correlaciones de fracción difusa.
       Solar Energy 45(1), 1-7.

    .. [3] Reindl, D.T., Beckmann, W.A., Duffie, J.A., 1990b. Evaluación de
       modelos de radiación en superficies inclinadas por hora. Solar Energy 45(1), 9-17.
    """

    cos_tt = aoi_projection(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    cos_tt = np.maximum(cos_tt, 0)  # GH 526

    # do not apply cos(zen) limit here (needed for HB below)
    cos_solar_zenith = tools.cosd(solar_zenith)

    # ratio of titled and horizontal beam irradiance
    rb = cos_tt / np.maximum(cos_solar_zenith, 0.01745)  # GH 432

    # Anisotropy Index
    AI = dni / dni_extra

    # DNI projected onto horizontal
    HB = dni * cos_solar_zenith
    HB = np.maximum(HB, 0)

    # these are the () and [] sub-terms of the second term of eqn 8
    term1 = 1 - AI
    term2 = 0.5 * (1 + tools.cosd(surface_tilt))
    with np.errstate(invalid="ignore", divide="ignore"):
        hb_to_ghi = np.where(ghi == 0, 0, np.divide(HB, ghi))
    term3 = 1 + np.sqrt(hb_to_ghi) * (tools.sind(0.5 * surface_tilt) ** 3)
    sky_diffuse = dhi * (AI * rb + term1 * term2 * term3)
    sky_diffuse = np.maximum(sky_diffuse, 0)

    return sky_diffuse


def get_ground_diffuse(surface_tilt, ghi, albedo=0.25, surface_type=None):
    """
    Estima la irradiancia difusa a partir de las reflexiones del suelo dada
    la irradiancia, el albedo y la inclinación de la superficie.

    Función para determinar la porción de irradiancia en una superficie inclinada
    debido a las reflexiones del suelo. Cualquiera de las entradas puede ser un DataFrame o
    un escalar.

    Parámetros
    ----------
    surface_tilt : numérico
        Ángulos de inclinación de la superficie en grados decimales. La inclinación debe ser >=0 y
        <=180. El ángulo de inclinación se define como grados desde la horizontal
        (por ejemplo, superficie hacia arriba = 0, superficie hacia el horizonte = 90).

    ghi : numérico
        Irradiancia horizontal global. [W/m^2]

    albedo : numérico, por defecto 0.25
        Reflectancia del suelo, típicamente 0.1-0.4 para superficies en la Tierra
        (tierra), puede aumentar sobre nieve, hielo, etc. También puede ser conocido como
        el coeficiente de reflexión. Debe ser >=0 y <=1. Será
        reemplazado si se suministra surface_type.

    surface_type: Ninguno o string, por defecto Ninguno
        Si no es Ninguno, reemplaza al albedo. El string puede ser uno de 'urban',
        'grass', 'fresh grass', 'snow', 'fresh snow', 'asphalt', 'concrete',
        'aluminum', 'copper', 'fresh steel', 'dirty steel', 'sea'.

    Retorna
    -------
    grounddiffuse : numérico
        Irradiancia reflejada por el suelo. [W/m^2]


    Referencias
    ----------
    .. [1] Loutzenhiser P.G. et. al. "Validación empírica de modelos para calcular
       la irradiancia solar en superficies inclinadas para la simulación de energía en edificios"
       2007, Solar Energy vol. 81. pp. 254-267.

    El cálculo es el último término de las ecuaciones 3, 4, 7, 8, 10, 11 y 12.

    .. [2] albedos de:
       http://files.pvsyst.com/help/albedo.htm
       y
       http://en.wikipedia.org/wiki/Albedo
       y
       https://doi.org/10.1175/1520-0469(1972)029<0959:AOTSS>2.0.CO;2
    """

    diffuse_irrad = ghi * albedo * (1 - np.cos(np.radians(surface_tilt))) * 0.5
    logging.debug(f"get_ground_diffuse called with surface_type: {surface_type}")



    try:
        diffuse_irrad.name = "diffuse_ground"
    except AttributeError:
        pass

    return diffuse_irrad


def aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    """
    Calcula el ángulo de incidencia del vector solar en una superficie.
    Este es el ángulo entre el vector solar y la normal de la superficie.

    Ingrese todos los ángulos en grados.

    Parámetros
    ----------
    inclinacion_superficie : numérico
        Inclinación del panel desde horizontal.
    azimuth_superficie : numérico
        Azimuth del panel desde el norte.
    zenit_solar : numérico
        Ángulo zenital solar.
    azimuth_solar : numérico
        Ángulo de azimuth solar.

    Retorna
    -------
    aoi : numérico
        Ángulo de incidencia en grados.
    """

    projection = aoi_projection(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth
    )
    aoi_value = np.rad2deg(np.arccos(projection))

    try:
        aoi_value.name = "aoi"
    except AttributeError:
        pass

    return aoi_value


def _get_perez_coefficients(perezmodel):
    """
    Encuentra los coeficientes para el modelo de Perez

    Parámetros
    ----------

    modelo_perez : cadena (opcional, por defecto='allsitescomposite1990')

        una cadena de caracteres que selecciona el conjunto de coeficientes de Perez
        deseado. Si no se proporciona un modelo como entrada, se utilizará el valor predeterminado,
        '1990'.

    Todas las posibles selecciones de modelo son:

        * '1990'
        * 'allsitescomposite1990' (igual que '1990')


    Retorna
    --------
    Coeficientes_F1, Coeficientes_F2 : (arreglo, arreglo)
        Coeficientes F1 y F2 para el modelo de Perez

    Referencias
    ----------
    .. [1] Loutzenhiser P.G. et. al. "Validación empírica de modelos para
       calcular la irradiancia solar en superficies inclinadas para la
       simulación de energía de edificios" 2007, Solar Energy vol. 81. pp. 254-267

    .. [2] Perez, R., Seals, R., Ineichen, P., Stewart, R., Menicucci, D.,
       1987. Una nueva versión simplificada del modelo de irradiancia difusa Perez
       para superficies inclinadas. Solar Energy 39(3), 221-232.

    .. [3] Perez, R., Ineichen, P., Seals, R., Michalsky, J., Stewart, R.,
       1990. Modelando la disponibilidad de luz natural y los componentes de irradiación
       directa y global desde la irradiación directa y global. Solar Energy 44 (5), 271-289.

    .. [4] Perez, R. et. al 1988. "Desarrollo y verificación del
       modelo de radiación difusa Perez". SAND88-7030

    """
    coeffdict = {

        'allsitescomposite1990': [
            [-0.0080,    0.5880,   -0.0620,   -0.0600,    0.0720,   -0.0220],
            [0.1300,    0.6830,   -0.1510,   -0.0190,    0.0660,   -0.0290],
            [0.3300,    0.4870,   -0.2210,    0.0550,   -0.0640,   -0.0260],
            [0.5680,    0.1870,   -0.2950,    0.1090,   -0.1520,   -0.0140],
            [0.8730,   -0.3920,   -0.3620,    0.2260,   -0.4620,    0.0010],
            [1.1320,   -1.2370,   -0.4120,    0.2880,   -0.8230,    0.0560],
            [1.0600,   -1.6000,   -0.3590,    0.2640,   -1.1270,    0.1310],
            [0.6780,   -0.3270,   -0.2500,    0.1560,   -1.3770,    0.2510]],
            }

    array = np.array(coeffdict[perezmodel])

    f1coeffs = array[:, 0:3]
    f2coeffs = array[:, 3:7]

    return f1coeffs, f2coeffs


def poa_components(aoi, dni, poa_sky_diffuse, poa_ground_diffuse):
    """
    Determina los componentes de la irradiancia en el plano del módulo.

    Combina el DNI con la irradiancia difusa del cielo y la irradiancia
    reflejada por el suelo para calcular los componentes totales, directos y
    difusos de la irradiancia en el plano del módulo.

    Parámetros
    ----------
    aoi : numérico
        Ángulo de incidencia de los rayos solares con respecto a la superficie
        del módulo, calculado con la función `aoi`.

    dni : numérico
        Irradiancia directa normal (W/m^2), medida en un archivo TMY o calculada
        con un modelo de cielo despejado.

    poa_sky_diffuse : numérico
        Irradiancia difusa del cielo (W/m^2) en el plano de los módulos,
        calculada por una función de traducción de irradiancia difusa.

    poa_ground_diffuse : numérico
        Irradiancia reflejada por el suelo (W/m^2) en el plano de los módulos,
        calculada por un modelo de albedo (por ejemplo, :func:`grounddiffuse`).

    Retorna
    -------
    irrads : OrderedDict o DataFrame
        Contiene las siguientes claves:

        * ``poa_global`` : Irradiancia total en el plano del módulo (W/m^2)
        * ``poa_direct`` : Irradiancia directa total en el plano del módulo (W/m^2)
        * ``poa_diffuse`` : Irradiancia difusa total en el plano del módulo (W/m^2)
        * ``poa_sky_diffuse`` : Irradiancia difusa en el plano desde el cielo (W/m^2)
        * ``poa_ground_diffuse`` : Irradiancia difusa en el plano desde el suelo (W/m^2)

    Notas
    ------
    La irradiancia beam negativa debido a un ángulo de incidencia (AOI) mayor a
    :math:`90^{\circ}` o un AOI menor a :math:`0^{\circ}` se establece en cero.
    """

    poa_direct = np.maximum(dni * np.cos(np.radians(aoi)), 0)
    poa_diffuse = poa_sky_diffuse + poa_ground_diffuse
    poa_global = poa_direct + poa_diffuse

    irrads = OrderedDict()
    irrads["poa_global"] = poa_global
    irrads["poa_direct"] = poa_direct
    irrads["poa_diffuse"] = poa_diffuse
    irrads["poa_sky_diffuse"] = poa_sky_diffuse
    irrads["poa_ground_diffuse"] = poa_ground_diffuse

    if isinstance(poa_direct, pd.Series):
        irrads = pd.DataFrame(irrads)

    return irrads

def clearness_index(ghi, solar_zenith, extra_radiation, min_cos_zenith=0.065,
                    max_clearness_index=2.0):
    """
    Calcula el índice de claridad.

    El índice de claridad es la relación entre la irradiancia global y la irradiancia extraterrestre
    en un plano horizontal [1]_.

    Parámetros
    ----------
    ghi : numérico
        Irradiancia global horizontal en W/m^2.

    solar_zenith : numérico
        Ángulo zenital solar verdadero (sin corrección de refracción) en grados decimales.

    extra_radiation : numérico
        Irradiancia incidente en la parte superior de la atmósfera.

    min_cos_zenith : numérico, predeterminado 0.065
        Valor mínimo de cos(zenith) permitido al calcular el índice de claridad global `kt`.
        Equivalente a un zenit de 86.273 grados.

    max_clearness_index : numérico, predeterminado 2.0
        Valor máximo del índice de claridad. El valor predeterminado, 2.0, permite eventos de
        sobre-irradiancia típicamente observados en datos subhorarios.
        El código Fortran de NREL's SRRL utilizaba 0.82 para datos horarios.

    Returns
    -------
    kt : numérico
        Índice de claridad

    Referencias
    ----------
    .. [1] Maxwell, E. L., "Un Modelo Cuasi-Físico para Convertir Insolación Global Horaria
           en Insolación Directa Normal", Informe Técnico No. SERI/TR-215-3087, Golden, CO:
           Solar Energy Research Institute, 1987.
    """
    cos_zenith = tools.cosd(solar_zenith)
    i0h = extra_radiation * np.maximum(cos_zenith, min_cos_zenith)
    # considerar agregar
    # with np.errstate(invalid='ignore', divide='ignore'):
    # al cálculo de kt, pero quizás es bueno permitir estas advertencias
    # a los usuarios que anulan min_cos_zenith
    kt = ghi / i0h
    kt = np.maximum(kt, 0)
    kt = np.minimum(kt, max_clearness_index)
    return kt


def disc(ghi, solar_zenith, datetime_or_doy, pressure=101325,
         min_cos_zenith=0.065, max_zenith=87, max_airmass=12):
    """
    Estima la Irradiancia Directa Normal a partir de la Irradiancia Global Horizontal
    utilizando el modelo DISC.

    El algoritmo DISC convierte la irradiancia global horizontal en irradiancia directa
    normal mediante relaciones empíricas entre los índices de claridad global y directa.

    La implementación de xm_solarlib limita el índice de claridad a 1.

    El informe original que describe el modelo DISC [1]_ utiliza la
    masa de aire relativa en lugar de la masa de aire absoluta (corregida por presión).
    Sin embargo, la implementación de NREL del modelo DISC [2]_
    utiliza la masa de aire absoluta. xm_solarlib Matlab también utiliza la masa de aire absoluta.
    La implementación de xm_solarlib en Python utiliza la masa de aire absoluta de forma predeterminada,
    pero se puede utilizar la masa de aire relativa si se suministra `pressure=None`.

    Parámetros
    ----------
    ghi : numérico
        Irradiancia global horizontal en W/m^2.

    solar_zenith : numérico
        Ángulos zenitales solares verdaderos (sin corrección de refracción) en grados decimales.

    datetime_or_doy : int, float, array, pd.DatetimeIndex
        Día del año o matriz de días del año, por ejemplo, pd.DatetimeIndex.dayofyear, o pd.DatetimeIndex.

    pressure : None o numérico, predeterminado 101325
        Presión en el sitio en Pascal. Si es None, se utiliza la masa de aire relativa
        en lugar de la masa de aire absoluta (corregida por presión).

    min_cos_zenith : numérico, predeterminado 0.065
        Valor mínimo de cos(zenith) permitido al calcular el índice de claridad global `kt`.
        Equivalente a un zenit de 86.273 grados.

    max_zenith : numérico, predeterminado 87
        Valor máximo de zenit permitido en el cálculo de DNI. DNI se establecerá en
        0 para momentos con valores de zenit mayores a `max_zenith`.

    max_airmass : numérico, predeterminado 12
        Valor máximo de la masa de aire permitido en el cálculo de Kn.
        El valor predeterminado (12) proviene del rango en el que Kn se ajustó
        a la masa de aire en el artículo original.

    Returns
    -------
    output : OrderedDict o DataFrame
        Contiene las siguientes claves:

        * ``dni``: La irradiancia directa normal modelada
          en W/m^2 proporcionada por el
          modelo de Código de Simulación de Insolación Directa (DISC).
        * ``kt``: Relación de la irradiancia global a la irradiancia extraterrestre
          en un plano horizontal.
        * ``airmass``: Masa de aire

    Referencias
    ----------
    .. [1] Maxwell, E. L., "Un Modelo Cuasi-Físico para Convertir Insolación Global Horaria
       en Insolación Directa Normal", Informe Técnico No. SERI/TR-215-3087, Golden, CO:
       Solar Energy Research Institute, 1987.

    .. [2] Maxwell, E. "Modelo DISC", Hoja de Excel.
       https://www.nrel.gov/grid/solar-resource/disc.html

    See Also
    --------
    dirint
    """

    # este es el cálculo de I0 de la referencia
    # SSC utiliza una constante solar = 1367.0 (verificado en 2018 08 15)
    I0 = get_extra_radiation(datetime_or_doy, 1370.)

    kt = clearness_index(ghi, solar_zenith, I0, min_cos_zenith=min_cos_zenith,
                         max_clearness_index=1)

    am = atmosphere.get_relative_airmass(solar_zenith, model='kasten1966')
    if pressure is not None:
        am = atmosphere.get_absolute_airmass(am, pressure)

    kn, am = _disc_kn(kt, am, max_airmass=max_airmass)
    dni = kn * I0

    bad_values = (solar_zenith > max_zenith) | (ghi < 0) | (dni < 0)
    dni = np.where(bad_values, 0, dni)

    output = OrderedDict()
    output['dni'] = dni
    output['kt'] = kt
    output['airmass'] = am

    if isinstance(datetime_or_doy, pd.DatetimeIndex):
        output = pd.DataFrame(output, index=datetime_or_doy)

    return output


def _disc_kn(clearness_index, airmass, max_airmass=12):
    """
    Calcula Kn para `disc`

    Parámetros
    ----------
    indice_de_claridad : numérico
    masa_de_aire : numérico
    max_masa_de_aire : float
        La masa de aire > max_masa_de_aire se establece en max_masa_de_aire antes de utilizarse
        en el cálculo de Kn.

    Returns
    -------
    Kn : numérico
    am : numérico
        Masa de aire utilizada en el cálculo de Kn. am <= max_masa_de_aire.
    """
    # Nombres cortos para las ecuaciones
    kt = clearness_index
    am = airmass

    am = np.minimum(am, max_airmass)  # GH 450

    is_cloudy = (kt <= 0.6)
    # Usar el método de Horner para calcular los polinomios de manera eficiente
    a = np.where(
        is_cloudy,
        0.512 + kt*(-1.56 + kt*(2.286 - 2.222*kt)),
        -5.743 + kt*(21.77 + kt*(-27.49 + 11.56*kt)))
    b = np.where(
        is_cloudy,
        0.37 + 0.962*kt,
        41.4 + kt*(-118.5 + kt*(66.05 + 31.9*kt)))
    c = np.where(
        is_cloudy,
        -0.28 + kt*(0.932 - 2.048*kt),
        -47.01 + kt*(184.2 + kt*(-222.0 + 73.81*kt)))

    delta_kn = a + b * np.exp(c*am)

    knc = 0.866 + am*(-0.122 + am*(0.0121 + am*(-0.000653 + 1.4e-05*am)))
    kn = knc - delta_kn
    return kn, am
