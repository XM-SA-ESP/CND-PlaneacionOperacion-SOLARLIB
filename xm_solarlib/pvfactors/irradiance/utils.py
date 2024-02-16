"""Módulo que contiene funciones y herramientas específicas de modelado de irradiancia.
Probablemente sea necesario trabajar más para mejorar la velocidad de cálculo aquí, y
en teoría, una gran parte de ``perez_diffuse_luminance`` debería ser manejada por xm_solarlib
"""
import numpy as np
import pandas as pd
from xm_solarlib import atmosphere, irradiance
from xm_solarlib.tools import cosd, sind
import math
from xm_solarlib.pvfactors.config import \
    SIGMA, N_SIGMA, GAUSSIAN_DIAMETER_CIRCUMSOLAR, RADIUS_CIRCUMSOLAR


def perez_diffuse_luminance(timestamps, surface_tilt, surface_azimuth,
                            solar_zenith, solar_azimuth, dni, dhi):
    """Función utilizada para calcular los términos de luminancia y factor de vista a partir del
     Modelo de transposición de luz difusa de Pérez, implementado en el
     Biblioteca ``xm_solarlib-python``.
     Esta función fue hecha a medida para permitir el cálculo del circunsolar.
     componente en la superficie posterior también. De lo contrario, el archivo ``xm_solarlib``
     la implementación lo ignoraría.

     Parámetros
     ----------
     marcas de tiempo: tipo matriz
         marcas de tiempo de simulación
     Surface_tilt: similar a una matriz
         Ángulos de inclinación de la superficie en grados decimales.
         Surface_tilt debe ser >=0 y <=180.
         El ángulo de inclinación se define como grados desde la horizontal.
         (por ejemplo, superficie orientada hacia arriba = 0, superficie orientada hacia el horizonte = 90)
     Surface_azimuth: similar a una matriz
         El acimut del panel girado,
         determinado proyectando el vector normal a la superficie del panel para
         la superficie de la tierra [grados].
     solar_zenith: similar a una matriz
         ángulos cenital solares
     solar_azimuth: similar a una matriz
         ángulos de azimut solar
     dni: tipo matriz
         valores para irradiancia normal directa
     dhi: en forma de matriz
         valores de irradiancia horizontal difusa

     Devoluciones
     -------
     df_inputs: `pandas.DataFrame`
         Marco de datos con las siguientes columnas:
         ['solar_cenit', 'solar_azimut', 'surface_tilt', 'surface_azimuth',
         'dhi', 'dni', 'vf_horizon', 'vf_circumsolar', 'vf_isotropic',
         'luminance_horizon', 'luminance_circuqmsolar', 'luminance_isotropic',
         'poa_isotropic', 'poa_circumsolar', 'poa_horizon', 'poa_total_diffuse']

     """
    # Create a dataframe to help filtering on all arrays
    df_inputs = pd.DataFrame(
        {'surface_tilt': surface_tilt, 'surface_azimuth': surface_azimuth,
         'solar_zenith': solar_zenith, 'solar_azimuth': solar_azimuth,
         'dni': dni, 'dhi': dhi},
        index=pd.DatetimeIndex(timestamps))

    dni_et = irradiance.get_extra_radiation(df_inputs.index.dayofyear)
    am = atmosphere.get_relative_airmass(df_inputs.solar_zenith)

    # Need to treat the case when the sun is hitting the back surface of pvrow
    aoi_proj = irradiance.aoi_projection(
        df_inputs.surface_tilt, df_inputs.surface_azimuth,
        df_inputs.solar_zenith, df_inputs.solar_azimuth)
    sun_hitting_back_surface = ((aoi_proj < 0) &
                                (df_inputs.solar_zenith <= 90))
    df_inputs_back_surface = df_inputs.loc[sun_hitting_back_surface].copy()
    # Reverse the surface normal to switch to back-surface circumsolar calc
    df_inputs_back_surface.loc[:, 'surface_azimuth'] = (
        df_inputs_back_surface.loc[:, 'surface_azimuth'] - 180.)
    df_inputs_back_surface.loc[:, 'surface_azimuth'] = np.mod(
        df_inputs_back_surface.loc[:, 'surface_azimuth'], 360.
    )
    df_inputs_back_surface.loc[:, 'surface_tilt'] = (
        180. - df_inputs_back_surface.surface_tilt)

    if df_inputs_back_surface.shape[0] > 0:
        # Use recursion to calculate circumsolar luminance for back surface
        df_inputs_back_surface = perez_diffuse_luminance(
            *breakup_df_inputs(df_inputs_back_surface))

    # Calculate Perez diffuse components
    components = irradiance.perez(df_inputs.surface_tilt,
                                  df_inputs.surface_azimuth,
                                  df_inputs.dhi, df_inputs.dni,
                                  dni_et,
                                  df_inputs.solar_zenith,
                                  df_inputs.solar_azimuth,
                                  am,
                                  return_components=True)

    # Calculate Perez view factors:
    a = irradiance.aoi_projection(
        df_inputs.surface_tilt,
        df_inputs.surface_azimuth, df_inputs.solar_zenith,
        df_inputs.solar_azimuth)
    a = np.maximum(a, 0)
    b = cosd(df_inputs.solar_zenith)
    b = np.maximum(b, cosd(85))

    vf_perez = pd.DataFrame({
        'vf_horizon': sind(df_inputs.surface_tilt),
        'vf_circumsolar': a / b,
        'vf_isotropic': (1. + cosd(df_inputs.surface_tilt)) / 2.
    },
        index=df_inputs.index
    )

    # Calculate diffuse luminance
    luminance = pd.DataFrame(
        np.array([
            components['horizon'] / vf_perez['vf_horizon'],
            components['circumsolar'] / vf_perez['vf_circumsolar'],
            components['isotropic'] / vf_perez['vf_isotropic']
        ]).T,
        index=df_inputs.index,
        columns=['luminance_horizon', 'luminance_circumsolar',
                 'luminance_isotropic']
    )
    luminance.loc[components['sky_diffuse'] == 0, :] = 0.

    # Format components column names
    components = components.rename(columns={'isotropic': 'poa_isotropic',
                                            'circumsolar': 'poa_circumsolar',
                                            'horizon': 'poa_horizon'})

    df_inputs = pd.concat([df_inputs, components, vf_perez, luminance],
                          axis=1, join='outer')
    df_inputs = df_inputs.rename(columns={'sky_diffuse': 'poa_total_diffuse'})

    # adjust the circumsolar luminance when it hits the back surface
    if df_inputs_back_surface.shape[0] > 0:
        df_inputs.loc[sun_hitting_back_surface, 'luminance_circumsolar'] = (
            df_inputs_back_surface.loc[:, 'luminance_circumsolar'])

    return df_inputs


def breakup_df_inputs(df_inputs):
    """Función auxiliar: a veces es más fácil proporcionar un marco de datos que una lista
     de matrices: esta función hace el trabajo de dividir el marco de datos en un
     lista de matrices 1-tenue esperadas

     Parámetros
     ----------
     df_inputs: ``pandas.DataFrame``
         Marco de datos indexado por marca de tiempo con las siguientes columnas:
         'surface_azimuth', 'surface_tilt', 'solar_zenith', 'solar_azimuth',
         'dni', 'dhi'

     Devoluciones
     -------
     todo 1-dim ``numpy.ndarray``
         ``marcas de tiempo``, ``tilt_angles``, ``surface_azimuth``,
         ``solar_zenith``, ``solar_azimut``, ``dni``, ``dhi``

     """
    timestamps = pd.to_datetime(df_inputs.index)
    surface_azimuth = df_inputs.surface_azimuth.values
    surface_tilt = df_inputs.surface_tilt.values
    solar_zenith = df_inputs.solar_zenith.values
    solar_azimuth = df_inputs.solar_azimuth.values
    dni = df_inputs.dni.values
    dhi = df_inputs.dhi.values

    return (timestamps, surface_tilt, surface_azimuth,
            solar_zenith, solar_azimuth, dni, dhi)


def calculate_circumsolar_shading(percentage_distance_covered,
                                  model='uniform_disk'):
    """Seleccione el modelo para calcular el sombreado circunsolar en función de la energía fotovoltaica actual
     condición de matriz.

     Parámetros
     ----------
     porcentaje_distancia_cubierta: flotante
         esto representa la cantidad de
         El diámetro circunsolar está cubierto por la fila vecina [en %]
     modelo: str, opcional
         nombre del modelo de sombreado circunsolar a utilizar:
         'uniform_disk' y 'gaussian' son los dos modelos disponibles actualmente
         (Valor predeterminado = 'disco_uniforme')

     Devoluciones
     -------
     flotar
         porcentaje de sombreado del área circunsolar

     """
    if model == 'uniform_disk':
        perc_shading = uniform_circumsolar_disk_shading(
            percentage_distance_covered)

    elif model == 'gaussian':
        perc_shading = gaussian_shading(percentage_distance_covered)

    else:
        raise ValueError(
            'calculate_circumsolar_shading: model does not exist: ' +
            '%s' % model)

    return perc_shading


def integral_default_gaussian(y, x):
    """Calcule el valor de la integral de xay de la función erf

     Parámetros
     ----------
     y: flotar
         limite superior
     x: flotar
         límite inferior

     Devoluciones
     -------
     flotar
         Valor calculado de la integral

     """
    return 0.5 * (math.erf(x) - math.erf(y))


def gaussian_shading(percentage_distance_covered):
    """Calcule el sombreado circunsolar suponiendo que el perfil de irradiancia en
     una sección 2D del disco circunsolar es gaussiana

     Parámetros
     ----------
     porcentaje_distancia_cubierta: flotante
         [en %], proporción de la
         Área del disco circunsolar cubierta por el pvrow vecino.

     Devoluciones
     -------
     flotar
         porcentaje de sombreado en términos de irradiancia (usando perfil gaussiano)

     """
    if percentage_distance_covered < 0.:
        perc_shading = 0.
    elif percentage_distance_covered > 100.:
        perc_shading = 100.
    else:
        y = - N_SIGMA * SIGMA
        x = (y +
             percentage_distance_covered / 100. *
             GAUSSIAN_DIAMETER_CIRCUMSOLAR)
        area = integral_default_gaussian(y, x)
        total_gaussian_area = integral_default_gaussian(- N_SIGMA * SIGMA,
                                                        N_SIGMA * SIGMA)
        perc_shading = area / total_gaussian_area * 100.

    return perc_shading


def gaussian(x, mu=0., sigma=1.):
    """Función de densidad gaussiana

     Parámetros
     ----------
     x: flotar
         argumento de función
     mu: flotante, opcional
         media de la gaussiana (valor predeterminado = 0.)
     sigma: flotante, opcional
         desviación estándar del gaussiano (valor predeterminado = 1).

     Devoluciones
     -------
     flotar
         valor de la función guassiana en el punto x

     """
    return (1. / (sigma * np.sqrt(2. * np.pi)) *
            np.exp(- 0.5 * np.power((x - mu) / sigma, 2)))


def uniform_circumsolar_disk_shading(percentage_distance_covered):
    """Calcule el porcentaje de sombreado de la irradiancia circunsolar. Esto
     El modelo considera circunsolar como un disco y calcula la
     porcentaje sombreado según el porcentaje de "distancia recorrida",
     que es la cantidad del diámetro del disco que está cubierto por el
     objeto vecino.

     Parámetros
     ----------
     porcentaje_distancia_cubierta: flotante
         distancia recorrida del diámetro del disco circunsolar
         [% - valores de 0 a 100]

     Devoluciones
     -------
     flotar
         valor de la función guassiana en el punto x

     """

    # Define a circumsolar disk
    r_circumsolar = RADIUS_CIRCUMSOLAR
    d_circumsolar = 2 * r_circumsolar
    area_circumsolar = np.pi * r_circumsolar**2.

    # Calculate circumsolar on case by case
    distance_covered = percentage_distance_covered / 100. * d_circumsolar

    if distance_covered <= 0:
        # No shading of circumsolar
        percent_shading = 0.
    elif (distance_covered > 0.) & (distance_covered <= r_circumsolar):
        # Theta is the full circle sector angle (not half) used to
        # calculate circumsolar shading
        theta = 2 * np.arccos((r_circumsolar - distance_covered) /
                              r_circumsolar)  # rad
        area_circle_sector = 0.5 * r_circumsolar ** 2 * theta
        area_triangle_sector = 0.5 * r_circumsolar ** 2 * np.sin(theta)
        area_shaded = area_circle_sector - area_triangle_sector
        percent_shading = area_shaded / area_circumsolar * 100.

    elif (distance_covered > r_circumsolar) & (distance_covered <
                                               d_circumsolar):
        distance_uncovered = d_circumsolar - distance_covered

        # Theta is the full circle sector angle (not half) used to
        # calculate circumsolar shading
        theta = 2 * np.arccos((r_circumsolar - distance_uncovered) /
                              r_circumsolar)  # rad
        area_circle_sector = 0.5 * r_circumsolar ** 2 * theta
        area_triangle_sector = 0.5 * r_circumsolar ** 2 * np.sin(theta)
        area_not_shaded = area_circle_sector - area_triangle_sector
        percent_shading = (1. - area_not_shaded / area_circumsolar) * 100.

    else:
        # Total shading of circumsolar: distance_covered >= d_circumsolar
        percent_shading = 100.

    return percent_shading


def calculate_horizon_band_shading(shading_angle, horizon_band_angle):
    """Calcula el porcentaje de sombreado de la banda del horizonte; que es justo el
     proporción de lo que está enmascarado por la fila vecina ya que suponemos
     valores de luminancia uniformes para la banda

     Parámetros
     ----------
     sombreado_ángulo: np.ndarray
         El ángulo de elevación que se utilizará para sombrear.
     horizonte_band_angle: flotante
         el ángulo de elevación total de la banda del horizonte

     Devoluciones
     -------
     flotar
         porcentaje de sombreado de la banda del horizonte

     """
    percent_shading = np.where(shading_angle >= horizon_band_angle, 100., 0.)
    percent_shading = np.where(
        (shading_angle >= 0) & (shading_angle < horizon_band_angle),
        100. * shading_angle / horizon_band_angle, percent_shading)
    return percent_shading