'''
Calcular la posición solar utilizando una variedad de métodos/paquetes.
'''

# Contributors:
# Rob Andrews (@Calama-Consulting), Calama Consulting, 2014
# Will Holmgren (@wholmgren), University of Arizona, 2014
# Tony Lorenzo (@alorenzo175), University of Arizona, 2015
# Cliff hansen (@cwhanse), Sandia National Laboratories, 2018

import os
import datetime as dt
try:
    from importlib import reload
except ImportError:
    try:
        from imp import reload
    except ImportError:
        pass

import numpy as np
import pandas as pd
import scipy.optimize as so
import warnings
import datetime

from xm_solarlib import atmosphere


NS_PER_HR = 1.e9 * 3600.  # nanoseconds per hour


def get_solarposition(time, latitude, longitude,
                      altitude=None, pressure=None,
                      method='nrel_numpy',
                      temperature=12, **kwargs):
    """
    Un envoltorio conveniente para los calculadores de posición solar.

    Parámetros
    ----------
    tiempo : pandas.DatetimeIndex
        Debe estar localizado o se asumirá UTC.

    latitud : float
        Latitud en grados decimales. Positivo al norte del ecuador, negativo
        al sur.

    longitud : float
        Longitud en grados decimales. Positivo al este del meridiano de Greenwich,
        negativo al oeste.

    altitud : None o float, por defecto None
        Si es None, se calcula a partir de la presión. Se asume que es 0 m
        si la presión también es None.

    presión : None o float, por defecto None
        Si es None, se calcula a partir de la altitud. Se asume que es 101325 Pa
        si la altitud también es None.

    método : string, por defecto 'nrel_numpy'
        'nrel_numpy' utiliza una implementación del algoritmo NREL SPA
        descrito en [1] (por defecto, recomendado): :py:func:`spa_python`

        'nrel_numba' utiliza una implementación del algoritmo NREL SPA
        descrito en [1], pero también compila el código primero:
        :py:func:`spa_python`

        'pyephem' utiliza el paquete PyEphem: :py:func:`pyephem`

<<<<<<< HEAD
        'ephemeris' utiliza el código efemérides de pvlib: :py:func:`ephemeris`
=======
        'ephemeris' uses the xm_solarlib ephemeris code: :py:func:`ephemeris`
>>>>>>> main

        'nrel_c' utiliza el código C NREL SPA [3]: :py:func:`spa_c`

    temperatura : float, por defecto 12
        Grados Celsius.

    kwargs
        Otros argumentos clave se pasan a la función de posición solar
        especificada por el argumento ``método``.

    Referencias
    ----------
    .. [1] I. Reda y A. Andreas, Algoritmo de posición solar para aplicaciones de radiación solar.
       Energía Solar, vol. 76, núm. 5, pp. 577-589, 2004.

    .. [2] I. Reda y A. Andreas, Corrección al algoritmo de posición solar para
       aplicaciones de radiación solar. Energía Solar, vol. 81, núm. 6, p. 838,
       2007.

    .. [3] Código SPA de NREL: http://rredc.nrel.gov/solar/codesandalgorithms/spa/
    """

    if altitude is None and pressure is None:
        altitude = 0.
        pressure = 101325.
    elif altitude is None:
        altitude = atmosphere.pres2alt(pressure)
    elif pressure is None:
        pressure = atmosphere.alt2pres(altitude)

    method = method.lower()
    if isinstance(time, dt.datetime):
        time = pd.DatetimeIndex([time, ])

    if method == 'nrel_c':
        ephem_df = spa_c(time, latitude, longitude, pressure, temperature,
                         **kwargs)
    elif method == 'nrel_numba':
        ephem_df = spa_python(time, latitude, longitude, altitude,
                              pressure, temperature,
                              how='numba', **kwargs)
    elif method == 'nrel_numpy':
        ephem_df = spa_python(time, latitude, longitude, altitude,
                              pressure, temperature,
                              how='numpy', **kwargs)
    elif method == 'pyephem':
        ephem_df = pyephem(time, latitude, longitude,
                           altitude=altitude,
                           pressure=pressure,
                           temperature=temperature, **kwargs)
    elif method == 'ephemeris':
        ephem_df = ephemeris(time, latitude, longitude, pressure, temperature,
                             **kwargs)
    else:
        raise ValueError('Invalid solar position method')

    return ephem_df

def spa_c(time, latitude, longitude, pressure=101325, altitude=0,
          temperature=12, delta_t=67.0,
          raw_spa_output=False):
    """
    Calcula la posición solar utilizando la implementación en C del código NREL SPA.

    Los archivos fuente de este código se encuentran en './spa_c_files/', junto con
    un archivo README que describe cómo se envuelve el código C en Python.
    Debido a restricciones de licencia, el código C debe descargarse por separado
    y utilizarse de acuerdo con su licencia.

    Esta función es más lenta y no más precisa que :py:func:`spa_python`.

    Parámetros
    ----------
    tiempo : pandas.DatetimeIndex
        Debe estar localizado o se asumirá UTC.
    latitud : float
        Latitud en grados decimales. Positivo al norte del ecuador, negativo
        al sur.
    longitud : float
        Longitud en grados decimales. Positivo al este del meridiano de Greenwich,
        negativo al oeste.
    presión : float, por defecto 101325
        Presión en Pascales.
    altitud : float, por defecto 0
        Altura sobre el nivel del mar. [m]
    temperatura : float, por defecto 12
        Temperatura en C
    delta_t : float, por defecto 67.0
        Diferencia entre el tiempo terrestre y UT1.
        USNO tiene valores anteriores y predicciones.
    salida_spa_cruda : bool, por defecto False
        Si es verdadero, devuelve la salida cruda de SPA.

    Returns
    -------
    DataFrame
        El DataFrame tendrá las siguientes columnas:
        elevación,
        azimut,
        cenit,
        elevación aparente,
        cenit aparente.

    Referencias
    ----------
    .. [1] Referencia NREL SPA:
       http://rredc.nrel.gov/solar/codesandalgorithms/spa/
       Archivos C NREL SPA: https://midcdmz.nrel.gov/spa/

    Nota: El campo "timezone" en los archivos C SPA se reemplaza por
    "time_zone" para evitar un conflicto de nombres con la función "__timezone" que se
    redefine en Python>=3.5. Este problema es
    `error en Python 24643 <https://bugs.python.org/issue24643>`_.

    .. [2] Delta T de USNO:
       http://www.usno.navy.mil/USNO/earth-orientation/eo-products/long-term

    Ver también
    --------
    pyephem, spa_python, ephemeris
    """

    # Added by Rob Andrews (@Calama-Consulting), Calama Consulting, 2014
    # Edited by Will Holmgren (@wholmgren), University of Arizona, 2014
    # Edited by Tony Lorenzo (@alorenzo175), University of Arizona, 2015

    try:
        from xm_solarlib.spa_c_files.spa_py import spa_calc
    except ImportError:
        raise ImportError('Could not import built-in SPA calculator. ' +
                          'You may need to recompile the SPA code.')

    # Si está localizado, conviértalo a UTC. De lo contrario, asuma UTC.
    try:
        time_utc = time.tz_convert('UTC')
    except TypeError:
        time_utc = time

    spa_out = []

    for date in time_utc:
        spa_out.append(spa_calc(year=date.year,
                                month=date.month,
                                day=date.day,
                                hour=date.hour,
                                minute=date.minute,
                                second=date.second,
                                time_zone=0,  # date uses utc time
                                latitude=latitude,
                                longitude=longitude,
                                elevation=altitude,
                                pressure=pressure / 100,
                                temperature=temperature,
                                delta_t=delta_t
                                ))

    spa_df = pd.DataFrame(spa_out, index=time)

    if raw_spa_output:
        # Cambie el nombre de "time_zone" en la salida cruda de spa_c_files.spa_py.spa_calc()
        # a "timezone" para que coincida con la API de pvlib.solarposition.spa_c()
        return spa_df.rename(columns={'time_zone': 'timezone'})
    else:
        dfout = pd.DataFrame({'azimuth': spa_df['azimuth'],
                              'apparent_zenith': spa_df['zenith'],
                              'apparent_elevation': spa_df['e'],
                              'elevation': spa_df['e0'],
                              'zenith': 90 - spa_df['e0']})

        return dfout

def spa_python(time, latitude, longitude,
               altitude=0, pressure=101325, temperature=12, delta_t=67.0,
               atmos_refract=None, how='numpy', numthreads=4):
    """
    Calcula la posición solar utilizando una implementación en Python del
    algoritmo NREL SPA.

    Los detalles del algoritmo NREL SPA se describen en [1]_.

    Si se ha instalado numba, las funciones se pueden compilar a
    código de máquina y la función se puede ejecutar en múltiples hilos.
    Sin numba, la función se evalúa a través de numpy con
    una ligera pérdida de rendimiento.

    Parámetros
    ----------
    tiempo : pandas.DatetimeIndex
        Debe estar localizado o se asumirá UTC.
    latitud : float
        Latitud en grados decimales. Positivo al norte del ecuador, negativo
        al sur.
    longitud : float
        Longitud en grados decimales. Positivo al este del meridiano de Greenwich,
        negativo al oeste.
    altitud : float, por defecto 0
        Distancia sobre el nivel del mar.
    presión : int o float, opcional, por defecto 101325
        Presión atmosférica promedio anual en Pascales.
    temperatura : int o float, opcional, por defecto 12
        Temperatura del aire promedio anual en grados Celsius.
    delta_t : float, opcional, por defecto 67.0
        Diferencia entre el tiempo terrestre y UT1.
        Si delta_t es None, utiliza spa.calculate_deltat
        utilizando time.year y time.month de pandas.DatetimeIndex.
        Para la mayoría de las simulaciones, el delta_t predeterminado es suficiente.
        *Nota: delta_t = None romperá el código que use nrel_numba,
        esto se solucionará en una versión futura.*
        El USNO tiene datos históricos y predicciones para delta_t [3]_.
    atmós_refract : None o float, opcional, por defecto None
        La refracción atmosférica aproximada (en grados)
        al amanecer y al atardecer.
    cómo : str, opcional, por defecto 'numpy'
        Las opciones son 'numpy' o 'numba'. Si numba >= 0.17.0
        está instalado, how='numba' compilará las funciones spa
        a código de máquina y las ejecutará en múltiples hilos.
    numthreads : int, opcional, por defecto 4
        Número de hilos a utilizar si how == 'numba'.

    Returns
    -------
    DataFrame
        El DataFrame tendrá las siguientes columnas:
        zenith aparente (grados),
        cenit (grados),
        elevación aparente (grados),
        elevación (grados),
        azimut (grados),
        ecuación del tiempo (minutos).


    Referencias
    ----------
    .. [1] I. Reda y A. Andreas, Algoritmo de posición solar para
       aplicaciones de radiación solar. Energía Solar, vol. 76, núm. 5, pp. 577-589, 2004.

    .. [2] I. Reda y A. Andreas, Corrección al algoritmo de posición solar para
       aplicaciones de radiación solar. Energía Solar, vol. 81, núm. 6, p. 838,
       2007.

    .. [3] USNO delta T:
       http://www.usno.navy.mil/USNO/earth-orientation/eo-products/long-term

    Ver también
    --------
    pyephem, spa_c, ephemeris
    """

    # Added by Tony Lorenzo (@alorenzo175), University of Arizona, 2015

    lat = latitude
    lon = longitude
    elev = altitude
    pressure = pressure / 100  # la presión debe estar en milibares para el cálculo
    atmos_refract = atmos_refract or 0.5667

    if not isinstance(time, pd.DatetimeIndex):
        try:
            time = pd.DatetimeIndex(time)
        except (TypeError, ValueError):
            time = pd.DatetimeIndex([time, ])

    unixtime = np.array(time.view(np.int64)/10**9)

    spa = _spa_python_import(how)

    delta_t = delta_t or spa.calculate_deltat(time.year, time.month)

    app_zenith, zenith, app_elevation, elevation, azimuth, eot = \
        spa.solar_position(unixtime, lat, lon, elev, pressure, temperature,
                           delta_t, atmos_refract, numthreads)

    result = pd.DataFrame({'apparent_zenith': app_zenith, 'zenith': zenith,
                           'apparent_elevation': app_elevation,
                           'elevation': elevation, 'azimuth': azimuth,
                           'equation_of_time': eot},
                          index=time)

    return result

def pyephem(time, latitude, longitude, altitude=0, pressure=101325,
            temperature=12, horizon='+0:00'):
    """
    Calcula la posición solar utilizando el paquete PyEphem.

    Parámetros
    ----------
    tiempo : pandas.DatetimeIndex
        Debe estar localizado o se asumirá UTC.
    latitud : float
        Latitud en grados decimales. Positivo al norte del ecuador, negativo
        al sur.
    longitud : float
        Longitud en grados decimales. Positivo al este del meridiano de Greenwich,
        negativo al oeste.
    altitud : float, por defecto 0
        Altura sobre el nivel del mar en metros. [m]
    presión : int o float, opcional, por defecto 101325
        Presión atmosférica en Pascales.
    temperatura : int o float, opcional, por defecto 12
        Temperatura del aire en grados Celsius.
    horizonte : string, opcional, por defecto '+0:00'
        grados de arco:minutos de arco desde el horizonte geométrico para el amanecer y
        el atardecer, por ejemplo, horizonte='+0:00' para usar el cruce del centro del sol con el
        horizonte geométrico para definir el amanecer y el atardecer,
        horizonte=SOL_HORIZONTE cuando el borde superior del sol cruza el
        horizonte geométrico

    Returns
    -------
    pandas.DataFrame
        el índice es el mismo que el argumento de entrada `tiempo`
        El DataFrame tendrá las siguientes columnas:
        elevación aparente, elevación,
        azimut aparente, azimut,
        cenit aparente, cenit.

    Ver también
    --------
    spa_python, spa_c, ephemeris
    """

    # Written by Will Holmgren (@wholmgren), University of Arizona, 2014
    try:
        import ephem
    except ImportError:
        raise ImportError('PyEphem must be installed')

    # si está localizado, conviértalo a UTC. de lo contrario, asuma UTC.
    try:
        time_utc = time.tz_convert('UTC')
    except TypeError:
        time_utc = time

    sun_coords = pd.DataFrame(index=time)

    obs, sun = _ephem_setup(latitude, longitude, altitude,
                            pressure, temperature, horizon)

    # crear y llenar listas de altitud y azimut del sol
    # esto es la altitud y azimut aparente corregidos por presión y temperatura.
    alts = []
    azis = []
    for thetime in time_utc:
        obs.date = ephem.Date(thetime)
        sun.compute(obs)
        alts.append(sun.alt)
        azis.append(sun.az)

    sun_coords['apparent_elevation'] = alts
    sun_coords['apparent_azimuth'] = azis

    # realizarlo nuevamente para p=0 para obtener alt/az sin atmósfera
    obs.pressure = 0
    alts = []
    azis = []
    for thetime in time_utc:
        obs.date = ephem.Date(thetime)
        sun.compute(obs)
        alts.append(sun.alt)
        azis.append(sun.az)

    sun_coords['elevation'] = alts
    sun_coords['azimuth'] = azis

    # convertir a grados. agregar cenit
    sun_coords = np.rad2deg(sun_coords)
    sun_coords['apparent_zenith'] = 90 - sun_coords['apparent_elevation']
    sun_coords['zenith'] = 90 - sun_coords['elevation']

    return sun_coords


def ephemeris(time, latitude, longitude, pressure=101325, temperature=12):
    """
    Calculadora de posición solar nativa de Python.
    La precisión de este código no está garantizada.
    Considere usar el código integrado spa_c o la biblioteca PyEphem.

    Parámetros
    ----------
    tiempo : pandas.DatetimeIndex
        Debe estar localizado o se asumirá UTC.
    latitud : float
        Latitud en grados decimales. Positivo al norte del ecuador, negativo
        al sur.
    longitud : float
        Longitud en grados decimales. Positivo al este del meridiano de Greenwich,
        negativo al oeste.
    presión : float o Series, por defecto 101325
        Presión ambiental (Pascales).
    temperatura : float o Series, por defecto 12
        Temperatura ambiente (Celsius).

    Returns
    -------

    DataFrame con las siguientes columnas:

        * elevación aparente: elevación del sol aparente teniendo en cuenta la
          refracción atmosférica.
        * elevación: elevación real (sin tener en cuenta la refracción) del sol
          en grados decimales, 0 = en el horizonte.
          El complemento del ángulo cenital.
        * azimut: Azimut del sol en grados al este del norte.
          Esto es el complemento del ángulo cenital aparente.
        * cenit aparente: cenit del sol aparente teniendo en cuenta la
          refracción atmosférica.
        * cenit: ángulo cenital solar
        * hora_solar: Hora solar en horas decimales (el mediodía solar es 12.00).

    Referencias
    -----------

    .. [1] Clase de Grover Hughes y materiales relacionados en Astronomía
       de Ingeniería en Sandia National Laboratories, 1985.

    Ver también
    --------
    pyephem, spa_c, spa_python

    """

    # Added by Rob Andrews (@Calama-Consulting), Calama Consulting, 2014
    # Edited by Will Holmgren (@wholmgren), University of Arizona, 2014

    # La mayoría de los comentarios en esta función provienen de PVLIB_MATLAB o de
    # del intento de pvlib-python de entender y solucionar problemas con el
    # algoritmo. Los comentarios no están basados en el material de referencia.
    # Esto ayuda un poco:

    # http://www.cv.nrao.edu/~rfisher/Ephemerides/times.html

    # la inversión de la longitud se debe al hecho de que este código fue
    # originalmente escrito para la convención de que las longitudes positivas eran para
    # ubicaciones al oeste del meridiano de Greenwich. Sin embargo, la convención correcta (a partir de 2009)
    # es usar longitudes negativas para ubicaciones al oeste del meridiano de Greenwich.
    # Por lo tanto, el usuario debe ingresar valores de longitud bajo la
    # convención correcta (por ejemplo, Albuquerque está en -106 longitud), pero debe
    # invertirse para su uso en el código.

    longitude = -1 * longitude

    abber = 20 / 3600.
    latr = np.radians(latitude)

    # el algoritmo SPA necesita que el tiempo se exprese en términos de
    # horas decimales UTC del día del año.

    # si está localizado, conviértalo a UTC. de lo contrario, asuma UTC.
    try:
        time_utc = time.tz_convert('UTC')
    except TypeError:
        time_utc = time

    # eliminar el día del año y calcular la hora decimal
    dayofyear = time_utc.dayofyear
    dec_hours = (time_utc.hour + time_utc.minute/60. + time_utc.second/3600. +
                time_utc.microsecond/3600.e6)

    # np.array necesario para pandas > 0.20
    univ_date = np.array(dayofyear)
    univ_hr = np.array(dec_hours)

    y_r = np.array(time_utc.year) - 1900
    yr_begin = 365 * y_r + np.floor((y_r - 1) / 4.) - 0.5

    e_zero = yr_begin + univ_date
    T = e_zero / 36525.

    # Calcular el Tiempo Sidéreo Medio de Greenwich (GMST)
    GMST0 = 6 / 24. + 38 / 1440. + (
        45.836 + 8640184.542 * T + 0.0929 * T ** 2) / 86400.
    GMST0 = 360 * (GMST0 - np.floor(GMST0))
    gmst_i = np.mod(GMST0 + 360 * (1.0027379093 * univ_hr / 24.), 360)

    # Tiempo Sidéreo Local Aparente
    loc_ast = np.mod((360 + gmst_i - longitude), 360)

    epoch_date = e_zero + univ_hr / 24.
    T1 = epoch_date / 36525.

    obliquity_r = np.radians(
        23.452294 - 0.0130125 * T1 - 1.64e-06 * T1 ** 2 + 5.03e-07 * T1 ** 3)
    ml_perigee = 281.22083 + 4.70684e-05 * epoch_date + 0.000453 * T1 ** 2 + (
        3e-06 * T1 ** 3)
    mean_anom = np.mod((358.47583 + 0.985600267 * epoch_date - 0.00015 *
                       T1 ** 2 - 3e-06 * T1 ** 3), 360)
    eccen = 0.01675104 - 4.18e-05 * T1 - 1.26e-07 * T1 ** 2
    eccen_anom = mean_anom
    E = 0

    while np.max(abs(eccen_anom - E)) > 0.0001:
        E = eccen_anom
        eccen_anom = mean_anom + np.degrees(eccen)*np.sin(np.radians(E))

    true_nom = (
        2 * np.mod(np.degrees(np.arctan2(((1 + eccen) / (1 - eccen)) ** 0.5 *
                   np.tan(np.radians(eccen_anom) / 2.), 1)), 360))
    ec_lon = np.mod(ml_perigee + true_nom, 360) - abber
    ec_lon_r = np.radians(ec_lon)
    dec_r = np.arcsin(np.sin(obliquity_r)*np.sin(ec_lon_r))

    rt_ascen = np.degrees(np.arctan2(np.cos(obliquity_r)*np.sin(ec_lon_r),
                                    np.cos(ec_lon_r)))

    hr_angle = loc_ast - rt_ascen
    hr_angle_r = np.radians(hr_angle)
    hr_angle = hr_angle - (360 * (abs(hr_angle) > 180))

    sun_az = np.degrees(np.arctan2(-np.sin(hr_angle_r),
                                  np.cos(latr)*np.tan(dec_r) -
                                  np.sin(latr)*np.cos(hr_angle_r)))
    sun_az[sun_az < 0] += 360

    sun_el = np.degrees(np.arcsin(
        np.cos(latr) * np.cos(dec_r) * np.cos(hr_angle_r) +
        np.sin(latr) * np.sin(dec_r)))

    solar_time = (180 + hr_angle) / 15.

    # Calcular corrección de refracción
    elevation = sun_el
    tan_el = pd.Series(np.tan(np.radians(elevation)), index=time_utc)
    refract = pd.Series(0, index=time_utc)

    refract[(elevation > 5) & (elevation <= 85)] = (
        58.1/tan_el - 0.07/(tan_el**3) + 8.6e-05/(tan_el**5))

    refract[(elevation > -0.575) & (elevation <= 5)] = (
        elevation *
        (-518.2 + elevation*(103.4 + elevation*(-12.79 + elevation*0.711))) +
        1735)

    refract[(elevation > -1) & (elevation <= -0.575)] = -20.774 / tan_el

    refract *= (283/(273. + temperature)) * (pressure/101325.) / 3600.

    apparent_sun_el = sun_el + refract

    # crear DataFrame de salida
    df_out = pd.DataFrame(index=time_utc)
    df_out['apparent_elevation'] = apparent_sun_el
    df_out['elevation'] = sun_el
    df_out['azimuth'] = sun_az
    df_out['apparent_zenith'] = 90 - apparent_sun_el
    df_out['zenith'] = 90 - sun_el
    df_out['solar_time'] = solar_time
    df_out.index = time

    return df_out

def _spa_python_import(how):
    "Compila spa.py de manera apropiada"

    from xm_solarlib import spa

    # verifica si el módulo spa fue compilado con numba
    using_numba = spa.USE_NUMBA

    if how == 'numpy' and using_numba:
        # el módulo spa fue compilado a código numba, por lo que necesitamos
        # recargar el módulo sin compilar
        # la variable de entorno XM_SOLARLIB_USE_NUMBA se utiliza para indicarle al módulo
        # que no compile con numba
        warnings.warn('Reloading spa to use numpy')
        os.environ['XM_SOLARLIB_USE_NUMBA'] = '0'
        spa = reload(spa)
        del os.environ['XM_SOLARLIB_USE_NUMBA']
    elif how == 'numba' and not using_numba:
        # El módulo spa no fue compilado a código numba, por lo que establecemos
        # XM_SOLARLIB_USE_NUMBA para que compile a numba al recargar.
        warnings.warn('Reloading spa to use numba')
        os.environ['XM_SOLARLIB_USE_NUMBA'] = '1'
        spa = reload(spa)
        del os.environ['XM_SOLARLIB_USE_NUMBA']
    elif how != 'numba' and how != 'numpy':
        raise ValueError("how must be either 'numba' or 'numpy'")

    return spa

def sun_rise_set_transit_spa(times, latitude, longitude, how='numpy',
                             delta_t=67.0, numthreads=4):
    """
    Calcula los horarios de salida, puesta y tránsito del sol utilizando el algoritmo NREL SPA.

    Los detalles del algoritmo NREL SPA se describen en [1]_.

    Si está instalado numba, las funciones se pueden compilar a código de máquina y la función se puede ejecutar en múltiples hilos.
    Sin numba, la función se evalúa mediante numpy con una ligera penalización de rendimiento.

    Parámetros
    ----------
    horarios : pandas.DatetimeIndex
        Debe estar localizado en la zona horaria de "latitud" y "longitud".
    latitud : float
        Latitud en grados, positiva al norte del ecuador, negativa al sur.
    longitud : float
        Longitud en grados, positiva al este del meridiano de Greenwich, negativa al oeste.
    metodo : str, opcional, valor predeterminado 'numpy'
        Las opciones son 'numpy' o 'numba'. Si numba >= 0.17.0 está instalado, how='numba' compilará las funciones spa a código de máquina y las ejecutará en múltiples hilos.
    delta_t : float, opcional, valor predeterminado 67.0
        Diferencia entre el tiempo terrestre y el UT1.
        Si delta_t es None, utiliza spa.calculate_deltat utilizando el año y el mes de times de pandas.DatetimeIndex.
        Para la mayoría de las simulaciones, el valor predeterminado de delta_t es suficiente.
        *Nota: delta_t = None romperá el código que use nrel_numba, esto se corregirá en una versión futura.*
    numthreads : int, opcional, valor predeterminado 4
        Número de hilos a utilizar si metodo == 'numba'.

    Devoluciones
    -------
    pandas.DataFrame
        El índice es el mismo que el argumento de entrada `horarios`.
        Las columnas son 'salida del sol', 'puesta del sol' y 'tránsito del sol'.

    Referencias
    ----------
    .. [1] Reda, I., Andreas, A., 2003. Solar position algorithm for solar
       radiation applications. Technical report: NREL/TP-560- 34302. Golden,
       USA, http://www.nrel.gov.
    """
    # Added by Tony Lorenzo (@alorenzo175), University of Arizona, 2015

    lat = latitude
    lon = longitude

    # los horarios deben estar localizados
    if times.tz:
        tzinfo = times.tz
    else:
        raise ValueError('times must be localized')

    # se debe convertir a medianoche UTC en el día de interés
    utcday = pd.DatetimeIndex(times.date).tz_localize('UTC')
    unixtime = np.array(utcday.view(np.int64)/10**9)

    spa = _spa_python_import(how)

    delta_t = delta_t or spa.calculate_deltat(times.year, times.month)

    transit, sunrise, sunset = spa.transit_sunrise_sunset(
        unixtime, lat, lon, delta_t, numthreads)

    # los arrays están en segundos desde el formato de época, deben convertirse a marcas de tiempo
    transit = pd.to_datetime(transit*1e9, unit='ns', utc=True).tz_convert(
        tzinfo).tolist()
    sunrise = pd.to_datetime(sunrise*1e9, unit='ns', utc=True).tz_convert(
        tzinfo).tolist()
    sunset = pd.to_datetime(sunset*1e9, unit='ns', utc=True).tz_convert(
        tzinfo).tolist()

    return pd.DataFrame(index=times, data={'sunrise': sunrise,
                                           'sunset': sunset,
                                           'transit': transit})


def _ephem_convert_to_seconds_and_microseconds(date):
    # utilidad de PyEphem 3.6.7.1 no lanzada
    "Convierte una fecha de PyEphem en segundos"
    microseconds = int(round(24 * 60 * 60 * 1000000 * date))
    seconds, microseconds = divmod(microseconds, 1000000)
    seconds -= 2209032000  # diferencia entre la época de 1900 y la época de 1970
    return seconds, microseconds


def _ephem_to_timezone(date, tzinfo):
    # utilidad de PyEphem 3.6.7.1 no lanzada
    "Convierte una fecha de PyEphem en un objeto datetime de Python con información de la zona horaria"
    seconds, microseconds = _ephem_convert_to_seconds_and_microseconds(date)
    date = dt.datetime.fromtimestamp(seconds, tzinfo)
    date = date.replace(microsecond=microseconds)
    return date


def _ephem_setup(latitude, longitude, altitude, pressure, temperature,
                 horizon):
    import ephem
    # inicializa un observador de PyEphem
    obs = ephem.Observer()
    obs.lat = str(latitude)
    obs.lon = str(longitude)
    obs.elevation = altitude
    obs.pressure = pressure / 100.  # convertir a mBar
    obs.temp = temperature
    obs.horizon = horizon

    # el sol de PyEphem
    sun = ephem.Sun()
    return obs, sun


def sun_rise_set_transit_ephem(times, latitude, longitude,
                               next_or_previous='next',
                               altitude=0,
                               pressure=101325,
                               temperature=12, horizon='0:00'):
    """
    Calcula las próximas horas de salida y puesta del sol usando el paquete PyEphem.

    Parámetros
    ----------
    times : pandas.DatetimeIndex
        Debe estar localizado
    latitud : float
        Latitud en grados, positiva al norte del ecuador, negativa al sur
    longitud : float
        Longitud en grados, positiva al este del meridiano de Greenwich, negativa al oeste
    siguiente_o_anterior : str
        'siguiente' o 'anterior' para la salida y puesta del sol con respecto a la hora proporcionada
    altitud : float, predeterminado 0
        distancia sobre el nivel del mar en metros.
    presion : int o float, opcional, predeterminado 101325
        presión del aire en Pascales.
    temperatura : int o float, opcional, predeterminado 12
        temperatura del aire en grados Celsius.
    horizonte : string, formato +/-X:YY
        grados de arco:minutos de arco desde el horizonte geométrico para la salida y puesta del sol, por ejemplo, horizonte='+0:00' para usar el cruce del centro del sol con el horizonte geométrico para definir la salida y puesta del sol,
        horizonte=SOL_HORIZONTE para cuando el borde superior del sol cruza el horizonte geométrico

    Devoluciones
    -------
    pandas.DataFrame
        el índice es el mismo que el argumento de entrada 'times'
        las columnas son 'sunrise' (salida del sol), 'sunset' (puesta del sol) y 'transit' (tránsito solar)

    Ver también
    --------
    pyephem
    """

    try:
        import ephem
    except ImportError:
        raise ImportError('PyEphem must be installed')

    # Los tiempos deben estar localizados
    if times.tz:
        tzinfo = times.tz
    else:
        raise ValueError('times must be localized')

    obs, sun = _ephem_setup(latitude, longitude, altitude,
                            pressure, temperature, horizon)
    # Crea listas de hora de salida y puesta del sol localizadas en time.tz
    if next_or_previous.lower() == 'next':
        rising = obs.next_rising
        setting = obs.next_setting
        transit = obs.next_transit
    elif next_or_previous.lower() == 'previous':
        rising = obs.previous_rising
        setting = obs.previous_setting
        transit = obs.previous_transit
    else:
        raise ValueError("next_or_previous must be either 'next' or" +
                         " 'previous'")

    sunrise = []
    sunset = []
    trans = []
    for thetime in times:
        thetime = thetime.to_pydatetime()
        # Versiones anteriores de pyephem ignoran la zona horaria al convertir a su
        # formato interno de datetime, así que convertimos a UTC aquí para admitir
        # todas las versiones.  GH #1449
        obs.date = ephem.Date(thetime.astimezone(datetime.timezone.utc))
        sunrise.append(_ephem_to_timezone(rising(sun), tzinfo))
        sunset.append(_ephem_to_timezone(setting(sun), tzinfo))
        trans.append(_ephem_to_timezone(transit(sun), tzinfo))

    return pd.DataFrame(index=times, data={'sunrise': sunrise,
                                           'sunset': sunset,
                                           'transit': trans})


def _calculate_simple_day_angle(dayofyear, offset=1):
    """
    Calculates the day angle for the Earth's orbit around the Sun.

    Parameters
    ----------
    dayofyear : numeric
    offset : int, default 1
        For the Spencer method, offset=1; for the ASCE method, offset=0

    Returns
    -------
    day_angle : numeric
    """
    return (2. * np.pi / 365.) * (dayofyear - offset)
