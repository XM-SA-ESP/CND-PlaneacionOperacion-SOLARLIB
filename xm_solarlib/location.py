"""
Este módulo contiene la clase Location (Ubicación).
"""

# Will Holmgren, University of Arizona, 2014-2016.

import os
import datetime

import pandas as pd
import pytz
import h5py

from xm_solarlib import solarposition, clearsky, atmosphere, irradiance


class Location:
    """
    Los objetos Location (Ubicación) son contenedores convenientes para datos de latitud, longitud,
    zona horaria y altitud asociados con una ubicación geográfica particular. También puedes asignar un nombre a un objeto de ubicación.

    Los objetos de ubicación tienen dos atributos de zona horaria:

        * ``tz`` es una cadena de zona horaria IANA.
        * ``pytz`` es un objeto de zona horaria pytz.

    Los objetos de ubicación admiten el método de impresión.

    Parámetros
    ----------
    latitude : float.
        Positivo es al norte del ecuador.
        Utilice la notación de grados decimales.

    longitude : float.
        Positivo es al este del meridiano de Greenwich.
        Utilice la notación de grados decimales.

    tz : str, int, float o pytz.timezone, predeterminado 'UTC'.
        Consulta
        http://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        para ver una lista de zonas horarias válidas.
        Los objetos pytz.timezone se convertirán a cadenas.
        Los enteros y flotantes deben estar en horas desde UTC.

    altitude : float, predeterminado 0.
        Altitud sobre el nivel del mar en metros.

    name : None o string, predeterminado None.
        Establece el atributo de nombre del objeto Location (Ubicación).

    """

    def __init__(self, latitude, longitude, tz='UTC', altitude=0, name=None):

        self.latitude = latitude
        self.longitude = longitude

        if isinstance(tz, str):
            self.tz = tz
            self.pytz = pytz.timezone(tz)
        elif isinstance(tz, datetime.timezone):
            self.tz = 'UTC'
            self.pytz = pytz.UTC
        elif isinstance(tz, datetime.tzinfo):
            self.tz = tz.zone
            self.pytz = tz
        elif isinstance(tz, (int, float)):
            self.tz = tz
            self.pytz = pytz.FixedOffset(tz*60)
        else:
            raise TypeError('Invalid tz specification')

        self.altitude = altitude

        self.name = name

    def __repr__(self):
        attrs = ['name', 'latitude', 'longitude', 'altitude', 'tz']
        return ('Location: \n  ' + '\n  '.join(
            f'{attr}: {getattr(self, attr)}' for attr in attrs))

    @classmethod
    def from_tmy(cls, tmy_metadata, tmy_data=None, **kwargs):
        """
        Crea un objeto basado en un diccionario de metadatos
        desde lectores de datos tmy2 o tmy3.

        Parámetros
        ----------
        tmy_metadata : dict
            Devuelto por tmy.readtmy2 o tmy.readtmy3
        tmy_data : None o DataFrame, predeterminado None
            Opcionalmente adjunta los datos TMY a este objeto.

        Retorna
        -------
        Ubicación (Location)
        """
        # No está completo, pero esperamos que entiendas la idea.
        # Puede que sea necesario código para manejar la diferencia entre tmy2 y tmy3.

        # determinar si estamos tratando con datos TMY2 o TMY3
        tmy2 = tmy_metadata.get('City', False)

        latitude = tmy_metadata['latitude']
        longitude = tmy_metadata['longitude']

        if tmy2:
            name = tmy_metadata['City']
        else:
            name = tmy_metadata['Name']

        tz = tmy_metadata['TZ']
        altitude = tmy_metadata['altitude']

        new_object = cls(latitude, longitude, tz=tz, altitude=altitude,
                         name=name, **kwargs)

        # No estamos seguros de si esto debe asignarse independientemente de la entrada.
        if tmy_data is not None:
            new_object.tmy_data = tmy_data
            new_object.weather = tmy_data

        return new_object

    @classmethod
    def from_epw(cls, metadata, data=None, **kwargs):
        """
        Crea un objeto de Ubicación basado en un diccionario de metadatos
        desde lectores de datos epw.

        Parámetros
        ----------
        metadata : dict
            Devuelto por epw.read_epw
        data : None o DataFrame, predeterminado None
            Opcionalmente adjunta los datos epw a este objeto.

        Retorna
        -------
        Objeto de Ubicación (Location) (o la subclase de Location (Ubicación)
        desde la que llamaste a este método).
        """

        latitude = metadata['latitude']
        longitude = metadata['longitude']

        name = metadata['city']

        tz = metadata['TZ']
        altitude = metadata['altitude']

        new_object = cls(latitude, longitude, tz=tz, altitude=altitude,
                         name=name, **kwargs)

        if data is not None:
            new_object.weather = data

        return new_object

    def get_solarposition(self, times, pressure=None, temperature=12,
                          **kwargs):
        """
        Utiliza la función :py:func:`pvlib.solarposition.get_solarposition`
        para calcular la posición solar (zenit, azimuth, etc.) en esta ubicación.

        Parámetros
        ----------
        times : pandas.DatetimeIndex
            Debe estar localizado o se asumirá UTC.
        pressure : None, float o similar a una matriz, predeterminado None
            Si es None, la presión se calculará utilizando
            :py:func:`pvlib.atmosphere.alt2pres` y ``self.altitude``.
        temperature : None, float o similar a una matriz, predeterminado 12

        kwargs
            pasados a :py:func:`pvlib.solarposition.get_solarposition`

        Retorna
        -------
        solar_position : DataFrame
            Las columnas dependen del argumento ``method`` kwarg, pero siempre incluyen
            ``zenith`` y ``azimuth``. Los ángulos están en grados.
        """
        if pressure is None:
            pressure = atmosphere.alt2pres(self.altitude)

        return solarposition.get_solarposition(times, latitude=self.latitude,
                                               longitude=self.longitude,
                                               altitude=self.altitude,
                                               pressure=pressure,
                                               temperature=temperature,
                                               **kwargs)

    def get_clearsky(self, times, model='ineichen', solar_position=None,
                     dni_extra=None, **kwargs):
        """
        Calcula las estimaciones de cielo despejado de GHI, DNI y/o DHI
        en esta ubicación.

        Parámetros
        ----------
        times: DatetimeIndex
        model: str, predeterminado 'ineichen'
            El modelo de cielo despejado a utilizar. Debe ser uno de
            'ineichen', 'haurwitz', 'simplified_solis'.
        solar_position : None o DataFrame, predeterminado None
            DataFrame con columnas 'apparent_zenith', 'zenith',
            'apparent_elevation'.
        dni_extra: None o numérico, predeterminado None
            Si es None, se calculará a partir de los tiempos.

        kwargs
            Parámetros adicionales pasados a las funciones relevantes. En muchos casos, se asumen valores climatológicos. ¡Consulte el código fuente para obtener más detalles!

        Retorna
        -------
        clearsky : DataFrame
            Los nombres de las columnas son: ``ghi, dni, dhi``.
        """
        if dni_extra is None:
            dni_extra = irradiance.get_extra_radiation(times)

        try:
            pressure = kwargs.pop('pressure')
        except KeyError:
            pressure = atmosphere.alt2pres(self.altitude)

        if solar_position is None:
            solar_position = self.get_solarposition(times, pressure=pressure)

        apparent_zenith = solar_position['apparent_zenith']
        apparent_elevation = solar_position['apparent_elevation']

        if model == 'ineichen':
            try:
                linke_turbidity = kwargs.pop('linke_turbidity')
            except KeyError:
                interp_turbidity = kwargs.pop('interp_turbidity', True)
                linke_turbidity = clearsky.lookup_linke_turbidity(
                    times, self.latitude, self.longitude,
                    interp_turbidity=interp_turbidity)

            try:
                airmass_absolute = kwargs.pop('airmass_absolute')
            except KeyError:
                airmass_absolute = self.get_airmass(
                    times, solar_position=solar_position)['airmass_absolute']

            cs = clearsky.ineichen(apparent_zenith, airmass_absolute,
                                   linke_turbidity, altitude=self.altitude,
                                   dni_extra=dni_extra, **kwargs)
        elif model == 'haurwitz':
            cs = clearsky.haurwitz(apparent_zenith)
        elif model == 'simplified_solis':
            cs = clearsky.simplified_solis(
                apparent_elevation, pressure=pressure, dni_extra=dni_extra,
                **kwargs)
        else:
            raise ValueError('{} is not a valid clear sky model. Must be '
                             'one of ineichen, simplified_solis, haurwitz'
                             .format(model))

        return cs

    def get_airmass(self, times=None, solar_position=None,
                    model='kastenyoung1989'):
        """
        Calcula la masa de aire relativa y absoluta.

        Elije automáticamente el zenit o el zenit aparente
        dependiendo del modelo seleccionado.

        Parámetros
        ----------
        times : None o DatetimeIndex, predeterminado None
            Solo se utiliza si no se proporciona solar_position.
        solar_position : None o DataFrame, predeterminado None
            DataFrame con columnas 'apparent_zenith', 'zenith'.
        model : str, predeterminado 'kastenyoung1989'
            Modelo de masa de aire relativa. Consulta
            :py:func:`pvlib.atmosphere.get_relative_airmass`
            para obtener una lista de modelos disponibles.

        Retorna
        -------
        airmass : DataFrame
            Las columnas son 'airmass_relative', 'airmass_absolute'

        See also
        --------
        xm_solarlib.atmosphere.get_relative_airmass
        """

        if solar_position is None:
            solar_position = self.get_solarposition(times)

        if model in atmosphere.APPARENT_ZENITH_MODELS:
            zenith = solar_position['apparent_zenith']
        elif model in atmosphere.TRUE_ZENITH_MODELS:
            zenith = solar_position['zenith']
        else:
            raise ValueError(f'{model} is not a valid airmass model')

        airmass_relative = atmosphere.get_relative_airmass(zenith, model)

        pressure = atmosphere.alt2pres(self.altitude)
        airmass_absolute = atmosphere.get_absolute_airmass(airmass_relative,
                                                           pressure)

        airmass = pd.DataFrame(index=solar_position.index)
        airmass['airmass_relative'] = airmass_relative
        airmass['airmass_absolute'] = airmass_absolute

        return airmass

    def get_sun_rise_set_transit(self, times, method='pyephem', **kwargs):
        """
        Calcula los tiempos de amanecer, atardecer y tránsito solar.

        Parámetros
        ----------
        times : DatetimeIndex
            Debe estar localizado en la Ubicación.
        method : str, predeterminado 'pyephem'
            'pyephem', 'spa' o 'geometric'

        kwargs :
            Pasados a las funciones relevantes. Consulta
            solarposition.sun_rise_set_transit_<method> para obtener detalles.

        Retorna
        -------
        resultado : DataFrame
            Los nombres de las columnas son: ``amanecer, atardecer, tránsito``.
        """

        if method == 'pyephem':
            result = solarposition.sun_rise_set_transit_ephem(
                times, self.latitude, self.longitude, **kwargs)
        elif method == 'spa':
            result = solarposition.sun_rise_set_transit_spa(
                times, self.latitude, self.longitude, **kwargs)
        elif method == 'geometric':
            sr, ss, tr = solarposition.sun_rise_set_transit_geometric(
                times, self.latitude, self.longitude, **kwargs)
            result = pd.DataFrame(index=times,
                                  data={'sunrise': sr,
                                        'sunset': ss,
                                        'transit': tr})
        else:
            raise ValueError('{} is not a valid method. Must be '
                             'one of pyephem, spa, geometric'
                             .format(method))
        return result

