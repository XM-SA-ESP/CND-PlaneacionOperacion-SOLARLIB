"""
El módulo "pvsystem" contiene funciones para modelar la producción y el rendimiento de módulos PV e inversores.
"""
import os
import io
from urllib.request import urlopen
from collections import OrderedDict
import numpy as np
import pandas as pd
import functools
import itertools
from dataclasses import dataclass
from typing import Optional, Union
from abc import ABC, abstractmethod
from scipy import constants
from xm_solarlib import ( iam, atmosphere, irradiance, inverter, spectrum, singlediode as _singlediode, temperature)
from xm_solarlib.tools import _build_kwargs, _build_args
from xm_solarlib._deprecation import warn_deprecated, deprecated
import xm_solarlib.tools as tools
from xm_solarlib.constantes import LENGTH_MISMATCH

_DC_MODEL_PARAMS = {
    'sapm': {
        'A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'B2', 'B3',
        'B4', 'B5', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
        'C7', 'Isco', 'Impo', 'Voco', 'Vmpo', 'Aisc', 'Aimp', 'bvoco',
        'Mbvoc', 'bvmpo', 'Mbvmp', 'N', 'Cells_in_Series',
        'IXO', 'IXXO', 'FD'},
    'desoto': {
        'alpha_sc', 'a_ref', 'i_l_ref', 'i_o_ref',
        'r_sh_ref', 'r_s'},
    'cec': {
        'alpha_sc', 'a_ref', 'i_l_ref', 'i_o_ref',
        'r_sh_ref', 'r_s', 'adjust'},
    'pvsyst': {
        'gamma_ref', 'mu_gamma', 'i_l_ref', 'i_o_ref',
        'r_sh_ref', 'r_sh_0', 'r_s', 'alpha_sc', 'egref',
        'cells_in_series'},
    'singlediode': {
        'alpha_sc', 'a_ref', 'i_l_ref', 'i_o_ref',
        'r_sh_ref', 'r_s'},
    'pvwatts': {'pdc0', 'gamma_pdc'}
}

def _unwrap_single_value(func):
    """Decorador para funciones que devuelven iterables.

    Si la longitud del iterable devuelto por `func` es 1, entonces
    se devuelve el único elemento del iterable. Si la longitud es
    mayor que 1, se devuelve el iterable completo.

    Agrega 'unwrap' como un argumento de palabra clave que se puede establecer en Falso
    para forzar que el valor de retorno sea una tupla, independientemente de su longitud.
    """
    @functools.wraps(func)
    def f(*args, **kwargs):
        unwrap = kwargs.pop('unwrap', True)
        x = func(*args, **kwargs)
        if unwrap and len(x) == 1:
            return x[0]
        return x
    return f


# No estoy seguro de si esto debería estar en el módulo pvsystem.
# Quizás algo más parecido a core.py? Puede que eventualmente crezca
# para importar muchas más funcionalidades de otros módulos.
class PVSystem:
    """
    La clase PVSystem define un conjunto estándar de atributos y funciones de modelado
    de sistemas fotovoltaicos (PV). Esta clase describe la colección e interacción de los
    componentes del sistema PV en lugar de un sistema instalado en el suelo. 
    Normalmente se utiliza en combinación con los objetos :py:class:`~pvlib.location.Location`
    y :py:class:`~pvlib.modelchain.ModelChain`.


    La clase admite topologías básicas de sistema que constan de:

        * `N` módulos totales dispuestos en serie (`modules_per_string=N`, `strings_per_inverter=1`).
        * `M` módulos totales dispuestos en paralelo (`modules_per_string=1`, `strings_per_inverter=M`).
        * `NxM` módulos totales dispuestos en `M` cadenas de `N` módulos cada una
          (`modules_per_string=N`, `strings_per_inverter=M`).

    La clase es complementaria a las funciones de nivel de módulo.

    Los atributos deberían ser cosas que generalmente no cambian acerca del sistema,
    como el tipo de módulo y el inversor. Los métodos de instancia aceptan argumentos
    para cosas que sí cambian, como la irradiancia y la temperatura.

    Parámetros
    ----------
    arrays : Array o iterable de Array, opcional
        Un Array o lista de Arrays que forman parte del sistema. Si no se especifica, se crea
        un solo Array a partir de los otros parámetros (por ejemplo, `surface_tilt`, `surface_azimuth`).
        Si se especifica como una lista, la lista debe contener al menos un Array;
        si la longitud de los arrays es 0, se genera un ValueError. Si se especifica `arrays`,
        se ignoran los siguientes parámetros de PVSystem:

        - `surface_tilt`
        - `surface_azimuth`
        - `albedo`
        - `surface_type`
        - `module`
        - `module_type`
        - `module_parameters`
        - `temperature_model_parameters`
        - `modules_per_string`
        - `strings_per_inverter`

    surface_tilt: float o similar a una matriz, por defecto 0
        Ángulos de inclinación de la superficie en grados decimales.
        El ángulo de inclinación se define como grados desde la horizontal
        (por ejemplo, superficie hacia arriba = 0, superficie hacia el horizonte = 90).

    surface_azimuth: float o similar a una matriz, por defecto 180
        Ángulo de azimut de la superficie del módulo.
        Norte=0, Este=90, Sur=180, Oeste=270.

    albedo : None o float, por defecto None
        Albedo de la superficie del suelo. Si es ``None``, entonces se utiliza
        ``surface_type`` para buscar un valor en ``irradiance.SURFACE_ALBEDOS``.
        Si ``surface_type`` también es None, se utiliza un albedo de la superficie del suelo de 0.25.

    surface_type : None o cadena de caracteres, por defecto None
        El tipo de superficie del suelo. Consulta ``irradiance.SURFACE_ALBEDOS`` para
        los valores válidos.

    module : None o cadena de caracteres, por defecto None
        El nombre del modelo de los módulos.
        Puede ser utilizado para buscar el diccionario de parámetros del módulo
        a través de algún otro método.

    module_type : None o cadena de caracteres, por defecto 'vidrio_polímero'
         Describe la construcción del módulo. Las cadenas válidas son 'vidrio_polímero'
         y 'vidrio_vidrio'. Utilizado para los cálculos de temperatura de la célula y del módulo.

    module_parameters : None, dict o Series, por defecto None
        Parámetros del módulo definidos por SAPM, CEC u otros.

    temperature_model_parameters : None, dict o Series, por defecto None.
        Parámetros del modelo de temperatura requeridos por uno de los modelos en
        pvlib.temperature (excluyendo poa_global, temp_air y wind_speed).


    modules_per_string: int o float, por defecto 1
        Consulta la discusión sobre la topología del sistema arriba.

    strings_per_inverter: int o float, por defecto 1
        Consulta la discusión sobre la topología del sistema arriba.

    inverter : None o cadena de caracteres, por defecto None
        El nombre del modelo de los inversores.
        Puede ser utilizado para buscar el diccionario de parámetros del inversor
        a través de algún otro método.

    inverter_parameters : None, dict o Series, por defecto None
        Parámetros del inversor definidos por SAPM, CEC u otros.

    racking_model : None o cadena de caracteres, por defecto 'rack_abierto'
        Cadenas válidas son 'rack_abierto', 'montaje_cerrado' e 'aislado_atrás'.
        Utilizado para identificar un conjunto de parámetros para el modelo de
        temperatura de la célula SAPM.

    losses_parameters : None, dict o Series, por defecto None
        Parámetros de pérdida definidos por PVWatts u otros.

    name : None o cadena de caracteres, por defecto None

    **kwargs
        Argumentos clave arbitrarios.
        Incluidos para compatibilidad, pero no utilizados.

    Raises
    ------
    ValueError
        Si `arrays` no es None y tiene longitud 0.

    See also
    --------
    xm_solarlib.location.Location
    """

    def __init__(self,
                 arrays=None,
                 albedo=None, surface_type=None,
                 module=None, module_type=None,
                 module_parameters=None,
                 temperature_model_parameters=None,
                 inverter=None, inverter_parameters=None,
                 racking_model=None, losses_parameters=None, name=None):

        surface_tilt=0
        surface_azimuth=180
        modules_per_string=1
        strings_per_inverter=1

        if arrays is None:
            if losses_parameters is None:
                array_losses_parameters = {}
            else:
                array_losses_parameters = _build_kwargs(['dc_ohmic_percent'],
                                                        losses_parameters)
            self.arrays = (Array(
                FixedMount(surface_tilt, surface_azimuth, racking_model),
                albedo,
                surface_type,
                module,
                module_type,
                module_parameters,
                temperature_model_parameters,
                modules_per_string,
                strings_per_inverter,
                array_losses_parameters,
            ),)
        elif isinstance(arrays, Array):
            self.arrays = (arrays,)
        elif len(arrays) == 0:
            raise ValueError("PVSystem must have at least one Array. "
                             "If you want to create a PVSystem instance "
                             "with a single Array pass `arrays=None` and pass "
                             "values directly to PVSystem attributes, e.g., "
                             "`surface_tilt=30`")
        else:
            self.arrays = tuple(arrays)

        self.inverter = inverter
        if inverter_parameters is None:
            self.inverter_parameters = {}
        else:
            self.inverter_parameters = inverter_parameters

        if losses_parameters is None:
            self.losses_parameters = {}
        else:
            self.losses_parameters = losses_parameters

        self.name = name

    def __repr__(self):
        pvsystem_repr = f'PVSystem:\n  name: {self.name}\n  '
        for array in self.arrays:
            pvsystem_repr += '\n  '.join(array.__repr__().split('\n'))
            pvsystem_repr += '\n  '
        pvsystem_repr += f'inverter: {self.inverter}'
        return pvsystem_repr


    def _validate_per_array(self, values, system_wide=False):
        """Verifica que `values` sea una tupla del mismo tamaño que
        `self.arrays`.

        Si `values` no es una tupla, se empaqueta en una tupla de longitud 1 antes
        de realizar la comprobación. Si las longitudes no son iguales, se genera un ValueError,
        de lo contrario, se devuelve la tupla `values`.

        Cuando `system_wide` es True y `values` no es una tupla, `values`
        se replica en una tupla del mismo tamaño que `self.arrays` y se devuelve esa tupla.
        """
        if system_wide and not isinstance(values, tuple):
            return (values,) * self.num_arrays
        if not isinstance(values, tuple):
            values = (values,)
        if len(values) != len(self.arrays):
            raise ValueError(LENGTH_MISMATCH)
        return values

    @_unwrap_single_value
    def _infer_cell_type(self):
        """
        Examina module_parameters y asigna la clave Technology para la base de datos CEC
        y la clave Material para la base de datos Sandia a una lista común de cadenas para
        el tipo de célula.

        Returns
        -------
        cell_type: str
        """
        return tuple(array._infer_cell_type() for array in self.arrays)

    @_unwrap_single_value
    def get_aoi(self, solar_zenith, solar_azimuth):
        """Obtiene el ángulo de incidencia en el/los Array(s) en el sistema.

        Parámetros
        ----------
        solar_zenith : float o Series.
            Ángulo cenital solar.
        solar_azimuth : float o Series.
            Ángulo azimut solar.

        Returns
        -------
        aoi : Series o tupla de Series
            El ángulo de incidencia
        """

        return tuple(array.get_aoi(solar_zenith, solar_azimuth)
                     for array in self.arrays)

    @_unwrap_single_value
    def get_irradiance(self, solar_zenith, solar_azimuth, dni, ghi, dhi,
                       dni_extra=None, airmass=None, albedo=None,
                       model='haydavies', **kwargs):
        """
        Utiliza la función :py:func:`irradiance.get_total_irradiance` para
        calcular los componentes de irradiancia del plano del array en las superficies inclinadas
        definidas por `surface_tilt` y `surface_azimuth` de cada array.

        Parámetros
        ----------
        solar_zenith : float o Series
            Ángulo cenital solar.
        solar_azimuth : float o Series
            Ángulo azimut solar.
        dni : float o Series o tupla de float o Series
            Irradiancia Directa Normal. [W/m2]
        ghi : float o Series o tupla de float o Series
            Irradiancia global horizontal. [W/m2]
        dhi : float o Series o tupla de float o Series
            Irradiancia difusa horizontal. [W/m2]
        dni_extra : None, float, Series o tupla de float o Series,\
            por defecto None
            Irradiancia directa normal extraterrestre. [W/m2]
        airmass : None, float o Series, por defecto None
            Masa de aire. [adimensional]
        albedo : None, float o Series, por defecto None
            Albedo de la superficie del suelo. [adimensional]
        model : String, por defecto 'haydavies'
            Modelo de irradiancia.

        kwargs
            Parámetros adicionales pasados a :func:`irradiance.get_total_irradiance`.

        Notas
        -----
        Cada uno de los parámetros `dni`, `ghi` y `dni` puede pasarse como una tupla
        para proporcionar diferentes irradiancias para cada array en el sistema. Si no
        se pasa como una tupla, se utiliza el mismo valor como entrada para cada Array.
        Si se pasa como una tupla, la longitud debe ser la misma que el número de Arrays.

        Returns
        -------
        poa_irradiance : DataFrame o tupla de DataFrame
            Los nombres de columna son: ``'poa_global', 'poa_direct', 'poa_diffuse',
            'poa_sky_diffuse', 'poa_ground_diffuse'``.

        See also
        --------
        xm_solarlib.irradiance.get_total_irradiance
        """
        dni = self._validate_per_array(dni, system_wide=True)
        ghi = self._validate_per_array(ghi, system_wide=True)
        dhi = self._validate_per_array(dhi, system_wide=True)

        albedo = self._validate_per_array(albedo, system_wide=True)

        return tuple(
            array.get_irradiance(solar_zenith, solar_azimuth,
                                 dni, ghi, dhi,
                                 dni_extra=dni_extra, airmass=airmass,
                                 albedo=albedo, model=model, **kwargs)
            for array, dni, ghi, dhi, albedo in zip(
                self.arrays, dni, ghi, dhi, albedo
            )
        )

    @_unwrap_single_value
    def get_iam(self, aoi, iam_model='physical'):
        """
        Determina el modificador del ángulo de incidencia utilizando el método especificado por
        ``modelo_iam``.

        Se espera que los parámetros para el modelo IAM seleccionado estén en
        ``PVSystem.module_parameters``. Los parámetros predeterminados están disponibles para
        los modelos 'físico', 'ashrae' y 'martin_ruiz'.

        Parámetros
        ----------
        aoi : numérico o tupla de numéricos
            El ángulo de incidencia en grados.

        modelo_iam : cadena, predeterminado 'físico'
            El modelo IAM a utilizar. Las cadenas válidas son 'físico', 'ashrae',
            'martin_ruiz', 'sapm' e 'interp'.
        Returns
        -------
        iam : numérico o tupla de numéricos
            El modificador de AOI.

        Raises
        ------
        ValueError
            si `modelo_iam` no es un nombre de modelo válido.
        """
        aoi = self._validate_per_array(aoi)
        return tuple(array.get_iam(aoi, iam_model)
                     for array, aoi in zip(self.arrays, aoi))

    @_unwrap_single_value
    def get_cell_temperature(self, poa_global, temp_air, wind_speed, model,
                             effective_irradiance=None):
        """
        Determina la temperatura de la celda utilizando el método especificado por el parámetro "modelo".

        Parámetros
        ----------
        poa_global : numérico o tupla de numéricos
            Irradiancia total incidente en W/m^2.

        temp_air : numérico o tupla de numéricos
            Temperatura ambiente en grados C.

        wind_speed : numérico o tupla de numéricos
            Velocidad del viento en m/s.

        model : str
            Modelos admitidos incluyen ``'sapm'``, ``'pvsyst'``,
            ``'faiman'``, ``'fuentes'`` y ``'noct_sam'``.

        effective_irradiance : numérico o tupla de numéricos, opcional
            La irradiancia que se convierte en fotocorriente en W/m^2.
            Solo se utiliza para algunos modelos.

        Returns
        -------
        numérico o tupla de numéricos
            Valores en grados C.

        Ver también
        --------
        Array.get_cell_temperature

        Notas
        -----
        Los parámetros `temp_air` y `wind_speed` pueden pasarse como tuplas
        para proporcionar diferentes valores para cada Array en el sistema. Si se pasan como
        una tupla, su longitud debe ser la misma que la cantidad de Arrays. Si no se pasa como
        una tupla, se utiliza el mismo valor para cada Array.
        """
        poa_global = self._validate_per_array(poa_global)
        temp_air = self._validate_per_array(temp_air, system_wide=True)
        wind_speed = self._validate_per_array(wind_speed, system_wide=True)
        # Not used for all models, but Array.get_cell_temperature handles it
        effective_irradiance = self._validate_per_array(effective_irradiance,
                                                        system_wide=True)

        return tuple(
            array.get_cell_temperature(poa_global, temp_air, wind_speed,
                                       model, effective_irradiance)
            for array, poa_global, temp_air, wind_speed, effective_irradiance
            in zip(
                self.arrays, poa_global, temp_air, wind_speed,
                effective_irradiance
            )
        )

    @_unwrap_single_value
    def calcparams_desoto(self, effective_irradiance, temp_cell):
        """
        Utiliza la función :py:func:`calcparams_desoto`, los parámetros de entrada
        y ``self.module_parameters`` para calcular las corrientes y resistencias del módulo.

        Parámetros
        ----------
        effective_irradiance : numérico o tupla de numéricos
            La irradiancia (W/m^2) que se convierte en fotocorriente.

        temp_cell : float o Series o tupla de float o Series
            La temperatura promedio de las celdas dentro de un módulo en C.

        Returns
        -------
        Ver pvsystem.calcparams_desoto para más detalles
        """
        effective_irradiance = self._validate_per_array(effective_irradiance)
        temp_cell = self._validate_per_array(temp_cell)

        build_kwargs = functools.partial(
            _build_kwargs,
            ['a_ref', 'i_l_ref', 'i_o_ref', 'r_sh_ref',
             'r_s', 'alpha_sc', 'egref', 'degdt',
             'irrad_ref', 'temp_ref']
        )

        return tuple(
            calcparams_desoto(
                effective_irradiance, temp_cell,
                **build_kwargs(array.module_parameters)
            )
            for array, effective_irradiance, temp_cell
            in zip(self.arrays, effective_irradiance, temp_cell)
        )

    @_unwrap_single_value
    def calcparams_cec(self, effective_irradiance, temp_cell):
        """
        Utiliza la función :py:func:`calcparams_cec`, los parámetros de entrada
        y ``self.module_parameters`` para calcular las corrientes y resistencias del módulo.

        Parámetros
        ----------
        effective_irradiance : numérico o tupla de numéricos
            La irradiancia (W/m^2) que se convierte en fotocorriente.

        temp_cell : float o Series o tupla de float o Series
            La temperatura promedio de las celdas dentro de un módulo en C.

        Returns
        -------
        Ver pvsystem.calcparams_cec para más detalles
        """
        effective_irradiance = self._validate_per_array(effective_irradiance)
        temp_cell = self._validate_per_array(temp_cell)

        build_kwargs = functools.partial(
            _build_kwargs,
            ['a_ref', 'i_l_ref', 'i_o_ref', 'r_sh_ref',
             'r_s', 'alpha_sc', 'adjust', 'egref', 'degdt',
             'irrad_ref', 'temp_ref']
        )

        return tuple(
            calcparams_cec(
                effective_irradiance, temp_cell,
                **build_kwargs(array.module_parameters)
            )
            for array, effective_irradiance, temp_cell
            in zip(self.arrays, effective_irradiance, temp_cell)
        )


    @_unwrap_single_value
    def sapm(self, effective_irradiance, temp_cell):
        """
        Utiliza la función :py:func:`sapm`, los parámetros de entrada,
        y ``self.module_parameters`` para calcular
        Voc, Isc, Ix, Ixx, Vmp e Imp.

        Parámetros
        ----------
        effective_irradiance : numérico o tupla de numéricos
            La irradiancia (W/m^2) que se convierte en fotocorriente.

        temp_cell : float o Series o tupla de float o Series
            La temperatura promedio de las celdas dentro de un módulo en C.

        Returns
        -------
        Ver pvsystem.sapm para más detalles
        """
        effective_irradiance = self._validate_per_array(effective_irradiance)
        temp_cell = self._validate_per_array(temp_cell)

        return tuple(
            sapm(effective_irradiance, temp_cell, array.module_parameters)
            for array, effective_irradiance, temp_cell
            in zip(self.arrays, effective_irradiance, temp_cell)
        )

    @_unwrap_single_value
    def sapm_spectral_loss(self, airmass_absolute):
        """
        Utiliza la función :py:func:`pvlib.spectrum.spectral_factor_sapm`,
        los parámetros de entrada y ``self.module_parameters`` para calcular F1.


        Parámetros
        ----------
        airmass_absolute : numérico
            Masa de aire absoluta.

        Returns
        -------
        F1 : numérico o tupla de numéricos
            El coeficiente de pérdida espectral SAPM.
        """
        return tuple(
            spectrum.spectral_factor_sapm(airmass_absolute,
                                          array.module_parameters)
            for array in self.arrays
        )

    @_unwrap_single_value
    def sapm_effective_irradiance(self, poa_direct, poa_diffuse,
                                  airmass_absolute, aoi,
                                  reference_irradiance=1000):
        """
        Utiliza la función :py:func:`sapm_effective_irradiance`, los parámetros de entrada
        y ``self.module_parameters`` para calcular
        la irradiancia efectiva.

        Parámetros
        ----------
        poa_direct : numérico o tupla de numéricos
            La irradiancia directa incidente en el módulo. [W/m^2]

        poa_diffuse : numérico o tupla de numéricos
            La irradiancia difusa incidente en el módulo. [W/m^2]

        airmass_absolute : numérico
            Masa de aire absoluta. [adimensional]

        aoi : numérico o tupla de numéricos
            Ángulo de incidencia. [grados]

        Returns
        -------
        effective_irradiance : numérico o tupla de numéricos
            La irradiancia efectiva SAPM. [W/m^2]
        """
        poa_direct = self._validate_per_array(poa_direct)
        poa_diffuse = self._validate_per_array(poa_diffuse)
        aoi = self._validate_per_array(aoi)
        return tuple(
            sapm_effective_irradiance(
                poa_direct, poa_diffuse, airmass_absolute, aoi,
                array.module_parameters)
            for array, poa_direct, poa_diffuse, aoi
            in zip(self.arrays, poa_direct, poa_diffuse, aoi)
        )

    @_unwrap_single_value
    def first_solar_spectral_loss(self, pw, airmass_absolute):
        """
        Utiliza :py:func:`pvlib.spectrum.spectral_factor_firstsolar` para
        calcular el factor de pérdida espectral. Los coeficientes del modelo son
        específicos para el tipo de celda del módulo y se determinan buscando
        uno de los siguientes claves en self.module_parameters (en orden):


        - 'first_solar_spectral_coefficients' (coeficientes proporcionados por el usuario)
        - 'Technology' - una cadena que describe el tipo de celda, puede leerse desde
        la base de datos de parámetros del módulo CEC
        - 'Material' - una cadena que describe el tipo de celda, puede leerse desde
        la base de datos de módulos de Sandia.

        Parámetros
        ----------
        pw : similar a una matriz
            agua precipitable atmosférica (cm).

        airmass_absolute : similar a una matriz
            masa de aire absoluta (corregida por presión).

        Returns
        -------
        modifier: similar a una matriz o tupla de similar a una matriz
            factor de desajuste espectral (adimensional) que se puede multiplicar
            con la irradiación de banda ancha que llega a las celdas de un módulo
            para estimar la irradiación efectiva, es decir, la irradiación que se
            convierte en corriente eléctrica.
        """

        pw = self._validate_per_array(pw, system_wide=True)

        def _spectral_correction(array, pw):
            if 'first_solar_spectral_coefficients' in \
                    array.module_parameters.keys():
                coefficients = \
                    array.module_parameters[
                        'first_solar_spectral_coefficients'
                    ]
                module_type = None
            else:
                module_type = array._infer_cell_type()
                coefficients = None

            return spectrum.spectral_factor_firstsolar(
                pw, airmass_absolute, module_type, coefficients
            )
        return tuple(
            itertools.starmap(_spectral_correction, zip(self.arrays, pw))
        )

    def singlediode(self, photocurrent, saturation_current,
                    resistance_series, resistance_shunt, nnsvth,
                    ivcurve_pnts=None):
        """
        Envoltura alrededor de la función :py:func:`pvlib.pvsystem.singlediode`.


        Ver :py:func:`pvlib.pvsystem.singlediode` para más detalles.
        """
        return singlediode(photocurrent, saturation_current,
                           resistance_series, resistance_shunt, nnsvth,
                           ivcurve_pnts=ivcurve_pnts)

    def i_from_v(self, voltage, photocurrent, saturation_current,
                 resistance_series, resistance_shunt, nnsvth):
        """
        Envoltura alrededor de la función :py:func:`pvlib.pvsystem.i_from_v`.

        Ver :py:func:`pvlib.pvsystem.i_from_v` para más detalles.


        .. versión cambiada:: 0.10.0
        Los argumentos de la función han sido reordenados.
        """
        return i_from_v(voltage, photocurrent, saturation_current,
                        resistance_series, resistance_shunt, nnsvth)

    def get_ac(self, model, p_dc, v_dc=None):
        r"""
        Calcula la potencia CA a partir de la potencia CC utilizando el modelo de inversor indicado
        por "modelo" y "self.inverter_parameters".

        Parámetros
        ----------
        model : str
            Debe ser uno de 'sandia', 'adr' o 'pvwatts'.
        p_dc : numérico, o tupla, lista o matriz de numéricos
            Potencia CC en cada entrada de MPPT del inversor. Utilice tupla, lista o
            matriz para inversores con múltiples entradas MPPT. Si el tipo es matriz,
            p_dc debe ser 2D con el eje 0 siendo las entradas MPPT. [W]
        v_dc : numérico, o tupla, lista o matriz de numéricos
            Voltaje CC en cada entrada de MPPT del inversor. Requerido cuando
            model='sandia' o model='adr'. Utilice tupla, lista o matriz para inversores
            con múltiples entradas MPPT. Si el tipo es matriz, v_dc debe ser 2D con el eje 0
            siendo las entradas MPPT. [V]

        Returns
        -------
        power_ac : numérico
            Salida de potencia CA para el inversor. [W]

        Raises
        ------
        ValueError
            Si el modelo no es uno de 'sandia', 'adr' o 'pvwatts'.
        ValueError
            Si el modelo='adr' y PVSystem tiene más de un array.

        See also
        --------
        xm_solarlib.inverter.sandia
        xm_solarlib.inverter.sandia_multi
        xm_solarlib.inverter.adr
        xm_solarlib.inverter.pvwatts
        xm_solarlib.inverter.pvwatts_multi
        """

        model = model.lower()
        multiple_arrays = self.num_arrays > 1
        if model == 'sandia':
            p_dc = self._validate_per_array(p_dc)
            v_dc = self._validate_per_array(v_dc)
            if multiple_arrays:
                return inverter.sandia_multi(
                    v_dc, p_dc, self.inverter_parameters)
            return inverter.sandia(v_dc[0], p_dc[0], self.inverter_parameters)
        elif model == 'pvwatts':
            kwargs = _build_kwargs(['eta_inv_nom', 'eta_inv_ref'],
                                   self.inverter_parameters)
            p_dc = self._validate_per_array(p_dc)
            if multiple_arrays:
                return inverter.pvwatts_multi(
                    p_dc, self.inverter_parameters['pdc0'], **kwargs)
            return inverter.pvwatts(
                p_dc[0], self.inverter_parameters['pdc0'], **kwargs)
        elif model == 'adr':
            if multiple_arrays:
                raise ValueError(
                    'The adr inverter function cannot be used for an inverter',
                    ' with multiple MPPT inputs')
            # While this is only used for single-array systems, calling
            # _validate_per_arry lets us pass in singleton tuples.
            p_dc = self._validate_per_array(p_dc)
            v_dc = self._validate_per_array(v_dc)
            return inverter.adr(v_dc[0], p_dc[0], self.inverter_parameters)
        else:
            raise ValueError(
                model + ' is not a valid AC power model.',
                ' model must be one of "sandia", "adr" or "pvwatts"')

    @_unwrap_single_value
    def scale_voltage_current_power(self, data):
        """
        Escala el voltaje, la corriente y la potencia del DataFrame `data`
        por `self.modules_per_string` y `self.strings_per_inverter`.

        Parámetros
        ----------
        data: DataFrame o tupla de DataFrame
            Puede contener columnas 'v_mp', 'v_oc', 'i_mp', 'i_x', 'i_xx',
            'i_sc', 'p_mp'.

        Returns
        -------
        scaled_data: DataFrame o tupla de DataFrame
            Una copia escalada de los datos de entrada.
        """
        data = self._validate_per_array(data)
        return tuple(
            scale_voltage_current_power(data,
                                        voltage=array.modules_per_string,
                                        current=array.strings)
            for array, data in zip(self.arrays, data)
        )

    @_unwrap_single_value
    def pvwatts_dc(self, g_poa_effective, temp_cell):
        """
        Calcula la potencia CC según el modelo PVWatts utilizando
        :py:func:`pvlib.pvsystem.pvwatts_dc`, `self.module_parameters['pdc0']`
        y `self.module_parameters['gamma_pdc']`.

        Consulta :py:func:`pvlib.pvsystem.pvwatts_dc` para obtener más detalles.

        """
        g_poa_effective = self._validate_per_array(g_poa_effective)
        temp_cell = self._validate_per_array(temp_cell)
        return tuple(
            pvwatts_dc(g_poa_effective, temp_cell,
                       array.module_parameters['pdc0'],
                       array.module_parameters['gamma_pdc'],
                       **_build_kwargs(['temp_ref'], array.module_parameters))
            for array, g_poa_effective, temp_cell
            in zip(self.arrays, g_poa_effective, temp_cell)
        )

    def pvwatts_losses(self):
        """
        Calcula las pérdidas de potencia CC según el modelo PVwatts utilizando
        :py:func:`pvlib.pvsystem.pvwatts_losses` y
        ``self.losses_parameters``.

        Consulta :py:func:`pvlib.pvsystem.pvwatts_losses` para obtener detalles.

        """
        kwargs = _build_kwargs(['soiling', 'shading', 'snow', 'mismatch',
                                'wiring', 'connections', 'lid',
                                'nameplate_rating', 'age', 'availability'],
                               self.losses_parameters)
        return pvwatts_losses(**kwargs)

    @_unwrap_single_value
    def dc_ohms_from_percent(self):
        """
        Calcula la resistencia equivalente de los cables para cada matriz utilizando
        :py:func:`pvlib.pvsystem.dc_ohms_from_percent`

        Consulta :py:func:`pvlib.pvsystem.dc_ohms_from_percent` para obtener detalles.

        """

        return tuple(array.dc_ohms_from_percent() for array in self.arrays)

    @property
    def num_arrays(self):
        """El número de matrices en el sistema."""
        return len(self.arrays)

    


class Array:
    """
    Un Array es un conjunto de módulos con la misma orientación.

    Específicamente, un array se define por su montaje, los
    parámetros del módulo, el número de cadenas paralelas de módulos
    y el número de módulos en cada cadena.

    Parámetros
    ----------
    mount: FixedMount, SingleAxisTrackerMount u otro
        Montaje para el array, ya sea en una estructura de inclinación fija o en un seguidor de eje único horizontal.
        El montaje se utiliza para determinar la orientación del módulo.
        Si no se proporciona, se utiliza un FixedMount con inclinación cero.

    albedo : None o float, valor por defecto None
        Albedo de la superficie del suelo. Si es ``None``, se utiliza el valor correspondiente en
        ``irradiance.SURFACE_ALBEDOS`` según el tipo de superficie.
        Si ``surface_type`` también es None, se utiliza un albedo de superficie del suelo de 0.25.

    surface_type : None o string, valor por defecto None
        Tipo de superficie del suelo. Consulta ``irradiance.SURFACE_ALBEDOS`` para conocer los valores válidos.

    module : None o string, valor por defecto None
        Nombre del modelo de los módulos.
        Puede utilizarse para buscar el diccionario de parámetros del módulo
        a través de algún otro método.

    module_type : None o string, valor por defecto None
         Describe la construcción del módulo. Las cadenas válidas son 'vidrio_polímero'
         y 'vidrio_vidrio'. Se utiliza para el cálculo de la temperatura de las celdas y el módulo.

    module_parameters : None, dict o Series, valor por defecto None
        Parámetros para el modelo del módulo, por ejemplo, SAPM, CEC u otros.

    temperature_model_parameters : None, dict o Series, valor por defecto None.
        Parámetros para el modelo de temperatura del módulo, por ejemplo, SAPM, Pvsyst u otros.

    modules_per_string: int, valor por defecto 1
        Número de módulos por cadena en el array.

    strings: int, valor por defecto 1
        Número de cadenas en paralelo en el array.

    array_losses_parameters: None, dict o Series, valor por defecto None.
        Las claves admitidas son 'dc_ohmic_percent'.

    name: None o str, valor por defecto None
        Nombre de la instancia del Array.
    """

    def __init__(self, mount,
                 albedo=None, surface_type=None,
                 module=None, module_type=None,
                 module_parameters=None,
                 temperature_model_parameters=None,
                 modules_per_string=1, strings=1,
                 array_losses_parameters=None,
                 name=None):
        self.mount = mount

        self.surface_type = surface_type
        if albedo is None:
            self.albedo = irradiance.SURFACE_ALBEDOS.get(surface_type, 0.25)
        else:
            self.albedo = albedo

        self.module = module
        if module_parameters is None:
            self.module_parameters = {}
        else:
            self.module_parameters = module_parameters

        self.module_type = module_type

        self.strings = strings
        self.modules_per_string = modules_per_string

        if temperature_model_parameters is None:
            self.temperature_model_parameters = \
                self._infer_temperature_model_params()
        else:
            self.temperature_model_parameters = temperature_model_parameters

        if array_losses_parameters is None:
            self.array_losses_parameters = {}
        else:
            self.array_losses_parameters = array_losses_parameters

        self.name = name

    def __repr__(self):
        attrs = ['name', 'mount', 'module',
                 'albedo', 'module_type',
                 'temperature_model_parameters',
                 'strings', 'modules_per_string']

        return 'Array:\n  ' + '\n  '.join(
            f'{attr}: {getattr(self, attr)}' for attr in attrs
        )

    def _infer_temperature_model_params(self):
        # try to infer temperature model parameters from from racking_model
        # and module_type
        param_set = f'{self.mount.racking_model}_{self.module_type}'
        if param_set in temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']:
            return temperature._temperature_model_params('sapm', param_set)
        elif 'freestanding' in param_set:
            return temperature._temperature_model_params('pvsyst',
                                                         'freestanding')
        elif 'insulated' in param_set:  # after SAPM to avoid confusing keys
            return temperature._temperature_model_params('pvsyst',
                                                         'insulated')
        else:
            return {}

    def _infer_cell_type(self):
        """
        Examina los parámetros del módulo y asigna la clave Technology para la base de datos de CEC
        y la clave Material para la base de datos de Sandia a una lista común de cadenas de tipo de celda.

        Devoluciones
        -------
        cell_type: str

        """

        _cell_type_dict = {'Multi-c-Si': 'multisi',
                           'Mono-c-Si': 'monosi',
                           'Thin Film': 'cigs',
                           'a-Si/nc': 'asi',
                           'CIS': 'cigs',
                           'CIGS': 'cigs',
                           '1-a-Si': 'asi',
                           'CdTe': 'cdte',
                           'a-Si': 'asi',
                           '2-a-Si': None,
                           '3-a-Si': None,
                           'HIT-Si': 'monosi',
                           'mc-Si': 'multisi',
                           'c-Si': 'multisi',
                           'Si-Film': 'asi',
                           'EFG mc-Si': 'multisi',
                           'GaAs': None,
                           'a-Si / mono-Si': 'monosi'}

        if 'Technology' in self.module_parameters.keys():
            # CEC module parameter set
            cell_type = _cell_type_dict[self.module_parameters['Technology']]
        elif 'Material' in self.module_parameters.keys():
            # Sandia module parameter set
            cell_type = _cell_type_dict[self.module_parameters['Material']]
        else:
            cell_type = None

        return cell_type

    def get_aoi(self, solar_zenith, solar_azimuth):
        """
        Obtiene el ángulo de incidencia en el array.

        Parámetros
        ----------
        solar_zenith : float o Series
            Ángulo cenital solar.
        solar_azimuth : float o Series
            Ángulo azimutal solar.

        Devoluciones
        -------
        aoi : Series
            El ángulo de incidencia.
        """
        orientation = self.mount.get_orientation(solar_zenith, solar_azimuth)
        return irradiance.aoi(orientation['surface_tilt'],
                              orientation['surface_azimuth'],
                              solar_zenith, solar_azimuth)

    def get_irradiance(self, solar_zenith, solar_azimuth, dni, ghi, dhi,
                       dni_extra=None, airmass=None, albedo=None,
                       model='haydavies', **kwargs):
        """
        Obtener los componentes de irradiancia en el plano del array.

        Utiliza la función :py:func:`pvlib.irradiance.get_total_irradiance` para
        calcular los componentes de irradiancia en el plano del array para una superficie
        definida por ``self.surface_tilt`` y ``self.surface_azimuth``.


        Parámetros
        ----------
        solar_zenith : float o Series.
            Ángulo cenital solar.
        solar_azimuth : float o Series.
            Ángulo azimutal solar.
        dni : float o Series
            Irradiancia directa normal. [W/m2]
        ghi : float o Series. [W/m2]
            Irradiancia global horizontal
        dhi : float o Series
            Irradiancia difusa horizontal. [W/m2]
        dni_extra : None, float o Series, valor por defecto None
            Irradiancia directa normal extraterrestre. [W/m2]
        airmass : None, float o Series, valor por defecto None
            Masa de aire. [adimensional]
        albedo : None, float o Series, valor por defecto None
            Albedo de la superficie del suelo. [adimensional]
        model : String, valor por defecto 'haydavies'
            Modelo de irradiancia.

        kwargs
            Parámetros adicionales pasados a
            :py:func:`pvlib.irradiance.get_total_irradiance`.


        Devoluciones
        -------
        poa_irradiance : DataFrame
            Los nombres de las columnas son: ``'poa_global', 'poa_direct', 'poa_diffuse',
            'poa_sky_diffuse', 'poa_ground_diffuse'``.

        Ver también
        --------
        :py:func:`xm_solarlib.irradiance.get_total_irradiance`
        """
        if albedo is None:
            albedo = self.albedo

        # not needed for all models, but this is easier
        if dni_extra is None:
            dni_extra = irradiance.get_extra_radiation(solar_zenith.index)

        if airmass is None:
            airmass = atmosphere.get_relative_airmass(solar_zenith)

        orientation = self.mount.get_orientation(solar_zenith, solar_azimuth)
        return irradiance.get_total_irradiance(orientation['surface_tilt'],
                                               orientation['surface_azimuth'],
                                               solar_zenith, solar_azimuth,
                                               dni, ghi, dhi,
                                               dni_extra=dni_extra,
                                               airmass=airmass,
                                               albedo=albedo,
                                               model=model,
                                               **kwargs)

    def get_iam(self, aoi, iam_model='physical'):
        """
        Determina el modificador del ángulo de incidencia utilizando el método especificado por
        ``iam_model``.

        Se espera que los parámetros para el modelo IAM seleccionado estén en
        ``Array.module_parameters``. Se proporcionan parámetros por defecto para
        los modelos 'physical', 'ashrae' y 'martin_ruiz'.

        Parámetros
        ----------
        aoi : numérico
            El ángulo de incidencia en grados.

        iam_model : string, valor por defecto 'physical'
            El modelo IAM a utilizar. Las cadenas válidas son 'physical', 'ashrae',
            'martin_ruiz' y 'sapm'.

        Devoluciones
        -------
        iam : numérico
            El modificador de AOI.

        Levanta
        ------
        ValueError
            si `iam_model` no es un nombre de modelo válido.
        """
        model = iam_model.lower()
        if model in ['ashrae', 'physical', 'martin_ruiz']:
            param_names = iam._IAM_MODEL_PARAMS[model]
            kwargs = _build_kwargs(param_names, self.module_parameters)
            func = getattr(iam, model)
            return func(aoi, **kwargs)
        elif model == 'sapm':
            return iam.sapm(aoi, self.module_parameters)
        elif model == 'interp':
            raise ValueError(model + ' is not implemented as an IAM model '
                             'option for Array')
        else:
            raise ValueError(model + ' is not a valid IAM model')

    def get_cell_temperature(self, poa_global, temp_air, wind_speed, model,
                             effective_irradiance=None):
        """
        Determina la temperatura de las celdas utilizando el método especificado por ``model``.

        Parámetros
        ----------
        poa_global : numérico
            Irradiancia incidente total [W/m^2]

        temp_air : numérico
            Temperatura ambiente del bulbo seco [C]

        wind_speed : numérico
            Velocidad del viento [m/s]

        model : str
            Modelos admitidos incluyen ``'sapm'``, ``'pvsyst'``,
            ``'faiman'``, ``'fuentes'``, y ``'noct_sam'``

        effective_irradiance : numérico, opcional
            La irradiancia que se convierte en fotocorriente en W/m^2.
            Solo se utiliza para algunos modelos.

        Devoluciones
        -------
        numérico
            Valores en grados Celsius.

        Ver también
        --------
        xm_solarlib.temperature.sapm_cell, xm_solarlib.temperature.pvsyst_cell,
        xm_solarlib.temperature.faiman, xm_solarlib.temperature.fuentes,
        xm_solarlib.temperature.noct_sam

        Notas
        -----
        Algunos modelos de temperatura tienen requisitos para los tipos de entrada;
        consulta la documentación de la función de modelo subyacente para obtener detalles.
        """
        # convenience wrapper to avoid passing args 2 and 3 every call
        _build_tcell_args = functools.partial(
            _build_args, input_dict=self.temperature_model_parameters,
            dict_name='temperature_model_parameters')

        if model == 'sapm':
            func = temperature.sapm_cell
            required = _build_tcell_args(['a', 'b', 'deltaT'])
            optional = _build_kwargs(['irrad_ref'],
                                     self.temperature_model_parameters)
        elif model == 'pvsyst':
            func = temperature.pvsyst_cell
            required = tuple()
            optional = {
                **_build_kwargs(['module_efficiency', 'alpha_absorption'],
                                self.module_parameters),
                **_build_kwargs(['u_c', 'u_v'],
                                self.temperature_model_parameters)
            }
        elif model == 'faiman':
            func = temperature.faiman
            required = tuple()
            optional = _build_kwargs(['u0', 'u1'],
                                     self.temperature_model_parameters)
        elif model == 'fuentes':
            func = temperature.fuentes
            required = _build_tcell_args(['noct_installed'])
            optional = _build_kwargs([
                'wind_height', 'emissivity', 'absorption',
                'surface_tilt', 'module_width', 'module_length'],
                self.temperature_model_parameters)
            if self.mount.module_height is not None:
                optional['module_height'] = self.mount.module_height
        elif model == 'noct_sam':
            func = functools.partial(temperature.noct_sam,
                                     effective_irradiance=effective_irradiance)
            required = _build_tcell_args(['noct', 'module_efficiency'])
            optional = _build_kwargs(['transmittance_absorptance',
                                      'array_height', 'mount_standoff'],
                                     self.temperature_model_parameters)
        else:
            raise ValueError(f'{model} is not a valid cell temperature model')

        temperature_cell = func(poa_global, temp_air, wind_speed,
                                *required, **optional)
        return temperature_cell

    def dc_ohms_from_percent(self):
        """
        Calcula la resistencia equivalente de los cables utilizando
        :py:func:`pvlib.pvsystem.dc_ohms_from_percent`


        Utiliza los parámetros del módulo del array según los siguientes modelos DC:

        CEC:

            * `self.module_parameters["V_mp_ref"]`
            * `self.module_parameters["I_mp_ref"]`

        SAPM:

            * `self.module_parameters["Vmpo"]`
            * `self.module_parameters["Impo"]`

        PVsyst u otro similar:

            * `self.module_parameters["Vmpp"]`
            * `self.module_parameters["Impp"]`

        Otros parámetros del array que se utilizan son:
        `self.losses_parameters["dc_ohmic_percent"]`,
        `self.modules_per_string` y
        `self.strings`.

        Consulta :py:func:`pvlib.pvsystem.dc_ohms_from_percent` para más detalles.

        """
        # get relevent Vmp and Imp parameters from CEC parameters
        if all([elem in self.module_parameters
                for elem in ['V_mp_ref', 'I_mp_ref']]):
            vmp_ref = self.module_parameters['V_mp_ref']
            imp_ref = self.module_parameters['I_mp_ref']

        # get relevant Vmp and Imp parameters from SAPM parameters
        elif all([elem in self.module_parameters
                  for elem in ['Vmpo', 'Impo']]):
            vmp_ref = self.module_parameters['Vmpo']
            imp_ref = self.module_parameters['Impo']

        # get relevant Vmp and Imp parameters if they are PVsyst-like
        elif all([elem in self.module_parameters
                  for elem in ['Vmpp', 'Impp']]):
            vmp_ref = self.module_parameters['Vmpp']
            imp_ref = self.module_parameters['Impp']

        # raise error if relevant Vmp and Imp parameters are not found
        else:
            raise ValueError('Parameters for Vmp and Imp could not be found '
                             'in the array module parameters. Module '
                             'parameters must include one set of '
                             '{"V_mp_ref", "I_mp_Ref"}, '
                             '{"Vmpo", "Impo"}, or '
                             '{"Vmpp", "Impp"}.'
                             )

        return dc_ohms_from_percent(
            vmp_ref,
            imp_ref,
            self.array_losses_parameters['dc_ohmic_percent'],
            self.modules_per_string,
            self.strings)


    def singlediode(self, photocurrent, saturation_current,
                        resistance_series, resistance_shunt, nnsvth,
                        ivcurve_pnts=None):
            """Envoltura alrededor de la función :py:func:`xm_solarlib.pvsystem.singlediode`.

            Consulta :py:func:`xm_solarlib.singlediode` para más detalles.
            """
            return singlediode(photocurrent, saturation_current,
                              resistance_series, resistance_shunt, nnsvth,
                              ivcurve_pnts=ivcurve_pnts)

def singlediode(photocurrent, saturation_current, resistance_series,
                resistance_shunt, nnsvth, ivcurve_pnts=None,
                method='lambertw'):
    """
    Resuelve la ecuación del diodo único para obtener una curva IV fotovoltaica.

    Resuelve la ecuación del diodo único [1]_:

    .. math::

        I = I_L -
            I_0 \left[
                \exp \left(\frac{V+I r_s}{n N_s V_{th}} \right)-1
            \right] -
            \frac{V + I r_s}{R_{sh}}

    para :math:`I` y :math:`V` cuando se proporcionan :math:`I_L, I_0, r_s, R_{sh},` y
    :math:`n N_s V_{th}`, que se describen a continuación. Los cinco puntos en la curva I-V
    especificados en [3]_ son devueltos. Si :math:`I_L, I_0, r_s, R_{sh},` y
    :math:`n N_s V_{th}` son todos escalares, se devuelve una sola curva. Si alguno
    es de tipo array (de la misma longitud), se calculan múltiples curvas IV.

    Los parámetros de entrada se pueden calcular a partir de datos meteorológicos utilizando una
    función para un modelo de un solo diodo, por ejemplo,
    :py:func:`~pvlib.pvsystem.calcparams_desoto`.


    Parámetros
    ----------
    photocurrent : numérico
        Corriente generada por la luz :math:`I_L` (fotocorriente)
        ``0 <= photocurrent``. [A]

    saturation_current : numérico
        Corriente de saturación del diodo :math:`I_0` bajo condiciones deseadas de la curva IV
        ``0 < saturation_current``. [A]

    resistance_series : numérico
        Resistencia en serie :math:`r_s` bajo condiciones deseadas de la curva IV.
        ``0 <= resistance_series < numpy.inf``. [ohm]

    resistance_shunt : numérico
        Resistencia en derivación :math:`R_{sh}` bajo condiciones deseadas de la curva IV.
        ``0 < resistance_shunt <= numpy.inf``. [ohm]

    nnsvth : numérico
        El producto de tres componentes: 1) el factor de idealidad usual del diodo
        :math:`n`, 2) el número de celdas en serie :math:`N_s`, y 3) el voltaje
        térmico de la celda :math:`V_{th}`. El voltaje térmico de la celda (en voltios) se puede
        calcular como :math:`k_B T_c / q`, donde :math:`k_B` es
        la constante de Boltzmann (J/K), :math:`T_c` es
        la temperatura de la unión p-n en Kelvin, y :math:`q` es la carga de un electrón
        (coulombs). ``0 < nnsvth``.  [V]

    ivcurve_pnts : None o int, predeterminado None
        Número de puntos en la curva IV deseada. Si es None o 0, no se producirán puntos en
        las curvas IV.

        .. deprecated:: 0.10.0
           Utilice :py:func:`pvlib.pvsystem.v_from_i` y
           :py:func:`pvlib.pvsystem.i_from_v` en su lugar.


    method : str, predeterminado 'lambertw'
        Determina el método utilizado para calcular puntos en la curva IV. Las
        opciones son ``'lambertw'``, ``'newton'``, o ``'brentq'``.

    Returns
    -------
    dict o pandas.DataFrame
        El objeto similar a un diccionario devuelto siempre contiene las claves/columnas:

            * i_sc - corriente de cortocircuito en amperios.
            * v_oc - voltaje de circuito abierto en voltios.
            * i_mp - corriente en el punto de máxima potencia en amperios.
            * v_mp - voltaje en el punto de máxima potencia en voltios.
            * p_mp - potencia en el punto de máxima potencia en vatios.
            * i_x - corriente en amperios, en ``v = 0.5*v_oc``.
            * i_xx - corriente en amperios, en ``v = 0.5*(v_oc+v_mp)``.

        Se devuelve un diccionario cuando los parámetros de entrada son escalares o
        ``ivcurve_pnts > 0``. Si ``ivcurve_pnts > 0``, el diccionario de salida también
        incluirá las claves:

            * i - corriente de la curva IV en amperios.
            * v - voltaje de la curva IV en voltios.

    Ver también
    --------
    calcparams_desoto
    calcparams_cec
    sapm
    xm_solarlib.singlediode.bishop88

    Notas
    -----
    Si el método es ``'lambertw'``, entonces la solución empleada para resolver la
    ecuación del diodo implícito utiliza la función de Lambert W para obtener una
    función explícita de :math:`V=f(I)` y :math:`I=f(V)` como se muestra en [2]_.

    Si el método es ``'newton'``, se utiliza el método de Newton-Raphson de búsqueda de raíces.
    Debería ser seguro para curvas IV bien comportadas, pero se recomienda el método
    ``'brentq'`` para mayor fiabilidad.

    Si el método es ``'brentq'``, se utiliza el método de búsqueda por bisección de Brent que
    garantiza la convergencia limitando el voltaje entre cero y
    circuito abierto.

    Si el método es ``'newton'`` o ``'brentq'`` y se indican ``ivcurve_pnts``, entonces
    :func:`pvlib.singlediode.bishop88` [4]_ se utiliza para calcular los puntos en la curva IV
    en voltajes de diodo desde cero hasta el voltaje de circuito abierto con un espaciado logarítmico que
    se acerca a medida que aumenta el voltaje. Si el método es ``'lambertw'``, entonces los puntos
    calculados en la curva IV están espaciados linealmente.


    Referencias
    ----------
    .. [1] S.R. Wenham, M.A. Green, M.E. Watt, "Applied Photovoltaics" ISBN
       0 86758 909 4

    .. [2] A. Jain, A. Kapoor, "Exact analytical solutions of the
       parameters of real solar cells using Lambert W-function", Solar
       Energy Materials and Solar Cells, 81 (2004) 269-277.

    .. [3] D. King et al, "Sandia Photovoltaic Array Performance Model",
       SAND2004-3535, Sandia National Laboratories, Albuquerque, NM

    .. [4] "Computer simulation of the effects of electrical mismatches in
       photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
       https://doi.org/10.1016/0379-6787(88)90059-2
    """
    if ivcurve_pnts:
        warn_deprecated('0.10.0', name='xm_solarlib.pvsystem.singlediode',
                        alternative=('xm_solarlib.pvsystem.v_from_i and '
                                     'xm_solarlib.pvsystem.i_from_v'),
                        obj_type='parameter ivcurve_pnts',
                        removal='0.11.0')
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nnsvth)  # collect args
    # Calculate points on the IV curve using the LambertW solution to the
    # single diode equation
    if method.lower() == 'lambertw':
        out = _singlediode._lambertw(*args, ivcurve_pnts)
        points = out[:7]
        if ivcurve_pnts:
            ivcurve_i, ivcurve_v = out[7:]
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        v_oc = _singlediode.bishop88_v_from_i(
            0.0, *args, method=method.lower()
        )
        i_mp, v_mp, p_mp = _singlediode.bishop88_mpp(
            *args, method=method.lower()
        )
        i_sc = _singlediode.bishop88_i_from_v(
            0.0, *args, method=method.lower()
        )
        i_x = _singlediode.bishop88_i_from_v(
            v_oc / 2.0, *args, method=method.lower()
        )
        i_xx = _singlediode.bishop88_i_from_v(
            (v_oc + v_mp) / 2.0, *args, method=method.lower()
        )
        points = i_sc, v_oc, i_mp, v_mp, p_mp, i_x, i_xx

        # calculate the IV curve if requested using bishop88
        if ivcurve_pnts:
            vd = v_oc * (
                (11.0 - np.logspace(np.log10(11.0), 0.0, ivcurve_pnts)) / 10.0
            )
            ivcurve_i, ivcurve_v, _ = _singlediode.bishop88(vd, *args)

    columns = ('i_sc', 'v_oc', 'i_mp', 'v_mp', 'p_mp', 'i_x', 'i_xx')

    if all(map(np.isscalar, args)) or ivcurve_pnts:
        out = {c: p for c, p in zip(columns, points)}

        if ivcurve_pnts:
            out.update(i=ivcurve_i, v=ivcurve_v)

        return out

    points = np.atleast_1d(*points)  # convert scalars to 1d-arrays
    points = np.vstack(points).T  # collect rows into DataFrame columns

    # save the first available pd.Series index, otherwise set to None
    index = next((a.index for a in args if isinstance(a, pd.Series)), None)

    out = pd.DataFrame(points, columns=columns, index=index)

    return out

def i_from_v(voltage, photocurrent, saturation_current, resistance_series,
             resistance_shunt, nnsvth, method='lambertw'):
    '''
    Corriente del dispositivo a un voltaje dado para el modelo de diodo único.

    Utiliza el modelo de diodo único (SDM) como se describe en, por ejemplo,
    Jain y Kapoor 2004 [1]_.
    La solución se basa en la Ec. 2 de [1] excepto cuando resistencia_serie=0,
    en cuyo caso se utiliza la solución explícita para la corriente.
    Los parámetros ideales del dispositivo se especifican mediante resistencia_derivación=np.inf y
    resistencia_serie=0.
    Los valores de entrada de esta función pueden ser escalares y pandas.Series, pero es
    responsabilidad del llamante asegurarse de que los argumentos sean todos float64
    y estén dentro de los rangos adecuados.

    .. versionchanged:: 0.10.0
       Se ha reordenado los argumentos de la función.

    Parámetros
    ----------
    voltaje : numérico
        El voltaje en Voltios en las condiciones deseadas de la curva IV.

    fotocorriente : numérico
        Corriente generada por la luz (fotocorriente) en amperios en las condiciones
        deseadas de la curva IV. A menudo abreviada como ``I_L``.
        0 <= fotocorriente

    corriente_saturación : numérico
        Corriente de saturación del diodo en amperios en las condiciones deseadas de la curva IV.
        A menudo abreviada como ``I_0``.
        0 < corriente_saturación

    resistencia_serie : numérico
        Resistencia en serie en ohmios en las condiciones deseadas de la curva IV.
        A menudo abreviada como ``rs``.
        0 <= resistencia_serie < numpy.inf

    resistencia_derivación : numérico
        Resistencia en derivación en ohmios en las condiciones deseadas de la curva IV.
        A menudo abreviada como ``rsh``.
        0 < resistencia_derivación <= numpy.inf

    nnsvth : numérico
        El producto de tres componentes. 1) El factor ideal del diodo usual
        (n), 2) el número de celdas en serie (Ns) y 3) la tensión térmica de la celda
        bajo las condiciones deseadas de la curva IV (Vth). La
        tensión térmica de la celda (en voltios) se puede calcular como
        ``k*temp_celda/q``, donde k es la constante de Boltzmann (J/K),
        temp_celda es la temperatura de la unión p-n en Kelvin y
        q es la carga de un electrón (coulombs).
        0 < nnsvth

    método : str
        Método a utilizar: ``'lambertw'``, ``'newton'``, o ``'brentq'``. *Nota*:
        ``'brentq'`` está limitado al primer cuadrante solamente.

    Devuelve
    -------
    corriente : np.ndarray o escalar

    Referencias
    ----------
    .. [1] A. Jain, A. Kapoor, "Soluciones analíticas exactas de los
       parámetros de celdas solares reales utilizando la función W de Lambert",
       Solar Energy Materials and Solar Cells, 81 (2004) 269-277.
    '''
    if method.lower() == 'lambertw':
        return _singlediode._lambertw_i_from_v(
            voltage, photocurrent, saturation_current, resistance_series,
            resistance_shunt, nnsvth
        )
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        args = (voltage, photocurrent, saturation_current, resistance_series,
                resistance_shunt, nnsvth)
        current = _singlediode.bishop88_i_from_v(*args, method=method.lower())
        # find the right size and shape for returns
        size, shape = _singlediode._get_size_and_shape(args)
        if size <= 1 and shape is not None:
            current = np.tile(current, shape)
        if np.isnan(current).any() and size <= 1:
            current = np.repeat(current, size)
            if shape is not None:
                current = current.reshape(shape)
        return current

def dc_ohms_from_percent(vmp_ref, imp_ref, dc_ohmic_percent,
                         modules_per_string=1,
                         strings=1):
    """
    Calcula la resistencia equivalente de los cables a partir de un porcentaje
    de pérdida ohmica en condiciones STC.

    La resistencia equivalente se calcula con la siguiente función:

    .. math::
        rw = (L_{stc} / 100) * (Varray / Iarray)

    :math:`rw` es la resistencia equivalente en ohmios
    :math:`Varray` es el Vmp de los módulos multiplicado por módulos por cadena
    :math:`Iarray` es el Imp de los módulos multiplicado por cadenas por arreglo
    :math:`L_{stc}` es el porcentaje de pérdida DC de entrada

    Parámetros
    ----------
    vmp_ref: numérico
        Voltaje en máxima potencia en condiciones de referencia [V]
    imp_ref: numérico
        Corriente en máxima potencia en condiciones de referencia [A]
    porcentaje_ohmico_dc: numérico, valor predeterminado 0
        pérdida DC de entrada en porcentaje, por ejemplo, una pérdida del 1.5% se ingresa como 1.5
    módulos_por_cadena: int, valor predeterminado 1
        Número de módulos por cadena en el arreglo.
    cadenas: int, valor predeterminado 1
        Número de cadenas en paralelo en el arreglo.

    Devuelve
    ----------
    rw: numérico
        Resistencia equivalente [ohmios]

    Ver También
    --------
    xm_solarlib.pvsystem.dc_ohmic_losses

    Referencias
    ----------
    .. [1] Ayuda de PVsyst 7. "Pérdida de cableado ohmico del arreglo".
       https://www.pvsyst.com/help/ohmic_loss.htm
    """
    vmp = modules_per_string * vmp_ref

    imp = strings * imp_ref

    rw = (dc_ohmic_percent / 100) * (vmp / imp)

    return rw


@dataclass
class AbstractMount(ABC):
    """
    Una clase base para que las clases de Montaje extiendan. No está destinada a ser
    instanciada directamente.
    """
    @abstractmethod
    def get_orientation(self, solar_zenith, solar_azimuth):
        """
        Determina la orientación del módulo.

        Parámetros
        ----------
        zenit_solar : numérico
            Ángulo zenital solar aparente [grados]
        azimuth_solar : numérico
            Ángulo de azimut solar [grados]

        Devuelve
        -------
        orientacion : tipo-dict
            Un objeto tipo diccionario con claves `'inclinación_superficie', 'azimut_superficie'`
            (generalmente un diccionario o un pandas.DataFrame)
        """



@dataclass
class FixedMount(AbstractMount):
    """
    Montaje fijo en orientación estática.

    Parámetros
    ----------
    surface_tilt : float, por defecto 0
        Ángulo de inclinación de la superficie. El ángulo de inclinación se define como el ángulo desde la horizontal
        (por ejemplo, superficie hacia arriba = 0, superficie hacia el horizonte = 90) [grados].

    surface_azimuth : float, por defecto 180
        Ángulo de azimut de la superficie del módulo. Norte=0, Este=90, Sur=180,
        Oeste=270. [grados].

    racking_model : str, opcional
        Cadenas válidas son 'open_rack', 'close_mount' e 'insulated_back'.
        Utilizado para identificar un conjunto de parámetros para el modelo de temperatura de la celda SAPM.

    module_height : float, opcional
       La altura sobre el suelo del centro del módulo [m]. Utilizado para
       el modelo de temperatura de la celda Fuentes.
    """

    surface_tilt: float = 0.0
    surface_azimuth: float = 180.0
    racking_model: Optional[str] = None
    module_height: Optional[float] = None

    def get_orientation(self, solar_zenith, solar_azimuth):
        # note -- docstring is automatically inherited from AbstractMount
        return {
            'surface_tilt': self.surface_tilt,
            'surface_azimuth': self.surface_azimuth,
        }



@dataclass
class SingleAxisTrackerMount(AbstractMount):
    """
    Montaje de seguidor de un solo eje para seguimiento solar dinámico.

    Parámetros
    ----------
    axis_tilt : float, por defecto 0
        La inclinación del eje de rotación (es decir, el eje y definido por
        axis_azimuth) con respecto a la horizontal. [grados]

    axis_azimuth : float, por defecto 180
        Un valor que denota la dirección de la brújula a lo largo de la cual se encuentra
        el eje de rotación, medido al este del norte. [grados]

    max_angle : float o tupla, por defecto 90
        Un valor que denota el ángulo máximo de rotación, en grados decimales,
        del seguidor de un solo eje desde su posición horizontal (horizontal
        si axis_tilt = 0). Si se proporciona un número flotante, representa el ángulo máximo de rotación,
        y se asume que el ángulo mínimo de rotación es el opuesto al ángulo máximo. Si se proporciona una tupla de (min_angle, max_angle),
        representa tanto el ángulo mínimo como el máximo de rotación.

        Una rotación a 'max_angle' es una rotación en sentido contrario a las agujas del reloj alrededor del
        eje y del sistema de coordenadas del seguidor. Por ejemplo, para un seguidor
        con 'axis_azimuth' orientado hacia el sur, una rotación a 'max_angle'
        es hacia el oeste, y una rotación hacia 'min_angle' es en la
        dirección opuesta, hacia el este. Por lo tanto, un max_angle de 180 grados
        (equivalente a max_angle = (-180, 180)) permite al seguidor alcanzar
        su capacidad de rotación completa.

    backtrack : bool, por defecto True
        Controla si el seguidor tiene la capacidad de "retroceder"
        para evitar sombreado de fila a fila. False indica que no hay capacidad de retroceso.
        True indica capacidad de retroceso.

    gcr : float, por defecto 2.0/7.0
        Un valor que denota la relación de cobertura de suelo de un sistema de seguimiento
        que utiliza retroceso; es decir, la relación entre el área de la superficie de la matriz fotovoltaica
        y el área total del suelo. Un sistema de seguimiento con módulos
        de 2 metros de ancho, centrados en el eje de seguimiento, con 6 metros
        entre los ejes de seguimiento tiene un GCR de 2/6=0.333. Si no se proporciona un GCR, el valor predeterminado es de 2/7. El GCR debe ser <=1. [adimensional]

    cross_axis_tilt : float, por defecto 0.0
        El ángulo, relativo a la horizontal, de la línea formada por la
        intersección entre la pendiente que contiene los ejes del seguidor y un plano
        perpendicular a los ejes del seguidor. La inclinación cruzada debe especificarse
        utilizando una convención diestra. Por ejemplo, los seguidores con azimuth del eje de 180 grados (hacia el sur) tendrán una inclinación cruzada negativa
        si el plano de ejes del seguidor desciende hacia el este y una inclinación cruzada positiva
        si el plano de ejes del seguidor sube hacia el este. Use
        :func:`~pvlib.tracking.calc_cross_axis_tilt` para calcular
        `cross_axis_tilt`. [grados]


    racking_model : str, opcional
        Cadenas válidas son 'open_rack', 'close_mount' e 'insulated_back'.
        Utilizado para identificar un conjunto de parámetros para el modelo de temperatura de la celda SAPM.

    module_height : float, opcional
       La altura sobre el suelo del centro del módulo [m]. Utilizado para
       el modelo de temperatura de la celda Fuentes.
    """
    axis_tilt: float = 0.0
    axis_azimuth: float = 0.0
    max_angle: Union[float, tuple] = 90.0
    backtrack: bool = True
    gcr: float = 2.0/7.0
    cross_axis_tilt: float = 0.0
    racking_model: Optional[str] = None
    module_height: Optional[float] = None

    def get_orientation(self, solar_zenith, solar_azimuth):
        # note -- docstring is automatically inherited from AbstractMount
        from xm_solarlib import tracking  # avoid circular import issue
        tracking_data = tracking.singleaxis(
            solar_zenith, solar_azimuth,
            self.axis_tilt, self.axis_azimuth,
            self.max_angle, self.backtrack,
            self.gcr, self.cross_axis_tilt
        )
        return tracking_data


def calcparams_desoto(effective_irradiance, temp_cell,
                      alpha_sc, a_ref, i_l_ref, i_o_ref, r_sh_ref, r_s,
                      egref=1.121, degdt=-0.0002677,
                      irrad_ref=1000, temp_ref=25):
    '''
    Calcula cinco valores de parámetros para la ecuación de un solo diodo a
    partir de la eficiencia de irradiación y la temperatura de la celda utilizando el modelo de De Soto et al.
    descrito en [1]_. Los cinco valores devueltos por calcparams_desoto
    pueden ser utilizados por singlediode para calcular una curva IV.

    Parámetros
    ----------
    eficiencia_irradiacion : numérico
        La eficiencia de irradiación (W/m2) que se convierte en corriente fotogenerada.

    temp_celda : numérico
        La temperatura promedio de la celda de las celdas dentro de un módulo en C.

    alpha_sc : flotante
        El coeficiente de temperatura de corriente de cortocircuito del
        módulo en unidades de A/C.

    a_ref : flotante
        El producto del factor de idealidad usual del diodo (n, adimensional),
        número de celdas en serie (Ns), y la tensión térmica de la celda en condiciones de referencia,
        en unidades de V.

    i_l_ref : flotante
        La corriente generada por la luz (o corriente fotogenerada) en condiciones de referencia,
        en amperios.

    i_o_ref : flotante
        La corriente oscura o corriente de saturación inversa del diodo en condiciones de referencia,
        en amperios.

    r_sh_ref : flotante
        La resistencia en derivación en condiciones de referencia, en ohmios.

    r_s : flotante
        La resistencia en serie en condiciones de referencia, en ohmios.

    egref : flotante
        La energía de la banda prohibida a temperatura de referencia en unidades de eV.
        1.121 eV para silicio cristalino. egref debe ser >0. Para parámetros
        de la base de datos de módulos SAM CEC, egref=1.121 es implícito para todos
        los tipos de celdas en el algoritmo de estimación de parámetros utilizado por NREL.

    degdt : flotante
        La dependencia de la temperatura de la energía de la banda prohibida en condiciones de referencia
        en unidades de 1/K. Puede ser un valor escalar
        (por ejemplo, -0.0002677 como en [1]_) o un DataFrame (esto puede ser útil si
        degdt se modela como una función de la temperatura). Para parámetros de
        la base de datos de módulos SAM CEC, degdt=-0.0002677 es implícito para todos los tipos de celdas
        en el algoritmo de estimación de parámetros utilizado por NREL.

    irradiancia_ref : flotante (opcional, por defecto=1000)
        Irradiancia de referencia en W/m^2.

    temp_ref : flotante (opcional, por defecto=25)
        Temperatura de celda de referencia en C.

    Returns
    -------
    Tupla de los siguientes resultados:

    corriente_fotogenerada : numérico
        Corriente generada por la luz en amperios.

    corriente_saturacion : numérico
        Corriente de saturación del diodo en amperios.

    resistencia_serie : numérico
        Resistencia en serie en ohmios.

    resistencia_derivacion : numérico
        Resistencia en derivación en ohmios.

    nnsvth : numérico
        El producto del factor de idealidad usual del diodo (n, adimensional),
        número de celdas en serie (Ns), y la tensión térmica de la celda
        en la eficiencia de irradiación y temperatura de celda especificadas.

    References
    ----------
    .. [1] W. De Soto et al., "Improvement and validation of a model for
       photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
       2006.

    .. [2] Página web del Sistema de Asesor de Modelos. https://sam.nrel.gov.

    .. [3] A. Dobos, "An Improved Coefficient Calculator for the California
       Energy Commission 6 Parameter Photovoltaic Module Model", Journal of
       Solar Energy Engineering, vol 134, 2012.

    .. [4] O. Madelung, "Semiconductors: Data Handbook, 3rd ed." ISBN
       3-540-40488-0

    See Also
    --------
    singlediode
    retrieve_sam

    Notes
    -----
    Si los parámetros de referencia en la estructura ModuleParameters se leen
    desde una base de datos o biblioteca de parámetros (por ejemplo, Sistema Asesor
    de Modelos), es importante usar los mismos valores de egref y degdt que
    se utilizaron para generar los parámetros de referencia, independientemente de las
    características reales de la banda prohibida del semiconductor. Por ejemplo, en
    el caso de la biblioteca del Sistema de Asesor de Modelos, creada como se describe
    en [3], egref y degdt para todos los módulos eran 1.121 y -0.0002677, respectivamente.

    Esta tabla de energías de banda prohibida de referencia (egref), dependencia de la energía de la banda prohibida
    con la temperatura (degdt) y respuesta típica al aire (M) se proporciona únicamente como referencia
    para aquellos que puedan generar sus propios parámetros de módulo de referencia (a_ref, IL_ref, I0_ref, etc.)
    basados en los diversos semiconductores fotovoltaicos. Nuevamente, enfatizamos la importancia de
    usar egref y degdt idénticos cuando se generan los parámetros de referencia y se modifican los parámetros de referencia
    (para irradiación, temperatura y masa de aire) según las ecuaciones de DeSoto.

     Silicio Cristalino (Si):
         * egref = 1.121
         * degdt = -0.0002677

         >>> M = np.polyval([-1.26E-4, 2.816E-3, -0.024459, 0.086257, 0.9181],
         ...                AMa) # doctest: +SKIP

         Fuente: [1]

     Telururo de Cadmio (CdTe):
         * egref = 1.475
         * degdt = -0.0003

         >>> M = np.polyval([-2.46E-5, 9.607E-4, -0.0134, 0.0716, 0.9196],
         ...                AMa) # doctest: +SKIP

         Fuente: [4]

     Diseleniuro de Cobre Indio (CIS):
         * egref = 1.010
         * degdt = -0.00011

         >>> M = np.polyval([-3.74E-5, 0.00125, -0.01462, 0.0718, 0.9210],
         ...                AMa) # doctest: +SKIP

         Fuente: [4]

     Diseleniuro de Cobre Indio Galio (CIGS):
         * egref = 1.15
         * degdt = ????

         >>> M = np.polyval([-9.07E-5, 0.0022, -0.0202, 0.0652, 0.9417],
         ...                AMa) # doctest: +SKIP

         Fuente: Wikipedia

     Arsenuro de Galio (GaAs):
         * egref = 1.424
         * degdt = -0.000433
         * M = desconocido

         Fuente: [4]
    '''

    # Constante de Boltzmann en eV/K, 8.617332478e-05
    k = constants.value('Boltzmann constant in eV/K')

    # reference temperature
    tref_k = temp_ref + 273.15
    tcell_k = temp_cell + 273.15

    e_g = egref * (1 + degdt*(tcell_k - tref_k))

    nnsvth = a_ref * (tcell_k / tref_k)

    # En la ecuación para IL, se utiliza el factor de eficiencia efectiva_irradiacion,
    # en lugar del producto S*M en [1]. eficiencia_efectiva_irradiacion es
    # equivalente al producto de S (irradiación que llega a las celdas de un módulo) *
    # M (factor de ajuste espectral) como se describe en [1].
    IL = effective_irradiance / irrad_ref * \
        (i_l_ref + alpha_sc * (tcell_k - tref_k))
    I0 = (i_o_ref * ((tcell_k / tref_k) ** 3) *
          (np.exp(egref / (k*(tref_k)) - (e_g / (k*(tcell_k))))))
    # Nota que la ecuación para rsh difiere de [1]. En [1] rsh se da como
    # rsh = Rsh_ref * (S_ref / S) donde S es la irradiación de banda ancha que llega
    # a las celdas del módulo. Si se desea, este comportamiento del modelo se puede duplicar
    # aplicando pérdidas de reflexión y suciedad a la irradiación de banda ancha del plano del módulo
    # y no aplicando un modificador de pérdida espectral, es decir,
    # spectral_modifier = 1.0.
    # Usar errstate para silenciar la advertencia de división
    with np.errstate(divide='ignore'):
        rsh = r_sh_ref * (irrad_ref / effective_irradiance)

    rs = r_s

    numeric_args = (effective_irradiance, temp_cell)
    out = (IL, I0, rs, rsh, nnsvth)

    if all(map(np.isscalar, numeric_args)):
        return out

    index = tools.get_pandas_index(*numeric_args)

    if index is None:
        return np.broadcast_arrays(*out)

    return tuple(pd.Series(a, index=index).rename(None) for a in out)


def calcparams_cec(effective_irradiance, temp_cell,
                   alpha_sc, a_ref, i_l_ref, i_o_ref, r_sh_ref, r_s,
                   adjust, egref=1.121, degdt=-0.0002677,
                   irrad_ref=1000, temp_ref=25):
    '''
    Calcula cinco valores de parámetros para la ecuación de un solo diodo en
    eficiencia de irradiación y temperatura de celda utilizando el modelo CEC.
    El modelo CEC [1]_ difiere del modelo De Soto et al.
    [3]_ por el parámetro Ajuste. Los cinco valores devueltos por
    calcparams_cec pueden ser utilizados por singlediode para calcular una curva IV.

    Parámetros
    ----------
    eficiencia_irradiacion : numérico
        La irradiación (W/m2) que se convierte en corriente fotogenerada.

    temp_celda : numérico
        La temperatura promedio de la celda de las celdas dentro de un módulo en C.

    alpha_sc : flotante
        El coeficiente de temperatura de corriente de cortocircuito del
        módulo en unidades de A/C.

    a_ref : flotante
        El producto del factor de idealidad usual del diodo (n, adimensional),
        número de celdas en serie (Ns), y la tensión térmica de la celda en condiciones de referencia,
        en unidades de V.

    i_l_ref : flotante
        La corriente generada por la luz (o corriente fotogenerada) en condiciones de referencia,
        en amperios.

    i_o_ref : flotante
        La corriente oscura o corriente de saturación inversa del diodo en condiciones de referencia,
        en amperios.

    r_sh_ref : flotante
        La resistencia en derivación en condiciones de referencia, en ohmios.

    r_s : flotante
        La resistencia en serie en condiciones de referencia, en ohmios.

    Ajuste : flotante
        El ajuste al coeficiente de temperatura para la corriente de cortocircuito,
        en porcentaje.

    egref : flotante
        La energía de la banda prohibida a temperatura de referencia en unidades de eV.
        1.121 eV para silicio cristalino. egref debe ser >0. Para parámetros
        de la base de datos de módulos SAM CEC, egref=1.121 es implícito para todos
        los tipos de celdas en el algoritmo de estimación de parámetros utilizado por NREL.

    degdt : flotante
        La dependencia de la temperatura de la energía de la banda prohibida en condiciones de referencia
        en unidades de 1/K. Puede ser un valor escalar
        (por ejemplo, -0.0002677 como en [3]) o un DataFrame (esto puede ser útil si
        degdt se modela como una función de la temperatura). Para parámetros de
        la base de datos de módulos SAM CEC, degdt=-0.0002677 es implícito para todos los tipos de celdas
        en el algoritmo de estimación de parámetros utilizado por NREL.

    irradiancia_ref : flotante (opcional, por defecto=1000)
        Irradiancia de referencia en W/m^2.

    temp_ref : flotante (opcional, por defecto=25)
        Temperatura de celda de referencia en C.

    Returns
    -------
    Tupla de los siguientes resultados:

    corriente_fotogenerada : numérico
        Corriente generada por la luz en amperios.

    corriente_saturacion : numérico
        Corriente de saturación del diodo en amperios.

    resistencia_serie : numérico
        Resistencia en serie en ohmios.

    resistencia_derivacion : numérico
        Resistencia en derivación en ohmios.

    nnsvth : numérico
        El producto del factor de idealidad usual del diodo (n, adimensional),
        número de celdas en serie (Ns), y la tensión térmica de la celda
        en la eficiencia de irradiación y temperatura de celda especificadas.

    References
    ----------
    .. [1] A. Dobos, "An Improved Coefficient Calculator for the California
       Energy Commission 6 Parameter Photovoltaic Module Model", Journal of
       Solar Energy Engineering, vol 134, 2012.

    .. [2] Página web del Sistema de Asesor de Modelos. https://sam.nrel.gov.

    .. [3] W. De Soto et al., "Improvement and validation of a model for
       photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
       2006.

    See Also
    --------
    calcparams_desoto
    singlediode
    retrieve_sam

    '''


    # pass adjusted temperature coefficient to desoto
    return calcparams_desoto(effective_irradiance, temp_cell,
                             alpha_sc*(1.0 - adjust/100),
                             a_ref, i_l_ref, i_o_ref,
                             r_sh_ref, r_s,
                             egref=egref, degdt=degdt,
                             irrad_ref=irrad_ref, temp_ref=temp_ref)


def _normalize_sam_product_names(names):
    '''
    Replace special characters within the product names to make them more
    suitable for use as Dataframe column names.
    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. July, 2019

    import warnings

    BAD_CHARS = ' -.()[]:+/",'
    GOOD_CHARS = '____________'

    mapping = str.maketrans(BAD_CHARS, GOOD_CHARS)
    names = pd.Series(data=names)
    norm_names = names.str.translate(mapping)

    n_duplicates = names.duplicated().sum()
    if n_duplicates > 0:
        warnings.warn('Original names contain %d duplicate(s).' % n_duplicates)

    n_duplicates = norm_names.duplicated().sum()
    if n_duplicates > 0:
        warnings.warn(
            'Normalized names contain %d duplicate(s).' % n_duplicates)

    return norm_names.values


def _parse_raw_sam_df(csvdata):

    df = pd.read_csv(csvdata, index_col=0, skiprows=[1, 2])

    df.columns = df.columns.str.replace(' ', '_')
    df.index = _normalize_sam_product_names(df.index)
    df = df.transpose()

    if 'ADRCoefficients' in df.index:
        ad_ce = 'ADRCoefficients'
        # for each inverter, parses a string of coefficients like
        # ' 1.33, 2.11, 3.12' into a list containing floats:
        # [1.33, 2.11, 3.12]
        df.loc[ad_ce] = df.loc[ad_ce].map(lambda x: list(
            map(float, x.strip(' []').split())))

    return df


def retrieve_sam(name=None, path=None):
    '''
    Retrieve latest module and inverter info from a local file or the
    SAM website.

    This function will retrieve either:

        * CEC module database
        * Sandia Module database
        * CEC Inverter database
        * Anton Driesse Inverter database

    and return it as a pandas DataFrame.

    Parameters
    ----------
    name : None or string, default None
        Name can be one of:

        * 'CECMod' - returns the CEC module database
        * 'CECInverter' - returns the CEC Inverter database
        * 'SandiaInverter' - returns the CEC Inverter database
          (CEC is only current inverter db available; tag kept for
          backwards compatibility)
        * 'SandiaMod' - returns the Sandia Module database
        * 'ADRInverter' - returns the ADR Inverter database

    path : None or string, default None
        Path to the SAM file. May also be a URL.

    Returns
    -------
    samfile : DataFrame
        A DataFrame containing all the elements of the desired database.
        Each column represents a module or inverter, and a specific
        dataset can be retrieved by the command

    Raises
    ------
    ValueError
        If no name or path is provided.

    Notes
    -----
    Files available at
        https://github.com/NREL/SAM/tree/develop/deploy/libraries
    Documentation for module and inverter data sets:
        https://sam.nrel.gov/photovoltaic/pv-sub-page-2.html

    Examples
    --------

    >>> from xm_solarlib import pvsystem
    >>> invdb = pvsystem.retrieve_sam('CECInverter')
    >>> inverter = invdb.AE_Solar_Energy__AE6_0__277V_
    >>> inverter
    Vac                          277
    Pso                    36.197575
    Paco                      6000.0
    Pdco                 6158.746094
    Vdco                       360.0
    C0                     -0.000002
    C1                     -0.000026
    C2                     -0.001253
    C3                       0.00021
    Pnt                          1.8
    Vdcmax                     450.0
    Idcmax                 17.107628
    Mppt_low                   100.0
    Mppt_high                  450.0
    CEC_Date                     NaN
    CEC_Type     Utility Interactive
    Name: AE_Solar_Energy__AE6_0__277V_, dtype: object
    '''

    if name is not None:
        name = name.lower()
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data')
        if name == 'cecmod':
            csvdata = os.path.join(
                data_path, 'sam-library-cec-modules-2019-03-05.csv')
        elif name == 'sandiamod':
            csvdata = os.path.join(
                data_path, 'sam-library-sandia-modules-2015-6-30.csv')
        elif name == 'adrinverter':
            csvdata = os.path.join(
                data_path, 'adr-library-cec-inverters-2019-03-05.csv')
        elif name in ['cecinverter', 'sandiainverter']:
            # Allowing either, to provide for old code,
            # while aligning with current expectations
            csvdata = os.path.join(
                data_path, 'sam-library-cec-inverters-2019-03-05.csv')
        else:
            raise ValueError(f'invalid name {name}')
    elif path is not None:
        if path.startswith('http'):
            response = urlopen(path)
            csvdata = io.StringIO(response.read().decode(errors='ignore'))
        else:
            csvdata = path
    elif name is None and path is None:
        raise ValueError("A name or path must be provided!")

    return _parse_raw_sam_df(csvdata)


def sapm(effective_irradiance, temp_cell, module):
    '''
    El Modelo de Rendimiento de Matriz Fotovoltaica Sandia (SAPM) genera 5 puntos
    en la curva I-V de un módulo fotovoltaico (Voc, Isc, Ix, Ixx, Vmp/Imp) según
    SAND2004-3535. Supone una temperatura de celda de referencia de 25°C.

    Parámetros
    ----------
    irradiacion_efectiva : numérico
        Irradiancia que llega a las celdas del módulo, después de reflejos y
        ajuste para el espectro. [W/m2]

    temp_celda : numérico
        Temperatura de la celda [C].

    modulo : tipo diccionario
        Un diccionario o Serie que define los parámetros SAPM. Consulta la sección de notas
        para obtener más detalles.

    Returns
    -------
    Un DataFrame con las columnas:

        * i_sc : Corriente de cortocircuito (A)
        * i_mp : Corriente en el punto de máxima potencia (A)
        * v_oc : Voltaje de circuito abierto (V)
        * v_mp : Voltaje en el punto de máxima potencia (V)
        * p_mp : Potencia en el punto de máxima potencia (W)
        * i_x : Corriente en V = 0.5Voc del módulo, define el cuarto punto en la curva I-V
        * i_xx : Corriente en V = 0.5(Voc+Vmp) del módulo, define el quinto punto en
          la curva I-V para modelar la forma de la curva

    Notes
    -----
    Los parámetros SAPM requeridos en "modulo" se enumeran en la siguiente tabla.

    La base de datos de módulos de Sandia contiene valores de parámetros para un conjunto limitado
    de módulos. La base de datos de módulos CEC no contiene estos parámetros.
    Ambas bases de datos se pueden acceder mediante :py:func:`retrieve_sam`.

    ================   ========================================================
    Clave                Descripción
    ================   ========================================================
    A0-A4              Los coeficientes de masa de aire utilizados en el cálculo de
                       la irradiación efectiva.
    B0-B5              Los coeficientes de ángulo de incidencia utilizados en el cálculo
                       de la irradiación efectiva.
    C0-C7              Los coeficientes determinados empíricamente que relacionan
                       Imp, Vmp, Ix e Ixx con la irradiación efectiva.
    Isco               Corriente de cortocircuito en condiciones de referencia (amperios).
    Impo               Corriente de máxima potencia en condiciones de referencia (amperios).
    Voco               Voltaje de circuito abierto en condiciones de referencia (amperios).
    Vmpo               Voltaje de máxima potencia en condiciones de referencia (amperios).
    Aisc               Coeficiente de temperatura de la corriente de cortocircuito en
                       condiciones de referencia (1/C).
    Aimp               Coeficiente de temperatura de la corriente de máxima potencia en
                       condiciones de referencia (1/C).
    bvoco              Coeficiente de temperatura de voltaje de circuito abierto en
                       condiciones de referencia (V/C).
    Mbvoc              Coeficiente que proporciona la dependencia de la irradiación
                       para el coeficiente de temperatura BetaVoc a irradiación de
                       referencia (V/C).
    bvmpo              Coeficiente de temperatura de voltaje de máxima potencia en
                       condiciones de referencia.
    Mbvmp              Coeficiente que proporciona la dependencia de la irradiación
                       para el coeficiente de temperatura BetaVmp a irradiación de
                       referencia (V/C).
    N                  "Factor de diodo" determinado empíricamente (adimensional).
    Celdas_en_Serie    Número de celdas en serie en las cadenas de celdas del módulo.
    IXO                Ix en condiciones de referencia.
    IXXO               Ixx en condiciones de referencia.
    FD                 Fracción de irradiación difusa utilizada por el módulo.
    ================   ========================================================

    References
    ----------
    .. [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
       Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
       NM.

    See Also
    --------
    retrieve_sam
    xm_solarlib.temperature.sapm_cell
    xm_solarlib.temperature.sapm_module
    '''

    temp_ref = 25
    irrad_ref = 1000

    q = constants.e  # Elementary charge in units of coulombs
    kb = constants.k  # Boltzmann's constant in units of J/K

    # avoid problem with integer input
    ee = np.array(effective_irradiance, dtype='float64') / irrad_ref

    # set up masking for 0, positive, and nan inputs
    ee_gt_0 = np.full_like(ee, False, dtype='bool')
    ee_eq_0 = np.full_like(ee, False, dtype='bool')
    notnan = ~np.isnan(ee)
    np.greater(ee, 0, where=notnan, out=ee_gt_0)
    np.equal(ee, 0, where=notnan, out=ee_eq_0)

    bvmpo = module['bvmpo'] + module['Mbvmp']*(1 - ee)
    bvoco = module['bvoco'] + module['Mbvoc']*(1 - ee)
    delta = module['N'] * kb * (temp_cell + 273.15) / q

    # avoid repeated computation
    logee = np.full_like(ee, np.nan)
    np.log(ee, where=ee_gt_0, out=logee)
    logee = np.where(ee_eq_0, -np.inf, logee)
    # avoid repeated __getitem__
    cells_in_series = module['Cells_in_Series']

    out = OrderedDict()

    out['i_sc'] = (
        module['Isco'] * ee * (1 + module['Aisc']*(temp_cell - temp_ref)))

    out['i_mp'] = (
        module['Impo'] * (module['C0']*ee + module['C1']*(ee**2)) *
        (1 + module['Aimp']*(temp_cell - temp_ref)))

    out['v_oc'] = np.maximum(0, (
        module['Voco'] + cells_in_series * delta * logee +
        bvoco*(temp_cell - temp_ref)))

    out['v_mp'] = np.maximum(0, (
        module['Vmpo'] +
        module['C2'] * cells_in_series * delta * logee +
        module['C3'] * cells_in_series * ((delta * logee) ** 2) +
        bvmpo*(temp_cell - temp_ref)))

    out['p_mp'] = out['i_mp'] * out['v_mp']

    out['i_x'] = (
        module['IXO'] * (module['C4']*ee + module['C5']*(ee**2)) *
        (1 + module['Aisc']*(temp_cell - temp_ref)))

    # the Ixx calculation in King 2004 has a typo (mixes up Aisc and Aimp)
    out['i_xx'] = (
        module['IXXO'] * (module['C6']*ee + module['C7']*(ee**2)) *
        (1 + module['Aisc']*(temp_cell - temp_ref)))

    if isinstance(out['i_sc'], pd.Series):
        out = pd.DataFrame(out)

    return out


sapm_spectral_loss = deprecated(
    since='0.10.0',
    alternative='xm_solarlib.spectrum.spectral_factor_sapm'
)(spectrum.spectral_factor_sapm)



def sapm_effective_irradiance(poa_direct, poa_diffuse, airmass_absolute, aoi,
                              module):
    r"""
    Calcula la irradiancia efectiva SAPM utilizando las funciones de pérdida espectral SAPM
    y pérdida de ángulo de incidencia SAPM.

    Parámetros
    ----------
    poa_directo : numérico
        La irradiancia directa incidente en el módulo. [W/m2]

    poa_difuso : numérico
        La irradiancia difusa incidente en el módulo. [W/m2]

    masa_aire_absoluta : numérico
        Masa de aire absoluta. [adimensional]

    aoi : numérico
        Ángulo de incidencia. [grados]

    modulo : tipo diccionario
        Un diccionario, Serie o DataFrame que define los parámetros de rendimiento SAPM.
        Consulta la sección de notas de :py:func:`sapm` para obtener más detalles.

    Returns
    -------
    irradiancia_efectiva : numérico
        Irradiancia efectiva teniendo en cuenta reflexiones y contenido espectral.
        [W/m2]

    Notes
    -----
    El modelo SAPM para irradiancia efectiva [1]_ traduce la irradiancia directa y difusa
    de banda ancha en el plano del arreglo a la irradiancia absorbida por las celdas de un
    módulo.

    El modelo es
    .. math::

        `ee = f_1(AM_a) (E_b f_2(AOI) + f_d E_d)`

    donde :math:`ee` es la irradiancia efectiva (W/m2), :math:`f_1` es un polinomio de cuarto
    grado en la masa de aire :math:`AM_a`, :math:`E_b` es la irradiancia de haz (directa) en el
    plano del arreglo, :math:`E_d` es la irradiancia difusa en el plano del arreglo, :math:`f_2`
    es un polinomio de quinto grado en el ángulo de incidencia :math:`AOI`, y :math:`f_d` es la
    fracción de la irradiancia difusa en el plano del arreglo que no se refleja.

    References
    ----------
    .. [1] D. King et al, "Sandia Photovoltaic Array Performance Model",
       SAND2004-3535, Sandia National Laboratories, Albuquerque, NM

    See also
    --------
    xm_solarlib.iam.sapm
    xm_solarlib.spectrum.spectral_factor_sapm
    xm_solarlib.pvsystem.sapm
    """

    F1 = spectrum.spectral_factor_sapm(airmass_absolute, module)
    F2 = iam.sapm(aoi, module)

    ee = F1 * (poa_direct * F2 + module['FD'] * poa_diffuse)

    return ee


def scale_voltage_current_power(data, voltage=1, current=1):
    """
    Escala el voltaje, la corriente y la potencia en los datos por los factores de voltaje y corriente.

    Parámetros
    ----------
    datos: DataFrame
        Puede contener las columnas `'v_mp', 'v_oc', 'i_mp' ,'i_x', 'i_xx', 'i_sc', 'p_mp'`.
    voltaje: numérico, por defecto 1
        La cantidad por la cual multiplicar los voltajes.
    corriente: numérico, por defecto 1
        La cantidad por la cual multiplicar las corrientes.

    Returns
    -------
    datos_escalados: DataFrame
        Una copia escalada de los datos de entrada.
        `'p_mp'` se escala por `voltaje * corriente`.
    """

    # as written, only works with a DataFrame
    # could make it work with a dict, but it would be more verbose
    voltage_keys = ['v_mp', 'v_oc']
    current_keys = ['i_mp', 'i_x', 'i_xx', 'i_sc']
    power_keys = ['p_mp']
    voltage_df = data.filter(voltage_keys, axis=1) * voltage
    current_df = data.filter(current_keys, axis=1) * current
    power_df = data.filter(power_keys, axis=1) * voltage * current
    df = pd.concat([voltage_df, current_df, power_df], axis=1)
    df_sorted = df[data.columns]  # retain original column order
    return df_sorted


def pvwatts_dc(g_poa_effective, temp_cell, pdc0, gamma_pdc, temp_ref=25.):
    r"""
    Implementa el modelo de potencia DC de PVWatts de NREL. El modelo DC de PVWatts [1]_ es:

    .. math::

        P_{dc} = \frac{G_{poa efectiva}}{1000} P_{dc0} (1 + \gamma_{pdc} (T_{celda} - T_{ref}))

    Ten en cuenta que ``pdc0`` también se utiliza como símbolo en
    :py:func:`pvlib.inverter.pvwatts`. ``pdc0`` en esta función se refiere a la potencia DC
    de los módulos en condiciones de referencia. ``pdc0`` en
    :py:func:`pvlib.inverter.pvwatts` se refiere al límite de entrada de potencia DC
    del inversor.


    Parámetros
    ----------
    g_poa_efectiva: numérico
        Irradiancia transmitida a las células fotovoltaicas. Para ser
        completamente consistente con PVWatts, el usuario debe haber aplicado
        previamente pérdidas por ángulo de incidencia, pero no suciedad, espectral,
        etc. [W/m^2]
    temp_celda: numérico
        Temperatura de la célula [°C].
    pdc0: numérico
        Potencia de los módulos a 1000 W/m^2 y temperatura de referencia de la célula. [W]
    gamma_pdc: numérico
        El coeficiente de temperatura de la potencia. Típicamente -0.002 a
        -0.005 por grado Celsius. [1/°C]
    temp_ref: numérico, por defecto 25.0
        Temperatura de referencia de la célula. PVWatts la define como 25 °C y
        se incluye aquí para flexibilidad. [°C]

    Returns
    -------
    pdc: numérico
        Potencia DC. [W]

    Referencias
    ----------
    .. [1] A. P. Dobos, "Manual de PVWatts Versión 5"
           http://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
           (2014).
    """  # noqa: E501

    pdc = (g_poa_effective * 0.001 * pdc0 *
           (1 + gamma_pdc * (temp_cell - temp_ref)))

    return pdc


def pvwatts_losses(soiling=2, shading=3, snow=0, mismatch=2, wiring=2,
                   connections=0.5, lid=1.5, nameplate_rating=1, age=0,
                   availability=3):
    """
    Implementa el modelo de pérdidas del sistema PVWatts de NREL.
    El modelo de pérdidas PVWatts [1]_ es:

    .. math::

        L_{total}(\%) = 100 [ 1 - \Pi_i ( 1 - \frac{L_i}{100} ) ]

    Todos los parámetros deben estar en unidades de %. Los parámetros
    pueden ser tipo array, pero todos los tamaños de arrays deben coincidir.

    Parámetros
    ----------
    soiling: numérico, por defecto 2
    shading: numérico, por defecto 3
    nieve: numérico, por defecto 0
    desajuste: numérico, por defecto 2
    cableado: numérico, por defecto 2
    conexiones: numérico, por defecto 0.5
    lid: numérico, por defecto 1.5
        Degradación inducida por la luz
    calificacion_placa: numérico, por defecto 1
    edad: numérico, por defecto 0
    disponibilidad: numérico, por defecto 3

    Retorna
    -------
    pérdidas: numérico
        Pérdidas del sistema en unidades de %.

    Referencias
    ----------
    .. [1] A. P. Dobos, "Manual de PVWatts Versión 5"
           http://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
           (2014).
    """

    params = [soiling, shading, snow, mismatch, wiring, connections, lid,
              nameplate_rating, age, availability]

    # manually looping over params allows for numpy/pandas to handle any
    # array-like broadcasting that might be necessary.
    perf = 1
    for param in params:
        perf *= 1 - param/100

    losses = (1 - perf) * 100.

    return losses