"""Módulo con Clases Base para modelos de irradiancia"""
import numpy as np
import logging

class BaseModel(object):
    """Clase base para modelos de irradiancia"""

    params = None
    cats = None
    irradiance_comp = None

    def fit(self, *args, **kwargs):
        """No se ha implementado"""
        raise NotImplementedError

    def transform(self, *args, **kwargs):
        """No se ha implementado"""
        raise NotImplementedError

    def get_full_modeling_vectors(self, *args, **kwargs):
        """No se ha implementado"""
        raise NotImplementedError

    @property
    def gnd_shaded(self):
        """No se ha implementado"""
        raise NotImplementedError

    @property
    def gnd_illum(self):
        """No se ha implementado"""
        raise NotImplementedError

    @property
    def pvrow_shaded(self):
        """No se ha implementado"""
        raise NotImplementedError

    @property
    def pvrow_illum(self):
        """No se ha implementado"""
        raise NotImplementedError

    @property
    def sky_luminance(self):
        """No se ha implementado"""
        raise NotImplementedError

    def get_ts_modeling_vectors(self, pvarray):
        """Obtener matrices de valores de irradiancia resumidos de un conjunto fotovoltaico, así como
         como los valores de reflectividad inversa (estos últimos deben denominarse
         "inv_rho") y los valores de irradiancia total de perez.

         Parámetros
         ----------
         pvarray: objeto de matriz fotovoltaica
             Conjunto fotovoltaico con los valores de irradiancia y reflectividad.

         Devoluciones
         -------
         irradiance_mat: lista
             Matriz de valores resumidos de irradiancia no reflectante para todos
             superficies de series temporales. Dimensión = [n_superficies, n_timesteps]
         rho_mat: lista
             Matriz de valores de reflectividad para todas las superficies de series temporales.
             Dimensión = [n_superficies, n_timesteps]
         invrho_mat: lista
             Lista de reflectividad inversa para todas las superficies de series temporales.
             Dimensión = [n_superficies, n_timesteps]
         total_perez_mat: lista
             Lista de valores de irradiancia transpuesta total de Pérez para todas las series temporales
             superficies
             Dimensión = [n_superficies, n_timesteps]
         """

        irradiance_mat = []
        rho_mat = []
        invrho_mat = []
        total_perez_mat = []
        # In principle, the list all ts surfaces should be ordered
        # with the ts surfaces' indices. ie first element has index 0, etc
        for ts_surface in pvarray.all_ts_surfaces:
            value = np.zeros(pvarray.n_states, dtype=float)
            for component in self.irradiance_comp:
                value += ts_surface.get_param(component)
            irradiance_mat.append(value)
            invrho_mat.append(ts_surface.get_param('inv_rho'))
            rho_mat.append(ts_surface.get_param('rho'))
            total_perez_mat.append(ts_surface.get_param('total_perez'))

        return irradiance_mat, rho_mat, invrho_mat, total_perez_mat

    def get_summed_components(self, pvarray, absorbed=True):
        """Obtener la suma de los componentes de irradiancia para el modelo de irradiancia,
         ya sea absorbido o sólo incidente.

         Parámetros
         ----------
         pvarray: objeto de matriz fotovoltaica
             Conjunto fotovoltaico con los valores de irradiancia y reflectividad.
         absorbido: bool, opcional
             Bandera para decidir si utilizar los componentes absorbidos no son
             (predeterminado = Verdadero)

         Devoluciones
         -------
         irradiance_mat: lista
             Matriz de valores resumidos de irradiancia no reflectante para todos
             superficies de series temporales. Dimensión = [n_superficies, n_timesteps]"""
        # Initialize
        list_components = (self.irradiance_comp_absorbed if absorbed
                           else self.irradiance_comp)
        # Build list
        irradiance_mat = []
        for ts_surface in pvarray.all_ts_surfaces:
            value = np.zeros(pvarray.n_states, dtype=float)
            for component in list_components:
                value += ts_surface.get_param(component)
            irradiance_mat.append(value)
        return np.array(irradiance_mat)

    def update_ts_surface_sky_term(self, ts_surface, name_sky_term='sky_term'):
        """Actualice el parámetro 'sky_term' de una superficie de serie temporal.

         Parámetros
         ----------
         ts_surface: :py:clase:`~pvfactors.geometry.timeseries.TsSurface`
             Superficie de serie temporal cuyo valor del parámetro 'sky_term' queremos
             actualizar
         name_sky_term: cadena, opcional
             Nombre del parámetro sky term (predeterminado = 'sky_term')
         """
        value = 0.
        for component in self.irradiance_comp:
            value += ts_surface.get_param(component)
        ts_surface.update_params({name_sky_term: value})

    def initialize_rho(self, rho_scalar, rho_calculated, default_value):
        """Inicializar valor de reflectividad:
         - si se pasa un valor escalar, úselo
         - de lo contrario intente utilizar el valor calculado
         - de lo contrario use el valor predeterminado

         Parámetros
         ----------
         rho_scalar: flotante
             Valor de reflectividad promedio global que se supone debe usarse
         rho_calculated: flotante
             Valor de reflectividad calculado
         valor_predeterminado: flotante
             Valor predeterminado a usar si todo falla

         Devoluciones
         -------
         rho_scalar: flotante
             Reflectividad media global
         """
        if np.isscalar(rho_scalar):
            logging.debug('Using scalar value for rho: %s', rho_scalar)
        elif np.isscalar(rho_calculated):
            rho_scalar = rho_calculated
        else:
            rho_scalar = default_value
        return rho_scalar