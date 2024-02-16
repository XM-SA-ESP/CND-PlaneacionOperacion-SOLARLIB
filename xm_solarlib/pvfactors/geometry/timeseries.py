
"""Herramientas de geometría de series temporales. Permiten la vectorización de la geometría.
cálculos."""

import numpy as np
from xm_solarlib.pvfactors.config import DISTANCE_TOLERANCE
from xm_solarlib.pvfactors.geometry.base import PVSurface, ShadeCollection
from shapely.geometry import GeometryCollection


class TsShadeCollection(object):
    """Colección de superficies de series temporales que están sombreadas o
     iluminado. Esto será utilizado tanto por la fila de tierra como por la de PV.
     geometrías."""

    def __init__(self, list_ts_surfaces, shaded):
        """Inicializar usando la lista de superficies y el estado de sombreado

         Parámetros
         ----------
         lista_ts_superficies: \
         lista de :py:class:`~pvfactors.geometry.timeseries.TsSurface`
             Lista de superficies de series temporales en la colección
         sombreado: booleano
             Estado de sombreado de la colección.
         """
        self._list_ts_surfaces = list_ts_surfaces
        self.shaded = shaded

    @property
    def list_ts_surfaces(self):
        """Lista de superficies de series temporales en la colección"""
        return self._list_ts_surfaces

    @property
    def length(self):
        """Longitud total de la colección"""
        length = 0.
        for ts_surf in self._list_ts_surfaces:
            length += ts_surf.length
        return length

    @property
    def n_ts_surfaces(self):
        """Número de superficies de series temporales en la colección"""
        return len(self._list_ts_surfaces)

    def get_param_weighted(self, param):
        """Obtener el parámetro de serie temporal para la colección, después de ponderar por
         longitud de la superficie.

         Parámetros
         ----------
         parámetro: cadena
             Nombre del parámetro

         Devoluciones
         -------
         np.ndarray
             Valores de parámetros ponderados
         """
        return self.get_param_ww(param) / self.length

    def get_param_ww(self, param):
        """Obtener el parámetro de serie temporal de la colección con peso,
         es decir, después de multiplicar por las longitudes de las superficies.

         Parámetros
         ----------
         parámetro: cadena
             Parámetro de superficie a devolver

         Devoluciones
         -------
         np.ndarray
             Valores de parámetros de series temporales multiplicados por pesos
         """

        value = 0
        for ts_surf in self._list_ts_surfaces:
            value += ts_surf.length * ts_surf.get_param(param)
        return value

    def update_params(self, new_dict):
        """Actualizar parámetros de superficie de serie temporal del segmento.

         Parámetros
         ----------
         new_dict: dict
             Parámetros para agregar o actualizar para las superficies.
         """
        for ts_surf in self._list_ts_surfaces:
            ts_surf.params.update(new_dict)

    def at(self, idx):
        """Generar una colección de tonos puntuales para el índice deseado.

         Parámetros
         ----------
         idx:int
             Índice a utilizar para generar la colección de sombras.

         Devoluciones
         -------
         colección: :py:clase:`~pvfactors.geometry.base.ShadeCollection`
         """
        list_surfaces = [ts_surf.at(idx) for ts_surf in self._list_ts_surfaces
                         if not ts_surf.at(idx).is_empty]
        return ShadeCollection(list_surfaces, shaded=self.shaded)


class TsSurface(object):
    """Clase de superficie de serie temporal: representación vectorizada de la superficie fotovoltaica
     geometrías."""

    def __init__(self, coords, n_vector=None, param_names=None, index=None,
                 shaded=False):
        """Inicializar la superficie de la serie temporal utilizando coordenadas de la serie temporal.

         Parámetros
         ----------
         coordenadas: :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas de serie temporal del segmento completo.
         índice: int, opcional
             Índice de segmento (Predeterminado = Ninguno)
         n_vector: np.ndarray, opcional
             Vectores normales de serie temporal del lado (Predeterminado = Ninguno)
         índice: int, opcional
             Índice de las superficies de la serie temporal (Predeterminado = Ninguno)
         sombreado: bool, opcional
             ¿La superficie está sombreada o no? (Predeterminado = Falso)
         """
        self.coords = coords
        self.param_names = [] if param_names is None else param_names
        # because if the coords change, they won't be altered. But speed...
        self.n_vector = n_vector
        self.params = dict.fromkeys(self.param_names)
        self.index = index
        self.shaded = shaded

    def at(self, idx):
        """Genere una geometría de segmento PV para el índice deseado.

         Parámetros
         ----------
         idx:int
             Índice que se utilizará para generar la geometría del segmento fotovoltaico

         Devoluciones
         -------
         segmento :py:clase:`~pvfactors.geometry.base.PVSurface` \
         o :py:clase:`~shapely.geometry.GeometryCollection`
             El objeto devuelto será una geometría vacía si su longitud es
             muy pequeño, de lo contrario será una geometría de superficie fotovoltaica
         """
        if self.length[idx] < DISTANCE_TOLERANCE:
            # return an empty geometry
            return GeometryCollection()
        else:
            # Get normal vector at idx
            n_vector = (self.n_vector[:, idx] if self.n_vector is not None
                        else None)
            # Get params at idx
            params = _get_params_at_idx(idx, self.params)
            # Return a pv surface geometry with given params
            return PVSurface(self.coords.at(idx), shaded=self.shaded,
                             index=self.index, normal_vector=n_vector,
                             param_names=self.param_names,
                             params=params)

    def plot_at_idx(self, idx, ax, color):
        """Trazar la fila PV de la serie temporal en un índice determinado, sólo si no es así
         demasiado pequeña.

         Parámetros
         ----------
         idx:int
             Índice que se utilizará para trazar la superficie fotovoltaica de la serie temporal
         hacha: :py:clase:`matplotlib.pyplot.axes` objeto
             Ejes para trazar
         color_shaded: str, opcional
             Color a utilizar para trazar la superficie fotovoltaica
         """
        if self.length[idx] > DISTANCE_TOLERANCE:
            self.at(idx).plot(ax, color=color)

    @property
    def b1(self):
        """Coordenadas de serie temporal del primer punto límite"""
        return self.coords.b1

    @property
    def b2(self):
        """Coordenadas de serie temporal del segundo punto límite"""
        return self.coords.b2

    @property
    def centroid(self):
        """Coordenadas de puntos de serie temporal del centroide de la superficie"""
        return self.coords.centroid

    def get_param(self, param):
        """Obtener valores de parámetros de serie temporal de superficie

         Parámetros
         ----------
         parámetro: cadena
             Parámetro de superficie a devolver

         Devoluciones
         -------
         np.ndarray
             Valores de parámetros de series temporales
         """
        return self.params[param]

    def update_params(self, new_dict):
        """Actualizar parámetros de superficie de serie temporal.

         Parámetros
         ----------
         new_dict: dict
             Parámetros para agregar o actualizar para la superficie
         """
        self.params.update(new_dict)

    @property
    def length(self):
        """Longitud de la serie temporal de la superficie"""
        return self.coords.length

    @property
    def highest_point(self):
        """Coordenadas del punto de serie temporal del punto más alto de la superficie"""
        return self.coords.highest_point

    @property
    def lowest_point(self):
        """Coordenadas del punto de serie temporal del punto más bajo de la superficie"""
        return self.coords.lowest_point

    @property
    def u_vector(self):
        """Vector ortogonal al vector normal de la superficie"""
        u_vector = (None if self.n_vector is None else
                    np.array([-self.n_vector[1, :], self.n_vector[0, :]]))
        return u_vector

    @property
    def is_empty(self):
        """Compruebe si la superficie está "vacía" comprobando si su longitud es siempre
         cero"""
        return np.nansum(self.length) < DISTANCE_TOLERANCE


class TsLineCoords(object):
    """Clase de coordenadas de línea de serie temporal: proporcionará una herramienta útil
     API para invocar coordenadas de series temporales."""

    def __init__(self, b1_ts_coords, b2_ts_coords, coords=None):
        """Inicializar las coordenadas de línea de la serie temporal utilizando la serie temporal
         coordenadas de sus límites.

         Parámetros
         ----------
         b1_ts_coords: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Coordenadas de serie temporal del primer punto límite.
         b2_ts_coords: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Coordenadas de serie temporal del segundo punto límite.
         coordenadas: np.ndarray, opcional
             Coordenadas de series temporales como matriz numpy
         """
        self.b1 = b1_ts_coords
        self.b2 = b2_ts_coords

    def at(self, idx):
        """Obtener coordenadas en un índice determinado

         Parámetros
         ----------
         idx:int
             Índice a utilizar para obtener coordenadas.
         """
        return self.as_array[:, :, idx]

    @classmethod
    def from_array(cls, coords_array):
        """Crear coordenadas de línea de serie temporal a partir de una gran variedad de coordenadas. 
        
        Parámetros 
        ---------- 
        matriz de coordenadas: np.ndarray Gran variedad de coordenadas. 
        """
        b1 = TsPointCoords.from_array(coords_array[0, :, :])
        b2 = TsPointCoords.from_array(coords_array[1, :, :])
        return cls(b1, b2)

    @property
    def length(self):
        """Longitud de la serie temporal de la línea."""
        return np.sqrt((self.b2.y - self.b1.y)**2
                       + (self.b2.x - self.b1.x)**2)

    @property
    def as_array(self):
        """Coordenadas de línea de serie temporal como matriz numerosa"""
        return np.array([[self.b1.x, self.b1.y], [self.b2.x, self.b2.y]])

    @property
    def centroid(self):
        """Coordenadas de puntos de serie temporal de las coordenadas de línea"""
        dy = self.b2.y - self.b1.y
        dx = self.b2.x - self.b1.x
        return TsPointCoords(self.b1.x + 0.5 * dx, self.b1.y + 0.5 * dy)

    @property
    def highest_point(self):
        """Coordenadas del punto de la serie temporal del punto más alto de las coordenadas de línea de la serie temporal"""
        is_b1_highest = self.b1.y >= self.b2.y
        x = np.where(is_b1_highest, self.b1.x, self.b2.x)
        y = np.where(is_b1_highest, self.b1.y, self.b2.y)
        return TsPointCoords(x, y)

    @property
    def lowest_point(self):
        """Coordenadas del punto de la serie temporal del punto más bajo de las coordenadas de línea de la serie temporal"""
        is_b1_highest = self.b1.y >= self.b2.y
        x = np.where(is_b1_highest, self.b2.x, self.b1.x)
        y = np.where(is_b1_highest, self.b2.y, self.b1.y)
        return TsPointCoords(x, y)

    def __repr__(self):
        """Usa la representación de matriz numpy de las coordenadas"""
        return str(self.as_array)


class TsPointCoords(object):
    """Coordenadas de puntos de series temporales: proporciona una API con formas proporcionadas para series temporales
     coordenadas del punto."""

    def __init__(self, x, y):
        """Inicializa las coordenadas de los puntos de la serie temporal utilizando una gran variedad de coordenadas.

         Parámetros
         ----------
         x: np.ndarray
             Serie temporal x coordenadas
         y: np.ndarray
             Series temporales y coordenadas
         """
        self.x = x
        self.y = y

    def at(self, idx):
        """Obtener coordenadas en un índice determinado

         Parámetros
         ----------
         idx:int
             Índice a utilizar para obtener coordenadas.
         """
        return self.as_array[:, idx]

    @property
    def as_array(self):
        """Coordenadas de puntos de serie temporal como matriz numerosa"""
        return np.array([self.x, self.y])

    @classmethod
    def from_array(cls, coords_array):
        """Cree coordenadas de puntos de series temporales a partir de una gran variedad de coordenadas.

         Parámetros
         ----------
         matriz_coords: np.ndarray
             Gran conjunto de coordenadas.
         """
        return cls(coords_array[0, :], coords_array[1, :])

    def __repr__(self):
        """Usa la representación de matriz numpy del punto"""
        return str(self.as_array)


def _get_params_at_idx(idx, params_dict):
    """Obtiene los valores de los parámetros en el índice dado. Devuelve el parámetro completo
     cuando es Ninguno, un escalar o un diccionario

     Parámetros
     ----------
     idx:int
         Índice en el que queremos los valores de los parámetros.
     params_dict: dictar
         Diccionario de parámetros

     Devoluciones
     -------
     Valor del parámetro en el índice
     """
    if params_dict is None:
        return None
    else:
        return {k: (val if (val is None) or np.isscalar(val)
                    or isinstance(val, dict) else val[idx])
                for k, val in params_dict.items()}