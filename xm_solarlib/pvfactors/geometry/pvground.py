from xm_solarlib.pvfactors.config import (
    MAX_X_GROUND, MIN_X_GROUND, Y_GROUND, DISTANCE_TOLERANCE, COLOR_DIC)
from xm_solarlib.pvfactors.geometry.base import (
    BaseSide, PVSegment, ShadeCollection, PVSurface)
from xm_solarlib.pvfactors.geometry.timeseries import (
    TsShadeCollection, TsLineCoords, TsPointCoords, TsSurface,
    _get_params_at_idx)
from shapely.geometry import LineString
import numpy as np
from copy import deepcopy
import logging


class TsGround(object):
    """Clase básica de serie temporal: esta clase es una versión vectorizada de la
     Clase de geometría de suelo fotovoltaico y almacenará terreno sombreado de series temporales
     y elementos de suelo iluminados, así como puntos de corte de filas fotovoltaicas."""

    x_min = MIN_X_GROUND
    x_max = MAX_X_GROUND

    def __init__(self, shadow_elements, illum_elements, param_names=None,
                 flag_overlap=None, cut_point_coords=None, y_ground=None):
        """Inicializar el terreno de la serie temporal usando una lista de superficies de la serie temporal
         para las sombras del suelo

         Parámetros
         ----------
         shadow_elements: \
         lista de :py:class:`~pvfactors.geometry.pvground.TsGroundElement`
             Elementos de suelo sombreados de series temporales.
         illum_elements: \
         lista de :py:class:`~pvfactors.geometry.pvground.TsGroundElement`
             Elementos terrestres iluminados de serie temporal.
         param_names: lista de cadenas, opcional
             Lista de nombres de parámetros de superficie para usar al crear geometrías
             (Predeterminado = Ninguno)
         flag_overlap: lista de bool, opcional
             Banderas que indican si las sombras del suelo se superponen, para todos
             pasos de tiempo (predeterminado = ninguno). Es decir. ¿Hay sombreado directo en las filas pv?
         cut_point_coords: \
         lista de :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`, \
         opcional
             Lista de coordenadas de puntos de corte, calculadas para filas PV de series temporales
             (Predeterminado = Ninguno)
         y_ground: flotante, opcional
             Coordenada Y del terreno plano [m] (Predeterminado=Ninguno)
         """
        # Lists of timeseries ground elements
        self.shadow_elements = shadow_elements
        self.illum_elements = illum_elements
        # Shade collections
        list_shaded_surf = []
        list_illum_surf = []
        for shadow_el in shadow_elements:
            list_shaded_surf += shadow_el.all_ts_surfaces
        for illum_el in illum_elements:
            list_illum_surf += illum_el.all_ts_surfaces
        self.illum = TsShadeCollection(list_illum_surf, False)
        self.shaded = TsShadeCollection(list_shaded_surf, True)
        # Other ground attributes
        self.param_names = [] if param_names is None else param_names
        self.flag_overlap = flag_overlap
        self.cut_point_coords = [] if cut_point_coords is None \
            else cut_point_coords
        self.y_ground = y_ground
        self.shaded_params = dict.fromkeys(self.param_names)
        self.illum_params = dict.fromkeys(self.param_names)

    @classmethod
    def from_ts_pvrows_and_angles(cls, list_ts_pvrows, alpha_vec, rotation_vec,
                                  y_ground=Y_GROUND, flag_overlap=None,
                                  param_names=None):
        """Crear terreno de serie temporal a partir de la lista de filas PV de serie temporal, y
         Conjunto fotovoltaico y ángulos solares.

         Parámetros
         ----------
         lista_ts_pvrows: \
         lista de :py:class:`~pvfactors.geometry.pvrow.TsPVRow`
             Filas de PV de la serie temporal que se utilizarán para calcular las sombras del terreno de la serie temporal
         alpha_vec: np.ndarray
             Ángulo formado por el vector solar 2d y el eje x del conjunto fotovoltaico [rad]
         rotación_vec: np.ndarray
             Valores de rotación de series temporales de la fila PV [grados]
         y_ground: flotante, opcional
             Coordenada y fija de terreno plano [m] (predeterminado = constante Y_GROUND)
         flag_overlap: lista de bool, opcional
             Banderas que indican si las sombras del suelo se superponen, para todos
             pasos de tiempo (predeterminado = ninguno). Es decir. ¿Hay sombreado directo en las filas pv?
         param_names: lista de cadenas, opcional
             Lista de nombres de parámetros de superficie para usar al crear geometrías
             (Predeterminado = Ninguno)
         """
        rotation_vec = np.deg2rad(rotation_vec)
        n_steps = len(rotation_vec)
        # Calculate coords of ground shadows and cutting points
        ground_shadow_coords = []
        cut_point_coords = []
        for ts_pvrow in list_ts_pvrows:
            # Get pvrow coords
            x1s_pvrow = ts_pvrow.full_pvrow_coords.b1.x
            y1s_pvrow = ts_pvrow.full_pvrow_coords.b1.y
            x2s_pvrow = ts_pvrow.full_pvrow_coords.b2.x
            y2s_pvrow = ts_pvrow.full_pvrow_coords.b2.y
            # --- Shadow coords calculation
            # Calculate x coords of shadow
            x1s_shadow = x1s_pvrow - (y1s_pvrow - y_ground) / np.tan(alpha_vec)
            x2s_shadow = x2s_pvrow - (y2s_pvrow - y_ground) / np.tan(alpha_vec)
            # Order x coords from left to right
            x1s_on_left = x1s_shadow <= x2s_shadow
            xs_left_shadow = np.where(x1s_on_left, x1s_shadow, x2s_shadow)
            xs_right_shadow = np.where(x1s_on_left, x2s_shadow, x1s_shadow)
            # Append shadow coords to list
            ground_shadow_coords.append(
                [[xs_left_shadow, y_ground * np.ones(n_steps)],
                 [xs_right_shadow, y_ground * np.ones(n_steps)]])
            # --- Cutting points coords calculation
            dx = (y1s_pvrow - y_ground) / np.tan(rotation_vec)
            cut_point_coords.append(
                TsPointCoords(x1s_pvrow - dx, y_ground * np.ones(n_steps)))

        ground_shadow_coords = np.array(ground_shadow_coords)
        return cls.from_ordered_shadows_coords(
            ground_shadow_coords, flag_overlap=flag_overlap,
            cut_point_coords=cut_point_coords, param_names=param_names,
            y_ground=y_ground)

    @classmethod
    def from_ordered_shadows_coords(cls, shadow_coords, flag_overlap=None,
                                    param_names=None, cut_point_coords=None,
                                    y_ground=Y_GROUND):
        """Crear terreno de serie temporal a partir de la lista de coordenadas de sombra del terreno.

         Parámetros
         ----------
         coordenadas_sombra: np.ndarray
             Lista de coordenadas ordenadas de sombra del suelo (de izquierda a derecha)
         flag_overlap: lista de bool, opcional
             Banderas que indican si las sombras del suelo se superponen, para todos
             pasos de tiempo (predeterminado = ninguno). Es decir. ¿Hay sombreado directo en las filas pv?
         param_names: lista de cadenas, opcional
             Lista de nombres de parámetros de superficie para usar al crear geometrías
             (Predeterminado = Ninguno)
         coordenadas_punto_corte: \
         lista de :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`, \
         opcional
             Lista de coordenadas de puntos de corte, calculadas para filas PV de series temporales
             (Predeterminado = Ninguno)
         y_ground: flotante, opcional
             Coordenada y fija de terreno plano [m] (predeterminado = constante Y_GROUND)
         """

        # Get cut point coords if any
        cut_point_coords = cut_point_coords or []
        # Create shadow coordinate objects
        list_shadow_coords = [TsLineCoords.from_array(coords)
                              for coords in shadow_coords]
        # If the overlap flags were passed, make sure shadows don't overlap
        if flag_overlap is not None:
            if len(list_shadow_coords) > 1:
                for idx, coords in enumerate(list_shadow_coords[:-1]):
                    coords.b2.x = np.where(flag_overlap,
                                           list_shadow_coords[idx + 1].b1.x,
                                           coords.b2.x)
        # Create shaded ground elements
        ts_shadows_elements = cls._shadow_elements_from_coords_and_cut_pts(
            list_shadow_coords, cut_point_coords, param_names)
        # Create illuminated ground elements
        ts_illum_elements = cls._illum_elements_from_coords_and_cut_pts(
            ts_shadows_elements, cut_point_coords, param_names, y_ground)
        return cls(ts_shadows_elements, ts_illum_elements,
                   param_names=param_names, flag_overlap=flag_overlap,
                   cut_point_coords=cut_point_coords, y_ground=y_ground)

    def _at_with_cut_points(self,idx, merge_if_flag_overlap, non_pt_shadow_elements ):
        # We want the ground surfaces broken up at the cut points
        if merge_if_flag_overlap:
            # We want to merge the shadow surfaces when they overlap
            list_shadow_surfaces = self._merge_shadow_surfaces(idx, non_pt_shadow_elements)
        else:
            # No need to merge the shadow surfaces
            list_shadow_surfaces = []
            for shadow_el in non_pt_shadow_elements:
                list_shadow_surfaces += shadow_el.non_point_surfaces_at(idx)
        # Get the illuminated surfaces
        list_illum_surfaces = []
        for illum_el in self.illum_elements:
            list_illum_surfaces += illum_el.non_point_surfaces_at(idx)
        return list_shadow_surfaces, list_illum_surfaces

    def at(self, idx, x_min_max=None, merge_if_flag_overlap=True, with_cut_points=True):
        """Genere una geometría de tierra fotovoltaica para el índice deseado. Esto
         sólo devolver superficies no puntuales dentro de los límites del terreno, es decir,
         superficies que no son puntos y que están dentro de x_min y x_max.

         Parámetros
         ----------
         idx:int
             Índice a utilizar para generar la geometría del suelo fotovoltaico
         x_min_max: tupla, opcional
             Lista de coordenadas x mínimas y máximas para la superficie plana [m]
             (Predeterminado = Ninguno)
         merge_if_flag_overlap: bool, opcional
             Decide si fusionar todas las sombras si se superponen o no
             (Predeterminado = Verdadero)
         with_cut_points: booleano, opcional
             Decida si desea incluir los puntos de corte guardados en el archivo creado.
             Geometría del suelo fotovoltaico (predeterminado = verdadero)

         Devoluciones
         -------
         pvground : :py:clase:`~pvfactors.geometry.pvground.PVGround`
         """
        # Get shadow elements that are not points at the given index
        non_pt_shadow_elements = [shadow_el for shadow_el in self.shadow_elements
            if shadow_el.coords.length[idx] > DISTANCE_TOLERANCE]

        if with_cut_points:
            list_shadow_surfaces, list_illum_surfaces = self._at_with_cut_points(idx,merge_if_flag_overlap, non_pt_shadow_elements)
        else:
            # No need to break up the surfaces at the cut points
            # We will need to build up new surfaces (since not done by classes)

            # Get the parameters at the given index
            illum_params = _get_params_at_idx(idx, self.illum_params)
            shaded_params = _get_params_at_idx(idx, self.shaded_params)

            if merge_if_flag_overlap and (self.flag_overlap is not None):
                # We want to merge the shadow surfaces when they overlap
                is_overlap = self.flag_overlap[idx]
                if is_overlap and (len(non_pt_shadow_elements) > 1):
                    coords = [non_pt_shadow_elements[0].b1.at(idx),
                              non_pt_shadow_elements[-1].b2.at(idx)]
                    list_shadow_surfaces = [PVSurface(
                        coords, shaded=True, param_names=self.param_names,
                        params=shaded_params)]
                else:
                    # No overlap for the given index or config
                    list_shadow_surfaces = [
                        PVSurface(shadow_el.coords.at(idx),
                                  shaded=True, params=shaded_params,
                                  param_names=self.param_names)
                        for shadow_el in non_pt_shadow_elements
                        if shadow_el.coords.length[idx]
                        > DISTANCE_TOLERANCE]
            else:
                # No need to merge the shadow surfaces
                list_shadow_surfaces = [
                    PVSurface(shadow_el.coords.at(idx),
                              shaded=True, params=shaded_params,
                              param_names=self.param_names)
                    for shadow_el in non_pt_shadow_elements
                    if shadow_el.coords.length[idx]
                    > DISTANCE_TOLERANCE]
            # Get the illuminated surfaces
            list_illum_surfaces = [PVSurface(illum_el.coords.at(idx),
                                             shaded=False, params=illum_params,
                                             param_names=self.param_names)
                                   for illum_el in self.illum_elements
                                   if illum_el.coords.length[idx]
                                   > DISTANCE_TOLERANCE]

        # Pass the created lists to the PVGround builder
        return PVGround.from_lists_surfaces(
            list_shadow_surfaces, list_illum_surfaces,
            param_names=self.param_names, y_ground=self.y_ground,
            x_min_max=x_min_max)

    def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                    color_illum=COLOR_DIC['pvrow_illum'], x_min_max=None,
                    merge_if_flag_overlap=True, with_cut_points=True,
                    with_surface_index=False):
        """Trazar el terreno de la serie temporal en un índice determinado.

         Parámetros
         ----------
         idx:int
             Índice que se utilizará para trazar el lado de la serie temporal
         hacha: :py:clase:`matplotlib.pyplot.axes` objeto
             Ejes para trazar
         color_shaded: str, opcional
             Color a utilizar para trazar las superficies sombreadas (Predeterminado =
             COLOR_DIC['pvrow_shaded'])
         x_min_max: tupla, opcional
             Lista de coordenadas x mínimas y máximas para la superficie plana [m]
             (Predeterminado = Ninguno)
         merge_if_flag_overlap: bool, opcional
             Decide si fusionar todas las sombras si se superponen o no
             (Predeterminado = Verdadero)
         with_cut_points: booleano, opcional
             Decida si desea incluir los puntos de corte guardados en el archivo creado.
             Geometría del suelo fotovoltaico (predeterminado = verdadero)
         with_surface_index: bool, opcional
             Trazar las superficies con sus valores de índice (Predeterminado = Falso)
         """
        pvground = self.at(idx, x_min_max=x_min_max,
                           merge_if_flag_overlap=merge_if_flag_overlap,
                           with_cut_points=with_cut_points)
        pvground.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
                      with_index=with_surface_index)

    def update_params(self, new_dict):
        """Actualiza los parámetros iluminados por otros nuevos, no sólo para el
         terreno de la serie temporal, sino también por sus elementos básicos y la serie temporal
         superficies de los elementos del suelo, para que estén todos sincronizados.

         Parámetros
         ----------
         new_dict: dict
             Nuevos parámetros
         """
        self.update_illum_params(new_dict)
        self.update_shaded_params(new_dict)

    def update_illum_params(self, new_dict):
        """Actualiza los parámetros iluminados por otros nuevos, no sólo para el
         terreno de la serie temporal, sino también por sus elementos básicos y la serie temporal
         superficies de los elementos del suelo, para que estén todos sincronizados.

         Parámetros
         ----------
         new_dict: dict
             Nuevos parámetros
         """
        self.illum_params.update(new_dict)
        for illum_el in self.illum_elements:
            illum_el.params.update(new_dict)
            for surf in illum_el.surface_list:
                surf.params.update(new_dict)

    def update_shaded_params(self, new_dict):
        """Actualice los parámetros sombreados con otros nuevos, no sólo para el
         terreno de la serie temporal, sino también por sus elementos básicos y la serie temporal
         superficies de los elementos del suelo, para que estén todos sincronizados.

         Parámetros
         ----------
         new_dict: dict
             Nuevos parámetros
         """
        self.shaded_params.update(new_dict)
        for shaded_el in self.shadow_elements:
            shaded_el.params.update(new_dict)
            for surf in shaded_el.surface_list:
                surf.params.update(new_dict)

    def get_param_weighted(self, param):
        """Obtener el parámetro de serie temporal para el terreno ts, después de ponderar por
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
        """Obtener parámetros de series temporales de las superficies del suelo con peso,
         es decir, después de multiplicar por las longitudes de las superficies.

         Parámetros
         ----------
         parámetro: cadena
             Parámetro de superficie a devolver

         Devoluciones
         -------
         np.ndarray
             Valores de parámetros de series temporales multiplicados por pesos

         aumentos
         ------
         Error de clave
             si el nombre del parámetro no está en los parámetros de una superficie
         """
        value = 0.
        for shadow_el in self.shadow_elements:
            value += shadow_el.get_param_ww(param)
        for illum_el in self.illum_elements:
            value += illum_el.get_param_ww(param)
        return value

    def shadow_coords_left_of_cut_point(self, idx_cut_pt):
        """Obtener las coordenadas de las sombras ubicadas en el lado izquierdo del punto de corte
         con índice dado. Las coordenadas de las sombras estarán acotadas.
         por las coordenadas del punto de corte y el mínimo predeterminado
         valores x del terreno.

         Parámetros
         ----------
         idx_cut_pt:int
             Índice del punto de interés del corte.

         Devoluciones
         -------
         lista de :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas de las sombras en el lado izquierdo del punto de corte.
         """
        cut_pt_coords = self.cut_point_coords[idx_cut_pt]
        return [shadow_el._coords_left_of_cut_point(shadow_el.coords,
                                                    cut_pt_coords)
                for shadow_el in self.shadow_elements]

    def shadow_coords_right_of_cut_point(self, idx_cut_pt):
        """Obtener coordenadas de sombras ubicadas en el lado derecho del corte
         punto con índice dado. Las coordenadas de las sombras estarán acotadas.
         por las coordenadas del punto de corte y el máximo predeterminado
         valores x del terreno.

         Parámetros
         ----------
         idx_cut_pt:int
             Índice del punto de interés del corte.

         Devoluciones
         -------
         lista de :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas de las sombras en el lado derecho del punto de corte.
         """
        cut_pt_coords = self.cut_point_coords[idx_cut_pt]
        return [shadow_el._coords_right_of_cut_point(shadow_el.coords,
                                                     cut_pt_coords)
                for shadow_el in self.shadow_elements]

    def ts_surfaces_side_of_cut_point(self, side, idx_cut_pt):
        """Obtenga una lista de todas las superficies del terreno y un lado de solicitud de
         un punto de corte

         Parámetros
         ----------
         lado: cadena
             Lado del punto de corte, ya sea "izquierdo" o "derecho"
         idx_cut_pt:int
             Índice del punto de corte, de cuyo lado queremos conseguir el terreno.
             superficies

         Devoluciones
         -------
         lista
             Lista de superficies de terreno de series temporales en el lado del punto de corte
         """
        list_ts_surfaces = []
        for shadow_el in self.shadow_elements:
            list_ts_surfaces += shadow_el.surface_dict[idx_cut_pt][side]
        for illum_el in self.illum_elements:
            list_ts_surfaces += illum_el.surface_dict[idx_cut_pt][side]
        return list_ts_surfaces

    @property
    def n_ts_surfaces(self):
        """Número de superficies de series temporales en el terreno ts"""
        return self.n_ts_shaded_surfaces + self.n_ts_illum_surfaces

    @property
    def n_ts_shaded_surfaces(self):
        """Número de superficies de series temporales sombreadas en el terreno ts"""
        n_ts_surfaces = 0
        for shadow_el in self.shadow_elements:
            n_ts_surfaces += shadow_el.n_ts_surfaces
        return n_ts_surfaces

    @property
    def n_ts_illum_surfaces(self):
        """Número de superficies de series temporales iluminadas en el suelo ts"""
        n_ts_surfaces = 0
        for illum_el in self.illum_elements:
            n_ts_surfaces += illum_el.n_ts_surfaces
        return n_ts_surfaces

    @property
    def all_ts_surfaces(self):
        """Número de superficies de series temporales en el terreno ts"""
        all_ts_surfaces = []
        for shadow_el in self.shadow_elements:
            all_ts_surfaces += shadow_el.all_ts_surfaces
        for illum_el in self.illum_elements:
            all_ts_surfaces += illum_el.all_ts_surfaces
        return all_ts_surfaces

    @property
    def length(self):
        """Duración del terreno de la serie temporal"""
        length = 0
        for shadow_el in self.shadow_elements:
            length += shadow_el.length
        for illum_el in self.illum_elements:
            length += illum_el.length
        return length

    @property
    def shaded_length(self):
        """Duración del terreno de la serie temporal"""
        length = 0
        for shadow_el in self.shadow_elements:
            length += shadow_el.length
        return length

    def non_point_shaded_surfaces_at(self, idx):
        """Devuelve una lista de superficies sombreadas, que no son puntos
         en el índice dado

         Parámetros
         ----------
         idx:int
             Índice en el que queremos que las superficies no sean puntos.

         Devoluciones
         -------
         lista de :py:clase:`~pvfactors.geometry.base.PVSurface`
         """
        logging.debug("Getting non point shaded surfaces at idx %s", idx)
        list_surfaces = []
        for shadow_el in self.shadow_elements:
            list_surfaces += shadow_el.non_point_surfaces_at(0)
        return list_surfaces

    def non_point_illum_surfaces_at(self, idx):
        """Devuelve una lista de superficies iluminadas, que no están
         puntos en el índice dado

         Parámetros
         ----------
         idx:int
             Índice en el que queremos que las superficies no sean puntos.

         Devoluciones
         -------
         lista de :py:clase:`~pvfactors.geometry.base.PVSurface`
         """
        logging.debug("Getting non point illum surfaces at idx %s", idx)
        list_surfaces = []
        for illum_el in self.illum_elements:
            list_surfaces += illum_el.non_point_surfaces_at(0)
        return list_surfaces

    def non_point_surfaces_at(self, idx):
        """Devuelve una lista de todas las superficies que no están
         puntos en el índice dado

         Parámetros
         ----------
         idx:int
             Índice en el que queremos que las superficies no sean puntos.

         Devoluciones
         -------
         lista de :py:clase:`~pvfactors.geometry.base.PVSurface`
         """
        return self.non_point_illum_surfaces_at(idx) \
            + self.non_point_shaded_surfaces_at(idx)

    def n_non_point_surfaces_at(self, idx):
        """Devuelve el número de :py:class:`~pvfactors.geometry.base.PVSurface`
         que no son puntos en un índice dado

         Parámetros
         ----------
         idx:int
             Índice en el que queremos que las superficies no sean puntos.

         Devoluciones
         -------
         En t
         """
        return len(self.non_point_surfaces_at(idx))

    @staticmethod
    def _shadow_elements_from_coords_and_cut_pts(
            list_shadow_coords, cut_point_coords, param_names):
        """Crear elementos de sombra en el suelo a partir de una lista de sombras ordenadas
         coordenadas (de izquierda a derecha) y las coordenadas del punto de corte del terreno.

         Notas
         -----
         Este método recortará las coordenadas de la sombra hasta el límite del terreno,
         es decir, las coordenadas de la sombra no deben estar fuera del rango
         [MIN_X_GROUND, MAX_X_GROUND].

         Parámetros
         ----------
         list_sombra_coords: \
         lista de :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
             Lista de coordenadas ordenadas de sombra del suelo (de izquierda a derecha)
         cut_point_coords: \
         lista de :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
             Lista de coordenadas de puntos de corte (de izquierda a derecha)
         param_names: lista
             Lista de nombres de parámetros para los elementos de tierra.

         Devoluciones
         -------
         lista_sombra_elementos: \
         lista de :py:class:`~pvfactors.geometry.pvground.TsGroundElement`
             Lista ordenada de elementos de sombra (de izquierda a derecha)
         """

        list_shadow_elements = []
        for shadow_coords in list_shadow_coords:
            shadow_coords.b1.x = np.clip(shadow_coords.b1.x, MIN_X_GROUND,
                                         MAX_X_GROUND)
            shadow_coords.b2.x = np.clip(shadow_coords.b2.x, MIN_X_GROUND,
                                         MAX_X_GROUND)
            list_shadow_elements.append(
                TsGroundElement(shadow_coords,
                                list_ordered_cut_pts_coords=cut_point_coords,
                                param_names=param_names, shaded=True))

        return list_shadow_elements

    @staticmethod
    def _illum_elements_from_coords_and_cut_pts(
            list_shadow_elements, cut_pt_coords, param_names, y_ground):
        """Crear elementos iluminados en el suelo a partir de una lista de sombras ordenadas.
         elementos (de izquierda a derecha) y las coordenadas del punto de corte del terreno.
         Este método asegurará que los elementos de tierra iluminados estén
         todo dentro de los límites del terreno [MIN_X_GROUND, MAX_X_GROUND].

         Parámetros
         ----------
         lista_sombra_coords: \
         lista de :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
             Lista de coordenadas ordenadas de sombra del suelo (de izquierda a derecha)
         coordenadas_punto_corte: \
         lista de :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
             Lista de coordenadas de puntos de corte (de izquierda a derecha)
         nombres_parámetros: lista
             Lista de nombres de parámetros para los elementos de tierra.

         Devoluciones
         -------
         lista_sombra_elementos: \
         lista de :py:class:`~pvfactors.geometry.pvground.TsGroundElement`
             Lista ordenada de elementos de sombra (de izquierda a derecha)
         """

        list_illum_elements = []
        if len(list_shadow_elements) == 0:
            msg = """There must be at least one shadow element on the ground,
            otherwise it probably means that no PV rows were created, so
            there's no point in running a simulation..."""
            raise ValueError(msg)
        n_steps = len(list_shadow_elements[0].coords.b1.x)
        y_ground_vec = y_ground * np.ones(n_steps)
        next_x = MIN_X_GROUND * np.ones(n_steps)
        # Build the groud elements from left to right, starting at x_min
        # and covering the ground with illuminated elements where there's no
        # shadow
        for shadow_element in list_shadow_elements:
            x1 = next_x
            x2 = shadow_element.coords.b1.x
            coords = TsLineCoords.from_array(
                np.array([[x1, y_ground_vec], [x2, y_ground_vec]]))
            list_illum_elements.append(TsGroundElement(
                coords, list_ordered_cut_pts_coords=cut_pt_coords,
                param_names=param_names, shaded=False))
            next_x = shadow_element.coords.b2.x
        # Add the last illuminated element to the list
        coords = TsLineCoords.from_array(
            np.array([[next_x, y_ground_vec],
                      [MAX_X_GROUND * np.ones(n_steps), y_ground_vec]]))
        list_illum_elements.append(TsGroundElement(
            coords, list_ordered_cut_pts_coords=cut_pt_coords,
            param_names=param_names, shaded=False))

        return list_illum_elements

    def _merge_shadow_surfaces(self, idx, non_pt_shadow_elements):
        list_shadow_surfaces = []
        if self.flag_overlap is not None:
            # Get the overlap flags
            is_overlap = self.flag_overlap[idx]
            n_shadow_elements = len(non_pt_shadow_elements)
            if is_overlap and (n_shadow_elements > 1):
                list_shadow_surfaces = self._merge_shadow_surfaces_if_is_overlap_and_many_shadawns(idx,list_shadow_surfaces, non_pt_shadow_elements, n_shadow_elements)
                
            else:
                # There's no need to merge anything
                for shadow_el in non_pt_shadow_elements:
                    list_shadow_surfaces += \
                        shadow_el.non_point_surfaces_at(idx)
        else:
            # There's no need to merge anything
            for shadow_el in non_pt_shadow_elements:
                list_shadow_surfaces += shadow_el.non_point_surfaces_at(idx)

        return list_shadow_surfaces
    
    def _first_surface(self, surface_to_merge, surface, list_shadow_surfaces: list):
        # first surface but definitely not last either
        if surface_to_merge is not None:
            coords = [surface_to_merge.boundary[0],
                    surface.boundary[1]]
            return PVSurface(coords, shaded=True,
                        param_names=self.param_names,
                        params=surface.params,
                        index=surface.index)
        else:
            return surface


    def _merge_shadow_surfaces_if_is_overlap_and_many_shadawns(self,idx, list_shadow_surfaces, non_pt_shadow_elements, n_shadow_elements):
        # If there's only one shadow, not point in going through this

        # Now go from left to right and merge shadow surfaces
        surface_to_merge = None
        for i_el, shadow_el in enumerate(non_pt_shadow_elements):
            surfaces = shadow_el.non_point_surfaces_at(idx)
            for i_surf, surface in enumerate(surfaces):
                if i_surf == len(surfaces) - 1:
                    # last surface, could also be first
                    if i_surf == 0 and surface_to_merge is not None:
                        surface = PVSurface(
                            [surface_to_merge.boundary[0], surface.boundary[1]], shaded=True,
                            param_names=self.param_names, params=surface.params, index=surface.index)
                    if i_el == n_shadow_elements - 1:
                        # last surface of last shadow element
                        list_shadow_surfaces.append(surface)
                    else:
                        # keep for merging with next element
                        surface_to_merge = surface
                elif i_surf == 0:
                    # first surface but definitely not last either
                    list_shadow_surfaces.append(self._first_surface(surface_to_merge, surface, list_shadow_surfaces))
                else:
                    # not first nor last surface
                    list_shadow_surfaces.append(surface)
        return list_shadow_surfaces

class TsGroundElement(object):
    """Clase especial para elementos terrestres de series temporales: un elemento terrestre ha conocido
     límites de coordenadas de la serie temporal, pero también tendrá un desglose de
     su área en n+1 superficies de series temporales ubicadas en las n+1 zonas terrestres
     definido por los n puntos de corte del terreno.
     Esto es crucial para calcular los factores de vista de forma vectorizada."""

    def __init__(self, coords, list_ordered_cut_pts_coords=None,
                 param_names=None, shaded=False):
        """Inicializar el elemento terreno de la serie temporal utilizando su serie temporal
         coordenadas de línea y construir las superficies de series temporales para todos los
         zonas de puntos de corte.

         Parámetros
         ----------
         coordenadas: :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas de línea de serie temporal del elemento terrestre.
         list_ordered_cut_pts_coords: lista, opcional
             Lista de todas las coordenadas de la serie temporal de puntos de corte
             (Predeterminado = [])
         param_names: lista de cadenas, opcional
             Lista de nombres de parámetros de superficie para usar al crear geometrías
             (Predeterminado = Ninguno)
         sombreado: bool, opcional
             Bandera que especifica si el elemento es una sombra o no (Predeterminado = Falso)
         """
        self.coords = coords
        self.param_names = param_names or []
        self.params = dict.fromkeys(self.param_names)
        self.shaded = shaded
        self.surface_dict = None  # will be necessary for view factor calcs
        self.surface_list = []  # will be necessary for vf matrix formation
        list_ordered_cut_pts_coords = list_ordered_cut_pts_coords or []
        if len(list_ordered_cut_pts_coords) > 0:
            self._create_all_ts_surfaces(list_ordered_cut_pts_coords)
        self.n_ts_surfaces = len(self.surface_list)

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
        """Coordenadas de puntos de serie temporal del centroide del elemento"""
        return self.coords.centroid

    @property
    def length(self):
        """Longitud de la serie temporal del terreno"""
        return self.coords.length

    @property
    def all_ts_surfaces(self):
        """Lista de todas las superficies ts que componen el elemento de tierra ts"""
        return self.surface_list

    def surfaces_at(self, idx):
        """Devuelve la lista de superficies (de izquierda a derecha) en el índice dado que
         forman el elemento tierra.

         Parámetros
         ----------
         idx:int
             Índice de interés

         Devoluciones
         -------
         lista de :py:clase:`~pvfactors.geometry.base.PVSurface`
         """
        return [surface.at(idx)
                for surface in self.surface_list]

    def non_point_surfaces_at(self, idx):
        """Devuelve la lista de superficies no puntuales (de izquierda a derecha) en un punto determinado
         Índice que constituye el elemento suelo.

         Parámetros
         ----------
         idx:int
             Índice de interés

         Devoluciones
         -------
         lista de :py:clase:`~pvfactors.geometry.base.PVSurface`
         """
        return [surface.at(idx)
                for surface in self.surface_list
                if surface.length[idx] > DISTANCE_TOLERANCE]

    def get_param_weighted(self, param):
        """Obtener el parámetro de serie temporal para el elemento terreno, después de ponderar por
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
        """Obtener el parámetro de serie temporal del elemento de tierra con peso,
         es decir, después de multiplicar por las longitudes de las superficies.

         Parámetros
         ----------
         parámetro: cadena
             Parámetro de superficie a devolver

         Devoluciones
         -------
         np.ndarray
             Valores de parámetros de series temporales multiplicados por pesos

         aumentos
         ------
         Error de clave
             si el nombre del parámetro no está en los parámetros de una superficie
         """
        value = 0.
        for ts_surf in self.surface_list:
            value += ts_surf.length * ts_surf.get_param(param)
        return value

    def _create_all_ts_surfaces(self, list_ordered_cut_pts):
        """Crea todas las n+1 superficies de series temporales que componen la serie temporal.
         elemento suelo, y que se encuentran en las n+1 zonas definidas por
         los n puntos de corte.

         Parámetros
         ----------
         list_ordered_cut_pts: lista de :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
             Lista de coordenadas de series temporales de todos los puntos de corte, ordenadas desde
             de izquierda a derecha
         """
        # Initialize dict
        self.surface_dict = {i: {'right': [], 'left': []}
                             for i in range(len(list_ordered_cut_pts))}
        n_cut_pts = len(list_ordered_cut_pts)

        next_coords = self.coords
        for idx_pt, cut_pt_coords in enumerate(list_ordered_cut_pts):
            # Get coords on left of cut pt
            coords_left = self._coords_left_of_cut_point(next_coords,
                                                         cut_pt_coords)
            # Save that surface in the required structures
            surface_left = TsSurface(coords_left, param_names=self.param_names,
                                     shaded=self.shaded)
            self.surface_list.append(surface_left)
            for i in range(idx_pt, n_cut_pts):
                self.surface_dict[i]['left'].append(surface_left)
            for j in range(0, idx_pt):
                self.surface_dict[j]['right'].append(surface_left)
            next_coords = self._coords_right_of_cut_point(next_coords,
                                                          cut_pt_coords)
        # Save the right most portion
        next_surface = TsSurface(next_coords, param_names=self.param_names,
                                 shaded=self.shaded)
        self.surface_list.append(next_surface)
        for j in range(0, n_cut_pts):
            self.surface_dict[j]['right'].append(next_surface)

    @staticmethod
    def _coords_right_of_cut_point(coords, cut_pt_coords):
        """Calcular las coordenadas de la línea de la serie temporal que están a la derecha del dado
         coordenadas del punto de corte, pero aún dentro del área del terreno

         Parámetros
         ----------
         coordenadas: :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas originales de la serie temporal
         cut_pt_coords:
         :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Coordenadas de serie temporal del punto de corte.
         Devoluciones
         -------
         :py:clase:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas de línea de serie temporal que se encuentran a la derecha del corte
             punto
         """
        coords = deepcopy(coords)
        coords.b1.x = np.maximum(coords.b1.x, cut_pt_coords.x)
        coords.b1.x = np.minimum(coords.b1.x, MAX_X_GROUND)
        coords.b2.x = np.maximum(coords.b2.x, cut_pt_coords.x)
        coords.b2.x = np.minimum(coords.b2.x, MAX_X_GROUND)
        return coords

    @staticmethod
    def _coords_left_of_cut_point(coords, cut_pt_coords):
        """Calcular las coordenadas de la línea de la serie temporal que quedan a la izquierda del dado
         coordenadas del punto de corte, pero aún dentro del área del terreno

         Parámetros
         ----------
         coordenadas: :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas originales de la serie temporal
         cut_pt_coords:
         :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Coordenadas de serie temporal del punto de corte.
         Devoluciones
         -------
         :py:clase:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas de línea de serie temporal que se encuentran a la izquierda del corte.
             punto
         """
        coords = deepcopy(coords)
        coords.b1.x = np.minimum(coords.b1.x, cut_pt_coords.x)
        coords.b1.x = np.maximum(coords.b1.x, MIN_X_GROUND)
        coords.b2.x = np.minimum(coords.b2.x, cut_pt_coords.x)
        coords.b2.x = np.maximum(coords.b2.x, MIN_X_GROUND)
        return coords


class PVGround(BaseSide):
    """Clase que define la geometría del terreno en paneles fotovoltaicos."""

    def __init__(self, list_segments=None, original_linestring=None):
        """Inicializar la geometría del suelo fotovoltaico.

         Parámetros
         ----------
         list_segments: lista de :py:class:`~pvfactors.geometry.base.PVSegment`, opcional
             Lista de segmentos fotovoltaicos que constituirán el terreno (predeterminado = [])
         original_linestring: :py:clase:`shapely.geometry.LineString`, opcional
             Línea continua completa de la que estará hecho el terreno.
             (Predeterminado = Ninguno)
         """
        list_segments = list_segments or []
        self.original_linestring = original_linestring
        super(PVGround, self).__init__(list_segments)

    @classmethod
    def as_flat(cls, x_min_max=None, shaded=False, y_ground=Y_GROUND,
                param_names=None):
        """Construya una superficie de suelo plana horizontal, formada por 1 segmento fotovoltaico.

         Parámetros
         ----------
         x_min_max: tupla, opcional
             Lista de coordenadas x mínimas y máximas para la superficie plana [m]
             (Predeterminado = Ninguno)
         sombreado: bool, opcional
             Estado sombreado de las superficies fotovoltaicas creadas (predeterminado = falso)
         y_ground: flotante, opcional
             Ubicación del terreno plano en el eje y en [m] (Predeterminado = Y_GROUND)
         param_names: lista de cadenas, opcional
             Nombres de los parámetros de la superficie, por ejemplo, reflectividad, incidente total.
             irradiancia, temperatura, etc. (Predeterminado = [])

         Devoluciones
         -------
         Objeto PVGround
         """
        param_names = param_names or []
        # Get ground boundaries
        if x_min_max is None:
            x_min, x_max = MIN_X_GROUND, MAX_X_GROUND
        else:
            x_min, x_max = x_min_max
        # Create PV segment for flat ground
        coords = [(x_min, y_ground), (x_max, y_ground)]
        seg = PVSegment.from_linestring_coords(coords, shaded=shaded,
                                               normal_vector=[0., 1.],
                                               param_names=param_names)
        return cls(list_segments=[seg], original_linestring=LineString(coords))

    @classmethod
    def from_lists_surfaces(
            cls, list_shaded_surfaces, list_illum_surfaces, x_min_max=None,
            y_ground=Y_GROUND, param_names=None):
        """Crea terreno a partir de listas de superficies fotovoltaicas iluminadas y sombreadas.

         Parámetros
         ----------
         lista_superficies_sombreadas: \
         lista de :py:clase:`~pvfactors.geometry.base.PVSurface`
             Lista de superficies fotovoltaicas de suelo sombreadas
         list_illum_surfaces: \
         lista de :py:clase:`~pvfactors.geometry.base.PVSurface`
             Lista de superficies fotovoltaicas terrestres iluminadas
         x_min_max: tupla, opcional
             Lista de coordenadas x mínimas y máximas para la superficie plana [m]
             (Predeterminado = Ninguno)
         y_ground: flotante, opcional
             Ubicación del terreno plano en el eje y en [m] (Predeterminado = Y_GROUND)
         param_names: lista de cadenas, opcional
             Nombres de los parámetros de la superficie, por ejemplo, reflectividad, incidente total.
             irradiancia, temperatura, etc. (Predeterminado = [])

         Devoluciones
         -------
         Objeto PVGround
         """
        param_names = param_names or []
        # Get ground boundaries
        if x_min_max is None:
            x_min, x_max = MIN_X_GROUND, MAX_X_GROUND
        else:
            x_min, x_max = x_min_max
        full_extent_coords = [(x_min, y_ground), (x_max, y_ground)]

        # Create the shade collections
        shaded_collection = ShadeCollection(
            list_surfaces=list_shaded_surfaces, shaded=True,
            param_names=param_names)
        illum_collection = ShadeCollection(
            list_surfaces=list_illum_surfaces, shaded=False,
            param_names=param_names)

        # Create the ground segment
        segment = PVSegment(illum_collection=illum_collection,
                            shaded_collection=shaded_collection)

        return cls(list_segments=[segment],
                   original_linestring=LineString(full_extent_coords))

    @property
    def boundary(self):
        """Límites de la línea lineal original del terreno."""
        return self.original_linestring.boundary