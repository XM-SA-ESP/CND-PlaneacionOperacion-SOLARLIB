import numpy as np
from xm_solarlib.pvfactors.config import COLOR_DIC
from xm_solarlib.pvfactors.geometry.base import PVSegment, BaseSide, _coords_from_center_tilt_length
from xm_solarlib.pvfactors.geometry.timeseries import TsShadeCollection, TsLineCoords, TsSurface
from xm_solarlib.tools import cosd, sind
from shapely.geometry import LineString, GeometryCollection


class TsSegment(object):
    """Un segmento es un segmento de serie temporal que tiene una colección 
    sombreada de serie temporal y una colección iluminada de serie temporal."""

    def __init__(self, coords, illum_collection, shaded_collection,
                 index=None, n_vector=None):
        """Inicializar segmento de serie temporal usando coordenadas de segmento y
         Superficies iluminadas y sombreadas en series temporales.

         Parámetros
         ----------
         coordenadas: :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas de serie temporal del segmento completo.
         colección_illum: \
         :py:clase:`~pvfactors.geometry.timeseries.TsShadeCollection`
             Colección de series temporales para la parte iluminada del segmento.
         colección_sombreada: \
         :py:clase:`~pvfactors.geometry.timeseries.TsShadeCollection`
             Colección de series temporales para la parte sombreada del segmento
         índice: int, opcional
             Índice de segmento (Predeterminado = Ninguno)
         n_vector: np.ndarray, opcional
             Vectores normales de serie temporal del lado (Predeterminado = Ninguno)
         """
        self.coords = coords
        self.illum = illum_collection
        self.shaded = shaded_collection
        self.index = index
        self.n_vector = n_vector

    def surfaces_at_idx(self, idx):
        """Obtener todas las geometrías de superficie fotovoltaica en un segmento de serie temporal para un determinado
         índice.

         Parámetros
         ----------
         idx:int
             Índice a utilizar para generar geometrías de superficie fotovoltaica

         Devoluciones
         -------
         lista de objetos :py:class:`~pvfactors.geometry.base.PVSurface`
             Lista de superficies fotovoltaicas
         """
        segment = self.at(idx)
        return segment.all_surfaces

    def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                    color_illum=COLOR_DIC['pvrow_illum']):
        """Trazar un segmento de serie temporal en un índice determinado.

         Parámetros
         ----------
         idx:int
             Índice que se utilizará para trazar el segmento de la serie temporal
         hacha: :py:clase:`matplotlib.pyplot.axes` objeto
             Ejes para trazar
         color_shaded: str, opcional
             Color a utilizar para trazar las superficies sombreadas (Predeterminado =
             COLOR_DIC['pvrow_shaded'])
         color_shaded: str, opcional
             Color a utilizar para trazar las superficies iluminadas (predeterminado =
             COLOR_DIC['pvrow_illum'])
         """
        segment = self.at(idx)
        segment.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
                     with_index=False)

    def at(self, idx):
        """Genere una geometría de segmento PV para el índice deseado.

         Parámetros
         ----------
         idx:int
             Índice que se utilizará para generar la geometría del segmento fotovoltaico

         Devoluciones
         -------
         segmento: :py:clase:`~pvfactors.geometry.base.PVSegment`
         """
        # Create illum collection
        illum_collection = self.illum.at(idx)
        # Create shaded collection
        shaded_collection = self.shaded.at(idx)
        # Create PV segment
        segment = PVSegment(illum_collection=illum_collection,
                            shaded_collection=shaded_collection,
                            index=self.index)
        return segment

    @property
    def length(self):
        """Longitud del segmento de la serie temporal."""
        return self.illum.length + self.shaded.length

    @property
    def shaded_length(self):
        """Longitud de la serie temporal de la parte sombreada del segmento."""
        return self.shaded.length

    @property
    def centroid(self):
        """Coordenadas de puntos de serie temporal del centroide del segmento"""
        return self.coords.centroid

    def get_param_weighted(self, param):
        """Obtener el parámetro de serie temporal para el segmento, después de ponderar por
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
        """Obtener parámetros de series temporales de las superficies del segmento con peso,
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
        return self.illum.get_param_ww(param) + self.shaded.get_param_ww(param)

    def update_params(self, new_dict):
        """Actualizar parámetros de superficie de serie temporal del segmento.

         Parámetros
         ----------
         new_dict: dict
             Parámetros para agregar o actualizar para las superficies.
         """
        self.illum.update_params(new_dict)
        self.shaded.update_params(new_dict)

    @property
    def highest_point(self):
        """Coordenadas del punto de serie temporal del punto más alto del segmento"""
        return self.coords.highest_point

    @property
    def lowest_point(self):
        """Coordenadas del punto de serie temporal del punto más bajo del segmento"""
        return self.coords.lowest_point

    @property
    def all_ts_surfaces(self):
        """Lista de todas las superficies de series temporales en el segmento"""
        return self.illum.list_ts_surfaces + self.shaded.list_ts_surfaces

    @property
    def n_ts_surfaces(self):
        """Número de superficies de series temporales en el segmento"""
        return self.illum.n_ts_surfaces + self.shaded.n_ts_surfaces


class TsSide(object):
    """Clase secundaria de serie temporal: esta clase es una versión vectorizada de la
     Geometrías del lado base. Las coordenadas y atributos (lista de segmentos,
     vector normal) están todos vectorizados."""

    def __init__(self, segments, n_vector=None):
        """Inicializar el lado de la serie temporal utilizando la lista de segmentos de la serie temporal.

         Parámetros
         ----------
         segmentos: lista de :py:class:`~pvfactors.geometry.pvrow.TsSegment`
             Lista de segmentos de series temporales del lado
         n_vector: np.ndarray, opcional
             Vectores normales de serie temporal del lado (Predeterminado = Ninguno)
         """
        self.list_segments = segments
        self.n_vector = n_vector

    @classmethod
    def from_raw_inputs(cls, xy_center, width, rotation_vec, cut,
                        shaded_length, n_vector=None, param_names=None):
        """Cree un lado de la serie temporal utilizando entradas de filas PV sin procesar.
         Nota: el sombreado siempre será cero cuando las filas de PV sean planas.

         Parámetros
         ----------
         xy_center: tupla de flotador
             Coordenadas x e y del punto central de la fila PV (invariante)
         width: flotador
             ancho de las filas fotovoltaicas [m]
         rotation_vec: np.ndarray
             Valores de rotación de series temporales de la fila PV [grados]
         cut: int
             Esquema de discretización del lado fotovoltaico.
             Creará segmentos de igual longitud.
         shaded_length: np.ndarray
             Valores de series temporales de longitud del lado sombreado desde el punto más bajo [m]
         n_vector: np.ndarray, opcional
             Vectores normales de series temporales del lado.
         param_names: lista de cadenas, opcional
             Lista de nombres de parámetros de superficie para usar al crear geometrías
             (Predeterminado = Ninguno)

         Devoluciones
         -------
         Nuevo objeto lateral de serie temporal
         """

        mask_tilted_to_left = rotation_vec >= 0

        # Create Ts segments
        x_center, y_center = xy_center
        radius = width / 2.
        segment_length = width / cut
        is_not_flat = rotation_vec != 0.

        # Calculate coords of shading point
        r_shade = radius - shaded_length
        x_sh = np.where(
            mask_tilted_to_left,
            r_shade * cosd(rotation_vec + 180.) + x_center,
            r_shade * cosd(rotation_vec) + x_center)
        y_sh = np.where(
            mask_tilted_to_left,
            r_shade * sind(rotation_vec + 180.) + y_center,
            r_shade * sind(rotation_vec) + y_center)

        # Calculate coords
        list_segments = []
        for i in range(cut):
            # Calculate segment coords
            r1 = radius - i * segment_length
            r2 = radius - (i + 1) * segment_length
            x1 = r1 * cosd(rotation_vec + 180.) + x_center
            y1 = r1 * sind(rotation_vec + 180.) + y_center
            x2 = r2 * cosd(rotation_vec + 180) + x_center
            y2 = r2 * sind(rotation_vec + 180) + y_center
            segment_coords = TsLineCoords.from_array(
                np.array([[x1, y1], [x2, y2]]))
            # Determine lowest and highest points of segment
            x_highest = np.where(mask_tilted_to_left, x2, x1)
            y_highest = np.where(mask_tilted_to_left, y2, y1)
            x_lowest = np.where(mask_tilted_to_left, x1, x2)
            y_lowest = np.where(mask_tilted_to_left, y1, y2)
            # Calculate illum and shaded coords
            x2_illum, y2_illum = x_highest, y_highest
            x1_shaded, y1_shaded, x2_shaded, y2_shaded = \
                x_lowest, y_lowest, x_lowest, y_lowest
            mask_all_shaded = (y_sh > y_highest) & (is_not_flat)
            mask_partial_shaded = (y_sh > y_lowest) & (~ mask_all_shaded) \
                & (is_not_flat)
            # Calculate second boundary point of shade
            x2_shaded = np.where(mask_all_shaded, x_highest, x2_shaded)
            x2_shaded = np.where(mask_partial_shaded, x_sh, x2_shaded)
            y2_shaded = np.where(mask_all_shaded, y_highest, y2_shaded)
            y2_shaded = np.where(mask_partial_shaded, y_sh, y2_shaded)
            x1_illum = x2_shaded
            y1_illum = y2_shaded
            illum_coords = TsLineCoords.from_array(
                np.array([[x1_illum, y1_illum], [x2_illum, y2_illum]]))
            shaded_coords = TsLineCoords.from_array(
                np.array([[x1_shaded, y1_shaded], [x2_shaded, y2_shaded]]))
            # Create illuminated and shaded collections
            is_shaded = False
            illum = TsShadeCollection(
                [TsSurface(illum_coords, n_vector=n_vector,
                           param_names=param_names, shaded=is_shaded)],
                is_shaded)
            is_shaded = True
            shaded = TsShadeCollection(
                [TsSurface(shaded_coords, n_vector=n_vector,
                           param_names=param_names, shaded=is_shaded)],
                is_shaded)
            # Create segment
            segment = TsSegment(segment_coords, illum, shaded,
                                n_vector=n_vector, index=i)
            list_segments.append(segment)

        return cls(list_segments, n_vector=n_vector)

    def surfaces_at_idx(self, idx):
        """Obtener todas las geometrías de superficie fotovoltaica en el lado de la serie temporal para un determinado
         índice.

         Parámetros
         ----------
         idx:int
             Índice a utilizar para generar geometrías de superficie fotovoltaica

         Devoluciones
         -------
         lista de objetos :py:class:`~pvfactors.geometry.base.PVSurface`
             Lista de superficies fotovoltaicas
         """
        side_geom = self.at(idx)
        return side_geom.all_surfaces

    def at(self, idx):
        """Generar una geometría lateral para el índice deseado.

         Parámetros
         ----------
         idx:int
             Índice a utilizar para generar geometría lateral

         Devoluciones
         -------
         lado: :py:clase:`~pvfactors.geometry.base.BaseSide`
         """
        list_geom_segments = []
        for ts_seg in self.list_segments:
            list_geom_segments.append(ts_seg.at(idx))
        side = BaseSide(list_geom_segments)
        return side

    def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                    color_illum=COLOR_DIC['pvrow_illum']):
        """Trazar el lado de la serie temporal en un índice determinado.

         Parámetros
         ----------
         idx:int
             Índice que se utilizará para trazar el lado de la serie temporal
         hacha: :py:clase:`matplotlib.pyplot.axes` objeto
             Ejes para trazar
         color_shaded: str, opcional
             Color a utilizar para trazar las superficies sombreadas (Predeterminado =
             COLOR_DIC['pvrow_shaded'])
         color_shaded: str, opcional
             Color a utilizar para trazar las superficies iluminadas (predeterminado =
             COLOR_DIC['pvrow_illum'])
         """
        side_geom = self.at(idx)
        side_geom.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
                       with_index=False)

    @property
    def shaded_length(self):
        """Serie temporal longitud sombreada del lado."""
        length = 0.
        for seg in self.list_segments:
            length += seg.shaded.length
        return length

    @property
    def length(self):
        """Longitud del lado de la serie temporal."""
        length = 0.
        for seg in self.list_segments:
            length += seg.length
        return length

    def get_param_weighted(self, param):
        """Obtener el parámetro de serie temporal para el lado, después de ponderar por
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
        """Obtener parámetros de series temporales de las superficies laterales con peso, es decir,
         después de multiplicar por las longitudes de las superficies.

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
        for seg in self.list_segments:
            value += seg.get_param_ww(param)
        return value

    def update_params(self, new_dict):
        """Update timeseries surface parameters of the side.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surfaces
        """
        for seg in self.list_segments:
            seg.update_params(new_dict)

    @property
    def n_ts_surfaces(self):
        """Número de superficies de series temporales en el lado ts"""
        n_ts_surfaces = 0
        for ts_segment in self.list_segments:
            n_ts_surfaces += ts_segment.n_ts_surfaces
        return n_ts_surfaces

    @property
    def all_ts_surfaces(self):
        """Lista de todas las superficies de la serie temporal"""
        all_ts_surfaces = []
        for ts_segment in self.list_segments:
            all_ts_surfaces += ts_segment.all_ts_surfaces
        return all_ts_surfaces

class TsPVRow(object):
    """Clase de fila PV de serie temporal: esta clase es una versión vectorizada de la
     Geometrías de filas fotovoltaicas. Las coordenadas y atributos (anverso y reverso)
     están todos vectorizados."""

    def __init__(self, ts_front_side, ts_back_side, xy_center, index=None,
                 full_pvrow_coords=None):
        """Inicialice la fila PV de la serie temporal con sus lados frontal y posterior.

         Parámetros
         ----------
         ts_front_side: :py:clase:`~pvfactors.geometry.pvrow.TsSide`
             Parte frontal de la serie temporal de la fila fotovoltaica
         ts_back_side: :py:clase:`~pvfactors.geometry.pvrow.TsSide`
             Parte posterior de la serie temporal de la fila PV
         xy_center: tupla de flotador
             Coordenadas x e y del punto central de la fila PV (invariante)
         índice: int, opcional
             índice de la fila PV (Predeterminado = Ninguno)
         full_pvrow_coords:\
         :py:clase:`~pvfactors.geometry.timeseries.TsLineCoords`, opcional
             Coordenadas de la serie temporal de la fila PV completa, de extremo a extremo
             (Predeterminado = Ninguno)
         """
        self.front = ts_front_side
        self.back = ts_back_side
        self.xy_center = xy_center
        self.index = index
        self.full_pvrow_coords = full_pvrow_coords

    @classmethod
    def from_raw_inputs(cls, xy_center, width, rotation_vec,
                        cut, shaded_length_front, shaded_length_back,
                        index=None, param_names=None):
        """Cree una fila de PV de serie temporal utilizando entradas sin procesar.
         Nota: el sombreado siempre será cero cuando las filas pv sean planas.

         Parámetros
         ----------
         xy_center: tupla de flotador
             Coordenadas x e y del punto central de la fila PV (invariante)
         width: flotador
             ancho de las filas fotovoltaicas [m]
         rotation_vec: np.ndarray
             Valores de rotación de series temporales de la fila PV [grados]
         cut: dictar
             Esquema de discretización de la fila PV. Por ejemplo, {'frente': 2, 'atrás': 4}.
             Creará segmentos de igual longitud en los lados designados.
         shaded_length_front: np.ndarray
             Valores de series temporales de longitud sombreada del lado frontal [m]
         shaded_length_back: np.ndarray
             Valores de series temporales de longitud sombreada del reverso [m]
         index: int, opcional
             Índice de la fila pv (predeterminado = Ninguno)
         param_names: lista de cadenas, opcional
             Lista de nombres de parámetros de superficie para usar al crear geometrías
             (Predeterminado = Ninguno)

         Devoluciones
         -------
         Nuevo objeto de fila PV de serie temporal
         """
        # Calculate full pvrow coords
        pvrow_coords = TsPVRow._calculate_full_coords(
            xy_center, width, rotation_vec)
        # Calculate normal vectors
        dx = pvrow_coords.b2.x - pvrow_coords.b1.x
        dy = pvrow_coords.b2.y - pvrow_coords.b1.y
        normal_vec_front = np.array([-dy, dx])
        # Calculate front side coords
        ts_front = TsSide.from_raw_inputs(
            xy_center, width, rotation_vec, cut.get('front', 1),
            shaded_length_front, n_vector=normal_vec_front,
            param_names=param_names)
        # Calculate back side coords
        ts_back = TsSide.from_raw_inputs(
            xy_center, width, rotation_vec, cut.get('back', 1),
            shaded_length_back, n_vector=-normal_vec_front,
            param_names=param_names)

        return cls(ts_front, ts_back, xy_center, index=index,
                   full_pvrow_coords=pvrow_coords)

    @staticmethod
    def _calculate_full_coords(xy_center, width, rotation):
        """Método para calcular las coordenadas completas de la fila PV.

         Parámetros
         ----------
         xy_center: tupla de flotador
             Coordenadas x e y del punto central de la fila PV (invariante)
         width: flotador
             ancho de las filas fotovoltaicas [m]
         rotation: np.ndarray
             Valores de rotación de series temporales de la fila PV [grados]

         Devoluciones
         -------
         coordenadas: :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas de serie temporal de la fila PV completa
         """
        x_center, y_center = xy_center
        radius = width / 2.
        # Calculate coords
        x1 = radius * cosd(rotation + 180.) + x_center
        y1 = radius * sind(rotation + 180.) + y_center
        x2 = radius * cosd(rotation) + x_center
        y2 = radius * sind(rotation) + y_center
        coords = TsLineCoords.from_array(np.array([[x1, y1], [x2, y2]]))
        return coords

    def surfaces_at_idx(self, idx):
        """Obtenga todas las geometrías de superficie fotovoltaica en la fila fotovoltaica de la serie temporal para un determinado
         índice.

         Parámetros
         ----------
         idx:int
             Índice a utilizar para generar geometrías de superficie fotovoltaica

         Devoluciones
         -------
         lista de objetos :py:class:`~pvfactors.geometry.base.PVSurface`
             Lista de superficies fotovoltaicas
         """
        pvrow = self.at(idx)
        return pvrow.all_surfaces

    def plot_at_idx(self, idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                    color_illum=COLOR_DIC['pvrow_illum'],
                    with_surface_index=False):
        """Trazar la fila PV de la serie temporal en un índice determinado.

         Parámetros
         ----------
         idx:int
             Índice que se utilizará para trazar filas de PV de series temporales
         ax: :py:clase:`matplotlib.pyplot.axes` objeto
             Ejes para trazar
         color_shaded: str, opcional
             Color a utilizar para trazar las superficies sombreadas (Predeterminado =
             COLOR_DIC['pvrow_shaded'])
         color_shaded: str, opcional
             Color a utilizar para trazar las superficies iluminadas (predeterminado =
             COLOR_DIC['pvrow_illum'])
         with_surface_index: bool, opcional
             Trazar las superficies con sus valores de índice (Predeterminado = Falso)
         """
        pvrow = self.at(idx)
        pvrow.plot(ax, color_shaded=color_shaded,
                   color_illum=color_illum, with_index=with_surface_index)

    def at(self, idx):
        """Genere una geometría de fila PV para el índice deseado.

         Parámetros
         ----------
         idx:int
             Índice que se utilizará para generar la geometría de fila PV

         Devoluciones
         -------
         pvrow : :py:clase:`~pvfactors.geometry.pvrow.PVRow`
         """
        front_geom = self.front.at(idx)
        back_geom = self.back.at(idx)
        original_line = LineString(
            self.full_pvrow_coords.as_array[:, :, idx])
        pvrow = PVRow(front_side=front_geom, back_side=back_geom,
                      index=self.index, original_linestring=original_line)
        return pvrow

    def update_params(self, new_dict):
        """Actualizar los parámetros de superficie de la serie temporal de la fila PV.

         Parámetros
         ----------
         new_dict: dict
             Parámetros para agregar o actualizar para las superficies.
         """
        self.front.update_params(new_dict)
        self.back.update_params(new_dict)

    @property
    def n_ts_surfaces(self):
        """Número de superficies de series temporales en la fila ts PV"""
        return self.front.n_ts_surfaces + self.back.n_ts_surfaces

    @property
    def all_ts_surfaces(self):
        """Lista de todas las superficies de la serie temporal"""
        return self.front.all_ts_surfaces + self.back.all_ts_surfaces

    @property
    def centroid(self):
        """Punto centroide de la fila pv de la serie temporal"""
        centroid = (self.full_pvrow_coords.centroid
                    if self.full_pvrow_coords is not None else None)
        return centroid

    @property
    def length(self):
        """Longitud de ambos lados de la fila PV de la serie temporal"""
        return self.front.length + self.back.length

    @property
    def highest_point(self):
        """Coordenadas del punto de serie temporal del punto más alto de la fila PV"""
        high_pt = (self.full_pvrow_coords.highest_point
                   if self.full_pvrow_coords is not None else None)
        return high_pt
    

class PVRowSide(BaseSide):
    """Un lado de una fila de PV representa toda la superficie de un lado de una fila de PV.
     En esencia contendrá un número fijo de
     :py:class:`~pvfactors.geometry.base.PVSegment` objetos que se unirán
     constituyen un lado de una fila PV: un lado de la fila PV también puede ser
     "discretizado" en múltiples segmentos"""

    def __init__(self, list_segments=[]):
        """Inicializar PVRowSide usando su clase base
         :py:clase:`pvfactors.geometry.base.BaseSide`

         Parámetros
         ----------
         list_segments: lista de :py:class:`~pvfactors.geometry.base.PVSegment`
             Lista de segmentos fotovoltaicos para el lado de la fila fotovoltaica.
         """
        super(PVRowSide, self).__init__(list_segments)


class PVRow(GeometryCollection):
    """Una fila PV está formada por dos lados de la fila PV, uno frontal y otro posterior."""

    def __init__(self, front_side=PVRowSide(), back_side=PVRowSide(),
                 index=None, original_linestring=None):
        """Inicializar fila PV.

         Parámetros
         ----------
         front_side: :py:class:`~pvfactors.geometry.pvrow.PVRowSide`, opcional
             Lado frontal de la fila PV (predeterminado = lado de la fila PV vacío)
         back_side: :py:class:`~pvfactors.geometry.pvrow.PVRowSide`, opcional
             Parte posterior de la fila PV (predeterminado = lado de la fila PV vacío)
         índice: int, opcional
             Índice de fila PV (predeterminado = Ninguno)
         original_linestring: :py:clase:`shapely.geometry.LineString`, opcional
             Cadena lineal continua completa de la que estará hecha la fila PV
             (Predeterminado = Ninguno)

         """
        self.front = front_side
        self.back = back_side
        self.index = index
        self.original_linestring = original_linestring
        self._all_surfaces = None
        super(PVRow, self).__init__([self.front, self.back])

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None, cut={}, param_names=[]):
        """Crear una fila PV con una sola superficie PV y usando una cadena lineal
         coordenadas.

         Parámetros
         ----------
         coords: lista
             Lista de coordenadas de cadena lineal para la superficie
         shaded: bool, opcional
             Estado de sombreado deseado para los lados de PVRow (Predeterminado = Falso)
         normal_vector: lista, opcional
             Vector normal para la superficie (Predeterminado = Ninguno)
         index: int, opcional
             Índice de fila PV (predeterminado = Ninguno)
         cut: dict, opcional
             Esquema para decidir cuántos segmentos crear en cada lado.
             Por ejemplo, {'front': 3, 'back': 2} conducirá a 3 segmentos en el frente
             y 2 segmentos en la parte posterior. (Predeterminado = {})
         param_names: lista de cadenas, opcional
             Nombres de los parámetros de la superficie, por ejemplo, reflectividad, incidente total.
             irradiancia, temperatura, etc. (Predeterminado = [])

         Devoluciones
         -------
         :py:clase:`~pvfactors.geometry.pvrow.PVRow` objeto
         """
        index_single_segment = 0
        front_side = PVRowSide.from_linestring_coords(
            coords, shaded=shaded, normal_vector=normal_vector,
            index=index_single_segment, n_segments=cut.get('front', 1),
            param_names=param_names)
        if normal_vector is not None:
            back_n_vec = - np.array(normal_vector)
        else:
            back_n_vec = - front_side.n_vector
        back_side = PVRowSide.from_linestring_coords(
            coords, shaded=shaded, normal_vector=back_n_vec,
            index=index_single_segment, n_segments=cut.get('back', 1),
            param_names=param_names)
        return cls(front_side=front_side, back_side=back_side, index=index,
                   original_linestring=LineString(coords))

    @classmethod
    def from_center_tilt_width(cls, xy_center, tilt, width, surface_azimuth,
                               axis_azimuth, shaded=False, normal_vector=None,
                               index=None, cut={}, param_names=[]):
        """Cree una fila PV utilizando principalmente las coordenadas del centro de la línea,
         un ángulo de inclinación y su longitud.

         Parámetros
         ----------
         xy_center: tupla
             Coordenadas x, y del punto central de la cadena lineal deseada
         inclinación: flotar
             ángulo de inclinación de la superficie deseado [grados]
         longitud: flotador
             longitud deseada de la cadena lineal [m]
         superficie_azimut: flotar
             Azimut superficial de la superficie fotovoltaica [grados]
         eje_azimut: flotante
             Azimut del eje de la superficie fotovoltaica, es decir, dirección del eje de rotación
             [grados]
         sombreado: bool, opcional
             Estado de sombreado deseado para los lados de PVRow (Predeterminado = Falso)
         normal_vector: lista, opcional
             Vector normal para la superficie (Predeterminado = Ninguno)
         índice: int, opcional
             Índice de fila PV (predeterminado = Ninguno)
         cortar: dict, opcional
             Esquema para decidir cuántos segmentos crear en cada lado.
             Por ejemplo, {'front': 3, 'back': 2} conducirá a 3 segmentos en el frente
             y 2 segmentos en la parte posterior. (Predeterminado = {})
         param_names: lista de cadenas, opcional
             Nombres de los parámetros de la superficie, por ejemplo, reflectividad, incidente total.
             irradiancia, temperatura, etc. (Predeterminado = [])

         Devoluciones
         -------
         :py:clase:`~pvfactors.geometry.pvrow.PVRow` objeto
         """
        coords = _coords_from_center_tilt_length(xy_center, tilt, width,
                                                 surface_azimuth, axis_azimuth)
        return cls.from_linestring_coords(coords, shaded=shaded,
                                          normal_vector=normal_vector,
                                          index=index, cut=cut,
                                          param_names=param_names)

    def plot(self, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
             color_illum=COLOR_DIC['pvrow_illum'], with_index=False):
        """Traza las superficies de la fila PV.

         Parámetros
         ----------
         ax: :py:clase:`matplotlib.pyplot.axes` objeto
             Ejes para trazar
         color_shaded: str, opcional
             Color a utilizar para trazar las superficies sombreadas (Predeterminado =
             COLOR_DIC['pvrow_shaded'])
         color_shaded: str, opcional
             Color a utilizar para trazar las superficies iluminadas (predeterminado =
             COLOR_DIC['pvrow_illum'])
         with_index: bool
             Bandera para anotar superficies con sus índices (Predeterminado = Falso)

         """
        self.front.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
                        with_index=with_index)
        self.back.plot(ax, color_shaded=color_shaded, color_illum=color_illum,
                       with_index=with_index)

    @property
    def boundary(self):
        """Límites de la cadena lineal original de la fila PV."""
        return self.original_linestring.boundary

    @property
    def highest_point(self):
        """Punto más alto de la fila PV."""
        b1, b2 = self.boundary
        highest_point = b1 if b1.y > b2.y else b2
        return highest_point

    @property
    def lowest_point(self):
        """Punto más bajo de la fila PV."""
        b1, b2 = self.boundary
        lowest_point = b1 if b1.y < b2.y else b2
        return lowest_point

    @property
    def all_surfaces(self):
        """List of all the surfaces in the PV row."""
        if self._all_surfaces is None:
            self._all_surfaces = []
            self._all_surfaces += self.front.all_surfaces
            self._all_surfaces += self.back.all_surfaces
        return self._all_surfaces

    @property
    def surface_indices(self):
        """Lista de todos los índices de superficie en la fila PV."""
        list_indices = []
        list_indices += self.front.surface_indices
        list_indices += self.back.surface_indices
        return list_indices

    def update_params(self, new_dict):
        """Actualice los parámetros de superficie tanto para el frente como para el reverso.

         Parámetros
         ----------
         new_dict: dict
             Parámetros para agregar o actualizar para la superficie
         """
        self.front.update_params(new_dict)
        self.back.update_params(new_dict)