
import numpy as np
from xm_solarlib.tools import cosd, sind
from xm_solarlib.pvfactors.config import COLOR_DIC, PLOT_FONTSIZE, DEFAULT_NORMAL_VEC, ALPHA_TEXT,\
      MAX_X_GROUND, DISTANCE_TOLERANCE
from xm_solarlib.pvfactors.geometry.utils import check_collinear, difference, is_collinear, contains, are_2d_vecs_collinear
from xm_solarlib.pvfactors.geometry.plot import plot_coords, plot_bounds, plot_line
from shapely.geometry import GeometryCollection, LineString
from shapely.geometry.collection import geos_geometrycollection_from_py
from shapely.ops import linemerge


def _check_uniform_shading(list_elements):
    """Comprueba que todos los :py:class:`~pvfactors.geometry.base.PVSurface` objetos en
     la lista tiene un sombreado uniforme

    Parametros
    ----------
    list_elements : lista de :py:class:`~pvfactors.geometry.base.PVSurface`

    Excepciones
    ------
    Exception
        si todos los elementos no tienen la misma bandera de sombreado
    """
    shaded = None
    for element in list_elements:
        if shaded is None:
            shaded = element.shaded
        else:
            is_uniform = shaded == element.shaded
            if not is_uniform:
                msg = "All elements should have same shading"
                raise ValueError(msg)


class BaseSide(GeometryCollection):
    """Un lado representa una colección fija de objetos de segmentos fotovoltaicos que deben
     todos serán colineales, con el mismo vector normal"""

    def __init__(self, list_segments=None):
        """Crea una geometría lateral.

        Parametros
        ----------
        list_segments: lista de :py:class:`~pvfactors.geometry.base.PVSegment`, opcional
             Lista de segmentos fotovoltaicos para el lado (Predeterminado = Ninguno)
        """
        list_segments = [] if list_segments is None else list_segments
        check_collinear(list_segments)
        self.list_segments = tuple(list_segments)
        self._all_surfaces = None
        super(BaseSide, self).__init__(list_segments)

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None, n_segments=1, param_names=None):
        """Crear un lado con una sola superficie fotovoltaica o múltiples superficies discretizadas.
         los identicos.

         Parámetros
         ----------
         coordenadas: lista
             Lista de coordenadas de cadena lineal para la superficie
         sombreado: bool, opcional
             Estado de sombreado deseado para la superficie fotovoltaica resultante
             (Predeterminado = Falso)
         normal_vector: lista, opcional
             Vector normal para la superficie (Predeterminado = Ninguno)
         índice: int, opcional
             Índice de los segmentos (Predeterminado = Ninguno)
         n_segmentos: int, opcional
             Número de segmentos de la misma longitud a usar (predeterminado = 1)
         param_names: lista de cadenas, opcional
             Nombres de los parámetros de la superficie, por ejemplo, reflectividad, incidente total.
             irradiancia, temperatura, etc. (Predeterminado = Ninguno)
         """
        if n_segments == 1:
            list_pvsegments = [PVSegment.from_linestring_coords(
                coords, shaded=shaded, normal_vector=normal_vector,
                index=index, param_names=param_names)]
        else:
            # Discretize coords and create segments accordingly
            linestring = LineString(coords)
            fractions = np.linspace(0., 1., num=n_segments + 1)
            list_points = [linestring.interpolate(fraction, normalized=True)
                           for fraction in fractions]
            list_pvsegments = []
            for idx in range(n_segments):
                new_coords = list_points[idx:idx + 2]
                pvsegment = PVSegment.from_linestring_coords(
                    new_coords, shaded=shaded, normal_vector=normal_vector,
                    index=index, param_names=param_names)
                list_pvsegments.append(pvsegment)
        return cls(list_segments=list_pvsegments)

    @property
    def n_vector(self):
        """Vector normal del lado."""
        if len(self.list_segments):
            return self.list_segments[0].n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @property
    def shaded_length(self):
        """Longitud sombreada del costado."""
        shaded_length = 0.
        for segment in self.list_segments:
            shaded_length += segment.shaded_length
        return shaded_length

    @property
    def n_surfaces(self):
        """Número de superficies en el objeto lateral."""
        n_surfaces = 0
        for segment in self.list_segments:
            n_surfaces += segment.n_surfaces
        return n_surfaces

    @property
    def all_surfaces(self):
        """Lista de todas las superficies del objeto Lado."""
        if self._all_surfaces is None:
            self._all_surfaces = []
            for segment in self.list_segments:
                self._all_surfaces += segment.all_surfaces
        return self._all_surfaces

    @property
    def surface_indices(self):
        """Lista de todos los índices de superficie en el objeto Lado."""
        list_indices = []
        for seg in self.list_segments:
            list_indices += seg.surface_indices
        return list_indices

    def plot(self, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
             color_illum=COLOR_DIC['pvrow_illum'], with_index=False):
        """Traza las superficies en el objeto lateral.

         Parámetros
         ----------
         hacha: :py:clase:`matplotlib.pyplot.axes` objeto
             Ejes para trazar
         color_shaded: str, opcional
             Color a utilizar para trazar las superficies sombreadas (Predeterminado =
             COLOR_DIC['pvrow_shaded'])
         color_illum: str, opcional
             Color a utilizar para trazar las superficies iluminadas (predeterminado =
             COLOR_DIC['pvrow_illum'])
         with_index: bool
             Bandera para anotar superficies con sus índices (Predeterminado = Falso)
         """
        for segment in self.list_segments:
            segment.plot(ax, color_shaded=color_shaded,
                         color_illum=color_illum, with_index=with_index)

    def cast_shadow(self, linestring):
        """Proyectar sombra en el costado usando una cadena lineal: reorganizará el
         Superficies fotovoltaicas entre las colecciones sombreadas e iluminadas del
         segmentos.

         Parámetros
         ----------
         linestring: :py:clase:`shapely.geometry.LineString`
             Cadena lineal que proyecta una sombra sobre el objeto lateral
         """
        for segment in self.list_segments:
            segment.cast_shadow(linestring)

    def merge_shaded_areas(self):
        """Fusionar áreas sombreadas de todos los segmentos fotovoltaicos"""
        for seg in self.list_segments:
            seg._shaded_collection.merge_surfaces()

    def cut_at_point(self, point):
        """Corte el lado en el punto si el lado lo contiene.

         Parámetros
         ----------
         point : :py:clase:`shapely.geometry.Point`
             Punto donde cortar la geometría lateral, si esta última contiene el
             anterior
         """
        if contains(self, point):
            for segment in self.list_segments:
                # Nothing will happen to the segments that do not contain
                # the point
                segment.cut_at_point(point)

    def get_param_weighted(self, param):
        """Obtener el parámetro de las superficies laterales, después de ponderar
         por longitud de superficie.

         Parámetros
         ----------
         param: str
             Parámetro de superficie a devolver

         Devoluciones
         -------
         float
             Valor de parámetro ponderado
         """
        value = self.get_param_ww(param) / self.length
        return value

    def get_param_ww(self, param):
        """Obtiene el parámetro de las superficies laterales con peso, es decir,
         después de multiplicar por las longitudes de las superficies.

         Parámetros
         ----------
         parámetro: cadena
             Parámetro de superficie a devolver

         Devoluciones
         -------
         float
             Valor del parámetro multiplicado por pesos.

         Excepciones
         ------
         KeyError
             si el nombre del parámetro no está en los parámetros de una superficie
         """
        value = 0
        for seg in self.list_segments:
            value += seg.get_param_ww(param)
        return value

    def update_params(self, new_dict):
        """Actualizar parámetros de superficie en el Lado.

         Parámetros
         ----------
         new_dict: dict
             Parámetros para agregar o actualizar para las superficies.
         """
        for seg in self.list_segments:
            seg.update_params(new_dict)


class BaseSurface(LineString):
    """Las superficies base serán extensiones de las clases :py:class:`LineString`,
     pero agregándole una orientación (vector normal).
     Entonces, dos superficies podrían usar la misma cadena lineal, pero tener direcciones opuestas.
     orientaciones."""

    def __init__(self, coords, normal_vector=None, index=None,
                 param_names=None, params=None):
        """Cree una superficie usando coordenadas de cadena de líneas.
         El vector normal puede tener dos direcciones para una LineString determinada,
         para que el usuario pueda proporcionarlo con el fin de ser específico,
         de lo contrario será automáticamente
         calculado, pero entonces la superficie no sabrá si se suponía que debía ser
         apuntando "arriba" o "abajo". Si la superficie está vacía, el vector normal
         tomará el valor predeterminado.

         Parámetros
         ----------
         coords: lista
             Lista de coordenadas de cadena lineal para la superficie
         normal_vector: lista, opcional
             Vector normal para la superficie (Predeterminado = Ninguno, así será
             calculado)
         index: int, opcional
             Índice de superficie (Predeterminado = Ninguno)
         param_names: lista de cadenas, opcional
             Nombres de los parámetros de la superficie, por ejemplo, reflectividad, incidente total.
             irradiancia, temperatura, etc. (Predeterminado = Ninguno)
         parámetros: dict, opcional
             Parámetros de flotación de superficie (predeterminado = Ninguno)
         """

        param_names = [] if param_names is None else param_names
        super(BaseSurface, self).__init__(coords)
        if normal_vector is None:
            self.n_vector = self._calculate_n_vector()
        else:
            self.n_vector = np.array(normal_vector)
        self.index = index
        self.param_names = param_names
        self.params = params if params is not None \
            else dict.fromkeys(self.param_names)

    def _calculate_n_vector(self):
        """Calcular el vector normal de la superficie, si la superficie no está vacía"""
        if not self.is_empty:
            b1, b2 = self.boundary
            dx = b2.x - b1.x
            dy = b2.y - b1.y
            return np.array([-dy, dx])
        else:
            return DEFAULT_NORMAL_VEC

    def plot(self, ax, color=None, with_index=False):
        """Traza la superficie en los ejes dados.

         Parámetros
         ----------
         ax: :py:clase:`matplotlib.pyplot.axes` objeto
             Ejes para trazar
         color: str, opcional
             Color a utilizar para trazar la superficie (Predeterminado = Ninguno)
         with_index: bool
             Bandera para anotar superficies con sus índices (Predeterminado = Falso)
         """
        plot_coords(ax, self)
        plot_bounds(ax, self)
        plot_line(ax, self, color)
        if with_index:
            # Prepare text location
            v = self.n_vector
            v_norm = v / np.linalg.norm(v)
            centroid = self.centroid
            alpha = ALPHA_TEXT
            x = centroid.x + alpha * v_norm[0]
            y = centroid.y + alpha * v_norm[1]
            # Add text
            if np.abs(x) < MAX_X_GROUND / 2.:
                ax.text(x, y, '{}'.format(self.index),
                        verticalalignment='center',
                        horizontalalignment='center')

    def difference(self, linestring):
        """Calcule la superficie restante después de retirar la pieza que pertenece
         cadena de líneas proporcionada,

         Parámetros
         ----------
         linestring: :py:clase:`shapely.geometry.LineString`
             Hilo de línea para quitar de la superficie

         Devoluciones
         -------
         :py:clase:`shapely.geometry.LineString`
            Diferencia resultante de la superficie actual menos la cadena lineal dada
         """
        return difference(self, linestring)

    def get_param(self, param):
        """Obtener el valor del parámetro de la superficie.

         Parámetros
         ----------
         param: cadena
             Parámetro de superficie a devolver

         Devoluciones
         -------
         Valor del parámetro a devolver

         Exceptions
         ------
         KeyError
             si el nombre del parámetro no está en los parámetros de superficie
         """
        return self.params[param]

    def update_params(self, new_dict):
        """Actualizar parámetros de superficie.

         Parámetros
         ----------
         new_dict: dict
             Parámetros para agregar o actualizar para la superficie
         """
        self.params.update(new_dict)


class PVSurface(BaseSurface):
    """Las superficies fotovoltaicas heredan de
     :py:clase:`~pvfactors.geometry.base.BaseSurface`. La única diferencia es
     que las superficies fotovoltaicas tienen un atributo "sombreado".
     """

    def __init__(self, coords=None, normal_vector=None, shaded=False,
                 index=None, param_names=None, params=None):
        """Inicializar superficie fotovoltaica.

         Parámetros
         ----------
         coords: lista, opcional
             Lista de coordenadas de cadena lineal para la superficie
         normal_vector: lista, opcional
             Vector normal para la superficie (Predeterminado = Ninguno, así será
             calculado)
         shaded: bool, opcional
             Bandera que indica si la superficie está sombreada o no (Predeterminado = Falso)
         index: int, opcional
             Índice de superficie (Predeterminado = Ninguno)
         param_names: lista de cadenas, opcional
             Nombres de los parámetros de la superficie, por ejemplo, reflectividad, incidente total.
             irradiancia, temperatura, etc. (Predeterminado = Ninguno)
         params: dict, opcional
             Parámetros de flotación de superficie (predeterminado = Ninguno)
         """

        param_names = [] if param_names is None else param_names
        super(PVSurface, self).__init__(coords, normal_vector, index=index,
                                        param_names=param_names, params=params)
        self.shaded = shaded


class ShadeCollection(GeometryCollection):
    """Un grupo de :py:class:`~pvfactors.geometry.base.PVSurface`
     objetos que tienen todos el mismo estado de sombreado. Las superficies fotovoltaicas no son
     necesariamente contiguos o colineales."""

    def __init__(self, list_surfaces=None, shaded=None, param_names=None):
        """Inicializar la colección de sombras.

         Parámetros
         ----------
         list_surfaces: lista, opcional
             Lista de objetos :py:class:`~pvfactors.geometry.base.PVSurface`
             (Predeterminado = Ninguno)
         shaded: bool, opcional
             Estado de sombreado de la colección. Si no se especifica, se derivará
             de la lista de superficies (Predeterminado = Ninguno)
         param_names: lista de cadenas, opcional
             Nombres de los parámetros de la superficie, por ejemplo, reflectividad, incidente total.
             irradiancia, temperatura, etc. (Predeterminado = Ninguno)

         """
        list_surfaces = [] if list_surfaces is None else list_surfaces
        param_names = [] if param_names is None else param_names
        _check_uniform_shading(list_surfaces)
        self.list_surfaces = list_surfaces
        self.shaded = self._get_shading(shaded)
        self.is_collinear = is_collinear(list_surfaces)
        self.param_names = param_names
        super(ShadeCollection, self).__init__(list_surfaces)

    def _get_shading(self, shaded):
        """Obtenga el sombreado de la superficie de la lista proporcionada de superficies fotovoltaicas.

         Parámetros
         ----------
         shaded: booleano
             Bandera de sombreado pasada durante la inicialización

         Devoluciones
         -------
         booleano
             Estado de sombreado de la colección.
         """
        if len(self.list_surfaces):
            return self.list_surfaces[0].shaded
        else:
            return shaded

    def plot(self, ax, color=None, with_index=False):
        """Traza las superficies en la colección de sombras.

         Parámetros
         ----------
         ax: :py:clase:`matplotlib.pyplot.axes` objeto
             Ejes para trazar
         color: str, opcional
             Color a utilizar para trazar la superficie (Predeterminado = Ninguno)
         with_index: bool
             Bandera para anotar superficies con sus índices (Predeterminado = Falso)
         """
        for surface in self.list_surfaces:
            surface.plot(ax, color=color, with_index=with_index)

    def add_linestring(self, linestring, normal_vector=None):
        """Agregar superficie fotovoltaica a la colección usando una cadena lineal

         Parámetros
         ----------
        linestring: :py:clase:`shapely.geometry.LineString`
             Cadena de líneas que se utilizará para agregar una superficie fotovoltaica a la colección
         normal_vector: lista, opcional
             Vector normal que se utilizará para crear la superficie fotovoltaica (predeterminado = Ninguno,
             intentaré conseguirlo de la colección)
         """
        if normal_vector is None:
            normal_vector = self.n_vector
        surf = PVSurface(coords=linestring.coords,
                         normal_vector=normal_vector, shaded=self.shaded,
                         param_names=self.param_names)
        self.add_pvsurface(surf)

    def add_pvsurface(self, pvsurface):
        """Agregar superficie fotovoltaica a la colección.

         Parámetros
         ----------
         pvsurface : :py:clase:`~pvfactors.geometry.base.PVSurface`
             Superficie fotovoltaica para añadir a la colección
         """
        self.list_surfaces.append(pvsurface)
        self.is_collinear = is_collinear(self.list_surfaces)
        super(ShadeCollection, self).__init__(self.list_surfaces)

    def remove_linestring(self, linestring):
        """Eliminar la cadena lineal de la colección de sombras.
         El método reorganizará las superficies fotovoltaicas para que funcione.

         Parámetros
         ----------
         linestring: :py:clase:`shapely.geometry.LineString`
             Cadena de línea para eliminar de la colección (mediante diferenciación)
         """
        new_list_surfaces = []
        for surface in self.list_surfaces:
            # Need to use buffer for intersects bc of floating point precision
            # errors in shapely
            if surface.buffer(DISTANCE_TOLERANCE).intersects(linestring):
                difference = surface.difference(linestring)
                # We want to make sure we can iterate on it, as
                # ``difference`` can be a multi-part geometry or not
                if not hasattr(difference, '__iter__'):
                    difference = [difference]
                for new_geom in difference:
                    if not new_geom.is_empty:
                        new_surface = PVSurface(
                            new_geom.coords, normal_vector=surface.n_vector,
                            shaded=surface.shaded,
                            param_names=surface.param_names)
                        new_list_surfaces.append(new_surface)
            else:
                new_list_surfaces.append(surface)

        self.list_surfaces = new_list_surfaces
        # Force update, even if list is empty
        self.update_geom_collection(self.list_surfaces)

    def update_geom_collection(self, list_surfaces):
        """Forzar actualización de la colección de geometría, incluso si la lista está vacía
         https://github.com/Toblerity/Shapely/blob/master/shapely/geometry/collection.py#L42

         Parámetros
         ----------
         list_surfaces: lista de :py:class:`~pvfactors.geometry.base.PVSurface`
             Nueva lista de superficies fotovoltaicas para actualizar la colección de sombras existente
         """
        self._geom, self._ndim = geos_geometrycollection_from_py(list_surfaces)

    def merge_surfaces(self):
        """Fusiona todas las superficies de la colección de sombras en una contigua
         superficie, incluso si no son contiguos, mediante el uso de límites."""
        if len(self.list_surfaces) > 1:
            merged_lines = linemerge(self.list_surfaces)
            minx, miny, maxx, maxy = merged_lines.bounds
            surf_1 = self.list_surfaces[0]
            new_pvsurf = PVSurface(
                coords=[(minx, miny), (maxx, maxy)],
                shaded=self.shaded, normal_vector=surf_1.n_vector,
                param_names=surf_1.param_names)
            self.list_surfaces = [new_pvsurf]
            self.update_geom_collection(self.list_surfaces)

    def cut_at_point(self, point):
        """Corte la colección en el punto si la colección lo contiene.

         Parámetros
         ----------
         point : :py:clase:`shapely.geometry.Point`
             Punto donde cortar la geometría de la colección, si esta última contiene el
             anterior
         """
        for idx, surface in enumerate(self.list_surfaces):
            if contains(surface, point):
                # Make sure that not hitting a boundary
                b1, b2 = surface.boundary
                not_hitting_b1 = b1.distance(point) > DISTANCE_TOLERANCE
                not_hitting_b2 = b2.distance(point) > DISTANCE_TOLERANCE
                if not_hitting_b1 and not_hitting_b2:
                    coords_1 = [b1, point]
                    coords_2 = [point, b2]
                    new_surf_1 = PVSurface(
                        coords_1, normal_vector=surface.n_vector,
                        shaded=surface.shaded,
                        param_names=surface.param_names)
                    new_surf_2 = PVSurface(
                        coords_2, normal_vector=surface.n_vector,
                        shaded=surface.shaded,
                        param_names=surface.param_names)
                    # Now update collection
                    self.list_surfaces[idx] = new_surf_1
                    self.list_surfaces.append(new_surf_2)
                    self.update_geom_collection(self.list_surfaces)
                    # No need to continue the loop
                    break

    def get_param_weighted(self, param):
        """Obtener el parámetro de las superficies de la colección, después de ponderar
         por longitud de superficie.

         Parámetros
         ----------
         parámetro: cadena
             Parámetro de superficie a devolver

         Devoluciones
         -------
         float
             Valor de parámetro ponderado
         """
        value = self.get_param_ww(param) / self.length
        return value

    def get_param_ww(self, param):
        """Obtiene el parámetro de las superficies de la colección con peso, es decir,
         después de multiplicar por las longitudes de las superficies.

         Parámetros
         ----------
         parámetro: cadena
             Parámetro de superficie a devolver

         Devoluciones
         -------
         flotar
             Valor del parámetro multiplicado por pesos.

         aumentos
         ------
         Error de clave
             si el nombre del parámetro no está en los parámetros de una superficie
         """
        value = 0
        for surf in self.list_surfaces:
            value += surf.get_param(param) * surf.length
        return value

    def update_params(self, new_dict):
        """Actualizar parámetros de superficie en la colección.

         Parámetros
         ----------
         new_dict: dict
             Parámetros para agregar o actualizar para la superficie
         """
        for surf in self.list_surfaces:
            surf.update_params(new_dict)

    @property
    def n_vector(self):
        """Vector normal único de la colección de sombras, si existe."""
        if not self.is_collinear:
            msg = "Cannot request n_vector if all elements not collinear"
            raise ValueError(msg)
        if len(self.list_surfaces):
            return self.list_surfaces[0].n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @property
    def n_surfaces(self):
        """Número de superficies en la colección."""
        return len(self.list_surfaces)

    @property
    def surface_indices(self):
        """Índices de las superficies de la colección."""
        return [surf.index for surf in self.list_surfaces]

    @classmethod
    def from_linestring_coords(cls, coords, shaded, normal_vector=None,
                               param_names=None):
        """Crea una colección de sombras con una única superficie fotovoltaica.

         Parámetros
         ----------
         coordenadas: lista
             Lista de coordenadas de cadena lineal para la superficie
         sombreado: booleano
             Estado de sombreado deseado para la colección.
         normal_vector: lista, opcional
             Vector normal para la superficie (Predeterminado = Ninguno)
         param_names: lista de cadenas, opcional
             Nombres de los parámetros de la superficie, por ejemplo, reflectividad, incidente total.
             irradiancia, temperatura, etc. (Predeterminado = Ninguno)
         """
        surf = PVSurface(coords=coords, normal_vector=normal_vector,
                         shaded=shaded, param_names=param_names)
        return cls([surf], shaded=shaded, param_names=param_names)


def _coords_from_center_tilt_length(xy_center, tilt, length,
                                    surface_azimuth, axis_azimuth):
    """Calcular las coordenadas ``shapely`` :py:class:`LineString` desde
     coordenadas centrales, ángulos de superficie y longitud de línea.
     El acimut del eje indica el eje de rotación de las filas (si son de un solo
     seguidores de ejes). En el plano 2D, el eje de rotación será el vector.
     normal a ese plano 2D y entrando en el plano 2D (al trazarlo).
     El azimut de la superficie siempre debe estar a 90 grados del azimut del eje,
     ya sea en dirección positiva o negativa.
     Por ejemplo, un trk de un solo eje con acimut del eje = 0 grados (Norte),
     tener valores de acimut de superficie iguales a 90 grados (este) o 270 grados (oeste).
     Los ángulos de inclinación deben ser siempre positivos. Dado el eje azimut y la superficie.
     azimut, se derivará un ángulo de rotación. Los ángulos de rotación positivos
     indican pvrows que apuntan a la izquierda, y los ángulos de rotación negativos
     indicar pvrows que apuntan a la derecha (sin importar cuál sea el acimut del eje
     es).
     Todas estas convenciones son necesarias para garantizar que, pase lo que pase,
     Los ángulos de inclinación y superficie son, todavía podemos identificar correctamente.
     las mismas filas pv: la fila PV más a la izquierda tendrá el índice 0, y la más a la derecha
     tendrá índice -1.

     Parámetros
     ----------
     xy_center: tupla
         Coordenadas x, y del punto central de la cadena lineal deseada
     inclinación: flotante o np.ndarray
         Ángulos de inclinación de la superficie deseados [grados]. Todos los valores deben ser positivos.
     longitud: flotador
         longitud deseada de la cadena lineal [m]
     Surface_azimuth: flotante o np.ndarray
         Ángulos de acimut de la superficie fotovoltaica [grados]
     eje_azimut: flotante
         Azimut del eje de la superficie fotovoltaica, es decir, dirección del eje de rotación
         [grados]

     Devoluciones
     -------
     list
         Lista de coordenadas de cadena lineal obtenidas a partir de entradas (podrían ser vectores)
         en la forma de [[x1, y1], [x2, y2]], donde xi y yi podrían ser matrices
         o valores escalares.
     """
    # PV row params
    x_center, y_center = xy_center
    radius = length / 2.
    # Get rotation
    rotation = _get_rotation_from_tilt_azimuth(surface_azimuth, axis_azimuth,
                                               tilt)
    # Calculate coords
    x1 = radius * cosd(rotation + 180.) + x_center
    y1 = radius * sind(rotation + 180.) + y_center
    x2 = radius * cosd(rotation) + x_center
    y2 = radius * sind(rotation) + y_center

    return [[x1, y1], [x2, y2]]


def _get_solar_2d_vectors(solar_zenith, solar_azimuth, axis_azimuth):
    """Proyección del vector solar 3d sobre la sección transversal de los sistemas:
     cuál es el plano 2D que estamos considerando.
     Esto es necesario para calcular las sombras.
     Recuerde que el plano 2D es tal que la dirección del torque
     el vector del tubo (o eje de rotación) entra (y es normal a) el plano 2D,
     tal que los ángulos de rotación positivos harán que las superficies fotovoltaicas se inclinen hacia la
     IZQUIERDA y viceversa.

     Parámetros
     ----------
     solar_zenith: matriz flotante o numpy
         Ángulo cenital solar [grados]
     solar_azimuth: matriz flotante o numpy
         Ángulo de azimut solar [grados]
     eje_azimut: flotante
         Azimut del eje de la superficie fotovoltaica, es decir, dirección del eje de rotación
         [grados]

     Devoluciones
     -------
     solar_2d_vector: matriz numerosa
         Dos componentes vectoriales del vector solar en el plano 2D, con el
         forma [x, y], donde xey pueden ser matrices
    """
    solar_2d_vector = np.array([
        # a drawing really helps understand the following
        sind(solar_zenith) * cosd(solar_azimuth - axis_azimuth - 90.),
        cosd(solar_zenith)])

    return solar_2d_vector


def _get_rotation_from_tilt_azimuth(surface_azimuth, axis_azimuth, tilt):
    """Calcule el ángulo de rotación utilizando el acimut de la superficie, el azimut del eje,
     y ángulos de inclinación de la superficie. Si bien los ángulos de inclinación de la superficie siempre deben ser
     positivos, los ángulos de rotación pueden ser negativos.
     En los factores pv, los ángulos de rotación positivos indicarán pvrows que apuntan al
     izquierda, y los ángulos de rotación negativos indicarán que las filas apuntan a la
     derecha (sin importar cuál sea el acimut del eje).

     Parámetros
     ----------
     inclinación: flotante o np.ndarray
         Ángulos de inclinación de la superficie deseados [grados]. Todos los valores deben ser positivos.
     Surface_azimuth: flotante o np.ndarray
         Ángulos de acimut de la superficie fotovoltaica [grados]
     eje_azimut: flotante
         Azimut del eje de la superficie fotovoltaica, es decir, dirección del eje de rotación
         [grados]

     Devoluciones
     -------
     flotador o np.ndarray
         Ángulo(s) de rotación calculado(s) en [grados]
     """

    # Calculate rotation of PV row (signed tilt angle)
    is_pointing_right = ((surface_azimuth - axis_azimuth) % 360.) > 180.
    rotation = np.where(is_pointing_right, tilt, -tilt)
    rotation[tilt == 0] = -0.0  # GH 125
    return rotation


class BasePVArray(object):
    """Clase base para paneles fotovoltaicos en factores pv. Proporcionará información básica
     capacidades."""

    registry_cols = ['geom', 'line_type', 'pvrow_index', 'side',
                     'pvsegment_index', 'shaded', 'surface_index']

    def __init__(self, axis_azimuth=None):
        """Inicializar la base del conjunto fotovoltaico.

         Parámetros
         ----------
         axis_azimuth: flotante, opcional
             Ángulo de acimut del eje de rotación [grados] (Predeterminado = Ninguno)
         """
        # All PV arrays should have a fixed axis azimuth in pvfactors
        self.axis_azimuth = axis_azimuth

        # The are required attributes of any PV array
        self.ts_pvrows = None
        self.ts_ground = None

    @property
    def n_ts_surfaces(self):
        """Número de superficies de series temporales en el conjunto fotovoltaico."""
        n_ts_surfaces = 0
        n_ts_surfaces += self.ts_ground.n_ts_surfaces
        for ts_pvrow in self.ts_pvrows:
            n_ts_surfaces += ts_pvrow.n_ts_surfaces
        return n_ts_surfaces

    @property
    def all_ts_surfaces(self):
        """Lista de todas las superficies de la serie temporal en el campo fotovoltaico"""
        all_ts_surfaces = []
        all_ts_surfaces += self.ts_ground.all_ts_surfaces
        for ts_pvrow in self.ts_pvrows:
            all_ts_surfaces += ts_pvrow.all_ts_surfaces
        return all_ts_surfaces

    @property
    def ts_surface_indices(self):
        """Lista de índices de todas las superficies de la serie temporal"""
        return [ts_surf.index for ts_surf in self.all_ts_surfaces]

    def plot_at_idx(self, idx, ax, merge_if_flag_overlap=True,
                    with_cut_points=True, x_min_max=None,
                    with_surface_index=False):
        """Trace todas las filas fotovoltaicas y el suelo en el conjunto fotovoltaico en el punto deseado.
         índice de pasos. Esto se puede llamar antes de transformar la matriz, y
         después de colocarlo.

         Parámetros
         ----------
         idx:int
             Índice de paso de tiempo seleccionado para trazar el conjunto fotovoltaico
         hacha: :py:clase:`matplotlib.pyplot.axes` objeto
             Ejes para trazar las geometrías del generador fotovoltaico.
         merge_if_flag_overlap: bool, opcional
             Decide si fusionar todas las sombras si se superponen
             (Predeterminado = Verdadero)
         with_cut_points: booleano, opcional
             Decida si desea incluir los puntos de corte guardados en el archivo creado.
             Geometría del suelo fotovoltaico (predeterminado = verdadero)
         x_min_max: tupla, opcional
             Lista de coordenadas x mínimas y máximas para el terreno plano
             superficie [m] (Predeterminado = Ninguno)
         with_surface_index: bool, opcional
             Trazar las superficies con sus valores de índice (Predeterminado = Falso)
         """
        # Plot pv array structures
        self.ts_ground.plot_at_idx(
            idx, ax, color_shaded=COLOR_DIC['ground_shaded'],
            color_illum=COLOR_DIC['ground_illum'],
            merge_if_flag_overlap=merge_if_flag_overlap,
            with_cut_points=with_cut_points, x_min_max=x_min_max,
            with_surface_index=with_surface_index)

        for ts_pvrow in self.ts_pvrows:
            ts_pvrow.plot_at_idx(
                idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                color_illum=COLOR_DIC['pvrow_illum'],
                with_surface_index=with_surface_index)

        # Plot formatting
        ax.axis('equal')
        if self.distance is not None:
            n_pvrows = self.n_pvrows
            ax.set_xlim(- 0.5 * self.distance,
                        (n_pvrows - 0.5) * self.distance)
        if self.height is not None:
            ax.set_ylim(- self.height, 2 * self.height)
        ax.set_xlabel("x [m]", fontsize=PLOT_FONTSIZE)
        ax.set_ylabel("y [m]", fontsize=PLOT_FONTSIZE)

    def fit(self, *args, **kwargs):
        """Not implemented."""
        raise NotImplementedError

    def update_params(self, new_dict):
        """Actualizar los parámetros de superficie de la serie temporal en la colección.
         Parámetros
         ----------
         new_dict: dict
             Parámetros para agregar o actualizar para las superficies.
         """
        self.ts_ground.update_params(new_dict)
        for ts_pvrow in self.ts_pvrows:
            ts_pvrow.update_params(new_dict)

    def _index_all_ts_surfaces(self):
        """Agregue índices únicos a todas las superficies del conjunto fotovoltaico."""
        for idx, ts_surface in enumerate(self.all_ts_surfaces):
            ts_surface.index = idx


class PVSegment(GeometryCollection):
    """Un segmento fotovoltaico será una colección de 2 segmentos colineales y contiguos.
     colecciones de sombra, una sombreada y otra iluminada. Se hereda de
     :py:class:`shapely.geometry.GeometryCollection` para que los usuarios aún puedan
     llamar a métodos y propiedades geométricos básicos, por ejemplo, longitud de llamada, etc.
     """

    def __init__(self, illum_collection=ShadeCollection(shaded=False),
                 shaded_collection=ShadeCollection(shaded=True), index=None):
        """Inicializar segmento fotovoltaico.

         Parámetros
         ----------
         colección_illum: \
         :py:clase:`~pvfactors.geometry.base.ShadeCollection`, opcional
             Colección iluminada del segmento fotovoltaico (predeterminado = sombra vacía
             colección sin sombreado)
         colección_sombreada: \
         :py:clase:`~pvfactors.geometry.base.ShadeCollection`, opcional
             Colección sombreada del segmento fotovoltaico (predeterminado = sombra vacía
             colección con sombreado)
         índice: int, opcional
             Índice del segmento PV (Predeterminado = Ninguno)
         """
        assert shaded_collection.shaded, "surface should be shaded"
        assert not illum_collection.shaded, "surface should not be shaded"
        self._check_collinear(illum_collection, shaded_collection)
        self._shaded_collection = shaded_collection
        self._illum_collection = illum_collection
        self.index = index
        self._all_surfaces = None
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    def _check_collinear(self, illum_collection, shaded_collection):
        """Compruebe que todas las superficies del segmento fotovoltaico sean colineales.

         Parámetros
         ----------
         colección_illum:
         :py:clase:`~pvfactors.geometry.base.ShadeCollection`, opcional
             Colección iluminada
         colección_sombreada:
         :py:clase:`~pvfactors.geometry.base.ShadeCollection`, opcional
             colección sombreada

         Excepciones
         ------
         Exception
             Si todas las superficies no son colineales
         """
        assert illum_collection.is_collinear
        assert shaded_collection.is_collinear
        # Check that if none or all of the collection is empty, n_vectors are
        # equal
        if (not illum_collection.is_empty) \
           and (not shaded_collection.is_empty):
            n_vec_ill = illum_collection.n_vector
            n_vec_shaded = shaded_collection.n_vector
            assert are_2d_vecs_collinear(n_vec_ill, n_vec_shaded)

    def plot(self, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
             color_illum=COLOR_DIC['pvrow_illum'], with_index=False):
        """Trazar las superficies en el segmento fotovoltaico.

         Parámetros
         ----------
         hacha: :py:clase:`matplotlib.pyplot.axes` objeto
             Ejes para trazar
         color_shaded: str, opcional
             Color a utilizar para trazar las superficies sombreadas (Predeterminado =
             COLOR_DIC['pvrow_shaded'])
         color_illum: str, opcional
             Color a utilizar para trazar las superficies iluminadas (predeterminado =
             COLOR_DIC['pvrow_illum'])
         with_index: bool
             Bandera para anotar superficies con sus índices (Predeterminado = Falso)
         """
        self._shaded_collection.plot(ax, color=color_shaded,
                                     with_index=with_index)
        self._illum_collection.plot(ax, color=color_illum,
                                    with_index=with_index)

    def cast_shadow(self, linestring):
        """Proyectar sombra en el segmento PV usando una cadena de líneas: reorganizará el
         Superficies fotovoltaicas entre las colecciones sombreadas e iluminadas del
         segmento

         Parámetros
         ----------
         linestring: :py:clase:`shapely.geometry.LineString`
             Linestring proyecta una sombra sobre el segmento fotovoltaico
         """
        # Using a buffer may slow things down, but it's quite crucial
        # in order for shapely to get the intersection accurately see:
        # https://stackoverflow.com/questions/28028910/how-to-deal-with-rounding-errors-in-shapely
        intersection = (self._illum_collection.buffer(DISTANCE_TOLERANCE)
                        .intersection(linestring))
        if not intersection.is_empty:
            # Split up only if interesects the illuminated collection
            # print(intersection)
            self._shaded_collection.add_linestring(intersection,
                                                   normal_vector=self.n_vector)
            # print(self._shaded_collection.length)
            self._illum_collection.remove_linestring(intersection)
            # print(self._illum_collection.length)
            super(PVSegment, self).__init__([self._shaded_collection,
                                             self._illum_collection])

    def cut_at_point(self, point):
        """Corte el segmento PV en el punto si el segmento lo contiene.

         Parámetros
         ----------
         point : :py:clase:`shapely.geometry.Point`
             Punto donde cortar la geometría de la colección, si esta última contiene el
             anterior
         """
        if contains(self, point):
            if contains(self._illum_collection, point):
                self._illum_collection.cut_at_point(point)
            else:
                self._shaded_collection.cut_at_point(point)

    def get_param_weighted(self, param):
        """Obtener el parámetro de las superficies del segmento, después de ponderar
         por longitud de superficie.

         Parámetros
         ----------
         parámetro: cadena
             Parámetro de superficie a devolver

         Devoluciones
         -------
         float
             Valor de parámetro ponderado
         """
        value = self.get_param_ww(param) / self.length
        return value

    def get_param_ww(self, param):
        """Obtiene el parámetro de las superficies del segmento con peso, es decir,
         después de multiplicar por las longitudes de las superficies.

         Parámetros
         ----------
         parámetro: cadena
             Parámetro de superficie a devolver

         Devoluciones
         -------
         flotar
             Valor del parámetro multiplicado por pesos.

         Excepcion
         ------
         KeyError
             si el nombre del parámetro no está en los parámetros de una superficie
         """
        value = 0
        value += self._shaded_collection.get_param_ww(param)
        value += self._illum_collection.get_param_ww(param)
        return value

    def update_params(self, new_dict):
        """Actualizar parámetros de superficie en la colección.

         Parámetros
         ----------
         new_dict: dict
             Parámetros para agregar o actualizar para las superficies.
         """
        self._shaded_collection.update_params(new_dict)
        self._illum_collection.update_params(new_dict)

    @property
    def n_vector(self):
        """Dado que se supone que las superficies sombreadas e iluminadas son colineales,
         esto debería devolver el vector normal de cualquiera de las superficies. Si ambos están vacíos,
         devuelve el valor predeterminado para el vector normal."""
        if not self.illum_collection.is_empty:
            return self.illum_collection.n_vector
        elif not self.shaded_collection.is_empty:
            return self.shaded_collection.n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @property
    def n_surfaces(self):
        """Número de superficies en la colección."""
        n_surfaces = self._illum_collection.n_surfaces \
            + self._shaded_collection.n_surfaces
        return n_surfaces

    @property
    def surface_indices(self):
        """Índices de las superficies del segmento fotovoltaico."""
        list_indices = []
        list_indices += self._illum_collection.surface_indices
        list_indices += self._shaded_collection.surface_indices
        return list_indices

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None, param_names=None):
        """Cree un segmento fotovoltaico con una única superficie fotovoltaica.

         Parámetros
         ----------
         coordenadas: lista
             Lista de coordenadas de cadena lineal para la superficie
         sombreado: bool, opcional
             Estado de sombreado deseado para la superficie fotovoltaica resultante
             (Predeterminado = Falso)
         normal_vector: lista, opcional
             Vector normal para la superficie (Predeterminado = Ninguno)
         index: int, opcional
             Índice del segmento (Predeterminado = Ninguno)
         param_names: lista de cadenas, opcional
             Nombres de los parámetros de la superficie, por ejemplo, reflectividad, incidente total.
             irradiancia, temperatura, etc. (Predeterminado = Ninguno)
         """
        col = ShadeCollection.from_linestring_coords(
            coords, shaded=shaded, normal_vector=normal_vector,
            param_names=param_names)
        # Realized that needed to instantiate other_col, otherwise could
        # end up with shared collection among different PV segments
        other_col = ShadeCollection(list_surfaces=[], shaded=not shaded,
                                    param_names=param_names)
        if shaded:
            return cls(illum_collection=other_col,
                       shaded_collection=col, index=index)
        else:
            return cls(illum_collection=col,
                       shaded_collection=other_col, index=index)

    @property
    def shaded_collection(self):
        """Colección sombreada del segmento fotovoltaico"""
        return self._shaded_collection

    @shaded_collection.setter
    def shaded_collection(self, new_collection):
        """Establezca la colección sombreada del segmento fotovoltaico con una nueva.

         Parámetros
         ----------
         new_collection: :py:clase:`pvfactors.geometry.base.ShadeCollection`
             Nueva colección para usar en la actualización.
         """
        assert new_collection.shaded, "surface should be shaded"
        self._shaded_collection = new_collection
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @shaded_collection.deleter
    def shaded_collection(self):
        """Elimine la colección sombreada de segmentos fotovoltaicos y reemplácela por una vacía.
         """
        self._shaded_collection = ShadeCollection(shaded=True)
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @property
    def illum_collection(self):
        """Colección iluminada del segmento fotovoltaico."""
        return self._illum_collection

    @illum_collection.setter
    def illum_collection(self, new_collection):
        """Conjunto colección iluminada del segmento fotovoltaico con uno nuevo.

         Parámetros
         ----------
         nueva_colección: :py:clase:`pvfactors.geometry.base.ShadeCollection`
             Nueva colección para usar en la actualización.
         """
        assert not new_collection.shaded, "surface should not be shaded"
        self._illum_collection = new_collection
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @illum_collection.deleter
    def illum_collection(self):
        """Eliminar la colección iluminada de segmentos fotovoltaicos y reemplazarlos con vacíos
         uno."""
        self._illum_collection = ShadeCollection(shaded=False)
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @property
    def shaded_length(self):
        """Longitud de la colección sombreada del segmento fotovoltaico.

         Devoluciones
         -------
         float
             Longitud de la colección sombreada
         """
        return self._shaded_collection.length

    @property
    def all_surfaces(self):
        """Lista de todos los :py:class:`pvfactors.geometry.base.PVSurface`

         Devoluciones
         -------
         lista de :py:clase:`~pvfactors.geometry.base.PVSurface`
             Superficies fotovoltaicas en el segmento fotovoltaico
         """
        if self._all_surfaces is None:
            self._all_surfaces = []
            self._all_surfaces += self._illum_collection.list_surfaces
            self._all_surfaces += self._shaded_collection.list_surfaces
        return self._all_surfaces
