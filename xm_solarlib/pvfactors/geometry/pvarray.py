"""Módulo que contiene clases de paneles fotovoltaicos, que utilizarán filas fotovoltaicas y tierra
geometrías."""

import numpy as np
from xm_solarlib.pvfactors.config import X_ORIGIN_PVROWS, DISTANCE_TOLERANCE
from xm_solarlib.pvfactors.geometry.base import \
    _get_solar_2d_vectors, BasePVArray, _get_rotation_from_tilt_azimuth
from xm_solarlib.pvfactors.geometry.pvrow import TsPVRow
from xm_solarlib.pvfactors.geometry.pvground import TsGround


class OrderedPVArray(BasePVArray):
    """Un conjunto fotovoltaico ordenado tiene un suelo horizontal plano y filas fotovoltaicas que
     están todos a la misma altura, con los mismos ángulos de inclinación y acimut de la superficie,
     y además todos igualmente espaciados. Estas simplificaciones permiten una gestión más rápida y sencilla.
     cálculos. En el conjunto fotovoltaico ordenado, la lista de filas fotovoltaicas debe ser
     ordenados de izquierda a derecha (a lo largo del eje x) en la geometría 2D."""

    y_ground = 0.  # ground will be at height = 0 by default

    def __init__(self, axis_azimuth=None, gcr=None, pvrow_height=None,
                 n_pvrows=None, pvrow_width=None, param_names=None,
                 cut=None):
        """Inicializar el conjunto fotovoltaico solicitado.
         La lista de filas de PV se ordenará de izquierda a derecha.

         Parámetros
         ----------
         axis_azimuth: flotante, opcional
             Ángulo de acimut del eje de rotación [grados] (Predeterminado = Ninguno)
         gcr: flotante, opcional
             Relación de cobertura del suelo (predeterminado = ninguno)
         pvrow_height: flotante, opcional
             Altura única de todas las filas PV en [m] (Predeterminado = Ninguno)
         n_pvrows: int, opcional
             Número de filas fotovoltaicas en el conjunto fotovoltaico (predeterminado = Ninguno)
         pvrow_width: flotante, opcional
             Ancho de las filas PV en el plano 2D en [m] (Predeterminado = Ninguno)
         param_names: lista de cadenas, opcional
             Lista de nombres de parámetros de superficie para las superficies fotovoltaicas
             (Predeterminado = Ninguno)
         cut: dict, opcional
             Diccionario anidado que indica si algunos lados de las filas PV deben ser
             discretizado y cómo (Predeterminado = Ninguno).
             Ejemplo: {1: {'front': 5}}, creará 5 segmentos en el frente
             lado de la fila PV con índice 1
         """
        # Initialize base parameters: common to all sorts of PV arrays
        super(OrderedPVArray, self).__init__(axis_azimuth=axis_azimuth)

        # These are the invariant parameters of the PV array
        self.gcr = gcr
        self.height = pvrow_height
        self.distance = (pvrow_width / gcr
                         if (pvrow_width is not None) and (gcr is not None)
                         else None)
        self.width = pvrow_width
        self.n_pvrows = n_pvrows
        self.param_names = [] if param_names is None else param_names
        self.cut = {} if cut is None else cut

        # These attributes will be updated at fitting time
        self.solar_2d_vectors = None
        self.n_states = None
        self.has_direct_shading = None
        self.rotation_vec = None
        self.shaded_length_front = None
        self.shaded_length_back = None

        # These attributes will be updated by the engine
        self.ts_vf_matrix = None
        self.ts_vf_aoi_matrix = None

    @classmethod
    def init_from_dict(cls, pvarray_params, param_names=None):
        """Crear una instancia del conjunto fotovoltaico ordenado a partir del diccionario de parámetros

         Parámetros
         ----------
         pvarray_params: dictar
             Los parámetros que definen el conjunto fotovoltaico
         param_names: lista de cadenas, opcional
             Lista de nombres de parámetros para pasar a superficies (Predeterminado = Ninguno)

         Devoluciones
         -------
         OrdenadoPVArray
             Matriz fotovoltaica ordenada inicializada
         """
        return cls(axis_azimuth=pvarray_params['axis_azimuth'],
                   gcr=pvarray_params['gcr'],
                   pvrow_height=pvarray_params['pvrow_height'],
                   n_pvrows=pvarray_params['n_pvrows'],
                   pvrow_width=pvarray_params['pvrow_width'],
                   cut=pvarray_params.get('cut', {}),
                   param_names=param_names)

    @classmethod
    def fit_from_dict_of_scalars(cls, pvarray_params, param_names=None):
        """Crear una instancia y ajustar un conjunto fotovoltaico ordenado utilizando el diccionario
         de entradas escalares.

         Parámetros
         ----------
         pvarray_params: dictar
             Los parámetros utilizados para la creación de instancias, el ajuste y la transformación.
         param_names: lista de cadenas, opcional
             Lista de nombres de parámetros para pasar a superficies (Predeterminado = Ninguno)

         Devoluciones
         -------
         OrdenadoPVArray
             Matriz fotovoltaica ordenada inicializada y ajustada
         """

        # Create pv array
        pvarray = cls.init_from_dict(pvarray_params,
                                     param_names=param_names)

        # Fit pv array to scalar values
        solar_zenith = np.array([pvarray_params['solar_zenith']])
        solar_azimuth = np.array([pvarray_params['solar_azimuth']])
        surface_tilt = np.array([pvarray_params['surface_tilt']])
        surface_azimuth = np.array([pvarray_params['surface_azimuth']])
        pvarray.fit(solar_zenith, solar_azimuth,
                    surface_tilt, surface_azimuth)

        return pvarray

    def fit(self, solar_zenith, solar_azimuth, surface_tilt, surface_azimuth):
        """Ajuste el conjunto fotovoltaico solicitado a la lista de ángulos solares y de superficie.
         Todos los resultados del conjunto fotovoltaico intermedio necesarios para construir las geometrías.
         Se calculará aquí utilizando la vectorización tanto como sea posible.

         Los resultados intermedios incluyen: coordenadas de fila PV para todas las marcas de tiempo,
         coordenadas del elemento terrestre para todas las marcas de tiempo, casos de directo
         sombreado,...

         Parámetros
         ----------
         solar_zenith: tipo matriz o flotante
             Ángulos cenital solares [grados]
         solar_azimuth: tipo matriz o flotante
             Ángulos de azimut solar [grados]
         Surface_tilt: tipo matriz o flotante
             Ángulos de inclinación de la superficie, de 0 a 180 [grados]
         Surface_azimuth: tipo matriz o flotante
             Ángulos de azimut de superficie [grados]
         """

        self.n_states = len(solar_zenith)
        # Calculate rotation angles
        rotation_vec = _get_rotation_from_tilt_azimuth(
            surface_azimuth, self.axis_azimuth, surface_tilt)
        # Save rotation vector
        self.rotation_vec = rotation_vec

        # Calculate the solar 2D vectors for all timestamps
        self.solar_2d_vectors = _get_solar_2d_vectors(
            solar_zenith, solar_azimuth, self.axis_azimuth)
        # Calculate the angle made by 2D sun vector and x-axis
        alpha_vec = np.arctan2(self.solar_2d_vectors[1],
                               self.solar_2d_vectors[0])

        # Calculate the coordinates of all PV rows for all timestamps
        self._calculate_pvrow_elements_coords(alpha_vec, rotation_vec)

        # Calculate ground elements coordinates for all timestamps
        self.ts_ground = TsGround.from_ts_pvrows_and_angles(
            self.ts_pvrows, alpha_vec, rotation_vec, y_ground=self.y_ground,
            flag_overlap=self.has_direct_shading,
            param_names=self.param_names)

        # Save surface rotation angles
        self.rotation_vec = rotation_vec

        # Index all timeseries surfaces
        self._index_all_ts_surfaces()

    def _calculate_pvrow_elements_coords(self, alpha_vec, rotation_vec):
        """Calcule elementos de coordenadas de fila PV de forma vectorizada, como
         Coordenadas de límites de fila PV y longitudes sombreadas.

         Parámetros
         ----------
         alpha_vec: tipo matriz o flotante
             Ángulo formado por el vector solar 2d y el eje x [rad]
         rotacion_vec: tipo matriz o flotante
             Ángulo de rotación de las filas PV [grados]
         """
        # Initialize timeseries pv rows
        self.ts_pvrows = []

        # Calculate interrow direct shading lengths
        self._calculate_interrow_shading(alpha_vec, rotation_vec)

        # Calculate coordinates of segments of each pv row side
        xy_centers = [(X_ORIGIN_PVROWS + idx * self.distance,
                       self.height + self.y_ground)
                      for idx in range(self.n_pvrows)]
        tilted_to_left = rotation_vec > 0.
        for idx_pvrow, xy_center in enumerate(xy_centers):
            # A special treatment needs to be applied to shaded lengths for
            # the PV rows at the edge of the PV array
            if idx_pvrow == 0:
                # the leftmost row doesn't have left neighbors
                shaded_length_front = np.where(tilted_to_left, 0.,
                                               self.shaded_length_front)
                shaded_length_back = np.where(tilted_to_left,
                                              self.shaded_length_back, 0.)
            elif idx_pvrow == (self.n_pvrows - 1):
                # the rightmost row does have right neighbors
                shaded_length_front = np.where(tilted_to_left,
                                               self.shaded_length_front, 0.)
                shaded_length_back = np.where(tilted_to_left, 0.,
                                              self.shaded_length_back)
            else:
                # use calculated shaded lengths
                shaded_length_front = self.shaded_length_front
                shaded_length_back = self.shaded_length_back
            # Create timeseries PV rows and add it to the list
            self.ts_pvrows.append(TsPVRow.from_raw_inputs(
                xy_center, self.width, rotation_vec,
                self.cut.get(idx_pvrow, {}), shaded_length_front,
                shaded_length_back, index=idx_pvrow,
                param_names=self.param_names))

    def _calculate_interrow_shading(self, alpha_vec, rotation_vec):
        """Calcule la longitud sombreada en la parte delantera y trasera de las filas fotovoltaicas cuando
         El sombreado directo ocurre y de forma vectorizada.

         Parámetros
         ----------
         alpha_vec: tipo matriz o flotante
             Ángulo formado por el vector solar 2d y el eje x [rad]
         rotacion_vec: tipo matriz o flotante
             Ángulo de rotación de las filas PV [grados]
         """

        if self.n_pvrows > 1:
            # Calculate intermediate values for direct shading
            alpha_vec_deg = np.rad2deg(alpha_vec)
            theta_t = 90. - rotation_vec
            theta_t_rad = np.deg2rad(theta_t)
            beta = theta_t + alpha_vec_deg
            beta_rad = np.deg2rad(beta)
            delta = self.distance * (
                np.sin(theta_t_rad) - np.cos(theta_t_rad) * np.tan(beta_rad))
            # Calculate temporary shaded lengths
            tmp_shaded_length_front = np.maximum(0, self.width - delta)
            tmp_shaded_length_back = np.maximum(0, self.width + delta)
            # The shaded length can't be longer than PV row (meaning sun can't
            # be under the horizon or something...)
            self.shaded_length_front = np.where(
                tmp_shaded_length_front > self.width, 0,
                tmp_shaded_length_front)
            self.shaded_length_back = np.where(
                tmp_shaded_length_back > self.width, 0,
                tmp_shaded_length_back)
        else:
            # Since there's 1 row, there can't be any direct shading
            self.shaded_length_front = np.zeros(self.n_states)
            self.shaded_length_back = np.zeros(self.n_states)

        # Flag times when there's direct shading
        self.has_direct_shading = (
            (self.shaded_length_front > DISTANCE_TOLERANCE)
            | (self.shaded_length_back > DISTANCE_TOLERANCE))