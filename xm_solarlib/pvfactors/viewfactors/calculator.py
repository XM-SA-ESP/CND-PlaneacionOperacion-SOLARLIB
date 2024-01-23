"""Módulo con clases y funciones para calcular vistas y factores de vista"""


from xm_solarlib.pvfactors.config import DISTANCE_TOLERANCE
from xm_solarlib.pvfactors.viewfactors.vfmethods import VFTsMethods
from xm_solarlib.pvfactors.viewfactors.aoimethods import AOIMethods
import numpy as np


class VFCalculator(object):
    """Esta clase de calculadora se utilizará para el cálculo de factores de vista.
     para :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray`, y lo hará
     confíe en ambos :py:class:`~pvfactors.viewfactors.vfmethods.VFTsMethods`
     y :py:class:`~pvfactors.viewfactors.aoimethods.AOIMethods`"""

    def __init__(self, faoi_fn_front=None, faoi_fn_back=None,
                 n_aoi_integral_sections=300):
        """Inicializar la calculadora de factor de vista con los métodos de cálculo
         que será utilizado. Los métodos AOI no serán instanciados si un
         Falta la función fAOI.

         Parámetros
         ----------
         faoi_fn_front: función u objeto, opcional
             Función (u objeto que contiene el método ``faoi``)
             que toma una lista (o una gran variedad) de ángulos de incidencia
             medido desde la superficie horizontal
             (con valores de 0 a 180 grados) y devuelve los valores fAOI para
             el lado frontal de las filas PV (predeterminado = Ninguno)
         faoi_fn_back: función u objeto, opcional
             Función (u objeto que contiene el método ``faoi``)
             que toma una lista (o una gran variedad) de ángulos de incidencia
             medido desde la superficie horizontal
             (con valores de 0 a 180 grados) y devuelve los valores fAOI para
             la parte posterior de las filas PV (predeterminado = Ninguno)
         n_integral_sections: int, opcional
             Número de divisiones integrales del intervalo de 0 a 180 grados
             a utilizar para la integral de pérdida fAOI (predeterminado = 300)
         """
        self.vf_ts_methods = VFTsMethods()
        # Do not instantiate AOIMethods if missing faoi function
        if (faoi_fn_front is None) or (faoi_fn_back is None):
            self.vf_aoi_methods = None
        else:
            # Check whether got function or object, and take ``faoi`` method
            # if object was passed
            faoi_fn_front = (faoi_fn_front.faoi
                             if hasattr(faoi_fn_front, 'faoi')
                             else faoi_fn_front)
            faoi_fn_back = (faoi_fn_back.faoi
                            if hasattr(faoi_fn_back, 'faoi') else faoi_fn_back)
            self.vf_aoi_methods = AOIMethods(
                faoi_fn_front, faoi_fn_back,
                n_integral_sections=n_aoi_integral_sections)
        # Saved matrices
        self.vf_matrix = None
        self.vf_aoi_matrix = None

    def fit(self, n_timestamps):
        """Ajuste la calculadora de factor de vista a las entradas de la serie temporal.

         Parámetros
         ----------
         n_marcas de tiempo: int
             Número de marcas de tiempo de simulación
         """
        if self.vf_aoi_methods is not None:
            self.vf_aoi_methods.fit(n_timestamps)

    def build_ts_vf_matrix(self, pvarray):
        """Calcular la matriz de factores de vista de series temporales para el dado
         matriz fotovoltaica ordenada

         Parámetros
         ----------
         pvarray: :py:clase:`~pvfactors.geometry.pvarray.OrderedPVArray`
             Conjunto fotovoltaico cuya serie temporal ve la matriz de factores para calcular

         Devoluciones
         -------
         np.ndarray
             Matriz de factores de vista de series temporales, con 3 dimensiones:
             [n_superficies, n_superficies, n_timesteps]
         """

        # Initialize matrix
        rotation_vec = pvarray.rotation_vec
        tilted_to_left = rotation_vec > 0
        n_steps = len(rotation_vec)
        n_ts_surfaces = pvarray.n_ts_surfaces
        vf_matrix = np.zeros((n_ts_surfaces + 1, n_ts_surfaces + 1, n_steps),
                             dtype=float)  # don't forget to include the sky

        # Get timeseries objects
        ts_ground = pvarray.ts_ground
        ts_pvrows = pvarray.ts_pvrows

        # Calculate ts view factors between pvrow and ground surfaces
        self.vf_ts_methods.vf_pvrow_gnd_surf(ts_pvrows, ts_ground,
                                             tilted_to_left, vf_matrix)
        # Calculate view factors between pv rows
        self.vf_ts_methods.vf_pvrow_to_pvrow(ts_pvrows, tilted_to_left,
                                             vf_matrix)
        # Calculate view factors to sky
        vf_matrix[:-1, -1, :] = 1. - np.sum(vf_matrix[:-1, :-1, :], axis=1)
        # This is not completely accurate yet, we need to set the sky vf
        # to zero when the surfaces have zero length
        for i, ts_surf in enumerate(pvarray.all_ts_surfaces):
            vf_matrix[i, -1, :] = np.where(ts_surf.length > DISTANCE_TOLERANCE,
                                           vf_matrix[i, -1, :], 0.)

        # Save in calculator
        self.vf_matrix = vf_matrix

        return vf_matrix

    def build_ts_vf_aoi_matrix(self, pvarray, rho_mat):
        """Calcule el factor de vista de los elementos de la matriz aoi de todas las filas PV
         superficies a todas las demás superficies, únicamente.
         Si los métodos AOI están disponibles, vf_aoi_matrix tendrá en cuenta
         para pérdidas por reflexión que son específicas de AOI. De lo contrario será
         Supongamos que todas las pérdidas por reflexión son difusas.

         Notas
         -----
         Cuando se utilizan métodos fAOI, esto no calculará
         ver factores desde las superficies del suelo hasta las superficies de las filas fotovoltaicas, para que los usuarios
         tendrá que correr
         :py:meth:`~pvfactors.viewfactors.calculator.VFCalculator.build_ts_vf_matrix`
         primero si quieren la matriz completa; de lo contrario, esas entradas se
         tienen valores cero en ellos.


         Parámetros
         ----------
         pvarray: :py:clase:`~pvfactors.geometry.pvarray.OrderedPVArray`
             Conjunto fotovoltaico cuya serie temporal ve la matriz AOI del factor para calcular
         rho_mat: np.ndarray
             Matriz 2D de valores de reflectividad para todas las superficies del
             Conjunto fotovoltaico + cielo.
             Forma = [n_ts_surfaces + 1, n_ts_surfaces + 1, n_timestamps]

         Devoluciones
         -------
         np.ndarray
             Matriz de factores de vista de series temporales para superficies de filas fotovoltaicas infinitesimales,
             y contabilizar las pérdidas de AOI, con 3 dimensiones:
             [n_superficies, n_superficies, n_timesteps]
         """
        # Initialize matrix
        rotation_vec = pvarray.rotation_vec
        tilted_to_left = rotation_vec > 0
        n_steps = len(rotation_vec)
        n_ts_surfaces = pvarray.n_ts_surfaces
        vf_aoi_matrix = np.zeros(
            (n_ts_surfaces + 1, n_ts_surfaces + 1, n_steps),
            dtype=float) if self.vf_matrix is None else self.vf_matrix

        # Get timeseries objects
        ts_ground = pvarray.ts_ground
        ts_pvrows = pvarray.ts_pvrows

        if self.vf_aoi_methods is None:
            # The reflection losses will be considered all diffuse.
            faoi_diffuse = 1. - rho_mat
            vf_aoi_matrix = faoi_diffuse * vf_aoi_matrix
        else:
            # Calculate vf_aoi between pvrow and ground surfaces
            self.vf_aoi_methods.vf_aoi_pvrow_to_gnd(ts_pvrows, ts_ground,
                                                    tilted_to_left,
                                                    vf_aoi_matrix)
            # Calculate vf_aoi between pvrows
            self.vf_aoi_methods.vf_aoi_pvrow_to_pvrow(
                ts_pvrows, tilted_to_left, vf_aoi_matrix)
            # Calculate vf_aoi between prows and sky
            self.vf_aoi_methods.vf_aoi_pvrow_to_sky(
                ts_pvrows, ts_ground, tilted_to_left, vf_aoi_matrix)

        # Save results
        self.vf_aoi_matrix = vf_aoi_matrix

        return vf_aoi_matrix

    def get_vf_ts_pvrow_element(self, pvrow_idx, pvrow_element, ts_pvrows,
                                ts_ground, rotation_vec, pvrow_width):
        """Calcular los factores de vista de la serie temporal del elemento pvrow de la serie temporal
         (segmento o superficie) a todos los demás elementos del conjunto fotovoltaico.

         Parámetros
         ----------
         pvrow_idx:int
             Índice de la fila PV de la serie temporal para la que queremos calcular el
             irradiancia de la superficie posterior
         elemento_pvrow: \
         :py:clase:`~pvfactors.geometry.timeseries.TsDualSegment` \
         o :py:class:`~pvfactors.geometry.timeseries.TsSurface`
             Elemento de fila PV de serie temporal para calcular los factores de vista
         ts_pvrows: lista de :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
             Lista de filas fotovoltaicas de serie temporal en el conjunto fotovoltaico
         ts_ground: :py:clase:`~pvfactors.geometry.timeseries.TsGround`
             Tierra de la serie temporal del conjunto fotovoltaico
         rotación_vec: np.ndarray
             Vector de rotación de serie temporal de las filas PV en [grados]
         pvrow_width: flotante
             Ancho de las filas fotovoltaicas de la serie temporal en la matriz en [m]

         Devoluciones
         -------
         view_factors: dictar
             Diccionario de los factores de visualización de series temporales para todo tipo de superficies.
             en el conjunto fotovoltaico. La lista de claves incluye: 'to_each_gnd_shadow',
             'to_gnd_shaded', 'to_gnd_illum', 'to_gnd_total', 'to_pvrow_total',
             'to_pvrow_shaded', 'to_pvrow_illum', 'to_sky'
         """
        tilted_to_left = rotation_vec > 0
        n_shadows = len(ts_pvrows)
        n_steps = len(rotation_vec)
        pvrow_element_coords = pvrow_element.coords
        pvrow_element_length = pvrow_element_coords.length

        # Get shadows on left and right sides of PV row
        shadows_coords_left = \
            ts_ground.shadow_coords_left_of_cut_point(pvrow_idx)
        shadows_coords_right = \
            ts_ground.shadow_coords_right_of_cut_point(pvrow_idx)
        # Calculate view factors to ground shadows
        list_vf_to_obstructed_gnd_shadows = []
        for i in range(n_shadows):
            shadow_left = shadows_coords_left[i]
            shadow_right = shadows_coords_right[i]
            # vfs to obstructed gnd shadows
            vf_obstructed_shadow = (
                self.vf_ts_methods.calculate_vf_to_shadow_obstruction_hottel(
                    pvrow_element, pvrow_idx, n_shadows, n_steps,
                    tilted_to_left, ts_pvrows, shadow_left, shadow_right,
                    pvrow_element_length))
            list_vf_to_obstructed_gnd_shadows.append(vf_obstructed_shadow)
        list_vf_to_obstructed_gnd_shadows = np.array(
            list_vf_to_obstructed_gnd_shadows)

        # Calculate view factors to shaded ground
        vf_shaded_gnd = np.sum(list_vf_to_obstructed_gnd_shadows, axis=0)

        # Calculate view factors to whole ground
        vf_gnd_total = self.vf_ts_methods.calculate_vf_to_gnd(
            pvrow_element_coords, pvrow_idx, n_shadows, n_steps,
            ts_ground.y_ground, ts_ground.cut_point_coords[pvrow_idx],
            pvrow_element_length, tilted_to_left, ts_pvrows)

        # Calculate view factors to illuminated ground
        vf_illum_gnd = vf_gnd_total - vf_shaded_gnd

        # Calculate view factors to pv rows
        vf_pvrow_total, vf_pvrow_shaded = \
            self.vf_ts_methods.calculate_vf_to_pvrow(
                pvrow_element_coords, pvrow_idx, n_shadows, n_steps, ts_pvrows,
                pvrow_element_length, tilted_to_left, pvrow_width,
                rotation_vec)
        vf_pvrow_illum = vf_pvrow_total - vf_pvrow_shaded

        # Calculate view factors to sky
        vf_to_sky = 1. - vf_gnd_total - vf_pvrow_total

        # return all timeseries view factors
        view_factors = {
            'to_each_gnd_shadow': list_vf_to_obstructed_gnd_shadows,
            'to_gnd_shaded': vf_shaded_gnd,
            'to_gnd_illum': vf_illum_gnd,
            'to_gnd_total': vf_gnd_total,
            'to_pvrow_total': vf_pvrow_total,
            'to_pvrow_shaded': vf_pvrow_shaded,
            'to_pvrow_illum': vf_pvrow_illum,
            'to_sky': vf_to_sky
        }

        return view_factors