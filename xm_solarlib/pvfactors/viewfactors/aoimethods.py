"""Módulo que contiene métodos de cálculo de pérdidas AOI"""

from xm_solarlib.pvfactors.config import DISTANCE_TOLERANCE
from xm_solarlib.pvfactors.geometry.timeseries import (
    TsPointCoords, TsSurface, TsLineCoords)

import xm_solarlib
from xm_solarlib.tools import cosd
import numpy as np


class AOIMethods:
    """Clase que contiene métodos relacionados con el cálculo de pérdidas de AOI para
     :py:clase:`~pvfactors.geometry.pvarray.OrderedPVArray` objetos."""

    def __init__(self, faoi_fn_front, faoi_fn_back, n_integral_sections=300):
        """Crear una instancia de clase con la función faoi y el número de secciones a usar
         calcular integrales de factores de vista con pérdidas faoi

         Parámetros
         ----------
         faoi_fn_front: función
             Función que toma una lista (o una gran variedad) de ángulos de incidencia
             medido desde la superficie horizontal
             (con valores de 0 a 180 grados) y devuelve los valores fAOI para
             el lado frontal de las filas fotovoltaicas
         faoi_fn_back: función
             Función que toma una lista (o una gran variedad) de ángulos de incidencia
             medido desde la superficie horizontal
             (con valores de 0 a 180 grados) y devuelve los valores fAOI para
             la parte posterior de las filas fotovoltaicas
         n_integral_sections: int, opcional
             Número de divisiones integrales del intervalo de 0 a 180 grados
             a utilizar para la integral de pérdida fAOI (predeterminado = 300)
         """
        # Check that faoi fn where passed
        faoi_fns_ok = callable(faoi_fn_front) and callable(faoi_fn_back)
        if not faoi_fns_ok:
            raise ValueError("The faoi_fn passed to the AOI methods are "
                                 "not callable. Please check the fAOI "
                                 "functions again")
        self.faoi_fn_front = faoi_fn_front
        self.faoi_fn_back = faoi_fn_back
        self.n_integral_sections = n_integral_sections
        # The following will be updated at fitting time
        self.interval = None
        self.aoi_angles_low = None
        self.aoi_angles_high = None
        self.integrand_front = None
        self.integrand_back = None

    def fit(self, n_timestamps):
        """Ajustar los métodos AOI a las entradas de series temporales: crear todo lo necesario
         atributos de integración.

         Parámetros
         ----------
         n_marcas de tiempo: int
             Número de marcas de tiempo de simulación
         """
        # Will use x values at the middle of the integral sections
        aoi_angles = np.linspace(0., 180., num=self.n_integral_sections + 1)
        # Assumes that at least 2 aoi angle values, otherwise what's the point
        self.interval = aoi_angles[1] - aoi_angles[0]
        # Get integral intervals' low, high, and middle points
        aoi_angles_low = aoi_angles[:-1]
        aoi_angles_high = aoi_angles_low + self.interval
        aoi_angles_middle = aoi_angles_low + self.interval / 2.
        # Calculate faoi values using middle points of integral intervals
        faoi_front = self.faoi_fn_front(aoi_angles_middle)
        faoi_back = self.faoi_fn_back(aoi_angles_middle)
        # Calculate small view factor values for each section
        vf_values = self._vf(aoi_angles_low, aoi_angles_high)
        # Multiply to get integrand
        integrand_front = faoi_front * vf_values
        integrand_back = faoi_back * vf_values
        # Replicate these values for all timestamps such that shapes
        # becomes: [n_timestamps, n_integral_sections]map
        self.aoi_angles_low = np.tile(aoi_angles_low, (n_timestamps, 1))
        self.aoi_angles_high = np.tile(aoi_angles_high, (n_timestamps, 1))
        self.integrand_front = np.tile(integrand_front, (n_timestamps, 1))
        self.integrand_back = np.tile(integrand_back, (n_timestamps, 1))

    def vf_aoi_pvrow_to_sky(self, ts_pvrows, ts_ground, tilted_to_left,
                            vf_matrix):
        """Calcule los factores de visualización entre la superficie de la fila fotovoltaica de la serie temporal y el cielo
         mientras contabiliza las pérdidas de AOI,
         y asigne valores a la matriz de factor de vista pasada usando
         los índices de superficie.

         Parámetros
         ----------
         ts_pvrows: lista de :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
             Lista de filas fotovoltaicas de serie temporal en el conjunto fotovoltaico
         ts_ground: :py:clase:`~pvfactors.geometry.timeseries.TsGround`
             Tierra de la serie temporal del conjunto fotovoltaico
         inclinado_a_izquierda: lista de bool
             Banderas que indican cuándo las filas de PV están estrictamente inclinadas hacia la izquierda
         vf_matrix: np.ndarray
             Ver matriz de factores para actualizar durante el cálculo. debería tener 3
             dimensiones de la siguiente manera: [n_surfaces, n_surfaces, n_timesteps]
         """
        sky_index = vf_matrix.shape[0] - 1
        # --- Build list of dummy sky surfaces
        # create sky left open area
        pt_1 = TsPointCoords(ts_ground.x_min * np.ones_like(tilted_to_left),
                             ts_ground.y_ground * np.ones_like(tilted_to_left))
        pt_2 = ts_pvrows[0].highest_point
        sky_left = TsSurface(TsLineCoords(pt_1, pt_2))
        # create sky right open area
        pt_1 = TsPointCoords(ts_ground.x_max * np.ones_like(tilted_to_left),
                             ts_ground.y_ground * np.ones_like(tilted_to_left))
        pt_2 = ts_pvrows[-1].highest_point
        sky_right = TsSurface(TsLineCoords(pt_2, pt_1))
        # Add sky surfaces in-between PV rows
        dummy_sky_surfaces = [sky_left]
        for idx_pvrow, ts_pvrow in enumerate(ts_pvrows[:-1]):
            right_ts_pvrow = ts_pvrows[idx_pvrow + 1]
            pt_1 = ts_pvrow.highest_point
            pt_2 = right_ts_pvrow.highest_point
            sky_surface = TsSurface(TsLineCoords(pt_1, pt_2))
            dummy_sky_surfaces.append(sky_surface)
        # Add sky right open area
        dummy_sky_surfaces.append(sky_right)

        # Now calculate vf_aoi for all PV row surfaces to sky
        for idx_pvrow, ts_pvrow in enumerate(ts_pvrows):
            # Get dummy sky surfaces
            sky_left = dummy_sky_surfaces[idx_pvrow]
            sky_right = dummy_sky_surfaces[idx_pvrow + 1]
            # Calculate vf_aoi for surfaces in PV row
            # front side
            front = ts_pvrow.front
            for front_surf in front.all_ts_surfaces:
                vf_aoi_left = self._vf_aoi_surface_to_surface(
                    front_surf, sky_left, is_back=False)
                vf_aoi_right = self._vf_aoi_surface_to_surface(
                    front_surf, sky_right, is_back=False)
                vf_aoi = np.where(tilted_to_left, vf_aoi_left, vf_aoi_right)
                vf_matrix[front_surf.index, sky_index, :] = vf_aoi
            # back side
            back = ts_pvrow.back
            for back_surf in back.all_ts_surfaces:
                vf_aoi_left = self._vf_aoi_surface_to_surface(
                    back_surf, sky_left, is_back=True)
                vf_aoi_right = self._vf_aoi_surface_to_surface(
                    back_surf, sky_right, is_back=True)
                vf_aoi = np.where(tilted_to_left, vf_aoi_right, vf_aoi_left)
                vf_matrix[back_surf.index, sky_index, :] = vf_aoi

    def vf_aoi_pvrow_to_pvrow(self, ts_pvrows, tilted_to_left, vf_matrix):
        """Calcule los factores de visualización entre superficies de filas fotovoltaicas de series temporales
         mientras contabiliza las pérdidas de AOI,
         y asigne valores a la matriz de factor de vista pasada usando
         los índices de superficie.

         Parámetros
         ----------
         ts_pvrows: lista de :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
             Lista de filas fotovoltaicas de serie temporal en el conjunto fotovoltaico
         inclinado_a_izquierda: lista de bool
             Banderas que indican cuándo las filas de PV están estrictamente inclinadas hacia la izquierda
         vf_matrix: np.ndarray
             Ver matriz de factores para actualizar durante el cálculo. debería tener 3
             dimensiones de la siguiente manera: [n_surfaces, n_surfaces, n_timesteps]
         """
        for idx_pvrow, ts_pvrow in enumerate(ts_pvrows[:-1]):
            # Get the next pv row
            right_ts_pvrow = ts_pvrows[idx_pvrow + 1]
            # front side
            front = ts_pvrow.front
            for surf_i in front.all_ts_surfaces:
                i = surf_i.index
                for surf_j in right_ts_pvrow.back.all_ts_surfaces:
                    j = surf_j.index
                    # vf aoi from i to j
                    vf_i_to_j = self._vf_aoi_surface_to_surface(
                        surf_i, surf_j, is_back=False)
                    vf_i_to_j = np.where(tilted_to_left, 0., vf_i_to_j)
                    # vf aoi from j to i
                    vf_j_to_i = self._vf_aoi_surface_to_surface(
                        surf_j, surf_i, is_back=True)
                    vf_j_to_i = np.where(tilted_to_left, 0., vf_j_to_i)
                    # save results
                    vf_matrix[i, j, :] = vf_i_to_j
                    vf_matrix[j, i, :] = vf_j_to_i
            # back side
            back = ts_pvrow.back
            for surf_i in back.all_ts_surfaces:
                i = surf_i.index
                for surf_j in right_ts_pvrow.front.all_ts_surfaces:
                    j = surf_j.index
                    # vf aoi from i to j
                    vf_i_to_j = self._vf_aoi_surface_to_surface(
                        surf_i, surf_j, is_back=True)
                    vf_i_to_j = np.where(tilted_to_left, vf_i_to_j, 0.)
                    # vf aoi from j to i
                    vf_j_to_i = self._vf_aoi_surface_to_surface(
                        surf_j, surf_i, is_back=False)
                    vf_j_to_i = np.where(tilted_to_left, vf_j_to_i, 0.)
                    # save results
                    vf_matrix[i, j, :] = vf_i_to_j
                    vf_matrix[j, i, :] = vf_j_to_i


    def vf_aoi_pvrow_to_gnd(self, ts_pvrows, ts_ground, tilted_to_left, vf_aoi_matrix):
        """
        Calculate view factors between time series PV row and ground surfaces considering non-diffuse AOI losses,
        and assign it to the passed aoi view factor matrix using surface indices.

        Parameters
        ----------
        ts_pvrows: list of :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
            List of time series PV rows in the PV array
        ts_ground: :py:class:`~pvfactors.geometry.timeseries.TsGround`
            Time series ground of the PV array
        tilted_to_left: list of bool
            Flags indicating when PV rows are strictly tilted to the left
        vf_aoi_matrix: np.ndarray
            View factor aoi matrix to update during the calculation. Should have 3 dimensions as follows: [n_surfaces, n_surfaces, n_timesteps]
        """

        n_pvrows = len(ts_pvrows)
        for idx_pvrow, ts_pvrow in enumerate(ts_pvrows):
            # Separate ground surfaces depending on side
            left_gnd_surfaces = ts_ground.ts_surfaces_side_of_cut_point('left', idx_pvrow)
            right_gnd_surfaces = ts_ground.ts_surfaces_side_of_cut_point('right', idx_pvrow)

            # Process front and back sides
            self._process_pvrow_side(ts_pvrow.front, left_gnd_surfaces, right_gnd_surfaces, idx_pvrow, n_pvrows, tilted_to_left, ts_pvrows, vf_aoi_matrix, is_back=False)
            self._process_pvrow_side(ts_pvrow.back, left_gnd_surfaces, right_gnd_surfaces, idx_pvrow, n_pvrows, tilted_to_left, ts_pvrows, vf_aoi_matrix, is_back=True)

    def _process_pvrow_side(self, pvrow_side, left_gnd_surfaces, right_gnd_surfaces, idx_pvrow, n_pvrows, tilted_to_left, ts_pvrows, vf_aoi_matrix, is_back):
        """
        Helper function to process each side of PV row.
        """
        for pvrow_surf in pvrow_side.all_ts_surfaces:
            ts_length = pvrow_surf.length
            i = pvrow_surf.index
            for gnd_surf in left_gnd_surfaces:
                j = gnd_surf.index
                vf_pvrow_to_gnd = self._vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
                    pvrow_surf, idx_pvrow, n_pvrows, tilted_to_left, ts_pvrows, gnd_surf, ts_length, is_back=is_back, is_left=True)
                vf_aoi_matrix[i, j, :] = vf_pvrow_to_gnd
            for gnd_surf in right_gnd_surfaces:
                j = gnd_surf.index
                vf_pvrow_to_gnd = self._vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
                    pvrow_surf, idx_pvrow, n_pvrows, tilted_to_left, ts_pvrows, gnd_surf, ts_length, is_back=is_back, is_left=False)
                vf_aoi_matrix[i, j, :] = vf_pvrow_to_gnd


    def _vf_aoi_surface_to_surface(self, surf_1, surf_2, is_back=True):
        """Calcule el factor de vista, teniendo en cuenta las pérdidas de AOI, de
         superficie 1 a la superficie 2.

         Notas
         -----
         Esto supone que surf_1 es infinitesimal (muy pequeño)

         Parámetros
         ----------
         surf_1: :py:clase:`~pvfactors.geometry.timeseries.TsSurface`
             Superficie infinitesimal a partir de la cual calcular el factor de vista
             Pérdidas de AOI
         surf_2: :py:clase:`~pvfactors.geometry.timeseries.TsSurface`
             Superficie a la que debe aplicarse el factor de visión con pérdidas de AOI
             calculado
         is_back: booleano
             Bandera que especifica si la superficie de la fila fotovoltaica está en la parte posterior o frontal
             de fila PV (Predeterminado = Verdadero)

         Devoluciones
         -------
         vf_aoi: np.ndarray
             Ver factores con pérdidas de aoi desde la superficie 1 a la superficie 2,
             la dimensión es [n_timesteps]
         """
        # skip calculation if either surface is empty (always zero length)
        skip = surf_1.is_empty or surf_2.is_empty
        if skip:
            vf_aoi = np.zeros_like(surf_2.length)
        else:
            # Get surface 1 params
            u_vector = surf_1.u_vector
            centroid = surf_1.centroid
            # Calculate AOI angles
            aoi_angles_1 = self._calculate_aoi_angles(u_vector, centroid,
                                                      surf_2.b1)
            aoi_angles_2 = self._calculate_aoi_angles(u_vector, centroid,
                                                      surf_2.b2)
            low_aoi_angles = np.where(aoi_angles_1 < aoi_angles_2, aoi_angles_1,
                                      aoi_angles_2)
            high_aoi_angles = np.where(aoi_angles_1 < aoi_angles_2, aoi_angles_2,
                                       aoi_angles_1)
            # Calculate vf_aoi
            vf_aoi_raw = self._calculate_vf_aoi_wedge_level(
                low_aoi_angles, high_aoi_angles, is_back=is_back)
            # Should be zero where either of the surfaces have zero length
            vf_aoi = np.where((surf_1.length < DISTANCE_TOLERANCE)
                              | (surf_2.length < DISTANCE_TOLERANCE), 0.,
                              vf_aoi_raw)
        return vf_aoi


    def _vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
            self, pvrow_surf, pvrow_idx, n_pvrows, tilted_to_left, ts_pvrows,
            gnd_surf, ts_length, is_back=True, is_left=True):
        """Calcule los factores de vista desde la superficie de la fila fotovoltaica de la serie temporal hasta un
         superficie del terreno en serie temporal, lo que representa las pérdidas de AOI.
         Esto devolverá la vista calculada.
         factores desde la superficie de la fila fotovoltaica hasta la superficie del suelo.
 
         Notas
         -----
         Esto supone que las superficies de las filas PV son infinitesimales (muy pequeñas)
 
         Parámetros
         ----------
         pvrow_surf: :py:clase:`~pvfactors.geometry.timeseries.TsSurface`
             Superficie de fila fotovoltaica de serie temporal que se utilizará para el cálculo
         pvrow_idx:int
             Índice de la fila PV de la serie temporal en la que se encuentra pvrow_surf
         n_pvrows: int
             Número de filas fotovoltaicas de serie temporal en el conjunto fotovoltaico y, por tanto, número
             de sombras que proyectan en el suelo
         inclinado_a_izquierda: lista de bool
             Banderas que indican cuándo las filas de PV están estrictamente inclinadas hacia la izquierda
         ts_pvrows: lista de :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
             Lista de filas fotovoltaicas de serie temporal en el conjunto fotovoltaico
         gnd_surf: :py:clase:`~pvfactors.geometry.timeseries.TsSurface`
             Superficie del terreno de serie temporal que se utilizará para el cálculo
         pvrow_surf_length: np.ndarray
             Longitud (ancho) de la superficie de la fila fotovoltaica de la serie temporal [m]
         is_back: booleano
             Bandera que especifica si la superficie de la fila fotovoltaica está en la parte posterior o frontal
             de fila PV (Predeterminado = Verdadero)
         is_left: booleano
             Bandera que especifica si la superficie gnd queda a la izquierda del punto de corte de la fila pv o
             no (Predeterminado = Verdadero)
 
         Devoluciones
         -------
         vf_aoi_pvrow_to_gnd_surf: np.ndarray
             Ver factores aoi desde la superficie de fila fotovoltaica de la serie temporal hasta la serie temporal
             superficie del suelo, la dimensión es [n_timesteps]
         """
        # skip calculation if either surface is empty (always zero length)
        skip = pvrow_surf.is_empty or gnd_surf.is_empty
        if skip:
            vf_aoi = np.zeros_like(gnd_surf.length)
        else:
            centroid = pvrow_surf.centroid
            u_vector = pvrow_surf.u_vector
            no_obstruction = (is_left & (pvrow_idx == 0)) \
                or ((not is_left) & (pvrow_idx == n_pvrows - 1))
            if no_obstruction:
                # There is no obstruction to the ground surface
                aoi_angles_1 = self._calculate_aoi_angles(u_vector, centroid,
                                                          gnd_surf.b1)
                aoi_angles_2 = self._calculate_aoi_angles(u_vector, centroid,
                                                          gnd_surf.b2)
            else:
                # Get lowest point of obstructing point
                idx_obstructing_pvrow = (pvrow_idx - 1 if is_left
                                         else pvrow_idx + 1)
                pt_obstr = ts_pvrows[idx_obstructing_pvrow
                                     ].full_pvrow_coords.lowest_point
                # adjust angle seen when there is obstruction
                aoi_angles_1 = self._calculate_aoi_angles_w_obstruction(
                    u_vector, centroid, gnd_surf.b1, pt_obstr, is_left)
                aoi_angles_2 = self._calculate_aoi_angles_w_obstruction(
                    u_vector, centroid, gnd_surf.b2, pt_obstr, is_left)
 
            low_aoi_angles = np.where(aoi_angles_1 < aoi_angles_2,
                                      aoi_angles_1, aoi_angles_2)
            high_aoi_angles = np.where(aoi_angles_1 < aoi_angles_2,
                                       aoi_angles_2, aoi_angles_1)
            vf_aoi_raw = self._calculate_vf_aoi_wedge_level(
                low_aoi_angles, high_aoi_angles, is_back=is_back)
            # Should be zero where either of the surfaces have zero length
            vf_aoi_raw = np.where((ts_length < DISTANCE_TOLERANCE)
                                  | (gnd_surf.length < DISTANCE_TOLERANCE), 0.,
                                  vf_aoi_raw)
 
            vf_aoi = self._final_result_pvrow_surf_to_gnd_surf_obstruction(tilted_to_left, vf_aoi_raw, is_back, is_left)
 
        return vf_aoi
    
    def _final_result_pvrow_surf_to_gnd_surf_obstruction(self,tilted_to_left, vf_aoi_raw , is_back, is_left):
    # Final result depends on whether front or back surface
        if is_left:
            vf_aoi = (np.where(tilted_to_left, 0., vf_aoi_raw) if is_back
                        else np.where(tilted_to_left, vf_aoi_raw, 0.))
        else:
            vf_aoi = (np.where(tilted_to_left, vf_aoi_raw, 0.) if is_back
                        else np.where(tilted_to_left, 0., vf_aoi_raw))
        return vf_aoi


    def _determine_no_obstruction(self, is_left, pvrow_idx, n_pvrows):
        """Determine if there is no obstruction."""
        return (is_left and pvrow_idx == 0) or (not is_left and pvrow_idx == n_pvrows - 1)

    def _calculate_aoi_angles_based_on_obstruction(
            self, no_obstruction, u_vector, centroid, gnd_surf, ts_pvrows, pvrow_idx, is_left):
        """Calculate AOI angles based on obstruction."""
        if no_obstruction:
            return (self._calculate_aoi_angles(u_vector, centroid, gnd_surf.b1),
                    self._calculate_aoi_angles(u_vector, centroid, gnd_surf.b2))
        else:
            idx_obstructing_pvrow = pvrow_idx - 1 if is_left else pvrow_idx + 1
            pt_obstr = ts_pvrows[idx_obstructing_pvrow].full_pvrow_coords.lowest_point
            return (self._calculate_aoi_angles_w_obstruction(u_vector, centroid, gnd_surf.b1, pt_obstr, is_left),
                    self._calculate_aoi_angles_w_obstruction(u_vector, centroid, gnd_surf.b2, pt_obstr, is_left))

    def _get_low_high_aoi_angles(self, aoi_angles_1, aoi_angles_2):
        """Get low and high AOI angles."""
        return (np.where(aoi_angles_1 < aoi_angles_2, aoi_angles_1, aoi_angles_2),
                np.where(aoi_angles_1 < aoi_angles_2, aoi_angles_2, aoi_angles_1))

    def _adjust_vf_aoi_based_on_orientation(self, vf_aoi_raw, tilted_to_left, is_back, is_left):
        """Adjust VF AOI based on orientation."""
        if is_left:
            return np.where(tilted_to_left, 0., vf_aoi_raw) if is_back else np.where(tilted_to_left, vf_aoi_raw, 0.)
        else:
            return np.where(tilted_to_left, vf_aoi_raw, 0.) if is_back else np.where(tilted_to_left, 0., vf_aoi_raw)

    def _calculate_aoi_angles_w_obstruction(
            self, u_vector, centroid, point_gnd, point_obstr,
            gnd_surf_is_left):
        """Calcule los ángulos AOI para una superficie de fila fotovoltaica del
         :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray` que ve
         una superficie del suelo, mientras está potencialmente obstruido por otro
         fila fotovoltaica

         Parámetros
         ----------
         u_vector: np.ndarray
             Vector de dirección de la superficie para calcular los ángulos AOI
         centroide: :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto centroide de la superficie de la fila PV para calcular los ángulos AOI
         punto: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto de la superficie del suelo que determinará el ángulo AOI.
         point_obstr: :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto potencialmente obstructivo para el cálculo del ángulo de visión.
         gnd_surf_is_left:bool
             Bandera que especifica si la superficie del suelo queda del corte de la fila fotovoltaica
             punto o no

         Devoluciones
         -------
         np.ndarray
             Ángulos AOI formados por un punto remoto y un centroide en la superficie,
             medido contra el vector de dirección de la superficie, teniendo en cuenta
             posible obstrucción [grados]
         """
        if point_obstr is None:
            # There is no obstruction
            point = point_gnd
        else:
            # Determine if there is obstruction by using the angles made by
            # specific strings with the x-axis
            alpha_pv = self._angle_with_x_axis(point_gnd, centroid)
            alpha_ob = self._angle_with_x_axis(point_gnd, point_obstr)
            if gnd_surf_is_left:
                is_obstructing = alpha_pv > alpha_ob
            else:
                is_obstructing = alpha_pv < alpha_ob
            x = np.where(is_obstructing, point_obstr.x, point_gnd.x)
            y = np.where(is_obstructing, point_obstr.y, point_gnd.y)
            point = TsPointCoords(x, y)

        aoi_angles = self._calculate_aoi_angles(u_vector, centroid, point)
        return aoi_angles

    def _calculate_vf_aoi_wedge_level(self, low_angles, high_angles,
                                      is_back=True):
        """Calcule los factores de vista modificados de faoi para una cuña definida por
         ángulos bajos y altos.

         Parámetros
         ----------
         ángulos_bajos: np.ndarray
             Ángulos AOI bajos (entre 0 y 180 grados), longitud = n_timestamps
         ángulos_altos: np.ndarray
             Ángulos AOI altos (entre 0 y 180 grados), longitud = n_timestamps.
             Debería ser más grande que ``low_angles``
         is_back: booleano
             Bandera que especifica si la superficie de la fila fotovoltaica está en la parte posterior o frontal
             de fila PV (Predeterminado = Verdadero)

         Devoluciones
         -------
         np.ndarray
             factores de vista modificados por faoi para cuña
             forma = (n_marcas de tiempo,)
         """
        # Calculate integrand: all d_vf_aoi values
        faoi_integrand = self._calculate_vfaoi_integrand(
            low_angles, high_angles, is_back=is_back)
        # Total vf_aoi will be sum of all smaller d_vf_aoi values
        total_vf_aoi = faoi_integrand.sum(axis=1)
        # Make sure vf is counted as zero if the wedge is super small
        total_vf_aoi = np.where(
            np.abs(high_angles - low_angles) < DISTANCE_TOLERANCE, 0.,
            total_vf_aoi)

        return total_vf_aoi

    def _calculate_vfaoi_integrand(self, low_angles, high_angles,
                                   is_back=True):
        """
         Calcule los factores de vista de la serie temporal con el integrando de pérdida aoi
         dados los ángulos bajos y altos que definen la superficie.

         Parámetros
         ----------
         ángulos_bajos: np.ndarray
             Ángulos AOI bajos (entre 0 y 180 grados), longitud = n_timestamps
         ángulos_altos: np.ndarray
             Ángulos AOI altos (entre 0 y 180 grados), longitud = n_timestamps.
             Debería ser más grande que ``low_angles``
         is_back: booleano
             Bandera que especifica si la superficie de la fila fotovoltaica está en la parte posterior o frontal
             de fila PV (Predeterminado = Verdadero)
         Devoluciones
         -------
         np.ndarray
             Valores del integrando vf_aoi para todas las marcas de tiempo.
             forma = (n_marcas de tiempo, n_secciones_integrales)
         """

        # Turn into dimension: [n_timestamps, n_integral_sections]
        low_angles_mat = np.tile(low_angles, (self.n_integral_sections, 1)).T
        high_angles_mat = np.tile(high_angles, (self.n_integral_sections, 1)).T

        # Filter out integrand values outside of range
        include_integral_section = ((low_angles_mat <= self.aoi_angles_high) &
                                    (high_angles_mat > self.aoi_angles_low))
        # The integrand values are different for front and back sides
        if is_back:
            faoi_integrand = np.where(include_integral_section,
                                      self.integrand_back, 0.)
        else:
            faoi_integrand = np.where(include_integral_section,
                                      self.integrand_front, 0.)

        return faoi_integrand

    @staticmethod
    def _calculate_aoi_angles(u_vector, centroid, point):
        """Calcule los ángulos AOI a partir del vector de dirección de la superficie,
         punto centroide de esa superficie y punto de otra superficie

         Parámetros
         ----------
         u_vector: np.ndarray
             Vector de dirección de la superficie para calcular los ángulos AOI
         centroide: :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto centroide de la superficie para calcular los ángulos AOI
         punto: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto de superficie remota que determinará el ángulo AOI

         Devoluciones
         -------
         np.ndarray
             Ángulos AOI formados por un punto remoto y un centroide en la superficie,
             medido contra el vector de dirección de la superficie [grados]
         """
        v_vector = np.array([point.x - centroid.x, point.y - centroid.y])
        dot_product = u_vector[0, :] * v_vector[0, :] \
            + u_vector[1, :] * v_vector[1, :]
        u_norm = np.linalg.norm(u_vector, axis=0)
        v_norm = np.linalg.norm(v_vector, axis=0)
        cos_theta = dot_product / (u_norm * v_norm)
        # because of round off errors, cos_theta can be slightly > 1,
        # or slightly < -1, so clip it
        cos_theta = np.clip(cos_theta, -1., 1.)
        aoi_angles = np.rad2deg(np.arccos(cos_theta))
        return aoi_angles

    @staticmethod
    def _vf(aoi_1, aoi_2):
        """Calcule el factor de vista desde una superficie infinitesimal hasta una banda infinita.

         Ver ilustración: http://www.thermalradiation.net/sectionb/B-71.html
         Aquí estamos usando ángulos medidos desde la horizontal.

         Parámetros
         ----------
         aoi_1: np.ndarray
             Ángulos inferiores que definen la banda infinita.
         aoi_2: np.ndarray
             Ángulos más altos que definen la banda infinita.

         Devoluciones
         -------
         np.ndarray
             Ver factores desde una superficie infinitesimal hasta una franja infinita

         """
        return 0.5 * np.abs(cosd(aoi_1) - cosd(aoi_2))

    @staticmethod
    def _angle_with_x_axis(pt_1, pt_2):
        """Ángulo con el eje x del vector que va de pt_1 a pt_2

         Parámetros
         ----------
         pt_1: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Coordenadas del punto de serie temporal del punto 1
         pt_2: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Coordenadas del punto de serie temporal del punto 2

         Devoluciones
         -------
         np.ndarray
             Ángulo entre el vector pt_1->pt_2 y el eje x
         """
        return np.arctan2(pt_2.y - pt_1.y, pt_2.x - pt_1.x)

    def rho_from_faoi_fn(self, is_back):
        """Calcular la reflectividad promedio global a partir de la función faoi
         para cualquier lado de la fila PV (requiere calcular factores de vista)

         Parámetros
         ----------
         is_back: booleano
             Bandera que especifica si se utiliza la función faoi frontal o posterior
         Devoluciones
         -------
         rho_average: flotante
             Valor de reflectividad promedio global de la superficie.
         """
        # Will use x values at the middle of the integral sections
        aoi_angles = np.linspace(0., 180., num=self.n_integral_sections + 1)
        # Assumes that at least 2 aoi angle values, otherwise what's the point
        self.interval = aoi_angles[1] - aoi_angles[0]
        # Get integral intervals' low, high, and middle points
        aoi_angles_low = aoi_angles[:-1]
        aoi_angles_high = aoi_angles_low + self.interval
        aoi_angles_middle = aoi_angles_low + self.interval / 2.
        # Calculate faoi values using middle points of integral intervals
        if is_back:
            faoi_values = self.faoi_fn_back(aoi_angles_middle)
        else:
            faoi_values = self.faoi_fn_front(aoi_angles_middle)
        # Calculate small view factor values for each section
        vf_values = self._vf(aoi_angles_low, aoi_angles_high)
        # Multiply to get integrand
        integrand_values = faoi_values * vf_values
        return (1. - integrand_values.sum())


def faoi_fn_from_xm_solarlib_sandia(pvmodule_name):
    """Generar una función faoi a partir del nombre de un módulo fotovoltaico xm_solarlib sandia

     Parámetros
     ----------
     pvmodule_name: cadena
         Nombre del módulo fotovoltaico en la base de datos del módulo xm_solarlib Sandia

     Devoluciones
     -------
     función_faoi
         Función que devuelve valores de pérdida positivos para entradas numéricas
         entre 0 y 180 grados.
     """
    # Get Sandia module database from xm_solarlib
    sandia_modules = xm_solarlib.pvsystem.retrieve_sam('SandiaMod')
    # Grab pv module sandia coeffs from database
    pvmodule = sandia_modules[pvmodule_name]

    def fn(angles):
        """Función de pérdida fAOI: calcula cuánta luz se absorbe en un momento dado
         ángulos de incidencia

         Parámetros
         ----------
         ángulos: np.ndarray o lista
             Ángulos medidos desde la superficie horizontal, desde 0 de 180 grados.

         Devoluciones
         -------
         np.ndarray
             valores fAOI
         """
        angles = np.array(angles) if isinstance(angles, list) else angles
        # Transform the inputs for the SAPM function
        angles = np.where(angles >= 90, angles - 90, 90. - angles)
        # Use xm_solarlib sapm aoi loss method
        return xm_solarlib.iam.sapm(angles, pvmodule, upper=1.)

    return fn