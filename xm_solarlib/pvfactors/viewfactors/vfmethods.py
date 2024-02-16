"""Módulo con herramientas de cálculo del factor de vista"""

from xm_solarlib.pvfactors.config import MIN_X_GROUND, MAX_X_GROUND, DISTANCE_TOLERANCE
from xm_solarlib.pvfactors.geometry.timeseries import TsLineCoords, TsPointCoords
from xm_solarlib.tools import cosd, sind
import numpy as np
import logging

class VFTsMethods(object):
    """Esta clase contiene todos los métodos utilizados para calcular series temporales.
     factores de visualización para todas las superficies en
     :py:clase:`~pvfactors.geometry.pvarray.OrderedPVArray`"""

    def vf_pvrow_gnd_surf(self, ts_pvrows, ts_ground, tilted_to_left,
                          vf_matrix):
        """Calcule los factores de visualización entre la fila fotovoltaica de la serie temporal y el terreno
         superficies y asígnelo a la matriz de factor de vista pasada usando
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

        n_pvrows = len(ts_pvrows)
        for idx_pvrow, ts_pvrow in enumerate(ts_pvrows):
            # Separate gnd surfaces depending on side
            left_gnd_surfaces = ts_ground.ts_surfaces_side_of_cut_point(
                'left', idx_pvrow)
            right_gnd_surfaces = ts_ground.ts_surfaces_side_of_cut_point(
                'right', idx_pvrow)
            # Front side
            front = ts_pvrow.front
            for pvrow_surf in front.all_ts_surfaces:
                if pvrow_surf.is_empty:
                    continue # do no run calculation for this surface
                ts_length = pvrow_surf.length
                i = pvrow_surf.index
                self._process_ground_surfaces(left_gnd_surfaces,vf_matrix, i ,  pvrow_surf, idx_pvrow, n_pvrows, tilted_to_left,ts_pvrows,ts_length,is_back= False, is_left= True )                
                self._process_ground_surfaces(right_gnd_surfaces,vf_matrix, i ,  pvrow_surf, idx_pvrow, n_pvrows, tilted_to_left,ts_pvrows,ts_length,is_back= False, is_left= False )
            # Back side
            back = ts_pvrow.back
            for pvrow_surf in back.all_ts_surfaces:
                if pvrow_surf.is_empty:
                    continue # do no run calculation for this surface
                ts_length = pvrow_surf.length
                i = pvrow_surf.index
                self._process_ground_surfaces(left_gnd_surfaces,vf_matrix, i ,  pvrow_surf, idx_pvrow, n_pvrows, tilted_to_left,ts_pvrows,ts_length,is_back= True, is_left= True )                
                self._process_ground_surfaces(right_gnd_surfaces,vf_matrix, i ,  pvrow_surf, idx_pvrow, n_pvrows, tilted_to_left,ts_pvrows,ts_length,is_back= True, is_left= False )


    def _process_ground_surfaces(self,gnd_surfaces, vf_matrix, i ,  pvrow_surf, idx_pvrow, n_pvrows, tilted_to_left,ts_pvrows,ts_length,is_back, is_left):
        """Process view factor calculations for ground surfaces relative to a PV row surface."""
        for gnd_surf in gnd_surfaces:
            if gnd_surf.is_empty:
                continue
            self._calculate_vf(vf_matrix, i ,  pvrow_surf, idx_pvrow, n_pvrows, tilted_to_left,ts_pvrows, gnd_surf,ts_length,is_back, is_left)


    def _calculate_vf(self,vf_matrix, i , pvrow_surf, idx_pvrow, n_pvrows, tilted_to_left,ts_pvrows, gnd_surf, ts_length, is_back ,is_left):
        j = gnd_surf.index
        vf_pvrow_to_gnd, vf_gnd_to_pvrow = (
            self.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
                pvrow_surf, idx_pvrow, n_pvrows,
                tilted_to_left, ts_pvrows, gnd_surf, ts_length,
                is_back=is_back, is_left=is_left)
            )
        vf_matrix[i, j, :] = vf_pvrow_to_gnd
        vf_matrix[j, i, :] = vf_gnd_to_pvrow


    def vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
            self, pvrow_surf, pvrow_idx, n_pvrows, tilted_to_left,
            ts_pvrows, gnd_surf, pvrow_surf_length, is_back=True,
            is_left=True):
        """Calcule los factores de vista desde la superficie de la fila fotovoltaica de la serie temporal hasta un
         Superficie del terreno en serie temporal. Esto devolverá la vista calculada.
         factores desde la superficie de la fila fotovoltaica hasta la superficie del suelo, Y desde la
         superficie del suelo a la superficie de la fila fotovoltaica (usando reciprocidad).

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
             Bandera que especifica si la superficie de la fila fotovoltaica está en la superficie frontal o posterior
             (Predeterminado = Verdadero)
         is_left: booleano
             Bandera que especifica si la superficie gnd queda a la izquierda del punto de corte de la fila pv o
             no (Predeterminado = Verdadero)

         Devoluciones
         -------
         vf_pvrow_to_gnd_surf: np.ndarray
             Ver factores desde la superficie de la fila fotovoltaica de la serie temporal hasta el terreno de la serie temporal
             superficie, la dimensión es [n_timesteps]
         vf_gnd_to_pvrow_surf: np.ndarray
             Ver factores desde la superficie del terreno de la serie temporal hasta la fila fotovoltaica de la serie temporal
             superficie, la dimensión es [n_timesteps]
         """

        pvrow_surf_lowest_pt = pvrow_surf.lowest_point
        pvrow_surf_highest_pt = pvrow_surf.highest_point
        no_obstruction = (is_left & (pvrow_idx == 0)) \
            or ((not is_left) & (pvrow_idx == n_pvrows - 1))
        if no_obstruction:
            # There is no obstruction to the gnd surface
            vf_pvrow_to_gnd_surf = self._vf_surface_to_surface(
                pvrow_surf.coords, gnd_surf, pvrow_surf_length)
        else:
            # Get lowest point of obstructing point
            idx_obstructing_pvrow = pvrow_idx - 1 if is_left else pvrow_idx + 1
            pt_obstr = ts_pvrows[idx_obstructing_pvrow
                                 ].full_pvrow_coords.lowest_point
            # Calculate vf from pv row to gnd surface
            vf_pvrow_to_gnd_surf = self._vf_hottel_gnd_surf(
                pvrow_surf_highest_pt, pvrow_surf_lowest_pt,
                gnd_surf.b1, gnd_surf.b2, pt_obstr, pvrow_surf_length,
                is_left)

        # Final result depends on whether front or back surface
        if is_left:
            vf_pvrow_to_gnd_surf = (
                np.where(tilted_to_left, 0., vf_pvrow_to_gnd_surf) if is_back
                else np.where(tilted_to_left, vf_pvrow_to_gnd_surf, 0.))
        else:
            vf_pvrow_to_gnd_surf = (
                np.where(tilted_to_left, vf_pvrow_to_gnd_surf, 0.) if is_back
                else np.where(tilted_to_left, 0., vf_pvrow_to_gnd_surf))

        # Use reciprocity to calculate ts vf from gnd surf to pv row surface
        gnd_surf_length = gnd_surf.length
        vf_gnd_to_pvrow_surf = np.where(
            gnd_surf_length > DISTANCE_TOLERANCE,
            vf_pvrow_to_gnd_surf * pvrow_surf_length / gnd_surf_length, 0.)

        return vf_pvrow_to_gnd_surf, vf_gnd_to_pvrow_surf

    def vf_pvrow_to_pvrow(self, ts_pvrows, tilted_to_left, vf_matrix):
        """Calculate view factors between surfaces of time series photovoltaic rows,
        and assign values to the passed view factor matrix using surface indices.
        """
        for idx_pvrow, ts_pvrow in enumerate(ts_pvrows[:-1]):
            right_ts_pvrow = ts_pvrows[idx_pvrow + 1]
            self._calculate_view_factors(ts_pvrow.front, right_ts_pvrow.back, tilted_to_left, vf_matrix, is_front_surface=True)
            self._calculate_view_factors(ts_pvrow.back, right_ts_pvrow.front, tilted_to_left, vf_matrix, is_front_surface=False)

    def _calculate_view_factors(self, source_surface, target_surface, tilted_to_left, vf_matrix, is_front_surface):
        for surf_i in source_surface.all_ts_surfaces:
            if surf_i.is_empty:
                continue
            i, length_i = surf_i.index, surf_i.length
            for surf_j in target_surface.all_ts_surfaces:
                if surf_j.is_empty:
                    continue
                j, length_j = surf_j.index, surf_j.length
                self._update_view_factor_matrix(surf_i, surf_j, i, j, length_i, length_j, tilted_to_left, vf_matrix, is_front_surface)


    def _update_view_factor_matrix(self, surf_i, surf_j, i, j, length_i, length_j, tilted_to_left, vf_matrix, is_front_surface):
        vf_i_to_j = self._vf_surface_to_surface(surf_i.coords, surf_j.coords, length_i)
        vf_i_to_j = np.where(tilted_to_left, 0., vf_i_to_j) if is_front_surface else np.where(tilted_to_left, vf_i_to_j, 0.)
        vf_j_to_i = np.where(surf_j.length > DISTANCE_TOLERANCE, vf_i_to_j * length_i / length_j, 0.)
        vf_matrix[i, j, :] = vf_i_to_j
        vf_matrix[j, i, :] = vf_j_to_i


    def calculate_vf_to_pvrow(self, pvrow_element_coords, pvrow_idx, n_pvrows,
                              n_steps, ts_pvrows, pvrow_element_length,
                              tilted_to_left, pvrow_width, rotation_vec):
        """Calcular factores de vista desde el elemento pvrow de la serie temporal hasta la serie temporal
         Filas de PV a su alrededor.

         Parámetros
         ----------
         pvrow_element_coords:
         :py:clase:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas de línea de serie temporal de pvrow_element
         pvrow_idx:int
             Índice de la fila PV de la serie temporal en la que se encuentra pvrow_element
         n_pvrows: int
             Número de filas fotovoltaicas de serie temporal en el conjunto fotovoltaico
         n_pasos: int
             Número de pasos de tiempo para los cuales calcular los factores pv
         ts_pvrows: lista de :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
             Geometrías de filas fotovoltaicas de series temporales que se utilizarán en el cálculo
         pvrow_element_length: flotante o np.ndarray
             Longitud (ancho) del elemento pvrow de la serie temporal [m]
         inclinado_a_izquierda: lista de bool
             Banderas que indican cuándo las filas de PV están estrictamente inclinadas hacia la izquierda
         pvrow_width: flotante
             Ancho de las filas fotovoltaicas de la serie temporal en el conjunto fotovoltaico [m], que es
             constante
         rotación_vec: np.ndarray
             Ángulos de rotación de las filas PV [grados]

         Devoluciones
         -------
         vf_to_pvrow: np.ndarray
             Ver factores desde la serie temporal pvrow_element hasta las filas fotovoltaicas vecinas
         vf_to_shaded_pvrow: np.ndarray
             Ver factores desde la serie temporal pvrow_element hasta las áreas sombreadas del
             filas fotovoltaicas vecinas
         """
        if pvrow_idx == 0:
            vf_left_pvrow = np.zeros(n_steps)
            vf_left_shaded_pvrow = np.zeros(n_steps)
        else:
            # Get vf to full pvrow
            left_ts_pvrow = ts_pvrows[pvrow_idx - 1]
            left_ts_pvrow_coords = left_ts_pvrow.full_pvrow_coords
            vf_left_pvrow = self._vf_surface_to_surface(
                pvrow_element_coords, left_ts_pvrow_coords,
                pvrow_element_length)
            # Get vf to shaded pvrow
            shaded_coords = self._create_shaded_side_coords(
                left_ts_pvrow.xy_center, pvrow_width,
                left_ts_pvrow.front.shaded_length, tilted_to_left,
                rotation_vec, left_ts_pvrow.full_pvrow_coords.lowest_point)
            vf_left_shaded_pvrow = self._vf_surface_to_surface(
                pvrow_element_coords, shaded_coords, pvrow_element_length)

        if pvrow_idx == (n_pvrows - 1):
            vf_right_pvrow = np.zeros(n_steps)
            vf_right_shaded_pvrow = np.zeros(n_steps)
        else:
            # Get vf to full pvrow
            right_ts_pvrow = ts_pvrows[pvrow_idx + 1]
            right_ts_pvrow_coords = right_ts_pvrow.full_pvrow_coords
            vf_right_pvrow = self._vf_surface_to_surface(
                pvrow_element_coords, right_ts_pvrow_coords,
                pvrow_element_length)
            # Get vf to shaded pvrow
            shaded_coords = self._create_shaded_side_coords(
                right_ts_pvrow.xy_center, pvrow_width,
                right_ts_pvrow.front.shaded_length, tilted_to_left,
                rotation_vec, right_ts_pvrow.full_pvrow_coords.lowest_point)
            vf_right_shaded_pvrow = self._vf_surface_to_surface(
                pvrow_element_coords, shaded_coords, pvrow_element_length)

        vf_to_pvrow = np.where(tilted_to_left, vf_right_pvrow, vf_left_pvrow)
        vf_to_shaded_pvrow = np.where(tilted_to_left, vf_right_shaded_pvrow,
                                      vf_left_shaded_pvrow)

        return vf_to_pvrow, vf_to_shaded_pvrow

    def calculate_vf_to_gnd(self, pvrow_element_coords, pvrow_idx, n_pvrows,
                            n_steps, y_ground, cut_point_coords,
                            pvrow_element_length, tilted_to_left, ts_pvrows):
        """Calcular factores de vista desde la serie temporal pvrow_element hasta el conjunto
         suelo.

         Parámetros
         ----------
         pvrow_element_coords:
         :py:clase:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas de línea de serie temporal del elemento pvrow
         pvrow_idx:int
             Índice de la fila PV de la serie temporal en la que se encuentra pvrow_element
         n_pvrows: int
             Número de filas fotovoltaicas de serie temporal en el conjunto fotovoltaico
         n_pasos: int
             Número de pasos de tiempo para los cuales calcular los factores pv
         y_ground: flotar
             Coordenada Y del terreno llano [m]
         cut_point_coords: lista de
         :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Lista de coordenadas de puntos de corte, calculadas para filas PV de series temporales
         pvrow_element_length: flotante o np.ndarray
             Longitud (ancho) de la serie temporal pvrow_element [m]
         inclinado_a_izquierda: lista de bool
             Banderas que indican cuándo las filas de PV están estrictamente inclinadas hacia la izquierda
         ts_pvrows: lista de :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
             Geometrías de filas fotovoltaicas de series temporales que se utilizarán en el cálculo

         Devoluciones
         -------
         vf_to_gnd: np.ndarray
             Ver factores desde la serie temporal pvrow_element hasta todo el terreno
         """

        pvrow_lowest_pt = ts_pvrows[pvrow_idx].full_pvrow_coords.lowest_point
        if pvrow_idx == 0:
            # There is no obstruction to view of the ground on the left
            coords_left_gnd = TsLineCoords(
                TsPointCoords(MIN_X_GROUND * np.ones(n_steps), y_ground),
                TsPointCoords(np.minimum(MAX_X_GROUND, cut_point_coords.x),
                              y_ground))
            vf_left_ground = self._vf_surface_to_surface(
                pvrow_element_coords, coords_left_gnd, pvrow_element_length)
        else:
            # The left PV row obstructs the view of the ground on the left
            left_pt_neighbor = \
                ts_pvrows[pvrow_idx - 1].full_pvrow_coords.lowest_point
            coords_gnd_proxy = TsLineCoords(left_pt_neighbor, pvrow_lowest_pt)
            vf_left_ground = self._vf_surface_to_surface(
                pvrow_element_coords, coords_gnd_proxy, pvrow_element_length)

        if pvrow_idx == (n_pvrows - 1):
            # There is no obstruction of the view of the ground on the right
            coords_right_gnd = TsLineCoords(
                TsPointCoords(np.maximum(MIN_X_GROUND, cut_point_coords.x),
                              y_ground),
                TsPointCoords(MAX_X_GROUND * np.ones(n_steps), y_ground))
            vf_right_ground = self._vf_surface_to_surface(
                pvrow_element_coords, coords_right_gnd, pvrow_element_length)
        else:
            # The right PV row obstructs the view of the ground on the right
            right_pt_neighbor = \
                ts_pvrows[pvrow_idx + 1].full_pvrow_coords.lowest_point
            coords_gnd_proxy = TsLineCoords(pvrow_lowest_pt, right_pt_neighbor)
            vf_right_ground = self._vf_surface_to_surface(
                pvrow_element_coords, coords_gnd_proxy, pvrow_element_length)

        # Merge the views of the ground for the back side
        vf_ground = np.where(tilted_to_left, vf_right_ground, vf_left_ground)

        return vf_ground

    def calculate_vf_to_shadow_obstruction_hottel(
            self, pvrow_element, pvrow_idx, n_shadows, n_steps, tilted_to_left,
            ts_pvrows, shadow_left, shadow_right, pvrow_element_length):
        """Calculate view factors from timeseries pvrow_element to the shadow
        of a specific timeseries PV row which is casted on the ground.

        Parameters
        ----------
        pvrow_element : :py:class:`~pvfactors.geometry.timeseries.TsDualSegment`\
            or :py:class:`~pvfactors.geometry.timeseries.TsSurface`
            Timeseries pvrow_element to use for calculation
        pvrow_idx : int
            Index of the timeseries PV row on the which the pvrow_element is
        n_shadows : int
            Number of timeseries PV rows in the PV array, and therefore number
            of shadows they cast on the ground
        n_steps : int
            Number of timesteps for which to calculate the pvfactors
        tilted_to_left : list of bool
            Flags indicating when the PV rows are strictly tilted to the left
        ts_pvrows : list of :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
            Timeseries PV row geometries that will be used in the calculation
        shadow_left : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Coordinates of the shadow that are on the left side of the cut
            point of the PV row on which the pvrow_element is
        shadow_right : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Coordinates of the shadow that are on the right side of the cut
            point of the PV row on which the pvrow_element is
        pvrow_element_length : float or np.ndarray
            Length (width) of the timeseries pvrow_element [m]

        Returns
        -------
        vf_to_shadow : np.ndarray
            View factors from timeseries pvrow_element to the ground shadow of
            a specific timeseries PV row
        """
        logging.debug("Calculating view factors to shadow obstruction%s",n_steps)
        pvrow_element_lowest_pt = pvrow_element.lowest_point
        pvrow_element_highest_pt = pvrow_element.highest_point
        # Calculate view factors to left shadows
        if pvrow_idx == 0:
            # There is no obstruction on the left
            vf_to_left_shadow = self._vf_surface_to_surface(
                pvrow_element.coords, shadow_left, pvrow_element_length)
        else:
            # There is potential obstruction on the left
            pt_obstr = ts_pvrows[pvrow_idx - 1].full_pvrow_coords.lowest_point
            is_shadow_left = True
            vf_to_left_shadow = self._vf_hottel_gnd_surf(
                pvrow_element_highest_pt, pvrow_element_lowest_pt,
                shadow_left.b1, shadow_left.b2, pt_obstr, pvrow_element_length,
                is_shadow_left)

        # Calculate view factors to right shadows
        if pvrow_idx == n_shadows - 1:
            # There is no obstruction on the right
            vf_to_right_shadow = self._vf_surface_to_surface(
                pvrow_element.coords, shadow_right, pvrow_element_length)
        else:
            # There is potential obstruction on the right
            pt_obstr = ts_pvrows[pvrow_idx + 1].full_pvrow_coords.lowest_point
            is_shadow_left = False
            vf_to_right_shadow = self._vf_hottel_gnd_surf(
                pvrow_element_highest_pt, pvrow_element_lowest_pt,
                shadow_right.b1, shadow_right.b2, pt_obstr,
                pvrow_element_length, is_shadow_left)

        # Filter since we're considering the back surface only
        vf_to_shadow = np.where(tilted_to_left, vf_to_right_shadow,
                                vf_to_left_shadow)

        return vf_to_shadow

    def _vf_hottel_gnd_surf(self, high_pt_pv, low_pt_pv, left_pt_gnd,
                            right_pt_gnd, obstr_pt, width, shadow_is_left):
        """
         Calcule los factores de vista de series temporales de una superficie fotovoltaica definida por baja
         y puntos altos, a una superficie del terreno definida por puntos izquierdo y derecho,
         teniendo en cuenta la posible obstrucción de las filas fotovoltaicas vecinas,
         definido por un punto de obstrucción, y todo esto usando el Hottel
         Método de cadena.

         Parámetros
         ----------
         high_pt_pv: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto más alto de la superficie fotovoltaica, para cada marca de tiempo
         low_pt_pv: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto más bajo de la superficie fotovoltaica, para cada marca de tiempo
         left_pt_gnd: :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto más a la izquierda de la superficie del suelo, para cada marca de tiempo
         right_pt_gnd: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto más a la derecha de la superficie del suelo, para cada marca de tiempo
         obstr_pt: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto de obstrucción de la fila fotovoltaica vecina, para cada marca de tiempo
         ancho: flotante o np.ndarray
             Ancho de la superficie de la fila fotovoltaica considerada, desde el punto más bajo hasta el punto más alto [m]
         sombra_is_izquierda: bool
             Lado de la superficie de sombra (o terreno) considerada con respecto a
             el punto del borde de la fila PV en el que se encuentra la superficie PV considerada
             situado

         Devoluciones
         -------
         vf_1_to_2: np.ndarray
             Ver factores desde la superficie fotovoltaica hasta la superficie del suelo (sombra)
         """

        if shadow_is_left:
            # When the shadow is left
            # - uncrossed strings are high_pv - left_gnd and low_pv - right_gnd
            # - crossed strings are high_pv - right_gnd and low_pv - left_gnd
            l1 = self._hottel_string_length(high_pt_pv, left_pt_gnd, obstr_pt,
                                            shadow_is_left)
            l2 = self._hottel_string_length(low_pt_pv, right_pt_gnd, obstr_pt,
                                            shadow_is_left)
            d1 = self._hottel_string_length(high_pt_pv, right_pt_gnd, obstr_pt,
                                            shadow_is_left)
            d2 = self._hottel_string_length(low_pt_pv, left_pt_gnd, obstr_pt,
                                            shadow_is_left)
        else:
            # When the shadow is right
            # - uncrossed strings are high_pv - right_gnd and low_pv - left_gnd
            # - crossed strings are high_pv - left_gnd and low_pv - right_gnd
            l1 = self._hottel_string_length(high_pt_pv, right_pt_gnd, obstr_pt,
                                            shadow_is_left)
            l2 = self._hottel_string_length(low_pt_pv, left_pt_gnd, obstr_pt,
                                            shadow_is_left)
            d1 = self._hottel_string_length(high_pt_pv, left_pt_gnd, obstr_pt,
                                            shadow_is_left)
            d2 = self._hottel_string_length(low_pt_pv, right_pt_gnd, obstr_pt,
                                            shadow_is_left)
        vf_1_to_2 = (d1 + d2 - l1 - l2) / (2. * width)
        # The formula doesn't work if surface is a point
        vf_1_to_2 = np.where(width > DISTANCE_TOLERANCE, vf_1_to_2, 0.)

        return vf_1_to_2

    def _hottel_string_length(self, pt_pv, pt_gnd, pt_obstr, shadow_is_left):
        """
         Calcule la longitud de una cadena según lo definido por el método Hottel String en el
         Cálculo de factores de visión, que permite tener en cuenta las obstrucciones.

         Parámetros
         ----------
         left_pt_gnd: :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto más a la izquierda de la superficie del suelo, para cada marca de tiempo
         right_pt_gnd: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto más a la derecha de la superficie del suelo, para cada marca de tiempo
         obstr_pt: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Punto de obstrucción de la fila fotovoltaica vecina, para cada marca de tiempo
         sombra_is_izquierda: bool
             Lado de la superficie de sombra (o terreno) considerada con respecto a
             el punto del borde de la fila PV en el que se encuentra la superficie PV considerada
             situado

         Devoluciones
         -------
         hottel_length: np.ndarray
             Devuelve la longitud de la serie temporal de la cadena, teniendo en cuenta
             obstrucciones, en [m]
         """
        # Calculate length of string without obstruction
        l_pv = self._distance(pt_pv, pt_gnd)
        if pt_obstr is None:
            # There can't be any obstruction
            hottel_length = l_pv
        else:
            # Determine if there is obstruction by using the angles made by
            # specific strings with the x-axis
            alpha_pv = self._angle_with_x_axis(pt_gnd, pt_pv)
            alpha_ob = self._angle_with_x_axis(pt_gnd, pt_obstr)
            if shadow_is_left:
                is_obstructing = alpha_pv > alpha_ob
            else:
                is_obstructing = alpha_pv < alpha_ob
            # Calculate length of string with obstruction
            l_obstr = (self._distance(pt_gnd, pt_obstr)
                       + self._distance(pt_obstr, pt_pv))
            # Merge based on whether there is obstruction or not
            hottel_length = np.where(is_obstructing, l_obstr, l_pv)
        return hottel_length

    def _vf_surface_to_surface(self, line_1, line_2, width_1):
        """Calcular factores de vista entre coordenadas de línea de series temporales y usar
         el método Hottel String para calcular los factores de vista (sin
         obstrucción).

         Parámetros
         ----------
         línea_1: :py:clase:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas de línea de serie temporal de la superficie 1
         línea_2: :py:clase:`~pvfactors.geometry.timeseries.TsLineCoords`
             Coordenadas de línea de serie temporal de la superficie 2
         ancho_1: flotante o np.ndarray
             Longitud de la línea_1 en [m]

         Devoluciones
         -------
         vf_1_to_2: np.ndarray
             Ver factores desde la línea_1 hasta la línea_2, para cada paso de tiempo
         """
        length_1 = self._distance(line_1.b1, line_2.b1)
        length_2 = self._distance(line_1.b2, line_2.b2)
        length_3 = self._distance(line_1.b1, line_2.b2)
        length_4 = self._distance(line_1.b2, line_2.b1)
        sum_1 = length_1 + length_2
        sum_2 = length_3 + length_4
        vf_1_to_2 = np.abs(sum_2 - sum_1) / (2. * width_1)
        # The formula doesn't work if the line is a point
        vf_1_to_2 = np.where(width_1 > DISTANCE_TOLERANCE, vf_1_to_2, 0.)

        return vf_1_to_2

    @staticmethod
    def _distance(pt_1, pt_2):
        """Calcular la distancia entre dos puntos de la serie temporal

         Parámetros
         ----------
         pt_1: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Coordenadas del punto de serie temporal del punto 1
         pt_2: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Coordenadas del punto de serie temporal del punto 2

         Devoluciones
         -------
         np.ndarray
             Distancia entre los dos puntos, para cada paso de tiempo
         """
        return np.sqrt((pt_2.y - pt_1.y)**2 + (pt_2.x - pt_1.x)**2)

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

    @staticmethod
    def _create_shaded_side_coords(xy_center, width, shaded_length,
                                   mask_tilted_to_left, rotation_vec,
                                   side_lowest_pt):
        """
         Cree las coordenadas de línea de la serie temporal para la parte sombreada de un
         Lado de la fila PV, según la longitud sombreada ingresada.

         Parámetros
         ----------
         xy_center: tupla de flotador
             Coordenadas x e y del punto central de la fila PV (invariante)
         ancho: flotador
             ancho de las filas fotovoltaicas [m]
         longitud_sombreada: np.ndarray
             Valores de series temporales de longitud del lado sombreado [m]
         inclinado_a_izquierda: lista de bool
             Banderas que indican cuándo las filas de PV están estrictamente inclinadas hacia la izquierda
         rotación_vec: np.ndarray
             Vector de rotación de serie temporal de las filas PV en [grados]
         side_lowest_pt: :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
             Coordenadas de la serie temporal del punto más bajo del lado de la fila fotovoltaica considerada

         Devoluciones
         -------
         side_shaded_coords: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Coordenadas de la serie temporal de la parte sombreada del lado de la fila PV
         """

        # Get invariant values
        x_center, y_center = xy_center
        radius = width / 2.

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

        side_shaded_coords = TsLineCoords(TsPointCoords(x_sh, y_sh),
                                          side_lowest_pt)

        return side_shaded_coords