"""Módulo que contiene modelos de irradiancia utilizados con geometrías de paneles fotovoltaicos"""

from xm_solarlib.tools import cosd
from xm_solarlib.irradiance import aoi as aoi_function
from xm_solarlib.irradiance import get_total_irradiance
import numpy as np
from  xm_solarlib.pvfactors.irradiance.utils import \
    perez_diffuse_luminance, calculate_horizon_band_shading, \
    calculate_circumsolar_shading
from  xm_solarlib.pvfactors.irradiance.base import BaseModel
from  xm_solarlib.pvfactors.config import \
    DEFAULT_HORIZON_BAND_ANGLE, SKY_REFLECTIVITY_DUMMY, \
    DEFAULT_CIRCUMSOLAR_ANGLE


class IsotropicOrdered(BaseModel):
    """Modelo de cielo isotrópico difuso para
     :py:clase:`~pvfactors.geometry.pvarray.OrderedPVArray`. Va a
     calcular los valores apropiados para una cúpula de cielo isotrópico y aplicar
     a la matriz fotovoltaica."""

    params = ['rho', 'inv_rho', 'direct', 'isotropic', 'reflection']
    cats = ['ground_illum', 'ground_shaded', 'front_illum_pvrow',
            'back_illum_pvrow', 'front_shaded_pvrow', 'back_shaded_pvrow']
    irradiance_comp = ['direct']
    irradiance_comp_absorbed = ['direct_absorbed']

    def __init__(self, rho_front=0.01, rho_back=0.03, module_transparency=0.,
                 module_spacing_ratio=0., faoi_fn_front=None,
                 faoi_fn_back=None):
        """Inicializa los valores del modelo de irradiancia que se guardarán más adelante.

         Parámetros
         ----------
         rho_front: flotante, opcional
             Reflectividad del lado frontal de las filas PV (predeterminado = 0,01)
         rho_back: flotante, opcional
             Reflectividad de la parte posterior de las filas PV (predeterminado = 0,03)
         module_transparency: flotante, opcional
             Módulo de transparencia (de 0 a 1), que dejará pasar algo de luz directa.
             pasar a través de los módulos fotovoltaicos en las filas fotovoltaicas y llegar a la zona sombreada
             suelo (predeterminado = 0., completamente oPaco)
         module_spacing_ratio: flotante, opcional
             Relación de espaciado de módulos (de 0 a 1), que es la relación del área
             cubierto por el espacio entre los módulos fotovoltaicos sobre el área total de la
             filas fotovoltaicas, y que determina cuánta luz directa llegará a la
             suelo sombreado a través de las filas fotovoltaicas
             (Predeterminado = 0., sin ningún espacio)
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
         """
        self.direct = dict.fromkeys(self.cats)
        self.total_perez = dict.fromkeys(self.cats)
        self.faoi_front = dict.fromkeys(self.irradiance_comp)
        self.faoi_back = dict.fromkeys(self.irradiance_comp)
        self.faoi_ground = None
        self.module_transparency = module_transparency
        self.module_spacing_ratio = module_spacing_ratio
        self.rho_front = rho_front
        self.rho_back = rho_back
        # Treatment of faoi functions
        faoi_fn_front = (faoi_fn_front.faoi if hasattr(faoi_fn_front, 'faoi')
                         else faoi_fn_front)
        faoi_fn_back = (faoi_fn_back.faoi if hasattr(faoi_fn_back, 'faoi')
                        else faoi_fn_back)
        self.faoi_fn_front = faoi_fn_front
        self.faoi_fn_back = faoi_fn_back
        # The following will be updated at fitting time
        self.albedo = None
        self.GHI = None
        self.dhi = None
        self.isotropic_luminance = None

    def fit(self, timestamps, dni, dhi, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth, albedo,
            ghi=None):
        """Utilice la vectorización para calcular los valores utilizados para el isotrópico
         modelo de irradiancia.

         Parámetros
         ----------
         marcas de tiempo: tipo matriz
             Lista de marcas de tiempo de la simulación.
         dni: tipo matriz
             Valores de irradiancia normal directa [W/m2]
         dhi: tipo matriz
             Valores de irradiancia horizontal difusa [W/m2]
         solar_zenith: similar a una matriz
             Ángulos cenital solares [grados]
         solar_azimuth: similar a una matriz
             Ángulos de azimut solar [grados]
         Surface_tilt: similar a una matriz
             Ángulos de inclinación de la superficie, de 0 a 180 [grados]
         Surface_azimuth: similar a una matriz
             Ángulos de azimut de superficie [grados]
         albedo: similar a una matriz
             Valores de albedo (o reflectividad del suelo)
         ghi: tipo matriz, opcional
             La irradiancia horizontal global [W/m2], si no se proporciona, se
             calculado a partir de dni y dhi (Predeterminado = Ninguno)
         """
        # Make sure getting array-like values
        if np.isscalar(dni):
            timestamps = [timestamps]
            dni = np.array([dni])
            dhi = np.array([dhi])
            solar_zenith = np.array([solar_zenith])
            solar_azimuth = np.array([solar_azimuth])
            surface_tilt = np.array([surface_tilt])
            surface_azimuth = np.array([surface_azimuth])
            ghi = None if ghi is None else np.array([ghi])
        # Length of arrays
        n = len(dni)
        # Make sure that albedo is a vector
        albedo = albedo * np.ones(n) if np.isscalar(albedo) else albedo

        # Save and calculate total POA values from Perez model
        ghi = dni * cosd(solar_zenith) + dhi if ghi is None else ghi
        self.GHI = ghi
        self.dhi = dhi
        self.n_steps = n
        perez_front_pvrow = get_total_irradiance(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            dni, ghi, dhi, albedo=albedo)

        # Save diffuse light
        self.isotropic_luminance = dhi
        self.albedo = albedo

        # dni seen by ground illuminated surfaces
        self.direct['ground_illum'] = dni * cosd(solar_zenith)
        self.direct['ground_shaded'] = (
            # Direct light through PV modules spacing
            self.direct['ground_illum'] * self.module_spacing_ratio
            # Direct light through PV modules, by transparency
            + self.direct['ground_illum'] * (1. - self.module_spacing_ratio)
            * self.module_transparency)

        # Calculate AOI on front pvrow using xm_solarlib implementation
        aoi_front_pvrow = aoi_function(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
        aoi_back_pvrow = 180. - aoi_front_pvrow

        # calculate faoi modifiers
        self._calculate_faoi_modifiers(aoi_front_pvrow, aoi_back_pvrow,
                                       surface_tilt, albedo)

        # dni seen by pvrow illuminated surfaces
        front_is_illum = aoi_front_pvrow <= 90
        # direct
        self.direct['front_illum_pvrow'] = np.where(
            front_is_illum, dni * cosd(aoi_front_pvrow), 0.)
        self.direct['front_shaded_pvrow'] = (
            # Direct light through PV modules spacing
            self.direct['front_illum_pvrow'] * self.module_spacing_ratio
            # Direct light through PV modules, by transparency
            + self.direct['front_illum_pvrow']
            * (1. - self.module_spacing_ratio)
            * self.module_transparency)
        self.direct['back_illum_pvrow'] = np.where(
            ~front_is_illum, dni * cosd(aoi_back_pvrow), 0.)
        self.direct['back_shaded_pvrow'] = (
            # Direct light through PV modules spacing
            self.direct['back_illum_pvrow'] * self.module_spacing_ratio
            # Direct light through PV modules, by transparency
            + self.direct['back_illum_pvrow']
            * (1. - self.module_spacing_ratio)
            * self.module_transparency)
        # perez
        self.total_perez['front_illum_pvrow'] = perez_front_pvrow['poa_global']
        self.total_perez['front_shaded_pvrow'] = (
            self.total_perez['front_illum_pvrow']
            - self.direct['front_illum_pvrow'])
        self.total_perez['ground_shaded'] = (self.dhi
                                             + self.direct['ground_shaded'])

    def transform(self, pvarray):
        """Aplicar valores de irradiancia calculados a la serie temporal del conjunto fotovoltaico
         Geometrías: asigna valores como parámetros a superficies de series temporales.

         Parámetros
         ----------
         pvarray: objeto de matriz fotovoltaica
             Conjunto fotovoltaico sobre el que se aplicarán los valores de irradiancia calculados
         """

        # Prepare variables
        n_steps = self.n_steps
        rho_front = self.rho_front * np.ones(n_steps)
        inv_rho_front = 1. / rho_front
        rho_back = self.rho_back * np.ones(n_steps)
        inv_rho_back = 1. / rho_back

        # Transform timeseries ground
        pvarray.ts_ground.update_illum_params(
            {'direct': self.direct['ground_illum'],
             'rho': self.albedo,
             'inv_rho': 1. / self.albedo,
             'total_perez': self.gnd_illum,
             'direct_absorbed': self.faoi_ground * self.direct['ground_illum']
             })
        pvarray.ts_ground.update_shaded_params(
            {'direct': self.direct['ground_shaded'],
             'rho': self.albedo,
             'inv_rho': 1. / self.albedo,
             'total_perez': self.gnd_shaded,
             'direct_absorbed': self.faoi_ground * self.direct['ground_shaded']
             })

        for ts_pvrow in pvarray.ts_pvrows:
            # Front
            for ts_seg in ts_pvrow.front.list_segments:
                ts_seg.illum.update_params(
                    {'direct': self.direct['front_illum_pvrow'],
                     'rho': rho_front,
                     'inv_rho': inv_rho_front,
                     'total_perez': self.pvrow_illum,
                     'direct_absorbed': self.faoi_front['direct']
                     * self.direct['front_illum_pvrow']})
                ts_seg.shaded.update_params(
                    {'direct': self.direct['front_shaded_pvrow'],
                     'rho': rho_front,
                     'inv_rho': inv_rho_front,
                     'total_perez': self.pvrow_shaded,
                     'direct_absorbed': self.faoi_front['direct']
                     * self.direct['front_shaded_pvrow']})
            # Back
            for ts_seg in ts_pvrow.back.list_segments:
                ts_seg.illum.update_params(
                    {'direct': self.direct['back_illum_pvrow'],
                     'rho': rho_back,
                     'inv_rho': inv_rho_back,
                     'total_perez': np.zeros(n_steps),
                     'direct_absorbed': self.faoi_back['direct']
                     * self.direct['back_illum_pvrow']})
                ts_seg.shaded.update_params(
                    {'direct': self.direct['back_shaded_pvrow'],
                     'rho': rho_back,
                     'inv_rho': inv_rho_back,
                     'total_perez': np.zeros(n_steps),
                     'direct_absorbed': self.faoi_back['direct']
                     * self.direct['back_shaded_pvrow']})

    def get_full_modeling_vectors(self, pvarray, idx):
        """Obtener los vectores de modelado utilizados en cálculos matriciales de matemáticas.
         modelo.

         Parámetros
         ----------
         pvarray: objeto de matriz fotovoltaica
             Conjunto fotovoltaico sobre el que se aplicarán los valores de irradiancia calculados
         idx: int, opcional
             Índice de los valores de irradiancia a aplicar al campo fotovoltaico (en el
             valores completos de la serie temporal)

         Devoluciones
         -------
         irradiance_vec: matriz numerosa
             Lista de valores resumidos de irradiancia no reflectante para todas las superficies
             y cielo
         rho_vec: matriz numerosa
             Lista de valores de reflectividad para todas las superficies y el cielo.
         invrho_vec: matriz numerosa
             Lista de reflectividad inversa para todas las superficies y el cielo.
         total_perez_vec: matriz numerosa
             Lista de valores de irradiancia transpuesta total de Pérez para todas las superficies
             y cielo
         """
        # Sum up the necessary parameters to form the irradiance vector
        irradiance_vec, rho_vec, inv_rho_vec, total_perez_vec = \
            self.get_modeling_vectors(pvarray)
        # Add sky values
        irradiance_vec.append(self.isotropic_luminance[idx])
        total_perez_vec.append(self.isotropic_luminance[idx])
        rho_vec.append(SKY_REFLECTIVITY_DUMMY)
        inv_rho_vec.append(SKY_REFLECTIVITY_DUMMY)

        return np.array(irradiance_vec), np.array(rho_vec), \
            np.array(inv_rho_vec), np.array(total_perez_vec)

    def get_full_ts_modeling_vectors(self, pvarray):
        """Obtener los vectores de modelado utilizados en los cálculos matriciales de
         el modelo matemático, incluidos los valores del cielo.

         Parámetros
         ----------
         pvarray: objeto de matriz fotovoltaica
             Conjunto fotovoltaico sobre el que se aplicarán los valores de irradiancia calculados

         Devoluciones
         -------
         irradiance_mat: np.ndarray
             Lista de valores resumidos de irradiancia no reflectante para todas las superficies
             y cielo. Dimensión = [n_superficies + 1, n_timesteps]
         rho_mat: np.ndarray
             Lista de valores de reflectividad para todas las superficies y el cielo.
             Dimensión = [n_superficies + 1, n_timesteps]
         invrho_mat: np.ndarray
             Lista de reflectividad inversa para todas las superficies y cielo.
             Dimensión = [n_superficies + 1, n_timesteps]
         total_perez_mat: np.ndarray
             Lista de valores de irradiancia transpuesta total de Pérez para todas las superficies
             y cielo. Dimensión = [n_superficies + 1, n_timesteps]
         """
        # Sum up the necessary parameters to form the irradiance vector
        irradiance_mat, rho_mat, inv_rho_mat, total_perez_mat = \
            self.get_ts_modeling_vectors(pvarray)
        # Add sky values
        irradiance_mat.append(self.isotropic_luminance)
        total_perez_mat.append(self.isotropic_luminance)
        rho_mat.append(SKY_REFLECTIVITY_DUMMY * np.ones(pvarray.n_states))
        inv_rho_mat.append(SKY_REFLECTIVITY_DUMMY * np.ones(pvarray.n_states))

        return np.array(irradiance_mat), np.array(rho_mat), \
            np.array(inv_rho_mat), np.array(total_perez_mat)

    @property
    def gnd_shaded(self):
        """Incidente de irradiancia de serie temporal total en áreas sombreadas del suelo"""
        return self.total_perez['ground_shaded']

    @property
    def gnd_illum(self):
        """Incidente de irradiancia de serie temporal total en áreas terrestres iluminadas"""
        return self.GHI

    @property
    def pvrow_shaded(self):
        """Incidencia de irradiancia de serie temporal total en el frente de la fila fotovoltaica iluminada
         áreas y calculadas por transposición de Pérez"""
        return self.total_perez['front_shaded_pvrow']

    @property
    def pvrow_illum(self):
        """Incidente de irradiancia de serie temporal total en el frente de la fila fotovoltaica sombreado
         áreas y calculadas por transposición de Pérez"""
        return self.total_perez['front_illum_pvrow']

    @property
    def sky_luminance(self):
        """Luminancia isotrópica total del cielo en series temporales"""
        return self.isotropic_luminance

    def _calculate_faoi_modifiers(self, aoi_front_pvrow, aoi_back_pvrow,
                                  surface_tilt, albedo):
        """Calcule los valores del modificador fAOI para todos los tipos de superficie: fila PV frontal,
         fila fotovoltaica posterior y tierra.

         Parámetros
         ----------
         aoi_front_pvrow: np.ndarray
             Valores del ángulo de incidencia de la luz directa para la parte frontal de la fila fotovoltaica [grados]
         aoi_back_pvrow: np.ndarray
             Valores del ángulo de incidencia de la luz directa para la parte posterior de la fila fotovoltaica [grados]
         inclinación_superficie: np.ndarray
             Inclinación de la superficie de las superficies de las filas fotovoltaicas, medida desde la horizontal,
             que van de 0 a 180 [grados]
         albedo: np.ndarray
             Valores de albedo terrestre
         """
        # Need to update aoi measurement: input measured from normal,
        # but functions require to measure it from surface horizontal
        aoi_front = np.abs(90. - aoi_front_pvrow)
        aoi_back = np.abs(90. - aoi_back_pvrow)
        # --- front
        self.faoi_front['direct'] = (
            self.faoi_fn_front(aoi_front)
            if self.faoi_fn_front is not None
            else (1. - self.rho_front) * np.ones_like(surface_tilt))

        # --- back
        self.faoi_back['direct'] = (
            self.faoi_fn_back(aoi_back)
            if self.faoi_fn_back is not None
            else (1. - self.rho_back) * np.ones_like(surface_tilt))

        # --- ground
        self.faoi_ground = (1. - albedo)


class HybridPerezOrdered(BaseModel):
    """El modelo se basa en el modelo de luz difusa de Pérez y
     aplicado a pvfactors :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray`
     objetos.
     El modelo aplica irradiancia directa, circunsolar y de horizonte a la energía fotovoltaica.
     superficies de la matriz.
     """

    params = ['rho', 'inv_rho', 'direct', 'isotropic', 'circumsolar',
              'horizon', 'reflection']
    cats = ['ground_illum', 'ground_shaded', 'front_illum_pvrow',
            'back_illum_pvrow', 'front_shaded_pvrow', 'back_shaded_pvrow']
    irradiance_comp = ['direct', 'circumsolar', 'horizon']
    irradiance_comp_absorbed = ['direct_absorbed', 'circumsolar_absorbed',
                                'horizon_absorbed']

    def __init__(self, horizon_band_angle=DEFAULT_HORIZON_BAND_ANGLE,
                 circumsolar_angle=DEFAULT_CIRCUMSOLAR_ANGLE,
                 circumsolar_model='uniform_disk', rho_front=0.01,
                 rho_back=0.03, module_transparency=0.,
                 module_spacing_ratio=0., faoi_fn_front=None,
                 faoi_fn_back=None):
        """Inicializa los valores del modelo de irradiancia que se guardarán más adelante.

         Parámetros
         ----------
         horizonte_band_angle: flotante, opcional
             Ancho de la banda del horizonte en [grados] (Predeterminado =
             DEFAULT_HORIZON_BAND_ANGLE)
         circumsolar_angle: flotante, opcional
             Diámetro del área circunsolar en [grados] (Por defecto =
             DEFAULT_CIRCUMSOLAR_ANGLE)
         modelo_circunsolar: str
             Modelo de sombreado circunsolar a utilizar (predeterminado = 'uniform_disk')
         rho_front: flotante, opcional
             Reflectividad del lado frontal de las filas PV (predeterminado = 0,01)
         rho_back: flotante, opcional
             Reflectividad de la parte posterior de las filas PV (predeterminado = 0,03)
         module_transparency: flotante, opcional
             Módulo de transparencia (de 0 a 1), que dejará pasar algo de luz directa.
             pasar a través de los módulos fotovoltaicos en las filas fotovoltaicas y llegar a la zona sombreada
             suelo (predeterminado = 0., completamente oPaco)
         module_spacing_ratio: flotante, opcional
             Relación de espaciado de módulos (de 0 a 1), que es la relación del área
             cubierto por el espacio entre los módulos fotovoltaicos sobre el área total de la
             filas fotovoltaicas, y que determina cuánta luz directa llegará a la
             suelo sombreado a través de las filas fotovoltaicas
             (Predeterminado = 0., sin ningún espacio)
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
         """
        self.direct = dict.fromkeys(self.cats)
        self.circumsolar = dict.fromkeys(self.cats)
        self.horizon = dict.fromkeys(self.cats)
        self.total_perez = dict.fromkeys(self.cats)
        self.faoi_front = dict.fromkeys(self.irradiance_comp)
        self.faoi_back = dict.fromkeys(self.irradiance_comp)
        self.faoi_ground = None
        self.horizon_band_angle = horizon_band_angle
        self.circumsolar_angle = circumsolar_angle
        self.circumsolar_model = circumsolar_model
        self.rho_front = rho_front
        self.rho_back = rho_back
        self.module_transparency = module_transparency
        self.module_spacing_ratio = module_spacing_ratio
        # Treatment of faoi functions
        faoi_fn_front = (faoi_fn_front.faoi if hasattr(faoi_fn_front, 'faoi')
                         else faoi_fn_front)
        faoi_fn_back = (faoi_fn_back.faoi if hasattr(faoi_fn_back, 'faoi')
                        else faoi_fn_back)
        self.faoi_fn_front = faoi_fn_front
        self.faoi_fn_back = faoi_fn_back
        # Attributes that will be updated at fitting time
        self.albedo = None
        self.GHI = None
        self.dni = None
        self.n_steps = None
        self.isotropic_luminance = None

    def fit(self, timestamps, dni, dhi, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth, albedo,
            ghi=None):
        """Utilice la vectorización para calcular los valores utilizados para el híbrido Pérez
         modelo de irradiancia.

         Parámetros
         ----------
         marcas de tiempo: tipo matriz
             Lista de marcas de tiempo de la simulación.
         dni: tipo matriz
             Valores de irradiancia normal directa [W/m2]
         dhi: tipo matriz
             Valores de irradiancia horizontal difusa [W/m2]
         solar_zenith: similar a una matriz
             Ángulos cenital solares [grados]
         solar_azimuth: similar a una matriz
             Ángulos de azimut solar [grados]
         Surface_tilt: similar a una matriz
             Ángulos de inclinación de la superficie, de 0 a 180 [grados]
         Surface_azimuth: similar a una matriz
             Ángulos de azimut de superficie [grados]
         albedo: similar a una matriz
             Valores de albedo (o reflectividad del suelo)
         ghi: tipo matriz, opcional
             La irradiancia horizontal global [W/m2], si no se proporciona, se
             calculado a partir de dni y dhi (Predeterminado = Ninguno)
         """
        # Make sure getting array-like values
        if np.isscalar(dni):
            timestamps = [timestamps]
            dni = np.array([dni])
            dhi = np.array([dhi])
            solar_zenith = np.array([solar_zenith])
            solar_azimuth = np.array([solar_azimuth])
            surface_tilt = np.array([surface_tilt])
            surface_azimuth = np.array([surface_azimuth])
            ghi = None if ghi is None else np.array([ghi])
        # Length of arrays
        n = len(dni)
        # Make sure that albedo is a vector
        albedo = albedo * np.ones(n) if np.isscalar(albedo) else albedo

        # Calculate terms from Perez model
        luminance_isotropic, luminance_circumsolar, poa_horizon, \
            poa_circumsolar_front, poa_circumsolar_back, \
            aoi_front_pvrow, aoi_back_pvrow = \
            self._calculate_luminance_poa_components(
                timestamps, dni, dhi, solar_zenith, solar_azimuth,
                surface_tilt, surface_azimuth)

        # calculate faoi modifiers
        self._calculate_faoi_modifiers(aoi_front_pvrow, aoi_back_pvrow,
                                       surface_tilt, albedo)

        # Save and calculate total POA values from Perez model
        ghi = dni * cosd(solar_zenith) + dhi if ghi is None else ghi
        self.GHI = ghi
        self.dhi = dhi
        self.n_steps = n
        perez_front_pvrow = get_total_irradiance(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            dni, ghi, dhi, albedo=albedo)
        total_perez_front_pvrow = perez_front_pvrow['poa_global']

        # Save isotropic luminance
        self.isotropic_luminance = luminance_isotropic
        self.albedo = albedo

        # Ground surfaces
        self.direct['ground_illum'] = dni * cosd(solar_zenith)
        self.direct['ground_shaded'] = (
            # Direct light through PV modules spacing
            self.direct['ground_illum'] * self.module_spacing_ratio
            # Direct light through PV modules, by transparency
            + self.direct['ground_illum'] * (1. - self.module_spacing_ratio)
            * self.module_transparency)
        self.circumsolar['ground_illum'] = luminance_circumsolar
        self.circumsolar['ground_shaded'] = (
            # Circumsolar light through PV modules spacing
            self.circumsolar['ground_illum'] * self.module_spacing_ratio
            # Circumsolar light through PV modules, by transparency
            + self.circumsolar['ground_illum']
            * (1. - self.module_spacing_ratio) * self.module_transparency)
        self.horizon['ground'] = np.zeros(n)

        # PV row surfaces
        front_is_illum = aoi_front_pvrow <= 90
        # direct
        self.direct['front_illum_pvrow'] = np.where(
            front_is_illum, dni * cosd(aoi_front_pvrow), 0.)
        self.direct['front_shaded_pvrow'] = (
            # Direct light through PV modules spacing
            self.direct['front_illum_pvrow'] * self.module_spacing_ratio
            # Direct light through PV modules, by transparency
            + self.direct['front_illum_pvrow']
            * (1. - self.module_spacing_ratio)
            * self.module_transparency)
        self.direct['back_illum_pvrow'] = np.where(
            ~front_is_illum, dni * cosd(aoi_back_pvrow), 0.)
        self.direct['back_shaded_pvrow'] = (
            # Direct light through PV modules spacing
            self.direct['back_illum_pvrow'] * self.module_spacing_ratio
            # Direct light through PV modules, by transparency
            + self.direct['back_illum_pvrow']
            * (1. - self.module_spacing_ratio)
            * self.module_transparency)
        # circumsolar
        self.circumsolar['front_illum_pvrow'] = np.where(
            front_is_illum, poa_circumsolar_front, 0.)
        self.circumsolar['front_shaded_pvrow'] = (
            # Direct light through PV modules spacing
            self.circumsolar['front_illum_pvrow'] * self.module_spacing_ratio
            # Direct light through PV modules, by transparency
            + self.circumsolar['front_illum_pvrow']
            * (1. - self.module_spacing_ratio)
            * self.module_transparency)
        self.circumsolar['back_illum_pvrow'] = np.where(
            ~front_is_illum, poa_circumsolar_back, 0.)
        self.circumsolar['back_shaded_pvrow'] = (
            # Direct light through PV modules spacing
            self.circumsolar['back_illum_pvrow'] * self.module_spacing_ratio
            # Direct light through PV modules, by transparency
            + self.circumsolar['back_illum_pvrow']
            * (1. - self.module_spacing_ratio)
            * self.module_transparency)
        # horizon
        self.horizon['front_pvrow'] = poa_horizon
        self.horizon['back_pvrow'] = poa_horizon
        # perez
        self.total_perez['front_illum_pvrow'] = total_perez_front_pvrow
        self.total_perez['front_shaded_pvrow'] = (
            total_perez_front_pvrow
            - self.direct['front_illum_pvrow']
            - self.circumsolar['front_illum_pvrow']
            + self.direct['front_shaded_pvrow']
            + self.circumsolar['front_shaded_pvrow']
        )
        self.total_perez['ground_shaded'] = (
            dhi
            - self.circumsolar['ground_illum']
            + self.circumsolar['ground_shaded']
            + self.direct['ground_shaded'])
        self.total_perez['ground_illum'] = ghi
        self.total_perez['sky'] = luminance_isotropic

    def transform(self, pvarray):
        """Aplicar valores de irradiancia calculados a la serie temporal del conjunto fotovoltaico
         Geometrías: asigna valores como parámetros a superficies de series temporales.

         Parámetros
         ----------
         pvarray: objeto de matriz fotovoltaica
             Conjunto fotovoltaico sobre el que se aplicarán los valores de irradiancia calculados
         """

        # Prepare variables
        n_steps = self.n_steps
        ts_pvrows = pvarray.ts_pvrows
        tilted_to_left = pvarray.rotation_vec > 0
        rho_front = self.rho_front * np.ones(n_steps)
        inv_rho_front = 1. / rho_front
        rho_back = self.rho_back * np.ones(n_steps)
        inv_rho_back = 1. / rho_back

        # Transform timeseries ground
        pvarray.ts_ground.update_illum_params({
            'direct': self.direct['ground_illum'],
            'circumsolar': self.circumsolar['ground_illum'],
            'horizon': np.zeros(n_steps),
            'rho': self.albedo,
            'inv_rho': 1. / self.albedo,
            'total_perez': self.gnd_illum,
            'direct_absorbed': self.faoi_ground * self.direct['ground_illum'],
            'circumsolar_absorbed': self.faoi_ground
            * self.circumsolar['ground_illum'],
            'horizon_absorbed': np.zeros(n_steps)})
        pvarray.ts_ground.update_shaded_params({
            'direct': self.direct['ground_shaded'],
            'circumsolar': self.circumsolar['ground_shaded'],
            'horizon': np.zeros(n_steps),
            'rho': self.albedo,
            'inv_rho': 1. / self.albedo,
            'total_perez': self.gnd_shaded,
            'direct_absorbed': self.faoi_ground * self.direct['ground_shaded'],
            'circumsolar_absorbed': self.faoi_ground
            * self.circumsolar['ground_shaded'],
            'horizon_absorbed': np.zeros(n_steps)})

        # Transform timeseries PV rows
        for idx_pvrow, ts_pvrow in enumerate(ts_pvrows):
            # Front
            for ts_seg in ts_pvrow.front.list_segments:
                ts_seg.illum.update_params({
                    'direct': self.direct['front_illum_pvrow'],
                    'circumsolar': self.circumsolar['front_illum_pvrow'],
                    'horizon': self.horizon['front_pvrow'],
                    'rho': rho_front,
                    'inv_rho': inv_rho_front,
                    'total_perez': self.pvrow_illum,
                    'direct_absorbed': self.faoi_front['direct']
                    * self.direct['front_illum_pvrow'],
                    'circumsolar_absorbed': self.faoi_front['circumsolar']
                    * self.circumsolar['front_illum_pvrow'],
                    'horizon_absorbed': self.faoi_front['horizon']
                    * self.horizon['front_pvrow']})
                ts_seg.shaded.update_params({
                    'direct': self.direct['front_shaded_pvrow'],
                    'circumsolar': self.circumsolar['front_shaded_pvrow'],
                    'horizon': self.horizon['front_pvrow'],
                    'rho': rho_front,
                    'inv_rho': inv_rho_front,
                    'total_perez': self.pvrow_shaded,
                    'direct_absorbed': self.faoi_front['direct']
                    * self.direct['front_shaded_pvrow'],
                    'circumsolar_absorbed': self.faoi_front['circumsolar']
                    * self.circumsolar['front_shaded_pvrow'],
                    'horizon_absorbed': self.faoi_front['horizon']
                    * self.horizon['front_pvrow']})
            # Back: apply back surface horizon shading
            for ts_seg in ts_pvrow.back.list_segments:
                # Illum
                # In ordered pv arrays, there should only be 1 surface -> 0
                centroid_illum = ts_seg.illum.list_ts_surfaces[0].centroid
                hor_shd_pct_illum = self._calculate_horizon_shading_pct_ts(
                    ts_pvrows, centroid_illum, idx_pvrow, tilted_to_left,
                    is_back_side=True)
                ts_seg.illum.update_params({
                    'direct': self.direct['back_illum_pvrow'],
                    'circumsolar': self.circumsolar['back_illum_pvrow'],
                    'horizon': self.horizon['back_pvrow'] *
                    (1. - hor_shd_pct_illum / 100.),
                    'horizon_unshaded': self.horizon['back_pvrow'],
                    'horizon_shd_pct': hor_shd_pct_illum,
                    'rho': rho_back,
                    'inv_rho': inv_rho_back,
                    'total_perez': np.zeros(n_steps),
                    'direct_absorbed': self.faoi_back['direct']
                    * self.direct['back_illum_pvrow'],
                    'circumsolar_absorbed': self.faoi_back['circumsolar']
                    * self.circumsolar['back_illum_pvrow'],
                    'horizon_absorbed': self.faoi_back['horizon']
                    * self.horizon['back_pvrow']
                    * (1. - hor_shd_pct_illum / 100.)})
                # Back
                # In ordered pv arrays, there should only be 1 surface -> 0
                centroid_shaded = ts_seg.shaded.list_ts_surfaces[0].centroid
                hor_shd_pct_shaded = self._calculate_horizon_shading_pct_ts(
                    ts_pvrows, centroid_shaded, idx_pvrow, tilted_to_left,
                    is_back_side=True)
                ts_seg.shaded.update_params({
                    'direct': self.direct['back_shaded_pvrow'],
                    'circumsolar': self.circumsolar['back_shaded_pvrow'],
                    'horizon': self.horizon['back_pvrow'] *
                    (1. - hor_shd_pct_shaded / 100.),
                    'horizon_unshaded': self.horizon['back_pvrow'],
                    'horizon_shd_pct': hor_shd_pct_shaded,
                    'rho': rho_back,
                    'inv_rho': inv_rho_back,
                    'total_perez': np.zeros(n_steps),
                    'direct_absorbed': self.faoi_back['direct']
                    * self.direct['back_shaded_pvrow'],
                    'circumsolar_absorbed': self.faoi_back['circumsolar']
                    * self.circumsolar['back_shaded_pvrow'],
                    'horizon_absorbed': self.faoi_back['horizon']
                    * self.horizon['back_pvrow']
                    * (1. - hor_shd_pct_shaded / 100.)})

    def get_full_modeling_vectors(self, pvarray, idx):
        """Obtener los vectores de modelado utilizados en cálculos matriciales de matemáticas.
         modelo.

         Parámetros
         ----------
         pvarray: objeto de matriz fotovoltaica
             Conjunto fotovoltaico sobre el que se aplicarán los valores de irradiancia calculados
         idx: int, opcional
             Índice de los valores de irradiancia a aplicar al campo fotovoltaico (en el
             valores completos de la serie temporal)

         Devoluciones
         -------
         irradiance_vec: matriz numerosa
             Lista de valores resumidos de irradiancia no reflectante para todas las superficies
             y cielo
         rho_vec: matriz numerosa
             Lista de valores de reflectividad para todas las superficies y el cielo.
         invrho_vec: matriz numerosa
             Lista de reflectividad inversa para todas las superficies y el cielo.
         total_perez_vec: matriz numerosa
             Lista de valores de irradiancia transpuesta total de Pérez para todas las superficies
             y cielo
         """

        # Sum up the necessary parameters to form the irradiance vector
        irradiance_vec, rho_vec, inv_rho_vec, total_perez_vec = self.get_modeling_vectors(pvarray)
        # Add sky values
        irradiance_vec.append(self.isotropic_luminance[idx])
        total_perez_vec.append(self.isotropic_luminance[idx])
        rho_vec.append(SKY_REFLECTIVITY_DUMMY)
        inv_rho_vec.append(SKY_REFLECTIVITY_DUMMY)

        return np.array(irradiance_vec), np.array(rho_vec), \
            np.array(inv_rho_vec), np.array(total_perez_vec)

    def get_full_ts_modeling_vectors(self, pvarray):
        """Obtener los vectores de modelado utilizados en los cálculos matriciales de
         el modelo matemático, incluidos los valores del cielo.

         Parámetros
         ----------
         pvarray: objeto de matriz fotovoltaica
             Conjunto fotovoltaico sobre el que se aplicarán los valores de irradiancia calculados

         Devoluciones
         -------
         irradiance_mat: np.ndarray
             Lista de valores resumidos de irradiancia no reflectante para todas las superficies
             y cielo. Dimensión = [n_superficies + 1, n_timesteps]
         rho_mat: np.ndarray
             Lista de valores de reflectividad para todas las superficies y el cielo.
             Dimensión = [n_superficies + 1, n_timesteps]
         invrho_mat: np.ndarray
             Lista de reflectividad inversa para todas las superficies y cielo.
             Dimensión = [n_superficies + 1, n_timesteps]
         total_perez_mat: np.ndarray
             Lista de valores de irradiancia transpuesta total de Pérez para todas las superficies
             y cielo. Dimensión = [n_superficies + 1, n_timesteps]
         """
        # Sum up the necessary parameters to form the irradiance vector
        irradiance_mat, rho_mat, inv_rho_mat, total_perez_mat = \
            self.get_ts_modeling_vectors(pvarray)
        # Add sky values
        irradiance_mat.append(self.isotropic_luminance)
        total_perez_mat.append(self.isotropic_luminance)
        rho_mat.append(SKY_REFLECTIVITY_DUMMY * np.ones(pvarray.n_states))
        inv_rho_mat.append(SKY_REFLECTIVITY_DUMMY * np.ones(pvarray.n_states))

        return np.array(irradiance_mat), np.array(rho_mat), \
            np.array(inv_rho_mat), np.array(total_perez_mat)

    @property
    def gnd_shaded(self):
        """Incidente de irradiancia de serie temporal total en áreas sombreadas del suelo"""
        return self.total_perez['ground_shaded']

    @property
    def gnd_illum(self):
        """Incidente de irradiancia de serie temporal total en áreas terrestres iluminadas"""
        return self.total_perez['ground_illum']

    @property
    def pvrow_shaded(self):
        """Incidencia de irradiancia de serie temporal total en el frente de la fila fotovoltaica iluminada
         áreas y calculadas por transposición de Pérez"""
        return self.total_perez['front_shaded_pvrow']

    @property
    def pvrow_illum(self):
        """Incidente de irradiancia de serie temporal total en el frente de la fila fotovoltaica sombreado
         áreas y calculadas por transposición de Pérez"""
        return self.total_perez['front_illum_pvrow']

    @property
    def sky_luminance(self):
        """Luminancia isotrópica total del cielo en series temporales"""
        return self.isotropic_luminance

    def _calculate_horizon_shading_pct_ts(self, ts_pvrows, ts_point_coords,
                                          pvrow_idx, tilted_to_left,
                                          is_back_side=True):
        """Calcular el porcentaje de sombreado de la banda del horizonte en las superficies del orden.
         Matriz fotovoltaica, de forma vectorizada.

         Parámetros
         ----------
         ts_pvrows: lista de :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
             Lista de filas fotovoltaicas de serie temporal en el conjunto fotovoltaico
         ts_point_coords: :py:clase:`~pvfactors.geometry.timeseries.TsPointCoords`
             Coordenadas de serie temporal del punto que sufre sombreado en la banda del horizonte
         pvrow_idx:int
             Índice de la fila de PV en la que se encuentra el punto anterior
         inclinado_a_izquierda: lista de bool
             Banderas que indican cuándo las filas de PV están estrictamente inclinadas hacia la izquierda
         is_back_side: booleano
             Bandera que indica si el punto está ubicado en la parte posterior de la fila PV

         Devoluciones
         -------
         horizonte_shading_pct: np.ndarray
             Vector porcentual de sombreado de irradiancia de la banda del horizonte
             (de 0 a 100)
         """
        n_pvrows = len(ts_pvrows)
        if pvrow_idx == 0:
            shading_pct_left = np.zeros_like(tilted_to_left)
        else:
            high_pt_left = ts_pvrows[
                pvrow_idx - 1].full_pvrow_coords.highest_point
            shading_angle_left = np.rad2deg(np.abs(np.arctan(
                (high_pt_left.y - ts_point_coords.y)
                / (high_pt_left.x - ts_point_coords.x))))
            shading_pct_left = calculate_horizon_band_shading(
                shading_angle_left, self.horizon_band_angle)

        if pvrow_idx == (n_pvrows - 1):
            shading_pct_right = np.zeros_like(tilted_to_left)
        else:
            high_pt_right = ts_pvrows[
                pvrow_idx + 1].full_pvrow_coords.highest_point
            shading_angle_right = np.rad2deg(np.abs(np.arctan(
                (high_pt_right.y - ts_point_coords.y)
                / (high_pt_right.x - ts_point_coords.x))))
            shading_pct_right = calculate_horizon_band_shading(
                shading_angle_right, self.horizon_band_angle)

        if is_back_side:
            shading_pct = np.where(tilted_to_left, shading_pct_right,
                                   shading_pct_left)
        else:
            shading_pct = np.where(tilted_to_left, shading_pct_left,
                                   shading_pct_right)

        return shading_pct

    def _calculate_circumsolar_shading_pct(self, surface, idx_neighbor, pvrows,
                                           solar_2d_vector):
        """Método modelo para calcular el sombreado circunsolar en superficies de
         el conjunto fotovoltaico solicitado.

        Parámetros
         ----------
         superficie :py:clase:`~pvfactors.geometry.base.PVSurface` objeto
             Superficie fotovoltaica para la cual se producirá algo de sombreado en la banda del horizonte
         idx_vecino: int
             Índice de la fila PV vecina (puede ser ``Ninguno``)
         pvrows: lista de objetos :py:class:`~pvfactors.geometry.pvrow.PVRow`
             Lista de filas PV en las que se utilizará ``idx_neighbor``
         solar_2d_vector: lista
             Vector solar en la representación del conjunto fotovoltaico 2D

         Devoluciones
         -------
         circ_shading_pct: flotante
             Porcentaje de irradiancia circunsolar sombreada (de 0 a 100)
         """


        if idx_neighbor is not None:
            # Calculate the solar and circumsolar elevation angles in 2D plane
            solar_2d_elevation = np.abs(
                np.arctan(solar_2d_vector[1] / solar_2d_vector[0])
            ) * 180. / np.pi
            lower_angle_circumsolar = (solar_2d_elevation -
                                       self.circumsolar_angle / 2.)
            centroid = surface.centroid
            neighbor_point = pvrows[idx_neighbor].highest_point
            shading_angle = np.abs(np.arctan(
                (neighbor_point.y - centroid.y) /
                (neighbor_point.x - centroid.x))) * 180. / np.pi
            percentage_circ_angle_covered = (shading_angle - lower_angle_circumsolar) \
                / self.circumsolar_angle * 100.
            circ_shading_pct = calculate_circumsolar_shading(
                percentage_circ_angle_covered, model=self.circumsolar_model)

        return circ_shading_pct

    @staticmethod
    def _calculate_luminance_poa_components(
            timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth):
        """Calcule los valores de irradiancia del plano de matriz y de luminancia tipo Pérez.

         Parámetros
         ----------
         marcas de tiempo: tipo matriz
             Lista de marcas de tiempo de la simulación.
         dni: tipo matriz
             Valores de irradiancia normal directa [W/m2]
         dhi: tipo matriz
             Valores de irradiancia horizontal difusa [W/m2]
         solar_zenith: similar a una matriz
             Ángulos cenital solares [grados]
         solar_azimuth: similar a una matriz
             Ángulos de azimut solar [grados]
         Surface_tilt: similar a una matriz
             Ángulos de inclinación de la superficie, de 0 a 180 [grados]
         Surface_azimuth: similar a una matriz
             Ángulos de azimut de superficie [grados]

         Devoluciones
         -------
         luminance_isotropic: matriz numerosa
             Valores de luminancia del área del domo celeste isotrópico
         luminance_circumsolar: matriz numerosa
             Valores de luminancia del área circunsolar.
         poa_horizon: matriz numerosa
             Irradiancia del plano de matriz proveniente de la banda del horizonte e incidente en
             los lados de las filas fotovoltaicas [W/m2]
         poa_circumsolar_front: matriz numerosa
             Irradiancia de plano de matriz procedente del área circunsolar y
             Incidencia en la parte frontal de las filas fotovoltaicas [W/m2]
         poa_circumsolar_back: matriz numerosa
             Irradiancia de plano de matriz procedente del área circunsolar y
             Incidencia en la parte trasera de las filas fotovoltaicas [W/m2]
         aoi_front_pvrow: matriz numerosa
             Ángulo de incidencia de la luz directa en el frente de las filas fotovoltaicas [grados]
         aoi_back_pvrow: matriz numerosa
             Ángulo de incidencia de la luz directa en la parte posterior de las filas fotovoltaicas [grados]
         """

        # Calculations from utils function
        df_inputs = perez_diffuse_luminance(
            timestamps, surface_tilt, surface_azimuth, solar_zenith,
            solar_azimuth, dni, dhi)

        luminance_isotropic = df_inputs.luminance_isotropic.values
        luminance_circumsolar = df_inputs.luminance_circumsolar.values
        poa_horizon = df_inputs.poa_horizon.values
        poa_circumsolar_front = df_inputs.poa_circumsolar.values

        # Calculate AOI on front pvrow using xm_solarlib implementation
        aoi_front_pvrow = aoi_function(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
        aoi_back_pvrow = 180. - aoi_front_pvrow

        # Will be used for back surface adjustments: from Perez model
        vf_circumsolar_backsurface = cosd(aoi_back_pvrow) / cosd(solar_zenith)
        poa_circumsolar_back = (luminance_circumsolar
                                * vf_circumsolar_backsurface)

        # Return only >0 values for poa_horizon
        poa_horizon = np.abs(poa_horizon)

        return luminance_isotropic, luminance_circumsolar, poa_horizon, \
            poa_circumsolar_front, poa_circumsolar_back, \
            aoi_front_pvrow, aoi_back_pvrow

    def _calculate_faoi_modifiers(self, aoi_front_pvrow, aoi_back_pvrow,
                                  surface_tilt, albedo):
        """Calcule los valores del modificador fAOI para todos los tipos de superficie: fila PV frontal,
         fila fotovoltaica posterior y tierra.

         Parámetros
         ----------
         aoi_front_pvrow: np.ndarray
             Valores del ángulo de incidencia de la luz directa para la parte frontal de la fila fotovoltaica [grados]
         aoi_back_pvrow: np.ndarray
             Valores del ángulo de incidencia de la luz directa para la parte posterior de la fila fotovoltaica [grados]
         inclinación_superficie: np.ndarray
             Inclinación de la superficie de las superficies de las filas fotovoltaicas, medida desde la horizontal,
             que van de 0 a 180 [grados]
         albedo: np.ndarray
             Valores de albedo terrestre
         """
        # Need to update aoi measurement: input measured from normal,
        # but functions require to measure it from surface horizontal
        aoi_front = np.abs(90. - aoi_front_pvrow)
        aoi_back = np.abs(90. - aoi_back_pvrow)
        # --- front
        if self.faoi_fn_front is not None:
            faoi_direct_rays = self.faoi_fn_front(aoi_front)
            self.faoi_front['direct'] = faoi_direct_rays
            # Assume that circumsolar is just a point for this
            self.faoi_front['circumsolar'] = faoi_direct_rays
            # Assume horizon band has zero width
            self.faoi_front['horizon'] = self.faoi_fn_front(surface_tilt)
        else:
            faoi_diffuse_front = ((1. - self.rho_front)
                                  * np.ones_like(surface_tilt))
            self.faoi_front['direct'] = faoi_diffuse_front
            self.faoi_front['circumsolar'] = faoi_diffuse_front
            self.faoi_front['horizon'] = faoi_diffuse_front

        # --- back
        if self.faoi_fn_back is not None:
            faoi_direct_rays = self.faoi_fn_back(aoi_back)
            self.faoi_back['direct'] = faoi_direct_rays
            # Assume that circumsolar is just a point for this
            self.faoi_back['circumsolar'] = faoi_direct_rays
            # Assume horizon band has zero width
            self.faoi_back['horizon'] = self.faoi_fn_back(surface_tilt)
        else:
            faoi_diffuse_back = ((1. - self.rho_back)
                                 * np.ones_like(surface_tilt))
            self.faoi_back['direct'] = faoi_diffuse_back
            self.faoi_back['circumsolar'] = faoi_diffuse_back
            self.faoi_back['horizon'] = faoi_diffuse_back

        # --- ground
        self.faoi_ground = (1. - albedo)