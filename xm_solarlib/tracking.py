import numpy as np
import pandas as pd

from xm_solarlib.tools import cosd, sind, acosd, asind
from xm_solarlib import irradiance


def singleaxis(
    apparent_zenith,
    apparent_azimuth,
    axis_tilt=0,
    axis_azimuth=0,
    max_angle=90,
    backtrack=True,
    gcr=2.0 / 7.0,
    cross_axis_tilt=0,
):
    """
    Determine el ángulo de rotación de un seguidor de un solo eje cuando se le proporcionan
    ángulos particulares de cenit y azimut solar.

    Consulta [1]_ para obtener detalles sobre las ecuaciones. Se puede especificar el seguimiento inverso
    y, si es así, se requiere una relación de cobertura del suelo.

    El ángulo de rotación se determina en un sistema de coordenadas diestro. El
    `axis_azimuth` del seguidor define el eje y positivo, el eje x positivo es
    90 grados en sentido horario desde el eje y y paralelo a la superficie de la Tierra,
    y el eje z positivo es normal tanto a los ejes x como y y está orientado hacia el cielo.
    El ángulo de rotación `tracker_theta` es una rotación diestra alrededor del eje y
    en el sistema de coordenadas x, y, z y indica la posición del seguidor con respecto a
    horizontal. Por ejemplo, si el `axis_azimuth` del seguidor es 180 (orientado al sur)
    y `axis_tilt` es cero, entonces un `tracker_theta` de cero es horizontal, un
    `tracker_theta` de 30 grados es una rotación de 30 grados hacia el oeste,
    y un `tracker_theta` de -90 grados es una rotación al plano vertical
    orientado hacia el este.

    Parámetros
    ----------
    apparent_zenith : float, arreglo 1D o Serie
        Ángulos aparentes de cenit solar en grados decimales.

    apparent_azimuth : float, arreglo 1D o Serie
        Ángulos aparentes de azimut solar en grados decimales.

    axis_tilt : float, por defecto 0
        La inclinación del eje de rotación (es decir, el eje y definido por
        ``axis_azimuth``) con respecto a la horizontal.
        ``axis_tilt`` debe ser >= 0 y <= 90. [grado]

    axis_azimuth : float, por defecto 0
        Un valor que denota la dirección de la brújula a lo largo de la cual se encuentra
        el eje de rotación. Medido en grados decimales al este del norte.

    max_angle : float o tupla, por defecto 90
        Un valor que denota el ángulo de rotación máximo, en grados decimales,
        del seguidor de un solo eje desde su posición horizontal (horizontal
        si axis_tilt = 0). Si se proporciona un número flotante, representa el
        ángulo de rotación máximo, y se asume que el ángulo de rotación mínimo es el
        opuesto del ángulo máximo. Si se proporciona una tupla de (min_angle, max_angle),
        representa tanto el ángulo de rotación mínimo como el máximo.

        Una rotación a 'max_angle' es una rotación en sentido contrario a las
        agujas del reloj alrededor del eje y del sistema de coordenadas del seguidor.
        Por ejemplo, para un seguidor con 'axis_azimuth' orientado al sur, una rotación a 'max_angle'
        es hacia el oeste, y una rotación hacia 'min_angle' es en la
        dirección opuesta, hacia el este. Por lo tanto, un max_angle de 180 grados
        (equivalente a max_angle = (-180, 180)) permite que el seguidor alcance
        su capacidad de rotación completa.

    backtrack : bool, por defecto True
        Controla si el seguidor tiene la capacidad de "seguimiento inverso"
        para evitar sombras de fila a fila. False indica que no hay capacidad de
        seguimiento inverso. True indica capacidad de seguimiento inverso.

    gcr : float, por defecto 2.0/7.0
        Un valor que denota la relación de cobertura del suelo de un sistema de
        seguimiento que utiliza seguimiento inverso; es decir, la relación entre el área
        de la superficie del arreglo fotovoltaico y el área total del suelo. Un sistema de
        seguimiento con módulos de 2 metros de ancho, centrados en el eje de seguimiento,
        con 6 metros entre los ejes de seguimiento tiene una relación de cobertura del suelo
        de 2/6=0.333. Si no se proporciona gcr, el valor predeterminado es 2/7. gcr debe ser <=1.

    cross_axis_tilt : float, por defecto 0.0
        El ángulo, con respecto a la horizontal, de la línea formada por la
        intersección entre la pendiente que contiene los ejes del seguidor y un plano
        perpendicular a los ejes del seguidor. El tilt cruzado debe especificarse
        utilizando una convención diestra. Por ejemplo, los seguidores con un
        eje de azimut de 180 grados (dirigido al sur) tendrán un tilt cruzado negativo si
        el plano de los ejes del seguidor desciende hacia el este y un tilt cruzado positivo si
        el plano de los ejes del seguidor desciende hacia el oeste. Usa
        :func:`~pvlib.tracking.calc_cross_axis_tilt` para calcular
        `cross_axis_tilt`. [grados]


    Devuelve
    -------
    dict o DataFrame con las siguientes columnas:
        * `tracker_theta`: El ángulo de rotación del seguidor es una rotación diestra
          definida por `axis_azimuth`.
          tracker_theta = 0 es horizontal. [grados]
        * `aoi`: El ángulo de incidencia de la radiación directa sobre la
          superficie del panel rotado. [grados]
        * `surface_tilt`: El ángulo entre la superficie del panel y la superficie terrestre,
          teniendo en cuenta la rotación del panel. [grados]
        * `surface_azimuth`: El azimut del panel rotado, determinado por
          proyectar el vector normal a la superficie del panel sobre la superficie terrestre. [grados]

    Ver también
    --------
    xm_solarlib.tracking.calc_axis_tilt
    xm_solarlib.tracking.calc_cross_axis_tilt
    xm_solarlib.tracking.calc_surface_orientation

    Referencias
    ----------
    .. [1] Kevin Anderson y Mark Mikofski, "Seguimiento Consciente de la Pendiente para
       Seguidores de Un Eje", Informe Técnico NREL/TP-5K00-76626, Julio 2020.
       https://www.nrel.gov/docs/fy20osti/76626.pdf
    """

    # MATLAB to Python conversion by
    # Will Holmgren (@wholmgren), U. Arizona. March, 2015.

    if isinstance(apparent_zenith, pd.Series):
        index = apparent_zenith.index
    else:
        index = None

    # convertir escalares a arreglos
    apparent_azimuth = np.atleast_1d(apparent_azimuth)
    apparent_zenith = np.atleast_1d(apparent_zenith)

    if apparent_azimuth.ndim > 1 or apparent_zenith.ndim > 1:
        raise ValueError("Input dimensions must not exceed 1")

    # Calcular la posición solar x, y, z usando el sistema de coordenadas como en [1], Ec. 1.

    # NOTA: elevación solar = 90 - cenit solar, luego use identidades trigonométricas:
    # sin(90-x) = cos(x) & cos(90-x) = sin(x)
    sin_zenith = sind(apparent_zenith)
    x = sin_zenith * sind(apparent_azimuth)
    y = sin_zenith * cosd(apparent_azimuth)
    z = cosd(apparent_zenith)

    # Supongamos que el marco de referencia del seguidor es diestro. El eje y positivo es
    # orientado a lo largo del eje de seguimiento; desde el norte, el eje y se gira en sentido horario
    # por el azimut del eje y se inclina desde la horizontal por el eje de seguimiento. El
    # eje x positivo está a 90 grados en sentido horario desde el eje y y es paralelo
    # a la horizontal (por ejemplo, si el eje y es sur, el eje x es oeste); el
    # eje z positivo es normal a los ejes x e y y está orientado hacia arriba.

    # Calcular la posición solar (xp, yp, zp) en el sistema de coordenadas del seguidor usando
    # [1] Ec. 4.

    cos_axis_azimuth = cosd(axis_azimuth)
    sin_axis_azimuth = sind(axis_azimuth)
    cos_axis_tilt = cosd(axis_tilt)
    sin_axis_tilt = sind(axis_tilt)
    xp = x * cos_axis_azimuth - y * sin_axis_azimuth
    # no es necesario calcular y'
    # yp = (x*cos_inclinacion_eje*sin_azimut_eje
    #       + y*cos_inclinacion_eje*cos_azimut_eje
    #       - z*sin_inclinacion_eje)
    zp = (
        x * sin_axis_tilt * sin_axis_azimuth
        + y * sin_axis_tilt * cos_axis_azimuth
        + z * cos_axis_tilt
    )

    # El ángulo de rotación ideal wid es la rotación para colocar el vector de posición solar
    # (xp, yp, zp) en el plano (y, z), que es normal al panel y
    # contiene el eje de rotación. wid = 0 indica que el panel está
    # horizontal. Aquí, nuestra convención es que una rotación en sentido horario es
    # positiva, para ver ángulos de rotación en el mismo marco de referencia que
    # el azimut. Por ejemplo, para un sistema con el eje de seguimiento orientado al sur, una
    # rotación hacia el este es negativa y una rotación hacia el oeste es
    # positiva. Esta es una rotación diestra alrededor del eje y del seguidor.

    # Calcular el ángulo desde el plano x-y hasta la proyección del vector solar sobre el plano x-z
    # usando [1] Eq. 5.

    wid = np.degrees(np.arctan2(xp, zp))

    # filtrar para el sol por encima del horizonte del panel
    zen_gt_90 = apparent_zenith > 90
    wid[zen_gt_90] = np.nan

    # Considerar el retroceso
    if backtrack:
        # Distancia entre filas en términos de longitudes de bastidores relativas a la inclinación cruzada del eje
        axes_distance = 1 / (gcr * cosd(cross_axis_tilt))

        # NOTA: tener en cuenta ángulos raros por debajo del conjunto, ver GH 824
        temp = np.abs(axes_distance * cosd(wid - cross_axis_tilt))

        # Ángulo de retroceso usando [1], Ec. 14
        with np.errstate(invalid="ignore"):
            wc = np.degrees(-np.sign(wid) * np.arccos(temp))

        # NOTA: en medio del día, arccos(temp) está fuera de rango porque
        # no hay sombra de fila a fila para evitar y el retroceso no es necesario
        # [1], Ec. 15-16
        with np.errstate(invalid="ignore"):
            tracker_theta = wid + np.where(temp < 1, wc, 0)
    else:
        tracker_theta = wid

    # NOTA: max_angle definido en relación con la rotación de punto cero, no la
    # normal del plano del sistema

    # Determinar los ángulos de rotación mínimo y máximo en función de max_angle.
    # Si max_angle es un solo valor, asumir que min_angle es el negativo.
    if np.isscalar(max_angle):
        min_angle = -max_angle
    else:
        min_angle, max_angle = max_angle

    # Recortar tracker_theta entre los ángulos mínimo y máximo.
    tracker_theta = np.clip(tracker_theta, min_angle, max_angle)

    # Calcular ángulos auxiliares
    surface = calc_surface_orientation(tracker_theta, axis_tilt, axis_azimuth)
    surface_tilt = surface["surface_tilt"]
    surface_azimuth = surface["surface_azimuth"]
    aoi = irradiance.aoi(
        surface_tilt, surface_azimuth, apparent_zenith, apparent_azimuth
    )

    # Agrupar DataFrame para los valores de retorno y filtrar para el sol por debajo del horizonte.
    out = {
        "tracker_theta": tracker_theta,
        "aoi": aoi,
        "surface_azimuth": surface_azimuth,
        "surface_tilt": surface_tilt,
    }
    if index is not None:
        out = pd.DataFrame(out, index=index)
        out[zen_gt_90] = np.nan
    else:
        out = {k: np.where(zen_gt_90, np.nan, v) for k, v in out.items()}

    return out


def calc_surface_orientation(tracker_theta, axis_tilt=0, axis_azimuth=0):
    """
    Calcula los ángulos de inclinación y azimuth de la superficie para una rotación de seguimiento dada.

    Parámetros
    ----------
    tracker_theta : numérico
        Ángulo de rotación del seguidor como una rotación en sentido horario alrededor
        del eje definido por ``axis_tilt`` y ``axis_azimuth``. Por ejemplo, con
        ``axis_tilt=0`` y ``axis_azimuth=180``, ``tracker_theta > 0`` resulta en un
        ``surface_azimuth`` hacia el Oeste, mientras que ``tracker_theta < 0`` resulta
        en un ``surface_azimuth`` hacia el Este. [grado]
    axis_tilt : float, por defecto 0
        La inclinación del eje de rotación con respecto a la horizontal.
        ``axis_tilt`` debe ser >= 0 y <= 90. [grado]
    axis_azimuth : float, por defecto 0
        Un valor que denota la dirección de la brújula a lo largo de la cual se encuentra
        el eje de rotación. Medido al este del norte. [grado]

    Devuelve
    -------
    dict o DataFrame
        Contiene las claves ``'surface_tilt'`` y ``'surface_azimuth'`` que representan
        la orientación del módulo teniendo en cuenta la rotación del seguidor y la
        orientación del eje. [grado]

    Referencias
    ----------
    .. [1] William F. Marion y Aron P. Dobos, "Ángulo de Rotación para el Seguimiento
       Óptimo de Seguidores de Un Eje", Informe Técnico NREL/TP-6A20-58891,
       Julio 2013. :doi:`10.2172/1089596`
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        surface_tilt = acosd(cosd(tracker_theta) * cosd(axis_tilt))

        # acotar(..., -1, +1) para prevenir problemas de arcsin(1 + epsilon):
        azimuth_delta = asind(
            np.clip(sind(tracker_theta) / sind(surface_tilt), a_min=-1, a_max=1)
        )
        # Combinar Ecuaciones 2, 3 y 4:
        azimuth_delta = np.where(
            abs(tracker_theta) < 90,
            azimuth_delta,
            -azimuth_delta + np.sign(tracker_theta) * 180,
        )
        # manejar el caso de surface_tilt=0:
        azimuth_delta = np.where(sind(surface_tilt) != 0, azimuth_delta, 90)
        surface_azimuth = (axis_azimuth + azimuth_delta) % 360

    out = {
        "surface_tilt": surface_tilt,
        "surface_azimuth": surface_azimuth,
    }
    if hasattr(tracker_theta, "index"):
        out = pd.DataFrame(out)
    return out
