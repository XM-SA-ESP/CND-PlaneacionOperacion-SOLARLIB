import numpy as np
import pandas as pd


def sapm(aoi, module, upper=None):
    r"""
    Determina el modificador del ángulo de incidencia (IAM) utilizando el modelo SAPM.

    Parámetros
    ----------
    aoi : numérico
        Ángulo de incidencia en grados. Los ángulos de entrada negativos devolverán
        ceros.

    module : tipo diccionario
        Un diccionario o Serie con los parámetros del modelo IAM SAPM.
        Consulta la sección de notas de :py:func:`sapm` para más detalles.

    upper : None o float, por defecto None
        Límite superior para los resultados.

    Retorna
    -------
    iam : numérico
        El coeficiente de pérdida de ángulo de incidencia de SAPM, denominado F2 en [1]_.

    Notas
    -----
    El SAPM [1]_ tradicionalmente no define un límite superior en la función
    de pérdida de AOI y pueden existir valores ligeramente superiores a 1 para
    ángulos de incidencia moderados (15-40 grados). Sin embargo, los usuarios pueden considerar
    imponer un límite superior de 1.

    Referencias
    ----------
    .. [1] King, D. et al, 2004, "Modelo de Rendimiento de Arreglo Fotovoltaico de Sandia",
       Informe SAND 3535, Laboratorios Nacionales de Sandia, Albuquerque, NM.

    .. [2] B.H. King et al, "Procedimiento para Determinar los Coeficientes para el
       Modelo de Rendimiento de Arreglo de Sandia (SAPM)," SAND2016-5284, Laboratorios Nacionales de Sandia (2016).

    .. [3] B.H. King et al, "Avances Recientes en Técnicas de Medición al Aire Libre para los
       Efectos del Ángulo de Incidencia," 42º IEEE PVSC (2015).
       DOI: 10.1109/PVSC.2015.7355849

    See Also
    --------
    xm_solarlib.iam.physical
    xm_solarlib.iam.ashrae
    xm_solarlib.iam.martin_ruiz
    xm_solarlib.iam.interp
    """

    aoi_coeff = [
        module["B5"],
        module["B4"],
        module["B3"],
        module["B2"],
        module["B1"],
        module["B0"],
    ]

    iam = np.polyval(aoi_coeff, aoi)
    iam = np.clip(iam, 0, upper)
    # nan tolerant masking
    aoi_lt_0 = np.full_like(aoi, False, dtype="bool")
    np.less(aoi, 0, where=~np.isnan(aoi), out=aoi_lt_0)
    iam = np.where(aoi_lt_0, 0, iam)

    if isinstance(aoi, pd.Series):
        iam = pd.Series(iam, aoi.index)

    return iam
