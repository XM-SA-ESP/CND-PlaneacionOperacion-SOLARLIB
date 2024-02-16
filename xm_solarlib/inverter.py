"""
Este módulo contiene funciones para el modelado de inversores y para ajustar modelos de inversores a datos.

Los modelos de inversores calculan la salida de potencia AC a partir de la entrada DC. Los parámetros del modelo
deben pasarse como un único diccionario.

Las funciones para estimar parámetros de modelos de inversores deben seguir el
patrón de nomenclatura 'fit_<nombre del modelo>', por ejemplo, fit_sandia.

"""

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit  # different than np.polyfit


def _sandia_eff(v_dc, p_dc, inverter):
    r"""
    Calcula la potencia AC del inversor sin recorte
    """
    paco = inverter["Paco"]
    pdco = inverter["Pdco"]
    vdco = inverter["Vdco"]
    C0 = inverter["C0"]
    C1 = inverter["C1"]
    C2 = inverter["C2"]
    C3 = inverter["C3"]
    pso = inverter["Pso"]

    A = pdco * (1 + C1 * (v_dc - vdco))
    B = pso * (1 + C2 * (v_dc - vdco))
    C = C0 * (1 + C3 * (v_dc - vdco))

    return (paco / (A - B) - C * (A - B)) * (p_dc - B) + C * (p_dc - B) ** 2


def _sandia_limits(power_ac, p_dc, paco, pnt, pso):
    r"""
    Aplica límites mínimos y máximos de potencia a `power_ac`
    """
    power_ac = np.minimum(paco, power_ac)
    min_ac_power = -1.0 * abs(pnt)
    below_limit = p_dc < pso
    try:
        power_ac[below_limit] = min_ac_power
    except TypeError:  # power_ac is a float
        if below_limit:
            power_ac = min_ac_power
    return power_ac


def sandia_multi(v_dc, p_dc, inverter):
    r"""
    Convierte la potencia y el voltaje DC en potencia AC para un inversor con múltiples
    entradas MPPT.

    Usa el modelo de inversor conectado a la red de Sandia [1]_. Extensión de [1]_
    para inversores con múltiples entradas desequilibradas como se describe en [2]_.

    Parámetros
    ----------
    v_dc : tupla, lista o array de numéricos
        Voltaje DC en cada entrada MPPT del inversor. Si el tipo es array, debe
        ser 2d con el eje 0 siendo las entradas MPPT. [V]

    p_dc : tupla, lista o array de numéricos
        Potencia DC en cada entrada MPPT del inversor. Si el tipo es array, debe
        ser 2d con el eje 0 siendo las entradas MPPT. [W]

    inverter : tipo diccionario
        Define los parámetros para el modelo de inversor en [1]_.

    Retorna
    -------
    power_ac : numérico
        Salida de potencia AC para el inversor. [W]

    Excepciones
    ------
    ValueError
        Si v_dc y p_dc tienen longitudes diferentes.

    Referencias
    ----------
    .. [1] D. King, S. Gonzalez, G. Galbraith, W. Boyson, "Modelo de Rendimiento
       para Inversores Fotovoltaicos Conectados a la Red", SAND2007-5036, Laboratorios Nacionales de Sandia.
    .. [2] C. Hansen, J. Johnson, R. Darbali-Zamora, N. Gurule. "Modelando la
       Eficiencia de Inversores con Múltiples Entradas", 49ª Conferencia Especialista en Fotovoltaica de IEEE, Filadelfia, PA, EE. UU. Junio de 2022.

    """

    if len(p_dc) != len(v_dc):
        raise ValueError("p_dc and v_dc have different lengths")
    power_dc = sum(p_dc)
    power_ac = 0.0 * power_dc

    for vdc, pdc in zip(v_dc, p_dc):
        power_ac += pdc / power_dc * _sandia_eff(vdc, power_dc, inverter)

    return _sandia_limits(
        power_ac, power_dc, inverter["Paco"], inverter["Pnt"], inverter["Pso"]
    )


def fit_sandia(ac_power, dc_power, dc_voltage, dc_voltage_level, p_ac_0, p_nt):
    r"""
    Determina parámetros para el modelo de inversor Sandia.

    Parámetros
    ----------
    ac_power : array_like
        Potencia de CA generada en cada punto de datos [W].
    dc_power : array_like
        Potencia de CC ingresada en cada punto de datos [W].
    dc_voltage : array_like
        Voltaje de entrada de CC en cada punto de datos [V].
    dc_voltage_level : array_like
        Nivel de voltaje de entrada de CC en cada punto de datos. Los valores deben ser 'Vmin',
        'Vnom' o 'Vmax'.
    p_ac_0 : float
        Potencia nominal de CA del inversor [W].
    p_nt : float
        Consumo de energía en la noche, es decir, la potencia consumida cuando el inversor no está entregando
        potencia de CA. [W]

    Returns
    -------
    dict
        Un conjunto de parámetros para el modelo de inversor Sandia [1]_. Consulta
        :py:func:`xm_solarlib.inverter.sandia` para obtener una descripción de las claves y valores.

    See Also
    --------
    xm_solarlib.inverter.sandia

    Notes
    -----
    El procedimiento de ajuste para estimar parámetros se describe en [2]_.
    Un punto de datos es un par de valores (dc_power, ac_power). Típicamente, el rendimiento del inversor se mide
    o describe en tres niveles de voltaje de entrada de CC,
    denominados 'Vmin', 'Vnom' y 'Vmax', y en cada nivel, la eficiencia del inversor
    se determina a varios niveles de potencia de salida. Por ejemplo,
    el protocolo de prueba del inversor CEC [3]_ especifica la medición de la potencia de entrada de CC
    que entrega potencia de salida de CA de 0.1, 0.2, 0.3, 0.5, 0.75 y 1.0 de
    la capacidad nominal de CA del inversor.

    Referencias
    ----------
    .. [1] D. King, S. Gonzalez, G. Galbraith, W. Boyson, "Modelo de Rendimiento
       para Inversores Fotovoltaicos Conectados a la Red", SAND2007-5036, Laboratorios Nacionales Sandia.
    .. [2] Página del Modelo de Inversor Sandia, Colaborativo de Modelado de Rendimiento Fotovoltaico
       https://pvpmc.sandia.gov/modeling-steps/dc-to-ac-conversion/sandia-inverter-model/
    .. [3] W. Bower, et al., "Protocolo de Prueba de Rendimiento para Evaluar
       Inversores Utilizados en Sistemas Fotovoltaicos Conectados a la Red", disponible en
       https://www.energy.ca.gov/sites/default/files/2020-06/2004-11-22_Sandia_Test_Protocol_ada.pdf
    """  # noqa: E501

    voltage_levels = ["Vmin", "Vnom", "Vmax"]

    # voltaje de entrada de dc promedio en cada nivel de voltaje
    v_d = np.array(
        [
            dc_voltage[dc_voltage_level == "Vmin"].mean(),
            dc_voltage[dc_voltage_level == "Vnom"].mean(),
            dc_voltage[dc_voltage_level == "Vmax"].mean(),
        ]
    )
    v_nom = v_d[1]  # model parameter
    # variable independiente para las regresiones, x_d
    x_d = v_d - v_nom

    # dataframe vacío para contener variables intermedias
    coeffs = pd.DataFrame(
        index=voltage_levels, columns=["a", "b", "c", "p_dc", "p_s0"], data=np.nan
    )

    def solve_quad(a, b, c):
        return (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)

    # [2] PASO 3E, ajustar una línea a (voltaje de CC, coeficiente_del_modelo)
    def extract_c(x_d, add):
        beta0, beta1 = polyfit(x_d, add, 1)
        c = beta1 / beta0
        return beta0, beta1, c

    for d in voltage_levels:
        x = dc_power[dc_voltage_level == d]
        y = ac_power[dc_voltage_level == d]
        # [2] STEP 3B
        # ajustar una cuadrática a (potencia de CC, potencia de CA)
        c, b, a = polyfit(x, y, 2)

        # [2] PASO 3D, resolver para p_dc y p_s0
        p_dc = solve_quad(a, b, (c - p_ac_0))
        p_s0 = solve_quad(a, b, c)

        # Agregar valores al dataframe en el índice d
        coeffs["a"][d] = a
        coeffs["p_dc"][d] = p_dc
        coeffs["p_s0"][d] = p_s0

    b_dc0, b_dc1, c1 = extract_c(x_d, coeffs["p_dc"])
    b_s0, b_s1, c2 = extract_c(x_d, coeffs["p_s0"])
    b_c0, b_c1, c3 = extract_c(x_d, coeffs["a"])

    p_dc0 = b_dc0
    p_s0 = b_s0
    c0 = b_c0

    return {
        "Paco": p_ac_0,
        "Pdco": p_dc0,
        "Vdco": v_nom,
        "Pso": p_s0,
        "C0": c0,
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "Pnt": p_nt,
    }
