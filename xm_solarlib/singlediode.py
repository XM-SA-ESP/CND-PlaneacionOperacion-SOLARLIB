"Funciones de bajo nivel para resolver la ecuación de un solo diodo."

import numpy as np
from xm_solarlib.tools import _golden_sect_dataframe

from scipy.optimize import brentq, newton
from scipy.special import lambertw

from xm_solarlib.constantes import NOT_IMPLEMENTED_ERROR_MSG

"Parámetros predeterminados del método de Newton para este módulo."
NEWTON_DEFAULT_PARAMS = {
    'tol': 1e-6,
    'maxiter': 100
}

"Tensión intrínseca por unión celular para a: Si, CdTe, Mertens y otros."
VOLTAGE_BUILTIN = 0.9  # [V]


def estimate_voc(photocurrent, saturation_current, nnsvth):
    """
    Estimación aproximada de la tensión de circuito abierto útil para limitar búsquedas de
    "i" o "v" al utilizar :func:`~pvlib.pvsystem.singlediode`.

    Parámetros
    ----------
    photocurrent : numérico
        corriente foto-generada [A]
    saturation_current : numérico
        corriente de saturación inversa del diodo [A]
    nnsvth : numérico
        producto de la tensión térmica ``Vth`` [V], el factor de idealidad del diodo ``n``,
        y el número de celdas en serie ``Ns``

    Retorna
    -------
    numérico
        estimación aproximada de la tensión de circuito abierto [V]

    Notas
    -----
    Calcular la tensión de circuito abierto, :math:`V_{oc}`, de un dispositivo ideal
    con resistencia en derivación infinita, :math:`R_{sh} \\to \\infty`, y resistencia en serie cero, :math:`r_s = 0`, nos da la siguiente ecuación [1]. Como estimación de :math:`V_{oc}`, es útil como límite superior para el método de bisección.

    .. math::

        V_{oc, est}=n Ns V_{th} \\log \\left( \\frac{I_L}{I_0} + 1 \\right)

    .. [1] http://www.pveducation.org/pvcdrom/open-circuit-voltage
    """     
    return nnsvth * np.log(np.asarray(photocurrent) / saturation_current + 1.0)


def bishop88(diode_voltage, photocurrent, saturation_current,
             resistance_series, resistance_shunt, nnsvth, d2mutau=0,
             nsvbi=np.Inf, breakdown_factor=0., breakdown_voltage=-5.5,
             breakdown_exp=3.28, gradients=False):
    r"""
    Cálculo explícito de puntos en la curva IV descrita por la ecuación del diodo único. Los valores se calculan según se describe en [1]_.

    La ecuación del diodo único con corriente de recombinación y ruptura de polarización inversa es

    .. math::

        I = I_{L} - I_{0} \left (\exp \frac{V_{d}}{nN_{s}V_{th}} - 1 \right )
        - \frac{V_{d}}{R_{sh}}
        - \frac{I_{L} \frac{d^{2}}{\mu \tau}}{N_{s} V_{bi} - V_{d}}
        - a \frac{V_{d}}{R_{sh}} \left (1 - \frac{V_{d}}{V_{br}} \right )^{-m}

    La entrada 'diode_voltage' debe ser :math:`V + I R_{s}`.

    .. warning::
       * Se requiere el uso de ``d2mutau`` con coeficientes de PVSyst
         para módulos de telururo de cadmio (CdTe) y silicio amorfo
         (a:Si) solamente.
       * No utilice ``d2mutau`` con coeficientes de CEC.

    Parámetros
    ----------
    diode_voltage : numérico
        voltaje del diodo :math:`V_d` [V]
    photocurrent : numérico
        corriente foto-generada :math:`I_{L}` [A]
    saturation_current : numérico
        corriente de saturación inversa del diodo :math:`I_{0}` [A]
    resistance_series : numérico
        resistencia en serie :math:`R_{s}` [ohmios]
    resistance_shunt: numérico
        resistencia en derivación :math:`R_{sh}` [ohmios]
    nnsvth : numérico
        producto del voltaje térmico :math:`V_{th}` [V], factor de idealidad del diodo
        :math:`n`, y número de celdas en serie :math:`N_{s}` [V]
    d2mutau : numérico, valor predeterminado 0
        parámetro de PVSyst para módulos de telururo de cadmio (CdTe) y silicio amorfo
        (a-Si) que tiene en cuenta la corriente de recombinación en la capa intrínseca. El valor es la razón del cuadrado del espesor de la capa intrínseca
        :math:`d^2` a la longitud de difusión de portadores de carga
        :math:`\mu \tau`. [V]
    nsvbi : numérico, valor predeterminado np.inf
        parámetro de PVSyst para módulos de telururo de cadmio (CdTe) y silicio amorfo
        (a-Si) que es el producto del número de celdas en serie del módulo fotovoltaico
        :math:`N_{s}` y el voltaje incorporado :math:`V_{bi}` de la capa intrínseca. [V].
    breakdown_factor : numérico, valor predeterminado 0
        fracción de la corriente óhmica involucrada en la ruptura por avalancha :math:`a`.
        El valor predeterminado de 0 excluye el término de polarización inversa del modelo. [adimensional]
    breakdown_voltage : numérico, valor predeterminado -5.5
        voltaje de ruptura inversa de la unión fotovoltaica :math:`V_{br}` [V]
    breakdown_exp : numérico, valor predeterminado 3.28
        exponente de ruptura por avalancha :math:`m` [adimensional]
    gradients : bool
        False devuelve solo I, V, y P. True también devuelve gradientes

    Devuelve
    -------
    tupla
        corrientes [A], voltajes [V], potencia [W], y opcionalmente
        :math:`\frac{dI}{dV_d}`, :math:`\frac{dV}{dV_d}`,
        :math:`\frac{dI}{dV}`, :math:`\frac{dP}{dV}`, y
        :math:`\frac{d^2 P}{dV dV_d}`

    Notas
    -----
    Los parámetros de pérdidas por recombinación de películas delgadas de PVSyst 'd2mutau' y
    'nsvbi' deben aplicarse solo a módulos fotovoltaicos de telururo de cadmio (CdTe) y silicio amorfo
    (a-Si), [2]_, [3]_. El voltaje incorporado :math:`V_{bi}` debe tener en cuenta todas las uniones. Por ejemplo: las células tándem y de triple unión
    tendrían voltajes incorporados de 1.8[V] y 2.7[V], respectivamente, basados
    en el valor predeterminado de 0.9[V] para una sola unión. El parámetro 'nsvbi'
    solo debe tener en cuenta el número de celdas en serie en una sola subcadena en paralelo si el módulo tiene celdas en paralelo mayores que 1.

    Referencias
    ----------
    .. [1] "Simulación por computadora de los efectos de desajustes eléctricos en
       circuitos de interconexión de celdas fotovoltaicas" JW Bishop, Solar Cell (1988)
       :doi:`10.1016/0379-6787(88)90059-2`

    .. [2] "Circuito equivalente mejorado y modelo analítico para celdas y módulos solares de silicio amorfo." J. Mertens, et al., IEEE Transactions
       on Electron Devices, Vol 45, No 2, Feb 1998.
       :doi:`10.1109/16.658676`

    .. [3] "Evaluación del rendimiento de un modelo de simulación para módulos fotovoltaicos de cualquier tecnología disponible", André Mermoud y Thibault Lejeune, 25th EUPVSEC,
       2010
       :doi:`10.4229/25thEUPVSEC2010-4BV.1.114`
    """
    # calcular la corriente de pérdida por recombinación cuando d2mutau > 0
    is_recomb = d2mutau > 0  # Verdadero donde hay pérdida de recombinación de película delgada
    v_recomb = np.where(is_recomb, nsvbi - diode_voltage, np.inf)
    i_recomb = np.where(is_recomb, photocurrent * d2mutau / v_recomb, 0)
    # calcular valores temporales para simplificar los cálculos
    v_star = diode_voltage / nnsvth  # voltaje no dimensional del diodo
    g_sh = 1.0 / resistance_shunt  # conductancia
    if breakdown_factor > 0:  # se considera polarización inversa
        brk_term = 1 - diode_voltage / breakdown_voltage
        brk_pwr = np.power(brk_term, -breakdown_exp)
        i_breakdown = breakdown_factor * diode_voltage * g_sh * brk_pwr
    else:
        i_breakdown = 0.
    i = (photocurrent - saturation_current * np.expm1(v_star)  # noqa: W503
         - diode_voltage * g_sh - i_recomb - i_breakdown)   # noqa: W503
    v = diode_voltage - i * resistance_series
    retval = (i, v, i*v)
    if gradients:
        # calcular gradientes de corriente de pérdida por recombinación cuando d2mutau > 0
        grad_i_recomb = np.where(is_recomb, i_recomb / v_recomb, 0)
        grad_2i_recomb = np.where(is_recomb, 2 * grad_i_recomb / v_recomb, 0)
        g_diode = saturation_current * np.exp(v_star) / nnsvth  # conductancia
        if breakdown_factor > 0:  # se considera polarización inversa
            brk_pwr_1 = np.power(brk_term, -breakdown_exp - 1)
            brk_pwr_2 = np.power(brk_term, -breakdown_exp - 2)
            brk_fctr = breakdown_factor * g_sh
            grad_i_brk = brk_fctr * (brk_pwr + diode_voltage *
                                     -breakdown_exp * brk_pwr_1)
            grad2i_brk = (brk_fctr * -breakdown_exp        # noqa: W503
                          * (2 * brk_pwr_1 + diode_voltage   # noqa: W503
                             * (-breakdown_exp - 1) * brk_pwr_2))  # noqa: W503
        else:
            grad_i_brk = 0.
            grad2i_brk = 0.
        grad_i = -g_diode - g_sh - grad_i_recomb - grad_i_brk  # di/dvd
        grad_v = 1.0 - grad_i * resistance_series  # dv/dvd
        grad = grad_i / grad_v  # di/dv
        grad_p = v * grad + i  # dp/dv
        grad2i = -g_diode / nnsvth - grad_2i_recomb - grad2i_brk  # d2i/dvd
        grad2v = -grad2i * resistance_series  # d2v/dvd
        grad2p = (
            grad_v * grad + v * (grad2i/grad_v - grad_i*grad2v/grad_v**2)
            + grad_i
        )  # d2p/dv/dvd
        retval += (grad_i, grad_v, grad, grad_p, grad2p)
    return retval


def bishop88_i_from_v(voltage, photocurrent, saturation_current,
                      resistance_series, resistance_shunt, nnsvth,
                      d2mutau=0, nsvbi=np.Inf, breakdown_factor=0.,
                      breakdown_voltage=-5.5, breakdown_exp=3.28,
                      method='newton', method_kwargs=None):
    """
    Encuentra la corriente dado cualquier voltaje.

    Parámetros
    ----------
    voltaje : numérico
        voltaje (V) en voltios [V]
    corriente_fotoelectrica : numérico
        corriente foto-generada (Iph o IL) [A]
    corriente_saturacion : numérico
        corriente oscura del diodo o corriente de saturación (Io o Isat) [A]
    resistencia_serie : numérico
        resistencia en serie (rs) en [Ohm]
    resistencia_derivacion : numérico
        resistencia en derivación (rsh) [Ohm]
    nnsvth : numérico
        producto del factor de idealidad del diodo (n), número de celdas en serie (Ns) y
        voltaje térmico (Vth = k_b * T / q_e) en voltios [V]
    d2mutau : numérico, valor predeterminado 0
        parámetro de PVSyst para módulos de telururo de cadmio (CdTe) y silicio amorfo
        (a-Si) que tiene en cuenta la corriente de recombinación en la
        capa intrínseca. El valor es la razón del cuadrado del espesor de la capa intrínseca
        :math:`d^2` a la longitud de difusión de portadores de carga
        :math:`\mu \tau`. [V]
    nsvbi : numérico, valor predeterminado np.inf
        parámetro de PVSyst para módulos de telururo de cadmio (CdTe) y silicio amorfo
        (a-Si) que es el producto del número de celdas en serie del módulo PV y el voltaje
        incorporado ``Vbi`` de la capa intrínseca. [V].
    factor_rotura : numérico, valor predeterminado 0
        fracción de la corriente óhmica involucrada en la ruptura por avalancha :math:`a`.
        El valor predeterminado de 0 excluye el término de polarización inversa del modelo. [adimensional]
    voltaje_rotura : numérico, valor predeterminado -5.5
        voltaje de ruptura inversa de la unión fotovoltaica :math:`V_{br}` [V]
    exponente_rotura : numérico, valor predeterminado 3.28
        exponente de ruptura por avalancha :math:`m` [adimensional]
    metodo : str, valor predeterminado 'newton'
       Puede ser ``'newton'`` o ``'brentq'``. ''metodo'' debe ser ``'newton'``
       si ``factor_rotura`` no es 0.
    argumentos_metodo : dict, opcional
        Argumentos clave pasados al método del buscador de raíces. Consulta
        los parámetros de :py:func:`scipy:scipy.optimize.brentq` y
        :py:func:`scipy:scipy.optimize.newton`. Se permite ``'full_output': True``,
        y ``optimizer_output`` sería devuelto. Consulta la sección de ejemplos.

    Devuelve
    -------
    corriente : numérico
        corriente (I) en el voltaje especificado (V). [A]
    optimizer_output : tupla, opcional, si se especifica en ``argumentos_metodo``
        consulta la documentación del buscador de raíces seleccionado.
        La raíz encontrada es el voltaje del diodo en [1]_.

    Ejemplos
    --------
    Usando los siguientes argumentos que pueden provenir de cualquier
    función ``calcparams_.*`` en :py:mod:`pvlib.pvsystem`:


    >>> args = {'corriente_fotoelectrica': 1., 'corriente_saturacion': 9e-10, 'nnsvth': 4.,
    ...         'resistencia_serie': 4., 'resistencia_derivacion': 5000.0}

    Usar los valores predeterminados:

    >>> i = bishop88_i_from_v(0.0, **args)

    Especificar tolerancias y número máximo de iteraciones:

    >>> i = bishop88_i_from_v(0.0, **args, metodo='newton',
    ...     argumentos_metodo={'tol': 1e-3, 'rtol': 1e-3, 'maxiter': 20})

    Obtener la salida completa del buscador de raíces:

    >>> i, metodo_salida = bishop88_i_from_v(0.0, **args, metodo='newton',
    ...     argumentos_metodo={'full_output': True})
    """
    # recopilar argumentos
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nnsvth, d2mutau, nsvbi,
            breakdown_factor, breakdown_voltage, breakdown_exp)
    method = method.lower()

    # crear un dict para argumentos_metodo si no se proporciona
    # este patrón evita errores con parámetros predeterminados mutables
    if not method_kwargs:
        method_kwargs = {}

    def fv(x, v, *a):
        # calcular la diferencia de voltaje dado el voltaje del diodo "x"
        return bishop88(x, *a)[1] - v

    if method == 'brentq':
        # primero limita la búsqueda usando voc
        voc_est = estimate_voc(photocurrent, saturation_current, nnsvth)

        # brentq solo funciona con entradas escalares, por lo que necesitamos una función de configuración
        # y np.vectorize para llamar repetidamente al optimizador con los argumentos correctos para entrada de matriz
        def vd_from_brent(voc, v, iph, isat, rs, rsh, gamma, d2mutau, nsvbi,
                          breakdown_factor, breakdown_voltage, breakdown_exp):
            return brentq(fv, 0.0, voc,
                          args=(v, iph, isat, rs, rsh, gamma, d2mutau, nsvbi,
                                breakdown_factor, breakdown_voltage,
                                breakdown_exp),
                          **method_kwargs)

        vd_from_brent_vectorized = np.vectorize(vd_from_brent)
        vd = vd_from_brent_vectorized(voc_est, voltage, *args)
    elif method == 'newton':
        # asegurarse de que todos los argumentos sean matrices numpy si el tamaño máximo > 1
        # si el voltaje es una matriz, entonces hacer una copia para usarla como suposición inicial, v0
        args, v0, method_kwargs = \
            _prepare_newton_inputs((voltage,), args, voltage, method_kwargs)
        vd = newton(func=lambda x, *a: fv(x, voltage, *a), x0=v0,
                    fprime=lambda x, *a: bishop88(x, *a, gradients=True)[4],
                    args=args,
                    **method_kwargs)
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MSG % method)

    # Cuando se especifica el parámetro 'full_output', el valor de 'vd' devuelto es una tupla con
    # muchos elementos, donde la raíz es el primero. Así que lo usamos para devolver
    # el resultado de bishop88 y devolver una tupla (escalar, tupla con los resultados del método)
    if method_kwargs.get('full_output') is True:
        return (bishop88(vd[0], *args)[0], vd)
    else:
        return bishop88(vd, *args)[0]


def bishop88_v_from_i(current, photocurrent, saturation_current,
                      resistance_series, resistance_shunt, nnsvth,
                      d2mutau=0, nsvbi=np.Inf, breakdown_factor=0.,
                      breakdown_voltage=-5.5, breakdown_exp=3.28,
                      method='newton', method_kwargs=None):
    """
    Encuentra la corriente dado cualquier voltaje.

    Parámetros
    ----------
    voltaje : numérico
        voltaje (V) en voltios [V]
    fotocorriente : numérico
        corriente fotogenerada (Iph o IL) [A]
    corriente_saturación : numérico
        corriente oscura o de saturación del diodo (Io o Isat) [A]
    resistencia_serie : numérico
        resistencia en serie (rs) en [Ohm]
    resistencia_derivación : numérico
        resistencia en derivación (rsh) [Ohm]
    nnsvth : numérico
        producto del factor de idealidad del diodo (n), número de celdas en serie (Ns) y
        voltaje térmico (Vth = k_b * T / q_e) en voltios [V]
    d2mutau : numérico, predeterminado 0
        parámetro de PVsyst para módulos de telururo de cadmio (CdTe) y silicio amorfo
        (a-Si) que tiene en cuenta la corriente de recombinación en la
        capa intrínseca. El valor es la razón del cuadrado del grosor de la capa intrínseca
        :math:`d^2` a la longitud de difusión de portadores de carga
        :math:`\\mu \\tau`. [V]
    nsvbi : numérico, predeterminado np.Inf
        parámetro de PVsyst para módulos de telururo de cadmio (CdTe) y silicio amorfo
        (a-Si) que es el producto del número de celdas en serie del módulo PV ``Ns`` y el
        voltaje integrado ``Vbi`` de la capa intrínseca. [V].
    factor_rotura : numérico, predeterminado 0
        fracción de corriente ohmica involucrada en la rotura por avalancha :math:`a`.
        El valor predeterminado de 0 excluye el término de polarización inversa del modelo. [adimensional]
    voltaje_rotura : numérico, predeterminado -5.5
        voltaje de rotura inversa de la unión fotovoltaica :math:`V_{br}`
        [V]
    exponente_rotura : numérico, predeterminado 3.28
        exponente de rotura por avalancha :math:`m` [adimensional]
    método : str, predeterminado 'newton'
       Puede ser ``'newton'`` o ``'brentq'``. ''método'' debe ser ``'newton'``
       si ``factor_rotura`` no es 0.
    método_kwargs : dict, opcional
        Argumentos clave pasados al método de búsqueda de raíces. Consulta
        los parámetros de :py:func:`scipy:scipy.optimize.brentq` y
        :py:func:`scipy:scipy.optimize.newton`.
        Se permite ``'full_output': True``, y se devolvería ``optimizer_output``.
        Consulta la sección de ejemplos.

    Devuelve
    -------
    corriente : numérico
        corriente (I) en el voltaje especificado (V). [A]
    optimizer_output : tupla, opcional, si se especifica en ``método_kwargs``
        consulta la documentación del buscador de raíces para el método seleccionado.
        La raíz encontrada es el voltaje del diodo en [1]_.

    Ejemplos
    --------
    Usando los siguientes argumentos que pueden provenir de cualquier
    función `calcparams_.*` en :py:mod:`pvlib.pvsystem`:

    >>> args = {'fotocorriente': 1., 'corriente_saturación': 9e-10, 'nnsvth': 4.,
    ...         'resistencia_serie': 4., 'resistencia_derivación': 5000.0}

    Usar valores predeterminados:

    >>> i = bishop88_i_from_v(0.0, **args)

    Especificar tolerancias y número máximo de iteraciones:

    >>> i = bishop88_i_from_v(0.0, **args, método='newton',
    ...     método_kwargs={'tol': 1e-3, 'rtol': 1e-3, 'maxiter': 20})

    Recuperar la salida completa del buscador de raíces:

    >>> i, método_output = bishop88_i_from_v(0.0, **args, método='newton',
    ...     método_kwargs={'full_output': True})
    """
    # recopilar argumentos
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nnsvth, d2mutau, nsvbi, breakdown_factor,
            breakdown_voltage, breakdown_exp)
    method = method.lower()

    # crear un dict para argumentos_método si no se proporciona
    # este patrón evita errores con parámetros predeterminados mutables
    if not method_kwargs:
        method_kwargs = {}

    # primero limite la búsqueda usando voc
    voc_est = estimate_voc(photocurrent, saturation_current, nnsvth)

    def fi(x, i, *a):
        # calcular la diferencia de corriente dado el voltaje del diodo "x"
        return bishop88(x, *a)[0] - i

    if method == 'brentq':
        # brentq solo funciona con entradas escalares, por lo que necesitamos una función de configuración
        # y np.vectorize para llamar repetidamente al optimizador con los argumentos correctos para entrada de matriz
        def vd_from_brent(voc, i, iph, isat, rs, rsh, gamma, d2mutau, nsvbi,
                          breakdown_factor, breakdown_voltage, breakdown_exp):
            return brentq(fi, 0.0, voc,
                          args=(i, iph, isat, rs, rsh, gamma, d2mutau, nsvbi,
                                breakdown_factor, breakdown_voltage,
                                breakdown_exp),
                          **method_kwargs)

        vd_from_brent_vectorized = np.vectorize(vd_from_brent)
        vd = vd_from_brent_vectorized(voc_est, current, *args)
    elif method == 'newton':
        # asegurarse de que todos los argumentos sean matrices numpy si el tamaño máximo > 1
        # si voc_est es una matriz, hacer una copia para usarla como suposición inicial, v0
        args, v0, method_kwargs = \
            _prepare_newton_inputs((current,), args, voc_est, method_kwargs)
        vd = newton(func=lambda x, *a: fi(x, current, *a), x0=v0,
                    fprime=lambda x, *a: bishop88(x, *a, gradients=True)[3],
                    args=args,
                    **method_kwargs)
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MSG % method)

    # Cuando se especifica el parámetro 'full_output', el valor de 'vd' devuelto es una tupla con
    # muchos elementos, donde la raíz es el primero. Así que lo usamos para devolver
    # el resultado de bishop88 y devolver una tupla (escalar, tupla con los resultados del método)
    if method_kwargs.get('full_output') is True:
        return (bishop88(vd[0], *args)[1], vd)
    else:
        return bishop88(vd, *args)[1]


def bishop88_mpp(photocurrent, saturation_current, resistance_series,
                 resistance_shunt, nnsvth, d2mutau=0, nsvbi=np.Inf,
                 breakdown_factor=0., breakdown_voltage=-5.5,
                 breakdown_exp=3.28, method='newton', method_kwargs=None):
    """
    Encuentra el punto de máxima potencia.

    Parámetros
    ----------
    fotocorriente : numérico
        corriente fotogenerada (Iph o IL) [A]
    corriente_saturación : numérico
        corriente oscura o de saturación del diodo (Io o Isat) [A]
    resistencia_serie : numérico
        resistencia en serie (rs) en [Ohm]
    resistencia_derivación : numérico
        resistencia en derivación (rsh) [Ohm]
    nnsvth : numérico
        producto del factor de idealidad del diodo (n), número de celdas en serie (Ns), y
        voltaje térmico (Vth = k_b * T / q_e) en voltios [V]
    d2mutau : numérico, predeterminado 0
        parámetro de PVsyst para módulos de telururo de cadmio (CdTe) y silicio amorfo
        (a-Si) que tiene en cuenta la corriente de recombinación en la
        capa intrínseca. El valor es la razón del cuadrado del grosor de la capa intrínseca
        :math:`d^2` a la longitud de difusión de portadores de carga
        :math:`\\mu \\tau`. [V]
    nsvbi : numérico, predeterminado np.inf
        parámetro de PVsyst para módulos de telururo de cadmio (CdTe) y silicio amorfo
        (a-Si) que es el producto del número de celdas en serie del módulo PV ``Ns`` y el
        voltaje integrado ``Vbi`` de la capa intrínseca. [V].
    factor_rotura : numérico, predeterminado 0
        fracción de corriente ohmica involucrada en la rotura por avalancha :math:`a`.
        El valor predeterminado de 0 excluye el término de polarización inversa del modelo. [adimensional]
    voltaje_rotura : numérico, predeterminado -5.5
        voltaje de rotura inversa de la unión fotovoltaica :math:`V_{br}`
        [V]
    exponente_rotura : numérico, predeterminado 3.28
        exponente de rotura por avalancha :math:`m` [adimensional]
    método : str, predeterminado 'newton'
       Puede ser ``'newton'`` o ``'brentq'``. ''método'' debe ser ``'newton'``
       si ``factor_rotura`` no es 0.
    método_kwargs : dict, opcional
        Argumentos clave pasados al método de búsqueda de raíces. Consulta
        los parámetros de :py:func:`scipy:scipy.optimize.brentq` y
        :py:func:`scipy:scipy.optimize.newton`.
        Se permite ``'full_output': True``, y se devolvería ``optimizer_output``.
        Consulta la sección de ejemplos.

    Devuelve
    -------
    tupla
        corriente de máxima potencia ``i_mp`` [A], voltaje de máxima potencia ``v_mp`` [V], y
        máxima potencia ``p_mp`` [W]
    optimizer_output : tupla, opcional, si se especifica en ``método_kwargs``
        consulta la documentación del buscador de raíces para el método seleccionado.
        La raíz encontrada es el voltaje del diodo en [1]_.

    Ejemplos
    --------
    Usando los siguientes argumentos que pueden provenir de cualquier
    función `calcparams_.*` en :py:mod:`pvlib.pvsystem`:

    >>> args = {'fotocorriente': 1., 'corriente_saturación': 9e-10, 'nnsvth': 4.,
    ...         'resistencia_serie': 4., 'resistencia_derivación': 5000.0}

    Usar valores predeterminados:

    >>> i_mp, v_mp, p_mp = bishop88_mpp(**args)

    Especificar tolerancias y número máximo de iteraciones:

    >>> i_mp, v_mp, p_mp = bishop88_mpp(**args, método='newton',
    ...     método_kwargs={'tol': 1e-3, 'rtol': 1e-3, 'maxiter': 20})

    Recuperar la salida completa del buscador de raíces:

    >>> (i_mp, v_mp, p_mp), método_output = bishop88_mpp(**args,
    ...     método='newton', método_kwargs={'full_output': True})
    """
    # recopilar argumentos
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nnsvth, d2mutau, nsvbi, breakdown_factor,
            breakdown_voltage, breakdown_exp)
    method = method.lower()

    # argumentos_método crea un diccionario si no se proporciona
    # este patrón evita errores con los parámetros predeterminados mutables
    if not method_kwargs:
        method_kwargs = {}

    # primero limitar la búsqueda usando voc
    voc_est = estimate_voc(photocurrent, saturation_current, nnsvth)

    def fmpp(x, *a):
        return bishop88(x, *a, gradients=True)[6]

    if method == 'brentq':
        # descomponer argumentos para que numpy.vectorize maneje la difusión
        vec_fun = np.vectorize(
            lambda voc, iph, isat, rs, rsh, gamma, d2mutau, nsvbi, vbr_a, vbr,
            vbr_exp: brentq(fmpp, 0.0, voc,
                            args=(iph, isat, rs, rsh, gamma, d2mutau, nsvbi,
                                  vbr_a, vbr, vbr_exp),
                            **method_kwargs)
        )
        vd = vec_fun(voc_est, *args)
    elif method == 'newton':
        # asegurarse de que todos los argumentos sean matrices numpy si el tamaño máximo > 1
        # si voc_est es una matriz, hacer una copia para usarla como suposición inicial, v0
        args, v0, method_kwargs = \
            _prepare_newton_inputs((), args, voc_est, method_kwargs)
        vd = newton(
            func=fmpp, x0=v0,
            fprime=lambda x, *a: bishop88(x, *a, gradients=True)[7], args=args,
            **method_kwargs)
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MSG % method)

    # Cuando se especifica el parámetro 'full_output', el valor de 'vd' devuelto es una tupla con
    # muchos elementos, donde la raíz es el primero. Así que lo usamos para devolver
    # el resultado de bishop88 y devolver
    # una tupla (tupla con la solución de bishop88, tupla con los resultados del método)
    if method_kwargs.get('full_output') is True:
        return (bishop88(vd[0], *args), vd)
    else:
        return bishop88(vd, *args)


def _get_size_and_shape(args):
    # encontrar el tamaño y la forma adecuados para los resultados
    size, shape = 0, None  # 0 o None ambos significan escalar
    for arg in args:
        try:
            this_shape = arg.shape  # intenta obtener la forma
        except AttributeError:
            this_shape = None
            try:
                this_size = len(arg)  # intenta obtener el tamaño
            except TypeError:
                this_size = 0
        else:
            this_size = arg.size  # si tiene forma, entonces también tiene tamaño
            if shape is None:
                shape = this_shape  # establece la forma si es None
        # actualizar tamaño y forma
        if this_size > size:
            size = this_size
            if this_shape is not None:
                shape = this_shape
    return size, shape


def _prepare_newton_inputs(i_or_v_tup, args, v0, method_kwargs):
    # propagar los argumentos para el método newton
    # el primer argumento debe ser una tupla, por ejemplo: (i,), (v,) o ()
    size, shape = _get_size_and_shape(i_or_v_tup + args)
    if size > 1:
        args = [np.asarray(arg) for arg in args]
    # newton utiliza una suposición inicial para la forma de salida
    # copiar v0 a una nueva matriz y propagarla a la forma del tamaño máximo
    if shape is not None:
        v0 = np.broadcast_to(v0, shape).copy()

    # establecer la tolerancia absoluta y el número máximo de iteraciones desde method_kwargs si no se proporcionan
    # aplicar valores predeterminados, pero dando prioridad a los valores especificados por el usuario
    method_kwargs = {**NEWTON_DEFAULT_PARAMS, **method_kwargs}

    return args, v0, method_kwargs


def _lambertw_v_from_i(current, photocurrent, saturation_current,
                       resistance_series, resistance_shunt, nnsvth):
    # Registra si las entradas eran todas escalares
    output_is_scalar = all(map(np.isscalar,
                               (current, photocurrent, saturation_current,
                                resistance_series, resistance_shunt, nnsvth)))

    # Esto transforma gsh=1/rsh, incluyendo rsh ideal=np.inf en gsh=0., lo cual
    # generalmente es más estable numéricamente
    conductance_shunt = 1. / resistance_shunt

    # Asegura que estemos trabajando con vistas de solo lectura de matrices numpy
    # Convierte Series en matrices para que no tengamos que preocuparnos por
    # problemas de transmisión multidimensional
    I, IL, I0, rs, gsh, a = \
        np.broadcast_arrays(current, photocurrent, saturation_current,
                            resistance_series, conductance_shunt, nnsvth)

    # Inicializa la salida V (I podría no ser float64)
    V = np.full_like(I, np.nan, dtype=np.float64)

    # Determina índices donde 0 < gsh requiere una solución implícita del modelo
    idx_p = 0. < gsh

    # Determina índices donde 0 = gsh permite una solución explícita del modelo
    idx_z = 0. == gsh

    # Soluciones explícitas donde gsh=0
    if np.any(idx_z):
        V[idx_z] = a[idx_z] * np.log1p((IL[idx_z] - I[idx_z]) / I0[idx_z]) - \
                   I[idx_z] * rs[idx_z]

    # Solo calcular usando LambertW si hay casos con gsh>0
    if np.any(idx_p):
        # Argumento de LambertW, no puede ser float128, puede desbordar a np.inf
        # el desbordamiento se maneja explícitamente a continuación, así que
        # ignoramos las advertencias aquí
        with np.errstate(over='ignore'):
            argw = (I0[idx_p] / (gsh[idx_p] * a[idx_p]) *
                    np.exp((-I[idx_p] + IL[idx_p] + I0[idx_p]) /
                           (gsh[idx_p] * a[idx_p])))

        # lambertw normalmente devuelve un valor complejo con parte imaginaria cero
        # puede desbordarse a np.inf
        lambertwterm = lambertw(argw).real

        # Registra los índices donde la entrada de lambertw se desbordó en la salida
        idx_inf = np.logical_not(np.isfinite(lambertwterm))

        # Solo vuelva a calcular LambertW si se desbordó
        if np.any(idx_inf):
            # Calcule usando log(argw) en caso de que argw sea realmente grande
            logargw = (np.log(I0[idx_p]) - np.log(gsh[idx_p]) -
                       np.log(a[idx_p]) +
                       (-I[idx_p] + IL[idx_p] + I0[idx_p]) /
                       (gsh[idx_p] * a[idx_p]))[idx_inf]

            # Tres iteraciones del método de Newton-Raphson para resolver
            # w+log(w)=logargw. La suposición inicial es w=logargw. Donde la evaluación directa
            # (arriba) resulta en NaN debido al desbordamiento, 3 iteraciones
            # del método de Newton dan aproximadamente 8 dígitos de precisión.
            w = logargw
            for _ in range(0, 3):
                w = w * (1. - np.log(w) + logargw) / (1. + w)
            lambertwterm[idx_inf] = w

        # Ec. 3 en Jain y Kapoor, 2004
        #  V = -I*(rs + rsh) + IL*rsh - a*lambertwterm + I0*rsh
        # Reformulado en términos de gsh=1/rsh para una mejor estabilidad numérica.
        V[idx_p] = (IL[idx_p] + I0[idx_p] - I[idx_p]) / gsh[idx_p] - \
            I[idx_p] * rs[idx_p] - a[idx_p] * lambertwterm

    if output_is_scalar:
        return V.item()
    else:
        return V


def _lambertw_i_from_v(voltage, photocurrent, saturation_current,
                       resistance_series, resistance_shunt, nnsvth):
    # Registra si las entradas eran todas escalares
    output_is_scalar = all(map(np.isscalar,
                               (voltage, photocurrent, saturation_current,
                                resistance_series, resistance_shunt, nnsvth)))

    # Esto transforma gsh=1/rsh, incluyendo rsh ideal=np.inf en gsh=0., lo cual
    # generalmente es más estable numéricamente
    conductance_shunt = 1. / resistance_shunt

    # Asegura que estemos trabajando con vistas de solo lectura de matrices numpy
    # Convierte Series en matrices para que no tengamos que preocuparnos por
    # problemas de transmisión multidimensional
    V, IL, I0, rs, gsh, a = \
        np.broadcast_arrays(voltage, photocurrent, saturation_current,
                            resistance_series, conductance_shunt, nnsvth)

    # Inicializa la salida I (V podría no ser float64)
    I = np.full_like(V, np.nan, dtype=np.float64)           # noqa: E741, N806

    # Determina índices donde 0 < rs requiere una solución implícita del modelo
    idx_p = 0. < rs

    # Determina índices donde 0 = rs permite una solución explícita del modelo
    idx_z = 0. == rs

    # Soluciones explícitas donde rs=0
    if np.any(idx_z):
        I[idx_z] = IL[idx_z] - I0[idx_z] * np.expm1(V[idx_z] / a[idx_z]) - \
                   gsh[idx_z] * V[idx_z]

    # Solo calcular usando LambertW si hay casos con rs>0
    # NO maneja la posibilidad de desbordamiento, problema de GitHub 298
    if np.any(idx_p):
        # Argumento de LambertW, no puede ser float128, puede desbordar a np.inf
        argw = rs[idx_p] * I0[idx_p] / (
                    a[idx_p] * (rs[idx_p] * gsh[idx_p] + 1.)) * \
               np.exp((rs[idx_p] * (IL[idx_p] + I0[idx_p]) + V[idx_p]) /
                      (a[idx_p] * (rs[idx_p] * gsh[idx_p] + 1.)))

        # lambertw normalmente devuelve un valor complejo con parte imaginaria cero
        # puede desbordarse a np.inf
        lambertwterm = lambertw(argw).real

        # Ec. 2 en Jain y Kapoor, 2004
        #  I = -V/(rs + rsh) - (a/rs)*lambertwterm + rsh*(IL + I0)/(rs + rsh)
        # Reformulado en términos de gsh=1/rsh para una mejor estabilidad numérica.

        I[idx_p] = (IL[idx_p] + I0[idx_p] - V[idx_p] * gsh[idx_p]) / \
                   (rs[idx_p] * gsh[idx_p] + 1.) - (
                               a[idx_p] / rs[idx_p]) * lambertwterm

    if output_is_scalar:
        return I.item()
    else:
        return I


def _lambertw(photocurrent, saturation_current, resistance_series,
              resistance_shunt, nnsvth, ivcurve_pnts=None):
    # recopilar argumentos
    params = {'photocurrent': photocurrent,
              'saturation_current': saturation_current,
              'resistance_series': resistance_series,
              'resistance_shunt': resistance_shunt, 'nnsvth': nnsvth}

    # Calcular la corriente de cortocircuito
    i_sc = _lambertw_i_from_v(0., **params)

    # Calcular el voltaje de circuito abierto
    v_oc = _lambertw_v_from_i(0., **params)

    # Establecer los elementos pequeños <0 en v_ca a 0
    if isinstance(v_oc, np.ndarray):
        v_oc[(v_oc < 0) & (v_oc > -1e-12)] = 0.
    elif isinstance(v_oc, (float, int)) and (v_oc < 0 and v_oc > -1e-12):
        v_oc = 0.

    # Encontrar el voltaje, v_mp, donde la potencia está maximizada.
    # Iniciar la búsqueda de sección dorada en v_ca * 1.14
    p_mp, v_mp = _golden_sect_dataframe(params, 0., v_oc * 1.14, _pwr_optfcn)

    # Encontrar Imp usando Lambert W
    i_mp = _lambertw_i_from_v(v_mp, **params)

    # Encontrar Ix e Ixx usando Lambert W
    i_x = _lambertw_i_from_v(0.5 * v_oc, **params)

    i_xx = _lambertw_i_from_v(0.5 * (v_oc + v_mp), **params)

    out = (i_sc, v_oc, i_mp, v_mp, p_mp, i_x, i_xx)

    # crear curva IV
    if ivcurve_pnts:
        ivcurve_v = (np.asarray(v_oc)[..., np.newaxis] *
                     np.linspace(0, 1, ivcurve_pnts))

        ivcurve_i = _lambertw_i_from_v(ivcurve_v.T, **params).T

        out += (ivcurve_i, ivcurve_v)

    return out


def _pwr_optfcn(df, loc):
    '''
Función para encontrar la potencia a partir de i_from_v.
'''

    current = _lambertw_i_from_v(df[loc], df['photocurrent'],
                                 df['saturation_current'],
                                 df['resistance_series'],
                                 df['resistance_shunt'], df['nnsvth'])

    return current * df[loc]
