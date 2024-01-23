"""
Collection of functions used in xm_solarlib_python
"""

import datetime as dt
import numpy as np
import pandas as pd
import pytz
import warnings


def cosd(angle):
    """
    Coseno con entrada de ángulo en grados

    Parámetros
    ----------
    angle : float o similar a una matriz
        Ángulo en grados

    Retorna
    -------
    resultado : float o similar a una matriz
        Coseno del ángulo
    """
    res = np.cos(np.radians(angle))
    return res


def sind(angle):
    """
    Sine with angle input in degrees

    Parámetros
    ----------
    angle : float
        Ángulo en grados

    Retorna
    -------
    resultado : float
        Seno del ángulo
    """
    res = np.sin(np.radians(angle))
    return res


def tand(angle):
    """
    Trigonometric tangent with angle input in degrees.

    Parámetros
    ----------
    angle : float
        Ángulo en grados

    Retorna
    -------
    result : float
        Tan of the angle
    """
    res = np.tan(np.radians(angle))
    return res


def tand(angle):
    """
    Trigonometric tangent with angle input in degrees.

    Parameters
    ----------
    angle : float
        Angle in degrees

    Returns
    -------
    result : float
        Tan of the angle
    """
    res = np.tan(np.radians(angle))
    return res


def asind(number):
    """
    Seno inverso que devuelve un ángulo en grados

    Parámetros
    ----------
    número : float
        Número de entrada

    Retorna
    -------
    resultado : float
        Resultado del arcsin
    """
    res = np.degrees(np.arcsin(number))
    return res


def acosd(number):
    """
    Coseno inverso que devuelve un ángulo en grados.

    Parámetros
    ----------
    número : float
        Número de entrada

    Retorna
    -------
    resultado : float
        Resultado del arccos
    """
    res = np.degrees(np.arccos(number))
    return res


def atand(number):
    """
    Tangente inversa trigonométrica que devuelve un ángulo en grados.

    Parámetros
    ----------
    número : float
        Número de entrada

    Retorna
    -------
    resultado : float
        resultado del arctan
    """
    res = np.degrees(np.arctan(number))
    return res


def localize_to_utc(time, location):
    """
    Convierte o localiza una serie de tiempo en UTC.

    Parámetros
    ----------
    tiempo : datetime.datetime, pandas.DatetimeIndex,
           o pandas.Series/DataFrame con un DatetimeIndex.
    ubicación : objeto xm_solarlib.Location

    Returns
    -------
    objeto pandas localizado en UTC.
    """
    if isinstance(time, dt.datetime):
        if time.tzinfo is None:
            time = pytz.timezone(location.tz).localize(time)
        time_utc = time.astimezone(pytz.utc)
    else:
        try:
            time_utc = time.tz_convert('UTC')
        except TypeError:
            time_utc = time.tz_localize(location.tz).tz_convert('UTC')

    return time_utc


def datetime_to_djd(time):
    """
    Convierte un datetime en el Día Juliano de Dublín.

    Parámetros
    ----------
    tiempo : datetime.datetime
        tiempo a convertir

    Returns
    -------
    float
        días fraccionales desde 12/31/1899+0000
    """

    if time.tzinfo is None:
        time_utc = pytz.utc.localize(time)
    else:
        time_utc = time.astimezone(pytz.utc)

    djd_start = pytz.utc.localize(dt.datetime(1899, 12, 31, 12))
    djd = (time_utc - djd_start).total_seconds() * 1.0/(60 * 60 * 24)

    return djd


def djd_to_datetime(djd, tz='UTC'):
    """
    Convierte un número decimal de Días Julianos de Dublín en un objeto datetime.datetime.

    Parámetros
    ----------
    djd : float
        días fraccionales desde 12/31/1899+0000
    tz : str, predeterminado 'UTC'
        zona horaria para localizar el resultado

    Returns
    -------
    datetime.datetime
       El datetime resultante localizado en tz
    """

    djd_start = pytz.utc.localize(dt.datetime(1899, 12, 31, 12))

    utc_time = djd_start + dt.timedelta(days=djd)
    return utc_time.astimezone(pytz.timezone(tz))


def _pandas_to_doy(pd_object):
    """
    Encuentra el día del año para un objeto similar a datetime de pandas.

    Útil para la evaluación retrasada del atributo dayofyear.

    Parámetros
    ----------
    pd_objeto : DatetimeIndex o Timestamp

    Returns
    -------
    día_del_año
    """
    return pd_object.dayofyear


def _doy_to_datetimeindex(doy, epoch_year=2014):
    """
    Convierte un valor escalar o un arreglo de días del año en un pd.DatetimeIndex.

    Parámetros
    ----------
    doy : numérico
        Contiene los días del año.

    Returns
    -------
    pd.DatetimeIndex
    """
    doy = np.atleast_1d(doy).astype('float')
    epoch = pd.Timestamp('{}-12-31'.format(epoch_year - 1))
    timestamps = [epoch + dt.timedelta(days=adoy) for adoy in doy]
    return pd.DatetimeIndex(timestamps)


def _datetimelike_scalar_to_doy(time):
    return pd.DatetimeIndex([pd.Timestamp(time)]).dayofyear


def _datetimelike_scalar_to_datetimeindex(time):
    return pd.DatetimeIndex([pd.Timestamp(time)])


def _scalar_out(arg):
    if np.isscalar(arg):
        output = arg
    else:  #
        # works if it's a 1 length array and
        # will throw a ValueError otherwise
        output = np.asarray(arg).item()

    return output


def _array_out(arg):
    if isinstance(arg, pd.Series):
        output = arg.values
    else:
        output = arg

    return output


def _build_kwargs(keys, input_dict):
    """
    Parámetros
    ----------
    claves : iterable
        Normalmente una lista de cadenas.
    input_dict : tipo dict
        Un diccionario del que intentar extraer cada clave.

    Returns
    -------
    kwargs : dict
        Un diccionario solo con las claves que estaban en input_dict.
    """

    kwargs = {}
    for key in keys:
        try:
            kwargs[key] = input_dict[key]
        except KeyError:
            pass

    return kwargs


def _build_args(keys, input_dict, dict_name):
    """
    Parámetros
    ----------
    claves : iterable
        Normalmente una lista de cadenas.
    input_dict : tipo dict
        Un diccionario del que extraer cada clave.
    nombre_dict : str
        Un nombre de variable para incluir en un mensaje de error para las claves faltantes.

    Returns
    -------
    kwargs : lista
        Una lista con valores correspondientes a las claves.
    """
    try:
        args = [input_dict[key] for key in keys]
    except KeyError as e:
        missing_key = e.args[0]
        msg = (f"Missing required parameter '{missing_key}'. Found "
               f"{input_dict} in {dict_name}.")
        raise KeyError(msg)
    return args


# Created April,2014
# Author: Rob Andrews, Calama Consulting
# Modified: November, 2020 by C. W. Hansen, to add atol and change exit
# criteria
def _golden_sect_dataframe(params, lower, upper, func, atol=1e-8):
    """
    Búsqueda vectorizada de la sección dorada para encontrar el máximo de una función de una
    sola variable.

    Parámetros
    ----------
    params : diccionario de números
        Parámetros que se pasarán a `func`. Cada entrada debe tener la misma longitud.

    inferior: número
        Límite inferior para la optimización. Debe tener la misma longitud que cada
        entrada de params.

    superior: número
        Límite superior para la optimización. Debe tener la misma longitud que cada
        entrada de params.

    func: función
        Función a optimizar. Debe estar en la forma
        resultado = f(diccionario o DataFrame, cadena), donde resultado es un diccionario o DataFrame
        que también contiene la salida de la función, y cadena es la clave
        correspondiente a la variable de entrada de la función.

    Returns
    -------
    número
        función evaluada en los puntos óptimos

    número
        puntos óptimos

    Notas
    -----
    Esta función encontrará los puntos donde la función está maximizada.
    Devuelve NaN donde inferior o superior es NaN, o donde func evalúa a NaN.

    Ver también
    --------
    xm_solarlib.singlediode._pwr_optfcn
    """
    if np.any(upper - lower < 0.):
        raise ValueError('upper >= lower is required')

    phim1 = (np.sqrt(5) - 1) / 2

    df = params.copy()  # shallow copy to avoid modifying caller's dict
    df['VH'] = upper
    df['VL'] = lower

    converged = False

    while not converged:

        phi = phim1 * (df['VH'] - df['VL'])
        df['V1'] = df['VL'] + phi
        df['V2'] = df['VH'] - phi

        df['f1'] = func(df, 'V1')
        df['f2'] = func(df, 'V2')
        df['SW_Flag'] = df['f1'] > df['f2']

        df['VL'] = df['V2']*df['SW_Flag'] + df['VL']*(~df['SW_Flag'])
        df['VH'] = df['V1']*~df['SW_Flag'] + df['VH']*(df['SW_Flag'])

        err = abs(df['V2'] - df['V1'])

        # handle all NaN case gracefully
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    message='All-NaN slice encountered')
            converged = np.all(err[~np.isnan(err)] < atol)

    # best estimate of location of maximum
    df['max'] = 0.5 * (df['V1'] + df['V2'])
    func_result = func(df, 'max')
    x = np.where(np.isnan(func_result), np.nan, df['max'])
    if np.isscalar(df['max']):
        # np.where always returns an ndarray, converting scalars to 0d-arrays
        x = x.item()

    return func_result, x


def _get_sample_intervals(times, win_length):
    """Calcula el intervalo de tiempo y las muestras por ventana para las funciones
    de detección de cielo despejado al estilo de Reno.
    """
    deltas = np.diff(times.values) / np.timedelta64(1, '60s')

    # determine if we can proceed
    if times.inferred_freq and len(np.unique(deltas)) == 1:
        sample_interval = times[1] - times[0]
        sample_interval = sample_interval.seconds / 60  # in minutes
        samples_per_window = int(win_length / sample_interval)
        return sample_interval, samples_per_window
    else:
        message = (
            'algorithm does not yet support unequal time intervals. consider '
            'resampling your data and checking for gaps from missing '
            'periods, leap days, etc.'
        )
        raise NotImplementedError(message)


def _degrees_to_index(degrees, coordinate):
    """
    Transforma grados de entrada en un número de índice de salida.
    Especifique un valor en grados y ya sea 'latitud' o 'longitud' para
    obtener el número de índice adecuado para estos dos tipos de coordenadas.
    Parámetros
    ----------
    grados : float o int
        Grados de latitud o longitud.
    coordenada : string
        Especifique si el argumento grados es latitud o longitud. Debe
        establecerse como 'latitud' o 'longitud', de lo contrario, se
        generará un error.
    Retorna
    -------
    índice : np.int16
        El número de índice de latitud o longitud para usar al buscar valores
        en la tabla de búsqueda de turbidez de Linke.
    """
    # Asignar inputmin, inputmax y outputmax en función del tipo de coordenada.
    if coordinate == 'latitude':
        inputmin = 90
        inputmax = -90
        outputmax = 2160
    elif coordinate == 'longitude':
        inputmin = -180
        inputmax = 180
        outputmax = 4320
    else:
        raise IndexError("coordinate must be 'latitude' or 'longitude'.")

    inputrange = inputmax - inputmin
    scale = outputmax/inputrange  # número de índices por grado
    center = inputmin + 1 / scale / 2  # desplazamiento al centro del índice
    outputmax -= 1  # desplazar el índice a indexación desde cero
    index = (degrees - center) * scale
    err = IndexError('Input, %g, is out of range (%g, %g).' %
                     (degrees, inputmin, inputmax))

    # Si el índice todavía está fuera de límites después del redondeo, generar un error.
    # Se usa 0.500001 en comparaciones en lugar de 0.5 para permitir un pequeño margen de error
    # que puede ocurrir al tratar con números de punto flotante.
    if index > outputmax:
        if index - outputmax <= 0.500001:
            index = outputmax
        else:
            raise err
    elif index < 0:
        if -index <= 0.500001:
            index = 0
        else:
            raise err
    # Si el índice no se estableció en outputmax o 0, redondearlo y convertirlo en un
    # número entero para que se pueda utilizar en indexación basada en enteros.
    else:
        index = int(np.around(index))

    return index


EPS = np.finfo('float64').eps  # machine precision NumPy-1.20
DX = EPS**(1/3)  # optimal differential element


def _first_order_centered_difference(f, x0, dx=DX, args=()):
    # simple replacement for scipy.misc.derivative, which is scheduled for
    # removal in scipy 1.12.0
    df = f(x0+dx, *args) - f(x0-dx, *args)
    return df / 2 / dx


def get_pandas_index(*args):
    """
    Obtiene el índice del primer DataFrame o Serie de pandas en una lista de
    argumentos.

    Parámetros
    ----------
    args: argumentos posicionales
        Los valores numéricos para buscar un índice de pandas.

    Returns
    -------
    Un índice de pandas o None
        Se devuelve None si no hay DataFrames o Series de pandas en la lista de args.
    """
    return next(
        (a.index for a in args if isinstance(a, (pd.DataFrame, pd.Series))),
        None
    )
