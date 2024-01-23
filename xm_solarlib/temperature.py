

TEMPERATURE_MODEL_PARAMETERS = {
    'sapm': {
        'open_rack_glass_glass': {'a': -3.47, 'b': -.0594, 'deltaT': 3},
        'close_mount_glass_glass': {'a': -2.98, 'b': -.0471, 'deltaT': 1},
        'open_rack_glass_polymer': {'a': -3.56, 'b': -.0750, 'deltaT': 3},
        'insulated_back_glass_polymer': {'a': -2.81, 'b': -.0455, 'deltaT': 0},
    },
    'pvsyst': {'freestanding': {'u_c': 29.0, 'u_v': 0},
               'insulated': {'u_c': 15.0, 'u_v': 0}}
}
"""Diccionario de parámetros de temperatura organizados por modelo.

En el nivel superior, hay claves para cada modelo. Actualmente existen dos modelos,
``'sapm'`` para el Modelo de Rendimiento de Matrices de Sandia y ``'pvsyst'``. Cada modelo
tiene un diccionario de configuraciones; un valor es en sí mismo un diccionario que contiene
parámetros del modelo. Recupere los parámetros indexando el modelo y la configuración
por nombre. Nota: las claves están en minúsculas y distinguen mayúsculas y minúsculas.

Ejemplo
-------
Recupere la configuración de vidrio-polímero de rack abierto para SAPM::

    from xm_solarlib.temperature import TEMPERATURE_MODEL_PARAMETERS
    temperature_model_parameters = (
        TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer'])
    # {'a': -3.56, 'b': -0.075, 'deltaT': 3}
"""


def _temperature_model_params(model, parameter_set):
    try:
        params = TEMPERATURE_MODEL_PARAMETERS[model]
        return params[parameter_set]
    except KeyError:
        msg = ('{} is not a named set of parameters for the {} cell'
               ' temperature model.'
               ' See xm_solarlib.temperature.TEMPERATURE_MODEL_PARAMETERS'
               ' for names'.format(parameter_set, model))
        raise KeyError(msg)