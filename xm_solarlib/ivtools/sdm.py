

def fit_cec_sam(celltype, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc,
                gamma_pmp, cells_in_series, temp_ref=25):
    """
     Estima los parámetros para el modelo de diodo único (SDM) de CEC utilizando el SAM
     SDK.

     Parámetros
     ----------
     tipo de celda: str
         El valor es uno de 'monoSi', 'multiSi', 'polySi', 'cis', 'cigs', 'cdte',
         'amorfo'
     v_mp: flotante
         Tensión en el punto de máxima potencia [V]
     i_mp: flotante
         Corriente en el punto de máxima potencia [A]
     v_oc: flotante
         Tensión de circuito abierto [V]
     i_sc: flotante
         Corriente de cortocircuito [A]
     alfa_sc: flotante
         Coeficiente de temperatura de la corriente de cortocircuito [A/C]
     beta_voc: flotante
         Coeficiente de temperatura de la tensión de circuito abierto [V/C]
     gamma_pmp: flotante
         Coeficiente de temperatura de potencia en el punto de máxima potencia [%/C]
     celdas_en_series: int
         Número de celdas en serie
     temp_ref: flotante, predeterminado 25
         Condición de temperatura de referencia [C]

     Devoluciones
     -------
     i_l_ref: flotante
         La corriente generada por la luz (o fotocorriente) en referencia.
         condiciones [A]
     i_o_ref: flotante
         La corriente de saturación inversa oscura o de diodo en referencia
         condiciones [A]
     r_s: flotar
         La resistencia en serie en condiciones de referencia, en ohmios.
     r_sh_ref: flotante
         La resistencia en derivación en condiciones de referencia, en ohmios.
     a_ref: flotante
         El producto del factor de idealidad del diodo habitual ``n`` (sin unidades),
         número de celdas en serie ``Ns`` y voltaje térmico de celda en
         condiciones de referencia [V]
     Ajustar: flotar
         El ajuste del coeficiente de temperatura para cortocircuito.
         actual, en porcentaje.

     aumentos
     ------
     Error de importación
         si NREL-PySAM no está instalado.
     Error de tiempo de ejecución
         si la extracción de parámetros no es exitosa.

     Notas
     -----
     El modelo CEC y el método de estimación se describen en [1]_.
     Se supone que las entradas ``v_mp``, ``i_mp``, ``v_oc`` e ``i_sc`` provienen de un
     curva IV única a irradiancia y temperatura celular constantes. La irradiancia es
     no utilizado explícitamente por el procedimiento de ajuste. El nivel de irradiancia al que
     Se determina la curva IV de entrada y se determina la temperatura de la celda especificada.
     ``temp_ref`` son las condiciones de referencia para los parámetros de salida
     ``i_l_ref``, ``i_o_ref``, ``r_s``, ``r_sh_ref``, ``a_ref`` y ``Ajustar``.

     Referencias
     ----------
     .. [1] A. Dobos, "Una calculadora de coeficientes mejorada para California
        Modelo de módulo fotovoltaico de 6 parámetros de la Comisión de Energía", Revista de
        Ingeniería de energía solar, vol 134, 2012. :doi:`10.1115/1.4005759`
     """

    try:
        from PySAM import PySSC
    except ImportError:
        raise ImportError("Requires NREL's PySAM package at "
                          "https://pypi.org/project/NREL-PySAM/.")

    datadict = {'tech_model': '6parsolve', 'financial_model': None,
                'celltype': celltype, 'Vmp': v_mp,
                'Imp': i_mp, 'Voc': v_oc, 'Isc': i_sc, 'alpha_isc': alpha_sc,
                'beta_voc': beta_voc, 'gamma_pmp': gamma_pmp,
                'Nser': cells_in_series, 'Tref': temp_ref}

    result = PySSC.ssc_sim_from_dict(datadict)
    if result['cmod_success'] == 1:
        return tuple([result[k] for k in ['Il', 'Io', 'Rs', 'Rsh', 'a',
                      'Adj']])
    else:
        raise RuntimeError('Parameter estimation failed')