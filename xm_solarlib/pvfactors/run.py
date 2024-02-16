from xm_solarlib.pvfactors.geometry import OrderedPVArray
from xm_solarlib.pvfactors.engine import PVEngine
from xm_solarlib.pvfactors.irradiance import HybridPerezOrdered
from xm_solarlib.pvfactors.viewfactors import VFCalculator


def run_timeseries_engine(fn_build_report, pvarray_parameters,
                          timestamps, dni, dhi, solar_zenith, solar_azimuth,
                          surface_tilt, surface_azimuth, albedo,
                          irradiance_model_params=None):
    """Run timeseries simulation without multiprocessing. This is the
    functional approach to the :py:class:`~pvfactors.engine.PVEngine` class.

    Parameters
    ----------
    fn_build_report : function
        Function that will build the report of the simulation
    pvarray_parameters : dict
        The parameters defining the PV array
    timestamps : array-like
        List of timestamps of the simulation.
    dni : array-like
        Direct normal irradiance values [W/m2]
    dhi : array-like
        Diffuse horizontal irradiance values [W/m2]
    solar_zenith : array-like
        Solar zenith angles [deg]
    solar_azimuth : array-like
        Solar azimuth angles [deg]
    surface_tilt : array-like
        Surface tilt angles, from 0 to 180 [deg]
    surface_azimuth : array-like
        Surface azimuth angles [deg]
    albedo : array-like
        Albedo values (or ground reflectivity)
    cls_pvarray : class of PV array, optional
        Class that will be used to build the PV array
        (Default =
        :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray` class)
    cls_engine : class of PV engine, optional
        Class of the engine to use to run the simulations (Default =
        :py:class:`~pvfactors.engine.PVEngine` class)
    cls_irradiance : class of irradiance model, optional
        The irradiance model that will be applied to the PV array
        (Default =
        :py:class:`~pvfactors.irradiance.models.HybridPerezOrdered` class)
    cls_vf : class of VF calculator, optional
        Calculator that will be used to calculate the view factor matrices
        (Default =
        :py:class:`~pvfactors.viewfactors.calculator.VFCalculator` class)
    fast_mode_pvrow_index : int, optional
        If a valid pvrow index is passed, then the PVEngine fast mode
        will be activated and the engine calculation will be done only
        for the back surface of the selected pvrow (Default = None)
    fast_mode_segment_index : int, optional
        If a segment index is passed, then the PVEngine fast mode
        will calculate back surface irradiance only for the
        selected segment of the selected back surface (Default = None)
    irradiance_model_params : dict, optional
        Dictionary of parameters that will be passed to the irradiance model
        class as kwargs at instantiation (Default = None)
    vf_calculator_params : dict, optional
        Dictionary of parameters that will be passed to the VF calculator
        class as kwargs at instantiation (Default = None)
    ghi : array-like, optional
        Global horizontal irradiance values [W/m2] (Default = None)

    Returns
    -------
    report
        Saved results from the simulation, as specified by user's report
        function
    """

    cls_pvarray=OrderedPVArray
    cls_engine=PVEngine
    cls_irradiance=HybridPerezOrdered
    cls_vf=VFCalculator
    fast_mode_pvrow_index=None
    fast_mode_segment_index=None
    ghi=None

    # Prepare input parameters
    irradiance_model_params = irradiance_model_params or {}
    vf_calculator_params = {}
    # Instantiate classes and engine
    irradiance_model = cls_irradiance(**irradiance_model_params)
    vf_calculator = cls_vf(**vf_calculator_params)
    pvarray = cls_pvarray.init_from_dict(pvarray_parameters)
    eng = cls_engine(pvarray, irradiance_model=irradiance_model,
                     vf_calculator=vf_calculator,
                     fast_mode_pvrow_index=fast_mode_pvrow_index,
                     fast_mode_segment_index=fast_mode_segment_index)
    # Fit engine
    eng.fit(timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth, albedo, ghi=ghi)

    # Run all timesteps
    report = (eng.run_full_mode(fn_build_report=fn_build_report))

    return report