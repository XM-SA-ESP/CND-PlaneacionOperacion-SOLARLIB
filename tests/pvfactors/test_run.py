from  xm_solarlib.pvfactors.run import run_timeseries_engine
import numpy as np
from unittest import mock


def test_run_timeseries_engine(fn_report_example, params_serial,
                               df_inputs_clearsky_8760):
    """Test that running timeseries engine with full mode works consistently"""
    df_inputs = df_inputs_clearsky_8760.iloc[:24, :]
    n = df_inputs.shape[0]

    # Get MET data
    timestamps = df_inputs.index
    dni = df_inputs.dni.values
    dhi = df_inputs.dhi.values
    solar_zenith = df_inputs.solar_zenith.values
    solar_azimuth = df_inputs.solar_azimuth.values
    surface_tilt = df_inputs.surface_tilt.values
    surface_azimuth = df_inputs.surface_azimuth.values

    report = run_timeseries_engine(
        fn_report_example, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'])

    assert len(report['qinc_front']) == n
    # Test value consistency
    np.testing.assert_almost_equal(np.nansum(report['qinc_back']),
                                   541.7115807694377)
    np.testing.assert_almost_equal(np.nansum(report['iso_back']),
                                   18.050083142438311)
    # Check a couple values
    np.testing.assert_almost_equal(report['qinc_back'][7],
                                   11.160301350847325)
    np.testing.assert_almost_equal(report['qinc_back'][-8],
                                   8.642850754173368)






class TestFastReportBuilder(object):

    @staticmethod
    def build(pvarray):

        return {'qinc_back': pvarray.ts_pvrows[1].back
                .get_param_weighted('qinc').tolist()}

    @staticmethod
    def merge(reports):

        report = reports[0]
        keys = report.keys()
        for other_report in reports[1:]:
            for key in keys:
                report[key] += other_report[key]

        return report


class TestFAOIReportBuilder(object):

    @staticmethod
    def build(pvarray):
        pvrow = pvarray.ts_pvrows[0]
        return {'qinc_front': pvrow.front.get_param_weighted('qinc').tolist(),
                'qabs_front': pvrow.front.get_param_weighted('qabs').tolist(),
                'qinc_back': pvrow.back.get_param_weighted('qinc').tolist(),
                'qabs_back': pvrow.back.get_param_weighted('qabs').tolist()}

    @staticmethod
    def merge(reports):
        report = reports[0]
        keys = report.keys()
        for other_report in reports[1:]:
            for key in keys:
                report[key] += other_report[key]
        return report