import pytest
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na
import msfc_ccd


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        msfc_ccd.ImageHeader(
            pixel=na.Cartesian2dVectorArrayRange(
                start=0,
                stop=32,
                axis=na.Cartesian2dVectorArray(x="x", y="y"),
            ),
            time_start=na.ScalarArray(
                ndarray=np.linspace(
                    astropy.time.Time("2024-03-25T20:49"),
                    astropy.time.Time("2024-03-25T21:49"),
                    num=5,
                ),
                axes="t",
            ),
            timedelta=na.ScalarArray(
                ndarray=[9.98, 9.99, 10.1, 10.06, 9.76] * u.s,
                axes="t",
            ),
            timedelta_requested=10 * u.s,
            serial_number="SN-001",
            camera_id="CAM-1",
            sequence_number=na.arange(0, 5, axis="t"),
            count=na.arange(0, 5, axis="t"),
            run_mode="sequence",
            status="completed",
            voltage_fpga_vccint=5 * u.V,
            voltage_fpga_vccaux=6 * u.V,
            voltage_fpga_vccbram=7 * u.V,
            temperature_fpga=20 * u.deg_C,
            temperature_adc_1=21 * u.deg_C,
            temperature_adc_2=22 * u.deg_C,
            temperature_adc_3=23 * u.deg_C,
            temperature_adc_4=24 * u.deg_C,
        )
    ],
)
class TestImageHeader:

    @pytest.mark.parametrize(
        argnames="item",
        argvalues=[
            dict(t=0),
        ],
    )
    def test__getitem__(
        self,
        a: msfc_ccd.ImageHeader,
        item: dict[str, int | slice] | na.AbstractArray,
    ):
        b = a[item]
        assert isinstance(b, msfc_ccd.ImageHeader)
        assert np.all(a.time[item] == b.time)

    def test_time_start(self, a: msfc_ccd.ImageHeader):
        result = a.time_start
        if not isinstance(result, astropy.time.Time):
            assert isinstance(result.ndarray, astropy.time.Time)

    def test_time(self, a: msfc_ccd.ImageHeader):
        result = a.time
        if not isinstance(result, astropy.time.Time):
            assert isinstance(result.ndarray, astropy.time.Time)
        assert np.all(result > a.time_start)

    def test_time_end(self, a: msfc_ccd.ImageHeader):
        result = a.time_end
        if not isinstance(result, astropy.time.Time):
            assert isinstance(result.ndarray, astropy.time.Time)
        assert np.all(result > a.time)

    def test_timedelta(self, a: msfc_ccd.ImageHeader):
        result = a.timedelta
        assert np.all(result >= 1 * u.s)
        assert np.all(result < 1000 * u.s)

    def test_timedelta_requested(self, a: msfc_ccd.ImageHeader):
        result = a.timedelta
        assert np.all(result >= 1 * u.s)
        assert np.all(result < 1000 * u.s)

    def test_serial_number(self, a: msfc_ccd.ImageHeader):
        result = a.serial_number
        assert isinstance(result, (str, na.AbstractScalar))

    def test_camera_id(self, a: msfc_ccd.ImageHeader):
        result = a.camera_id
        assert isinstance(result, (str, na.AbstractScalar))

    def test_sequence_number(self, a: msfc_ccd.ImageHeader):
        result = a.sequence_number
        assert np.all(result >= 0)

    def test_count(self, a: msfc_ccd.ImageHeader):
        result = a.count
        assert np.all(result >= 0)

    def test_run_mode(self, a: msfc_ccd.ImageHeader):
        result = a.run_mode
        assert isinstance(result, (str, na.AbstractScalar))

    def test_status(self, a: msfc_ccd.ImageHeader):
        result = a.status
        assert isinstance(result, (str, na.AbstractScalar))

    def test_voltage_fpga_vccint(self, a: msfc_ccd.abc.AbstractImageData):
        result = a.voltage_fpga_vccint
        assert np.all(result >= 0 * u.V)
        assert np.all(result < 50 * u.V)

    def test_voltage_fpga_vccaux(self, a: msfc_ccd.ImageHeader):
        result = a.voltage_fpga_vccaux
        assert np.all(result >= 0 * u.V)
        assert np.all(result < 50 * u.V)

    def test_voltage_fpga_vccbram(self, a: msfc_ccd.ImageHeader):
        result = a.voltage_fpga_vccbram
        assert np.all(result >= 0 * u.V)
        assert np.all(result < 50 * u.V)

    def test_temperature_fpga(self, a: msfc_ccd.ImageHeader):
        result = a.temperature_fpga
        assert np.all(result >= 0 * u.deg_C)
        assert np.all(result < 100 * u.deg_C)

    def test_temperature_adc_1(self, a: msfc_ccd.ImageHeader):
        result = a.temperature_adc_1
        assert np.all(result >= 0 * u.deg_C)
        assert np.all(result < 100 * u.deg_C)

    def test_temperature_adc_2(self, a: msfc_ccd.ImageHeader):
        result = a.temperature_adc_2
        assert np.all(result >= 0 * u.deg_C)
        assert np.all(result < 100 * u.deg_C)

    def test_temperature_adc_3(self, a: msfc_ccd.ImageHeader):
        result = a.temperature_adc_3
        assert np.all(result >= 0 * u.deg_C)
        assert np.all(result < 100 * u.deg_C)

    def test_temperature_adc_4(self, a: msfc_ccd.ImageHeader):
        result = a.temperature_adc_4
        assert np.all(result >= 0 * u.deg_C)
        assert np.all(result < 100 * u.deg_C)
