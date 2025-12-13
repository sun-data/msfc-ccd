import pytest
import astropy.units as u
import named_arrays as na
import msfc_ccd
from optika._tests.test_mixins import AbstractTestPrintable


class AbstractTestAbstractCamera(
    AbstractTestPrintable,
):

    @pytest.mark.parametrize("value", [1, 10])
    def test_calibrate_timedelta_exposure(
        self,
        a: msfc_ccd.abc.AbstractCamera,
        value: int,
    ):
        result = a.calibrate_timedelta_exposure(value)
        assert result > 0 * u.s

    @pytest.mark.parametrize("value", [1, 10])
    def test_calibrate_voltage_fpga(
        self,
        a: msfc_ccd.abc.AbstractCamera,
        value: int,
    ):
        result = a.calibrate_voltage_fpga(value)
        assert result > 0 * u.V

    @pytest.mark.parametrize("value", [1, 10])
    def test_calibrate_temperature_fpga(
        self,
        a: msfc_ccd.abc.AbstractCamera,
        value: int,
    ):
        result = a.calibrate_temperature_fpga(value)
        assert na.unit(result).is_equivalent(u.deg_C)

    @pytest.mark.parametrize("value", [1, 10])
    def test_calibrate_temperature_adc_1(
        self,
        a: msfc_ccd.abc.AbstractCamera,
        value: int,
    ):
        result = a.calibrate_temperature_adc_1(value)
        assert na.unit(result).is_equivalent(u.deg_C)

    @pytest.mark.parametrize("value", [1, 10])
    def test_calibrate_temperature_adc_234(
        self,
        a: msfc_ccd.abc.AbstractCamera,
        value: int,
    ):
        result = a.calibrate_temperature_adc_234(value)
        assert na.unit(result).is_equivalent(u.deg_C)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        msfc_ccd.Camera(),
    ],
)
class TestCamera(
    AbstractTestAbstractCamera,
):
    pass
