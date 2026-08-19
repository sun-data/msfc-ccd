import pytest
import astropy.units as u
import named_arrays as na
import msfc_ccd
from optika._tests.test_mixins import AbstractTestPrintable


class AbstractTestAbstractCamera(
    AbstractTestPrintable,
):

    def test_gain(
        self,
        a: msfc_ccd.abc.AbstractCamera,
    ):
        result = a.gain
        if result is not None:
            assert na.unit(result).is_equivalent(u.electron / u.DN)

    def test_dn_to_electrons(
        self,
        a: msfc_ccd.abc.AbstractCamera,
    ):
        b = 100 * u.DN
        if a.gain is None:
            with pytest.raises(ValueError, match="`gain` is `None`"):
                a.dn_to_electrons(b)
        else:
            result = a.dn_to_electrons(b)
            assert na.unit(result).is_equivalent(u.electron)

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
        msfc_ccd.Camera(
            gain=na.ScalarArray(
                ndarray=[[2.5, 2.6], [2.7, 2.8]] * u.electron / u.DN,
                axes=("tap_x", "tap_y"),
            ),
        ),
    ],
)
class TestCamera(
    AbstractTestAbstractCamera,
):
    pass
