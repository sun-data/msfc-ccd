import pytest
import numpy as np
import astropy.units as u
import msfc_ccd
from optika.sensors.materials import AbstractSiliconSensorMaterial
from optika._tests.test_mixins import AbstractTestPrintable


class AbstractTestAbstractSensor(
    AbstractTestPrintable,
):

    def test_num_tap_x(self, a: msfc_ccd.abc.AbstractSensor):
        result = a.num_tap_x
        assert result == 2

    def test_num_tap_y(self, a: msfc_ccd.abc.AbstractSensor):
        assert a.num_tap_y == 2

    def test_manufacturer(self, a: msfc_ccd.abc.AbstractSensor):
        assert isinstance(a.manufacturer, str)

    def test_family(self, a: msfc_ccd.abc.AbstractSensor):
        assert isinstance(a.family, str)

    def test_serial_number(self, a: msfc_ccd.abc.AbstractSensor):
        if a.serial_number is not None:
            assert isinstance(a.serial_number, str)

    def test_material(self, a: msfc_ccd.abc.AbstractSensor):
        assert isinstance(a.material, AbstractSiliconSensorMaterial)

    def test_width_pixel(self, a: msfc_ccd.abc.AbstractSensor):
        assert a.width_pixel > 0 * u.um

    def test_width_active(self, a: msfc_ccd.abc.AbstractSensor):
        assert a.width_active > 0 * u.um

    def test_num_pixels(self, a: msfc_ccd.abc.AbstractSensor):
        assert a.num_pixel.x > 0
        assert a.num_pixel.y > 0

    def test_num_pixels_active(self, a: msfc_ccd.abc.AbstractSensor):
        assert a.num_pixel_active.x > 0
        assert a.num_pixel_active.y > 0

    def test_num_blank(self, a: msfc_ccd.abc.AbstractSensor):
        assert isinstance(a.num_blank, int)
        assert a.num_blank > 0

    def test_num_overscan(self, a: msfc_ccd.abc.AbstractSensor):
        assert isinstance(a.num_overscan, int)
        assert a.num_overscan > 0

    def test_num_masked(self, a: msfc_ccd.abc.AbstractSensor):
        assert isinstance(a.num_masked, int)
        assert a.num_masked > 0

    def test_cte(self, a: msfc_ccd.abc.AbstractSensor):
        assert a.cte > 0

    def test_readout_noise(self, a: msfc_ccd.abc.AbstractSensor):
        assert a.readout_noise > 0 * u.electron

    def test_temperature(self, a: msfc_ccd.abc.AbstractSensor):
        assert a.temperature > 0 * u.K

    def test_dark_current(self, a: msfc_ccd.abc.AbstractSensor):
        assert a.dark_current() > 0 * u.electron / u.s


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        msfc_ccd.TeledyneCCD230(
            serial_number="42",
        ),
        msfc_ccd.TeledyneCCD230(
            readout_mode="full-frame",
        ),
    ],
)
class TestTeledyneCCD230(
    AbstractTestAbstractSensor,
):

    def test_width_package(self, a: msfc_ccd.TeledyneCCD230):
        assert a.width_package.x > 0 * u.mm
        assert a.width_package.y > 0 * u.mm

    def test_num_pixel_active_mode(self, a: msfc_ccd.TeledyneCCD230):
        result = a.num_pixel_active
        assert result.x == a.num_pixel.x
        if a.readout_mode == "full-frame":
            assert result.y == a.num_pixel.y
        else:
            assert result.y == a.num_pixel.y // 2

    def test_dark_current_datasheet(self, a: msfc_ccd.TeledyneCCD230):
        """The typical value and the temperature dependence are the datasheet's."""
        typical = a.dark_current(248 * u.K)
        assert abs(typical - 0.2 * u.electron / u.s) < 1e-9 * u.electron / u.s
        ratio = a.dark_current(273.15 * u.K) / typical
        assert (
            abs(
                ratio
                - 1.14e6
                * 273.15**3
                * np.exp(-9080 / 273.15)
                / (1.14e6 * 248**3 * np.exp(-9080 / 248))
            )
            < 1e-9
        )
