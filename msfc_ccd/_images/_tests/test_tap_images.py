import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import msfc_ccd
from . import test_images

_camera = msfc_ccd.Camera(
    gain=na.ScalarArray(
        ndarray=[[2.5, 2.6], [2.7, 2.8]] * u.electron / u.DN,
        axes=("tx", "ty"),
    ),
    axis_tap_x="tx",
    axis_tap_y="ty",
)


class AbstractTestAbstractTapImage(
    test_images.AbstractTestAbstractCameraData,
):

    def test_axis_tap_x(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.axis_tap_x
        assert isinstance(result, str)
        assert result in a.outputs.shape

    def test_axis_tap_y(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.axis_tap_y
        assert isinstance(result, str)
        assert result in a.outputs.shape

    def test_tap(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.tap
        for ax in result:
            assert isinstance(ax, str)
            assert isinstance(result[ax], na.AbstractScalarArray)

    def test_label(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.label
        for s in result.ndarray.flat:
            assert isinstance(s, str)

    def test_where_blank(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.where_blank()
        assert result.sum() == a.camera.sensor.num_blank

    def test_where_overscan(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.where_overscan()
        assert result.sum() == a.camera.sensor.num_overscan

    def test_bias(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.bias()
        axis_tap_x = a.camera.axis_tap_x
        axis_tap_y = a.camera.axis_tap_y
        assert na.unit(result.outputs) == na.unit(a.outputs)
        assert result.shape[axis_tap_x] == a.shape[axis_tap_x]
        assert result.shape[axis_tap_y] == a.shape[axis_tap_y]

    def test_unbiased(self, a: msfc_ccd.abc.AbstractTapData):
        super().test_unbiased(a)
        result = a.unbiased
        assert isinstance(result, msfc_ccd.TapData)
        assert na.unit(result.outputs) == na.unit(a.outputs)
        assert np.abs(result.outputs.mean()) < 1 * u.DN

    def test_active(self, a: msfc_ccd.abc.AbstractTapData):
        super().test_active(a)
        sensor = a.camera.sensor
        num_nap = sensor.num_blank + sensor.num_overscan
        result = a.active
        assert isinstance(result, msfc_ccd.TapData)
        assert result.shape[a.axis_x] == a.shape[a.axis_x] - num_nap
        assert result.shape[a.axis_y] == a.shape[a.axis_y]


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        msfc_ccd.fits.open(msfc_ccd.samples.path_dark_esis1, _camera).taps,
    ],
)
class TestTapImage(
    AbstractTestAbstractTapImage,
):
    pass
