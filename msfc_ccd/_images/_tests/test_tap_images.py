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

    def test_bias_ignores_overscan(self, a: msfc_ccd.abc.AbstractTapData):
        b = a.replace(outputs=a.outputs + 1000 * u.DN * a.where_overscan())
        assert np.all(a.bias().outputs == b.bias().outputs)
        assert np.all(
            a.bias(num_overscan=None).outputs != b.bias(num_overscan=None).outputs
        )

    def test_unbiased(self, a: msfc_ccd.abc.AbstractTapData):
        super().test_unbiased(a)
        result = a.unbiased
        assert isinstance(result, msfc_ccd.TapData)
        assert na.unit(result.outputs) == na.unit(a.outputs)
        assert np.abs(result.outputs.mean()) < 1 * u.DN

    def test_dark(self, a: msfc_ccd.abc.AbstractTapData):
        axis = "_test_dark"
        num = 8
        sigma = 3 * u.DN
        rng = np.random.default_rng(seed=42)
        noise = na.ScalarArray(
            ndarray=rng.normal(size=(num,) + a.outputs.ndarray.shape) * sigma,
            axes=(axis,) + a.outputs.axes,
        )
        outputs = a.outputs + noise
        outputs[{axis: 2, a.axis_x: 100, a.axis_y: 100}] = 60000 * u.DN
        b = a.replace(outputs=outputs)
        result = b.dark(axis)
        assert isinstance(result, msfc_ccd.TapData)
        assert axis not in result.outputs.shape
        assert result.outputs.shape[a.axis_x] == a.outputs.shape[a.axis_x]
        assert result.outputs.shape[a.axis_y] == a.outputs.shape[a.axis_y]
        residual = result.outputs - a.outputs
        assert np.all(np.abs(residual[{a.axis_x: 100, a.axis_y: 100}]) < 10 * sigma)
        assert np.all(np.abs(residual.mean()) < 0.1 * sigma)
        assert np.all(residual.std() < sigma / np.sqrt(num / 2))

    def test_readout_noise(self, a: msfc_ccd.abc.AbstractTapData):
        axis = "_test_readout_noise"
        num = 5
        sigma = 3 * u.DN
        rng = np.random.default_rng(seed=42)
        outputs = a.outputs + na.ScalarArray(
            ndarray=rng.normal(size=(num,) + a.outputs.ndarray.shape) * sigma,
            axes=(axis,) + a.outputs.axes,
        )
        outputs[{axis: 2, a.axis_x: 100, a.axis_y: 100}] = 60000 * u.DN
        b = a.replace(outputs=outputs)
        result = b.readout_noise(axis)
        assert isinstance(result, msfc_ccd.TapData)
        assert result.outputs.shape[axis] == num - 1
        assert a.axis_x not in result.outputs.shape
        assert a.axis_y not in result.outputs.shape
        assert np.all(np.abs(result.outputs - sigma) < 0.05 * sigma)

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
