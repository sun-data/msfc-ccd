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

    def test_amplifier(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.amplifier
        assert result.shape[a.axis_tap_x] == a.shape[a.axis_tap_x]
        assert result.shape[a.axis_tap_y] == a.shape[a.axis_tap_y]
        assert set(result.ndarray.flat) == {"1/E", "2/H", "3/G", "4/F"}

    def test_where_blank(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.where_blank()
        assert result.sum() == a.camera.sensor.num_blank

    def test_where_overscan(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.where_overscan()
        assert result.sum() == a.camera.sensor.num_overscan

    def test_where_masked(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.where_masked()
        assert result.sum() == a.camera.sensor.num_masked
        assert result[{a.axis_y: 0}]

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

    def test_hits(self, a: msfc_ccd.abc.AbstractTapData):
        sensor = a.camera.sensor
        num_blank = sensor.num_blank
        charge = 600 * u.DN

        outputs = a.outputs.copy()

        # Isolated events, spaced far enough apart to stay isolated
        index = [
            {a.axis_x: num_blank + 10 + 71 * k, a.axis_y: 10 + 53 * k} for k in range(8)
        ]
        for i in index:
            outputs[i] = outputs[i] + charge

        # Two adjacent hot pixels, which are not an isolated event
        index_pair = {a.axis_x: num_blank + 800, a.axis_y: 400}
        index_next = {a.axis_x: num_blank + 801, a.axis_y: 400}
        outputs[index_pair] = outputs[index_pair] + charge
        outputs[index_next] = outputs[index_next] + charge

        result = a.replace(outputs=outputs).hits()

        assert isinstance(result, msfc_ccd.TapData)
        assert result.shape[a.axis_x] == a.active.shape[a.axis_x]
        assert result.shape[a.axis_y] == a.shape[a.axis_y]

        # Every isolated event is found, with its charge
        for i in index:
            i = {a.axis_x: i[a.axis_x] - num_blank, a.axis_y: i[a.axis_y]}
            assert np.all(np.isfinite(result.outputs[i]))
            assert np.all(np.abs(result.outputs[i] - charge) < 0.1 * charge)

        # The adjacent pair is not
        i = {a.axis_x: index_pair[a.axis_x] - num_blank, a.axis_y: index_pair[a.axis_y]}
        assert not np.any(np.isfinite(result.outputs[i]))

    def test_hits_single(self, a: msfc_ccd.abc.AbstractTapData):
        num_blank = a.camera.sensor.num_blank
        charge = 600 * u.DN

        # The dark level and the noise of the active pixels, so that pixels
        # can be set to a known height above the dark level
        level = np.median(a.active.outputs, axis=a.axis_xy)
        deviation = np.abs(a.active.outputs - level)
        sigma = 1.482602218505602 * np.median(deviation, axis=a.axis_xy)

        outputs = a.outputs.copy()

        # A single-pixel event
        index = {a.axis_x: num_blank + 100, a.axis_y: 100}
        outputs[index] = level + charge

        # An event with one neighbor between the split and event thresholds
        index_split = {a.axis_x: num_blank + 300, a.axis_y: 300}
        index_neighbor = {a.axis_x: num_blank + 301, a.axis_y: 300}
        outputs[index_split] = level + charge
        outputs[index_neighbor] = level + 4 * sigma

        b = a.replace(outputs=outputs)
        result = b.hits(threshold_split=3)
        result_all = b.hits()

        i = {a.axis_x: index[a.axis_x] - num_blank, a.axis_y: index[a.axis_y]}
        assert np.all(np.abs(result.outputs[i] - charge) < 1 * u.DN)

        i = {
            a.axis_x: index_split[a.axis_x] - num_blank,
            a.axis_y: index_split[a.axis_y],
        }
        assert not np.any(np.isfinite(result.outputs[i]))
        assert np.all(np.isfinite(result_all.outputs[i]))

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
