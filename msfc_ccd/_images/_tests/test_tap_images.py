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

    def test_gain(self, a: msfc_ccd.abc.AbstractTapData):
        gain = 2.5 * u.electron / u.DN
        fe55 = msfc_ccd.Fe55()
        rng = np.random.default_rng(seed=42)

        num_blank = a.camera.sensor.num_blank
        num_x, num_y, step = 30, 15, 33

        # Fe 55 events on a grid, spaced far enough apart to stay isolated
        probability_beta = fe55.probability_k_beta / (
            fe55.probability_k_alpha + fe55.probability_k_beta
        )
        where_beta = rng.random(size=(num_x, num_y)) < probability_beta
        charge = np.where(where_beta, fe55.charge_k_beta, fe55.charge_k_alpha)
        charge = charge / gain + rng.normal(scale=14, size=charge.shape) * u.DN
        charge = na.ScalarArray(charge, axes=(a.axis_x, a.axis_y))

        index = {
            a.axis_x: slice(num_blank + 8, num_blank + 8 + num_x * step, step),
            a.axis_y: slice(8, 8 + num_y * step, step),
        }
        outputs = a.outputs.copy()
        outputs[index] = outputs[index] + charge

        result = a.replace(outputs=outputs).gain()

        assert isinstance(result, msfc_ccd.TapData)
        assert a.axis_x not in result.outputs.shape
        assert a.axis_y not in result.outputs.shape
        assert na.unit(result.outputs).is_equivalent(u.electron / u.DN)
        assert np.all(np.abs(result.outputs - gain) < 0.02 * gain)

    @classmethod
    def _darks(
        cls,
        a: msfc_ccd.abc.AbstractTapData,
        axis: str,
        rate: u.Quantity,
    ) -> msfc_ccd.abc.AbstractTapData:
        """Build a sequence of darks with a known dark current rate."""
        timedelta = na.ScalarArray(
            ndarray=[2, 3, 4, 7, 12] * u.s,
            axes=axis,
        )

        # The dark current accumulates only in the light-sensitive pixels,
        # so the blank columns used for the bias are untouched.
        where_active = ~(a.where_blank() | a.where_overscan())
        outputs = a.outputs + rate * timedelta * where_active

        # Cosmic rays, which arrive in proportion to the exposure time and so
        # would otherwise be counted as dark current.
        # They land only in the active pixels, since a cosmic ray in the blank
        # columns would corrupt the bias instead.
        rng = np.random.default_rng(seed=42)
        shape = outputs.shape
        num_blank = a.camera.sensor.num_blank
        num_x = shape[a.axis_x] - num_blank - a.camera.sensor.num_overscan
        num = np.round(10 * timedelta / timedelta[{axis: 0}]).astype(int)
        for i in range(shape[axis]):
            for _ in range(num[{axis: i}].ndarray.item()):
                index = {
                    axis: i,
                    a.axis_x: num_blank + rng.integers(num_x).item(),
                    a.axis_y: rng.integers(shape[a.axis_y]).item(),
                }
                outputs[index] = 60000 * u.DN

        return a.replace(
            inputs=a.inputs.replace(timedelta=timedelta),
            outputs=outputs,
        )

    def test_dark_current(self, a: msfc_ccd.abc.AbstractTapData):
        axis = "_test_dark_current"
        rate = 0.05 * u.DN / u.s
        result = self._darks(a, axis, rate).dark_current(axis)

        assert isinstance(result, msfc_ccd.TapData)
        assert axis not in result.outputs.shape
        assert a.axis_x not in result.outputs.shape
        assert a.axis_y not in result.outputs.shape
        assert np.all(np.abs(result.outputs - rate) < 0.01 * rate)

    def test_dark_current_linear(self, a: msfc_ccd.abc.AbstractTapData):
        """The rate must not depend on which exposures were used."""
        axis = "_test_dark_current_linear"
        rate = 0.05 * u.DN / u.s
        b = self._darks(a, axis, rate)

        short = b[{axis: slice(None, 3)}].dark_current(axis)
        long = b[{axis: slice(2, None)}].dark_current(axis)

        assert np.all(np.abs(long.outputs - short.outputs) < 0.01 * rate)

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
